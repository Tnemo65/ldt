"""Resume benchmark_v7 from checkpoint."""
import sys, warnings, time, gc
sys.path.insert(0, r'C:\proj\ldt\results\v7')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import benchmark_v7 as bm
from benchmark_v7 import (
    evaluate_batch, evaluate_streaming,
    MONTHS, SEEDS, DIFFICULTIES, LABEL_BUDGETS,
    TRAIN_N, VAL_N, TEST_N, ANOMALY_N, ANOMALY_RATE,
    ANOMALY_PARAMS,
    SklearnIF, SklearnOCSVM, DenoisingAE, CADIFEiaBatch,
    RandomBaseline, sHST_River, MemStream, CADIFEiaStream,
    METRICS,
)

OUT_DIR = Path(r'C:\proj\ldt\results\v7')
CHECKPOINT = OUT_DIR / 'checkpoint_v7.csv'

# Load checkpoint
df_checkpoint = pd.read_csv(CHECKPOINT)
print(f'Loaded checkpoint: {len(df_checkpoint)} rows')

# Build jobs from checkpoint metadata
completed = set(zip(
    df_checkpoint['algorithm'],
    df_checkpoint['difficulty'],
    df_checkpoint['fold'],
    df_checkpoint['seed'],
    df_checkpoint['label_budget'].fillna(0).astype(int)
))
print(f'Completed unique jobs: {len(completed)}')

# Load data
print('\nLoading data...', flush=True)
monthly, monthly_X = [], []
for m in MONTHS:
    df = bm.clean(bm.load_month(2024, m))
    X = bm.features(df)
    monthly.append(df)
    monthly_X.append(X.astype(np.float32))
    print(f'  Month {m:02d}: {len(df):,} records', flush=True)

# Build all jobs
print('\nBuilding jobs...', flush=True)
all_jobs = []
for fold_idx, test_month in enumerate(MONTHS[1:], 1):
    train_months = MONTHS[:test_month - 1]
    val_month = train_months[-1] if train_months else test_month - 1
    if not train_months:
        continue

    train_X = np.vstack([monthly_X[m - 1] for m in train_months])
    train_df = pd.concat([monthly[m - 1] for m in train_months], ignore_index=True)

    val_X = monthly_X[val_month - 1][-VAL_N:]
    val_df = monthly[val_month - 1].iloc[-VAL_N:].reset_index(drop=True)

    n_train_keep = len(train_X) - VAL_N
    train_X = train_X[:n_train_keep]
    train_df = train_df.iloc[:n_train_keep].reset_index(drop=True)
    if len(train_X) > TRAIN_N:
        rng_sub = np.random.RandomState(42)
        idx = rng_sub.choice(len(train_X), TRAIN_N, replace=False)
        train_X = train_X[idx]
        train_df = train_df.iloc[idx].reset_index(drop=True)

    test_df = monthly[test_month - 1]

    for diff in DIFFICULTIES:
        params = ANOMALY_PARAMS[diff]
        seed_s = SEEDS[fold_idx % len(SEEDS)]

        rng_src = np.random.RandomState(seed_s)
        if len(test_df) > TEST_N:
            src_idx = rng_src.choice(len(test_df), TEST_N, replace=False)
            test_df_sub = test_df.iloc[src_idx].reset_index(drop=True)
        else:
            test_df_sub = test_df.reset_index(drop=True)

        test_df_inj, y_labels = bm.inject_anomalies(test_df_sub, params, seed_s)
        X_test = bm.features(test_df_inj).astype(np.float32)
        y_labels = np.array(y_labels, dtype=np.int8)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(train_X).astype(np.float32)
        X_val_s = scaler.transform(val_X).astype(np.float32)
        X_test_s = scaler.transform(X_test).astype(np.float32)
        y_val = np.zeros(len(X_val_s), dtype=np.int8)

        batch_algo_info = [
            ('Random',        RandomBaseline,    {}),
            ('sklearn_IF',    SklearnIF,        {}),
            ('sklearn_OCSVM', SklearnOCSVM,    {}),
            ('DenoisingAE',   DenoisingAE,     {}),
            ('CA-DIF-EIA',    CADIFEiaBatch,   {'ablation': 'full'}),
            ('AE+IF',         CADIFEiaBatch,    {'ablation': 'ae_if'}),
            ('IF-baseline',   CADIFEiaBatch,   {'ablation': 'baseline'}),
        ]
        for name, cls, kw in batch_algo_info:
            for seed in SEEDS:
                key = (name, diff, fold_idx, seed, 0)
                if key in completed:
                    continue
                all_jobs.append({
                    'type': 'batch', 'fold': fold_idx, 'test_month': test_month,
                    'diff': diff, 'algo_name': name, 'algo_cls': cls,
                    'seed': seed, 'X_train': X_train_s,
                    'X_val': X_val_s, 'X_test': X_test_s,
                    'y_val': y_val, 'y_test': y_labels, 'kwargs': kw,
                    'label_budget': 0,
                })

        stream_algo_info = [
            ('Random',            RandomBaseline,    {}),
            ('sHST-River',       sHST_River,       {}),
            ('MemStream',        MemStream,        {}),
            ('CA-DIF-EIA-Stream', CADIFEiaStream,  {}),
        ]
        for name, cls, kw in stream_algo_info:
            for seed in SEEDS:
                for lb in LABEL_BUDGETS:
                    key = (name, diff, fold_idx, seed, lb)
                    if key in completed:
                        continue
                    all_jobs.append({
                        'type': 'stream', 'fold': fold_idx, 'test_month': test_month,
                        'diff': diff, 'algo_name': name, 'algo_cls': cls,
                        'seed': seed, 'X_train': X_train_s,
                        'X_val': X_val_s, 'X_test': X_test_s,
                        'y_val': y_val, 'y_test': y_labels, 'kwargs': kw,
                        'label_budget': lb,
                    })

print(f'Total remaining jobs: {len(all_jobs)}')
if len(all_jobs) == 0:
    print('ALL DONE! Resume will compute stats from checkpoint.')
else:
    # Run remaining jobs
    print(f'\nRunning {len(all_jobs)} remaining jobs...')
    t0 = time.perf_counter()
    new_results = []
    CHECKPOINT_INTERVAL = 25

    for i, job in enumerate(all_jobs):
        try:
            if job['type'] == 'batch':
                res = evaluate_batch(
                    job['algo_cls'],
                    job['X_train'], job['X_val'], job['X_test'],
                    job['y_val'], job['y_test'],
                    job['seed'], **job['kwargs']
                )
            else:
                res = evaluate_streaming(
                    job['algo_cls'],
                    job['X_train'], job['X_val'], job['X_test'],
                    job['y_val'], job['y_test'],
                    job['seed'],
                    label_budget=job['label_budget'], **job['kwargs']
                )
            res['error'] = ''
        except Exception as e:
            res = {m: float('nan') for m in METRICS + ['train_ms', 'score_ms', 'labels_consumed', 'anomaly_rate']}
            res['error'] = str(e)[:80]

        res.update({
            'fold': job['fold'], 'month': job['test_month'],
            'difficulty': job['diff'], 'algorithm': job['algo_name'],
            'seed': job['seed'],
            'label_budget': job['label_budget'],
        })
        new_results.append(res)

        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            remain = (len(all_jobs) - i - 1) / rate / 60
            print(f'  {i+1}/{len(all_jobs)} ({rate:.2f}/s, ~{remain:.1f}m left)')
            combined = pd.concat([df_checkpoint, pd.DataFrame(new_results)], ignore_index=True)
            combined.to_csv(CHECKPOINT, index=False)

        gc.collect()

    # Final save
    combined = pd.concat([df_checkpoint, pd.DataFrame(new_results)], ignore_index=True)
    combined.to_csv(CHECKPOINT, index=False)
    print(f'\n  Saved {len(combined)} rows to checkpoint')

    t_done = time.perf_counter() - t0
    print(f'  Done in {t_done/60:.1f} min')

# Load full results and run stats
print('\n' + '='*60)
print('COMPUTING FINAL STATS FROM ALL RESULTS')
print('='*60)

results = pd.read_csv(CHECKPOINT)
print(f'Total rows: {len(results)}')
errors = results['error'].notna().sum()
print(f'Errors: {errors}')
if errors > 0:
    print(results[results['error'].notna()][['algorithm','error']].drop_duplicates())

# Stats
print('\n=== Summary by Algorithm ===')
print(results.groupby('algorithm')['AUC_PR'].agg(['mean','std','count']).sort_values('mean', ascending=False).to_string())

print('\n=== Batch AUC-PR by Difficulty ===')
batch = results[~results['algorithm'].isin(['Random','sHST-River','MemStream','CA-DIF-EIA-Stream'])]
pivot = batch.pivot_table(index='algorithm', columns='difficulty', values='AUC_PR', aggfunc='mean')
print(pivot.round(4).to_string())

print('\n=== Friedman Test ===')
stat_results = bm.statistical_analysis(results, 'all', results['algorithm'].unique().tolist())
for name, res in stat_results.items():
    if not res.get('significant', False):
        print(f'  {name}: Friedman p={res.get("friedman_p", 1):.4f} -- NOT SIGNIFICANT')
    else:
        print(f'  {name}: Friedman p={res.get("friedman_p", 0):.4f} -- SIGNIFICANT')
        print(f'    Ranks: {dict(res.get("avg_ranks", {}))}')
        for pair in res.get('pairwise_comparisons', []):
            sig = "YES" if pair.get('significant') else "no"
            print(f'    {pair["target"]} vs {pair["baseline"]}: p_holm={pair["p_corrected"]:.4f}, sig={sig}')

print('\n=== BAR Score (Streaming) ===')
stream = results[results['algorithm'].isin(['sHST-River','MemStream','CA-DIF-EIA-Stream']) & (results['label_budget'] > 0)]
for lb in sorted(stream['label_budget'].unique()):
    sub = stream[stream['label_budget'] == lb]
    print(f'\n  Budget {int(lb)}:')
    for algo, row in sub.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).items():
        bar = 100 * row / lb if lb > 0 else 0
        print(f'    {algo}: AUC_PR={row:.4f}, BAR={bar:.2f}')

print('\nDONE!')
