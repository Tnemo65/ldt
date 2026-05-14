"""Fast v7 benchmark: 2 folds, 2 seeds, 2 budgets for quick results."""
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
    MONTHS, DIFFICULTIES,
    TRAIN_N, VAL_N, TEST_N, ANOMALY_N,
    ANOMALY_PARAMS,
    SklearnIF, SklearnOCSVM, DenoisingAE, CADIFEiaBatch,
    RandomBaseline, sHST_River, MemStream, CADIFEiaStream,
    METRICS,
)

OUT_DIR = Path(r'C:\proj\ldt\results\v7')
FAST_SEEDS = [42, 123]
FAST_BUDGETS = [0, 500]
FAST_FOLDS = [2, 3]  # test months: Feb, Mar

# Clean slate
for f in ['checkpoint_fast.csv', 'benchmark_results_fast.csv',
          'benchmark_fast_results.md',
          'fig_fast_overview.png', 'fig_fast_difficulty.png',
          'fig_fast_ablation.png', 'fig_fast_bar.png', 'fig_fast_pareto.png']:
    path = OUT_DIR / f
    if path.exists():
        path.unlink()
        print(f'Removed old {f}')

print('='*70)
print('FAST v7 BENCHMARK (2 folds, 2 seeds, 2 budgets)')
print('='*70)

# Load data
print('\n[1/4] Loading data...', flush=True)
monthly, monthly_X = [], []
for m in MONTHS:
    df = bm.clean(bm.load_month(2024, m))
    X = bm.features(df)
    monthly.append(df)
    monthly_X.append(X.astype(np.float32))
    print(f'  Month {m:02d}: {len(df):,} records', flush=True)

# Build jobs
print('\n[2/4] Building jobs...', flush=True)
jobs = []
for fold_idx, test_month in enumerate(MONTHS[1:], 1):
    if test_month not in FAST_FOLDS:
        continue
    train_months = MONTHS[:test_month - 1]
    val_month = train_months[-1]
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
        seed_s = FAST_SEEDS[fold_idx % len(FAST_SEEDS)]

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

        batch_info = [
            ('Random',        RandomBaseline,    {}),
            ('sklearn_IF',    SklearnIF,        {}),
            ('sklearn_OCSVM', SklearnOCSVM,     {}),
            ('DenoisingAE',   DenoisingAE,      {}),
            ('CA-DIF-EIA',    CADIFEiaBatch,   {'ablation': 'full'}),
            ('AE+IF',         CADIFEiaBatch,   {'ablation': 'ae_if'}),
            ('IF-baseline',   CADIFEiaBatch,   {'ablation': 'baseline'}),
        ]
        for name, cls, kw in batch_info:
            for seed in FAST_SEEDS:
                jobs.append({
                    'type': 'batch', 'fold': fold_idx, 'test_month': test_month,
                    'diff': diff, 'algo_name': name, 'algo_cls': cls,
                    'seed': seed,
                    'X_train': X_train_s.copy(),
                    'X_val': X_val_s.copy(),
                    'X_test': X_test_s.copy(),
                    'y_val': y_val.copy(),
                    'y_test': y_labels.copy(),
                    'kwargs': kw, 'label_budget': 0,
                })

        stream_info = [
            ('Random',            RandomBaseline,    {}),
            ('sHST-River',       sHST_River,       {}),
            ('MemStream',        MemStream,         {}),
            ('CA-DIF-EIA-Stream', CADIFEiaStream,  {}),
        ]
        for name, cls, kw in stream_info:
            for seed in FAST_SEEDS:
                for lb in FAST_BUDGETS:
                    jobs.append({
                        'type': 'stream', 'fold': fold_idx, 'test_month': test_month,
                        'diff': diff, 'algo_name': name, 'algo_cls': cls,
                        'seed': seed,
                        'X_train': X_train_s.copy(),
                        'X_val': X_val_s.copy(),
                        'X_test': X_test_s.copy(),
                        'y_val': y_val.copy(),
                        'y_test': y_labels.copy(),
                        'kwargs': kw, 'label_budget': lb,
                    })

print(f'  Total: {len(jobs)} jobs')

# Run
print(f'\n[3/4] Running {len(jobs)} jobs...', flush=True)
t0 = time.perf_counter()
results = []

for i, job in enumerate(jobs):
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
        res['error'] = str(e)[:120]
        print(f'  ERROR job {i} ({job["algo_name"]}): {e}', flush=True)

    res.update({
        'fold': job['fold'], 'month': job['test_month'],
        'difficulty': job['diff'], 'algorithm': job['algo_name'],
        'seed': job['seed'],
        'label_budget': job['label_budget'],
    })
    results.append(res)

    if (i + 1) % 10 == 0:
        elapsed = time.perf_counter() - t0
        rate = (i + 1) / elapsed
        remain = (len(jobs) - i - 1) / rate / 60
        print(f'  {i+1}/{len(jobs)} ({rate:.2f}/s, ~{remain:.1f}m left)', flush=True)
        pd.DataFrame(results).to_csv(OUT_DIR / 'checkpoint_fast.csv', index=False)
    gc.collect()

t_done = time.perf_counter() - t0
print(f'\n  Done in {t_done/60:.1f} min', flush=True)

# Save
bench_df = pd.DataFrame(results)
bench_df.to_csv(OUT_DIR / 'benchmark_results_fast.csv', index=False)

# Report
print('\n' + '='*60)
print('RESULTS')
print('='*60)

print('\n--- AUC-PR by Algorithm (overall) ---')
print(bench_df.groupby('algorithm')['AUC_PR'].agg(['mean','std','count']).sort_values('mean', ascending=False).to_string())

print('\n--- Batch AUC-PR by Difficulty ---')
batch = bench_df[bench_df['algorithm'].isin(['sklearn_IF','sklearn_OCSVM','DenoisingAE','CA-DIF-EIA','AE+IF','IF-baseline','Random'])]
pivot = batch.pivot_table(index='algorithm', columns='difficulty', values='AUC_PR', aggfunc='mean')
print(pivot.round(4).to_string())

print('\n--- Streaming AUC-PR by Label Budget ---')
stream = bench_df[bench_df['algorithm'].isin(['sHST-River','MemStream','CA-DIF-EIA-Stream','Random'])]
for lb in sorted(stream['label_budget'].unique()):
    sub = stream[stream['label_budget'] == lb]
    print(f'\n  Budget {int(lb)}:')
    for algo, row in sub.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).items():
        bar = 100 * row / lb if lb > 0 else 0
        print(f'    {algo}: AUC_PR={row:.4f}, BAR={bar:.2f}')

print('\n--- Errors ---')
errors = bench_df[bench_df['error'].notna()]
print(f'Total errors: {len(errors)}')
if len(errors) > 0:
    for _, row in errors.iterrows():
        print(f'  {row["algorithm"]} fold={row["fold"]} diff={row["difficulty"]}: {row["error"]}')

print('\nDONE!')
