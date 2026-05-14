#!/usr/bin/env python3
"""Run full baseline benchmark with incremental saves and real-time output."""
import sys, os, time
sys.path.insert(0, 'c:/proj/ldt/memstream_src')
os.chdir('c:/proj/ldt')

from pathlib import Path
import importlib.util
import pandas as pd

spec = importlib.util.spec_from_file_location("b", "c:/proj/ldt/memstream_src/scripts/benchmark_rigorous.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

CACHE_DIR = Path('c:/proj/ldt/results/v5_clean/fold_cache')
OUT_DIR = Path('c:/proj/ldt/results/rigorous_v1')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]
DIFFICULTIES = ['easy', 'medium', 'hard']
N_FOLDS = 11

metadata = b.load_metadata(CACHE_DIR)
fold_entries = []
for fold_num in range(1, N_FOLDS + 1):
    for diff in DIFFICULTIES:
        entry = next((e for e in metadata if e['fold'] == fold_num and e['difficulty'] == diff), None)
        if entry:
            fold_entries.append(entry)

ALGOS_BASELINE = [
    b.SklearnIF,
    b.SklearnLOF,
    b.MemStream_,
    b.sHST_River,
    b.IForestASD_,
    b.sHST_Mem_Ensemble,
    b.CADIFEia,
    b.HBOS,
]

total = len(ALGOS_BASELINE) * len(fold_entries) * len(SEEDS)
done = 0
all_results = []

log_file = OUT_DIR / 'benchmark_log.txt'
log = open(log_file, 'w')

def log_print(msg):
    print(msg)
    log.write(msg + '\n')
    log.flush()

log_print(f"Starting benchmark: {total} experiments")
log_print(f"GPU: {b.GPU_OK}, Device: {b.DEVICE}")
log_print(f"Algorithms: {[a.name for a in ALGOS_BASELINE]}")
log_print("=" * 60)

t_start = time.perf_counter()

for algo_cls in ALGOS_BASELINE:
    t_algo = time.perf_counter()
    log_print(f"\n[{algo_cls.name}]")

    for entry in fold_entries:
        fold_num = entry['fold']
        diff = entry['difficulty']

        # Load data once per fold (share across seeds)
        data = b.load_fold_data(fold_num, diff, CACHE_DIR)
        if data['bad_features']:
            log_print(f"  Fold {fold_num:02d}/{diff}: bad_features={data['bad_features']}")

        for seed in SEEDS:
            row = b.evaluate_algorithm(algo_cls, data, seed, use_calibration=True)
            row['algorithm'] = algo_cls.name
            all_results.append(row)
            done += 1

            if done % 50 == 0:
                elapsed = time.perf_counter() - t_start
                rate = done / elapsed
                eta = (total - done) / rate / 60
                pct = done / total * 100
                log_print(f"  Progress: {done}/{total} ({pct:.1f}%) "
                          f"{rate:.1f}/s ETA={eta:.1f}min")

            # Incremental save every 100 rows
            if done % 100 == 0:
                df = pd.DataFrame(all_results)
                df.to_csv(OUT_DIR / 'results_detailed.csv', index=False)

    # Save after each algo
    df = pd.DataFrame(all_results)
    df.to_csv(OUT_DIR / 'results_detailed.csv', index=False)
    t_algo_elapsed = time.perf_counter() - t_algo
    log_print(f"  [{algo_cls.name}] done in {t_algo_elapsed/60:.1f}min")

# Final save
df = pd.DataFrame(all_results)
df.to_csv(OUT_DIR / 'results_detailed.csv', index=False)

t_total = time.perf_counter() - t_start
log_print(f"\nTotal: {total} experiments in {t_total/60:.1f}min")
log_print(f"Average: {total/t_total:.1f} experiments/sec")
log_print(f"Results: {OUT_DIR / 'results_detailed.csv'}")
log.close()

# Quick summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if not df.empty:
    summary = df.groupby('algorithm').agg(
        AUC_ROC_mean=('AUC_ROC', 'mean'),
        AUC_ROC_std=('AUC_ROC', 'std'),
        AUC_PR_mean=('AUC_PR', 'mean'),
        AUC_PR_std=('AUC_PR', 'std'),
    ).round(4)
    print(summary.to_string())
    summary.to_csv(OUT_DIR / 'results_summary.csv')
