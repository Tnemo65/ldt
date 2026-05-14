#!/usr/bin/env python3
"""Run CA-MemStream benchmark - 3 seeds, sequential."""
import sys, os, time
sys.path.insert(0, 'c:/proj/ldt/memstream_src')
os.chdir('c:/proj/ldt')

from pathlib import Path
import importlib.util
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

spec = importlib.util.spec_from_file_location("b", "c:/proj/ldt/memstream_src/scripts/benchmark_rigorous.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

CACHE_DIR = Path('c:/proj/ldt/results/v5_clean/fold_cache')
OUT_DIR = Path('c:/proj/ldt/results/rigorous_v1')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]  # 3 seeds
DIFFICULTIES = ['easy', 'medium', 'hard']
N_FOLDS = 11

metadata = b.load_metadata(CACHE_DIR)
fold_entries = []
for fold_num in range(1, N_FOLDS + 1):
    for diff in DIFFICULTIES:
        entry = next((e for e in metadata if e['fold'] == fold_num and e['difficulty'] == diff), None)
        if entry:
            fold_entries.append(entry)

ALGOS_CA = [b.CA_MemStream, b.CA_MemStream_BAR]

total = len(ALGOS_CA) * len(fold_entries) * len(SEEDS)
done = 0
all_results = []

log_file = OUT_DIR / 'ca_benchmark_log.txt'
log = open(log_file, 'w')

def log_print(msg):
    print(msg)
    sys.stdout.flush()
    log.write(msg + '\n')
    log.flush()

log_print(f"CA-MemStream benchmark: {total} experiments")
log_print(f"Seeds: {SEEDS}, Alg: {[a.name for a in ALGOS_CA]}")
log_print(f"Started: {time.strftime('%H:%M:%S')}")

t_start = time.perf_counter()

for algo_cls in ALGOS_CA:
    t_algo = time.perf_counter()
    log_print(f"\n[{algo_cls.name}]")

    for entry in fold_entries:
        fold_num = entry['fold']
        diff = entry['difficulty']

        data = b.load_fold_data(fold_num, diff, CACHE_DIR)

        for seed in SEEDS:
            row = b.evaluate_algorithm(algo_cls, data, seed, use_calibration=True)
            row['algorithm'] = algo_cls.name
            all_results.append(row)
            done += 1

            elapsed = time.perf_counter() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate / 60 if rate > 0 else 0
            pct = done / total * 100
            log_print(f"  {done:3d}/{total} ({pct:5.1f}%) {rate:.1f}/s ETA={eta:.1f}min | {algo_cls.name} fold{fold_num:02d}/{diff} seed={seed}")

            if done % 33 == 0:
                df = pd.DataFrame(all_results)
                df.to_csv(OUT_DIR / 'results_ca_memstream.csv', index=False)

    df = pd.DataFrame(all_results)
    df.to_csv(OUT_DIR / 'results_ca_memstream.csv', index=False)
    t_elapsed = time.perf_counter() - t_algo
    log_print(f"  [{algo_cls.name}] done in {t_elapsed/60:.1f}min")

t_total = time.perf_counter() - t_start
log_print(f"\nTotal: {total} in {t_total/60:.1f}min")
log.close()

df = pd.DataFrame(all_results)
df.to_csv(OUT_DIR / 'results_ca_memstream.csv', index=False)

print("\n=== SUMMARY ===")
if not df.empty:
    summary = df.groupby('algorithm').agg(
        mean=('AUC_PR', 'mean'),
        std=('AUC_PR', 'std'),
    ).round(4)
    print(summary.to_string())
