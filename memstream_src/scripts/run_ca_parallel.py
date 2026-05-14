#!/usr/bin/env python3
"""Run CA-MemStream benchmark - parallel version using joblib."""
import sys, os, time
sys.path.insert(0, 'c:/proj/ldt/memstream_src')
os.chdir('c:/proj/ldt')

from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Load benchmark module
spec = importlib.util.spec_from_file_location("b", "c:/proj/ldt/memstream_src/scripts/benchmark_rigorous.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

CACHE_DIR = Path('c:/proj/ldt/results/v5_clean/fold_cache')
OUT_DIR = Path('c:/proj/ldt/results/rigorous_v1')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Configuration
# ============================================================
SEEDS = [42, 123, 456]  # 3 seeds
DIFFICULTIES = ['easy', 'medium', 'hard']
N_FOLDS = 11

# Load fold entries
metadata = b.load_metadata(CACHE_DIR)
fold_entries = []
for fold_num in range(1, N_FOLDS + 1):
    for diff in DIFFICULTIES:
        entry = next((e for e in metadata if e['fold'] == fold_num and e['difficulty'] == diff), None)
        if entry:
            fold_entries.append(entry)

ALGOS_CA = [b.CA_MemStream, b.CA_MemStream_BAR]

print(f"CA-MemStream benchmark: {len(ALGOS_CA)} algos x {len(fold_entries)} folds x {len(SEEDS)} seeds = "
      f"{len(ALGOS_CA)*len(fold_entries)*len(SEEDS)} experiments")
print(f"Using 3 seeds for speed (still statistically valid)")
print(f"Output: {OUT_DIR}")

total = len(ALGOS_CA) * len(fold_entries) * len(SEEDS)
print(f"Starting at {time.strftime('%H:%M:%S')}")


def run_experiment(args):
    """Run a single experiment."""
    algo_cls, fold_num, diff, seed = args
    try:
        data = b.load_fold_data(fold_num, diff, CACHE_DIR)
        row = b.evaluate_algorithm(algo_cls, data, seed, use_calibration=True)
        row['algorithm'] = algo_cls.name
        return row
    except Exception as e:
        return {
            'fold': fold_num, 'difficulty': diff, 'seed': seed,
            'algorithm': algo_cls.name,
            'AUC_ROC': np.nan, 'AUC_PR': np.nan, 'error': str(e)
        }


# Build experiment list
experiments = []
for algo_cls in ALGOS_CA:
    for entry in fold_entries:
        for seed in SEEDS:
            experiments.append((algo_cls, entry['fold'], entry['difficulty'], seed))

print(f"Running {len(experiments)} experiments with joblib (n_jobs=-1)...")
t_start = time.perf_counter()

# Run in parallel
results = Parallel(n_jobs=-1, verbose=1, backend='loky')(
    delayed(run_experiment)(exp) for exp in experiments
)

t_total = time.perf_counter() - t_start

# Save results
df = pd.DataFrame(results)
df.to_csv(OUT_DIR / 'results_ca_memstream.csv', index=False)

print(f"\nDone in {t_total/60:.1f} min")
print(f"Saved: {OUT_DIR / 'results_ca_memstream.csv'}")
print(f"Results: {len(df)} rows")

# Quick summary
if not df.empty and 'AUC_PR' in df.columns:
    summary = df.groupby('algorithm').agg(
        AUC_ROC_mean=('AUC_ROC', 'mean'),
        AUC_PR_mean=('AUC_PR', 'mean'),
    ).round(4)
    print("\nSummary:")
    print(summary.to_string())
    summary.to_csv(OUT_DIR / 'summary_ca_memstream.csv')
