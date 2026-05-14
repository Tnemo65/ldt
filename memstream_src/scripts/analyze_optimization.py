#!/usr/bin/env python3
"""Deep analysis of benchmark results for optimization targets."""
import sys, os
sys.path.insert(0, 'c:/proj/ldt/memstream_src')
os.chdir('c:/proj/ldt')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

OUT_DIR = Path('c:/proj/ldt/results/rigorous_v1')

# Load results
df = pd.read_csv(OUT_DIR / 'results_detailed.csv')
df = df[df['algorithm'] != 'sklearn_LOF']  # Remove broken LOF

# Load CA-MemStream quick results
df_ca = pd.read_csv(OUT_DIR / 'results_ca_quick.csv')

print("=" * 80)
print("PART 1: PER-FOLD ANALYSIS")
print("=" * 80)

print("\n--- MemStream_ per-fold AUC-PR (mean across 5 seeds) ---")
ms = df[df['algorithm'] == 'MemStream_']
for diff in ['easy', 'medium', 'hard']:
    sub = ms[ms['difficulty'] == diff]
    print(f"\n{diff.upper()}:")
    pivot = sub.pivot_table(values='AUC_PR', index='fold', aggfunc='mean')
    for fold, val in pivot.iterrows():
        bar = '#' * int(float(val.values[0]) * 20)
        print(f"  Fold {fold:02d}: {float(val.values[0]):.4f} {bar}")

print("\n--- Per-fold standard deviation (stability analysis) ---")
for algo in ['MemStream_', 'HBOS', 'sklearn_IF', 'CADIFEia']:
    a = df[df['algorithm'] == algo]
    for diff in ['easy', 'medium', 'hard']:
        sub = a[a['difficulty'] == diff]
        pivot = sub.pivot_table(values='AUC_PR', index='fold', aggfunc=['mean', 'std'])
        overall_std = pivot[('AUC_PR', 'std')].mean()
        print(f"  {algo:<20} {diff:8s}: avg_fold_std={overall_std:.4f}")

print("\n" + "=" * 80)
print("PART 2: HYPERPARAMETER SENSITIVITY")
print("=" * 80)

print("""
Key hyperparameters to tune:

MemStream_ (kNN+Memory):
  - bufsz: buffer size (current: 500) - affects warmup quality
  - memsz: memory size (current: 200) - larger = more capacity
  - k: neighbors (current: 10, implicit in score_one)

sklearn_IF:
  - n_estimators: number of trees (current: 200)
  - max_samples: samples per tree (current: all)
  - contamination: prior (current: 0.05)
  - max_features: features per split (current: auto)

HBOS:
  - n_bins: histogram bins (current: 10)
  - alpha: bin width regularization (current: 0.1)

CADIFEia:
  - n_bins: context bins (current: 5)
  - n_estimators: trees per context (current: varies)
""")

print("=" * 80)
print("PART 3: ENSEMBLE OPPORTUNITIES")
print("=" * 80)

print("\n--- Score correlation between algorithms ---")
# Compute per-experiment scores
algos = ['MemStream_', 'HBOS', 'sklearn_IF', 'CADIFEia']

# For fold 1 easy
fold1_easy = df[(df['fold'] == 1) & (df['difficulty'] == 'easy') & (df['seed'] == 42)]
scores = {}
for algo in algos:
    row = fold1_easy[fold1_easy['algorithm'] == algo]
    if len(row) > 0:
        scores[algo] = row['AUC_PR'].values[0]

print("Fold 1 easy, seed 42 AUC-PR:")
for algo, score in sorted(scores.items(), key=lambda x: -x[1]):
    print(f"  {algo:<22}: {score:.4f}")

print("\n--- Algorithm diversity (fold-level rank correlation) ---")
# For each fold/difficulty, rank algorithms
for diff in ['easy']:
    print(f"\n{diff.upper()} rankings per fold:")
    sub = df[df['difficulty'] == diff]
    for fold in [1, 2, 3, 4, 5]:
        fold_data = sub[(sub['fold'] == fold) & (sub['seed'] == 42)]
        ranked = fold_data.sort_values('AUC_PR', ascending=False)
        top3 = ranked.head(3)['algorithm'].tolist()
        print(f"  Fold {fold}: {' > '.join(top3[:3])}")

print("\n" + "=" * 80)
print("PART 4: MEDIAN VS MEAN (for skewed distributions)")
print("=" * 80)

summary = df.groupby('algorithm')['AUC_PR'].agg(['mean', 'std', 'median'])
summary['cv'] = summary['std'] / summary['mean'] * 100
summary['mean_rank'] = 0
summary['median_rank'] = 0

for diff in ['easy', 'medium', 'hard']:
    sub = df[df['difficulty'] == diff]
    mean_ranks = sub.groupby('algorithm')['AUC_PR'].mean().rank(ascending=False)
    median_ranks = sub.groupby('algorithm')['AUC_PR'].median().rank(ascending=False)
    for algo in summary.index:
        if algo in mean_ranks.index:
            summary.loc[algo, 'mean_rank'] += mean_ranks[algo]
        if algo in median_ranks.index:
            summary.loc[algo, 'median_rank'] += median_ranks[algo]

summary['mean_rank'] /= 3
summary['median_rank'] /= 3
summary = summary.sort_values('median_rank')

print(f"\n{'Algorithm':<22} {'Mean':>8} {'Median':>8} {'CV%':>6} {'Mean_Rank':>10} {'Med_Rank':>10}")
print("-" * 70)
for algo, row in summary.iterrows():
    print(f"{algo:<22} {row['mean']:>8.4f} {row['median']:>8.4f} {row['cv']:>6.1f}% "
          f"{row['mean_rank']:>10.1f} {row['median_rank']:>10.1f}")

print("\n" + "=" * 80)
print("PART 5: CA-MemStream vs MemStream_ detailed")
print("=" * 80)

print("\nCA-MemStream (3 folds, seed=42):")
ca_pivot = df_ca.pivot_table(values='AUC_PR', index=['fold', 'difficulty'], columns='algorithm')
for idx, row in ca_pivot.iterrows():
    fold, diff = idx
    ms_val = row.get('CA_MemStream', np.nan)
    bar_val = row.get('CA_MemStream_BAR', np.nan)
    diff_str = f"{diff:6s}"
    print(f"  Fold {fold} {diff_str}: CA-MemStream={ms_val:.4f} BAR={bar_val:.4f}")

print("\nMemStream_ (same folds, seed=42):")
ms_same = df[(df['algorithm'] == 'MemStream_') & (df['fold'].isin([1, 2, 3])) & (df['seed'] == 42)]
ms_pivot = ms_same.pivot_table(values='AUC_PR', index=['fold', 'difficulty'])
for idx, val in ms_pivot.iterrows():
    print(f"  Fold {idx[0]} {idx[1]:6s}: MemStream_={val.values[0]:.4f}")

print("\n" + "=" * 80)
print("PART 6: ANOMALY RATE ANALYSIS")
print("=" * 80)

print("\n--- Anomaly rate per fold/difficulty ---")
for diff in ['easy', 'medium', 'hard']:
    sub = df[(df['difficulty'] == diff) & (df['fold'] == 1) & (df['algorithm'] == 'sklearn_IF')]
    n_anom = sub['n_anomalies'].values[0]
    n_test = sub['AUC_ROC'].shape[0]
    # Estimate from AUC_ROC row count
    print(f"  {diff:8s}: n_anom={n_anom}, anomaly_rate={n_anom/200000*100:.2f}%")

print("\n--- Does anomaly rate correlate with AUC-PR? ---")
# Correlation between anomaly rate and performance
for algo in ['MemStream_', 'HBOS', 'sklearn_IF']:
    a = df[df['algorithm'] == algo]
    # Merge with n_anomalies
    anom_counts = df[df['algorithm'] == 'sklearn_IF'][['fold', 'difficulty', 'n_anomalies']].drop_duplicates()
    a = a.merge(anom_counts, on=['fold', 'difficulty'])
    r, p = stats.pearsonr(a['n_anomalies'], a['AUC_PR'])
    print(f"  {algo:<20}: r={r:.3f} p={p:.4f} "
          f"({'positive' if r > 0 else 'negative'} correlation)")
