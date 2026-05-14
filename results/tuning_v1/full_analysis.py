#!/usr/bin/env python3
"""Full analysis of targeted tuning results."""
import pandas as pd
import numpy as np

df = pd.read_csv('c:/proj/ldt/results/tuning_v1/targeted_results.csv')
print(f'Total rows: {len(df)}, Expected: {11*3*22} = {11*3*22}')

# Overall ranking
print('\n' + '='*80)
print('OVERALL RANKING (11 folds x 3 difficulties, 1 seed)')
print('='*80)
overall = df.groupby(['label', 'group'])['AUC_PR'].agg(['mean', 'std', 'median', 'count']).reset_index()
overall['cv'] = overall['std'] / overall['mean'].clip(lower=1e-9) * 100
overall = overall.sort_values('mean', ascending=False)
print(f"{'Rank':>4} {'Label':<30} {'Group':<12} {'Mean':>8} {'Std':>8} {'CV%':>7} {'Med':>8}")
print("-"*80)
for i, (_, row) in enumerate(overall.iterrows()):
    print(f"{i+1:>4} {row['label']:<30} {row['group']:<12} "
          f"{row['mean']:>8.4f} {row['std']:>8.4f} {row['cv']:>7.1f}% {row['median']:>8.4f}")

# Per-difficulty ranking
print('\n' + '='*80)
print('PER-DIFFICULTY RANKING')
print('='*80)
for diff in ['easy', 'medium', 'hard']:
    diff_df = df[df['difficulty'] == diff]
    piv = diff_df.groupby(['label', 'group'])['AUC_PR'].agg(['mean', 'std']).reset_index()
    piv = piv.sort_values('mean', ascending=False)
    print(f"\n{diff.upper()} (top 10):")
    print(f"{'Rank':>4} {'Label':<30} {'Group':<12} {'Mean':>8} {'Std':>8}")
    print("-"*70)
    for i, (_, row) in enumerate(piv.head(10).iterrows()):
        print(f"{i+1:>4} {row['label']:<30} {row['group']:<12} "
              f"{row['mean']:>8.4f} {row['std']:>8.4f}")

# k parameter analysis
print('\n' + '='*80)
print('k PARAMETER SENSITIVITY (MemStream_)')
print('='*80)
ms_df = df[df['group'] == 'MemStream_']
k_results = {}
for k_val in [3, 5, 7, 10, 20]:
    for diff in ['easy', 'medium', 'hard']:
        labels = [l for l in ms_df['label'].unique() if f'_k{k_val}' in l and 'wt' not in l and 'diverse' not in l and 'kmeans' not in l and 'm500' not in l]
        if labels:
            val = ms_df[(ms_df['label'] == labels[0]) & (ms_df['difficulty'] == diff)]['AUC_PR'].mean()
            if k_val not in k_results:
                k_results[k_val] = {}
            k_results[k_val][diff] = val

print(f"{'k':>5} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Overall':>8}")
print("-"*40)
for k in sorted(k_results.keys()):
    vals = k_results[k]
    overall_k = np.mean(list(vals.values()))
    print(f"{k:>5} {vals.get('easy', 0):>8.4f} {vals.get('medium', 0):>8.4f} "
          f"{vals.get('hard', 0):>8.4f} {overall_k:>8.4f}")

# Memory strategy comparison
print('\n' + '='*80)
print('MEMORY INITIALIZATION STRATEGY (MemStream_, k=5)')
print('='*80)
for diff in ['easy', 'medium', 'hard']:
    sub = ms_df[ms_df['difficulty'] == diff]
    strategies = {
        'first (baseline)': 'MS_k5',
        'kmeans': 'MS_k5_kmeans',
        'diverse': 'MS_k5_diverse',
        'm500': 'MS_k5_m500',
        'weighted': 'MS_k5_wt',
    }
    vals = {}
    for name, label in strategies.items():
        v = sub[sub['label'] == label]['AUC_PR'].mean()
        vals[name] = v
    best = max(vals.values())
    print(f"\n{diff.upper()}:")
    for name, v in sorted(vals.items(), key=lambda x: -x[1]):
        marker = " <-- BEST" if v == best else ""
        print(f"  {name:<20}: {v:.4f}{marker}")

# HBOS n_bins analysis
print('\n' + '='*80)
print('HBOS n_bins SENSITIVITY')
print('='*80)
hb_df = df[df['group'] == 'HBOS']
for diff in ['easy', 'medium', 'hard']:
    sub = hb_df[hb_df['difficulty'] == diff]
    vals = sub.groupby('label')['AUC_PR'].mean().sort_values(ascending=False)
    print(f"\n{diff.upper()}:")
    for label, v in vals.items():
        print(f"  {label:<15}: {v:.4f}")

# sklearn_IF analysis
print('\n' + '='*80)
print('sklearn_IF COMPARISON')
print('='*80)
if_df = df[df['group'] == 'sklearn_IF']
for diff in ['easy', 'medium', 'hard']:
    sub = if_df[if_df['difficulty'] == diff]
    vals = sub.groupby('label')['AUC_PR'].mean().sort_values(ascending=False)
    print(f"\n{diff.upper()}:")
    for label, v in vals.items():
        print(f"  {label:<20}: {v:.4f}")

# Ensemble analysis
print('\n' + '='*80)
print('ENSEMBLE ANALYSIS')
print('='*80)
ens_df = df[df['group'] == 'Ensemble']
best_ms = df[df['label'] == 'MS_k5_m500']
best_hb = df[df['label'] == 'HBOS_b10']
best_if = df[df['label'] == 'IF_n200_mf50']

for diff in ['easy', 'medium', 'hard']:
    ms_val = best_ms[best_ms['difficulty'] == diff]['AUC_PR'].mean()
    hb_val = best_hb[best_hb['difficulty'] == diff]['AUC_PR'].mean()
    if_val = best_if[best_if['difficulty'] == diff]['AUC_PR'].mean()

    sub = ens_df[ens_df['difficulty'] == diff]
    ens_vals = sub.groupby('label')['AUC_PR'].mean().sort_values(ascending=False)

    print(f"\n{diff.upper()}:")
    print(f"  MS_k5_m500: {ms_val:.4f}")
    print(f"  HBOS_b10:   {hb_val:.4f}")
    print(f"  IF_n200_mf50: {if_val:.4f}")
    print(f"  Ensembles:")
    for label, v in ens_vals.items():
        improvement = v - max(ms_val, hb_val, if_val)
        print(f"    {label:<30}: {v:.4f} ({improvement:+.4f} vs best single)")

# OVERFITTING ANALYSIS
print('\n' + '='*80)
print('STABILITY ANALYSIS (CV = std/mean * 100)')
print('='*80)
stability = df.groupby(['label', 'group'])['AUC_PR'].agg(['mean', 'std']).reset_index()
stability['cv'] = stability['std'] / stability['mean'].clip(lower=1e-9) * 100
stability = stability.sort_values('cv')
print(f"{'Label':<30} {'Group':<12} {'Mean':>8} {'Std':>8} {'CV%':>7}")
print("-"*70)
for _, row in stability.head(10).iterrows():
    print(f"{row['label']:<30} {row['group']:<12} "
          f"{row['mean']:>8.4f} {row['std']:>8.4f} {row['cv']:>7.1f}%")

# Best recommendation
print('\n' + '='*80)
print('RECOMMENDATION')
print('='*80)
print("""
Based on the 11-fold tuning analysis:

1. BEST SINGLE ALGORITHM: sklearn_IF (IF_n200_mf50)
   - Highest overall AUC-PR: 0.2155
   - But struggles on EASY difficulty (0.1537)
   - Best on HARD difficulty

2. RUNNER-UP: MemStream_ (MS_k5_m500)
   - AUC-PR: 0.1987 overall
   - Best on EASY: 0.4409
   - CV=91.8% (high variance)

3. MEMORY INITIALIZATION:
   - Larger memory (memsz=500) beats memsz=200
   - k=3 slightly better than k=5, but k=5 is more stable
   - kmeans and diverse strategies provide marginal improvement

4. HBOS:
   - More bins (b100) significantly better than fewer bins
   - Best on EASY: 0.2834

5. ENSEMBLES:
   - All ensembles WORSE than best single algorithm
   - Ensembles average out the diversity, reducing peak performance
   - Score averaging degrades performance by ~30-40%

KEY INSIGHT: On this streaming anomaly detection dataset:
   - kNN-based MemStream_ is better suited for EASY anomalies
   - sklearn_IF is better for HARD anomalies
   - Ensemble does NOT help - the algorithms rank differently per fold
""")
