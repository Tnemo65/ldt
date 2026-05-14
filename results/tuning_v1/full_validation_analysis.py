#!/usr/bin/env python3
"""Full analysis of multi-seed validation results."""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('c:/proj/ldt/results/tuning_v1/validation_multi_seed.csv')
print(f'Rows: {len(df)}, Expected: {11*3*5*10} = {11*3*5*10}')

# Overall ranking
print('\n' + '='*80)
print('OVERALL RANKING (11 folds x 5 seeds x 3 difficulties)')
print('='*80)
overall = df.groupby('label')['AUC_PR'].agg(['mean', 'std', 'median', 'count']).reset_index()
overall['cv'] = overall['std'] / overall['mean'].clip(lower=1e-9) * 100
overall = overall.sort_values('mean', ascending=False)
print(f"{'Rank':>4} {'Label':<22} {'Mean':>8} {'Std':>8} {'Med':>8} {'CV%':>7} {'n':>5}")
for i, (_, r) in enumerate(overall.iterrows()):
    print(f"{i+1:>4} {r['label']:<22} {r['mean']:>8.4f} {r['std']:>8.4f} "
          f"{r['median']:>8.4f} {r['cv']:>7.1f}% {int(r['count']):>5}")

# Per-difficulty
print('\n' + '='*80)
print('PER-DIFFICULTY RANKING')
print('='*80)
for diff in ['easy', 'medium', 'hard']:
    sub = df[df['difficulty'] == diff]
    piv = sub.groupby('label')['AUC_PR'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    print(f"\n{diff.upper()}:")
    for i, (label, vals) in enumerate(piv.iterrows()):
        if i >= 10:
            break
        print(f"  {i+1:>2}. {label:<22}: {vals['mean']:.4f} +/- {vals['std']:.4f}")

# Statistical tests
print('\n' + '='*80)
print('STATISTICAL ANALYSIS')
print('='*80)

best_label = overall.iloc[0]['label']
best_mean = overall.iloc[0]['mean']
pivot = df.pivot_table(values='AUC_PR', index=['fold', 'difficulty', 'seed'], columns='label')

print(f"\nWilcoxon signed-rank vs {best_label} (best, mean={best_mean:.4f}):")
print(f"{'Algo':<22} {'Diff':>8} {'p-value':>10} {'Sig':>6} {'Effect':>10}")
print("-" * 60)

for algo in pivot.columns:
    if algo == best_label:
        continue
    a = pivot[best_label].dropna().values
    b = pivot[algo].dropna().values
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) < 3:
        continue
    try:
        stat, p = stats.wilcoxon(a, b, alternative='two-sided')
    except:
        p = 1.0
    diff_mean = a.mean() - b.mean()
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = diff_mean / (pooled_std + 1e-10)
    if abs(d) < 0.2:
        eff = 'neg'
    elif abs(d) < 0.5:
        eff = 'small'
    elif abs(d) < 0.8:
        eff = 'medium'
    else:
        eff = 'large'
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"  {algo:<20} {diff_mean:>+8.4f} {p:>10.4f} {sig:>6} {eff:>10}")

# Holm-Bonferroni correction
print('\nHolm-Bonferroni corrected p-values:')
pvals = {}
for algo in pivot.columns:
    if algo == best_label:
        continue
    a = pivot[best_label].dropna().values
    b = pivot[algo].dropna().values
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) < 3:
        continue
    try:
        stat, p = stats.wilcoxon(a, b, alternative='two-sided')
        pvals[algo] = p
    except:
        pvals[algo] = 1.0

sorted_p = sorted(pvals.items(), key=lambda x: x[1])
m = len(sorted_p)
adjusted = {}
for i, (algo, p) in enumerate(sorted_p):
    adj_p = min(p * (m - i), 1.0)
    if i > 0:
        adj_p = min(adj_p, adjusted[sorted_p[i-1][0]])
    adjusted[algo] = adj_p

for algo, p in sorted(adjusted.items(), key=lambda x: x[1]):
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"  {algo:<20}: adj_p={p:.4f} {sig}")

# Win/loss count
print('\nWin/Loss count vs best algorithm:')
for algo in pivot.columns:
    if algo == best_label:
        continue
    a = pivot[best_label].dropna().values
    b = pivot[algo].dropna().values
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    wins = (a > b).sum()
    losses = (a < b).sum()
    ties = (a == b).sum()
    print(f"  {algo:<22}: wins={wins} losses={losses} ties={ties} "
          f"({wins/len(a)*100:.1f}% win rate)")

# Comparison with baseline
print('\n' + '='*80)
print('COMPARISON WITH BASELINE (MemStream__baseline, sklearn_IF_baseline)')
print('='*80)
for label in overall['label'].unique():
    if label in ['MS_baseline', 'sklearn_IF_baseline', best_label]:
        continue
    ms_base = df[df['label'] == 'MS_baseline']['AUC_PR'].mean()
    if_base = df[df['label'] == 'sklearn_IF_baseline']['AUC_PR'].mean()
    this_mean = df[df['label'] == label]['AUC_PR'].mean()
    ms_improvement = (this_mean - ms_base) / ms_base * 100
    if_improvement = (this_mean - if_base) / if_base * 100
    print(f"  {label:<22}: vs MS_base={ms_improvement:>+6.1f}% vs IF_base={if_improvement:>+6.1f}%")

# Stability analysis
print('\n' + '='*80)
print('STABILITY ANALYSIS (lower CV = more stable)')
print('='*80)
stability = df.groupby('label')['AUC_PR'].agg(['mean', 'std']).reset_index()
stability['cv'] = stability['std'] / stability['mean'].clip(lower=1e-9) * 100
stability = stability.sort_values('cv')
for _, r in stability.iterrows():
    print(f"  {r['label']:<22}: CV={r['cv']:>6.1f}% mean={r['mean']:.4f}")

# Final recommendation
print('\n' + '='*80)
print('FINAL RECOMMENDATION')
print('='*80)
top3 = overall.head(3)
for i, (_, r) in enumerate(top3.iterrows()):
    print(f"  {i+1}. {r['label']}: AUC-PR={r['mean']:.4f} +/- {r['std']:.4f}")
print(f"\nBest overall: {best_label} (AUC-PR={best_mean:.4f})")

# Check if any config significantly beats the baseline
baseline_mean = df[df['label'] == 'sklearn_IF_baseline']['AUC_PR'].mean()
improvement = (best_mean - baseline_mean) / baseline_mean * 100
print(f"Improvement over sklearn_IF_baseline: {improvement:+.1f}%")

ms_base_mean = df[df['label'] == 'MS_baseline']['AUC_PR'].mean()
ms_improvement = (best_mean - ms_base_mean) / ms_base_mean * 100
print(f"Improvement over MemStream__baseline: {ms_improvement:+.1f}%")
