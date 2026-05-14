#!/usr/bin/env python3
"""Analyze v7 benchmark results comprehensively."""
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from pathlib import Path

df = pd.read_csv('c:/proj/ldt/results/v7/checkpoint_v7.csv')

print(f"Total rows: {len(df)}")
print(f"Algorithms: {sorted(df['algorithm'].unique())}")
print(f"Difficulties: {sorted(df['difficulty'].unique())}")
print(f"Folds: {sorted(df['fold'].unique())}")
print(f"Seeds: {sorted(df['seed'].unique())}")
print(f"Label budgets: {sorted(df['label_budget'].unique())}")
print()

# Separate batch and streaming
batch = df[df['label_budget'] == 0].copy()
streaming = df[df['label_budget'] == 500].copy()

print("=" * 80)
print("BATCH ALGORITHMS (label_budget=0)")
print("=" * 80)

# Overall
print("\n--- Overall (batch) ---")
overall = batch.groupby('algorithm')['AUC_PR'].agg(['mean', 'std', 'median']).sort_values('mean', ascending=False)
print(overall)

# Per difficulty
for diff in ['easy', 'medium', 'hard']:
    sub = batch[batch['difficulty'] == diff]
    piv = sub.groupby('algorithm')['AUC_PR'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    print(f"\n--- {diff.upper()} (batch) ---")
    for algo, vals in piv.iterrows():
        print(f"  {algo:<20}: {vals['mean']:.4f} +/- {vals['std']:.4f}")

print("\n" + "=" * 80)
print("STREAMING ALGORITHMS (label_budget=500)")
print("=" * 80)

# Overall
print("\n--- Overall (streaming) ---")
overall_s = streaming.groupby('algorithm')['AUC_PR'].agg(['mean', 'std', 'median']).sort_values('mean', ascending=False)
print(overall_s)

# Per difficulty
for diff in ['easy', 'medium', 'hard']:
    sub = streaming[streaming['difficulty'] == diff]
    piv = sub.groupby('algorithm')['AUC_PR'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    print(f"\n--- {diff.upper()} (streaming) ---")
    for algo, vals in piv.iterrows():
        print(f"  {algo:<20}: {vals['mean']:.4f} +/- {vals['std']:.4f}")

# Statistical tests
print("\n" + "=" * 80)
print("STATISTICAL SIGNIFICANCE")
print("=" * 80)

# Friedman test on batch algorithms - align by fold, difficulty, seed
batch_algos = ['DenoisingAE', 'AE+IF', 'CA-DIF-EIA', 'IF-baseline', 'sklearn_IF', 'sklearn_OCSVM', 'Random']

# Create aligned arrays for Friedman
batch_friedman_data = []
for algo in batch_algos:
    sub = batch[batch['algorithm'] == algo].sort_values(['fold', 'difficulty', 'seed'])
    vals = sub['AUC_PR'].values
    if len(vals) > 0:
        batch_friedman_data.append(vals)

if len(batch_friedman_data) >= 2:
    min_len = min(len(v) for v in batch_friedman_data)
    aligned_data = [v[:min_len] for v in batch_friedman_data]
    stat, p = friedmanchisquare(*aligned_data)
    print(f"\nFriedman test (batch, n={min_len}): chi2={stat:.2f} p={p:.2e}")

# Streaming Friedman
stream_algos = ['MemStream', 'sHST-River', 'CA-DIF-EIA-Stream', 'Random']
stream_friedman_data = []
for algo in stream_algos:
    sub = streaming[streaming['algorithm'] == algo].sort_values(['fold', 'difficulty', 'seed'])
    vals = sub['AUC_PR'].values
    if len(vals) > 0:
        stream_friedman_data.append(vals)

if len(stream_friedman_data) >= 2:
    min_len = min(len(v) for v in stream_friedman_data)
    aligned_data = [v[:min_len] for v in stream_friedman_data]
    stat, p = friedmanchisquare(*aligned_data)
    print(f"Friedman test (streaming, n={min_len}): chi2={stat:.2f} p={p:.2e}")

# Wilcoxon on batch
print("\nWilcoxon signed-rank (batch, vs DenoisingAE):")
de_scores = batch[batch['algorithm'] == 'DenoisingAE'].sort_values(['fold', 'difficulty', 'seed'])['AUC_PR'].values
for algo in ['AE+IF', 'CA-DIF-EIA', 'IF-baseline', 'sklearn_IF']:
    algo_scores = batch[batch['algorithm'] == algo].sort_values(['fold', 'difficulty', 'seed'])['AUC_PR'].values
    if len(algo_scores) == len(de_scores):
        stat, p = wilcoxon(de_scores, algo_scores, alternative='greater')
        diff = de_scores.mean() - algo_scores.mean()
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  DenoisingAE vs {algo:<15}: diff={diff:+.4f} p={p:.4f} {sig}")

# Wilcoxon on streaming
print("\nWilcoxon signed-rank (streaming, vs MemStream):")
ms_scores = streaming[streaming['algorithm'] == 'MemStream'].sort_values(['fold', 'difficulty', 'seed'])['AUC_PR'].values
for algo in ['sHST-River', 'CA-DIF-EIA-Stream']:
    algo_scores = streaming[streaming['algorithm'] == algo].sort_values(['fold', 'difficulty', 'seed'])['AUC_PR'].values
    if len(algo_scores) == len(ms_scores):
        stat, p = wilcoxon(ms_scores, algo_scores, alternative='greater')
        diff = ms_scores.mean() - algo_scores.mean()
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  MemStream vs {algo:<20}: diff={diff:+.4f} p={p:.4f} {sig}")

# Label budget analysis
print("\n" + "=" * 80)
print("LABEL BUDGET IMPACT (MemStream)")
print("=" * 80)
for budget in sorted(df['label_budget'].unique()):
    sub = df[(df['algorithm'] == 'MemStream') & (df['label_budget'] == budget)]
    mean = sub['AUC_PR'].mean()
    std = sub['AUC_PR'].std()
    print(f"  Budget {budget:5d}: AUC-PR={mean:.4f} +/- {std:.4f}")
