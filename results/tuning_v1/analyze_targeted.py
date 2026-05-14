#!/usr/bin/env python3
"""Analyze partial results."""
import pandas as pd
import numpy as np

df = pd.read_csv('c:/proj/ldt/results/tuning_v1/targeted_results.csv')
print(f'Rows: {len(df)}')

# Overall
print('\n=== OVERALL (2 folds) ===')
for g in ['MemStream_', 'HBOS', 'sklearn_IF', 'Ensemble']:
    sub = df[df['group'] == g]
    top = sub.groupby('label')['AUC_PR'].mean().sort_values(ascending=False)
    for label, val in top.items():
        std = sub[sub['label'] == label]['AUC_PR'].std()
        print(f'  [{g}] {label}: {val:.4f} +/- {std:.4f}')

# k analysis
print('\n=== k PARAMETER (MemStream_) ===')
for diff in ['easy', 'medium', 'hard']:
    sub = df[(df['group'] == 'MemStream_') & (df['difficulty'] == diff)]
    k_results = {}
    for label in sub['label'].unique():
        if '_k' in label:
            k = int(label.split('_k')[1].split('_')[0])
        else:
            continue
        val = sub[sub['label'] == label]['AUC_PR'].mean()
        k_results[k] = val
    print(f'  {diff}: {sorted(k_results.items())}')

# Ensemble vs single
print('\n=== ENSEMBLE vs SINGLE ===')
for diff in ['easy', 'medium', 'hard']:
    ms5 = df[(df['label'] == 'MS_k5') & (df['difficulty'] == diff)]['AUC_PR'].mean()
    hb10 = df[(df['label'] == 'HBOS_b10') & (df['difficulty'] == diff)]['AUC_PR'].mean()
    best_ms = df[(df['group'] == 'MemStream_') & (df['difficulty'] == diff)].groupby('label')['AUC_PR'].mean().max()
    ens = df[(df['label'] == 'Ens_MS5_HBOS10') & (df['difficulty'] == diff)]['AUC_PR'].mean()
    print(f'  {diff}: best_MS={best_ms:.4f} HBOS_b10={hb10:.4f} Ens={ens:.4f}')
