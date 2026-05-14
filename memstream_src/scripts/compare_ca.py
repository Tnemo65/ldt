#!/usr/bin/env python3
"""Compare CA-MemStream vs baselines on same folds."""
import sys, os
sys.path.insert(0, 'c:/proj/ldt/memstream_src')
os.chdir('c:/proj/ldt')

import pandas as pd
from pathlib import Path
import numpy as np

OUT_DIR = Path('c:/proj/ldt/results/rigorous_v1')

# Load CA-MemStream results
df_ca = pd.read_csv(OUT_DIR / 'results_ca_quick.csv')

# Load baseline results
df_base = pd.read_csv(OUT_DIR / 'results_detailed.csv')

print("=" * 80)
print("COMPARISON: CA-MemStream vs Baselines (3 folds, seed=42)")
print("=" * 80)

# Filter baselines to same folds/seeds
base_same = df_base[
    (df_base['seed'] == 42) &
    (df_base['fold'].isin([1, 2, 3]))
]

print(f"\nBaseline experiments: {len(base_same)}")
print(f"CA-MemStream experiments: {len(df_ca)}")

# Summary by difficulty
print(f"\n{'Algorithm':<25} {'EASY AUC-PR':>15} {'MEDIUM AUC-PR':>15} {'HARD AUC-PR':>15}")
print("-" * 75)

all_algos = ['sklearn_IF', 'MemStream_', 'sHST_Mem_Ensemble', 'HBOS', 'CADIFEia',
             'sHST_River', 'IForestASD_', 'CA_MemStream', 'CA_MemStream_BAR']

for algo in all_algos:
    if algo in ['CA_MemStream', 'CA_MemStream_BAR']:
        src = df_ca
    else:
        src = base_same

    vals = {}
    for diff in ['easy', 'medium', 'hard']:
        v = src[(src['algorithm'] == algo) & (src['difficulty'] == diff)]['AUC_PR']
        if len(v) > 0:
            vals[diff] = f"{v.mean():.4f}±{v.std():.3f}"
        else:
            vals[diff] = "N/A"

    marker = ">>>" if algo in ['CA_MemStream', 'CA_MemStream_BAR'] else "   "
    print(f"{marker} {algo:<22} {vals.get('easy', 'N/A'):>15} {vals.get('medium', 'N/A'):>15} {vals.get('hard', 'N/A'):>15}")

print()
print("=" * 80)
print("KEY COMPARISON: CA-MemStream vs MemStream_ (same kNN-based Memory Module)")
print("=" * 80)

for diff in ['easy', 'medium', 'hard']:
    ca = df_ca[(df_ca['algorithm'] == 'CA_MemStream') & (df_ca['difficulty'] == diff)]['AUC_PR']
    ms = base_same[(base_same['algorithm'] == 'MemStream_') & (base_same['difficulty'] == diff)]['AUC_PR']

    if len(ca) > 0 and len(ms) > 0:
        improvement = (ca.mean() - ms.mean()) / ms.mean() * 100
        print(f"  {diff.upper()}: CA-MemStream={ca.mean():.4f} vs MemStream_={ms.mean():.4f} "
              f"({improvement:+.1f}%)")

print()
print("NOTE: CA_MemStream_BAR produces identical results to CA_MemStream")
print("      (bar_rate=0%, drift_events=0 - BAR only activates in streaming mode)")
print()
print("NOTE: CA-MemStream (AE+Memory) vs MemStream_ (kNN+Memory):")
print("      - Same memory module, different anomaly scoring")
print("      - AE captures non-linear reconstruction error")
print("      - kNN captures density-based distance")
