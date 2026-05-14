#!/usr/bin/env python3
"""Debug: check results file and fix issues."""
import pandas as pd
import numpy as np

df = pd.read_csv('c:/proj/ldt/results/tuning_v1/targeted_results.csv')
print(f'Total rows: {len(df)}')

# Check which folds have what
for fold in sorted(df['fold'].unique()):
    sub = df[df['fold'] == fold]
    for diff in sorted(sub['difficulty'].unique()):
        s = sub[sub['difficulty'] == diff]
        nan_count = s['AUC_ROC'].isna().sum()
        print(f'Fold {fold} {diff}: {len(s)} configs, {nan_count} NaN')

# Check errors
errors = df[df['AUC_ROC'].isna()]
if not errors.empty:
    print(f'\nError examples (first 5):')
    for _, r in errors.head(5).iterrows():
        print(f'  {r["label"]} fold={r["fold"]} {r["difficulty"]}: AUC_ROC=NaN')

# Check the MS configs that got NaN - see if they got any scores
ms_nan = df[(df['group'] == 'MemStream_') & (df['AUC_ROC'].isna())]
ms_ok = df[(df['group'] == 'MemStream_') & (~df['AUC_ROC'].isna())]
print(f'\nMS configs with NaN: {ms_nan["label"].unique()}')
print(f'MS configs OK: {ms_ok["label"].unique()}')

# Check if fold 4 data has a different structure
f4 = df[df['fold'] == 4]
print(f'\nFold 4 configs: {len(f4)}')
print(f'Fold 4 groups: {f4["group"].unique()}')
for label in f4['label'].unique():
    sub = f4[f4['label'] == label]
    nan = sub['AUC_ROC'].isna().sum()
    print(f'  {label}: {len(sub)} rows, {nan} NaN')
