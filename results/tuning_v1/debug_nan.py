#!/usr/bin/env python3
import pandas as pd
import numpy as np

df = pd.read_csv('c:/proj/ldt/results/tuning_v1/targeted_results.csv')
print(f'Rows: {len(df)}')

# Check NaN distribution
print('\nNaN counts by label:')
for label in sorted(df['label'].unique()):
    sub = df[df['label'] == label]
    nan_auc_roc = sub['AUC_ROC'].isna().sum()
    nan_auc_pr = sub['AUC_PR'].isna().sum()
    print(f'  {label}: NaN_AUC_ROC={nan_auc_roc}/{len(sub)} NaN_AUC_PR={nan_auc_pr}/{len(sub)}')

# Fold-level detail for MemStream_ configs
print('\nMemStream_ per-fold AUC-ROC (first 3 folds):')
ms = df[df['group'] == 'MemStream_']
for fold in sorted(ms['fold'].unique())[:3]:
    for diff in ['easy', 'medium', 'hard']:
        sub = ms[(ms['fold'] == fold) & (ms['difficulty'] == diff)]
        print(f'  Fold {fold} {diff}:')
        for _, r in sub.iterrows():
            print(f'    {r["label"]}: AUC_ROC={r["AUC_ROC"]:.4f} AUC_PR={r["AUC_PR"]:.4f}')
