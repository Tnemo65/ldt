#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('c:/proj/ldt/results/tuning_v1/phase1_results.csv')
print(f'Rows: {len(df)}')

for fold in sorted(df['fold'].unique()):
    for diff in ['easy', 'medium', 'hard']:
        sub = df[(df['fold'] == fold) & (df['difficulty'] == diff)]
        if sub.empty:
            continue
        print(f'\nFold {fold} {diff}:')
        top = sub.nlargest(3, 'AUC_PR')
        for _, r in top.iterrows():
            print(f'  [{r["group"]}] {r["label"]}: AUC-PR={r["AUC_PR"]:.4f} AUC-ROC={r["AUC_ROC"]:.4f}')
