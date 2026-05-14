import pandas as pd
import numpy as np

df = pd.read_csv('results/v8/checkpoint_v8.csv')

print("=" * 70)
print("DETAILED COMPARISON: MemStream vs CA-DIF-EIA-Stream")
print("=" * 70)

for algo in ['MemStream', 'CA-DIF-EIA-Stream']:
    sub = df[df['algorithm'] == algo]
    print(f"\n=== {algo} ===")
    print(f"  Rows: {len(sub)}")
    print(f"  AUC_PR  : mean={sub['AUC_PR'].mean():.6f}  std={sub['AUC_PR'].std():.6f}")
    print(f"  AUC_ROC : mean={sub['AUC_ROC'].mean():.6f}  std={sub['AUC_ROC'].std():.6f}")
    print(f"  F1      : mean={sub['F1'].mean():.6f}  std={sub['F1'].std():.6f}")
    print(f"  Precision: mean={sub['Precision'].mean():.6f}")
    print(f"  Recall   : mean={sub['Recall'].mean():.6f}")
    # By difficulty
    for diff in ['easy', 'medium', 'hard']:
        s = sub[sub['difficulty'] == diff]
        if len(s) > 0:
            print(f"  {diff:8s}: AUC_PR={s['AUC_PR'].mean():.6f}  F1={s['F1'].mean():.6f}  Prec={s['Precision'].mean():.6f}  Rec={s['Recall'].mean():.6f}")
    # By budget
    for lb in sorted(sub['label_budget'].unique()):
        s = sub[sub['label_budget'] == lb]
        if len(s) > 0:
            print(f"  budget={int(lb):4d}: AUC_PR={s['AUC_PR'].mean():.6f}  labels_used={s['labels_consumed'].mean():.0f}")
    print()

# Speed analysis
print("=" * 70)
print("SPEED ANALYSIS")
print("=" * 70)
speed_cols = ['fit_time', 'predict_time', 'total_time', 'throughput']
for algo in ['MemStream', 'CA-DIF-EIA-Stream', 'DenoisingAE', 'IsolationForest']:
    sub = df[df['algorithm'] == algo]
    if len(sub) > 0:
        print(f"\n=== {algo} ===")
        for col in speed_cols:
            if col in sub.columns:
                vals = sub[col].dropna()
                if len(vals) > 0:
                    print(f"  {col:20s}: mean={vals.mean():.6f}  std={vals.std():.6f}  min={vals.min():.6f}  max={vals.max():.6f}")

print()
print("=" * 70)
print("TOP 10 OVERALL (all seeds, budgets averaged)")
print("=" * 70)
available_cols = ['AUC_PR', 'AUC_ROC', 'F1', 'Precision', 'Recall']
grp = df.groupby('algorithm').agg({
    'AUC_PR': 'mean',
    'AUC_ROC': 'mean',
    'F1': 'mean',
    'Precision': 'mean',
    'Recall': 'mean'
}).sort_values('AUC_PR', ascending=False)
print(grp.head(10).to_string())
