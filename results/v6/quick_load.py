"""Quick load test."""
import sys
sys.path.insert(0, r'C:\proj\ldt\results\v6')
import benchmark_v6 as bm
import numpy as np
import pandas as pd

# Override for speed
bm.MONTHS = [1, 2, 3]
bm.SEEDS = [42]
bm.DIFFICULTIES = ['easy']
bm.LABEL_BUDGETS = [0]

print("Loading 3 months...")
monthly, monthly_X = [], []
for m in bm.MONTHS:
    df = bm.clean(bm.load_month(2024, m))
    X = bm.features(df)
    monthly.append(df)
    monthly_X.append(X.astype(np.float32))
    print(f'  Month {m:02d}: {len(df):,} records')

print("\nFold 1 setup...")
fold_idx = 1
test_month = bm.MONTHS[1]
train_months = bm.MONTHS[:fold_idx]
val_month = train_months[-1]

train_X = np.vstack([monthly_X[m - 1] for m in train_months])
train_df = pd.concat([monthly[m - 1] for m in train_months], ignore_index=True)
last_X = monthly_X[val_month - 1]
last_df = monthly[val_month - 1]
val_X = last_X[-bm.VAL_N:]
val_df = last_df.iloc[-bm.VAL_N:].reset_index(drop=True)
n_train_keep = len(train_X) - bm.VAL_N
train_X = train_X[:n_train_keep]
train_df = train_df.iloc[:n_train_keep].reset_index(drop=True)

print(f"  Train: {len(train_X):,}, Val: {len(val_X):,}")

# Test sklearn_IF
print("\nTesting sklearn_IF...")
algo = bm.SklearnIF(seed=42)
algo.fit(train_X.astype(np.float32), val_X.astype(np.float32), np.zeros(len(val_X), dtype=np.int8))
scores = algo.decision_function(val_X.astype(np.float32))
print(f"  Scores: min={scores.min():.4f}, max={scores.max():.4f}")
print("OK!")
