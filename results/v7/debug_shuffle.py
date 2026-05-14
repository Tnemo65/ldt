"""Verify the shuffle bug."""
import sys, warnings
sys.path.insert(0, r'C:\proj\ldt\results\v7')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import benchmark_v7 as bm

DATA_DIR = Path(r'C:\proj\ldt\data\raw')
df = bm.clean(bm.load_month(2024, 1))
test_df = df.iloc[:2000].reset_index(drop=True)

seed = 42
n_anom = 100
params = {'type': 'extreme_fare', 'fare_range': (150, 500), 'n': n_anom}

rng = np.random.RandomState(seed)
inj_idx = rng.choice(len(test_df), n_anom, replace=False)
df_a = test_df.iloc[inj_idx].copy().reset_index(drop=True)
df_a['fare_amount'] = rng.uniform(150, 500, size=n_anom)

# Buggy version (current)
df_combined_buggy = pd.concat([test_df, df_a], ignore_index=True)
df_combined_buggy = df_combined_buggy.sample(frac=1, random_state=seed).reset_index(drop=True)
labels_buggy = np.concatenate([np.zeros(len(test_df)), np.ones(n_anom)])

# Fixed version
df_combined_fixed = pd.concat([test_df, df_a], ignore_index=True)
# Sample index first, apply to both
perm = df_combined_fixed.sample(frac=1, random_state=seed).index
df_combined_fixed = df_combined_fixed.loc[perm].reset_index(drop=True)
labels_fixed = np.concatenate([np.zeros(len(test_df)), np.ones(n_anom)])
labels_fixed = labels_fixed[perm.to_numpy()]

# Check features
X_buggy = bm.features(df_combined_buggy).astype(np.float32)
X_fixed = bm.features(df_combined_fixed).astype(np.float32)

# Buggy: fare difference between labeled anomalies and normal
anom_buggy = labels_buggy == 1
print('=== BUGGY (labels not shuffled) ===')
print(f'  Labeled-anom fare mean: {X_buggy[anom_buggy, 2].mean():.2f}')
print(f'  Labeled-normal fare mean: {X_buggy[~anom_buggy, 2].mean():.2f}')
print(f'  Should be similar (labels mismatched)')

# Fixed: fare difference
anom_fixed = labels_fixed == 1
print('\n=== FIXED (labels shuffled with data) ===')
print(f'  Labeled-anom fare mean: {X_fixed[anom_fixed, 2].mean():.2f}')
print(f'  Labeled-normal fare mean: {X_fixed[~anom_fixed, 2].mean():.2f}')
print(f'  Should be very different (labels matched)')

# The true anomalies: high fare rows
true_high_fare = X_fixed[:, 2] > 100
print(f'\nTrue anomalies (fare > 100): {true_high_fare.sum()} rows')
print(f'Fixed labels matched to true anomalies: {np.sum(anom_fixed & true_high_fare)} / {n_anom}')
print(f'Fixed labels correctly NOT matched: {np.sum(~anom_fixed & ~true_high_fare)} / {len(X_fixed)-n_anom}')
