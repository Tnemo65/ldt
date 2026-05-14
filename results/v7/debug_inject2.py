"""Deep debug of inject_anomalies step by step."""
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
print(f'inj_idx range: {inj_idx.min()} - {inj_idx.max()}')
print(f'inj_idx sample: {inj_idx[:10]}')

df_a = test_df.iloc[inj_idx].copy().reset_index(drop=True)
print(f'\ndf_a shape: {df_a.shape}')
print(f'df_a columns: {df_a.columns.tolist()}')
print(f'df_a dtypes:\n{df_a.dtypes}')
print(f'\ndf_a fare before injection:')
print(f'  mean: {df_a["fare_amount"].mean():.2f}')
print(f'  min: {df_a["fare_amount"].min():.2f}')
print(f'  max: {df_a["fare_amount"].max():.2f}')

# Apply injection
new_fare = rng.uniform(150, 500, size=n_anom)
print(f'\nNew fare values (first 10): {new_fare[:10]}')
print(f'New fare mean: {new_fare.mean():.2f}')

df_a['fare_amount'] = new_fare
print(f'\ndf_a fare after setting column:')
print(f'  mean: {df_a["fare_amount"].mean():.2f}')
print(f'  min: {df_a["fare_amount"].min():.2f}')
print(f'  max: {df_a["fare_amount"].max():.2f}')
print(f'  dtype: {df_a["fare_amount"].dtype}')

# Check individual values
for i in range(5):
    print(f'  row {i}: fare={df_a.iloc[i]["fare_amount"]:.2f}')

# Now combine
df_combined = pd.concat([test_df, df_a], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=seed).reset_index(drop=True)
labels = np.concatenate([np.zeros(len(test_df), dtype=np.int8),
                         np.ones(n_anom, dtype=np.int8)])

# Check labels
print(f'\nTotal: {len(df_combined)}, Labels: {len(labels)}, Labelsum: {labels.sum()}')

# Find anomalies in combined
anom_mask = labels == 1
anom_indices = np.where(anom_mask)[0]
print(f'Anomaly indices in combined: {anom_indices[:10]}')

# Check fare at anomaly positions
for idx in anom_indices[:5]:
    fare = df_combined.iloc[idx]['fare_amount']
    print(f'  idx={idx}: fare={fare:.2f}, dtype={type(fare)}')

# The REAL issue: is the injection data actually in df_combined?
# Let's check: are the new fare values in df_combined?
print(f'\nUnique fare values from injection: {len(set(new_fare))} unique')
print(f'fare values in combined near 150-500:')
high_fare = df_combined[df_combined['fare_amount'] > 100]['fare_amount']
print(f'  Count: {len(high_fare)}')
if len(high_fare) > 0:
    print(f'  Mean: {high_fare.mean():.2f}')
    print(f'  Values: {high_fare.values[:10]}')

print('\n=== THE REAL TEST ===')
# Get features and check anomaly fare
X_combined = bm.features(df_combined).astype(np.float32)
print(f'Combined features shape: {X_combined.shape}')
print(f'Feature 2 (fare) for anomalies: mean={X_combined[anom_mask, 2].mean():.2f}')
print(f'Feature 2 (fare) for normal: mean={X_combined[~anom_mask, 2].mean():.2f}')
