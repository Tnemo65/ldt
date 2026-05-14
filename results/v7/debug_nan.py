"""Verify: are anomalous rows surviving NaN filtering?"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r'C:\proj\ldt\results\v7')
import benchmark_v7 as bm

bm.MONTHS = [1]
df = bm.clean(bm.load_month(2024, 1))

rng = np.random.RandomState(42)
test_df = df.iloc[rng.choice(len(df), bm.TEST_N, replace=False)].reset_index(drop=True)

params = {'type': 'extreme_fare', 'fare_range': (150, 500), 'n': 500}
test_df_inj, y_labels = bm.inject_anomalies(test_df, params, 42)
print(f"After injection: {len(test_df_inj)} rows, {y_labels.sum()} anomalies")

# Check NaN in fare_amount before features()
print(f"\nNaN in fare_amount: {test_df_inj['fare_amount'].isna().sum()}")
print(f"NaN in trip_distance: {test_df_inj['trip_distance'].isna().sum()}")
print(f"NaN in dur_min: {test_df_inj['dur_min'].isna().sum()}")

X = bm.features(test_df_inj)
print(f"\nX shape: {X.shape}")
print(f"Anomaly feature 2 (fare): mean={X[y_labels==1, 2].mean():.2f}, std={X[y_labels==1, 2].std():.2f}")
print(f"Normal feature 2 (fare): mean={X[y_labels==0, 2].mean():.2f}, std={X[y_labels==0, 2].std():.2f}")

# Why is anomaly fare only slightly above normal?
# Check raw values
anom_df = test_df_inj.iloc[y_labels==1]
print(f"\nRaw anomaly fare: min={anom_df['fare_amount'].min():.1f}, max={anom_df['fare_amount'].max():.1f}, mean={anom_df['fare_amount'].mean():.1f}")
