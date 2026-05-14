"""Debug: why are anomalies indistinguishable from normal?"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, r'C:\proj\ldt\results\v7')
import benchmark_v7 as bm
from sklearn.preprocessing import StandardScaler

bm.MONTHS = [1]
df = bm.clean(bm.load_month(2024, 1))

# Normal stats
X = bm.features(df)
print("=== NORMAL DATA ===")
print(f"Records: {len(df)}")
print(f"Shape: {X.shape}")
print(f"fare_amount: {df['fare_amount'].describe().to_string()}")
print(f"trip_distance: {df['trip_distance'].describe().to_string()}")
print(f"dur_min: {df['dur_min'].describe().to_string()}")

# Inject anomalies
rng = np.random.RandomState(42)
inj_idx = rng.choice(len(df), 500, replace=False)
df_a = df.iloc[inj_idx].copy().reset_index(drop=True)

# Easy: meter_mult 10-20x
df_e = df_a.copy()
df_e['fare_amount'] = df_e['trip_distance'] * 2.5 * rng.uniform(10, 20, size=500)
print(f"\n=== EASY ANOMALIES ===")
print(f"fare_amount: min={df_e['fare_amount'].min():.1f}, max={df_e['fare_amount'].max():.1f}, mean={df_e['fare_amount'].mean():.1f}")

# After clean() percentile filtering (1%-99%)
fare_lo, fare_hi = df['fare_amount'].quantile(0.01), df['fare_amount'].quantile(0.99)
print(f"Fare range after 1-99%% filter: {fare_lo:.1f} - {fare_hi:.1f}")
kept = (df_e['fare_amount'] >= fare_lo) & (df_e['fare_amount'] <= fare_hi)
print(f"Anomalies kept after filtering: {kept.sum()}/500 ({kept.mean():.1%})")

# Compare features
X_norm = bm.features(df.iloc[rng.choice(len(df), 1000, replace=False)].reset_index(drop=True))
X_easy = bm.features(df_e)

print(f"\n=== FEATURE COMPARISON (after scaling) ===")
scaler = StandardScaler()
scaler.fit(X_norm)
X_norm_s = scaler.transform(X_norm)
X_easy_s = scaler.transform(X_easy)

# Key features: [2]=fare, [6]=fare/dist, [7]=fare/dur, [16]=fare/dur_norm
for fi, fname in [(2,'fare'), (6,'fare/dist'), (7,'fare/dur'), (16,'fare/dur_norm')]:
    print(f"  {fname:15s}: norm mean={X_norm_s[:,fi].mean():7.3f} ±{X_norm_s[:,fi].std():6.3f}, "
          f"easy mean={X_easy_s[:,fi].mean():7.3f} ±{X_easy_s[:,fi].std():6.3f}, "
          f"diff={X_easy_s[:,fi].mean()-X_norm_s[:,fi].mean():+.3f}")

# Mahalanobis-like check: how many sigma away?
for fi, fname in [(2,'fare'), (6,'fare/dist'), (7,'fare/dur')]:
    norm_mean = X_norm_s[:,fi].mean()
    norm_std = X_norm_s[:,fi].std()
    z_easy = (X_easy_s[:,fi] - norm_mean) / (norm_std + 1e-9)
    pct_above_2sigma = (np.abs(z_easy) > 2).mean()
    pct_above_3sigma = (np.abs(z_easy) > 3).mean()
    print(f"  {fname:15s}: {pct_above_2sigma:.1%} > 2sigma, {pct_above_3sigma:.1%} > 3sigma")
