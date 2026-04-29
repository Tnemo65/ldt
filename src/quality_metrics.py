"""
Step 2: Define quality metrics + anomaly thresholds
==================================================
Purpose: Use baseline from Step 1 to detect anomalies in NEW data (July 2024).

Fixes applied:
  - Fixed threshold correctly uses TRAINING data (Jan), not test data
  - VECTORIZED quality scores (81s -> <1s)
  - Uses shared utils (pipeline_utils.py)
  - NaN counts tracked in baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time

import pipeline_utils as pu

DATA_DIR = Path('d:/final/data/raw')
OUTPUT_DIR = Path('d:/final/output')
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Step 2a: Load baseline ────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2a: Loading baseline from Step 1")
print("=" * 60)

with open(OUTPUT_DIR / 'baseline_stats.json') as f:
    baseline = json.load(f)

print(f"Loaded baseline: {len(baseline)} context groups")
print(f"Columns: fare_amount, trip_distance, total_amount, passenger_count")

# ── Step 2b: Load test data ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2b: Loading test data (July 2024)")
print("=" * 60)

df = pd.read_parquet(DATA_DIR / 'yellow_tripdata_2024-07.parquet')
print(f"Loaded: {len(df):,} rows, {df.shape[1]} cols")

# ── Step 2c: Vectorized context assignment ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2c: VECTORIZED context assignment")
print("=" * 60)

t0 = time.time()
hours = df['tpep_pickup_datetime'].dt.hour
ratecodes = df['RatecodeID']
time_bin = pd.cut(hours, bins=[-1, 5, 11, 17, 24],
                  labels=['night', 'morning', 'afternoon', 'evening']).astype(str)
ratecode_bin = np.where(ratecodes == 1, 'standard', 'special')
df['context'] = time_bin + '_' + ratecode_bin
print(f"Context assigned in {time.time() - t0:.2f}s (vectorized)")

# Verify coverage
test_contexts = set(df['context'].unique())
baseline_contexts = set(baseline.keys())
missing = test_contexts - baseline_contexts
if missing:
    print(f"  WARNING: Test contexts not in baseline: {missing}")
else:
    print(f"  All {len(test_contexts)} contexts covered in baseline: OK")

# ── Step 2d: Load training data for fixed threshold ─────────────────────────────
print("\n" + "=" * 60)
print("STEP 2d: Computing fixed threshold from TRAINING data (Jan)")
print("=" * 60)

df_train = pd.read_parquet(DATA_DIR / 'yellow_tripdata_2024-01.parquet')
# NOTE: we use df_train just for computing fixed thresholds
# The actual rule checks use per-context thresholds from baseline
fixed_fare_p5 = df_train['fare_amount'].quantile(0.05)
fixed_fare_p95 = df_train['fare_amount'].quantile(0.95)
del df_train  # free memory

print(f"Fixed thresholds (from TRAINING Jan): p5=${fixed_fare_p5:.2f}, p95=${fixed_fare_p95:.2f}")

# ── Step 2e: Apply quality checks (fully vectorized) ────────────────────────────
print("\n" + "=" * 60)
print("STEP 2e: Applying quality checks (fully vectorized)")
print("=" * 60)

t0 = time.time()
n_total = len(df)

# Pre-allocate all violation columns as boolean
df['negative_fare'] = df['fare_amount'] < 0
df['zero_distance'] = df['trip_distance'] <= 0
df['null_passenger'] = df['passenger_count'].isna()
df['null_ratecode'] = df['RatecodeID'].isna()

# Context-aware outlier flags (vectorized per context) using z-score > 3
df['fare_outlier'] = False
df['distance_outlier'] = False
df['total_outlier'] = False
df['speed_extreme'] = False

ctx_arr = df['context'].values
fare_dev_arr = np.zeros(len(df), dtype=float)
dist_dev_arr = np.zeros(len(df), dtype=float)
total_dev_arr = np.zeros(len(df), dtype=float)

for ctx, b in baseline.items():
    mask = (ctx_arr == ctx)
    if mask.sum() == 0:
        continue

    for col, arr in [('fare_amount', fare_dev_arr), ('trip_distance', dist_dev_arr),
                      ('total_amount', total_dev_arr)]:
        if col not in b:
            continue
        s = b[col]
        if pd.isna(s.get('std')) or s['std'] < 0.01:
            continue
        vals = df.loc[mask, col].fillna(s['mean']).values
        z = np.abs(vals - s['mean']) / s['std']
        arr[mask] = z

    durations = (df.loc[mask, 'tpep_dropoff_datetime'] - df.loc[mask, 'tpep_pickup_datetime']).dt.total_seconds()
    speeds = np.where(durations.values > 0,
                      df.loc[mask, 'trip_distance'].values / (durations.values / 3600), 0)
    df.loc[mask, 'speed_extreme'] = (speeds > 100) | (durations.values <= 0)

df['fare_outlier'] = (fare_dev_arr > 3)
df['distance_outlier'] = (dist_dev_arr > 3)
df['total_outlier'] = (total_dev_arr > 3)

print(f"Checks applied in {time.time() - t0:.2f}s")

# Anomaly = any violation
df['is_anomaly'] = (
    df['fare_outlier'] | df['distance_outlier'] | df['total_outlier'] |
    df['negative_fare'] | df['zero_distance'] | df['null_passenger'] |
    df['null_ratecode'] | df['speed_extreme']
)

# Fixed threshold outlier (for comparison)
df['fixed_outlier'] = (
    (df['fare_amount'] < fixed_fare_p5) | (df['fare_amount'] > fixed_fare_p95)
)

# ── Step 2f: Quality metrics summary ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2f: Quality metrics summary")
print("=" * 60)

n_anomaly = df['is_anomaly'].sum()
violation_cols = ['fare_outlier', 'distance_outlier', 'total_outlier',
                   'negative_fare', 'zero_distance', 'null_passenger',
                   'null_ratecode', 'speed_extreme']

print(f"\nTotal records: {n_total:,}")
print(f"Total anomalies: {n_anomaly:,} ({n_anomaly/n_total*100:.2f}%)")
print(f"\nViolation breakdown:")

for col in violation_cols:
    n = int(df[col].sum())
    pct = n / n_total * 100
    note = ""
    if col in ['fare_outlier', 'distance_outlier', 'total_outlier']:
        note = " (z-score > 3, context-aware)"
    elif col in ['null_passenger', 'null_ratecode']:
        note = " (REAL DQ issue)"
    elif col in ['negative_fare', 'zero_distance']:
        note = " (REAL DQ issue)"
    print(f"  {col:<20}: {n:>8,} ({pct:5.2f}%){note}")

# ── Step 2g: Fixed vs Context-aware comparison ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2g: Context-aware vs Fixed threshold comparison")
print("=" * 60)

ctx_only = ((df['fare_outlier']) & (~df['fixed_outlier'])).sum()
fixed_only = ((~df['fare_outlier']) & (df['fixed_outlier'])).sum()
both = ((df['fare_outlier']) & (df['fixed_outlier'])).sum()
neither = ((~df['fare_outlier']) & (~df['fixed_outlier'])).sum()

print(f"\n  Fixed thresholds from TRAINING (Jan): p5=${fixed_fare_p5:.2f}, p95=${fixed_fare_p95:.2f}")
print(f"\n  Agreement analysis:")
print(f"    Both flagged:           {both:>10,} ({both/n_total*100:.2f}%)")
print(f"    Context-aware ONLY:      {ctx_only:>10,} ({ctx_only/n_total*100:.2f}%)  <- Context model adds value")
print(f"    Fixed threshold ONLY:    {fixed_only:>10,} ({fixed_only/n_total*100:.2f}%)  <- Fixed under/over-flags")
print(f"    Neither flagged:       {neither:>10,} ({neither/n_total*100:.2f}%)")

# ── Step 2h: Anomaly by context group ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2h: Anomaly rate by context group")
print("=" * 60)

ctx_summary = df.groupby('context').agg(
    total=('fare_amount', 'count'),
    anomalies=('is_anomaly', 'sum'),
    fare_outliers=('fare_outlier', 'sum'),
    distance_outliers=('distance_outlier', 'sum'),
    neg_fare=('negative_fare', 'sum'),
    null=('null_passenger', 'sum'),
).reset_index()

ctx_summary['anomaly_rate'] = ctx_summary['anomalies'] / ctx_summary['total'] * 100
ctx_summary['fare_outlier_rate'] = ctx_summary['fare_outliers'] / ctx_summary['total'] * 100
ctx_summary['distance_outlier_rate'] = ctx_summary['distance_outliers'] / ctx_summary['total'] * 100

print(f"\n{'Context':<22} {'Total':>10} {'Anomaly%':>10} {'FareOut%':>10} {'DistOut%':>10}")
print("-" * 65)
for _, row in ctx_summary.sort_values('anomaly_rate', ascending=False).iterrows():
    print(f"{row['context']:<22} {row['total']:>10,} {row['anomaly_rate']:>9.2f}% "
          f"{row['fare_outlier_rate']:>9.2f}% {row['distance_outlier_rate']:>9.2f}%")

# ── Step 2i: VECTORIZED quality scores (<1s instead of 81s) ──────────────────
print("\n" + "=" * 60)
print("STEP 2i: Computing quality scores (VECTORIZED)")
print("=" * 60)

t0 = time.time()
df['quality_score'] = pu.compute_quality_scores(df, violation_cols)
print(f"Quality scores computed in {time.time() - t0:.2f}s (vectorized, no apply)")

print(f"\nQuality score distribution:")
print(df['quality_score'].describe())

# By context
qs_by_ctx = df.groupby('context')['quality_score'].mean().sort_values()
print(f"\nMean quality score by context:")
for ctx, score in qs_by_ctx.items():
    bar = '#' * int(score / 5)
    print(f"  {ctx:<22}: {score:6.2f}  {bar}")

# ── Step 2j: Save results ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2j: Saving results")
print("=" * 60)

summary = {
    'test_dataset': 'yellow_tripdata_2024-07.parquet',
    'training_dataset': 'yellow_tripdata_2024-01.parquet',
    'n_total': int(n_total),
    'n_anomaly': int(n_anomaly),
    'anomaly_rate': float(n_anomaly / n_total),
    'context_groups': len(baseline),
    'violation_counts': {col: int(df[col].sum()) for col in violation_cols},
    'quality_score_mean': float(df['quality_score'].mean()),
    'quality_score_median': float(df['quality_score'].median()),
    'fixed_threshold_p5': float(fixed_fare_p5),
    'fixed_threshold_p95': float(fixed_fare_p95),
}

with open(OUTPUT_DIR / 'step2_quality_metrics.json', 'w') as f:
    json.dump(summary, f, indent=2)

anomaly_df = df[df['is_anomaly']].head(1000)
anomaly_df.to_parquet(OUTPUT_DIR / 'step2_anomaly_samples.parquet')

print(f"Saved: {OUTPUT_DIR / 'step2_quality_metrics.json'}")
print(f"Saved: {OUTPUT_DIR / 'step2_anomaly_samples.parquet'} ({len(anomaly_df)} samples)")

# ── Step 2k: Key insights ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2k: Key insights")
print("=" * 60)

print(f"""
KEY FINDING: Context-aware thresholds vs Fixed thresholds
  - Context-aware fare outliers: {df['fare_outlier'].sum():,} ({df['fare_outlier'].mean()*100:.2f}%)
  - Fixed threshold outliers:     {df['fixed_outlier'].sum():,} ({df['fixed_outlier'].mean()*100:.2f}%)

  Context-aware finds {ctx_only:,} records that fixed threshold MISSES (different behavior)
  Fixed threshold finds {fixed_only:,} records that context-aware MISSES (under-flagging)
""")

print("DONE: Step 2 complete")
