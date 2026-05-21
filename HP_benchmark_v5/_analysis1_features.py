import sys, numpy as np, json, os
sys.path.insert(0, r'C:\proj\ldt\HP_benchmark_v5')

print('='*80)
print('ANALYSIS 1: FEATURE DISTRIBUTIONS')
print('='*80)

import pyarrow.parquet as pq
import pandas as pd

# Load clean train data (first 50K rows via pandas)
pf = pq.ParquetFile(r'C:\proj\ldt\HP_benchmark_v5\data\train_clean.parquet')
total = pf.metadata.num_rows
n_read = min(50000, total)
tbl = pf.read_row_group(0).slice(0, n_read)
df = tbl.to_pandas()

print(f'\n=== TRAIN DATA STATS (first 50K of {total:,}) ===')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')

# Raw columns stats
raw_cols = ['trip_distance', 'fare_amount', 'total_amount', 'duration_s', 'speed_mph', 'passenger_count', 'tip_amount']
print(f'\n=== RAW FEATURE STATS ===')
for col in raw_cols:
    vals = df[col].dropna().values.astype(float)
    if len(vals) == 0:
        print(f'{col}: EMPTY')
        continue
    print(f'{col}: min={np.min(vals):.3f}, max={np.max(vals):.3f}, mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, median={np.median(vals):.3f}, p25={np.percentile(vals,25):.3f}, p75={np.percentile(vals,75):.3f}')

print(f'\n=== DATE RANGE ===')
df['date'] = pd.to_datetime(df['date'])
print(f'Date range: {df["date"].min()} to {df["date"].max()}')
month_counts = df["date"].dt.month.value_counts().sort_index()
for m, cnt in month_counts.items():
    print(f'  Month {m}: {cnt:,}')

print(f'\n=== RATE CODE DISTRIBUTION ===')
print(df['RatecodeID'].value_counts().sort_index())

print(f'\n=== ANOMALY RATE IN VALID ===')
gt_mask = np.load(r'C:\proj\ldt\HP_benchmark_v5\data\valid\ground_truth_mask.npy')
print(f'Total: {len(gt_mask):,}, Anomalies: {gt_mask.sum():,} ({gt_mask.mean()*100:.2f}%)')

# Per-type analysis
per_type = json.load(open(r'C:\proj\ldt\HP_benchmark_v5\data\valid\ground_truth_per_type.json'))
print(f'\n=== ANOMALY TYPES ===')
for t, info in sorted(per_type.items(), key=lambda x: int(x[0]) if x[0].lstrip('-').isdigit() else 999):
    print(f'Type {t}: count={info["count"]}, desc={info.get("description","")}')
