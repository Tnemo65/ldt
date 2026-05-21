import sys, numpy as np, json
sys.path.insert(0, r'C:\proj\ldt\HP_benchmark_v5')

print('='*80)
print('ANALYSIS 5: DEEP GROUND TRUTH ANALYSIS')
print('='*80)

import pyarrow.parquet as pq

# Load injection log
inj_log = json.load(open(r'C:\proj\ldt\HP_benchmark_v5\data\valid\injection_log.json'))
print(f'\n=== INJECTION LOG ===')
print(f'Keys: {list(inj_log.keys())[:10]}...')
for k in list(inj_log.keys())[:3]:
    v = inj_log[k]
    if isinstance(v, dict):
        print(f'{k}: {list(v.keys())}')
    elif isinstance(v, list):
        print(f'{k}: list of {len(v)} items, first={v[0] if v else "empty"}')
    else:
        print(f'{k}: {v}')

# Load per-type ground truth
per_type = json.load(open(r'C:\proj\ldt\HP_benchmark_v5\data\valid\ground_truth_per_type.json'))

print(f'\n=== PER-TYPE DETAILS ===')
for t, info in sorted(per_type.items(), key=lambda x: int(x[0]) if x[0].lstrip('-').isdigit() else 999):
    print(f'\nType {t}: {info.get("name","")}')
    print(f'  desc: {info.get("desc","")}')
    print(f'  target_n: {info.get("target_n","")}')
    print(f'  confirmed_n: {info.get("confirmed_n","")}')
    print(f'  ratio_pct: {info.get("ratio_pct","")}')
    print(f'  key_feature: {info.get("key_feature","")}')
    print(f'  key_signal: {info.get("key_signal","")}')
    print(f'  indices count: {len(info.get("indices",[]))}')

# Load raw valid polluted data and check actual feature values for anomaly types
print(f'\n=== FEATURE VALUES PER ANOMALY TYPE ===')

# Load raw valid polluted data
pf = pq.ParquetFile(r'C:\proj\ldt\HP_benchmark_v5\data\valid_polluted.parquet')
tbl = pf.read_row_group(0)
raw_df = tbl.to_pandas()

print(f'Raw DF shape: {raw_df.shape}')
print(f'Columns: {list(raw_df.columns)}')

# Get ground truth
gt_mask = np.load(r'C:\proj\ldt\HP_benchmark_v5\data\valid\ground_truth_mask.npy')
gt_mask = gt_mask[-len(raw_df):]

# Analyze anomalies vs normal
anom_idx = np.where(gt_mask == 1)[0]
norm_idx = np.where(gt_mask == 0)[0]

print(f'\nAnomalies: {len(anom_idx)}, Normal: {len(norm_idx)}')

# Key features
for col in ['fare_amount', 'trip_distance', 'duration_s', 'total_amount', 'tip_amount', 'speed_mph']:
    if col not in raw_df.columns:
        continue
    a_vals = raw_df[col].iloc[anom_idx].values.astype(float)
    n_vals = raw_df[col].iloc[norm_idx].values.astype(float)
    a_nan = np.isnan(a_vals).sum()
    n_nan = np.isnan(n_vals).sum()
    a_vals = a_vals[~np.isnan(a_vals)]
    n_vals = n_vals[~np.isnan(n_vals)]
    if len(a_vals) == 0 or len(n_vals) == 0:
        continue
    print(f'{col}: ANOM mean={np.mean(a_vals):.2f} median={np.median(a_vals):.2f}, NORMAL mean={np.mean(n_vals):.2f} median={np.median(n_vals):.2f}, ratio_mean={np.mean(a_vals)/max(np.mean(n_vals),0.01):.2f}x')

# Check ratecode distribution for anomalies
print(f'\n=== RATE CODE DISTRIBUTION ===')
print(f'Anomalies:')
print(raw_df['RatecodeID'].iloc[anom_idx].value_counts().head(10))
print(f'\nNormal:')
print(raw_df['RatecodeID'].iloc[norm_idx].value_counts().head(10))

# Check duration clipping effect
print(f'\n=== DURATION ANALYSIS ===')
dur_anom = raw_df['duration_s'].iloc[anom_idx].values
dur_norm = raw_df['duration_s'].iloc[norm_idx].values
print(f'Anom duration: min={np.nanmin(dur_anom):.0f}, max={np.nanmax(dur_anom):.0f}, mean={np.nanmean(dur_anom):.0f}')
print(f'Norm duration: min={np.nanmin(dur_norm):.0f}, max={np.nanmax(dur_norm):.0f}, mean={np.nanmean(dur_norm):.0f}')
print(f'Note: feature extraction clips duration to [1, 360] seconds')

# Check actual injection rules from inject_anomalies_memstream.py
print(f'\n=== CHECKING INJECTION LOG ENTRIES ===')
# Check first few injection entries
if 'injections' in inj_log:
    injections = inj_log['injections']
    print(f'Total injections: {len(injections)}')
    for inj in injections[:3]:
        print(f'  {inj}')
elif 'entries' in inj_log:
    entries = inj_log['entries']
    print(f'Total entries: {len(entries)}')
    for e in entries[:3]:
        print(f'  {e}')
else:
    print(f'Keys: {list(inj_log.keys())}')
