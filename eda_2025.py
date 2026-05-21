import pandas as pd
import numpy as np
import warnings
import json
warnings.filterwarnings('ignore')

files = [
    "c:/proj/ldt/data/raw/raw_202501.parquet",
    "c:/proj/ldt/data/raw/raw_202502.parquet",
    "c:/proj/ldt/data/raw/raw_202503.parquet",
    "c:/proj/ldt/data/raw/raw_202504.parquet",
    "c:/proj/ldt/data/raw/raw_202505.parquet",
    "c:/proj/ldt/data/raw/raw_202506.parquet",
    "c:/proj/ldt/data/raw/raw_202507.parquet",
    "c:/proj/ldt/data/raw/raw_202508.parquet",
    "c:/proj/ldt/data/raw/raw_202509.parquet",
    "c:/proj/ldt/data/raw/raw_202510.parquet",
    "c:/proj/ldt/data/raw/raw_202511.parquet",
    "c:/proj/ldt/data/raw/raw_202512.parquet",
]

results = {}

# =============================================================================
# SECTION 1: DATASET SIZE & VOLUME
# =============================================================================
print("=" * 80)
print("SECTION 1: DATASET SIZE & VOLUME")
print("=" * 80)
total = 0
monthly_counts = {}
for f in files:
    month = f.split("_")[-1].replace(".parquet", "")
    df = pd.read_parquet(f)
    cnt = len(df)
    total += cnt
    monthly_counts[month] = cnt
    print(f"  {month}: {cnt:,} rows")

print(f"\n  TOTAL (2025): {total:,} rows")
print(f"  Monthly average: {total/12:,.0f} rows")

results['section1_total'] = total
results['section1_monthly'] = monthly_counts

# =============================================================================
# SECTION 2: RAW DATA SCHEMA & COLUMNS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 2: RAW DATA SCHEMA & COLUMNS")
print("=" * 80)
df_sample = pd.read_parquet(files[0])
print(f"  Total columns: {len(df_sample.columns)}")
print(f"  Columns: {list(df_sample.columns)}")
print(f"  Dtypes:\n{df_sample.dtypes.to_string()}")
results['section2_columns'] = list(df_sample.columns)
results['section2_ncols'] = len(df_sample.columns)

# =============================================================================
# SECTION 3: COMPLETENESS VIOLATIONS (Table 4.1)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 3: COMPLETENESS VIOLATIONS (Table 4.1)")
print("=" * 80)
print(f"  Total records across 2025: {total:,}")
cols_null_check = [
    'passenger_count', 'RatecodeID', 'store_and_fwd_flag',
    'congestion_surcharge', 'Airport_fee',  # 'speed_mph' is derived — skip here
]
print(f"\n  {'Column':<25} {'NULL Count':>12} {'NULL Rate':>10} {'Thesis':>10}")
print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*10}")

completeness_results = {}
for col in cols_null_check:
    cnt = 0
    for f in files:
        df = pd.read_parquet(f, columns=[col])
        cnt += df[col].isnull().sum()
    rate = cnt / total * 100
    completeness_results[col] = {'count': int(cnt), 'rate': round(rate, 4)}
    print(f"  {col:<25} {cnt:>12,} {rate:>9.4f}%")

results['section3_completeness'] = completeness_results

# PULocationID, DOLocationID range
print(f"\n  --- Range checks ---")
for col, lo, hi in [('PULocationID', 1, 263), ('DOLocationID', 1, 263), ('passenger_count', 1, 9)]:
    cnt = 0
    for f in files:
        df = pd.read_parquet(f, columns=[col])
        cnt += ((df[col] < lo) | (df[col] > hi)).sum()
    rate = cnt / total * 100
    print(f"  {col} out of [{lo},{hi}]: {cnt:,} ({rate:.4f}%)")
    completeness_results[f"{col}_range"] = {'count': int(cnt), 'rate': round(rate, 4)}

# =============================================================================
# SECTION 4: MONTHLY NULL RATE TREND (passenger_count)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 4: MONTHLY NULL RATE TREND (passenger_count)")
print("=" * 80)
print(f"\n  {'Month':<12} {'Rows':>15} {'NULL Count':>12} {'NULL Rate':>10}")
print(f"  {'-'*12} {'-'*15} {'-'*12} {'-'*10}")
monthly_null = {}
for f in files:
    month = f.split("_")[-1].replace(".parquet", "")
    df = pd.read_parquet(f, columns=['passenger_count'])
    rows = len(df)
    null_cnt = df['passenger_count'].isnull().sum()
    null_rate = null_cnt / rows * 100
    monthly_null[month] = {'rows': int(rows), 'null_count': int(null_cnt), 'null_rate': round(null_rate, 4)}
    print(f"  2025-{month[-2:]:<7} {rows:>15,} {null_cnt:>12,} {null_rate:>9.4f}%")

results['section4_monthly_null'] = monthly_null

# =============================================================================
# SECTION 5: VALIDITY VIOLATIONS — NEGATIVE FINANCIAL VALUES
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 5: VALIDITY VIOLATIONS — NEGATIVE FINANCIAL VALUES (Table 4.2)")
print("=" * 80)
fin_cols = ['fare_amount', 'total_amount', 'extra', 'mta_tax']
print(f"\n  {'Column':<20} {'Neg Count':>12} {'Rate':>8} {'Min':>12} {'Max':>12} {'Mean':>12} {'Median':>10}")
print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

fin_results = {}
for col in fin_cols:
    vals = []
    for f in files:
        df = pd.read_parquet(f, columns=[col])
        vals.append(df[col])
    combined = pd.concat(vals, ignore_index=True)
    neg = combined[combined < 0]
    cnt = len(neg)
    rate = cnt / len(combined) * 100
    mn = float(neg.min()) if len(neg) > 0 else None
    mx = float(neg.max()) if len(neg) > 0 else None
    mean_v = float(neg.mean()) if len(neg) > 0 else None
    median_v = float(neg.median()) if len(neg) > 0 else None
    fin_results[col] = {'count': int(cnt), 'rate': round(rate, 4), 'min': round(mn, 2) if mn else None,
                        'max': round(mx, 2) if mx else None, 'mean': round(mean_v, 2) if mean_v else None,
                        'median': round(median_v, 2) if median_v else None}
    print(f"  {col:<20} {cnt:>12,} {rate:>7.4f}% {str(round(mn,2))[:12]:>12} {str(round(mx,2))[:12]:>12} {str(round(mean_v,2))[:12]:>12} {str(round(median_v,2))[:10]:>10}")

results['section5_financial'] = fin_results

# Overlap fare < 0 AND total < 0
overlap = 0
fare_neg = 0
total_neg = 0
for f in files:
    df = pd.read_parquet(f, columns=['fare_amount', 'total_amount'])
    fare_neg += (df['fare_amount'] < 0).sum()
    total_neg += (df['total_amount'] < 0).sum()
    overlap += ((df['fare_amount'] < 0) & (df['total_amount'] < 0)).sum()

overlap_rate = overlap / total * 100
print(f"\n  fare_amount < 0 total: {fare_neg:,}")
print(f"  total_amount < 0 total: {total_neg:,}")
print(f"  Overlap (fare < 0 AND total < 0): {overlap:,} ({overlap_rate:.4f}%)")
fin_results['overlap'] = {'count': int(overlap), 'rate': round(overlap_rate, 4)}

# =============================================================================
# SECTION 6: VALIDITY VIOLATIONS — PHYSICAL PLAUSIBILITY
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 6: VALIDITY VIOLATIONS — PHYSICAL PLAUSIBILITY")
print("=" * 80)
print(f"\n  Total records: {total:,}")

physical_results = {}

# Speed — compute from trip_distance / duration_s
print(f"\n  Computing speed_mph from trip_distance / duration_s...")
speed_gt_80 = speed_gt_100 = speed_gt_200 = 0
speed_max = 0
for f in files:
    df = pd.read_parquet(f, columns=['trip_distance', 'tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    df['duration_s'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()
    df['speed_mph'] = df['trip_distance'] / (df['duration_s'] / 3600)
    df['speed_mph'] = df['speed_mph'].replace([np.inf, -np.inf], np.nan)
    speed_gt_80 += df['speed_mph'].gt(80).sum()
    speed_gt_100 += df['speed_mph'].gt(100).sum()
    speed_gt_200 += df['speed_mph'].gt(200).sum()
    curr_max = df['speed_mph'].max()
    if not pd.isna(curr_max) and curr_max > speed_max:
        speed_max = curr_max

physical_results = {}
for thresh, val in [(80, speed_gt_80), (100, speed_gt_100), (200, speed_gt_200)]:
    rate = val / total * 100
    key = f'speed_gt_{thresh}'
    physical_results[key] = {'count': int(val), 'rate': round(rate, 4)}
    print(f"  Speed > {thresh} mph: {val:,} ({rate:.4f}%)")

physical_results['speed_max'] = float(speed_max)
print(f"  Speed max: {speed_max:,.2f} mph")

# Duration from datetime
print(f"\n  Computing duration_s from pickup/dropoff datetimes...")
dur_zero = dur_long = 0
for f in files:
    df = pd.read_parquet(f, columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance'])
    dur = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()
    dur_zero += (dur == 0).sum()
    dur_long += (dur > 4 * 3600).sum()

dur_zero_rate = dur_zero / total * 100
dur_long_rate = dur_long / total * 100
physical_results['duration_zero'] = {'count': int(dur_zero), 'rate': round(dur_zero_rate, 4)}
physical_results['duration_gt_4h'] = {'count': int(dur_long), 'rate': round(dur_long_rate, 4)}
print(f"  Duration == 0: {dur_zero:,} ({dur_zero_rate:.4f}%)")
print(f"  Duration > 4 hours: {dur_long:,} ({dur_long_rate:.4f}%)")

# fare_amount max, trip_distance max
for col in ['fare_amount', 'trip_distance']:
    mx = -np.inf
    for f in files:
        df = pd.read_parquet(f, columns=[col])
        curr = df[col].max()
        if not pd.isna(curr) and curr > mx:
            mx = curr
    print(f"  {col} max: {mx:,.2f}")
    physical_results[f'{col}_max'] = float(mx)

# fare_per_mile
fare_max = trip_max = 0
for f in files:
    df = pd.read_parquet(f, columns=['fare_amount', 'trip_distance'])
    df_clean = df[(df['fare_amount'] > 0) & (df['trip_distance'] > 0)].copy()
    df_clean['fpm'] = df_clean['fare_amount'] / df_clean['trip_distance']
    curr = df_clean['fpm'].max()
    if not pd.isna(curr) and curr > fare_max:
        fare_max = curr
print(f"  fare_per_mile max: {fare_max:,.2f}")
physical_results['fare_per_mile_max'] = float(fare_max)

results['section6_physical'] = physical_results

# =============================================================================
# SECTION 7: ZERO-DISTANCE & TEMPORAL
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 7: VALIDITY VIOLATIONS — ZERO-DISTANCE & TEMPORAL")
print("=" * 80)

zero_results = {}

# Zero distance
zero_dist_cnt = 0
zero_dist_pos_fare = 0
zero_fare_means = []
for f in files:
    df = pd.read_parquet(f, columns=['trip_distance', 'fare_amount'])
    zero_dist_cnt += (df['trip_distance'] == 0).sum()
    zd = df[df['trip_distance'] == 0]
    zero_dist_pos_fare += (zd['fare_amount'] > 0).sum()
    if len(zd) > 0:
        zero_fare_means.append(zd['fare_amount'].mean())

zero_dist_rate = zero_dist_cnt / total * 100
zero_fare_mean = np.mean(zero_fare_means) if zero_fare_means else 0
print(f"  Zero-distance trips: {zero_dist_cnt:,} ({zero_dist_rate:.4f}%)")
print(f"  Zero-distance + positive fare: {zero_dist_pos_fare:,}")
print(f"  Mean fare for zero-distance: {zero_fare_mean:.2f}")

zero_results['zero_dist_count'] = int(zero_dist_cnt)
zero_results['zero_dist_rate'] = round(zero_dist_rate, 4)
zero_results['zero_dist_pos_fare'] = int(zero_dist_pos_fare)
zero_results['zero_dist_fare_mean'] = round(zero_fare_mean, 2)

# Datetime columns
dt_cols = []
for c in df_sample.columns:
    if any(x in c.lower() for x in ['pickup', 'datetime', 'date', 'dropoff', 'time']):
        dt_cols.append(c)
print(f"\n  Date/time columns: {dt_cols}")

# Check year range
for dt_col in dt_cols:
    try:
        years = []
        for f in files:
            df = pd.read_parquet(f, columns=[dt_col])
            if df[dt_col].dtype == 'object' or 'datetime' not in str(df[dt_col].dtype):
                df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
            years.append(df[dt_col].dt.year)
        all_years = pd.concat(years, ignore_index=True)
        print(f"\n  Column '{dt_col}':")
        print(f"    Min year: {all_years.min()}")
        print(f"    Max year: {all_years.max()}")
        print(f"    Year value counts:\n{all_years.value_counts().sort_index()}")
        
        # Count outside 2025
        outside_2025 = ((all_years < 2025) | (all_years > 2025)).sum()
        rate = outside_2025 / len(all_years) * 100
        print(f"    Outside year 2025: {outside_2025:,} ({rate:.4f}%)")
        zero_results[f'{dt_col}_outside_2025'] = {'count': int(outside_2025), 'rate': round(rate, 4)}
    except Exception as e:
        print(f"    Error analyzing {dt_col}: {e}")

results['section7_zero_temporal'] = zero_results

# =============================================================================
# SECTION 8: SCHEMA VALIDATION
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 8: SCHEMA VALIDATION (Layer 1)")
print("=" * 80)
print(f"  Schema rejection: sample from {files[0]}")

df_s = pd.read_parquet(files[0])
n = len(df_s)
print(f"  Sample size: {n:,} rows")

# Schema rejections: NULL in passenger_count OR RatecodeID
schema_rej = df_s['passenger_count'].isnull() | df_s['RatecodeID'].isnull()
schema_rej_count = schema_rej.sum()
schema_rej_rate = schema_rej_count / n * 100
print(f"  NULL passenger_count OR RatecodeID: {schema_rej_count:,} ({schema_rej_rate:.4f}%)")
print(f"  Thesis claim (for 2024): ~10.1%")

# All columns NULL check
all_null = df_s['passenger_count'].isnull().sum()
all_null_rate = all_null / n * 100
print(f"  NULL passenger_count only: {all_null:,} ({all_null_rate:.4f}%)")

results['section8_schema'] = {
    'sample_size': int(n),
    'schema_rej_null_passenger_or_ratecode': int(schema_rej_count),
    'schema_rej_rate': round(schema_rej_rate, 4)
}

# =============================================================================
# SAVE RESULTS
# =============================================================================
with open('c:/proj/ldt/data_eda_results.json', 'w') as fp:
    json.dump(results, fp, indent=2)
print("\n\n  Results saved to data_eda_results.json")

# =============================================================================
# PRINT SUMMARY TABLE
# =============================================================================
print("\n\n" + "=" * 80)
print("SUMMARY: ACTUAL vs THESIS VALUES")
print("=" * 80)
print(f"\n  {'Check':<50} {'Actual':>15} {'Thesis':>15} {'Match':>8}")
print(f"  {'-'*50} {'-'*15} {'-'*15} {'-'*8}")

checks = [
    ("Total records 2025", total, "~30M (simulated)", None),
    ("Total columns", len(df_sample.columns), "24", None),
    ("NULL passenger_count rate", completeness_results['passenger_count']['rate'], 9.94, "approx"),
    ("NULL RatecodeID rate", completeness_results['RatecodeID']['rate'], 9.94, "approx"),
    ("NULL store_and_fwd_flag rate", completeness_results['store_and_fwd_flag']['rate'], 9.94, "approx"),
    ("NULL congestion_surcharge rate", completeness_results['congestion_surcharge']['rate'], 9.94, "approx"),
    ("NULL Airport_fee rate", completeness_results['Airport_fee']['rate'], 9.94, "approx"),
    ("NULL speed_mph rate", completeness_results['speed_mph']['rate'], 1.89, "approx"),
]

for label, actual, thesis, mode in checks:
    if mode == "approx":
        match = "~" if actual is not None else "?"
    else:
        match = "OK" if actual == thesis else "DIFF"
    print(f"  {label:<50} {str(actual):>15} {str(thesis):>15} {match:>8}")

print("\n  DONE.")
