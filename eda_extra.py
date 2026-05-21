import pandas as pd
import numpy as np
import warnings
import json
warnings.filterwarnings('ignore')

files = [f"c:/proj/ldt/data/raw/raw_{m}.parquet" for m in [f"2025{m:02d}" for m in range(1,13)]]

total = 48722602  # from previous run

# Compute speed_mph NULL rate across all files
print("Computing speed_mph NULL rate...")
speed_null = 0
for f in files:
    df = pd.read_parquet(f, columns=['trip_distance', 'tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    dur = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()
    speed = df['trip_distance'] / (dur / 3600)
    speed_null += speed.replace([np.inf, -np.inf], np.nan).isna().sum()

speed_null_rate = speed_null / total * 100
print(f"  speed_mph NULL count: {speed_null:,} ({speed_null_rate:.4f}%)")

# Also check extra columns: cbd_congestion_fee, improvement_surcharge
print("\nChecking extra columns...")
for col in ['cbd_congestion_fee', 'improvement_surcharge']:
    cnt = 0
    for f in files:
        df = pd.read_parquet(f, columns=[col])
        cnt += df[col].isnull().sum()
    rate = cnt / total * 100
    print(f"  {col} NULL: {cnt:,} ({rate:.4f}%)")

# Extra: improvement_surcharge negative
print("\nChecking improvement_surcharge < 0...")
cnt = 0
for f in files:
    df = pd.read_parquet(f, columns=['improvement_surcharge'])
    cnt += (df['improvement_surcharge'] < 0).sum()
rate = cnt / total * 100
print(f"  improvement_surcharge < 0: {cnt:,} ({rate:.4f}%)")
