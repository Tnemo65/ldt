"""
Shared data loader: streams through all monthly parquet files
and computes the aggregates needed by multiple visualization scripts.
Caches results to disk so each figure script runs fast.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time

RAW_DIR = Path('d:/final/data/raw')
CACHE_DIR = Path('d:/final/visualization/cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ALL_MONTHS = (
    [f'2024-{m:02d}' for m in range(1, 13)] +
    [f'2025-{m:02d}' for m in range(1, 13)] +
    [f'2026-{m:02d}' for m in range(1, 3)]
)


def load_month(month):
    """Load a single month parquet, return DataFrame with month column."""
    path = RAW_DIR / f'yellow_tripdata_{month}.parquet'
    df = pd.read_parquet(path)
    df['month'] = month
    return df


def compute_monthly_stats(force_reload=False):
    """Return DataFrame with one row per month: total, violations, rates, avg metrics."""
    cache = CACHE_DIR / 'monthly_stats.json'
    if cache.exists() and not force_reload:
        return pd.read_json(cache, precise_float=True)

    records = []
    for m in ALL_MONTHS:
        path = RAW_DIR / f'yellow_tripdata_{m}.parquet'
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

        n = len(df)
        dur_min = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds().values / 60
        speed = np.where(dur_min > 0, df['trip_distance'].values / (dur_min / 60), 0)

        v_neg_fare  = (df['fare_amount'] < 0).sum()
        v_zero_dist = (df['trip_distance'] <= 0).sum()
        v_null_pass = df['passenger_count'].isna().sum()
        v_null_rc   = df['RatecodeID'].isna().sum()
        any_viol    = ((df['fare_amount'] < 0) | (df['trip_distance'] <= 0) |
                       df['passenger_count'].isna() | df['RatecodeID'].isna()).sum()

        records.append({
            'year_month': m,
            'total': n,
            'violations': any_viol,
            'violation_rate': any_viol / n * 100,
            'null_passenger_rate': v_null_pass / n,
            'null_ratecode_rate':   v_null_rc   / n,
            'neg_fare_rate':  v_neg_fare / n,
            'zero_dist_rate': v_zero_dist / n,
            'avg_fare':     df['fare_amount'].mean(),
            'avg_distance': df['trip_distance'].mean(),
            'avg_speed':    np.nanmean(speed[np.isfinite(speed)]),
            'avg_duration': np.nanmean(dur_min[np.isfinite(dur_min)]),
            'vendor1_count': (df['VendorID'] == 1).sum(),
            'vendor2_count': (df['VendorID'] == 2).sum(),
            'vendor6_count': (df['VendorID'] == 6).sum(),
            'vendor1_violations': ((df['VendorID'] == 1) & ((df['fare_amount'] < 0) | (df['trip_distance'] <= 0) | df['passenger_count'].isna())).sum(),
            'vendor2_violations': ((df['VendorID'] == 2) & ((df['fare_amount'] < 0) | (df['trip_distance'] <= 0) | df['passenger_count'].isna())).sum(),
        })
        del df

    df_stats = pd.DataFrame(records)
    df_stats['year_month_str'] = df_stats['year_month'].astype(str)
    df_stats.to_json(cache, orient='records', indent=2)
    return df_stats


def compute_vendor_monthly(force_reload=False):
    """Return DataFrame with per-vendor per-month violation rates."""
    cache = CACHE_DIR / 'vendor_monthly.json'
    if cache.exists() and not force_reload:
        return pd.read_json(cache, precise_float=True)

    records = []
    for m in ALL_MONTHS:
        path = RAW_DIR / f'yellow_tripdata_{m}.parquet'
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for vid in [1, 2, 6]:
            vd = df[df['VendorID'] == vid]
            n_v = len(vd)
            if n_v > 0:
                v_v = ((vd['fare_amount'] < 0) | (vd['trip_distance'] <= 0) | vd['passenger_count'].isna()).sum()
                records.append({
                    'year_month': m, 'VendorID': vid,
                    'total': n_v, 'violations': v_v,
                    'violation_rate': v_v / n_v * 100
                })
        del df

    df_vendor = pd.DataFrame(records)
    df_vendor.to_json(cache, orient='records', indent=2)
    return df_vendor


def compute_violation_by_month(force_reload=False):
    """Return DataFrame with monthly violation counts by type for T5 figure."""
    cache = CACHE_DIR / 'violation_by_month.json'
    if cache.exists() and not force_reload:
        return pd.read_json(cache, precise_float=True)

    VIOL_COLS = [
        'v_null_passenger', 'v_zero_distance', 'v_neg_fare',
        'v_null_ratecode', 'v_null_pu', 'v_null_do',
        'v_dropoff_before_pickup', 'v_speed_high',
    ]

    records = []
    for m in ALL_MONTHS:
        path = RAW_DIR / f'yellow_tripdata_{m}.parquet'
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

        df['v_null_passenger']  = df['passenger_count'].isna()
        df['v_zero_distance']   = df['trip_distance'] <= 0
        df['v_neg_fare']        = df['fare_amount'] < 0
        df['v_null_ratecode']   = df['RatecodeID'].isna()
        df['v_null_pu']         = df['PULocationID'].isna()
        df['v_null_do']         = df['DOLocationID'].isna()
        df['v_dropoff_before_pickup'] = df['tpep_dropoff_datetime'] < df['tpep_pickup_datetime']

        dur_min = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
        speed = np.where(dur_min > 0, df['trip_distance'] / (dur_min / 60), 0)
        df['v_speed_high'] = speed > 100

        row = {'year_month': m, 'total': len(df)}
        for col in VIOL_COLS:
            row[col] = int(df[col].sum())
        records.append(row)
        del df

    df_viol = pd.DataFrame(records)
    df_viol.to_json(cache, orient='records', indent=2)
    return df_viol


def load_jan_sample(sample_size=200_000, random_state=42):
    """Load Jan 2024 sample with derived features for scatter/barchart figures."""
    path = RAW_DIR / 'yellow_tripdata_2024-01.parquet'
    df = pd.read_parquet(path)
    df['tpep_pickup_datetime']  = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    hours = df['tpep_pickup_datetime'].dt.hour
    ratecodes = df['RatecodeID']
    time_bin = pd.cut(hours, bins=[-1, 5, 11, 17, 24],
                      labels=['night', 'morning', 'afternoon', 'evening']).astype(str)
    ratecode_bin = np.where(ratecodes == 1, 'standard', 'special')
    df['time_bin']     = time_bin
    df['ratecode_bin'] = ratecode_bin
    df['context']      = time_bin + '_' + ratecode_bin

    df['duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    df['v_null_passenger'] = df['passenger_count'].isna()
    df['v_zero_distance']  = df['trip_distance'] <= 0
    df['v_neg_fare']        = df['fare_amount'] < 0
    df['v_null_ratecode']   = df['RatecodeID'].isna()

    any_viol = (df['v_null_passenger'] | df['v_zero_distance'] |
                df['v_neg_fare'] | df['v_null_ratecode'])
    df['any_violation'] = any_viol

    return df.sample(n=min(sample_size, len(df)), random_state=random_state)


def load_month_sample(month='2024-01', sample_size=200_000, random_state=42):
    """Load a specific month sample with derived features."""
    path = RAW_DIR / f'yellow_tripdata_{month}.parquet'
    df = pd.read_parquet(path)
    df['tpep_pickup_datetime']  = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    hours = df['tpep_pickup_datetime'].dt.hour
    ratecodes = df['RatecodeID']
    time_bin = pd.cut(hours, bins=[-1, 5, 11, 17, 24],
                      labels=['night', 'morning', 'afternoon', 'evening']).astype(str)
    ratecode_bin = np.where(ratecodes == 1, 'standard', 'special')
    df['time_bin']     = time_bin
    df['ratecode_bin'] = ratecode_bin
    df['context']      = time_bin + '_' + ratecode_bin

    df['duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    df['v_null_passenger'] = df['passenger_count'].isna()
    df['v_zero_distance']  = df['trip_distance'] <= 0
    df['v_neg_fare']        = df['fare_amount'] < 0
    df['v_null_ratecode']   = df['RatecodeID'].isna()

    any_viol = (df['v_null_passenger'] | df['v_zero_distance'] |
                df['v_neg_fare'] | df['v_null_ratecode'])
    df['any_violation'] = any_viol

    return df.sample(n=min(sample_size, len(df)), random_state=random_state)
