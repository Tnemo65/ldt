"""
Shared data loader for NYC Taxi anomaly detection experiments.
Provides consistent data loading, cleaning, and feature extraction
across all hyperparameter experiments.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

N_FEATURES = 34
N_NEIGHBORHOODS = 10
DATA_PATH = 'C:/proj/ldt/data/nyc_taxi_300k.parquet'

# Denominators used by FeatureVectorizer (must match exactly)
FARE_PER_MILE_DENOM = 2.5
FARE_PER_MIN_DENOM = 0.67
SPEED_DENOM = 12.0
EPS = 1e-8
JFK_FLAT_FARE = 70.0


def location_to_neighborhood(loc_id):
    """Map NYC taxi LocationID to neighborhood index (0-9)."""
    if pd.isna(loc_id):
        return 9
    z = int(loc_id)
    if 1 <= z <= 43:     return 0  # manhattan
    elif 44 <= z <= 103: return 4  # bronx
    elif 104 <= z <= 127: return 1  # brooklyn
    elif 128 <= z <= 148: return 2  # queens_lower
    elif 149 <= z <= 161: return 3  # queens_upper
    elif 162 <= z <= 181: return 5  # staten_island
    elif 182 <= z <= 196: return 6  # ewr
    elif 217 <= z <= 229: return 7  # jfk
    elif 230 <= z <= 234: return 8  # nalp
    else:                 return 9  # unknown


def zone_to_grid(zone_id):
    """Map LocationID to 4x4 grid coordinates."""
    z = int(zone_id) if not pd.isna(zone_id) else 0
    if z <= 0:
        return 0, 0
    return (z - 1) % 16, (z - 1) // 16


def get_context_id(hour, dow, ratecode):
    """Encode (hour, day_of_week, ratecode) into context cell ID (0-7).

    Cell encoding:
      bit2: is_special  = ratecode > 1
      bit1: is_night   = hour >= 20 OR hour < 6
      bit0: is_weekend = dow >= 5
    """
    is_special = 1 if ratecode > 1 else 0
    is_night   = 1 if (hour >= 20 or hour < 6) else 0
    is_weekend = 1 if dow >= 5 else 0
    return (is_special << 2) | (is_night << 1) | is_weekend


def clean(df):
    """Apply standard cleaning filters to NYC taxi data."""
    df = df.copy()
    for col in ['PULocationID', 'DOLocationID', 'fare_amount',
                'trip_distance', 'passenger_count', 'RatecodeID']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['PULocationID', 'DOLocationID', 'fare_amount',
                            'trip_distance', 'passenger_count', 'RatecodeID'])
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    df = df[(df['PULocationID'] >= 1) & (df['PULocationID'] <= 265)]
    df = df[(df['DOLocationID'] >= 1) & (df['DOLocationID'] <= 265)]
    df = df[(df['RatecodeID'] >= 1) & (df['RatecodeID'] <= 99)]
    df['fare_amount']   = df['fare_amount'].abs()
    df['trip_distance'] = df['trip_distance'].abs()
    pickup  = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    dropoff = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df['duration_s'] = (dropoff - pickup).dt.total_seconds()
    df = df[(df['duration_s'] > 0) & (df['duration_s'] < 86400)]
    df['speed_mph'] = df['trip_distance'] / (df['duration_s'] / 3600)
    df = df[(df['speed_mph'] > 0) & (df['speed_mph'] < 100)]
    for col in ['fare_amount', 'trip_distance', 'duration_s']:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df = df[(df[col] >= lo) & (df[col] <= hi)]
    df['dur_min']   = df['duration_s'] / 60.0
    df['total_amt'] = df.get('total_amount', df['fare_amount'])
    return df.reset_index(drop=True)


def extract_features(df):
    """Extract 34D feature vector from dataframe. Matches FeatureVectorizer exactly."""
    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    pickup   = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    dropoff  = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    hour     = pickup.dt.hour.fillna(12).astype(np.int32).values
    dow      = pickup.dt.dayofweek.fillna(0).astype(np.int32).values
    dist     = df['trip_distance'].fillna(0).values.astype(np.float32)
    dur      = df['dur_min'].fillna(1).values.astype(np.float32)
    fare     = df['fare_amount'].fillna(0).values.astype(np.float32)
    pax      = df['passenger_count'].fillna(1).values.astype(np.float32)
    total    = df['total_amt'].fillna(0).values.astype(np.float32)
    spd      = df['speed_mph'].fillna(0).values.astype(np.float32)
    ratecode = df['RatecodeID'].fillna(1).values.astype(np.float32)
    pu_loc   = df['PULocationID'].fillna(0).values
    do_loc   = df['DOLocationID'].fillna(0).values

    eps = np.float32(EPS)

    # Raw features
    X[:, 0] = dist;  X[:, 1] = dur;  X[:, 2] = fare
    X[:, 3] = pax;   X[:, 4] = total; X[:, 5] = spd

    # Derived ratios
    X[:, 6]  = fare / np.maximum(dist, eps)
    X[:, 7]  = fare / np.maximum(dur, eps)
    X[:, 8]  = fare / np.maximum(pax, eps)

    # Temporal
    X[:, 9]  = hour.astype(np.float32)
    X[:, 10] = dow.astype(np.float32)
    X[:, 11] = (dow >= 5).astype(np.float32)

    # Spatial grid
    pu_gx = np.zeros(n, dtype=np.float32); pu_gy = np.zeros(n, dtype=np.float32)
    do_gx = np.zeros(n, dtype=np.float32); do_gy = np.zeros(n, dtype=np.float32)
    for i in range(n):
        pux, puy = zone_to_grid(pu_loc[i])
        dox, doy = zone_to_grid(do_loc[i])
        pu_gx[i], pu_gy[i] = float(pux), float(puy)
        do_gx[i], do_gy[i] = float(dox), float(doy)
    X[:, 12] = pu_gx; X[:, 13] = pu_gy
    X[:, 14] = do_gx; X[:, 15] = do_gy

    # Normalized
    X[:, 16] = X[:, 6] / np.float32(FARE_PER_MILE_DENOM)
    X[:, 17] = X[:, 7] / np.float32(FARE_PER_MIN_DENOM)
    X[:, 18] = spd / np.float32(SPEED_DENOM)
    X[:, 19] = pax / np.maximum(dist, eps)

    # Cyclic temporal
    X[:, 20] = np.sin(np.float32(2 * np.pi) * hour.astype(np.float32) / 24.0)
    X[:, 21] = np.cos(np.float32(2 * np.pi) * hour.astype(np.float32) / 24.0)
    X[:, 22] = np.sin(np.float32(2 * np.pi) * dow.astype(np.float32) / 7.0)
    X[:, 23] = np.cos(np.float32(2 * np.pi) * dow.astype(np.float32) / 7.0)
    X[:, 24] = dist * dist

    # Ratecode one-hot
    for i, rc in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        X[:, 25 + i] = (ratecode == rc).astype(np.float32)

    X[:, 30] = ((hour >= 20) | (hour <= 6)).astype(np.float32)
    X[:, 31] = np.log1p(fare)
    X[:, 32] = np.log1p(dist)
    X[:, 33] = np.abs(pu_gy - do_gy)

    return np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)


def load_data(n_warmup=10000, n_test=15000, data_path=None):
    """Load NYC taxi data and split into warmup / test sets.

    Args:
        n_warmup: Number of records for model warmup (training)
        n_test:   Number of records for evaluation
        data_path: Path to parquet file. Defaults to nyc_taxi_300k.parquet

    Returns:
        dict with keys: X_warmup, X_test, y_test, df_warmup, df_test,
                        nb_warmup, nb_test, hr_warmup, hr_test,
                        dw_warmup, dw_test, rc_warmup, rc_test
    """
    path = data_path or DATA_PATH
    df_raw = pd.read_parquet(path)
    df_clean = clean(df_raw)

    # Take contiguous slice
    total_needed = n_warmup + n_test
    if len(df_clean) < total_needed:
        raise ValueError(
            f"Dataset too small: need {total_needed} records, got {len(df_clean)}")
    df_clean = df_clean.iloc[:total_needed].reset_index(drop=True)
    df_warmup = df_clean.iloc[:n_warmup].reset_index(drop=True)
    df_test   = df_clean.iloc[n_warmup:n_warmup + n_test].reset_index(drop=True)

    X_warmup = extract_features(df_warmup)
    X_test   = extract_features(df_test)

    # Extract context metadata
    def get_meta(df_):
        pickup = pd.to_datetime(df_['tpep_pickup_datetime'], errors='coerce')
        hr = pickup.dt.hour.fillna(12).astype(int).values
        dw = pickup.dt.dayofweek.fillna(0).astype(int).values
        rc = df_['RatecodeID'].fillna(1).astype(float).values
        nb = np.array([location_to_neighborhood(loc) for loc in df_['PULocationID'].fillna(1).values], dtype=int)
        return nb, hr, dw, rc

    nb_w, hr_w, dw_w, rc_w = get_meta(df_warmup)
    nb_t, hr_t, dw_t, rc_t = get_meta(df_test)

    # Fraud injection
    from .fraud_injection import inject_fraud
    df_test_inj, y_test = inject_fraud(df_test, np.random.RandomState(42), anomaly_rate=0.05)
    X_test_inj = extract_features(df_test_inj)

    return {
        'X_warmup': X_warmup, 'X_test': X_test_inj, 'y_test': y_test,
        'df_warmup': df_warmup, 'df_test': df_test_inj,
        'nb_warmup': nb_w, 'nb_test': nb_t,
        'hr_warmup': hr_w, 'hr_test': hr_t,
        'dw_warmup': dw_w, 'dw_test': dw_t,
        'rc_warmup': rc_w, 'rc_test': rc_t,
        'n_warmup': n_warmup, 'n_test': len(X_test_inj),
        'n_anomalies': int(y_test.sum()),
    }
