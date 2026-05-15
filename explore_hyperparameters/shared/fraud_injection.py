"""
Fraud injection for NYC Taxi anomaly detection.
Provides multiple fraud types and drift injection for experiments.
"""

import numpy as np
import pandas as pd
from .data_loader import (
    N_FEATURES, EPS, JFK_FLAT_FARE, location_to_neighborhood,
    zone_to_grid, get_context_id, clean, extract_features
)


def is_canary_clean(df):
    """Check if a record passes all canary (clean) filters."""
    n = len(df)
    fare  = df['fare_amount'].fillna(0).values
    dist  = df['trip_distance'].fillna(0).values
    dur   = df['dur_min'].fillna(1).values
    pax   = df['passenger_count'].fillna(1).values
    spd   = df['speed_mph'].fillna(0).values
    tip   = df.get('tip_amount', pd.Series(np.zeros(n))).fillna(0).values
    ptype = df.get('payment_type', pd.Series(np.ones(n))).fillna(1).values

    clean_mask = np.ones(n, dtype=bool)
    clean_mask &= (fare > 0) & (fare <= 500)
    clean_mask &= (dist > 0) | (fare == 0)
    fpm = np.where(dur > 0, fare / np.maximum(dur, 0.01), 0)
    clean_mask &= (fpm <= 5.0) | (dur == 0)
    clean_mask &= (spd > 0) & (spd <= 80)
    clean_mask &= (pax >= 1) & (pax <= 6)
    clean_mask &= ~((ptype == 1) & (tip == 0))
    return clean_mask


def inject_fraud(df, rng, fraud_type='mixed', anomaly_rate=0.05,
                  type1_pct=0.60, type2_pct=0.30, type3_pct=0.10):
    """Inject realistic fraud into NYC taxi data.

    Types:
      Type1: Short-trip fare fraud (dist<1mi, fare $40-80 injected)
      Type2: Duration manipulation (dist 2-4mi, duration x8-15x)
      Type3: Ratecode mismatch (JFK flat fare with standard ratecode)

    Args:
        df:         Cleaned dataframe
        rng:        np.random.RandomState
        fraud_type: 'mixed', 'type1_only', 'type2_only', 'type3_only'
        anomaly_rate: Fraction of test set to inject anomalies
        type1_pct:  Fraction of anomalies that are Type1
        type2_pct:  Fraction of anomalies that are Type2
        type3_pct:  Fraction of anomalies that are Type3

    Returns:
        (df_injected, y_labels) where y_labels[i]=1 if record i is fraud
    """
    df = df.copy()
    n  = len(df)
    y  = np.zeros(n, dtype=np.int8)
    n_anom = int(n * anomaly_rate)

    canary_clean = is_canary_clean(df)
    ratecode = df['RatecodeID'].fillna(1).values.astype(np.float32)
    dist_arr = df['trip_distance'].fillna(0).values.astype(np.float32)

    is_standard = (ratecode == 1.0)

    # Build candidate pools
    type1_pool = np.where(is_standard & canary_clean & (dist_arr < 1.0))[0]
    type2_pool = np.where(is_standard & canary_clean &
                          (dist_arr >= 2.0) & (dist_arr <= 4.0))[0]
    type3_pool = np.where(is_standard & canary_clean)[0]

    if fraud_type == 'type1_only':
        type1_pct, type2_pct, type3_pct = 1.0, 0.0, 0.0
    elif fraud_type == 'type2_only':
        type1_pct, type2_pct, type3_pct = 0.0, 1.0, 0.0
    elif fraud_type == 'type3_only':
        type1_pct, type2_pct, type3_pct = 0.0, 0.0, 1.0

    n1 = int(n_anom * type1_pct)
    n2 = int(n_anom * type2_pct)
    n3 = int(n_anom * type3_pct)

    pool1 = type1_pool[:n1] if len(type1_pool) >= n1 else np.array([], dtype=int)
    pool2 = type2_pool[:n2] if len(type2_pool) >= n2 else np.array([], dtype=int)
    pool3 = type3_pool[:n3] if len(type3_pool) >= n3 else np.array([], dtype=int)

    # Ensure we have exactly n_anom by sampling with replacement if needed
    pool = np.concatenate([pool1, pool2, pool3])
    if len(pool) < n_anom:
        shortfall = n_anom - len(pool)
        extra = rng.choice(pool, size=shortfall, replace=True)
        pool  = np.concatenate([pool, extra])
    pool = pool[:n_anom]
    y[pool] = 1

    # Apply fraud modifications
    for idx in pool:
        if idx in pool1:
            # Type 1: Short trip + inflated fare ($40-80)
            new_fare = float(rng.uniform(40.0, 80.0))
            df.at[df.index[idx], 'fare_amount'] = new_fare
            df.at[df.index[idx], 'total_amt']   = new_fare
        elif idx in pool2:
            # Type 2: Duration manipulation (x8-15x)
            dur_mult = rng.uniform(8.0, 15.0)
            old_dur  = float(df.at[df.index[idx], 'dur_min'])
            old_dist = float(df.at[df.index[idx], 'trip_distance'])
            new_dur  = old_dur * dur_mult
            df.at[df.index[idx], 'dur_min']   = new_dur
            df.at[df.index[idx], 'speed_mph'] = old_dist / max(new_dur / 60.0, 0.01)
        else:
            # Type 3: JFK flat fare charged but no actual JFK trip
            df.at[df.index[idx], 'fare_amount'] = JFK_FLAT_FARE
            df.at[df.index[idx], 'total_amt']   = JFK_FLAT_FARE
            df.at[df.index[idx], 'RatecodeID']  = 2.0

    return df, y


def inject_drift(df, rng, drift_type='abrupt', drift_pct=0.10,
                  drift_feature_idx=6, drift_magnitude=5.0):
    """Inject distribution drift into features.

    Args:
        df: DataFrame
        rng: np.random.RandomState
        drift_type: 'abrupt' or 'gradual'
        drift_pct: Fraction of records to modify
        drift_feature_idx: Which feature to shift (default: fare_per_mile)
        drift_magnitude: Number of std deviations to shift

    Returns:
        df with drift applied
    """
    df = df.copy()
    n_drift = int(len(df) * drift_pct)
    drift_idx = rng.choice(len(df), size=n_drift, replace=False)

    X_raw = extract_features(df)
    feature_mean = X_raw[:, drift_feature_idx].mean()
    feature_std  = X_raw[:, drift_feature_idx].std() + 1e-8

    if drift_type == 'abrupt':
        # All drifted records shifted uniformly
        X_raw[drift_idx, drift_feature_idx] += drift_magnitude * feature_std
    elif drift_type == 'gradual':
        # Linear increase from middle onwards
        mid = len(df) // 2
        for i in range(mid, len(df)):
            if i in drift_idx:
                progress = (i - mid) / (len(df) - mid)
                X_raw[i, drift_feature_idx] += drift_magnitude * progress * feature_std

    # Note: df is modified in-place but we only track feature changes
    # For evaluation purposes, we return drift positions
    return df, drift_idx


def get_pool_info(df, rng, anomaly_rate=0.05):
    """Return pool information for fraud injection."""
    canary_clean = is_canary_clean(df)
    ratecode = df['RatecodeID'].fillna(1).values.astype(np.float32)
    dist_arr = df['trip_distance'].fillna(0).values.astype(np.float32)
    is_standard = (ratecode == 1.0)

    type1_pool = np.where(is_standard & canary_clean & (dist_arr < 1.0))[0]
    type2_pool = np.where(is_standard & canary_clean &
                          (dist_arr >= 2.0) & (dist_arr <= 4.0))[0]
    type3_pool = np.where(is_standard & canary_clean)[0]

    return {
        'type1_available': len(type1_pool),
        'type2_available': len(type2_pool),
        'type3_available': len(type3_pool),
        'canary_clean_pct': canary_clean.mean(),
    }
