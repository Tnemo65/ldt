#!/usr/bin/env python3
"""
Multi-Strategy Anomaly Injection for Evaluation.

Injects anomalies using multiple strategies:
- Point anomalies (sudden spike in fare_amount)
- Contextual anomalies (high fare at unusual hour)
- Collective anomalies (consecutive anomalous trips)
- Uniform and Gaussian noise injection

Returns df with anomalies + labels array.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from datetime import datetime, timedelta


def inject_anomalies(
    df: pd.DataFrame,
    n_anomalies: int = 1000,
    seed: int = 42,
    anomaly_rate: Optional[float] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Inject anomalies into a NYC taxi DataFrame using multiple strategies.

    Args:
        df: Input DataFrame with taxi records
        n_anomalies: Target number of anomalies to inject
        seed: Random seed for reproducibility
        anomaly_rate: If provided, use this as the target anomaly rate instead of n_anomalies

    Returns:
        Tuple of (modified_df, labels)
        - modified_df: DataFrame with injected anomalies
        - labels: numpy array (1=anomaly, 0=normal)
    """
    np.random.seed(seed)
    df = df.copy()

    n = len(df)
    labels = np.zeros(n, dtype=int)

    if anomaly_rate is not None:
        n_anomalies = int(n * anomaly_rate)

    # Randomly select indices for anomalies
    anomaly_indices = np.random.choice(n, size=n_anomalies, replace=False)
    anomaly_indices = sorted(anomaly_indices)

    # Distribute anomaly types
    n_point = int(n_anomalies * 0.4)  # 40% point anomalies
    n_contextual = int(n_anomalies * 0.25)  # 25% contextual anomalies
    n_collective = int(n_anomalies * 0.2)  # 20% collective anomalies
    n_noise = n_anomalies - n_point - n_contextual - n_collective  # Remaining as noise

    idx = 0
    anomaly_types = []

    # 1. Point anomalies
    for _ in range(n_point):
        if idx >= len(anomaly_indices):
            break
        i = anomaly_indices[idx]
        anomaly_types.append(('point', i))
        _inject_point_anomaly(df, i)
        labels[i] = 1
        idx += 1

    # 2. Contextual anomalies
    for _ in range(n_contextual):
        if idx >= len(anomaly_indices):
            break
        i = anomaly_indices[idx]
        anomaly_types.append(('contextual', i))
        _inject_contextual_anomaly(df, i)
        labels[i] = 1
        idx += 1

    # 3. Collective anomalies (consecutive records)
    n_collective_groups = max(1, n_collective // 5)
    for _ in range(n_collective_groups):
        if idx >= len(anomaly_indices):
            break
        # Select 3-5 consecutive indices
        group_size = np.random.randint(3, 6)
        end_idx = min(idx + group_size, len(anomaly_indices))
        for j in range(idx, end_idx):
            anomaly_types.append(('collective', anomaly_indices[j]))
            _inject_collective_anomaly(df, anomaly_indices[j])
            labels[anomaly_indices[j]] = 1
        idx = end_idx

    # 4. Noise anomalies (remaining)
    for _ in range(n_noise):
        if idx >= len(anomaly_indices):
            break
        i = anomaly_indices[idx]
        anomaly_type = np.random.choice(['uniform', 'gaussian'])
        anomaly_types.append((anomaly_type, i))
        _inject_noise_anomaly(df, i, noise_type=anomaly_type)
        labels[i] = 1
        idx += 1

    return df, labels


def _inject_point_anomaly(df: pd.DataFrame, idx: int) -> None:
    """Inject a point anomaly: sudden spike in fare or distance."""
    anomaly_type = np.random.choice(['fare_spike', 'distance_spike', 'speed_spike'])

    if anomaly_type == 'fare_spike':
        # Spike fare by 5-10x
        multiplier = np.random.uniform(5, 10)
        df.iloc[idx, df.columns.get_loc('fare_amount')] *= multiplier

    elif anomaly_type == 'distance_spike':
        # Spike distance to unrealistic values
        df.iloc[idx, df.columns.get_loc('trip_distance')] = np.random.uniform(50, 200)

    elif anomaly_type == 'speed_spike':
        # Make speed unrealistic (high speed with short time)
        df.iloc[idx, df.columns.get_loc('trip_distance')] = np.random.uniform(30, 100)
        if 'trip_duration' in df.columns:
            df.iloc[idx, df.columns.get_loc('trip_duration')] = 0.1


def _inject_contextual_anomaly(df: pd.DataFrame, idx: int) -> None:
    """Inject a contextual anomaly: unusual values for the context."""
    # Parse datetime to get hour
    dt_str = df.iloc[idx].get('tpep_pickup_datetime', '')
    try:
        dt = pd.to_datetime(dt_str)
        hour = dt.hour
    except:
        hour = 12  # Default

    # High fare at unusual hour (e.g., 3 AM)
    if 2 <= hour <= 5:
        # Very expensive trip in the middle of night
        df.iloc[idx, df.columns.get_loc('fare_amount')] = np.random.uniform(100, 500)
        df.iloc[idx, df.columns.get_loc('trip_distance')] = np.random.uniform(5, 20)

    # Or unusual passenger count
    elif np.random.random() < 0.5:
        df.iloc[idx, df.columns.get_loc('passenger_count')] = 0  # Invalid

    # Or negative values
    else:
        df.iloc[idx, df.columns.get_loc('fare_amount')] = -np.random.uniform(10, 100)


def _inject_collective_anomaly(df: pd.DataFrame, idx: int) -> None:
    """Inject a collective anomaly: patterns in consecutive records."""
    anomaly_type = np.random.choice(['fare_pattern', 'tip_pattern'])

    if anomaly_type == 'fare_pattern':
        # Gradual increase in fares
        offset = np.random.uniform(2, 5)
        current_fare = df.iloc[idx].get('fare_amount', 10)
        df.iloc[idx, df.columns.get_loc('fare_amount')] = current_fare + offset

    elif anomaly_type == 'tip_pattern':
        # Unusual tip pattern (100% tip)
        fare = df.iloc[idx].get('fare_amount', 20)
        df.iloc[idx, df.columns.get_loc('tip_amount')] = fare  # 100% tip


def _inject_noise_anomaly(df: pd.DataFrame, idx: int, noise_type: str = 'gaussian') -> None:
    """Inject noise anomalies."""
    if noise_type == 'uniform':
        # Add uniform noise
        for col in ['fare_amount', 'trip_distance', 'tip_amount']:
            if col in df.columns:
                original = df.iloc[idx].get(col, 10)
                noise = np.random.uniform(-original * 0.5, original * 0.5)
                df.iloc[idx, df.columns.get_loc(col)] = original + noise

    elif noise_type == 'gaussian':
        # Add Gaussian noise (larger variance)
        for col in ['fare_amount', 'trip_distance', 'tip_amount']:
            if col in df.columns:
                original = df.iloc[idx].get(col, 10)
                noise = np.random.normal(0, original * 0.3)
                df.iloc[idx, df.columns.get_loc(col)] = max(0, original + noise)


def generate_synthetic_test_data(
    n_normal: int = 10000,
    n_anomalies: int = 1000,
    seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic NYC taxi data with known anomalies for testing.

    Args:
        n_normal: Number of normal records
        n_anomalies: Number of anomaly records
        seed: Random seed

    Returns:
        Tuple of (df, labels)
    """
    np.random.seed(seed)

    n_total = n_normal + n_anomalies
    base_date = datetime(2024, 1, 1, 0, 0, 0)

    records = []
    for i in range(n_total):
        # Time progression (time-ordered)
        dt = base_date + timedelta(minutes=i * 2)

        # Generate normal features
        hour = dt.hour
        is_rush = 6 <= hour < 10 or 16 <= hour < 20

        # Base fare varies by time
        base_fare = 8 + (5 if is_rush else 0) + np.random.uniform(-2, 2)
        distance = np.random.exponential(3) + 0.5

        record = {
            'tpep_pickup_datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'tpep_dropoff_datetime': (dt + timedelta(minutes=int(distance * 2 + 5))).strftime('%Y-%m-%d %H:%M:%S'),
            'PULocationID': np.random.randint(1, 264),
            'DOLocationID': np.random.randint(1, 264),
            'fare_amount': max(2.5, base_fare),
            'tip_amount': base_fare * np.random.uniform(0, 0.25),
            'tolls_amount': 0.0,
            'improvement_surcharge': 0.3,
            'trip_distance': max(0.1, distance),
            'passenger_count': np.random.randint(1, 5),
            'ratecodeID': 1,
            'payment_type': np.random.randint(1, 3),
            'trip_duration': distance * 2 + np.random.uniform(-2, 2),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Inject anomalies
    df_anom, labels = inject_anomalies(df, n_anomalies=n_anomalies, seed=seed)

    return df_anom, labels


if __name__ == '__main__':
    # Test the injection
    print("Generating synthetic test data...")
    df, labels = generate_synthetic_test_data(n_normal=5000, n_anomalies=500)

    print(f"Generated {len(df)} records")
    print(f"Anomalies: {int(np.sum(labels))} ({np.mean(labels)*100:.1f}%)")

    # Show sample anomalies
    anomaly_df = df[labels == 1].head(5)
    print("\nSample anomalies:")
    print(anomaly_df[['fare_amount', 'trip_distance', 'tip_amount']].to_string())
