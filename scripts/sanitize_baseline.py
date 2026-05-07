"""
3-step baseline sanitization (CRITICAL).
Spec: Lines 2848-2882
"""

import pandas as pd
from pathlib import Path
import sys

def sanitize_baseline(input_path: Path, output_path: Path):
    """3-step sanitization: physical filter + IQR outliers + verification."""

    print("Loading January 2024 data...")
    df = pd.read_parquet(input_path)
    print(f"Raw records: {len(df):,}")

    # Step 1: Physical violation filter
    print("\nStep 1: Physical violation filter")
    df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime',
                            'trip_distance', 'fare_amount'])
    df = df[df['fare_amount'] > 0]
    df = df[df['trip_distance'] > 0]
    df = df[df['passenger_count'].between(1, 6)]

    # Speed filter (assume max 100 mph)
    df['trip_duration'] = (pd.to_datetime(df['tpep_dropoff_datetime']) -
                            pd.to_datetime(df['tpep_pickup_datetime'])).dt.total_seconds() / 3600
    df['speed_mph'] = df['trip_distance'] / df['trip_duration'].clip(lower=0.01)
    df = df[df['speed_mph'] <= 100]

    print(f"After physical filter: {len(df):,}")

    # Step 2: IQR outlier removal (3×IQR, stricter)
    print("\nStep 2: IQR outlier removal")
    for feature in ['fare_amount', 'trip_distance', 'trip_duration']:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR

        before = len(df)
        df = df[(df[feature] >= lower) & (df[feature] <= upper)]
        removed = before - len(df)
        print(f"  {feature}: removed {removed:,} outliers")

    print(f"After IQR filter: {len(df):,}")

    # Step 3: Verification
    print("\nStep 3: Verification")
    null_rate = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100

    violations = (
        (df['fare_amount'] <= 0) |
        (df['trip_distance'] <= 0) |
        (df['passenger_count'] > 6) |
        (df['passenger_count'] == 0)
    )
    viol_rate = violations.sum() / len(df) * 100

    print(f"Null rate: {null_rate:.3f}%")
    print(f"Violation rate: {viol_rate:.3f}%")

    if null_rate >= 0.5 or viol_rate >= 0.5:
        print("❌ FAILED: Metrics exceed thresholds")
        sys.exit(1)

    # Save clean baseline
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\n✅ Saved clean baseline: {len(df):,} records")
    print(f"Output: {output_path}")

    return df

def main():
    input_path = Path('data/raw/yellow_tripdata_2024-01.parquet')
    output_path = Path('data/clean/jan_2024_clean_baseline.parquet')

    sanitize_baseline(input_path, output_path)

if __name__ == "__main__":
    main()
