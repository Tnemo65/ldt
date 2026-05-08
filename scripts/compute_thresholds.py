"""
Compute 4D context-aware thresholds (95th percentile).
Spec: Lines 2901-2904
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.vectorizer import FeatureVectorizer
from sklearn.preprocessing import StandardScaler

def compute_thresholds(data_path: Path, output_path: Path):
    """Compute 95th percentile thresholds per 4D context."""

    print("Loading clean baseline...")
    df = pd.read_parquet(data_path)
    print(f"Records: {len(df):,}")

    # Load neighborhood mapping
    with open('src/config/neighborhood_mapping.json') as f:
        neighbor_map = json.load(f)['mapping']

    # Add derived features
    df['pickup_dt'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['dropoff_dt'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['duration'] = (df['dropoff_dt'] - df['pickup_dt']).dt.total_seconds()
    df['hour'] = df['pickup_dt'].dt.hour
    df['day'] = df['pickup_dt'].dt.dayofweek

    # 4D context dimensions
    df['trip_type'] = pd.cut(df['trip_distance'], bins=[0, 2, 10, 100],
                               labels=['short', 'medium', 'long'])
    df['time_window'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24],
                                 labels=['night', 'morning', 'afternoon', 'evening'])
    df['day_type'] = df['day'].apply(lambda x: 'weekend' if x >= 5 else 'weekday')
    df['neighborhood'] = df['PULocationID'].astype(str).map(neighbor_map)

    # Mock anomaly scores (random for now, will be replaced by iForest)
    np.random.seed(42)
    df['anomaly_score'] = np.random.random(len(df))

    # Compute 95th percentile per 4D context
    print("\nComputing 4D thresholds (95th percentile)...")
    thresholds = {}

    for trip_type in ['short', 'medium', 'long']:
        for time_win in ['night', 'morning', 'afternoon', 'evening']:
            for day_type in ['weekday', 'weekend']:
                for neighbor in set(df['neighborhood'].dropna()):
                    key = f"{trip_type}_{time_win}_{day_type}_{neighbor}"

                    mask = (
                        (df['trip_type'] == trip_type) &
                        (df['time_window'] == time_win) &
                        (df['day_type'] == day_type) &
                        (df['neighborhood'] == neighbor)
                    )

                    if mask.sum() > 100:  # Minimum samples
                        threshold = df.loc[mask, 'anomaly_score'].quantile(0.95)
                        thresholds[key] = float(threshold)

    # Global fallback
    global_threshold = df['anomaly_score'].quantile(0.95)

    print(f"Computed {len(thresholds)} context-specific thresholds")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'version': '1.0',
            'percentile': 95,
            'global_threshold': float(global_threshold),
            'thresholds': thresholds
        }, f, indent=2)

    print(f"✅ Saved to {output_path}")

def main():
    data_path = Path('data/clean/jan_2024_clean_baseline.parquet')
    output_path = Path('src/config/threshold_matrix.json')
    compute_thresholds(data_path, output_path)

if __name__ == "__main__":
    main()
