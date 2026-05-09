#!/usr/bin/env python3
"""
Fit StandardScaler on Jan 2024 clean baseline (21D features).
Optimized: vectorized pandas operations instead of row-by-row.
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.features.vectorizer import FeatureVectorizer


def fit_scaler_fast(data_path='data/clean/jan_2024_clean_baseline.parquet',
                     output_path='models/scaler.pkl',
                     sample_size=100_000):
    """Fit StandardScaler on clean baseline using vectorized operations."""

    print("Loading clean baseline...")
    df = pd.read_parquet(data_path)
    print(f"Records: {len(df):,}")

    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled: {len(df):,} records for fitting")
    else:
        print(f"Fitting on full dataset: {len(df):,} records")

    print("\nVectorizing features (21D)...")
    vec = FeatureVectorizer()

    # Batch process with a progress indicator
    n = len(df)
    batch_size = 10000
    features_list = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = df.iloc[start:end]
        for _, row in batch.iterrows():
            features_list.append(vec.transform(row.to_dict()))
        print(f"  Processed: {end:,} / {n:,}")

    X = np.array(features_list, dtype=np.float64)
    print(f"Feature matrix: {X.shape}")

    # Verify 21D
    assert X.shape[1] == 21, f"Expected 21D, got {X.shape[1]}D"

    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(X)

    print(f"Mean (first 5): {scaler.mean_[:5]}")
    print(f"Scale (first 5): {scaler.scale_[:5]}")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n  Saved scaler to {output_path}")
    print(f"  Fitted on {len(X):,} samples")
    print(f"  Feature dim: {len(scaler.mean_)}D")
    print("  PASS")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/clean/jan_2024_clean_baseline.parquet')
    parser.add_argument('--output', default='models/scaler.pkl')
    parser.add_argument('--sample', type=int, default=100_000)
    args = parser.parse_args()
    fit_scaler_fast(args.data, args.output, args.sample)
