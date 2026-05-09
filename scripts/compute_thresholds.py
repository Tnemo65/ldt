#!/usr/bin/env python3
"""
Compute 4D context-aware thresholds (95th percentile) using real iForest inference.

Workflow:
  1. Load trained iForest model (models/iforest_model.pkl)
  2. Load fitted 21D scaler (models/scaler.pkl)
  3. Run inference on sample of clean baseline to get real anomaly scores
  4. Compute 95th percentile per (trip_type x time_window x day_type x neighborhood)

Prerequisites:
  python src/ml/train_iforest.py

Spec: Lines 2901-2904
"""

import argparse
import sys
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd


def compute_thresholds(
    data_path='data/clean/jan_2024_clean_baseline.parquet',
    model_path='models/iforest_model.pkl',
    scaler_path='models/scaler.pkl',
    output_path='src/config/threshold_matrix.json',
    sample_size=200_000,
):
    """Compute 95th percentile thresholds per 4D context using real iForest scores."""

    # Load neighborhood mapping
    with open('src/config/neighborhood_mapping.json') as f:
        neighbor_map = json.load(f)['mapping']

    # Load data
    print("Loading clean baseline...")
    df = pd.read_parquet(data_path)
    print(f"Records: {len(df):,}")

    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled: {len(df):,} records for inference")
    else:
        print(f"Running inference on: {len(df):,} records")

    # Add derived features for context
    df['pickup_dt'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['dropoff_dt'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['hour'] = df['pickup_dt'].dt.hour
    df['day'] = df['pickup_dt'].dt.dayofweek

    # 4D context dimensions
    df['trip_type'] = pd.cut(
        df['trip_distance'], bins=[0, 2, 10, 100], labels=['short', 'medium', 'long']
    )
    df['time_window'] = pd.cut(
        df['hour'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening']
    )
    df['day_type'] = df['day'].apply(lambda x: 'weekend' if x >= 5 else 'weekday')
    df['neighborhood'] = df['PULocationID'].astype(str).map(neighbor_map)

    # Load scaler
    print("Loading 21D StandardScaler...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"  Scaler dim: {len(scaler.mean_)}D")

    # Load model
    if not Path(model_path).exists():
        print(f"\n  Model not found: {model_path}")
        print("  Run: python src/ml/train_iforest.py")
        print("  Then re-run this script.")
        return False

    print("Loading iForest model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load vectorizer
    sys.path.insert(0, '.')
    from src.features.vectorizer import FeatureVectorizer
    vectorizer = FeatureVectorizer()

    # Real inference (batch for speed)
    print("\nRunning IsolationForest inference (batch mode)...")
    vectorizer = FeatureVectorizer()
    X = vectorizer.transform_batch(df)
    X_scaled = scaler.transform(X)
    # sklearn: negative scores (lower = more anomalous), negate for River-like semantics
    raw_scores = model.score_samples(X_scaled)
    anomaly_scores = (-raw_scores).tolist()

    df['anomaly_score'] = anomaly_scores
    print(f"\n  Score stats: mean={np.mean(anomaly_scores):.4f}, "
          f"std={np.std(anomaly_scores):.4f}, "
          f"max={np.max(anomaly_scores):.4f}")

    # Compute 95th percentile per 4D context
    print("\nComputing 4D thresholds (95th percentile)...")
    thresholds = {}

    valid_neighborhoods = set(df['neighborhood'].dropna())
    for trip_type in ['short', 'medium', 'long']:
        for time_win in ['night', 'morning', 'afternoon', 'evening']:
            for day_type in ['weekday', 'weekend']:
                for neighbor in valid_neighborhoods:
                    key = f"{trip_type}_{time_win}_{day_type}_{neighbor}"
                    mask = (
                        (df['trip_type'] == trip_type) &
                        (df['time_window'] == time_win) &
                        (df['day_type'] == day_type) &
                        (df['neighborhood'] == neighbor)
                    )
                    if mask.sum() > 100:
                        threshold = float(df.loc[mask, 'anomaly_score'].quantile(0.95))
                        thresholds[key] = threshold

    global_threshold = float(df['anomaly_score'].quantile(0.95))

    print(f"Computed {len(thresholds)} context-specific thresholds")
    print(f"Global 95th percentile: {global_threshold:.6f}")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'version': '1.0',
            'percentile': 95,
            'global_threshold': global_threshold,
            'thresholds': thresholds,
            'model_used': str(model_path),
            'sample_size': n,
        }, f, indent=2)

    print(f"\n  Saved: {output_path}")
    print("  PASS")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute 4D context thresholds with real iForest scores')
    parser.add_argument('--data', default='data/clean/jan_2024_clean_baseline.parquet')
    parser.add_argument('--model', default='models/iforest_model.pkl')
    parser.add_argument('--scaler', default='models/scaler.pkl')
    parser.add_argument('--output', default='src/config/threshold_matrix.json')
    parser.add_argument('--sample', type=int, default=200_000)
    args = parser.parse_args()

    success = compute_thresholds(
        data_path=args.data,
        model_path=args.model,
        scaler_path=args.scaler,
        output_path=args.output,
        sample_size=args.sample,
    )
    sys.exit(0 if success else 1)
