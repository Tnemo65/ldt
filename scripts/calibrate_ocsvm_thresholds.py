#!/usr/bin/env python3
"""
Calibrate OneClassSVM thresholds on clean baseline.
Uses percentile-based approach (e.g., 95th percentile of clean scores).
"""

import pickle
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, '.')
from src.features.vectorizer import FeatureVectorizer


def calibrate_thresholds(
    model_path: str = 'models/ocsvm_model.pkl',
    data_path: str = 'data/clean/jan_2024_clean_baseline.parquet',
    scaler_path: str = 'models/scaler.pkl',
    percentile: int = 95,
    sample_size: int = 50000
):
    """Calibrate thresholds on clean data."""
    print("="*60)
    print("OneClassSVM Threshold Calibration")
    print("="*60)

    # Load model and scaler
    print(f"\n1. Loading model and scaler...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load and sample data
    print(f"\n2. Loading clean data...")
    df = pd.read_parquet(data_path)

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"   Sampled {sample_size:,} records")
    else:
        print(f"   Using all {len(df):,} records")

    # Vectorize and score
    print(f"\n3. Scoring clean records...")
    vectorizer = FeatureVectorizer()
    scores = []

    for idx, row in df.iterrows():
        try:
            features = vectorizer.transform(row.to_dict())
            features_scaled = scaler.transform([features])[0]
            feature_dict = {i: float(v) for i, v in enumerate(features_scaled)}
            score = model.score_one(feature_dict)
            scores.append(score)
        except:
            pass

        if (idx + 1) % 10000 == 0:
            print(f"   Scored: {idx+1:,}")

    scores = np.array(scores)
    print(f"   ✓ Scored {len(scores):,} records")

    # Compute threshold
    print(f"\n4. Computing {percentile}th percentile threshold...")
    threshold = np.percentile(scores, percentile)

    print(f"\n   Score distribution:")
    print(f"   - Min:    {scores.min():.4f}")
    print(f"   - 5th:    {np.percentile(scores, 5):.4f}")
    print(f"   - 25th:   {np.percentile(scores, 25):.4f}")
    print(f"   - Median: {np.median(scores):.4f}")
    print(f"   - 75th:   {np.percentile(scores, 75):.4f}")
    print(f"   - 95th:   {np.percentile(scores, 95):.4f}")
    print(f"   - 98th:   {np.percentile(scores, 98):.4f}")
    print(f"   - Max:    {scores.max():.4f}")
    print(f"\n   Threshold ({percentile}th): {threshold:.4f}")

    # Create threshold file
    threshold_data = {
        'version': '1.0',
        'model': 'OneClassSVM',
        'percentile': percentile,
        'global_threshold': float(threshold),
        'thresholds': {},  # Can add per-context thresholds later
        'calibration_info': {
            'sample_size': len(scores),
            'score_mean': float(scores.mean()),
            'score_std': float(scores.std()),
            'score_min': float(scores.min()),
            'score_max': float(scores.max())
        }
    }

    output_path = 'models/ocsvm_thresholds.json'
    with open(output_path, 'w') as f:
        json.dump(threshold_data, f, indent=2)

    print(f"\n5. Saved thresholds to: {output_path}")
    print(f"\n{'='*60}")
    print("✅ Calibration Complete!")
    print(f"   Threshold: {threshold:.4f}")
    print(f"   Interpretation: Scores > {threshold:.4f} are anomalies")
    print(f"{'='*60}")

    return threshold


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--percentile', type=int, default=95)
    parser.add_argument('--sample-size', type=int, default=50000)
    args = parser.parse_args()

    calibrate_thresholds(percentile=args.percentile, sample_size=args.sample_size)
