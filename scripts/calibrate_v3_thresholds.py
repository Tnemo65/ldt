#!/usr/bin/env python3
"""
Calibrate aggressive thresholds for v3 model.
v3 is too sensitive - need very high percentiles (99th+) to reduce FPR.

Strategy:
- Sample clean baseline data
- Score with v3 model
- Try multiple percentiles: 99th, 99.5th, 99.9th
- Save thresholds
"""

import pickle
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, '.')
from src.features.vectorizer import FeatureVectorizer


def calibrate_v3_thresholds(
    model_path: str = 'models/iforest_model_v3.pkl',
    data_path: str = 'data/clean/jan_2024_clean_baseline.parquet',
    scaler_path: str = 'models/scaler.pkl',
    percentiles: list = [99.0, 99.5, 99.9],
    sample_size: int = 100000
):
    """Calibrate thresholds for v3 model at high percentiles."""
    print("="*70)
    print("v3 MODEL AGGRESSIVE THRESHOLD CALIBRATION")
    print("="*70)

    # Load artifacts
    print(f"\n1. Loading artifacts...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"   ✓ Model and scaler loaded")

    # Load and sample data
    print(f"\n2. Loading clean baseline...")
    df = pd.read_parquet(data_path)

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"   ✓ Sampled {sample_size:,} records")
    else:
        print(f"   ✓ Using all {len(df):,} records")

    # Vectorize and score
    print(f"\n3. Scoring clean records with v3 model...")
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
            print(f"   Scored: {idx+1:,} / {sample_size:,}")

    scores = np.array(scores)
    print(f"   ✓ Scored {len(scores):,} clean records")

    # Compute percentile thresholds
    print(f"\n4. Computing threshold percentiles...")
    print(f"\n   Score distribution (clean data):")
    print(f"   - Min:     {scores.min():.6f}")
    print(f"   - 1st:     {np.percentile(scores, 1):.6f}")
    print(f"   - 5th:     {np.percentile(scores, 5):.6f}")
    print(f"   - 25th:    {np.percentile(scores, 25):.6f}")
    print(f"   - Median:  {np.median(scores):.6f}")
    print(f"   - 75th:    {np.percentile(scores, 75):.6f}")
    print(f"   - 90th:    {np.percentile(scores, 90):.6f}")
    print(f"   - 95th:    {np.percentile(scores, 95):.6f}")
    print(f"   - 98th:    {np.percentile(scores, 98):.6f}")
    print(f"   - 99th:    {np.percentile(scores, 99):.6f}")
    print(f"   - 99.5th:  {np.percentile(scores, 99.5):.6f}")
    print(f"   - 99.9th:  {np.percentile(scores, 99.9):.6f}")
    print(f"   - Max:     {scores.max():.6f}")

    # Save threshold files for each percentile
    print(f"\n5. Saving threshold files...")
    for percentile in percentiles:
        threshold = np.percentile(scores, percentile)

        threshold_data = {
            'version': '3.0',
            'model': 'iForest_v3',
            'percentile': percentile,
            'global_threshold': float(threshold),
            'thresholds': {},  # Can add per-context later if needed
            'calibration_info': {
                'sample_size': len(scores),
                'score_mean': float(scores.mean()),
                'score_std': float(scores.std()),
                'score_min': float(scores.min()),
                'score_max': float(scores.max())
            }
        }

        output_path = f'models/v3_thresholds_p{int(percentile*10):04d}.json'
        with open(output_path, 'w') as f:
            json.dump(threshold_data, f, indent=2)

        print(f"   ✓ {percentile}th percentile: {threshold:.6f} → {output_path}")

    print(f"\n{'='*70}")
    print("✅ CALIBRATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nThreshold files created:")
    for percentile in percentiles:
        threshold = np.percentile(scores, percentile)
        print(f"   {percentile}th: {threshold:.6f} (models/v3_thresholds_p{int(percentile*10):04d}.json)")

    print(f"\nNext step: Test each threshold with validation script")
    print(f"   python scripts/validate_model_synthetic.py \\")
    print(f"     --model models/iforest_model_v3.pkl \\")
    print(f"     --thresholds models/v3_thresholds_pXXXX.json \\")
    print(f"     --subset 10000")

    return scores


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=100000)
    parser.add_argument('--percentiles', type=float, nargs='+', default=[99.0, 99.5, 99.9])
    args = parser.parse_args()

    calibrate_v3_thresholds(
        percentiles=args.percentiles,
        sample_size=args.sample_size
    )
