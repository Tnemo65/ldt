#!/usr/bin/env python3
"""
Train OneClassSVM on clean baseline as alternative to HalfSpaceTrees.
OneClassSVM learns the boundary of normal behavior.

Usage:
  python src/ml/train_ocsvm.py
  python src/ml/train_ocsvm.py --data data/clean/jan_2024_clean_baseline.parquet
"""

import argparse
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from river.anomaly import OneClassSVM

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.vectorizer import FeatureVectorizer


def train_ocsvm(
    data_path: str,
    output_dir: str = 'models',
    scaler_path: str = 'models/scaler.pkl',
    model_name: str = 'ocsvm_model.pkl',
    nu: float = 0.1
):
    """Train OneClassSVM on clean baseline (Jan 2024 ONLY).

    CRITICAL: Train ONLY on clean baseline to learn normal behavior.
    Do NOT include anomalies in training data.

    Args:
        data_path: Path to clean baseline parquet file
        output_dir: Directory to save trained model
        scaler_path: Path to fitted StandardScaler
        model_name: Output model filename (default: ocsvm_model.pkl)
        nu: Upper bound on fraction of outliers (default: 0.1)

    Returns:
        Trained model instance
    """
    print("="*60)
    print("OneClassSVM Training (River)")
    print("="*60)

    # 1. Load data
    print(f"\n1. Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"   ✓ Loaded {len(df):,} records")

    # 2. Batch vectorize features
    print(f"\n2. Extracting 21D feature vectors...")
    vectorizer = FeatureVectorizer()
    X = vectorizer.transform_batch(df)
    print(f"   Extracted {X.shape[0]:,} x {X.shape[1]}D feature vectors")

    # 3. Load fitted scaler
    print(f"\n3. Loading fitted StandardScaler from: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("   ✓ Scaler loaded")

    # 4. Scale features
    print(f"\n4. Scaling features...")
    X_scaled = scaler.transform(X)
    print(f"   ✓ Features scaled (mean≈0, std≈1)")

    # 5. Train OneClassSVM
    print(f"\n5. Training OneClassSVM...")
    print(f"   Config:")
    print(f"   - Algorithm: One-Class SVM")
    print(f"   - nu (outlier fraction): {nu}")

    model = OneClassSVM(nu=nu)

    # Stream learning (one record at a time)
    for i, features in enumerate(X_scaled):
        # Convert numpy array to dict (River expects dict input)
        feature_dict = {idx: float(val) for idx, val in enumerate(features)}
        model.learn_one(feature_dict)

        # Progress
        if (i + 1) % 100000 == 0:
            print(f"   Trained: {i + 1:,} / {len(X):,} ({(i+1)/len(X)*100:.1f}%)")

    print(f"   ✓ Training complete: {len(X):,} records")

    # 6. Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / model_name
    print(f"\n6. Saving model to: {model_file}")

    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    print(f"   ✓ Model saved ({model_file.stat().st_size / 1e6:.1f} MB)")

    # 7. Quick validation
    print(f"\n7. Quick validation...")
    test_sample = X_scaled[:10]
    scores = []
    for features in test_sample:
        feature_dict = {idx: float(val) for idx, val in enumerate(features)}
        score = model.score_one(feature_dict)
        scores.append(score)
    print(f"   Sample scores: {[f'{s:.3f}' for s in scores]}")
    print(f"   Mean score: {np.mean(scores):.3f}")
    print(f"   Std score: {np.std(scores):.3f}")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"   Model: {model_file}")
    print(f"   Records: {X.shape[0]:,}")
    print(f"   Features: 21D")
    print(f"   Algorithm: OneClassSVM (nu={nu})")
    print(f"{'='*60}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train OneClassSVM on clean baseline')
    parser.add_argument(
        '--data',
        type=str,
        default='data/clean/jan_2024_clean_baseline.parquet',
        help='Path to clean baseline parquet file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Output directory for model (default: models)'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default='models/scaler.pkl',
        help='Path to fitted StandardScaler (default: models/scaler.pkl)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='ocsvm_model.pkl',
        help='Output model filename (default: ocsvm_model.pkl)'
    )
    parser.add_argument(
        '--nu',
        type=float,
        default=0.1,
        help='Upper bound on fraction of outliers (default: 0.1)'
    )

    args = parser.parse_args()

    # Validate inputs
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data file not found: {data_path}")
        return 1

    scaler_path = Path(args.scaler)
    if not scaler_path.exists():
        print(f"❌ Error: Scaler not found: {scaler_path}")
        print("   Hint: Run Phase 0 tasks to fit StandardScaler")
        return 1

    # Train
    try:
        train_ocsvm(
            data_path=str(data_path),
            output_dir=args.output,
            scaler_path=str(scaler_path),
            model_name=args.model_name,
            nu=args.nu
        )
        return 0
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
