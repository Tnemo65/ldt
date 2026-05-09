#!/usr/bin/env python3
"""
Train iForestASD (HalfSpaceTrees) on clean baseline.
Spec: Lines 3414-3465 (Cold start training on Jan 2024)

Usage:
  python src/ml/train_iforest.py
  python src/ml/train_iforest.py --data data/clean/jan_2024_clean_baseline.parquet
"""

import argparse
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from river.anomaly import HalfSpaceTrees
import io
import locale

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.vectorizer import FeatureVectorizer


def train_iforest(
    data_path: str,
    output_dir: str = 'models',
    scaler_path: str = 'models/scaler.pkl',
    n_trees: int = 200,
    height: int = 10,
    window_size: int = 512,
    model_name: str = 'iforest_model.pkl',
    sample_size: int = None,
):
    """Train iForestASD on clean baseline (Jan 2024 ONLY).

    CRITICAL: Train ONLY on clean baseline to learn normal behavior.
    Do NOT include anomalies in training data.

    Args:
        data_path: Path to clean baseline parquet file
        output_dir: Directory to save trained model
        scaler_path: Path to fitted StandardScaler
        n_trees: Number of trees (default: 200, prototype-validated)
        height: Tree height (default: 10, prototype-validated)
        window_size: Window size (default: 512, prototype-validated)
        model_name: Output model filename (default: iforest_model.pkl)

    Returns:
        Trained model instance
    """
    print("="*60)
    print("iForestASD Training (River HalfSpaceTrees)")
    print("="*60)

    # 1. Load data
    print(f"\n1. Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"   Loaded {len(df):,} records")

    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        print(f"   Sampled: {len(df):,} records")

    # 2. Vectorize features
    print(f"\n2. Extracting 21D enhanced feature vectors (with ratio features)...")
    vectorizer = FeatureVectorizer()

    X = []
    for idx, row in df.iterrows():
        try:
            features = vectorizer.transform(row.to_dict())
            X.append(features)
        except Exception as e:
            if idx % 100000 == 0:
                print(f"   Warning: Failed to vectorize record {idx}: {e}")

        if (idx + 1) % 500000 == 0:
            print(f"   Vectorized: {idx + 1:,} / {len(df):,}")

    X = np.array(X)
    print(f"   Extracted {X.shape[0]:,} feature vectors (shape: {X.shape})")

    # 3. Load fitted scaler
    print(f"\n3. Loading fitted StandardScaler from: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("   Scaler loaded")

    # 4. Scale features
    print(f"\n4. Scaling features...")
    X_scaled = scaler.transform(X)
    print(f"   Features scaled (mean=0, std=1)")

    # 5. Train iForestASD (River HalfSpaceTrees)
    print(f"\n5. Training iForestASD (HalfSpaceTrees)...")
    print(f"   Config:")
    print(f"   - n_trees: {n_trees}")
    print(f"   - height: {height}")
    print(f"   - window_size: {window_size}")
    print(f"   - seed: 42")

    model = HalfSpaceTrees(
        n_trees=n_trees,
        height=height,
        window_size=window_size,
        seed=42
    )

    # Stream learning (one record at a time)
    for i, features in enumerate(X_scaled):
        # Convert numpy array to dict (River expects dict input)
        feature_dict = {idx: float(val) for idx, val in enumerate(features)}
        model.learn_one(feature_dict)

        # Progress
        if (i + 1) % 100000 == 0:
            print(f"   Trained: {i + 1:,} / {len(X):,} ({(i+1)/len(X)*100:.1f}%)")

    print(f"   Training complete: {len(X):,} records")

    # 6. Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / model_name
    print(f"\n6. Saving model to: {model_file}")

    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    print(f"   Model saved ({model_file.stat().st_size / 1e6:.1f} MB)")

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
    print(f"   Records: {len(X):,}")
    print(f"   Features: 21D (15D base + 6D ratio)")
    print(f"{'='*60}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train iForestASD on clean baseline')
    parser.add_argument(
        '--data',
        type=str,
        default='data/clean/jan_2024_clean_baseline.parquet',
        help='Path to clean baseline parquet file'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample N records for faster training (default: all)'
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
        '--n-trees',
        type=int,
        default=200,
        help='Number of trees (default: 200, prototype-validated)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=10,
        help='Tree height (default: 10, prototype-validated)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=512,
        help='Window size (default: 512, prototype-validated)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='iforest_model.pkl',
        help='Output model filename (default: iforest_model.pkl)'
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return 1

    scaler_path = Path(args.scaler)
    if not scaler_path.exists():
        print(f"Error: Scaler not found: {scaler_path}")
        print("   Hint: Run Phase 0 tasks to fit StandardScaler")
        return 1

    try:
        train_iforest(
            data_path=str(data_path),
            output_dir=args.output,
            scaler_path=str(scaler_path),
            n_trees=args.n_trees,
            height=args.height,
            window_size=args.window_size,
            model_name=args.model_name,
            sample_size=args.sample,
        )
        return 0
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
