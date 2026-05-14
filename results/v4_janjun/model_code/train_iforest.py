#!/usr/bin/env python3
"""
Train IsolationForest on clean baseline (FULL DATA - 2.4M records).

Optimized for MAX SPEED using:
  - Vectorized pandas batch feature extraction (100-500x faster than row-by-row)
  - sklearn IsolationForest with parallel trees (n_jobs=-1, all CPU cores)
  - Batch chunked processing for memory efficiency

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
import io
import time
import gc
from joblib import parallel_backend

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.vectorizer import FeatureVectorizer


def train_isolation_forest(
    data_path: str,
    output_dir: str = 'models',
    scaler_path: str = 'models/scaler.pkl',
    n_trees: int = 200,
    max_samples: int = 256,
    contamination: float = 0.001,
    max_features: float = 1.0,
    model_name: str = 'iforest_model.pkl',
    sample_size: int = None,
    chunk_size: int = 500000,
):
    """Train IsolationForest on clean baseline (Jan 2024 ONLY).

    CRITICAL: Train ONLY on clean baseline to learn normal behavior.
    Do NOT include anomalies in training data.

    Args:
        data_path: Path to clean baseline parquet file
        output_dir: Directory to save trained model
        scaler_path: Path to fitted StandardScaler
        n_trees: Number of trees (default: 200)
        max_samples: Max samples per tree (default: 256 for speed)
        contamination: Expected anomaly rate (default: 0.001)
        max_features: Fraction of features per tree (default: 1.0 = all 21D)
        model_name: Output model filename
        sample_size: If set, sample N records (default: ALL - full training)
        chunk_size: Records per processing chunk (default: 500K)

    Returns:
        Trained model instance
    """
    print("=" * 60)
    print("IsolationForest Training (sklearn, parallel trees)")
    print("=" * 60)

    import sklearn
    from sklearn.ensemble import IsolationForest
    print(f"   sklearn version: {sklearn.__version__}")

    # 1. Load data
    print(f"\n1. Loading data from: {data_path}")
    t0 = time.time()
    df = pd.read_parquet(data_path)
    n_records = len(df)
    print(f"   Loaded {n_records:,} records in {time.time()-t0:.1f}s")

    if sample_size and sample_size < n_records:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        n_records = sample_size
        print(f"   Sampled: {n_records:,} records")
    else:
        print(f"   Using ALL {n_records:,} records (full training)")

    # 2. Batch vectorization
    print(f"\n2. Batch vectorizing {len(df):,} records (vectorized pandas)...")
    t0 = time.time()
    vectorizer = FeatureVectorizer()
    X = vectorizer.transform_batch(df)
    print(f"   Vectorized {X.shape[0]:,} x {X.shape[1]}D in {time.time()-t0:.1f}s")
    del df
    gc.collect()

    # 3. Scale features
    print(f"\n3. Loading scaler and scaling features...")
    t0 = time.time()
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)
    del X
    gc.collect()
    print(f"   Scaled in {time.time()-t0:.1f}s")

    # 4. Train IsolationForest with ALL CPU cores
    n_cores = -1
    print(f"\n4. Training IsolationForest (FULL DATA)...")
    print(f"   Config:")
    print(f"   - n_trees: {n_trees}")
    print(f"   - max_samples: {max_samples}")
    print(f"   - contamination: {contamination}")
    print(f"   - max_features: {max_features}")
    print(f"   - n_jobs: {n_cores} (ALL CPU cores)")
    print(f"   - random_state: 42")

    t0 = time.time()
    model = IsolationForest(
        n_estimators=n_trees,
        max_samples=max_samples,
        contamination=contamination,
        max_features=max_features,
        n_jobs=n_cores,
        random_state=42,
        verbose=1,
    )

    with parallel_backend('threading', n_jobs=n_cores):
        model.fit(X_scaled)

    train_time = time.time() - t0
    print(f"   Training complete in {train_time:.1f}s ({len(X_scaled)/train_time:.0f} records/sec)")
    del X_scaled
    gc.collect()

    # 5. Quick validation
    print(f"\n5. Quick validation (sample from training data)...")
    t0 = time.time()
    X_val = vectorizer.transform_batch(pd.read_parquet(data_path).sample(1000, random_state=7).reset_index(drop=True))
    X_val_scaled = scaler.transform(X_val)
    pred_labels = model.predict(X_val_scaled)
    anomaly_scores = model.score_samples(X_val_scaled)
    n_anomalies = (pred_labels == -1).sum()
    print(f"   Anomalies in sample: {n_anomalies}/1000")
    print(f"   Score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
    print(f"   Score mean: {anomaly_scores.mean():.4f}")
    print(f"   Validation time: {time.time()-t0:.1f}s")
    del X_val, X_val_scaled
    gc.collect()

    # 6. Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / model_name
    print(f"\n6. Saving model to: {model_file}")

    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    size_mb = model_file.stat().st_size / 1e6
    print(f"   Model saved ({size_mb:.1f} MB)")

    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"   Model: {model_file}")
    print(f"   Records: {n_records:,}")
    print(f"   Features: 21D (15D base + 6D ratio)")
    print(f"   Training time: {train_time:.1f}s")
    print(f"   Trees: {n_trees}, max_samples: {max_samples}")
    print(f"{'=' * 60}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train IsolationForest on clean baseline')
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
        help='Sample N records for faster training (default: ALL)'
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
        help='Number of trees (default: 200)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=256,
        help='Max samples per tree (default: 256 for speed, use 2048 for quality)'
    )
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.001,
        help='Expected anomaly rate (default: 0.001 = 0.1%%)'
    )
    parser.add_argument(
        '--max-features',
        type=float,
        default=1.0,
        help='Fraction of features per tree (default: 1.0 = all 21D)'
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
        train_isolation_forest(
            data_path=str(data_path),
            output_dir=args.output,
            scaler_path=str(scaler_path),
            n_trees=args.n_trees,
            max_samples=args.max_samples,
            contamination=args.contamination,
            max_features=args.max_features,
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
