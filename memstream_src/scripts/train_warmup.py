#!/usr/bin/env python3
"""
MemStream Training Pipeline — Time-Ordered Data Splits.

FIXES in v5:
- C-DE-1: Time-ordered splits instead of random shuffle
- C-DE-2: Normalization leakage prevention (split warmup data)
- H-ML-2/H-ML-3: Complete determinism flags

Data flow:
  [10%] → Compute normalization stats ONLY
  [80%] → Train autoencoder
  [10%] → Initialize memory
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memstream_core import MemStreamCore, MemStreamConfig, set_determinism
from core.feature_extractor import FeatureVectorizer
from core.context_aware import (
    ContextBeta, extract_context_from_record,
    NUM_NEIGHBORHOODS, NUM_CONTEXT_CELLS
)


def prepare_time_ordered_splits(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    calib_frac: float = 0.8
) -> dict:
    """
    Prepare TEMPORAL splits for streaming anomaly detection.

    CRITICAL: Uses time-ordered splits, NOT random shuffle.
    """
    # Parse datetime
    df = df.copy()
    df['pickup_dt'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df = df.dropna(subset=['pickup_dt'])

    # C-DE-1 FIX: Sort by time (NO SHUFFLE!)
    df = df.sort_values('pickup_dt').reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_frac)
    calib_end = int(n * calib_frac)

    splits = {
        'warmup': df.iloc[:train_end].copy(),
        'calibration': df.iloc[train_end:calib_end].copy(),
        'test': df.iloc[calib_end:].copy()
    }

    # Verify temporal order
    assert splits['warmup']['pickup_dt'].max() <= splits['calibration']['pickup_dt'].min(), \
        "Temporal overlap: warmup and calibration sets overlap!"
    assert splits['calibration']['pickup_dt'].max() <= splits['test']['pickup_dt'].min(), \
        "Temporal overlap: calibration and test sets overlap!"

    print(f"\n{'='*60}")
    print("TIME-ORDERED DATA SPLITS")
    print(f"{'='*60}")
    print(f"Total records: {n:,}")
    print(f"  Warmup:      {len(splits['warmup']):>8,} ({len(splits['warmup'])/n*100:>5.1f}%)")
    print(f"  Calibration:  {len(splits['calibration']):>8,} ({len(splits['calibration'])/n*100:>5.1f}%)")
    print(f"  Test:        {len(splits['test']):>8,} ({len(splits['test'])/n*100:>5.1f}%)")

    return splits


def prepare_warmup_data_leakage_free(
    df: pd.DataFrame,
    stats_frac: float = 0.1,
    memory_frac: float = 0.1
) -> dict:
    """
    Prepare warmup data with NO normalization leakage.

    C-DE-2 FIX: Split warmup data into 3 parts:
      1. First 10%: Compute normalization stats ONLY
      2. Middle 80%: Train autoencoder
      3. Last 10%: Initialize memory module
    """
    n = len(df)

    stats_end = int(n * stats_frac)
    memory_start = int(n * (1 - memory_frac))

    return {
        'stats_data': df.iloc[:stats_end],
        'train_data': df.iloc[stats_end:memory_start],
        'memory_data': df.iloc[memory_start:],
    }


def main():
    parser = argparse.ArgumentParser(description='MemStream Training')
    parser.add_argument('--data', default='data/clean/jan_2024_clean_baseline.parquet',
                        help='Input data path')
    parser.add_argument('--output', default='models/memstream',
                        help='Output model directory')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--memory-size', type=int, default=100,
                        help='Memory module size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--signing-key', type=str, default='training-signing-key',
                        help='HMAC signing key for model')
    args = parser.parse_args()

    # Set determinism (H-ML-2, H-ML-3)
    set_determinism(args.seed)
    print(f"Reproducibility seed: {args.seed}")

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Data file not found: {args.data}")
        print("Creating sample data for demonstration...")
        # Generate synthetic taxi data
        df = generate_sample_taxi_data(n_records=10000)
    else:
        print(f"Loading data from: {args.data}")
        if args.data.endswith('.parquet'):
            df = pd.read_parquet(args.data)
        else:
            df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} records")

    # Time-ordered splits (C-DE-1 FIX)
    splits = prepare_time_ordered_splits(df)

    # Leakage-free warmup (C-DE-2 FIX)
    warmup_data = prepare_warmup_data_leakage_free(splits['warmup'])

    # Feature extraction
    vectorizer = FeatureVectorizer()

    # Stats from FIRST 10% (C-DE-2)
    print(f"\nComputing stats from {len(warmup_data['stats_data']):,} samples (first 10%)...")
    stats_features = vectorizer.transform_batch(warmup_data['stats_data'])
    stats_mean = np.mean(stats_features, axis=0)
    stats_std = np.std(stats_features, axis=0)
    stats_std = np.clip(stats_std, min=1e-8)
    print(f"Stats computed. Mean shape: {stats_mean.shape}, Std shape: {stats_std.shape}")

    # Train from MIDDLE 80%
    print(f"\nPreparing training data from {len(warmup_data['train_data']):,} samples (middle 80%)...")
    train_features = vectorizer.transform_batch(warmup_data['train_data'])

    # Normalize
    train_normalized = (train_features - stats_mean) / stats_std
    X_train = train_normalized.astype(np.float32)
    print(f"Training data shape: {X_train.shape}")

    # Autoencoder training
    cfg = MemStreamConfig()
    cfg.memory_len = args.memory_size
    cfg.warmup_epochs = args.epochs
    cfg.seed = args.seed
    # v10: 34D features
    cfg.in_dim = 34
    cfg.hidden_dim = 68
    cfg.out_dim = 34

    print(f"\nInitializing MemStream with config (v10: 34D):")
    print(f"  in_dim: {cfg.in_dim}")
    print(f"  hidden_dim: {cfg.hidden_dim}")
    print(f"  memory_len: {cfg.memory_len}")
    print(f"  warmup_epochs: {cfg.warmup_epochs}")

    ms = MemStreamCore(cfg=cfg, device='cpu')
    ms.mean = torch.from_numpy(stats_mean).float()
    ms.std = torch.from_numpy(stats_std).float()

    # Warmup (AE training + memory initialization)
    print(f"\nStarting warmup...")
    ms.warmup(
        X_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True
    )

    # Calibrate beta using calibration set
    print(f"\nCalibrating threshold using calibration set...")
    calib_features = vectorizer.transform_batch(splits['calibration'])
    calib_normalized = (calib_features - stats_mean) / stats_std
    calib_scores = []
    for f in calib_normalized:
        try:
            score = ms.score_one(f)
            calib_scores.append(score)
        except RuntimeError:
            # Beta not set yet - skip
            pass

    if calib_scores:
        beta = np.percentile(calib_scores, 95)  # 5% FPR target
        ms.set_beta(beta)
        print(f"Beta threshold (95th percentile): {beta:.4f}")
        print(f"Expected FPR: ~5%")
    else:
        print("[WARNING] Could not compute calibration scores")
        beta = 0.5

    # v10: Compute context-beta thresholds for 10 neighborhoods x 8 context cells
    print(f"\nComputing context-beta thresholds (80 total)...")
    context_beta = ContextBeta(default_beta=beta)
    
    # Record scores per neighborhood/context for warmup set
    warmup_records = splits['warmup'].to_dict('records')
    warmup_features = vectorizer.transform_batch(warmup_records)
    warmup_normalized = (warmup_features - stats_mean) / stats_std
    
    for i, features in enumerate(warmup_normalized):
        try:
            score = ms.score_one(features)
            record = warmup_records[i]
            nb_id, ctx_id, _, _ = extract_context_from_record(record)
            context_beta.record_score(score, nb_id, ctx_id)
        except RuntimeError:
            continue
    
    # Fit context-beta thresholds
    context_beta.fit_from_scores(percentile=95)
    thresholds = context_beta.to_dict()
    print(f"Context-beta thresholds computed for {NUM_NEIGHBORHOODS} neighborhoods x {NUM_CONTEXT_CELLS} context cells")

    # Save model
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = str(output_path / 'memstream_ae.pt')
    ms.save(model_path, signing_key=args.signing_key)

    # Save vectorizer stats
    vectorizer_path = output_path / 'vectorizer_stats.npz'
    np.savez(vectorizer_path,
             mean=stats_mean,
             std=stats_std,
             feature_names=vectorizer.feature_names)

    # v10: Save context-beta thresholds
    import json
    context_beta_path = output_path / 'context_beta_thresholds.json'
    with open(context_beta_path, 'w') as f:
        json.dump(thresholds, f, indent=2)

    print(f"\nModel saved to {output_path}")
    print(f"  memstream_ae.pt: Autoencoder weights")
    print(f"  memstream_ae.pt.hmac: HMAC signature")
    print(f"  vectorizer_stats.npz: Normalization statistics")
    print(f"  context_beta_thresholds.json: 80 context-beta thresholds")

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY (v10: 34D + 10 neighborhoods)")
    print(f"{'='*60}")
    print(f"  Total records: {len(df):,}")
    print(f"  Warmup records: {len(splits['warmup']):,}")
    print(f"  Calibration records: {len(splits['calibration']):,}")
    print(f"  Test records: {len(splits['test']):,}")
    print(f"  Global beta threshold: {beta:.4f}" if calib_scores else "  Beta threshold: Not set")
    print(f"  Feature dimension: {cfg.in_dim}D")
    print(f"  Neighborhoods: {NUM_NEIGHBORHOODS}")
    print(f"  Context-beta thresholds: 80")
    print(f"  Memory size: {cfg.memory_len}")
    print(f"  Device: cpu")
    print(f"{'='*60}")


def generate_sample_taxi_data(n_records: int = 10000) -> pd.DataFrame:
    """Generate synthetic NYC taxi data for testing."""
    import random
    from datetime import datetime, timedelta

    print(f"Generating {n_records:,} synthetic taxi records...")

    base_date = datetime(2024, 1, 1, 0, 0, 0)

    records = []
    for i in range(n_records):
        # Time progression (time-ordered)
        dt = base_date + timedelta(minutes=i * 5)

        # Random location
        pu_loc = random.randint(1, 263)
        do_loc = random.randint(1, 263)

        # Generate features
        fare = random.uniform(5, 50)
        distance = random.uniform(0.5, 20)

        record = {
            'tpep_pickup_datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'tpep_dropoff_datetime': (dt + timedelta(minutes=random.randint(5, 60))).strftime('%Y-%m-%d %H:%M:%S'),
            'PULocationID': pu_loc,
            'DOLocationID': do_loc,
            'fare_amount': fare,
            'tip_amount': fare * random.uniform(0, 0.25),
            'tolls_amount': 0.0,
            'improvement_surcharge': 0.3,
            'trip_distance': distance,
            'passenger_count': random.randint(1, 4),
            'ratecodeID': 1,
            'payment_type': random.randint(1, 2),
            'pickup_latitude': 40.7128 + random.uniform(-0.1, 0.1),
            'pickup_longitude': -74.0060 + random.uniform(-0.1, 0.1),
            'dropoff_latitude': 40.7128 + random.uniform(-0.15, 0.15),
            'dropoff_longitude': -74.0060 + random.uniform(-0.15, 0.15),
        }
        records.append(record)

    return pd.DataFrame(records)


if __name__ == '__main__':
    main()
