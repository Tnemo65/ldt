#!/usr/bin/env python3
"""
Model Training - Offline Training of ML Models
===============================================
Trains anomaly detection models with correct contamination=0.01 (1% natural outliers in clean data).

CRITICAL: contamination=0.01 is NOT the test anomaly rate (0.022).
It represents expected natural outliers in clean training data.

Usage:
    python exp_offline_train.py --data-file ./data/processed/train.csv --output-dir ./models

Models trained:
    1. IsolationForest: contamination=0.01, n_estimators=100
    2. LOF: contamination=0.01, novelty=True, n_neighbors=20
    3. OneClassSVM: nu=0.02, kernel='rbf', with StandardScaler
    4. RRCF: 100 trees, tree_size=256, incremental insertion
"""

import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# For RRCF (Robust Random Cut Forest)
try:
    import rrcf
except ImportError:
    rrcf = None


def train_isolation_forest(df: pd.DataFrame, contamination: float = 0.01, random_state: int = 42) -> IsolationForest:
    """
    Train IsolationForest model with correct contamination parameter.

    CRITICAL: contamination=0.01 means 1% natural outliers in clean data, NOT test anomaly rate.

    Args:
        df: DataFrame with computed features (must have feature columns after column 4)
        contamination: Expected fraction of outliers (0.01 = 1%)
        random_state: Random seed for reproducibility

    Returns:
        Trained IsolationForest model
    """
    # Extract features (skip non-feature columns: window_start, trip_count, hour, dow)
    feature_cols = df.columns[4:]
    X = df[feature_cols].values

    model = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X)
    return model


def train_lof(df: pd.DataFrame, contamination: float = 0.01) -> LocalOutlierFactor:
    """
    Train Local Outlier Factor model with novelty=True.

    CRITICAL: contamination=0.01 means 1% natural outliers in clean data, NOT test anomaly rate.
    novelty=True allows prediction on new unseen data.

    Args:
        df: DataFrame with computed features
        contamination: Expected fraction of outliers (0.01 = 1%)

    Returns:
        Trained LOF model
    """
    # Extract features (skip non-feature columns)
    feature_cols = df.columns[4:]
    X = df[feature_cols].values

    model = LocalOutlierFactor(
        contamination=contamination,
        novelty=True,  # Critical: allows prediction on new data
        n_neighbors=20,
        n_jobs=-1
    )

    model.fit(X)
    return model


def train_ocsvm(df: pd.DataFrame, nu: float = 0.02) -> tuple:
    """
    Train One-Class SVM model with StandardScaler.

    Args:
        df: DataFrame with computed features
        nu: Upper bound on fraction of training errors (0.02 = 2%)

    Returns:
        Tuple of (trained model, StandardScaler)
    """
    # Extract features (skip non-feature columns)
    feature_cols = df.columns[4:]
    X = df[feature_cols].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = OneClassSVM(
        nu=nu,
        kernel='rbf',
        gamma='auto'
    )

    model.fit(X_scaled)
    return model, scaler


def train_rrcf(df: pd.DataFrame, num_trees: int = 100, tree_size: int = 256) -> dict:
    """
    Train Robust Random Cut Forest with incremental insertion.

    RRCF detects anomalies by identifying points with high CoDisp (change of displacement).

    Args:
        df: DataFrame with computed features
        num_trees: Number of trees in forest (100)
        tree_size: Max size of each tree (256)

    Returns:
        dict: {
            'forest': {tree_id: tree},
            'num_trees': num_trees,
            'tree_size': tree_size,
            'feature_cols': list of feature column names
        }
    """
    if rrcf is None:
        raise ImportError("rrcf package not installed. Install with: pip install rrcf")

    # Extract features (skip non-feature columns)
    feature_cols = list(df.columns[4:])
    X = df[feature_cols].values

    # Create forest with num_trees trees
    forest = {}

    for tree_id in range(num_trees):
        # Create new tree
        tree = rrcf.RCTree()

        # Incremental insertion of points from this tree's sample
        # Distribute points across trees in round-robin fashion
        indices = np.arange(len(X))
        # Shuffle indices for randomness
        np.random.seed(42 + tree_id)  # Deterministic seeding
        shuffled = np.random.permutation(indices)

        # Each tree gets roughly 1/num_trees of the data
        points_per_tree = (len(X) + num_trees - 1) // num_trees
        start_idx = tree_id * points_per_tree
        end_idx = min(start_idx + points_per_tree, len(X))

        # Insert points into tree
        for idx in shuffled[start_idx:end_idx]:
            # Create point as numpy array and insert
            point = X[idx]
            tree.insert_point(point, index=idx)

        forest[tree_id] = tree

    return {
        'forest': forest,
        'num_trees': num_trees,
        'tree_size': tree_size,
        'feature_cols': feature_cols
    }


def compute_context_stats_for_models(df: pd.DataFrame) -> dict:
    """
    Compute context statistics from training data for statistical models.

    This is the same as compute_context_stats from exp_00_prepare.
    Kept here for convenience in the training pipeline.

    Args:
        df: Training DataFrame with columns: trip_count, hour, dow

    Returns:
        dict: {(dow, hour): {'mean': X, 'std': Y, 'median': Z, 'q25': A, 'q75': B}}
    """
    # Validate required columns
    required_cols = ['trip_count', 'hour', 'dow']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    stats = {}

    for (dow, hour), group in df.groupby(['dow', 'hour']):
        std_val = group['trip_count'].std()
        stats[(dow, hour)] = {
            'mean': group['trip_count'].mean(),
            'std': std_val if pd.notna(std_val) and std_val > 0 else 1.0,
            'median': group['trip_count'].median(),
            'q25': group['trip_count'].quantile(0.25),
            'q75': group['trip_count'].quantile(0.75)
        }

    return stats


def main():
    """
    CLI for training all anomaly detection models.

    Trains 4 models:
    1. IsolationForest (contamination=0.01)
    2. LOF (contamination=0.01, novelty=True)
    3. OneClassSVM (nu=0.02)
    4. RRCF (100 trees, tree_size=256)

    Saves models and context stats as pickle files.
    """
    parser = argparse.ArgumentParser(
        description='Train anomaly detection models with correct contamination=0.01'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help='Path to training data CSV file (with features)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save trained models'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_file}...")
    df = pd.read_parquet(args.data_file)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Train IsolationForest
    print("\n[1/5] Training IsolationForest (contamination=0.01)...")
    iso_forest = train_isolation_forest(df, contamination=0.01, random_state=42)
    iso_forest_path = output_dir / 'isolation_forest.pkl'
    with open(iso_forest_path, 'wb') as f:
        pickle.dump(iso_forest, f)
    print(f"  Saved to {iso_forest_path}")

    # Train LOF
    print("[2/5] Training LOF (contamination=0.01, novelty=True)...")
    lof_model = train_lof(df, contamination=0.01)
    lof_path = output_dir / 'lof.pkl'
    with open(lof_path, 'wb') as f:
        pickle.dump(lof_model, f)
    print(f"  Saved to {lof_path}")

    # Train OneClassSVM
    print("[3/5] Training OneClassSVM (nu=0.02)...")
    ocsvm_model, scaler = train_ocsvm(df, nu=0.02)
    ocsvm_path = output_dir / 'ocsvm.pkl'
    with open(ocsvm_path, 'wb') as f:
        pickle.dump((ocsvm_model, scaler), f)
    print(f"  Saved to {ocsvm_path}")

    # Train RRCF
    print("[4/5] Training RRCF (100 trees, tree_size=256)...")
    if rrcf is None:
        print("  WARNING: rrcf package not installed, skipping RRCF training")
        rrcf_model = None
    else:
        rrcf_model = train_rrcf(df, num_trees=100, tree_size=256)
        rrcf_path = output_dir / 'rrcf.pkl'
        # RRCF trees contain lambda functions and non-picklable objects
        # Use dill library which handles these correctly
        try:
            import dill
            # Save the entire model dict with dill
            with open(rrcf_path, 'wb') as f:
                dill.dump(rrcf_model, f)
            print(f"  Saved to {rrcf_path}")
        except Exception as e:
            print(f"  WARNING: Could not serialize RRCF model: {e}")
            rrcf_model = None

    # Compute and save context stats
    print("[5/5] Computing context statistics...")
    context_stats = compute_context_stats_for_models(df)
    stats_path = output_dir / 'context_stats.pkl'
    with open(stats_path, 'wb') as f:
        pickle.dump(context_stats, f)
    print(f"  Saved to {stats_path}")

    print("\nModel training complete!")
    print(f"All models saved to {output_dir}")


if __name__ == '__main__':
    main()
