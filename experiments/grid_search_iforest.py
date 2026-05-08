#!/usr/bin/env python3
"""
Hyperparameter Grid Search for iForestASD.
Task 2.11-2.15: Test 9 configs (3×3 grid), log to MLflow

Grid:
- n_trees: [50, 100, 200]
- window_size: [128, 256, 512]
- Total: 9 configurations

Evaluation:
- Train on Jan 2024 clean baseline
- Validate on 50K synthetic anomalies
- Metrics: F1, Recall, FPR, Training time

Best config selected by: F1 score with FPR < 5% constraint
"""

import argparse
import sys
from pathlib import Path
import pickle
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.vectorizer import FeatureVectorizer

# Try importing MLflow (optional)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠ MLflow not available - results will only be saved locally")

from river.anomaly import HalfSpaceTrees


def train_model(
    data_path: str,
    scaler_path: str,
    n_trees: int,
    height: int,
    window_size: int
):
    """Train iForestASD with given hyperparameters.

    Args:
        data_path: Path to training data
        scaler_path: Path to fitted scaler
        n_trees: Number of trees
        height: Tree height
        window_size: Window size

    Returns:
        Trained model and training time
    """
    print(f"\n  Training: n_trees={n_trees}, height={height}, window={window_size}")

    # Load data
    df = pd.read_parquet(data_path)
    print(f"    Loaded {len(df):,} records")

    # Vectorize
    vectorizer = FeatureVectorizer()
    X = []

    for idx, row in df.iterrows():
        try:
            features = vectorizer.transform(row.to_dict())
            X.append(features)
        except Exception:
            pass

        if (idx + 1) % 500000 == 0:
            print(f"    Vectorized: {idx + 1:,} / {len(df):,}")

    X = np.array(X)
    print(f"    Features: {X.shape}")

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X)

    # Train model
    print(f"    Training...")
    start_time = time.time()

    model = HalfSpaceTrees(
        n_trees=n_trees,
        height=height,
        window_size=window_size,
        seed=42
    )

    for i, features in enumerate(X_scaled):
        feature_dict = {idx: float(val) for idx, val in enumerate(features)}
        model.learn_one(feature_dict)

        if (i + 1) % 500000 == 0:
            print(f"    Trained: {i + 1:,} / {len(X):,}")

    training_time = time.time() - start_time
    print(f"    ✓ Training complete: {training_time:.1f}s")

    return model, training_time


def validate_model(
    model,
    scaler,
    data_path: str,
    labels_path: str,
    thresholds_path: str
):
    """Validate model on synthetic anomalies.

    Args:
        model: Trained iForestASD model
        scaler: Fitted StandardScaler
        data_path: Path to validation data with anomalies
        labels_path: Path to ground truth labels
        thresholds_path: Path to context thresholds

    Returns:
        Metrics dict (recall, fpr, precision, f1)
    """
    print(f"\n  Validating...")

    # Load validation data
    df = pd.read_parquet(data_path)
    labels_df = pd.read_csv(labels_path)

    print(f"    Loaded {len(df):,} records")

    # Load thresholds
    with open(thresholds_path) as f:
        thresholds = json.load(f)

    # Score all records
    vectorizer = FeatureVectorizer()
    y_pred = []

    for idx, row in df.iterrows():
        try:
            features = vectorizer.transform(row.to_dict())
            features_scaled = scaler.transform([features])[0]
            feature_dict = {i: float(val) for i, val in enumerate(features_scaled)}

            score = model.score_one(feature_dict)

            # Get threshold (simplified - use global for now)
            threshold = thresholds.get('global_threshold', 0.5)

            is_anomaly = score > threshold
            y_pred.append(1 if is_anomaly else 0)

        except Exception:
            y_pred.append(0)

        if (idx + 1) % 10000 == 0:
            print(f"    Scored: {idx + 1:,} / {len(df):,}")

    # Calculate metrics
    y_true = labels_df['is_anomaly'].values
    y_pred = np.array(y_pred)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'recall': recall,
        'fpr': fpr,
        'precision': precision,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

    print(f"    ✓ Recall: {recall:.3f}, FPR: {fpr:.3f}, F1: {f1:.3f}")

    return metrics


def run_grid_search(
    train_data_path: str = 'data/clean/jan_2024_clean_baseline.parquet',
    val_data_path: str = 'data/clean/jan_2024_with_50k_anomalies.parquet',
    labels_path: str = 'data/clean/anomaly_labels.csv',
    scaler_path: str = 'models/scaler.pkl',
    thresholds_path: str = 'models/context_thresholds_v2.json',
    output_dir: str = 'experiments/grid_search_results'
):
    """Run hyperparameter grid search.

    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data
        labels_path: Path to ground truth labels
        scaler_path: Path to fitted scaler
        thresholds_path: Path to context thresholds
        output_dir: Directory to save results

    Returns:
        Best configuration and all results
    """
    print("="*60)
    print("iForestASD Hyperparameter Grid Search")
    print("="*60)

    # Define grid
    param_grid = {
        'n_trees': [50, 100, 200],
        'height': [10],  # Keep fixed at prototype value
        'window_size': [128, 256, 512]
    }

    grid = list(ParameterGrid(param_grid))
    print(f"\nGrid size: {len(grid)} configurations")
    print(f"Parameters:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")

    # Load scaler once
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Run grid search
    results = []

    for i, params in enumerate(grid):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(grid)}")
        print(f"{'='*60}")
        print(f"Params: {params}")

        try:
            # Train model
            model, training_time = train_model(
                train_data_path,
                scaler_path,
                n_trees=params['n_trees'],
                height=params['height'],
                window_size=params['window_size']
            )

            # Validate model
            metrics = validate_model(
                model,
                scaler,
                val_data_path,
                labels_path,
                thresholds_path
            )

            # Combine results
            result = {
                **params,
                **metrics,
                'training_time': training_time,
                'timestamp': datetime.utcnow().isoformat()
            }

            results.append(result)

            # Log to MLflow if available
            if MLFLOW_AVAILABLE:
                with mlflow.start_run(run_name=f"grid_search_{i+1}"):
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    mlflow.log_metric('training_time', training_time)

            print(f"\n  ✅ Config {i+1} complete")

        except Exception as e:
            print(f"\n  ❌ Config {i+1} failed: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_path = Path(output_dir) / f'grid_search_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results_df.to_csv(results_path, index=False)

    print(f"\n{'='*60}")
    print(f"Grid Search Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved: {results_path}")

    # Find best config (highest F1 with FPR < 5%)
    valid_results = results_df[results_df['fpr'] < 0.05]

    if len(valid_results) > 0:
        best_idx = valid_results['f1'].idxmax()
        best_config = valid_results.loc[best_idx]

        print(f"\n✅ Best Configuration (F1={best_config['f1']:.3f}, FPR={best_config['fpr']:.3f}):")
        print(f"  n_trees: {int(best_config['n_trees'])}")
        print(f"  height: {int(best_config['height'])}")
        print(f"  window_size: {int(best_config['window_size'])}")
        print(f"  Recall: {best_config['recall']:.3f}")
        print(f"  Precision: {best_config['precision']:.3f}")
        print(f"  Training time: {best_config['training_time']:.1f}s")
    else:
        print(f"\n⚠ No configuration achieved FPR < 5%")
        best_config = results_df.loc[results_df['f1'].idxmax()]
        print(f"\nBest by F1 alone: {best_config.to_dict()}")

    return best_config, results_df


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter grid search for iForestASD')
    parser.add_argument('--train-data', type=str, default='data/clean/jan_2024_clean_baseline.parquet')
    parser.add_argument('--val-data', type=str, default='data/clean/jan_2024_with_50k_anomalies.parquet')
    parser.add_argument('--labels', type=str, default='data/clean/anomaly_labels.csv')
    parser.add_argument('--scaler', type=str, default='models/scaler.pkl')
    parser.add_argument('--thresholds', type=str, default='models/context_thresholds_v2.json')
    parser.add_argument('--output-dir', type=str, default='experiments/grid_search_results')

    args = parser.parse_args()

    # Run grid search
    best_config, results = run_grid_search(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        labels_path=args.labels,
        scaler_path=args.scaler,
        thresholds_path=args.thresholds,
        output_dir=args.output_dir
    )

    return 0


if __name__ == '__main__':
    exit(main())
