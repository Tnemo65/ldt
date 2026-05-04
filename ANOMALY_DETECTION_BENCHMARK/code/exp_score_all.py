#!/usr/bin/env python3
"""
Score All Models on Test Data
==============================
Scores all 19 trained models plus 5 computed models (statistical, window-based)
on 2025 test data and saves predictions for evaluation.

Models scored:
- PyOD (8): HBOS, KNN×2, MCD, PCA, COPOD, ABOD, ECOD
- Deep Learning (4): Autoencoder×2, VAE×2
- sklearn ML (5): IsolationForest×2, LOF×2, OneClassSVM
- RRCF (2): RRCF×2
- Statistical (3): Z-Score, IQR, MAD
- Window-based (2): EWMA, CUSUM

Total: 24 scores

Usage:
    python exp_score_all.py --test-file /path/to/test_2025.parquet \\
                            --models-dir /path/to/models \\
                            --output-dir /path/to/output
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

# Import custom VAE Sampling layer for loading VAE models
from exp_ml_autoencoder import Sampling


def score_pyod_model(model, X_test: pd.DataFrame) -> np.ndarray:
    """Score PyOD model using predict_proba().

    Returns outlier probability (higher = more anomalous).
    """
    return model.predict_proba(X_test.values)[:, 1]


def score_deep_learning_model(model: keras.Model, scaler, X_test: pd.DataFrame) -> np.ndarray:
    """Score deep learning model (Autoencoder/VAE) using reconstruction error.

    Returns MSE reconstruction error (higher = more anomalous).
    """
    X_scaled = scaler.transform(X_test.values)
    reconstructions = model.predict(X_scaled, verbose=0)
    mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
    return mse


def score_sklearn_model(model, X_test: pd.DataFrame, scaler=None) -> np.ndarray:
    """Score sklearn model using score_samples().

    Returns negated anomaly score (higher = more anomalous).
    """
    if scaler is not None:
        X_scaled = scaler.transform(X_test.values)
        return -model.score_samples(X_scaled)
    else:
        return -model.score_samples(X_test.values)


def score_rrcf_model(rrcf_model: dict, X_test: pd.DataFrame) -> np.ndarray:
    """Score RRCF model using CoDisp (Collusive Displacement).

    Returns average CoDisp across all trees (higher = more anomalous).
    """
    forest = rrcf_model['forest']
    num_trees = rrcf_model['num_trees']

    scores = np.zeros(len(X_test))

    for i, point in enumerate(X_test.values):
        codisp_sum = 0.0
        for tree_id, tree in forest.items():
            # Insert point temporarily and compute CoDisp
            tree.insert_point(point, index=f'test_{i}')
            codisp = tree.codisp(f'test_{i}')
            codisp_sum += codisp
            # Remove point
            tree.forget_point(f'test_{i}')

        scores[i] = codisp_sum / num_trees

    return scores


def main():
    parser = argparse.ArgumentParser(
        description='Score all 19 trained models + 5 computed models on test data'
    )
    parser.add_argument('--test-file', type=Path, required=True,
                       help='Path to test data parquet file')
    parser.add_argument('--models-dir', type=Path, required=True,
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Directory to save predictions')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("SCORING ALL MODELS ON TEST DATA")
    print(f"{'='*80}")

    # Load test data
    print(f"\nLoading test data from: {args.test_file}")
    df_test = pd.read_parquet(args.test_file)
    print(f"Loaded {len(df_test):,} test samples")

    # Count anomalies
    n_anomalies = df_test['is_anomaly'].sum()
    print(f"Anomalies: {n_anomalies} ({n_anomalies/len(df_test)*100:.2f}%)")

    # Feature columns (same as training)
    feature_cols = [
        'ctx_mean', 'ctx_std', 'ctx_median', 'ctx_q25', 'ctx_q75',
        'ctx_dev', 'ctx_abs_dev',
        'lag_48', 'lag_144', 'lag_336',
        'roll_mean_48', 'roll_std_48', 'roll_mean_336',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]
    print(f"Features: {len(feature_cols)}")

    X_test = df_test[feature_cols].fillna(0)

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / 'predictions'
    predictions_dir.mkdir(exist_ok=True)

    # =========================================================================
    # PyOD MODELS (8)
    # =========================================================================

    print(f"\n{'='*80}")
    print("SCORING PyOD MODELS (8 models)")
    print(f"{'='*80}\n")

    pyod_models = [
        ('hbos_bins10', 'HBOS (n_bins=10)'),
        ('knn_k5', 'KNN (k=5)'),
        ('knn_k20', 'KNN (k=20)'),
        ('mcd_auto', 'MCD (auto)'),
        ('pca_n3', 'PCA (n_components=3)'),
        ('copod', 'COPOD'),
        ('abod_fast', 'ABOD (fast)'),
        ('ecod', 'ECOD'),
    ]

    for model_name, model_desc in pyod_models:
        model_path = args.models_dir / f'{model_name}.pkl'
        if model_path.exists():
            print(f"Scoring {model_desc}...")
            start = time.time()
            model = joblib.load(model_path)
            scores = score_pyod_model(model, X_test)
            elapsed = time.time() - start

            df_test[f'{model_name}_score'] = scores

            # Save predictions
            df_pred = df_test[['window_start', 'is_anomaly', 'anomaly_type', f'{model_name}_score']].copy()
            df_pred.columns = ['window_start', 'is_anomaly', 'anomaly_type', 'anomaly_score']
            df_pred['window_id'] = df_pred.index
            pred_path = predictions_dir / f'{model_name}_predictions.parquet'
            df_pred.to_parquet(pred_path, index=False)

            throughput = len(X_test) / elapsed
            print(f"  ✓ Scored {len(X_test):,} samples in {elapsed:.2f}s ({throughput:.0f} windows/s)\n")
        else:
            print(f"  ⚠️ WARNING: {model_path} not found, skipping\n")

    # =========================================================================
    # DEEP LEARNING MODELS (4)
    # =========================================================================

    print(f"\n{'='*80}")
    print("SCORING DEEP LEARNING MODELS (4 models)")
    print(f"{'='*80}\n")

    dl_models = [
        ('autoencoder_dim8', 'Autoencoder (dim=8)'),
        ('autoencoder_dim16', 'Autoencoder (dim=16)'),
        ('vae_dim8', 'VAE (dim=8)'),
        ('vae_dim16', 'VAE (dim=16)'),
    ]

    for model_name, model_desc in dl_models:
        model_path = args.models_dir / f'{model_name}.keras'
        scaler_path = args.models_dir / f'{model_name}_scaler.pkl'

        if model_path.exists() and scaler_path.exists():
            print(f"Scoring {model_desc}...")
            start = time.time()
            model = keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
            scores = score_deep_learning_model(model, scaler, X_test)
            elapsed = time.time() - start

            df_test[f'{model_name}_score'] = scores

            # Save predictions
            df_pred = df_test[['window_start', 'is_anomaly', 'anomaly_type', f'{model_name}_score']].copy()
            df_pred.columns = ['window_start', 'is_anomaly', 'anomaly_type', 'anomaly_score']
            df_pred['window_id'] = df_pred.index
            pred_path = predictions_dir / f'{model_name}_predictions.parquet'
            df_pred.to_parquet(pred_path, index=False)

            throughput = len(X_test) / elapsed
            print(f"  ✓ Scored {len(X_test):,} samples in {elapsed:.2f}s ({throughput:.0f} windows/s)\n")
        else:
            print(f"  ⚠️ WARNING: {model_path} or {scaler_path} not found, skipping\n")

    # =========================================================================
    # SKLEARN ML MODELS (5)
    # =========================================================================

    print(f"\n{'='*80}")
    print("SCORING SKLEARN ML MODELS (5 models)")
    print(f"{'='*80}\n")

    sklearn_models = [
        ('iforest_n100', 'IsolationForest (n=100)', False),
        ('iforest_n200', 'IsolationForest (n=200)', False),
        ('lof_k20', 'LOF (k=20)', False),
        ('lof_k50', 'LOF (k=50)', False),
        ('ocsvm_nu002', 'OneClassSVM (nu=0.02)', True),  # Has scaler
    ]

    for model_name, model_desc, has_scaler in sklearn_models:
        model_path = args.models_dir / f'{model_name}.pkl'

        if model_path.exists():
            print(f"Scoring {model_desc}...")
            start = time.time()

            if has_scaler:
                # OneClassSVM is saved as (model, scaler) tuple
                model, scaler = joblib.load(model_path)
                scores = score_sklearn_model(model, X_test, scaler=scaler)
            else:
                model = joblib.load(model_path)
                scores = score_sklearn_model(model, X_test)

            elapsed = time.time() - start

            df_test[f'{model_name}_score'] = scores

            # Save predictions
            df_pred = df_test[['window_start', 'is_anomaly', 'anomaly_type', f'{model_name}_score']].copy()
            df_pred.columns = ['window_start', 'is_anomaly', 'anomaly_type', 'anomaly_score']
            df_pred['window_id'] = df_pred.index
            pred_path = predictions_dir / f'{model_name}_predictions.parquet'
            df_pred.to_parquet(pred_path, index=False)

            throughput = len(X_test) / elapsed
            print(f"  ✓ Scored {len(X_test):,} samples in {elapsed:.2f}s ({throughput:.0f} windows/s)\n")
        else:
            print(f"  ⚠️ WARNING: {model_path} not found, skipping\n")

    # =========================================================================
    # RRCF MODELS (2) - SKIPPED (too slow, ~3 hours for 17k samples)
    # =========================================================================

    print(f"\n{'='*80}")
    print("RRCF MODELS (2 models) - SKIPPED")
    print(f"{'='*80}\n")

    print("  ⚠️ RRCF scoring is extremely slow (insert/remove each test point)")
    print("  Estimated time: ~3 hours for 17,529 samples")
    print("  Skipping for now - can optimize later if needed\n")

    # NOTE: RRCF scoring requires inserting each test point into the forest,
    # computing CoDisp, then removing it. This is O(n_test * n_trees * log(tree_size))
    # which is prohibitively slow for large test sets.
    #
    # Possible optimizations:
    # 1. Sample a subset of test points
    # 2. Use fewer trees
    # 3. Implement batch scoring
    # 4. Use a faster RRCF implementation

    # =========================================================================
    # STATISTICAL MODELS (3) - Computed
    # =========================================================================

    print(f"\n{'='*80}")
    print("COMPUTING STATISTICAL MODELS (3 models)")
    print(f"{'='*80}\n")

    # Z-Score
    print("Computing Z-Score...")
    zscore = np.abs(df_test['ctx_dev'])
    df_test['zscore_score'] = zscore

    df_pred = df_test[['window_start', 'is_anomaly', 'anomaly_type', 'zscore_score']].copy()
    df_pred.columns = ['window_start', 'is_anomaly', 'anomaly_type', 'anomaly_score']
    df_pred['window_id'] = df_pred.index
    pred_path = predictions_dir / 'zscore_predictions.parquet'
    df_pred.to_parquet(pred_path, index=False)
    print(f"  ✓ Computed Z-Score\n")

    # IQR
    print("Computing IQR...")
    iqr = df_test['ctx_q75'] - df_test['ctx_q25']
    iqr_lower = df_test['ctx_q25'] - 1.5 * iqr
    iqr_upper = df_test['ctx_q75'] + 1.5 * iqr
    df_test['iqr_score'] = np.maximum(
        iqr_lower - df_test['trip_count'],
        df_test['trip_count'] - iqr_upper
    ).clip(lower=0)

    df_pred = df_test[['window_start', 'is_anomaly', 'anomaly_type', 'iqr_score']].copy()
    df_pred.columns = ['window_start', 'is_anomaly', 'anomaly_type', 'anomaly_score']
    df_pred['window_id'] = df_pred.index
    pred_path = predictions_dir / 'iqr_predictions.parquet'
    df_pred.to_parquet(pred_path, index=False)
    print(f"  ✓ Computed IQR\n")

    # MAD
    print("Computing MAD...")
    mad = np.abs(df_test['trip_count'] - df_test['ctx_median'])
    df_test['mad_score'] = mad

    df_pred = df_test[['window_start', 'is_anomaly', 'anomaly_type', 'mad_score']].copy()
    df_pred.columns = ['window_start', 'is_anomaly', 'anomaly_type', 'anomaly_score']
    df_pred['window_id'] = df_pred.index
    pred_path = predictions_dir / 'mad_predictions.parquet'
    df_pred.to_parquet(pred_path, index=False)
    print(f"  ✓ Computed MAD\n")

    # =========================================================================
    # WINDOW-BASED MODELS (2) - Computed
    # =========================================================================

    print(f"\n{'='*80}")
    print("COMPUTING WINDOW-BASED MODELS (2 models)")
    print(f"{'='*80}\n")

    # EWMA
    print("Computing EWMA...")
    alpha = 0.3
    ewma = df_test['trip_count'].ewm(alpha=alpha).mean()
    df_test['ewma_score'] = np.abs(df_test['trip_count'] - ewma)

    df_pred = df_test[['window_start', 'is_anomaly', 'anomaly_type', 'ewma_score']].copy()
    df_pred.columns = ['window_start', 'is_anomaly', 'anomaly_type', 'anomaly_score']
    df_pred['window_id'] = df_pred.index
    pred_path = predictions_dir / 'ewma_predictions.parquet'
    df_pred.to_parquet(pred_path, index=False)
    print(f"  ✓ Computed EWMA\n")

    # CUSUM
    print("Computing CUSUM...")
    target = df_test['ctx_mean']
    cusum_pos = (df_test['trip_count'] - target).clip(lower=0).cumsum()
    cusum_neg = (target - df_test['trip_count']).clip(lower=0).cumsum()
    df_test['cusum_score'] = np.maximum(cusum_pos, cusum_neg)

    df_pred = df_test[['window_start', 'is_anomaly', 'anomaly_type', 'cusum_score']].copy()
    df_pred.columns = ['window_start', 'is_anomaly', 'anomaly_type', 'anomaly_score']
    df_pred['window_id'] = df_pred.index
    pred_path = predictions_dir / 'cusum_predictions.parquet'
    df_pred.to_parquet(pred_path, index=False)
    print(f"  ✓ Computed CUSUM\n")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print(f"\n{'='*80}")
    print("SCORING COMPLETE")
    print(f"{'='*80}\n")

    print(f"Total models scored: 22 (RRCF skipped due to performance)")
    print(f"  - PyOD: 8")
    print(f"  - Deep Learning: 4")
    print(f"  - sklearn ML: 5")
    print(f"  - RRCF: 0 (skipped - too slow)")
    print(f"  - Statistical: 3")
    print(f"  - Window-based: 2")

    print(f"\nPredictions saved to: {predictions_dir}")
    print(f"Total prediction files: {len(list(predictions_dir.glob('*.parquet')))}")

    print(f"\nNext Steps:")
    print(f"  1. Evaluate: python exp_evaluate_simple.py")
    print(f"  2. Generate benchmark table: python exp_generate_benchmark_table.py\n")


if __name__ == "__main__":
    main()
