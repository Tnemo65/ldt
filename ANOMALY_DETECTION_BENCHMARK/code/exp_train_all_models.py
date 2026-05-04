#!/usr/bin/env python3
"""
Train All 18 Anomaly Detection Models - Consolidated Script
============================================================
Trains all 18 models with data-driven parameters (20 total configurations).

Based on data analysis (analyze_data_for_params.py):
- PCA: n_components=3 (95% variance)
- KNN: n_neighbors=5 (local) or 20 (contextual)
- RRCF: tree_size=256 (strong daily correlation 0.88)
- Deep Learning: encoding_dim=8 or 16
- MCD: Will underperform (non-Gaussian data)
- COPOD/ECOD: Expected to excel (parameter-free)

Usage:
    python exp_train_all_models.py --train-file /nfs/interns/dacthinh/repos/benchmark/anomalies/output_anomaly/train_2024.parquet \\
                                    --output-dir /nfs/interns/dacthinh/repos/benchmark/anomalies/output_anomaly/models

Models Trained (20 configurations):
    PyOD (8): HBOS, KNN×2, MCD, PCA, COPOD, ABOD, ECOD
    Deep Learning (4): Autoencoder×2, VAE×2
    Existing ML (6): IsolationForest×2, LOF×2, OneClassSVM, RRCF×2
    Statistical (3): Z-Score, IQR, MAD (computed, not trained)
    Sequential (2): EWMA, CUSUM (stateful, trained separately)
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Import PyOD models directly
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from pyod.models.copod import COPOD
from pyod.models.abod import ABOD
from pyod.models.ecod import ECOD

# Import sklearn models
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Import deep learning models
from exp_ml_autoencoder import train_autoencoder, train_vae

# Import RRCF
try:
    import rrcf
    RRCF_AVAILABLE = True
except ImportError:
    RRCF_AVAILABLE = False
    print("Warning: rrcf not installed, RRCF models will be skipped")

warnings.filterwarnings('ignore')


def get_feature_columns(df: pd.DataFrame) -> list:
    """Extract feature column names from dataframe.

    Excludes: window_start, trip_count, hour, dow (first 4 columns)
    Includes: All computed features (context, lag, rolling, temporal)
    """
    # Get all columns after the first 4 (window_start, trip_count, hour, dow)
    feature_cols = df.columns[4:].tolist()

    # Remove any remaining non-feature columns
    exclude_cols = ['is_anomaly', 'anomaly_type', 'label']
    feature_cols = [col for col in feature_cols if col not in exclude_cols]

    return feature_cols


def train_all_models(df_train: pd.DataFrame, output_dir: Path) -> dict:
    """Train all 23 model configurations.

    Args:
        df_train: Training dataframe with features
        output_dir: Directory to save trained models

    Returns:
        dict: {model_name: {'model': model, 'train_time': seconds, 'path': filepath}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract features
    feature_cols = get_feature_columns(df_train)
    X_train = df_train[feature_cols]

    print(f"\n{'='*80}")
    print(f"Training {len(X_train)} samples with {len(feature_cols)} features")
    print(f"Features: {feature_cols}")
    print(f"{'='*80}\n")

    results = {}

    # =========================================================================
    # PyOD MODELS (9 configurations)
    # =========================================================================

    print(f"\n{'='*80}")
    print("TRAINING PyOD MODELS (9 configurations)")
    print(f"{'='*80}\n")

    # 1. HBOS - Histogram-Based Outlier Score
    print("[1/20] Training HBOS (n_bins=10)...")
    start = time.time()
    model = HBOS(contamination=0.01, n_bins=10)
    model.fit(X_train.values)
    train_time = time.time() - start
    model_path = output_dir / 'hbos_bins10.pkl'
    joblib.dump(model, model_path)
    results['hbos_bins10'] = {'model': model, 'train_time': train_time, 'path': model_path}
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # 2. KNN - k=5 (local patterns)
    print("[2/20] Training KNN (n_neighbors=5, local patterns)...")
    start = time.time()
    model = KNN(contamination=0.01, n_neighbors=5, method='largest')
    model.fit(X_train.values)
    train_time = time.time() - start
    model_path = output_dir / 'knn_k5.pkl'
    joblib.dump(model, model_path)
    results['knn_k5'] = {'model': model, 'train_time': train_time, 'path': model_path}
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # 3. KNN - k=20 (contextual patterns)
    print("[3/20] Training KNN (n_neighbors=20, contextual patterns)...")
    start = time.time()
    model = KNN(contamination=0.01, n_neighbors=20, method='largest')
    model.fit(X_train.values)
    train_time = time.time() - start
    model_path = output_dir / 'knn_k20.pkl'
    joblib.dump(model, model_path)
    results['knn_k20'] = {'model': model, 'train_time': train_time, 'path': model_path}
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # 4. MCD - auto support_fraction
    print("[4/20] Training MCD (support_fraction=None, auto)...")
    print("  ⚠️ WARNING: Data is non-Gaussian, MCD may underperform")
    start = time.time()
    model = MCD(contamination=0.01, support_fraction=None)
    model.fit(X_train.values)
    train_time = time.time() - start
    model_path = output_dir / 'mcd_auto.pkl'
    joblib.dump(model, model_path)
    results['mcd_auto'] = {'model': model, 'train_time': train_time, 'path': model_path}
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # 5. PCA - n_components=3 (95% variance from analysis)
    print("[5/20] Training PCA (n_components=3, 95% variance)...")
    start = time.time()
    model = PCA(contamination=0.01, n_components=3)
    model.fit(X_train.values)
    train_time = time.time() - start
    model_path = output_dir / 'pca_n3.pkl'
    joblib.dump(model, model_path)
    results['pca_n3'] = {'model': model, 'train_time': train_time, 'path': model_path}
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # 6. COPOD - parameter-free
    print("[6/20] Training COPOD (parameter-free)...")
    print("  ✓ Expected to excel on non-Gaussian data")
    start = time.time()
    model = COPOD(contamination=0.01)
    model.fit(X_train.values)
    train_time = time.time() - start
    model_path = output_dir / 'copod.pkl'
    joblib.dump(model, model_path)
    results['copod'] = {'model': model, 'train_time': train_time, 'path': model_path}
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # 7. ABOD - fast method (O(n³) exact is prohibitive)
    print("[7/20] Training ABOD (method='fast')...")
    start = time.time()
    model = ABOD(contamination=0.01, method='fast')
    model.fit(X_train.values)
    train_time = time.time() - start
    model_path = output_dir / 'abod_fast.pkl'
    joblib.dump(model, model_path)
    results['abod_fast'] = {'model': model, 'train_time': train_time, 'path': model_path}
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # 8. ECOD - parameter-free
    print("[8/20] Training ECOD (parameter-free, distribution-agnostic)...")
    print("  ✓ Expected to excel on non-Gaussian data")
    start = time.time()
    model = ECOD(contamination=0.01)
    model.fit(X_train.values)
    train_time = time.time() - start
    model_path = output_dir / 'ecod.pkl'
    joblib.dump(model, model_path)
    results['ecod'] = {'model': model, 'train_time': train_time, 'path': model_path}
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # =========================================================================
    # DEEP LEARNING MODELS (4 configurations)
    # =========================================================================

    print(f"\n{'='*80}")
    print("TRAINING DEEP LEARNING MODELS (4 configurations)")
    print(f"{'='*80}\n")

    # 9. Autoencoder - encoding_dim=8
    print("[9/20] Training Autoencoder (encoding_dim=8, aggressive compression)...")
    start = time.time()
    model, scaler = train_autoencoder(X_train, encoding_dim=8, epochs=50, batch_size=32)
    train_time = time.time() - start
    model_path = output_dir / 'autoencoder_dim8.keras'
    scaler_path = output_dir / 'autoencoder_dim8_scaler.pkl'
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    results['autoencoder_dim8'] = {
        'model': model, 'scaler': scaler, 'train_time': train_time,
        'path': model_path, 'scaler_path': scaler_path
    }
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # 10. Autoencoder - encoding_dim=16
    print("[10/20] Training Autoencoder (encoding_dim=16, preserve more info)...")
    start = time.time()
    model, scaler = train_autoencoder(X_train, encoding_dim=16, epochs=50, batch_size=32)
    train_time = time.time() - start
    model_path = output_dir / 'autoencoder_dim16.keras'
    scaler_path = output_dir / 'autoencoder_dim16_scaler.pkl'
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    results['autoencoder_dim16'] = {
        'model': model, 'scaler': scaler, 'train_time': train_time,
        'path': model_path, 'scaler_path': scaler_path
    }
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # 11. VAE - latent_dim=8
    print("[11/20] Training VAE (latent_dim=8)...")
    start = time.time()
    model, scaler = train_vae(X_train, latent_dim=8, epochs=50, batch_size=32)
    train_time = time.time() - start
    model_path = output_dir / 'vae_dim8.keras'
    scaler_path = output_dir / 'vae_dim8_scaler.pkl'
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    results['vae_dim8'] = {
        'model': model, 'scaler': scaler, 'train_time': train_time,
        'path': model_path, 'scaler_path': scaler_path
    }
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # 12. VAE - latent_dim=16
    print("[12/20] Training VAE (latent_dim=16)...")
    start = time.time()
    model, scaler = train_vae(X_train, latent_dim=16, epochs=50, batch_size=32)
    train_time = time.time() - start
    model_path = output_dir / 'vae_dim16.keras'
    scaler_path = output_dir / 'vae_dim16_scaler.pkl'
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    results['vae_dim16'] = {
        'model': model, 'scaler': scaler, 'train_time': train_time,
        'path': model_path, 'scaler_path': scaler_path
    }
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # =========================================================================
    # EXISTING ML MODELS (6 configurations)
    # =========================================================================

    print(f"\n{'='*80}")
    print("TRAINING EXISTING ML MODELS (6 configurations)")
    print(f"{'='*80}\n")

    # 13. IsolationForest - n_estimators=100
    print("[13/20] Training IsolationForest (n_estimators=100)...")
    start = time.time()
    model = IsolationForest(contamination=0.01, n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train.values)
    train_time = time.time() - start
    model_path = output_dir / 'iforest_n100.pkl'
    joblib.dump(model, model_path)
    results['iforest_n100'] = {'model': model, 'train_time': train_time, 'path': model_path}
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # 14. IsolationForest - n_estimators=200
    print("[14/20] Training IsolationForest (n_estimators=200)...")
    start = time.time()
    model = IsolationForest(contamination=0.01, n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train.values)
    train_time = time.time() - start
    model_path = output_dir / 'iforest_n200.pkl'
    joblib.dump(model, model_path)
    results['iforest_n200'] = {'model': model, 'train_time': train_time, 'path': model_path}
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # 15. LOF - n_neighbors=20
    print("[15/20] Training LOF (n_neighbors=20)...")
    start = time.time()
    model = LocalOutlierFactor(contamination=0.01, novelty=True, n_neighbors=20, n_jobs=-1)
    model.fit(X_train.values)
    train_time = time.time() - start
    model_path = output_dir / 'lof_k20.pkl'
    joblib.dump(model, model_path)
    results['lof_k20'] = {'model': model, 'train_time': train_time, 'path': model_path}
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # 16. LOF - n_neighbors=50
    print("[16/20] Training LOF (n_neighbors=50)...")
    start = time.time()
    model = LocalOutlierFactor(contamination=0.01, novelty=True, n_neighbors=50, n_jobs=-1)
    model.fit(X_train.values)
    train_time = time.time() - start
    model_path = output_dir / 'lof_k50.pkl'
    joblib.dump(model, model_path)
    results['lof_k50'] = {'model': model, 'train_time': train_time, 'path': model_path}
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # 17. OneClassSVM - nu=0.02 (already optimal)
    print("[17/20] Training OneClassSVM (nu=0.02)...")
    start = time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.values)
    model = OneClassSVM(nu=0.02, kernel='rbf', gamma='auto')
    model.fit(X_scaled)
    train_time = time.time() - start
    model_path = output_dir / 'ocsvm_nu002.pkl'
    scaler_path = output_dir / 'ocsvm_nu002_scaler.pkl'
    joblib.dump((model, scaler), model_path)
    results['ocsvm_nu002'] = {
        'model': model, 'scaler': scaler, 'train_time': train_time,
        'path': model_path, 'scaler_path': scaler_path
    }
    print(f"  ✓ Saved to {model_path} ({train_time:.2f}s)\n")

    # =========================================================================
    # TREE-BASED MODELS (RRCF) (2 configurations)
    # =========================================================================

    print(f"\n{'='*80}")
    print("TRAINING TREE-BASED MODELS - RRCF (2 configurations)")
    print(f"{'='*80}\n")

    if not RRCF_AVAILABLE:
        print("  ⚠️ WARNING: rrcf not installed, skipping RRCF models")
        print("  Install with: pip install rrcf\n")
    else:
        # 18. RRCF - num_trees=100, tree_size=256
        print("[18/20] Training RRCF (num_trees=100, tree_size=256)...")
        print("  ✓ tree_size=256 justified by strong daily correlation (0.88)")
        start = time.time()

        # Build RRCF forest manually
        num_trees = 100
        forest = {}
        for tree_id in range(num_trees):
            tree = rrcf.RCTree()
            # Distribute points across trees in round-robin fashion
            indices = np.arange(len(X_train))
            np.random.seed(42 + tree_id)  # Deterministic seeding
            shuffled = np.random.permutation(indices)

            # Each tree gets roughly 1/num_trees of the data
            points_per_tree = (len(X_train) + num_trees - 1) // num_trees
            start_idx = tree_id * points_per_tree
            end_idx = min(start_idx + points_per_tree, len(X_train))

            # Insert points into tree
            for idx in shuffled[start_idx:end_idx]:
                point = X_train.values[idx]
                tree.insert_point(point, index=idx)

            forest[tree_id] = tree

        rrcf_model = {
            'forest': forest,
            'num_trees': num_trees,
            'tree_size': 256,
            'feature_cols': feature_cols
        }
        train_time = time.time() - start

        # Save with dill (RRCF requires dill, not pickle/joblib)
        import dill
        model_path = output_dir / 'rrcf_t100_s256.pkl'
        with open(model_path, 'wb') as f:
            dill.dump(rrcf_model, f)
        results['rrcf_t100_s256'] = {'model': rrcf_model, 'train_time': train_time, 'path': model_path}
        print(f"  ✓ Saved to {model_path} with dill ({train_time:.2f}s)\n")

        # 19. RRCF - num_trees=200, tree_size=256
        print("[19/20] Training RRCF (num_trees=200, tree_size=256)...")
        start = time.time()

        # Build RRCF forest manually
        num_trees = 200
        forest = {}
        for tree_id in range(num_trees):
            tree = rrcf.RCTree()
            indices = np.arange(len(X_train))
            np.random.seed(42 + tree_id)
            shuffled = np.random.permutation(indices)

            points_per_tree = (len(X_train) + num_trees - 1) // num_trees
            start_idx = tree_id * points_per_tree
            end_idx = min(start_idx + points_per_tree, len(X_train))

            for idx in shuffled[start_idx:end_idx]:
                point = X_train.values[idx]
                tree.insert_point(point, index=idx)

            forest[tree_id] = tree

        rrcf_model = {
            'forest': forest,
            'num_trees': num_trees,
            'tree_size': 256,
            'feature_cols': feature_cols
        }
        train_time = time.time() - start

        model_path = output_dir / 'rrcf_t200_s256.pkl'
        with open(model_path, 'wb') as f:
            dill.dump(rrcf_model, f)
        results['rrcf_t200_s256'] = {'model': rrcf_model, 'train_time': train_time, 'path': model_path}
        print(f"  ✓ Saved to {model_path} with dill ({train_time:.2f}s)\n")

    # =========================================================================
    # STATISTICAL & SEQUENTIAL MODELS (computed, not trained)
    # =========================================================================

    print(f"\n{'='*80}")
    print("STATISTICAL & SEQUENTIAL MODELS (not trained here)")
    print(f"{'='*80}\n")

    print("[20/20] Statistical models (Z-Score, IQR, MAD):")
    print("  → Computed from context statistics, no training needed")
    print("\nSequential models (EWMA, CUSUM):")
    print("  → Stateful, trained separately with pre-warming")
    print(f"  → See exp_kafka_window.py for implementation\n")

    return results


def print_summary(results: dict):
    """Print training summary statistics."""
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}\n")

    print(f"Total Models Trained: {len(results)}")
    print(f"\nTraining Times:")

    # Sort by training time
    sorted_results = sorted(results.items(), key=lambda x: x[1]['train_time'])

    total_time = 0
    for model_name, info in sorted_results:
        total_time += info['train_time']
        print(f"  {model_name:25s}: {info['train_time']:>7.2f}s")

    print(f"\n  {'TOTAL':25s}: {total_time:>7.2f}s ({total_time/60:.1f} min)")

    print(f"\nModel Files Saved:")
    for model_name, info in results.items():
        print(f"  ✓ {info['path']}")
        if 'scaler_path' in info:
            print(f"    + {info['scaler_path']}")

    print(f"\n{'='*80}")
    print("✓ ALL MODELS TRAINED SUCCESSFULLY")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train all 18 anomaly detection models (20 configurations)'
    )
    parser.add_argument(
        '--train-file',
        type=str,
        required=True,
        help='Path to 2024 training data parquet file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save trained models'
    )

    args = parser.parse_args()

    # Load training data
    print(f"Loading training data from: {args.train_file}")
    df_train = pd.read_parquet(args.train_file)
    print(f"Loaded {len(df_train)} samples with {len(df_train.columns)} columns\n")

    # Train all models
    output_dir = Path(args.output_dir)
    results = train_all_models(df_train, output_dir)

    # Print summary
    print_summary(results)

    print(f"\nNext Steps:")
    print(f"  1. Score all models: python exp_score_all.py")
    print(f"  2. Evaluate: python exp_evaluate_simple.py")
    print(f"  3. Generate benchmark table: python exp_generate_benchmark_table.py\n")


if __name__ == '__main__':
    main()
