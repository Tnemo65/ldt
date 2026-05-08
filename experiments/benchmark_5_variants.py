#!/usr/bin/env python3
"""
Benchmark 5 Algorithm Variants - Validate Context-Aware Approach.
Task 2.16-2.20: Compare 5 variants to validate value of ratio features and context-aware thresholds

Variants:
1. Baseline Static - iForest + global threshold + 15D raw features (simplest)
2. Baseline Ratio - iForest + global threshold + 21D features (prototype: 92% Recall, 5% FPR)
3. Proposed Context-Aware - iForest + per-cluster thresholds + 21D (full system, target <4% FPR)
4. Opponent ARF - Adaptive Random Forest (established streaming algorithm)
5. Opponent LODA - Lightweight Online Detector (fast baseline)

Evaluation:
- Train on Jan 2024 clean baseline
- Validate on 50K synthetic anomalies
- 5 random seeds × 5 variants = 25 runs
- Metrics: Recall, FPR, Precision, F1, Throughput, Memory

Expected ranking:
1. Proposed Context-Aware (best FPR <4%)
2. Baseline Ratio (good Recall 92%)
3. Opponent ARF (competitive but slower)
4. Baseline Static (worst FPR 63%+)
5. Opponent LODA (fast but lower accuracy)
"""

import argparse
import sys
from pathlib import Path
import pickle
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
import psutil
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.vectorizer import FeatureVectorizer

from river.anomaly import HalfSpaceTrees, OneClassSVM
try:
    from river.forest import ARFClassifier
    ARF_AVAILABLE = True
except ImportError:
    ARF_AVAILABLE = False
    print("⚠ ARF not available in River")

# Try importing MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# Prototype-validated iForest config
IFOREST_CONFIG = {
    'n_trees': 200,
    'height': 10,
    'window_size': 512,
}


def get_context_key(record: dict, neighborhood_map: dict = None) -> str:
    """Generate 4D context key for per-cluster thresholds.

    Args:
        record: Trip record dict
        neighborhood_map: Zone ID → cluster name mapping (from neighborhood_mapping.json)

    Returns:
        Context key like "short_morning_weekday_zone_high_volume_1"
    """
    try:
        from datetime import datetime as dt

        pickup_dt = record.get('tpep_pickup_datetime')
        if isinstance(pickup_dt, str):
            pickup_dt = dt.fromisoformat(pickup_dt)

        # Trip type
        distance = record.get('trip_distance', 0)
        trip_type = 'short' if distance < 2 else ('medium' if distance < 10 else 'long')

        # Time window
        hour = pickup_dt.hour
        if 6 <= hour < 10:
            time_window = 'morning'
        elif 17 <= hour < 20:
            time_window = 'evening'
        elif 22 <= hour or hour < 6:
            time_window = 'night'
        else:
            time_window = 'midday'

        # Day type
        day_type = 'weekend' if pickup_dt.weekday() >= 5 else 'weekday'

        # Neighborhood (from KMeans clustering via mapping file)
        zone_id = str(record.get('PULocationID', 0))
        if neighborhood_map:
            neighborhood = neighborhood_map.get(zone_id, 'unknown')
        else:
            # Fallback to hardcoded (for backwards compatibility)
            zone_int = int(zone_id)
            if zone_int <= 50:
                neighborhood = 'manhattan'
            elif zone_int <= 100:
                neighborhood = 'brooklyn'
            elif zone_int in [132, 138]:
                neighborhood = 'airport'
            else:
                neighborhood = 'outer'

        return f"{trip_type}_{time_window}_{day_type}_{neighborhood}"
    except:
        return "unknown_unknown_unknown_unknown"


class VariantConfig:
    """Configuration for a single algorithm variant."""

    def __init__(self, name, model_factory, features_dim, threshold_strategy, description):
        self.name = name
        self.model_factory = model_factory
        self.features_dim = features_dim  # '15D' or '21D'
        self.threshold_strategy = threshold_strategy
        self.description = description


def train_variant(
    variant: VariantConfig,
    train_data_path: str,
    scaler_path: str,
    seed: int
):
    """Train a variant model.

    Args:
        variant: Variant configuration
        train_data_path: Path to training data
        scaler_path: Path to fitted scaler
        seed: Random seed

    Returns:
        Trained model, training time, memory usage
    """
    print(f"\n  Training {variant.name} (seed={seed})...")

    # Load data
    df = pd.read_parquet(train_data_path)

    # Vectorize
    vectorizer = FeatureVectorizer()
    X = []

    for idx, row in df.iterrows():
        try:
            features = vectorizer.transform(row.to_dict())

            # Use only first 15D for 15D variants
            if variant.features_dim == '15D':
                features = features[:15]

            X.append(features)
        except:
            pass

        if (idx + 1) % 500000 == 0:
            print(f"    Vectorized: {idx + 1:,}")

    X = np.array(X)

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X)

    # Train
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    start_time = time.time()

    model = variant.model_factory(seed=seed)

    for i, features in enumerate(X_scaled):
        feature_dict = {idx: float(val) for idx, val in enumerate(features)}
        model.learn_one(feature_dict)

        if (i + 1) % 500000 == 0:
            print(f"    Trained: {i + 1:,}")

    training_time = time.time() - start_time
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_mb = mem_after - mem_before

    print(f"    ✓ Time: {training_time:.1f}s, Memory: +{memory_mb:.0f}MB")

    return model, training_time, memory_mb


def validate_variant(
    variant: VariantConfig,
    model,
    scaler,
    val_data_path: str,
    labels_path: str,
    thresholds_path: str,
    neighborhood_map: dict
):
    """Validate a variant model.

    Args:
        variant: Variant configuration
        model: Trained model
        scaler: Fitted scaler
        val_data_path: Path to validation data
        labels_path: Path to ground truth labels
        thresholds_path: Path to context thresholds
        neighborhood_map: Zone ID → cluster name mapping

    Returns:
        Metrics dict
    """
    print(f"\n  Validating {variant.name}...")

    # Load data
    df = pd.read_parquet(val_data_path)
    labels_df = pd.read_csv(labels_path)

    # Load thresholds
    with open(thresholds_path) as f:
        thresholds = json.load(f)

    # Score
    vectorizer = FeatureVectorizer()
    y_pred = []
    scores = []

    start_time = time.time()

    for idx, row in df.iterrows():
        try:
            features = vectorizer.transform(row.to_dict())

            if variant.features_dim == '15D':
                features = features[:15]

            features_scaled = scaler.transform([features])[0]
            feature_dict = {i: float(val) for i, val in enumerate(features_scaled)}

            score = model.score_one(feature_dict)
            scores.append(score)

            # Get threshold based on strategy
            if variant.threshold_strategy == 'global_95th':
                threshold = np.percentile(scores, 95)
            elif variant.threshold_strategy == 'global_96th':
                threshold = np.percentile(scores, 96)
            elif variant.threshold_strategy == 'per_cluster_adaptive':
                context_key = get_context_key(row.to_dict(), neighborhood_map)
                threshold = thresholds.get('thresholds', {}).get(
                    context_key,
                    thresholds.get('global_threshold', 0.5)
                )
            else:
                threshold = thresholds.get('global_threshold', 0.5)

            y_pred.append(1 if score > threshold else 0)

        except:
            y_pred.append(0)

        if (idx + 1) % 10000 == 0:
            print(f"    Scored: {idx + 1:,}")

    scoring_time = time.time() - start_time
    throughput = len(df) / scoring_time

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
        'throughput_eps': throughput,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

    print(f"    ✓ Recall: {recall:.3f}, FPR: {fpr:.3f}, F1: {f1:.3f}, Throughput: {throughput:.0f} eps")

    return metrics


def run_benchmark(
    train_data_path: str = 'data/clean/jan_2024_clean_baseline.parquet',
    val_data_path: str = 'data/clean/jan_2024_with_50k_anomalies.parquet',
    labels_path: str = 'data/clean/anomaly_labels.csv',
    scaler_path: str = 'models/scaler.pkl',
    thresholds_path: str = 'models/context_thresholds_v2.json',
    mapping_path: str = 'src/config/neighborhood_mapping.json',
    n_seeds: int = 5,
    output_dir: str = 'experiments/benchmark_results'
):
    """Run 5-variant benchmark with multiple seeds.

    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data
        labels_path: Path to ground truth labels
        scaler_path: Path to fitted scaler
        thresholds_path: Path to context thresholds
        mapping_path: Path to neighborhood cluster mapping
        n_seeds: Number of random seeds
        output_dir: Directory to save results

    Returns:
        Results DataFrame
    """
    print("="*60)
    print("5-Variant Benchmark - Context-Aware Validation")
    print("="*60)

    # Define variants
    variants = [
        VariantConfig(
            name='baseline_static',
            model_factory=lambda seed: HalfSpaceTrees(**IFOREST_CONFIG, seed=seed),
            features_dim='15D',
            threshold_strategy='global_95th',
            description='Simplest: global threshold, raw features'
        ),
        VariantConfig(
            name='baseline_ratio',
            model_factory=lambda seed: HalfSpaceTrees(**IFOREST_CONFIG, seed=seed),
            features_dim='21D',
            threshold_strategy='global_96th',
            description='Prototype: ratio features reduce variance'
        ),
        VariantConfig(
            name='proposed_context_aware',
            model_factory=lambda seed: HalfSpaceTrees(**IFOREST_CONFIG, seed=seed),
            features_dim='21D',
            threshold_strategy='per_cluster_adaptive',
            description='Full system: ratio + per-cluster thresholds'
        ),
    ]

    # ARF if available
    if ARF_AVAILABLE:
        variants.append(VariantConfig(
            name='opponent_arf',
            model_factory=lambda seed: ARFClassifier(n_models=200, max_features='sqrt', seed=seed),
            features_dim='21D',
            threshold_strategy='global_95th',
            description='Adaptive Random Forest'
        ))

    # OneClassSVM (lightweight baseline)
    variants.append(VariantConfig(
        name='opponent_ocsvm',
        model_factory=lambda seed: OneClassSVM(nu=0.1),
        features_dim='21D',
        threshold_strategy='global_95th',
        description='Fast baseline: One-Class SVM (lightweight)'
    ))

    # Load neighborhood mapping (for context key generation)
    with open(mapping_path) as f:
        neighborhood_map = json.load(f)['mapping']

    print(f"\nVariants: {len(variants)}")
    print(f"Seeds: {n_seeds}")
    print(f"Total runs: {len(variants) * n_seeds}")

    # Load scaler once
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Run benchmark
    results = []

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Variant: {variant.name}")
        print(f"{'='*60}")
        print(f"Description: {variant.description}")
        print(f"Features: {variant.features_dim}")
        print(f"Threshold: {variant.threshold_strategy}")

        for seed in range(42, 42 + n_seeds):
            print(f"\n  Seed {seed}:")

            try:
                # Train
                model, train_time, memory_mb = train_variant(
                    variant, train_data_path, scaler_path, seed
                )

                # Validate
                metrics = validate_variant(
                    variant, model, scaler, val_data_path, labels_path, thresholds_path, neighborhood_map
                )

                # Combine results
                result = {
                    'variant': variant.name,
                    'description': variant.description,
                    'features': variant.features_dim,
                    'threshold_strategy': variant.threshold_strategy,
                    'seed': seed,
                    'training_time': train_time,
                    'memory_mb': memory_mb,
                    **metrics,
                    'timestamp': datetime.utcnow().isoformat()
                }

                results.append(result)

                # Log to MLflow if available
                if MLFLOW_AVAILABLE:
                    with mlflow.start_run(run_name=f"{variant.name}_seed{seed}"):
                        mlflow.log_params({
                            'variant': variant.name,
                            'features': variant.features_dim,
                            'seed': seed
                        })
                        mlflow.log_metrics(metrics)

            except Exception as e:
                print(f"\n  ❌ Failed: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_path = Path(output_dir) / f'benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results_df.to_csv(results_path, index=False)

    print(f"\n{'='*60}")
    print(f"Benchmark Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved: {results_path}")

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Summary Statistics (Mean ± Std)")
    print(f"{'='*60}")

    summary = results_df.groupby('variant').agg({
        'recall': ['mean', 'std'],
        'fpr': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'throughput_eps': ['mean', 'std']
    }).round(3)

    print(summary)

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Benchmark 5 algorithm variants')
    parser.add_argument('--train-data', type=str, default='data/clean/jan_2024_clean_baseline.parquet')
    parser.add_argument('--val-data', type=str, default='data/clean/jan_2024_with_50k_anomalies.parquet')
    parser.add_argument('--labels', type=str, default='data/clean/anomaly_labels.csv')
    parser.add_argument('--scaler', type=str, default='models/scaler.pkl')
    parser.add_argument('--thresholds', type=str, default='models/context_thresholds_v2.json')
    parser.add_argument('--mapping', type=str, default='src/config/neighborhood_mapping.json',
                       help='Neighborhood cluster mapping (zone ID → cluster name)')
    parser.add_argument('--n-seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--output-dir', type=str, default='experiments/benchmark_results')

    args = parser.parse_args()

    results = run_benchmark(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        labels_path=args.labels,
        scaler_path=args.scaler,
        thresholds_path=args.thresholds,
        mapping_path=args.mapping,
        n_seeds=args.n_seeds,
        output_dir=args.output_dir
    )

    return 0


if __name__ == '__main__':
    exit(main())
