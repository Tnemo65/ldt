#!/usr/bin/env python3
"""
Compare multiple anomaly detection models on synthetic data.

Tests all available models and compares their performance:
- sklearn IsolationForest (new, trained on full 2.4M data)
- GaussianScorer (if available)

Usage:
  python scripts/compare_models.py
  python scripts/compare_models.py --subset 10000  # Quick test
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, '.')
from src.features.vectorizer import FeatureVectorizer


def load_model(model_path: str):
    """Load a trained model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def is_river_model(model):
    """Detect if model is River HalfSpaceTrees (vs sklearn IsolationForest)."""
    return hasattr(model, 'score_one')


def score_records(model, X_scaled, model_name: str):
    """Score records and return anomaly scores.

    Handles both River HalfSpaceTrees (score_one dict) and
    sklearn IsolationForest (score_samples array).
    """
    print(f"   Scoring with {model_name}...")

    if is_river_model(model):
        # River HalfSpaceTrees: batch via score_many if available, else loop
        scores = []
        for i, features in enumerate(X_scaled):
            feature_dict = {idx: float(val) for idx, val in enumerate(features)}
            score = model.score_one(feature_dict)
            scores.append(score)
            if (i + 1) % 5000 == 0:
                print(f"      {i+1:,} / {len(X_scaled):,}")
        return np.array(scores)
    else:
        # sklearn IsolationForest: batch score_samples
        # Returns negative scores (lower = more anomalous)
        raw_scores = model.score_samples(X_scaled)
        return -raw_scores  # Negate so higher = more anomalous


def compute_metrics(y_true, y_pred, threshold):
    """Compute confusion matrix and metrics."""
    y_pred_binary = (y_pred >= threshold).astype(int)

    tp = ((y_true == 1) & (y_pred_binary == 1)).sum()
    fp = ((y_true == 0) & (y_pred_binary == 1)).sum()
    tn = ((y_true == 0) & (y_pred_binary == 0)).sum()
    fn = ((y_true == 1) & (y_pred_binary == 0)).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'recall': recall, 'fpr': fpr,
        'precision': precision, 'f1': f1
    }


def find_best_threshold(y_true, scores):
    """Find threshold that maximizes F1 score."""
    thresholds = np.percentile(scores, [90, 95, 96, 97, 98, 99])
    best_f1 = 0
    best_threshold = None
    best_metrics = None

    for threshold in thresholds:
        metrics = compute_metrics(y_true, scores, threshold)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def main():
    parser = argparse.ArgumentParser(description='Compare anomaly detection models')
    parser.add_argument(
        '--subset',
        type=int,
        default=None,
        help='Test on subset of records (for quick validation)'
    )
    args = parser.parse_args()

    print("="*70)
    print("MODEL COMPARISON ON SYNTHETIC ANOMALIES")
    print("="*70)

    # 1. Load data
    print("\n1. Loading data...")
    df = pd.read_parquet('data/clean/jan_2024_with_50k_anomalies.parquet')
    labels = pd.read_csv('data/clean/anomaly_labels.csv')

    if args.subset:
        print(f"   Using subset: {args.subset:,} records")
        df = df.iloc[:args.subset]
        labels = labels.iloc[:args.subset]

    print(f"   Total: {len(df):,}")
    print(f"   Anomalies: {labels['is_anomaly'].sum():,}")
    print(f"   Clean: {(~labels['is_anomaly']).sum():,}")

    # 2. Batch vectorize
    print("\n2. Batch vectorizing features...")
    vectorizer = FeatureVectorizer()
    X = vectorizer.transform_batch(df)
    print(f"   Vectorized: {X.shape}")

    # 3. Scale
    print("\n3. Scaling features...")
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)

    # 4. Load models
    print("\n4. Loading models...")
    models = {}

    model_files = [
        ('sklearn IsolationForest (full)', 'models/iforest_model.pkl'),
        ('GaussianScorer', 'models/gaussian_model.pkl'),
    ]

    for name, path in model_files:
        if Path(path).exists():
            models[name] = load_model(path)
            print(f"   ✓ {name}")
        else:
            print(f"   ⊗ {name} (not found)")

    if not models:
        print("\n❌ No models found!")
        return 1

    # 5. Score all models
    print("\n5. Scoring records with all models...")
    results = {}

    y_true = labels['is_anomaly'].values

    for name, model in models.items():
        scores = score_records(model, X_scaled, name)
        threshold, metrics = find_best_threshold(y_true, scores)

        results[name] = {
            'scores': scores,
            'threshold': threshold,
            'metrics': metrics
        }

    # 6. Display comparison
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)

    print(f"\n{'Model':<25} {'Threshold':<12} {'Recall':<10} {'FPR':<10} {'F1':<10}")
    print("-" * 70)

    for name, result in results.items():
        metrics = result['metrics']
        threshold = result['threshold']

        status = "✅" if metrics['recall'] >= 0.75 and metrics['fpr'] < 0.05 else "❌"

        print(f"{name:<25} {threshold:<12.4f} {metrics['recall']:<10.3f} {metrics['fpr']:<10.3f} {metrics['f1']:<10.3f} {status}")

    # Find best model
    print("\n" + "="*70)
    print("BEST MODEL")
    print("="*70)

    best_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1'])
    best_result = results[best_name]

    print(f"\n🏆 Best: {best_name}")
    print(f"   Threshold: {best_result['threshold']:.4f}")
    print(f"   Recall: {best_result['metrics']['recall']:.3f}")
    print(f"   FPR: {best_result['metrics']['fpr']:.3f}")
    print(f"   F1: {best_result['metrics']['f1']:.3f}")

    if best_result['metrics']['recall'] >= 0.75 and best_result['metrics']['fpr'] < 0.05:
        print("\n✅ MEETS GO/NO-GO CRITERIA!")
    else:
        print("\n❌ Does not meet criteria yet")
        print(f"   Recall: {best_result['metrics']['recall']:.3f} (target: ≥0.75)")
        print(f"   FPR: {best_result['metrics']['fpr']:.3f} (target: <0.05)")

    return 0


if __name__ == '__main__':
    exit(main())
