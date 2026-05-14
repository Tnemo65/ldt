#!/usr/bin/env python3
"""
Hybrid vs Baseline Benchmark.

Compares:
- CA-DQStream (full hybrid with MemStream)
- vs IsolationForest baseline
- vs original MemStream

Metrics: precision, recall, F1, FPR, latency, BAR score
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memstream_core import MemStreamCore, BARController, set_determinism
from core.feature_extractor import FeatureVectorizer
from core.context_aware import ContextAwareFeatureVectorizer, get_4d_context
from scripts.inject_anomalies_multi import inject_anomalies, generate_synthetic_test_data


class BenchmarkRunner:
    """Runner for hybrid vs baseline benchmarks."""

    def __init__(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        seed: int = 42
    ):
        self.df = df
        self.labels = labels
        self.seed = seed
        self.results = {}

    def benchmark_isolation_forest(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100
    ) -> Dict:
        """Benchmark IsolationForest."""
        print("  Training IsolationForest...")
        start_time = time.time()

        # Extract features
        vectorizer = FeatureVectorizer()
        features = vectorizer.transform_batch(self.df)

        # Handle NaN
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

        # Train
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=self.seed,
            n_jobs=-1
        )
        model.fit(features)

        train_time = time.time() - start_time

        # Predict
        start_time = time.time()
        predictions = model.predict(features)
        predictions = (predictions == -1).astype(int)  # -1 = anomaly in sklearn

        inference_time = time.time() - start_time

        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.labels, predictions, average='binary', zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(self.labels, predictions).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        return {
            'model': 'IsolationForest',
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'fpr': float(fpr),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'train_time': train_time,
            'inference_time': inference_time,
            'records_per_second': len(self.df) / inference_time if inference_time > 0 else 0,
        }

    def benchmark_memstream_original(
        self,
        model: MemStreamCore,
        threshold: float = None
    ) -> Dict:
        """Benchmark original MemStream (25D)."""
        print("  Running MemStream (25D)...")
        start_time = time.time()

        vectorizer = FeatureVectorizer()
        scores = []
        predictions = []

        for idx, row in self.df.iterrows():
            features = vectorizer.transform(row.to_dict())
            score = model.score_one(features)
            scores.append(score)

            pred = 1 if score > (threshold or model.max_thres.item()) else 0
            predictions.append(pred)

        inference_time = time.time() - start_time
        scores = np.array(scores)
        predictions = np.array(predictions)

        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.labels, predictions, average='binary', zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(self.labels, predictions).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        return {
            'model': 'MemStream_original (25D)',
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'fpr': float(fpr),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'inference_time': inference_time,
            'records_per_second': len(self.df) / inference_time if inference_time > 0 else 0,
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
        }

    def benchmark_ca_memstream(
        self,
        model: MemStreamCore,
        threshold: float = None
    ) -> Dict:
        """Benchmark Context-Aware MemStream (40D)."""
        print("  Running CA-MemStream (40D)...")
        start_time = time.time()

        vectorizer = ContextAwareFeatureVectorizer()
        bar_controller = BARController()
        scores = []
        predictions = []
        memory_updates = 0

        for idx, row in self.df.iterrows():
            record = row.to_dict()
            ctx = get_4d_context(record)
            features = vectorizer.transform(record, ctx)

            score = model.score_one(features)
            scores.append(score)

            # BAR controller
            should_update, reason = bar_controller.should_update_memory(
                ctx['neighborhood'], score
            )
            if should_update:
                model.memory_update(features)
                memory_updates += 1

            pred = 1 if score > (threshold or model.max_thres.item()) else 0
            predictions.append(pred)

        inference_time = time.time() - start_time
        scores = np.array(scores)
        predictions = np.array(predictions)

        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.labels, predictions, average='binary', zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(self.labels, predictions).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        bar_stats = bar_controller.get_stats()

        return {
            'model': 'CA-MemStream (40D)',
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'fpr': float(fpr),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'inference_time': inference_time,
            'records_per_second': len(self.df) / inference_time if inference_time > 0 else 0,
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'bar_rate': bar_stats['bar_rate'],
            'bar_rate_pct': bar_stats['bar_rate_pct'],
            'memory_updates': memory_updates,
            'drift_events': bar_stats['drift_events'],
        }


def main():
    parser = argparse.ArgumentParser(description='Hybrid vs Baseline Benchmark')
    parser.add_argument('--data', type=str, help='Test CSV path (optional)')
    parser.add_argument('--model-25d', type=str, help='25D model path (optional)')
    parser.add_argument('--model-40d', type=str, help='40D model path (optional)')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path')
    parser.add_argument('--n-records', type=int, default=5000, help='Number of records (if generating)')
    parser.add_argument('--n-anomalies', type=int, default=500, help='Number of anomalies')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_determinism(args.seed)

    print("=" * 60)
    print("HYBRID VS BASELINE BENCHMARK")
    print("=" * 60)

    # Load or generate data
    if args.data:
        print(f"\n[1] Loading data from {args.data}...")
        df = pd.read_csv(args.data)
    else:
        print(f"\n[1] Generating synthetic test data...")
        df, labels = generate_synthetic_test_data(
            n_normal=args.n_records - args.n_anomalies,
            n_anomalies=args.n_anomalies,
            seed=args.seed
        )
        print(f"  Generated {len(df)} records")

    if 'labels' not in dir():
        print(f"  Injecting {args.n_anomalies} anomalies...")
        df, labels = inject_anomalies(df, n_anomalies=args.n_anomalies, seed=args.seed)

    print(f"  Total records: {len(df):,}")
    print(f"  Anomalies: {int(np.sum(labels)):,} ({np.mean(labels)*100:.1f}%)")

    # Initialize benchmark runner
    runner = BenchmarkRunner(df, labels, seed=args.seed)

    # Benchmark IsolationForest
    print("\n[2] Benchmarking IsolationForest...")
    if_model = runner.benchmark_isolation_forest(contamination=0.05)
    print(f"  F1: {if_model['f1']:.4f}, FPR: {if_model['fpr']:.4f}")

    # Benchmark MemStream 25D
    print("\n[3] Benchmarking MemStream (25D)...")
    if args.model_25d:
        ms_25d = MemStreamCore.load(args.model_25d, signing_key='training-signing-key')
        ms_25d_model = runner.benchmark_memstream_original(ms_25d)
    else:
        # Train a quick MemStream model for testing
        print("  Training quick MemStream model...")
        from core.memstream_core import MemStreamCore, MemStreamConfig
        from scripts.train_warmup import generate_sample_taxi_data

        # Generate training data
        train_df, _ = generate_synthetic_test_data(n_normal=2000, n_anomalies=0, seed=args.seed)
        vectorizer = FeatureVectorizer()
        train_features = vectorizer.transform_batch(train_df[:1500])

        cfg = MemStreamConfig()
        ms_25d = MemStreamCore(cfg=cfg)
        ms_25d.warmup(train_features, epochs=50, verbose=False)

        # Calibrate
        calib_features = vectorizer.transform_batch(train_df[1500:])
        calib_scores = [ms_25d.score_one(f) for f in calib_features]
        ms_25d.set_beta(np.percentile(calib_scores, 95))

        ms_25d_model = runner.benchmark_memstream_original(ms_25d)

    print(f"  F1: {ms_25d_model['f1']:.4f}, FPR: {ms_25d_model['fpr']:.4f}")

    # Benchmark CA-MemStream 40D
    print("\n[4] Benchmarking CA-MemStream (40D)...")
    if args.model_40d:
        ms_40d = MemStreamCore.load(args.model_40d, signing_key='training-signing-key')
        ca_ms_model = runner.benchmark_ca_memstream(ms_40d)
    else:
        # Train a quick CA-MemStream model for testing
        print("  Training quick CA-MemStream model...")
        from core.memstream_core import MemStreamCore, MemStreamConfig, get_4d_context

        train_df, _ = generate_synthetic_test_data(n_normal=2000, n_anomalies=0, seed=args.seed)
        vectorizer_40d = ContextAwareFeatureVectorizer()

        train_features = []
        for _, row in train_df.iterrows():
            ctx = get_4d_context(row.to_dict())
            features = vectorizer_40d.transform(row.to_dict(), ctx)
            train_features.append(features)
        train_features = np.array(train_features)

        cfg_40d = MemStreamConfig()
        cfg_40d.in_dim = 40
        cfg_40d.out_dim = 40
        ms_40d = MemStreamCore(cfg=cfg_40d)
        ms_40d.warmup(train_features[:1500], epochs=50, verbose=False)

        # Calibrate
        calib_features = [vectorizer_40d.transform(row.to_dict(), get_4d_context(row.to_dict()))
                         for _, row in train_df[1500:].iterrows()]
        calib_scores = [ms_40d.score_one(f) for f in calib_features]
        ms_40d.set_beta(np.percentile(calib_scores, 95))

        ca_ms_model = runner.benchmark_ca_memstream(ms_40d)

    print(f"  F1: {ca_ms_model['f1']:.4f}, FPR: {ca_ms_model['fpr']:.4f}")
    print(f"  BAR Rate: {ca_ms_model['bar_rate_pct']:.2f}%")

    # Compute improvements
    print("\n[5] Computing improvements...")
    f1_improvement = (ca_ms_model['f1'] - if_model['f1']) / max(if_model['f1'], 0.001) * 100
    fpr_improvement = (if_model['fpr'] - ca_ms_model['fpr']) / max(if_model['fpr'], 0.001) * 100

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n{'Model':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FPR':<12}")
    print("-" * 73)
    print(f"{'IsolationForest':<25} {if_model['precision']:.4f}      {if_model['recall']:.4f}      {if_model['f1']:.4f}      {if_model['fpr']:.4f}")
    print(f"{'MemStream (25D)':<25} {ms_25d_model['precision']:.4f}      {ms_25d_model['recall']:.4f}      {ms_25d_model['f1']:.4f}      {ms_25d_model['fpr']:.4f}")
    print(f"{'CA-MemStream (40D)':<25} {ca_ms_model['precision']:.4f}      {ca_ms_model['recall']:.4f}      {ca_ms_model['f1']:.4f}      {ca_ms_model['fpr']:.4f}")

    print(f"\n{'Model':<25} {'Latency (ms/rec)':<18} {'Records/sec':<15} {'BAR Rate':<12}")
    print("-" * 70)
    if_lat = (if_model['inference_time'] / len(df)) * 1000
    ms_lat = (ms_25d_model['inference_time'] / len(df)) * 1000
    ca_lat = (ca_ms_model['inference_time'] / len(df)) * 1000

    print(f"{'IsolationForest':<25} {if_lat:.4f} ms            {if_model['records_per_second']:.0f}            -")
    print(f"{'MemStream (25D)':<25} {ms_lat:.4f} ms            {ms_25d_model['records_per_second']:.0f}            -")
    print(f"{'CA-MemStream (40D)':<25} {ca_lat:.4f} ms            {ca_ms_model['records_per_second']:.0f}            {ca_ms_model['bar_rate_pct']:.2f}%")

    print(f"\n{'='*60}")
    print("IMPROVEMENTS (CA-MemStream vs IsolationForest):")
    print(f"  F1 Improvement: {f1_improvement:+.1f}%")
    print(f"  FPR Improvement: {fpr_improvement:+.1f}%")
    print(f"  BAR Rate: {ca_ms_model['bar_rate_pct']:.2f}% (target: 1-5%)")
    print("=" * 60)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_records': len(df),
        'n_anomalies': int(np.sum(labels)),
        'anomaly_rate': float(np.mean(labels)),
        'isolation_forest': if_model,
        'memstream_25d': ms_25d_model,
        'ca_memstream_40d': ca_ms_model,
        'improvements': {
            'f1_improvement_pct': f1_improvement,
            'fpr_improvement_pct': fpr_improvement,
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[6] Results saved to {output_path}")


if __name__ == '__main__':
    main()
