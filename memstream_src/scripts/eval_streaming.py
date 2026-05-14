#!/usr/bin/env python3
"""
Streaming Evaluation Script for CA-DQStream + MemStream.

Simulates streaming evaluation on test data with:
- Latency tracking (p50, p95, p99)
- Precision, recall, F1, FPR over time windows
- Drift detection events
- Results written to JSON
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict, deque

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memstream_core import MemStreamCore, BARController, set_determinism
from core.feature_extractor import FeatureVectorizer
from scripts.inject_anomalies_multi import inject_anomalies


class StreamingEvaluator:
    """Streaming evaluator for online anomaly detection."""

    def __init__(
        self,
        model: MemStreamCore,
        vectorizer: FeatureVectorizer,
        window_size: int = 1000
    ):
        self.model = model
        self.vectorizer = vectorizer
        self.window_size = window_size

        # Latency tracking
        self.latencies: List[float] = []
        self.latency_window = deque(maxlen=window_size)

        # Metrics tracking
        self.scores: List[float] = []
        self.predictions: List[int] = []
        self.labels: List[int] = []
        self.timestamps: List[float] = []

        # Per-window metrics
        self.window_tp = 0
        self.window_fp = 0
        self.window_tn = 0
        self.window_fn = 0
        self.window_total = 0

        # Drift events
        self.drift_events: List[Dict] = []
        self.bar_controller = BARController()

    def evaluate_stream(self, df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Evaluate model on streaming data."""
        print(f"Evaluating on {len(df):,} records...")

        start_time = time.time()

        for idx, row in df.iterrows():
            record = row.to_dict()
            label = labels[idx]

            # Extract features
            t0 = time.time()
            features = self.vectorizer.transform(record)

            # Score
            try:
                score = self.model.score_one(features)
                is_anomaly = score > self.model.max_thres.item()
            except RuntimeError:
                score = 0.0
                is_anomaly = False

            # Track latency
            latency = (time.time() - t0) * 1000  # ms
            self.latency_window.append(latency)

            # BAR controller - determine if memory should update
            neighborhood = self._extract_neighborhood(record)
            should_update, reason = self.bar_controller.should_update_memory(neighborhood, score)

            if should_update:
                self.model.memory_update(features)

            # Record prediction
            self.scores.append(score)
            self.predictions.append(1 if is_anomaly else 0)
            self.labels.append(int(label))
            self.timestamps.append(time.time())

            # Update window metrics
            self._update_window_metrics(label, 1 if is_anomaly else 0)

            # Check for drift events
            if reason == 'drift_detected':
                self.drift_events.append({
                    'timestamp': time.time(),
                    'neighborhood': neighborhood,
                    'score': score,
                    'index': idx
                })

            # Progress
            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1:,} / {len(df):,} records...")

        total_time = time.time() - start_time

        # Compute final metrics
        results = self._compute_metrics()
        results['timing'] = {
            'total_time_seconds': total_time,
            'records_per_second': len(df) / total_time if total_time > 0 else 0,
        }

        return results

    def _extract_neighborhood(self, record: Dict) -> str:
        """Extract neighborhood from record."""
        zone_id = int(float(record.get('PULocationID', 1)))
        if zone_id <= 50:
            return 'manhattan'
        elif zone_id <= 100:
            return 'brooklyn'
        elif zone_id <= 150:
            return 'queens'
        elif zone_id <= 200:
            return 'bronx'
        elif zone_id in [132, 138]:
            return 'airport'
        else:
            return 'staten_island'

    def _update_window_metrics(self, label: int, prediction: int):
        """Update running confusion matrix."""
        self.window_total += 1

        if prediction == 1 and label == 1:
            self.window_tp += 1
        elif prediction == 1 and label == 0:
            self.window_fp += 1
        elif prediction == 0 and label == 0:
            self.window_tn += 1
        else:  # prediction == 0 and label == 1
            self.window_fn += 1

    def _compute_metrics(self) -> Dict:
        """Compute evaluation metrics."""
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        scores = np.array(self.scores)

        # Confusion matrix elements
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        # Core metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Latency percentiles
        latencies = np.array(self.latency_window)
        latency_p50 = np.percentile(latencies, 50) if len(latencies) > 0 else 0
        latency_p95 = np.percentile(latencies, 95) if len(latencies) > 0 else 0
        latency_p99 = np.percentile(latencies, 99) if len(latencies) > 0 else 0

        # BAR statistics
        bar_stats = self.bar_controller.get_stats()

        return {
            'confusion_matrix': {
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn),
            },
            'metrics': {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'fpr': float(fpr),  # False Positive Rate
                'total_records': int(len(predictions)),
                'anomaly_rate': float(np.mean(labels)),
            },
            'latency': {
                'p50_ms': float(latency_p50),
                'p95_ms': float(latency_p95),
                'p99_ms': float(latency_p99),
                'mean_ms': float(np.mean(latencies)) if len(latencies) > 0 else 0,
            },
            'bar_controller': bar_stats,
            'drift_events': {
                'total': len(self.drift_events),
                'events': self.drift_events[:100],  # First 100 events
            },
            'score_statistics': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'p50': float(np.percentile(scores, 50)),
                'p95': float(np.percentile(scores, 95)),
                'p99': float(np.percentile(scores, 99)),
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Streaming Evaluation')
    parser.add_argument('--data', type=str, required=True, help='Test CSV path')
    parser.add_argument('--model', type=str, required=True, help='Model path (.pt)')
    parser.add_argument('--signing-key', type=str, default='training-signing-key',
                        help='HMAC signing key')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path')
    parser.add_argument('--n-anomalies', type=int, default=1000,
                        help='Number of anomalies to inject')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    set_determinism(args.seed)

    print("=" * 60)
    print("STREAMING EVALUATION")
    print("=" * 60)

    # Load data
    print(f"\n[1] Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  Loaded {len(df):,} records")

    # Inject anomalies
    print(f"\n[2] Injecting {args.n_anomalies:,} anomalies...")
    df_anom, labels = inject_anomalies(df, n_anomalies=args.n_anomalies, seed=args.seed)
    print(f"  Total records: {len(df_anom):,}")
    print(f"  Anomalies: {int(np.sum(labels)):,} ({np.mean(labels)*100:.2f}%)")

    # Load model
    print(f"\n[3] Loading model from {args.model}...")
    model = MemStreamCore.load(args.model, signing_key=args.signing_key)
    print(f"  Model loaded: in_dim={model.cfg.in_dim}")
    print(f"  Beta threshold: {model.max_thres.item():.4f}")

    # Initialize vectorizer
    vectorizer = FeatureVectorizer()

    # Run streaming evaluation
    print("\n[4] Running streaming evaluation...")
    evaluator = StreamingEvaluator(model, vectorizer)
    results = evaluator.evaluate_stream(df_anom, labels)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nCore Metrics:")
    print(f"  Precision: {results['metrics']['precision']:.4f}")
    print(f"  Recall:    {results['metrics']['recall']:.4f}")
    print(f"  F1:        {results['metrics']['f1']:.4f}")
    print(f"  FPR:       {results['metrics']['fpr']:.4f}")

    print(f"\nLatency:")
    print(f"  p50: {results['latency']['p50_ms']:.2f} ms")
    print(f"  p95: {results['latency']['p95_ms']:.2f} ms")
    print(f"  p99: {results['latency']['p99_ms']:.2f} ms")

    print(f"\nBAR Controller:")
    print(f"  BAR Rate: {results['bar_controller']['bar_rate_pct']:.2f}%")
    print(f"  Drift Events: {results['drift_events']['total']}")

    print(f"\nTiming:")
    print(f"  Total Time: {results['timing']['total_time_seconds']:.2f} s")
    print(f"  Records/sec: {results['timing']['records_per_second']:.0f}")

    # Save results
    print(f"\n[5] Saving results to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {output_path}")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
