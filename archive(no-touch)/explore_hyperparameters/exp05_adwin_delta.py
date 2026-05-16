#!/usr/bin/env python3
"""
Exp 05: ADWIN Delta
Priority: HIGH
Primary Metric: Precision/Recall on drift injection

Rationale: ADWIN delta controls how sensitive the drift detector is.
Low delta (0.001): Very sensitive, detects subtle drifts quickly.
High delta (0.02): Conservative, only triggers on clear drifts.
"""

import sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np

ROOT = Path(__file__).parent
OUT  = ROOT / 'results'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
from shared import load_data


class SimpleADWIN:
    """Simplified ADWIN for drift detection experiments."""

    def __init__(self, delta=0.002, max_window=1000):
        self.delta = delta
        self.max_window = max_window
        self.window = []
        self.detections = 0
        self.n_updates = 0

    def update(self, value):
        """Update ADWIN with a new value. Returns True if drift detected."""
        self.n_updates += 1
        self.window.append(value)
        if len(self.window) > self.max_window:
            self.window.pop(0)

        if len(self.window) < 50:
            return False

        mid = len(self.window) // 2
        recent = self.window[mid:]
        older  = self.window[:mid]

        mean_r = np.mean(recent)
        mean_o = np.mean(older)
        n_r, n_o = len(recent), len(older)

        # Hoeffding bound
        m = 1.0 / (1.0 / n_r + 1.0 / n_o)
        epsilon = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 / self.delta))

        if abs(mean_r - mean_o) > epsilon:
            self.detections += 1
            # Shrink window
            self.window = self.window[mid:]
            return True
        return False

    def reset(self):
        self.window = []
        self.detections = 0
        self.n_updates = 0

    def mean(self):
        return np.mean(self.window) if self.window else 0.0


def inject_feature_drift(X, rng, feature_idx=6, drift_start=0.50,
                          magnitude=3.0):
    """Inject gradual feature drift after drift_start fraction of records."""
    n = len(X)
    start_idx = int(n * drift_start)
    drift_idx = np.arange(start_idx, n)
    feature_std = X[:start_idx, feature_idx].std() + 1e-8

    X_drifted = X.copy()
    for i in drift_idx:
        progress = (i - start_idx) / (n - start_idx)
        X_drifted[i, feature_idx] += magnitude * progress * feature_std
    return X_drifted, start_idx


def run():
    ts = time.strftime('%Y%m%d_%H%M%S')
    print("=" * 60)
    print("  EXP 05: ADWIN Delta")
    print("  Priority: HIGH  |  Metric: Precision/Recall on drift injection")
    print("=" * 60)

    print("\nLoading data...")
    data = load_data(n_warmup=10000, n_test=15000)
    print(f"  Warmup: {data['n_warmup']:,}  Test: {data['n_test']:,}")

    # Grid: ADWIN deltas
    grid = [0.001, 0.002, 0.005, 0.01, 0.02]
    results = []

    rng = np.random.RandomState(42)

    for delta in grid:
        print(f"\n  [ADWIN delta={delta}]")

        # Create ADWIN instance
        adwin = SimpleADWIN(delta=delta, max_window=1000)

        # Create drifted test set
        X_drifted, drift_start = inject_feature_drift(
            data['X_test'].copy(), rng, feature_idx=6,
            drift_start=0.50, magnitude=3.0
        )

        # Track detection
        detected = False
        detection_time = None
        false_positives = 0

        for i in range(len(X_drifted)):
            # Use feature value as ADWIN input (simplified)
            value = float(X_drifted[i, 6])  # fare_per_mile feature
            drift = adwin.update(value)

            if drift and not detected and i >= drift_start:
                detected = True
                detection_time = i
            elif drift and (i < drift_start or detected):
                false_positives += 1

        # Compute metrics
        total_drift_time = len(X_drifted) - drift_start
        latency = (detection_time - drift_start) if detection_time else -1

        recall = 1.0 if detected else 0.0
        precision = 1.0 / (1.0 + false_positives) if detected else 0.0

        m = {
            'drift_start_idx': int(drift_start),
            'detection_time': int(detection_time) if detection_time else None,
            'latency': int(latency) if latency >= 0 else None,
            'detected': detected,
            'false_positives': int(false_positives),
            'total_adwin_updates': adwin.n_updates,
            'recall': float(recall),
            'precision': float(precision),
        }

        print(f"    Drift start: {drift_start}  Detected: {detected}  "
              f"Latency: {latency if latency >= 0 else 'N/A'}")
        print(f"    False positives: {false_positives}  Recall: {recall:.4f}  "
              f"Precision: {precision:.4f}")

        results.append({'delta': delta, 'metrics': m})

    best_recall = max(results, key=lambda r: r['metrics']['recall'])
    best_precision = max(results, key=lambda r: r['metrics']['precision'])

    print(f"\n{'='*60}")
    print(f"  RESULTS:")
    print(f"  {'delta':>8}  {'Detected':>9}  {'Latency':>9}  {'FPs':>6}  "
          f"{'Recall':>8}  {'Precision':>10}")
    print(f"  {'-'*60}")
    for r in results:
        m = r['metrics']
        lat = str(m['latency']) if m['latency'] is not None else 'N/A'
        print(f"  {r['delta']:>8.4f}  {str(m['detected']):>9}  "
              f"{lat:>9}  {m['false_positives']:>6}  "
              f"{m['recall']:>8.4f}  {m['precision']:>10.4f}")

    output = {
        'experiment': 'exp05_adwin_delta',
        'hyperparameter': 'adwin_delta',
        'timestamp': ts,
        'priority': 'HIGH',
        'primary_metric': 'Recall',
        'grid': grid,
        'results': results,
        'best_recall': best_recall,
        'best_precision': best_precision,
        'recommendation': f"delta={best_recall['delta']} gives best drift recall. "
                          f"For balanced precision/recall, use delta={best_precision['delta']}.",
    }
    out_path = OUT / f'exp05_adwin_delta_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    run()
