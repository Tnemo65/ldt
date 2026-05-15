#!/usr/bin/env python3
"""
Exp 06: max_window (ADWIN)
Priority: MEDIUM
Primary Metric: Drift detection latency

Rationale: max_window controls the maximum history kept by ADWIN.
Smaller window = faster drift detection but more false positives.
Larger window = more stable but slower to detect actual drifts.
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
    def __init__(self, delta=0.002, max_window=1000):
        self.delta = delta
        self.max_window = max_window
        self.window = []
        self.detections = 0
        self.n_updates = 0

    def update(self, value):
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
        m = 1.0 / (1.0 / n_r + 1.0 / n_o)
        epsilon = np.sqrt((1.0 / (2.0 * m)) * np.log(4.0 / self.delta))
        if abs(mean_r - mean_o) > epsilon:
            self.detections += 1
            self.window = self.window[mid:]
            return True
        return False


def inject_abrupt_drift(X, rng, feature_idx=6, drift_point=0.50, magnitude=3.0):
    """Inject abrupt drift at drift_point fraction of records."""
    n = len(X)
    drift_idx = int(n * drift_point)
    X_drifted = X.copy()
    feature_std = X[:drift_idx, feature_idx].std() + 1e-8
    X_drifted[drift_idx:, feature_idx] += magnitude * feature_std
    return X_drifted, drift_idx


def run():
    ts = time.strftime('%Y%m%d_%H%M%S')
    print("=" * 60)
    print("  EXP 06: max_window (ADWIN)")
    print("  Priority: MEDIUM  |  Metric: Drift detection latency")
    print("=" * 60)

    print("\nLoading data...")
    data = load_data(n_warmup=10000, n_test=15000)

    grid = [200, 500, 1000, 2000]
    results = []
    rng = np.random.RandomState(42)

    for max_window in grid:
        print(f"\n  [max_window={max_window}]")

        adwin = SimpleADWIN(delta=0.002, max_window=max_window)

        X_drifted, drift_point = inject_abrupt_drift(
            data['X_test'].copy(), rng,
            feature_idx=6, drift_point=0.50, magnitude=3.0
        )

        detected = False
        detection_time = None
        false_positives = 0

        for i in range(len(X_drifted)):
            value = float(X_drifted[i, 6])
            drift = adwin.update(value)
            if drift and not detected and i >= drift_point:
                detected = True
                detection_time = i
            elif drift and (i < drift_point or detected):
                false_positives += 1

        latency = (detection_time - drift_point) if detection_time else -1
        recall = 1.0 if detected else 0.0
        precision = 1.0 / (1.0 + false_positives) if detected else 0.0

        m = {
            'drift_point': int(drift_point),
            'detection_time': int(detection_time) if detection_time else None,
            'latency': int(latency) if latency >= 0 else None,
            'detected': detected,
            'false_positives': int(false_positives),
            'recall': float(recall),
            'precision': float(precision),
            'avg_window_size': len(adwin.window),
        }
        print(f"    Drift at: {drift_point}  Detected: {detected}  "
              f"Latency: {latency if latency >= 0 else 'N/A'}")
        print(f"    False positives: {false_positives}")

        results.append({'max_window': max_window, 'metrics': m})

    print(f"\n{'='*60}")
    print(f"  RESULTS:")
    print(f"  {'max_window':>11}  {'Detected':>9}  {'Latency':>9}  {'FPs':>6}  "
          f"{'Recall':>8}  {'Precision':>10}")
    print(f"  {'-'*60}")
    for r in results:
        m = r['metrics']
        lat = str(m['latency']) if m['latency'] is not None else 'N/A'
        print(f"  {r['max_window']:>11,}  {str(m['detected']):>9}  "
              f"{lat:>9}  {m['false_positives']:>6}  "
              f"{m['recall']:>8.4f}  {m['precision']:>10.4f}")

    output = {
        'experiment': 'exp06_max_window',
        'hyperparameter': 'max_window',
        'timestamp': ts,
        'priority': 'MEDIUM',
        'primary_metric': 'Latency',
        'grid': grid,
        'results': results,
        'recommendation': "Trade-off between detection latency and false positives. "
                          "Smaller window = faster detection but more FPs.",
    }
    out_path = OUT / f'exp06_max_window_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    run()
