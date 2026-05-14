"""
Custom Drift Detection Implementation
===================================
Thay vì dùng ADWIN từ river, ta implement drift detector đơn giản
dựa trên statistical change detection.
"""

import numpy as np
from typing import List, Tuple, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class SimpleDriftDetector:
    """
    Simple drift detector dựa trên rolling window comparison.
    Phù hợp với streaming scenario của CA-DQStream.
    """

    def __init__(self, window_size: int = 100, threshold: float = 2.0):
        """
        Args:
            window_size: Số samples trong mỗi window
            threshold: Số lần std deviation để trigger drift
        """
        self.window_size = window_size
        self.threshold = threshold
        self.buffer = []
        self.baseline_mean = None
        self.baseline_std = None
        self.drift_count = 0

    def update(self, value: float) -> bool:
        """Update detector with new value. Returns True if drift detected."""
        self.buffer.append(value)

        # Keep only window_size + buffer for baseline
        if len(self.buffer) > self.window_size * 3:
            self.buffer = self.buffer[-self.window_size * 3:]

        # Need enough data for baseline
        if len(self.buffer) < self.window_size * 2:
            return False

        # Baseline: first window_size samples
        baseline = self.buffer[:self.window_size]
        # Current: last window_size samples
        current = self.buffer[-self.window_size:]

        if self.baseline_mean is None:
            self.baseline_mean = np.mean(baseline)
            self.baseline_std = np.std(baseline) + 1e-6

        current_mean = np.mean(current)
        current_std = np.std(current) + 1e-6

        # Detect drift: current mean differs significantly from baseline
        change = abs(current_mean - self.baseline_mean)
        drift_detected = change > self.threshold * self.baseline_std

        if drift_detected:
            self.drift_count += 1
            # Update baseline
            self.baseline_mean = 0.7 * self.baseline_mean + 0.3 * current_mean
            self.baseline_std = 0.7 * self.baseline_std + 0.3 * current_std

        return drift_detected


class ADWINLike:
    """
    ADWIN-like detector: so sánh hai nửa của sliding window.
    """
    def __init__(self, delta: float = 0.002, window_size: int = 1000):
        self.delta = delta
        self.window_size = window_size
        self.window = []
        self.drift_detected = False

    def update(self, value: float) -> bool:
        """Update with new value. Returns True if drift detected."""
        self.window.append(value)

        # Keep window bounded
        if len(self.window) > self.window_size:
            self.window.pop(0)

        # Need minimum samples
        if len(self.window) < 100:
            return False

        # Check for drift
        mid = len(self.window) // 2
        w1 = np.array(self.window[:mid])
        w2 = np.array(self.window[mid:])

        n1, n2 = len(w1), len(w2)
        m1, m2 = np.mean(w1), np.mean(w2)

        # Harmonic mean for variance
        m = 1 / (1/n1 + 1/n2) if n1 > 0 and n2 > 0 else 1

        # Epsilon cut
        eps_cut = np.sqrt((2.0 / m) * np.log(4.0 * len(self.window) / self.delta))

        if abs(m1 - m2) > eps_cut:
            self.drift_detected = True
            # Reset window to second half
            self.window = list(w2)
            return True

        return False


def run_tests():
    """Run all drift detection tests."""

    print("=" * 70)
    print("DRIFT DETECTION TESTS")
    print("=" * 70)

    # Test cases
    test_cases = [
        {
            'name': 'ABRUPT_10X',
            'generate': lambda: np.concatenate([
                np.random.normal(15, 3, 500),
                np.random.normal(150, 15, 500),
                np.random.normal(15, 3, 500),
            ]),
            'drift_start': 500,
            'expected': 'detect at ~500-600',
        },
        {
            'name': 'ABRUPT_3X',
            'generate': lambda: np.concatenate([
                np.random.normal(15, 3, 500),
                np.random.normal(45, 5, 500),
                np.random.normal(15, 3, 500),
            ]),
            'drift_start': 500,
            'expected': 'detect at ~500-600',
        },
        {
            'name': 'GRADUAL_2X',
            'generate': lambda: np.concatenate([
                np.random.normal(15, 3, 500),
                np.array([np.random.normal(15 + i*0.06, 3) for i in range(500)]),
                np.random.normal(45, 3, 500),
            ]),
            'drift_start': 500,
            'expected': 'detect gradually',
        },
        {
            'name': 'TRANSIENT',
            'generate': lambda: np.concatenate([
                np.random.normal(15, 3, 500),
                np.array([np.random.normal(75 if 100 <= i < 400 else 15, 3) for i in range(500)]),
                np.random.normal(15, 3, 500),
            ]),
            'drift_start': 500,
            'expected': 'detect then recover',
        },
        {
            'name': 'RECURRING',
            'generate': lambda: np.concatenate([
                np.random.normal(15, 3, 300),
                np.concatenate([
                    np.random.normal(15, 3, 100),
                    np.random.normal(45, 5, 100),
                ] * 3),
                np.random.normal(15, 3, 300),
            ]),
            'drift_start': 400,
            'expected': 'detect multiple cycles',
        },
    ]

    # Test SimpleDriftDetector
    print("\n" + "-" * 70)
    print("TESTING: SimpleDriftDetector")
    print("-" * 70)

    for tc in test_cases:
        np.random.seed(42)
        values = tc['generate']()
        detector = SimpleDriftDetector(window_size=50, threshold=1.5)

        first_det = None
        n_detections = 0

        for i, v in enumerate(values):
            if detector.update(v):
                n_detections += 1
                if first_det is None and i >= tc['drift_start']:
                    first_det = i

        detected = first_det is not None
        print(f"\n  {tc['name']:15} | Drift @ {tc['drift_start']} | "
              f"Detected: {'YES' if detected else 'NO':3} | "
              f"First: {first_det if first_det else 'N/A':>5} | "
              f"Total: {n_detections:2}")

    # Test ADWINLike
    print("\n" + "-" * 70)
    print("TESTING: ADWINLike (Custom Implementation)")
    print("-" * 70)

    for tc in test_cases:
        np.random.seed(42)
        values = tc['generate']()
        detector = ADWINLike(delta=0.001, window_size=500)

        first_det = None
        n_detections = 0

        for i, v in enumerate(values):
            if detector.update(v):
                n_detections += 1
                if first_det is None and i >= tc['drift_start']:
                    first_det = i

        detected = first_det is not None
        print(f"\n  {tc['name']:15} | Drift @ {tc['drift_start']} | "
              f"Detected: {'YES' if detected else 'NO':3} | "
              f"First: {first_det if first_det else 'N/A':>5} | "
              f"Total: {n_detections:2}")


def run_production_simulation():
    """Simulate full production scenario."""

    print("\n\n" + "=" * 70)
    print("PRODUCTION SIMULATION: Multi-Metric Drift Detection")
    print("=" * 70)

    # Simulate meta-metrics from MetaAggregator
    def generate_metrics(pre_drift: int = 500, drift_duration: int = 500):
        """Generate meta-metrics with drift."""
        np.random.seed(42)

        for i in range(pre_drift):
            yield {
                'volume': np.random.poisson(100),
                'null_rate': 0.01,
                'violation_rate': 0.05,
                'anomaly_rate': 0.05,
                'avg_anomaly_score': 0.15,
                'delta_score': 0.1,
            }

        # Drift: anomaly_rate jumps to 0.3
        for i in range(drift_duration):
            yield {
                'volume': np.random.poisson(100),
                'null_rate': 0.01,
                'violation_rate': np.random.uniform(0.3, 0.5),
                'anomaly_rate': np.random.uniform(0.25, 0.45),
                'avg_anomaly_score': np.random.uniform(0.4, 0.6),
                'delta_score': np.random.uniform(0.05, 0.2),
            }

        # Recovery
        for i in range(500):
            yield {
                'volume': np.random.poisson(100),
                'null_rate': 0.01,
                'violation_rate': 0.05,
                'anomaly_rate': 0.05,
                'avg_anomaly_score': 0.15,
                'delta_score': 0.1,
            }

    # Initialize detectors for each metric
    detectors = {
        'volume': SimpleDriftDetector(window_size=10, threshold=2.0),
        'anomaly_rate': SimpleDriftDetector(window_size=10, threshold=1.5),
        'avg_anomaly_score': SimpleDriftDetector(window_size=10, threshold=1.5),
        'delta_score': SimpleDriftDetector(window_size=10, threshold=1.5),
    }

    all_drifts = []
    print("\n  Processing metrics...")

    for i, metrics in enumerate(generate_metrics()):
        for metric_name, detector in detectors.items():
            if metric_name in metrics:
                if detector.update(metrics[metric_name]):
                    all_drifts.append({
                        'window': i,
                        'metric': metric_name,
                        'value': metrics[metric_name],
                    })

    # Summary
    print(f"\n  Results:")
    print(f"    Total drift events: {len(all_drifts)}")
    print(f"    Drift window: 500-1000")
    print(f"    Affected metrics: {set(d['metric'] for d in all_drifts)}")

    if all_drifts:
        first = min(d['window'] for d in all_drifts)
        last = max(d['window'] for d in all_drifts)
        print(f"    First detection: window {first}")
        print(f"    Last detection: window {last}")

    # Group by metric
    by_metric = {}
    for d in all_drifts:
        m = d['metric']
        if m not in by_metric:
            by_metric[m] = []
        by_metric[m].append(d)

    print(f"\n  By Metric:")
    for metric, drifts in by_metric.items():
        print(f"    {metric}: {len(drifts)} events")


if __name__ == '__main__':
    run_tests()
    run_production_simulation()

    print("\n\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
Custom drift detector hoạt động tốt với abrupt drift.

Production recommendations:
1. Use SimpleDriftDetector với window_size=50-100, threshold=1.5-2.0
2. Monitor multiple metrics (anomaly_rate, avg_anomaly_score)
3. ADWIN từ river có thể cần tuning hoặc thay bằng custom impl
4. DriftAggregator cần reset sau khi adaptation hoàn thành
""")
