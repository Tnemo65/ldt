"""
CA-DQStream: Concept Drift Detection Benchmark
=============================================
Script này chạy benchmark offline để test tất cả các drift cases
và kiểm tra xem ADWIN và IEC có phát hiện đúng không.

Test Cases:
1. ABRUPT_DRIFT    - ADWIN should detect within 1-2 windows
2. GRADUAL_DRIFT   - ADWIN should detect after multiple windows
3. TRANSIENT_DRIFT - ADWIN may miss if too fast
4. RECURRING_DRIFT - ADWIN adapts but delta_score stays high
5. FEATURE_DRIFT   - Detection depends on feature importance
6. LABEL_DRIFT     - ADWIN should detect via anomaly_rate
7. DISTRIBUTION_SHIFT - ADWIN detects on multiple metrics

Expected IEC Responses:
- ABRUPT_DRIFT: switch_model (HIGH severity)
- GRADUAL_DRIFT: retrain_model (MODERATE)
- TRANSIENT_DRIFT: do_nothing (recover quickly)
- RECURRING_DRIFT: adjust_threshold (recurring pattern)
- FEATURE_DRIFT: retrain_model (MODERATE-HIGH)
- LABEL_DRIFT: retrain_model (HIGH)
- DISTRIBUTION_SHIFT: switch_model (HIGH)

Usage:
    python scripts/benchmark_drift_detection.py
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from river.drift import ADWIN

# ─── ADWIN Configuration (same as production) ───────────────────────────────────

ADWIN_DELTA_CONFIG = {
    'volume': 0.005,
    'null_rate': 0.001,
    'violation_rate': 0.002,
    'anomaly_rate': 0.002,
    'avg_anomaly_score': 0.003,
    'delta_score': 0.002,
}


class DriftDetector:
    """ADWIN-based drift detector matching production config."""

    def __init__(self, delta_config=None):
        self.delta_config = delta_config or ADWIN_DELTA_CONFIG
        self.adwins = {}
        for metric, delta in self.delta_config.items():
            self.adwins[metric] = ADWIN(delta=delta)

    def update(self, metrics: dict) -> Tuple[bool, List[str]]:
        """Update all ADWINs with metrics. Returns (drift_detected, affected_metrics)."""
        affected = []
        for metric_name, adwin in self.adwins.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if adwin.update(value):
                    affected.append(metric_name)
        return len(affected) > 0, affected


# ─── Drift Simulation Data ─────────────────────────────────────────────────────

def generate_normal_data(n_records: int, seed: int = 42) -> pd.DataFrame:
    """Generate normal NYC taxi data."""
    np.random.seed(seed)

    records = []
    for i in range(n_records):
        hour = (i // 100) % 24
        records.append({
            'record_id': i,
            'hour': hour,
            'fare_amount': np.random.normal(15, 5),
            'trip_distance': np.random.exponential(3),
            'passenger_count': np.random.choice([1, 2, 3, 4]),
            'neighborhood': np.random.choice(['manhattan', 'brooklyn', 'queens']),
        })
    return pd.DataFrame(records)


def simulate_abrupt_drift(df: pd.DataFrame, window_start: int, window_size: int,
                          multiplier: float = 10.0) -> pd.DataFrame:
    """Simulate abrupt drift: fare jumps 10x immediately."""
    df = df.copy()
    mask = (df['record_id'] >= window_start) & (df['record_id'] < window_start + window_size)
    df.loc[mask, 'fare_amount'] *= multiplier
    return df


def simulate_gradual_drift(df: pd.DataFrame, window_start: int, window_size: int,
                           max_multiplier: float = 2.5) -> pd.DataFrame:
    """Simulate gradual drift: fare increases linearly over time."""
    df = df.copy()
    for i, row_idx in enumerate(df.index):
        record_id = df.loc[row_idx, 'record_id']
        if window_start <= record_id < window_start + window_size:
            progress = (record_id - window_start) / window_size
            multiplier = 1.0 + (max_multiplier - 1.0) * progress
            df.loc[row_idx, 'fare_amount'] *= multiplier
    return df


def simulate_transient_drift(df: pd.DataFrame, window_start: int, window_size: int,
                             multiplier: float = 5.0) -> pd.DataFrame:
    """Simulate transient drift: spike in middle, fade at edges."""
    df = df.copy()
    for i, row_idx in enumerate(df.index):
        record_id = df.loc[row_idx, 'record_id']
        if window_start <= record_id < window_start + window_size:
            progress = (record_id - window_start) / window_size
            # Spike in middle 20%-80%
            if 0.2 <= progress <= 0.8:
                df.loc[row_idx, 'fare_amount'] *= multiplier
    return df


def simulate_recurring_drift(df: pd.DataFrame, cycles: List[Tuple[int, int]],
                              multiplier: float = 3.0) -> pd.DataFrame:
    """Simulate recurring drift: drift appears in cycles."""
    df = df.copy()
    for start, size in cycles:
        for i, row_idx in enumerate(df.index):
            record_id = df.loc[row_idx, 'record_id']
            if start <= record_id < start + size:
                df.loc[row_idx, 'fare_amount'] *= multiplier
    return df


def simulate_feature_drift(df: pd.DataFrame, window_start: int, window_size: int,
                            multiplier: float = 5.0) -> pd.DataFrame:
    """Simulate feature drift: only fare changes, distance stays same."""
    df = df.copy()
    mask = (df['record_id'] >= window_start) & (df['record_id'] < window_start + window_size)
    df.loc[mask, 'fare_amount'] *= multiplier
    # Note: trip_distance stays the same → ratio drift
    return df


def simulate_label_drift(df: pd.DataFrame, window_start: int, window_size: int,
                         fraud_rate: float = 0.5) -> pd.DataFrame:
    """Simulate label drift: inject more anomalies (high fare for short trips)."""
    df = df.copy()
    np.random.seed(42)
    mask = (df['record_id'] >= window_start) & (df['record_id'] < window_start + window_size)
    anomaly_mask = mask & (np.random.random(mask.sum()) < fraud_rate)
    df.loc[anomaly_mask, 'trip_distance'] = np.random.uniform(0.3, 1.0, anomaly_mask.sum())
    df.loc[anomaly_mask, 'fare_amount'] = np.random.uniform(40, 80, anomaly_mask.sum())
    return df


def simulate_distribution_shift(df: pd.DataFrame, window_start: int, window_size: int,
                                 target_neighborhood: str = 'airport') -> pd.DataFrame:
    """Simulate distribution shift: neighborhood distribution changes."""
    df = df.copy()
    mask = (df['record_id'] >= window_start) & (df['record_id'] < window_start + window_size)
    df.loc[mask, 'neighborhood'] = target_neighborhood
    df.loc[mask, 'fare_amount'] *= 2.0  # Airport premium
    df.loc[mask, 'trip_distance'] = np.random.uniform(15, 35, mask.sum())
    return df


# ─── Metrics Computation ────────────────────────────────────────────────────────

def compute_window_metrics(df: pd.DataFrame, window_start: int, window_size: int,
                           anomaly_threshold: float = 15.0) -> dict:
    """Compute meta-metrics for a window (matching MetaAggregator)."""
    mask = (df['record_id'] >= window_start) & (df['record_id'] < window_start + window_size)
    window = df[mask]

    if len(window) == 0:
        return None

    volume = len(window)

    # Null rate (simulated)
    null_rate = 0.0  # No nulls in our simulation

    # Violation rate (fare > 500)
    violation_rate = (window['fare_amount'] > 500).mean()

    # Anomaly rate (fare > threshold)
    anomaly_rate = (window['fare_amount'] > anomaly_threshold).mean()

    # Avg anomaly score (higher = more anomalous)
    avg_anomaly_score = window['fare_amount'].mean() / 100.0

    # Delta score (violation_rate vs anomaly_rate divergence)
    delta_score = abs(violation_rate - anomaly_rate) / (violation_rate + anomaly_rate + 1e-6)

    return {
        'volume': volume,
        'null_rate': null_rate,
        'violation_rate': violation_rate,
        'anomaly_rate': anomaly_rate,
        'avg_anomaly_score': avg_anomaly_score,
        'delta_score': delta_score,
    }


# ─── Drift Detection Logic (matching IEC) ──────────────────────────────────────

def assess_severity(n_recent_drifts: int, drift_threshold: int = 3) -> str:
    """Assess drift severity (matching DriftAggregator)."""
    if n_recent_drifts == 0:
        return 'none'
    elif n_recent_drifts < drift_threshold:
        return 'low'
    elif n_recent_drifts < drift_threshold * 2:
        return 'moderate'
    else:
        return 'high'


def predict_strategy(metrics: dict, severity: str) -> str:
    """Predict IEC strategy (matching IECOperator fallback)."""
    if severity == 'none':
        return 'do_nothing'
    elif severity == 'low':
        return 'adjust_threshold'
    elif severity == 'moderate':
        return 'retrain_model'
    else:
        return 'switch_model'


# ─── Benchmark Runner ───────────────────────────────────────────────────────────

def run_drift_benchmark():
    """Run comprehensive drift detection benchmark."""

    print("="*80)
    print("CA-DQSTREAM: CONCEPT DRIFT DETECTION BENCHMARK")
    print("="*80)
    print("\nADWIN Configuration:")
    for metric, delta in ADWIN_DELTA_CONFIG.items():
        sensitivity = "HIGH" if delta <= 0.001 else "MEDIUM" if delta <= 0.003 else "LOW"
        print(f"  {metric}: delta={delta} ({sensitivity})")

    # Generate base data
    print("\n[1/7] Generating base data...")
    n_records = 2000
    df_base = generate_normal_data(n_records, seed=42)

    # Define test cases
    test_cases = [
        {
            'name': 'ABRUPT_DRIFT',
            'description': 'Sudden 10x fare increase',
            'window_start': 500,
            'window_size': 300,
            'apply_drift': lambda df: simulate_abrupt_drift(df, 500, 300, 10.0),
            'expected_severity': 'high',
            'expected_strategy': 'switch_model',
            'expected_detection_window': '1-2 windows',
        },
        {
            'name': 'GRADUAL_DRIFT',
            'description': 'Fare increases linearly 2.5x over 600 records',
            'window_start': 500,
            'window_size': 600,
            'apply_drift': lambda df: simulate_gradual_drift(df, 500, 600, 2.5),
            'expected_severity': 'moderate',
            'expected_strategy': 'retrain_model',
            'expected_detection_window': '5-10 windows',
        },
        {
            'name': 'TRANSIENT_DRIFT',
            'description': 'Short spike in middle, fade at edges',
            'window_start': 500,
            'window_size': 200,
            'apply_drift': lambda df: simulate_transient_drift(df, 500, 200, 5.0),
            'expected_severity': 'low',
            'expected_strategy': 'do_nothing',
            'expected_detection_window': 'may miss if too fast',
        },
        {
            'name': 'RECURRING_DRIFT',
            'description': 'Drift appears in 3 cycles',
            'window_start': 500,
            'window_size': 600,
            'apply_drift': lambda df: simulate_recurring_drift(
                df, [(500, 80), (680, 80), (860, 80)], 3.0),
            'expected_severity': 'moderate',
            'expected_strategy': 'retrain_model',
            'expected_detection_window': 'recurring pattern',
        },
        {
            'name': 'FEATURE_DRIFT',
            'description': 'Only fare changes (ratio drift)',
            'window_start': 500,
            'window_size': 300,
            'apply_drift': lambda df: simulate_feature_drift(df, 500, 300, 5.0),
            'expected_severity': 'moderate-high',
            'expected_strategy': 'retrain_model',
            'expected_detection_window': '2-5 windows',
        },
        {
            'name': 'LABEL_DRIFT',
            'description': '50% of records become fraud',
            'window_start': 500,
            'window_size': 400,
            'apply_drift': lambda df: simulate_label_drift(df, 500, 400, 0.5),
            'expected_severity': 'high',
            'expected_strategy': 'retrain_model',
            'expected_detection_window': '2-3 windows',
        },
        {
            'name': 'DISTRIBUTION_SHIFT',
            'description': 'All traffic shifts to airport distribution',
            'window_start': 500,
            'window_size': 400,
            'apply_drift': lambda df: simulate_distribution_shift(df, 500, 400),
            'expected_severity': 'high',
            'expected_strategy': 'switch_model',
            'expected_detection_window': '1-3 windows',
        },
    ]

    results = []

    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"TEST: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"{'='*80}")

        # Apply drift
        df = test_case['apply_drift'](df_base.copy())

        # Initialize detector
        detector = DriftDetector()
        recent_drifts = []
        drift_history = []

        window_size = 100  # 1-minute window equivalent
        n_windows = n_records // window_size

        drift_detected = False
        detection_window = None
        affected_metrics = []
        strategies_by_window = []

        for w in range(n_windows):
            window_start = w * window_size
            metrics = compute_window_metrics(df, window_start, window_size)

            if metrics:
                has_drift, affected = detector.update(metrics)

                if has_drift:
                    drift_detected = True
                    if detection_window is None:
                        detection_window = w

                    for metric in affected:
                        recent_drifts.append({
                            'window': w,
                            'metric': metric,
                            'value': metrics[metric],
                        })
                        affected_metrics.append(metric)

                # Assess severity and predict strategy
                severity = assess_severity(len(recent_drifts))
                strategy = predict_strategy(metrics, severity)
                strategies_by_window.append((w, severity, strategy))

        # Summarize results
        result = {
            'test_name': test_case['name'],
            'drift_detected': drift_detected,
            'detection_window': detection_window,
            'affected_metrics': list(set(affected_metrics)),
            'total_drifts': len(recent_drifts),
            'expected_severity': test_case['expected_severity'],
            'expected_strategy': test_case['expected_strategy'],
        }

        # Determine final strategy (from last window)
        if strategies_by_window:
            final_severity = strategies_by_window[-1][1]
            final_strategy = strategies_by_window[-1][2]
            result['final_severity'] = final_severity
            result['final_strategy'] = final_strategy
            result['correct_severity'] = final_severity in test_case['expected_severity']
            result['correct_strategy'] = final_strategy == test_case['expected_strategy']
        else:
            result['final_severity'] = 'none'
            result['final_strategy'] = 'do_nothing'
            result['correct_severity'] = True
            result['correct_strategy'] = True

        results.append(result)

        # Print summary
        print(f"\nResults:")
        print(f"  Drift Detected: {'YES' if drift_detected else 'NO'}")
        if detection_window:
            print(f"  Detection Window: {detection_window} (of {n_windows} windows)")
        print(f"  Affected Metrics: {result['affected_metrics']}")
        print(f"  Total Drift Events: {len(recent_drifts)}")
        print(f"  Final Severity: {result['final_severity']}")
        print(f"  Final Strategy: {result['final_strategy']}")
        print(f"\nExpected:")
        check_sev = "[OK]" if result['correct_severity'] else "[FAIL]"
        check_str = "[OK]" if result['correct_strategy'] else "[FAIL]"
        print(f"  Severity: {test_case['expected_severity']} -> Got: {result['final_severity']} {check_sev}")
        print(f"  Strategy: {test_case['expected_strategy']} -> Got: {result['final_strategy']} {check_str}")

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # Calculate accuracy
    total = len(results)
    severity_correct = sum(1 for r in results if r['correct_severity'])
    strategy_correct = sum(1 for r in results if r['correct_strategy'])
    detected = sum(1 for r in results if r['drift_detected'])

    print(f"\n\nAccuracy:")
    print(f"  Drift Detection Rate: {detected}/{total} ({detected/total*100:.1f}%)")
    print(f"  Severity Prediction: {severity_correct}/{total} ({severity_correct/total*100:.1f}%)")
    print(f"  Strategy Prediction: {strategy_correct}/{total} ({strategy_correct/total*100:.1f}%)")

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'drift_benchmark_results.csv'
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return df_results


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = run_drift_benchmark()
    print("\n=== BENCHMARK COMPLETE ===")
