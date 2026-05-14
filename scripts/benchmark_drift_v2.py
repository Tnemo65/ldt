"""
CA-DQStream: Concept Drift Detection Benchmark (Enhanced)
=========================================================
Script này chạy benchmark offline để test tất cả các drift cases
với ADWIN được cấu hình nhạy hơn.

Test Cases:
1. ABRUPT_DRIFT    - ADWIN should detect within 1-2 windows
2. GRADUAL_DRIFT   - ADWIN should detect after multiple windows
3. TRANSIENT_DRIFT - ADWIN may miss if too fast
4. RECURRING_DRIFT - ADWIN adapts but delta_score stays high
5. FEATURE_DRIFT   - Detection depends on feature importance
6. LABEL_DRIFT     - ADWIN should detect via anomaly_rate
7. DISTRIBUTION_SHIFT - ADWIN detects on multiple metrics
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from river.drift import ADWIN

# Enhanced ADWIN config - more sensitive for testing
ADWIN_DELTA_CONFIG = {
    'volume': 0.01,         # Higher = more sensitive
    'null_rate': 0.005,     # Higher = more sensitive
    'violation_rate': 0.01, # Higher = more sensitive
    'anomaly_rate': 0.01,   # Higher = more sensitive
    'avg_anomaly_score': 0.01,  # Higher = more sensitive
    'delta_score': 0.01,    # Higher = more sensitive
}


class DriftDetector:
    """ADWIN-based drift detector."""

    def __init__(self, delta_config=None):
        self.delta_config = delta_config or ADWIN_DELTA_CONFIG
        self.adwins = {}
        for metric, delta in self.delta_config.items():
            self.adwins[metric] = ADWIN(delta=delta)
        self.drift_count = 0

    def update(self, metrics: dict) -> Tuple[bool, List[str]]:
        """Update all ADWINs with metrics."""
        affected = []
        for metric_name, adwin in self.adwins.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                drift = adwin.update(value)
                if drift:
                    affected.append(metric_name)
                    self.drift_count += 1
        return len(affected) > 0, affected

    def reset(self):
        """Reset all ADWIN instances."""
        for metric, delta in self.delta_config.items():
            self.adwins[metric] = ADWIN(delta=delta)
        self.drift_count = 0


# ─── Data Generation ────────────────────────────────────────────────────────────

def generate_normal_data(n_records: int, seed: int = 42) -> pd.DataFrame:
    """Generate normal NYC taxi data with consistent statistics."""
    np.random.seed(seed)

    records = []
    for i in range(n_records):
        records.append({
            'record_id': i,
            'fare_amount': np.random.normal(15, 3),  # mean=15, std=3
            'trip_distance': max(0.1, np.random.exponential(2.5)),
            'passenger_count': float(np.random.choice([1, 2, 3, 4])),
            'neighborhood': np.random.choice(['manhattan', 'brooklyn', 'queens', 'bronx']),
        })
    return pd.DataFrame(records)


def apply_abrupt_drift(df: pd.DataFrame, start: int, size: int, mult: float) -> pd.DataFrame:
    """Abrupt: fare jumps 10x immediately."""
    df = df.copy()
    mask = (df['record_id'] >= start) & (df['record_id'] < start + size)
    df.loc[mask, 'fare_amount'] *= mult
    return df


def apply_gradual_drift(df: pd.DataFrame, start: int, size: int, max_mult: float) -> pd.DataFrame:
    """Gradual: fare increases linearly from 1x to max_mult."""
    df = df.copy()
    for idx in df.index:
        rid = df.loc[idx, 'record_id']
        if start <= rid < start + size:
            progress = (rid - start) / size
            mult = 1.0 + (max_mult - 1.0) * min(progress, 1.0)
            df.loc[idx, 'fare_amount'] *= mult
    return df


def apply_transient_drift(df: pd.DataFrame, start: int, size: int, mult: float) -> pd.DataFrame:
    """Transient: spike in middle 20%-80%, fade at edges."""
    df = df.copy()
    for idx in df.index:
        rid = df.loc[idx, 'record_id']
        if start <= rid < start + size:
            progress = (rid - start) / size
            if 0.2 <= progress <= 0.8:
                df.loc[idx, 'fare_amount'] *= mult
    return df


def apply_recurring_drift(df: pd.DataFrame, cycles: List, mult: float) -> pd.DataFrame:
    """Recurring: drift appears in cycles."""
    df = df.copy()
    for start, size in cycles:
        for idx in df.index:
            rid = df.loc[idx, 'record_id']
            if start <= rid < start + size:
                df.loc[idx, 'fare_amount'] *= mult
    return df


def apply_feature_drift(df: pd.DataFrame, start: int, size: int, mult: float) -> pd.DataFrame:
    """Feature: only fare changes, creating ratio drift."""
    df = df.copy()
    mask = (df['record_id'] >= start) & (df['record_id'] < start + size)
    df.loc[mask, 'fare_amount'] *= mult
    return df


def apply_label_drift(df: pd.DataFrame, start: int, size: int, fraud_rate: float) -> pd.DataFrame:
    """Label: inject fraud (short trip, high fare)."""
    df = df.copy()
    np.random.seed(99)
    mask = (df['record_id'] >= start) & (df['record_id'] < start + size)
    indices = df[mask].index
    n_fraud = int(len(indices) * fraud_rate)
    fraud_indices = np.random.choice(indices, size=n_fraud, replace=False)
    df.loc[fraud_indices, 'trip_distance'] = np.random.uniform(0.3, 1.0, n_fraud)
    df.loc[fraud_indices, 'fare_amount'] = np.random.uniform(40, 80, n_fraud)
    return df


def apply_distribution_shift(df: pd.DataFrame, start: int, size: int) -> pd.DataFrame:
    """Distribution: neighborhood and fare both shift."""
    df = df.copy()
    mask = (df['record_id'] >= start) & (df['record_id'] < start + size)
    df.loc[mask, 'neighborhood'] = 'airport'
    df.loc[mask, 'fare_amount'] = df.loc[mask, 'fare_amount'] * 2.5
    return df


# ─── Metrics Computation ────────────────────────────────────────────────────────

def compute_window_metrics(window_df: pd.DataFrame, baseline_fare_mean: float = 15.0) -> dict:
    """Compute meta-metrics matching MetaAggregator output."""
    if len(window_df) == 0:
        return None

    volume = len(window_df)
    null_rate = 0.0

    # Violation rate: fare > 500 (canary rule)
    violation_rate = (window_df['fare_amount'] > 500).mean()

    # Anomaly rate: fare > 2x baseline
    threshold = baseline_fare_mean * 2
    anomaly_rate = (window_df['fare_amount'] > threshold).mean()

    # Avg anomaly score: normalized fare mean
    avg_anomaly_score = window_df['fare_amount'].mean() / 100.0

    # Delta score: divergence between violation and anomaly rate
    delta_score = abs(violation_rate - anomaly_rate) / (violation_rate + anomaly_rate + 1e-6)

    return {
        'volume': float(volume),
        'null_rate': null_rate,
        'violation_rate': float(violation_rate),
        'anomaly_rate': float(anomaly_rate),
        'avg_anomaly_score': float(avg_anomaly_score),
        'delta_score': float(delta_score),
    }


# ─── Severity Assessment (matching IEC) ────────────────────────────────────────

def assess_severity(n_drifts: int, threshold: int = 3) -> str:
    """Assess drift severity."""
    if n_drifts == 0:
        return 'none'
    elif n_drifts < threshold:
        return 'low'
    elif n_drifts < threshold * 2:
        return 'moderate'
    else:
        return 'high'


def predict_strategy(severity: str) -> str:
    """Predict IEC strategy."""
    strategies = {
        'none': 'do_nothing',
        'low': 'adjust_threshold',
        'moderate': 'retrain_model',
        'high': 'switch_model',
    }
    return strategies.get(severity, 'do_nothing')


# ─── Benchmark Runner ───────────────────────────────────────────────────────────

def run_benchmark():
    """Run comprehensive drift detection benchmark."""

    print("=" * 70)
    print("CA-DQSTREAM: CONCEPT DRIFT DETECTION BENCHMARK")
    print("=" * 70)

    print("\n[Config] ADWIN Delta (higher = more sensitive):")
    for k, v in ADWIN_DELTA_CONFIG.items():
        print(f"  {k}: {v}")

    # Base data
    n_records = 3000
    print(f"\n[1/7] Generating {n_records} base records...")
    df_base = generate_normal_data(n_records)

    # Test cases
    test_cases = [
        {
            'name': 'ABRUPT_DRIFT',
            'desc': 'Sudden 10x fare increase',
            'drift_start': 500,
            'drift_size': 300,
            'apply': lambda d: apply_abrupt_drift(d, 500, 300, 10.0),
            'expected_severity': 'high',
            'expected_strategy': 'switch_model',
        },
        {
            'name': 'GRADUAL_DRIFT',
            'desc': 'Fare increases linearly 3x over 600 records',
            'drift_start': 500,
            'drift_size': 600,
            'apply': lambda d: apply_gradual_drift(d, 500, 600, 3.0),
            'expected_severity': 'moderate',
            'expected_strategy': 'retrain_model',
        },
        {
            'name': 'TRANSIENT_DRIFT',
            'desc': 'Short spike in middle 20%-80%',
            'drift_start': 500,
            'drift_size': 200,
            'apply': lambda d: apply_transient_drift(d, 500, 200, 5.0),
            'expected_severity': 'low',
            'expected_strategy': 'do_nothing',
        },
        {
            'name': 'RECURRING_DRIFT',
            'desc': 'Drift in 3 cycles',
            'drift_start': 500,
            'drift_size': 500,
            'apply': lambda d: apply_recurring_drift(
                d, [(500, 80), (680, 80), (860, 80)], 3.0),
            'expected_severity': 'moderate',
            'expected_strategy': 'retrain_model',
        },
        {
            'name': 'FEATURE_DRIFT',
            'desc': 'Only fare changes (ratio drift)',
            'drift_start': 500,
            'drift_size': 300,
            'apply': lambda d: apply_feature_drift(d, 500, 300, 5.0),
            'expected_severity': 'moderate',
            'expected_strategy': 'retrain_model',
        },
        {
            'name': 'LABEL_DRIFT',
            'desc': '50% fraud injection (short trip, high fare)',
            'drift_start': 500,
            'drift_size': 400,
            'apply': lambda d: apply_label_drift(d, 500, 400, 0.5),
            'expected_severity': 'high',
            'expected_strategy': 'retrain_model',
        },
        {
            'name': 'DISTRIBUTION_SHIFT',
            'desc': 'All traffic shifts to airport (2.5x fare)',
            'drift_start': 500,
            'drift_size': 400,
            'apply': lambda d: apply_distribution_shift(d, 500, 400),
            'expected_severity': 'high',
            'expected_strategy': 'switch_model',
        },
    ]

    results = []

    for tc in test_cases:
        print(f"\n{'=' * 70}")
        print(f"TEST: {tc['name']}")
        print(f"DESC: {tc['desc']}")
        print(f"{'=' * 70}")

        # Apply drift
        df = tc['apply'](df_base.copy())

        # Reset detector
        detector = DriftDetector()
        all_drifts = []
        affected_metrics_set = set()

        window_size = 50  # 50 records per window
        n_windows = n_records // window_size

        first_detection_window = None
        strategies_by_window = []

        for w in range(n_windows):
            w_start = w * window_size
            w_end = w_start + window_size
            window_df = df[(df['record_id'] >= w_start) & (df['record_id'] < w_end)]

            metrics = compute_window_metrics(window_df)
            if metrics:
                has_drift, affected = detector.update(metrics)

                if has_drift:
                    if first_detection_window is None:
                        first_detection_window = w
                    for m in affected:
                        affected_metrics_set.add(m)
                        all_drifts.append({'window': w, 'metric': m, 'value': metrics[m]})

                severity = assess_severity(len(all_drifts))
                strategy = predict_strategy(severity)
                strategies_by_window.append((w, severity, strategy))

        # Final results
        final_severity = strategies_by_window[-1][1] if strategies_by_window else 'none'
        final_strategy = strategies_by_window[-1][2] if strategies_by_window else 'do_nothing'

        detected = len(all_drifts) > 0
        severity_correct = final_severity in tc['expected_severity']
        strategy_correct = final_strategy == tc['expected_strategy']

        result = {
            'test_name': tc['name'],
            'detected': detected,
            'first_detection_window': first_detection_window,
            'total_drifts': len(all_drifts),
            'affected_metrics': list(affected_metrics_set),
            'final_severity': final_severity,
            'final_strategy': final_strategy,
            'expected_severity': tc['expected_severity'],
            'expected_strategy': tc['expected_strategy'],
            'severity_correct': severity_correct,
            'strategy_correct': strategy_correct,
        }
        results.append(result)

        # Print
        print(f"\nResults:")
        print(f"  Detected: {'YES' if detected else 'NO'}")
        if first_detection_window is not None:
            print(f"  First Detection Window: {first_detection_window}/{n_windows}")
        print(f"  Total Drift Events: {len(all_drifts)}")
        print(f"  Affected Metrics: {list(affected_metrics_set)}")
        print(f"  Final Severity: {final_severity} (expected: {tc['expected_severity']})")
        print(f"  Final Strategy: {final_strategy} (expected: {tc['expected_strategy']})")
        print(f"  Severity Match: {'OK' if severity_correct else 'FAIL'}")
        print(f"  Strategy Match: {'OK' if strategy_correct else 'FAIL'}")

    # Summary
    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    df_results = pd.DataFrame(results)

    total = len(results)
    detected_count = sum(1 for r in results if r['detected'])
    severity_ok = sum(1 for r in results if r['severity_correct'])
    strategy_ok = sum(1 for r in results if r['strategy_correct'])

    print(f"\nDetection Rate: {detected_count}/{total} ({detected_count/total*100:.0f}%)")
    print(f"Severity Accuracy: {severity_ok}/{total} ({severity_ok/total*100:.0f}%)")
    print(f"Strategy Accuracy: {strategy_ok}/{total} ({strategy_ok/total*100:.0f}%)")

    print("\nDetailed Results:")
    print("-" * 70)
    for r in results:
        status = "OK" if (r['severity_correct'] and r['strategy_correct']) else "FAIL"
        print(f"  {r['test_name']:25} | Detected: {str(r['detected']):5} | Severity: {r['final_severity']:10} | Strategy: {r['final_strategy']:15} | {status}")

    # Save
    output_path = Path(__file__).parent.parent / 'results' / 'drift_benchmark_results.csv'
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return df_results


if __name__ == '__main__':
    results = run_benchmark()
    print("\n=== BENCHMARK COMPLETE ===")
