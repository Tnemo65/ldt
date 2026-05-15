#!/usr/bin/env python3
"""
Test các shared modules trước khi sử dụng trong production.

Chạy: python test_utils.py
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_viz_utils():
    """Test viz_utils imports và basic functions."""
    print("\n[1] Testing viz_utils...")
    
    from viz_utils import (
        VizStyle, ColorPalette, COLORS,
        save_fig, create_grid, create_metrics_text,
        plot_confusion_matrix, plot_roc_pr_curves, plot_score_distribution
    )
    
    # Test colors
    assert COLORS['normal'] == '#4A90D9', "Normal color mismatch"
    assert COLORS['anomaly'] == '#E24A33', "Anomaly color mismatch"
    
    # Test metrics text
    text = create_metrics_text(
        config={'memory_len': 50000},
        metrics={'f1': 0.85, 'precision': 0.9},
        n_normal=97000,
        n_anomaly=3000
    )
    assert 'memory_len' in text, "Config not in text"
    # Note: metrics keys are formatted with proper casing
    assert 'f1' in text.lower() or 'F1' in text, "F1 not in text"
    
    # Test grid creation
    fig, axes = create_grid(6, n_cols=3)
    assert len(axes) == 6, "Wrong number of axes"
    plt.close(fig)
    
    print("  [OK] viz_utils")
    return True


def test_eval_utils():
    """Test eval_utils imports và functions."""
    print("\n[2] Testing eval_utils...")
    
    from eval_utils import (
        EvalMetrics, compute_all_metrics, find_optimal_threshold,
        sweep_thresholds, ThresholdSweep
    )
    
    # Create synthetic data
    np.random.seed(42)
    n_normal = 9000
    n_anomaly = 1000
    
    scores = np.concatenate([
        np.random.gamma(2, 1, n_normal),  # Normal: low scores
        np.random.gamma(5, 2, n_anomaly)  # Anomaly: high scores
    ])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Shuffle
    perm = np.random.permutation(len(scores))
    scores = scores[perm]
    labels = labels[perm]
    
    # Test find_optimal_threshold
    threshold, f1 = find_optimal_threshold(scores, labels)
    assert 0 < threshold < max(scores), f"Threshold out of range: {threshold}"
    assert 0 <= f1 <= 1, f"F1 out of range: {f1}"
    
    # Test sweep_thresholds
    sweep = sweep_thresholds(scores, labels, n_points=50)
    assert isinstance(sweep, ThresholdSweep), "Wrong return type"
    assert len(sweep.thresholds) == 50, "Wrong sweep length"
    
    best_t, best_f1 = sweep.find_best()
    assert 0 < best_t < max(scores), "Best threshold out of range"
    
    # Test compute_all_metrics
    metrics = compute_all_metrics(scores, labels, threshold)
    assert isinstance(metrics, EvalMetrics), "Wrong return type"
    assert 0 <= metrics.f1 <= 1, f"F1 out of range: {metrics.f1}"
    assert 0 <= metrics.precision <= 1, f"Precision out of range"
    assert 0 <= metrics.recall <= 1, f"Recall out of range"
    assert metrics.tp + metrics.fp + metrics.tn + metrics.fn == len(labels), "CM counts mismatch"
    
    # Test to_dict
    d = metrics.to_dict()
    assert 'f1' in d and 'auc_roc' in d and 'auc_pr' in d
    
    # Test summary
    summary = metrics.summary()
    assert 'F1=' in summary, "Summary missing F1"
    
    print(f"  [OK] eval_utils")
    print(f"    - Threshold: {threshold:.4f}")
    print(f"    - F1: {metrics.f1:.4f}")
    print(f"    - Precision: {metrics.precision:.4f}")
    print(f"    - Recall: {metrics.recall:.4f}")
    print(f"    - AUC-ROC: {metrics.auc_roc:.4f}")
    print(f"    - AUC-PR: {metrics.auc_pr:.4f}")
    return True


def test_fraud_utils():
    """Test fraud_utils imports và functions."""
    print("\n[3] Testing fraud_utils...")
    
    from fraud_utils import (
        FraudType, FraudConfig, inject_anomalies,
        get_anomaly_summary, split_by_fraud_type
    )
    import pandas as pd
    
    # Create test dataframe
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame({
        'trip_distance': np.random.uniform(0.5, 20, n),
        'fare_amount': np.random.uniform(5, 50, n),
        'total_amount': np.random.uniform(10, 60, n),
        'RatecodeID': np.random.choice([1, 2, 3, 4, 5], n),
    })
    
    # Test FraudConfig
    config = FraudConfig(
        anomaly_rate=0.05,
        fraud_type=FraudType.MIXED,
        seed=42
    )
    assert config.anomaly_rate == 0.05
    assert config.fraud_type == FraudType.MIXED
    
    # Test inject_anomalies with simple config
    df_test, labels = inject_anomalies(df, anomaly_rate=0.03, seed=42)
    summary = get_anomaly_summary(labels)
    
    assert summary['n_anomaly'] > 0, "No anomalies injected"
    assert summary['n_normal'] > 0, "No normal samples"
    assert abs(summary['anomaly_rate'] - 0.03) < 0.01, f"Wrong anomaly rate: {summary['anomaly_rate']}"
    
    # Test FraudType enum
    assert FraudType.SHORT_TRIP.value == 'short_trip'
    assert FraudType.RATECODE_MISMATCH.value == 'ratecode_mismatch'
    
    # Test split_by_fraud_type
    type1, type3 = split_by_fraud_type(df_test, labels, method='distance')
    assert type1.sum() + type3.sum() == labels.sum(), "Split counts mismatch"
    
    print("  [OK] fraud_utils")
    print(f"    - Injected {summary['n_anomaly']} anomalies ({summary['anomaly_rate']*100:.2f}%)")
    return True


def test_integration():
    """Integration test: use all modules together."""
    print("\n[4] Testing integration...")
    
    from eval_utils import compute_all_metrics, find_optimal_threshold
    from fraud_utils import inject_anomalies, FraudType
    import pandas as pd
    
    # Create realistic test data
    np.random.seed(42)
    n = 10000
    df = pd.DataFrame({
        'trip_distance': np.random.uniform(0.5, 20, n),
        'fare_amount': np.random.uniform(5, 50, n),
        'total_amount': np.random.uniform(10, 60, n),
        'RatecodeID': np.random.choice([1, 2, 3, 4, 5], n),
    })
    
    # Inject anomalies
    df_test, labels = inject_anomalies(
        df, 
        anomaly_rate=0.05,
        fraud_type=FraudType.MIXED,
        seed=42
    )
    
    # Generate "scores" (simulated)
    normal_scores = np.random.gamma(2, 0.5, (labels == 0).sum())
    anomaly_scores = np.random.gamma(4, 1.5, (labels == 1).sum())
    scores = np.zeros(len(labels))
    scores[labels == 0] = normal_scores
    scores[labels == 1] = anomaly_scores
    
    # Compute metrics
    threshold, f1 = find_optimal_threshold(scores, labels)
    metrics = compute_all_metrics(scores, labels, threshold)
    
    assert metrics.f1 > 0.3, f"F1 too low for synthetic data: {metrics.f1}"
    
    print("  [OK] Integration")
    print(f"    - End-to-end: {f1:.4f} F1 score achieved")
    return True


def main():
    print("="*60)
    print("Testing Shared Modules")
    print("="*60)
    
    all_passed = True
    
    try:
        all_passed &= test_viz_utils()
    except Exception as e:
        print(f"  X viz_utils FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_eval_utils()
    except Exception as e:
        print(f"  X eval_utils FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_fraud_utils()
    except Exception as e:
        print(f"  X fraud_utils FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_integration()
    except Exception as e:
        print(f"  X Integration FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("PASSED - ALL TESTS OK!")
        print("Ban co the su dung cac modules trong code chinh.")
    else:
        print("FAILED - SOME TESTS FAILED!")
        print("Sua loi truoc khi su dung.")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    success = main()
    sys.exit(0 if success else 1)
