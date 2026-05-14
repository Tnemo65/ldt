#!/usr/bin/env python3
"""
Step 5: Verify METER model predictions.
Loads trained model from models/ and tests:
1. All 4 strategy classes
2. Edge cases
3. METER vs fallback rule comparison
4. IECOperator integration check
"""

import sys
from pathlib import Path
import pickle
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_DIR = Path('models')
METADATA_PATH = MODEL_DIR / 'meter_metadata.json'
MODEL_PATH = MODEL_DIR / 'meter_hypernetwork.pkl'
SCALER_PATH = MODEL_DIR / 'meter_scaler.pkl'


STRATEGIES = {
    0: 'do_nothing',
    1: 'adjust_threshold',
    2: 'retrain_model',
    3: 'switch_model'
}


def fallback_strategy(anomaly_rate, null_rate, violation_rate, avg_score):
    """Rule-based fallback from IECOperator._fallback_strategy()."""
    # Simplified severity estimation
    n_drift_signals = 0
    if anomaly_rate > 0.10:
        n_drift_signals += 1
    if null_rate > 0.05:
        n_drift_signals += 1
    if violation_rate > 0.10:
        n_drift_signals += 1
    if avg_score > 0.60:
        n_drift_signals += 1

    if n_drift_signals == 0:
        return 'do_nothing'
    elif n_drift_signals == 1:
        return 'adjust_threshold'
    elif n_drift_signals == 2:
        return 'retrain_model'
    else:
        return 'switch_model'


def meter_strategy(model, scaler, feature_cols, meta_metrics):
    """Predict strategy using trained METER model."""
    feat = np.array([[meta_metrics.get(c, 0.0) for c in feature_cols]])
    feat_s = scaler.transform(feat)
    pred_id = int(model.predict(feat_s)[0])
    probs = model.predict_proba(feat_s)[0]
    conf = float(probs[pred_id])
    return STRATEGIES[pred_id], conf


def test_all_classes(model, scaler, feature_cols):
    """Test that all 4 strategy classes can be triggered."""
    print("\n" + "=" * 60)
    print("Test 1: All 4 strategy classes")
    print("=" * 60)

    test_cases = [
        # (volume, null_rate, violation_rate, anomaly_rate, avg_score, delta_score)
        (1000, 0.005, 0.03, 0.04, 0.38, 0.10, 0, 'do_nothing', 'Normal operation'),
        (950, 0.02, 0.05, 0.09, 0.52, 0.12, 1, 'adjust_threshold', 'Threshold drift'),
        (800, 0.06, 0.12, 0.22, 0.62, 0.20, 2, 'retrain_model', 'Distribution shift'),
        (550, 0.15, 0.30, 0.55, 0.88, 0.45, 3, 'switch_model', 'Concept drift (severe)'),
    ]

    all_pass = True
    for (vol, null_r, viol_r, anom_r, avg_s, delta, expected_id, expected_name, desc) in test_cases:
        meta = {
            'volume': vol, 'null_rate': null_r, 'violation_rate': viol_r,
            'anomaly_rate': anom_r, 'avg_anomaly_score': avg_s, 'delta_score': delta
        }
        pred_name, conf = meter_strategy(model, scaler, feature_cols, meta)
        match = "PASS" if pred_name == expected_name else "FAIL"
        if pred_name != expected_name:
            all_pass = False
        print(f"  [{match}] {desc:30s} -> {pred_name} (expected={expected_name}, conf={conf:.2f})")

    return all_pass


def test_edge_cases(model, scaler, feature_cols):
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("Test 2: Edge cases")
    print("=" * 60)

    edge_cases = [
        ({'volume': 0, 'null_rate': 0.0, 'violation_rate': 0.0,
          'anomaly_rate': 0.0, 'avg_anomaly_score': 0.0, 'delta_score': 0.0},
         "Zero values"),
        ({'volume': 9999, 'null_rate': 0.99, 'violation_rate': 0.99,
          'anomaly_rate': 0.99, 'avg_anomaly_score': 0.99, 'delta_score': 0.99},
         "Max values"),
        ({'volume': 500, 'null_rate': 0.0, 'violation_rate': 0.0,
          'anomaly_rate': 0.0, 'avg_anomaly_score': 0.0, 'delta_score': 0.0},
         "Low volume only"),
        ({'volume': 1000, 'null_rate': 0.0, 'violation_rate': 0.0,
          'anomaly_rate': 0.50, 'avg_anomaly_score': 0.49, 'delta_score': 0.0},
         "Anomaly rate high but avg_score below threshold"),
        ({'volume': 1000, 'null_rate': 0.0, 'violation_rate': 0.50,
          'anomaly_rate': 0.0, 'avg_anomaly_score': 0.0, 'delta_score': 0.50},
         "Violation rate high but anomaly rate zero"),
    ]

    for meta, desc in edge_cases:
        pred_name, conf = meter_strategy(model, scaler, feature_cols, meta)
        print(f"  {desc:50s} -> {pred_name} (conf={conf:.2f})")


def test_meter_vs_fallback(model, scaler, feature_cols):
    """Compare METER predictions vs fallback rule-based."""
    print("\n" + "=" * 60)
    print("Test 3: METER vs Fallback rule agreement")
    print("=" * 60)

    test_cases = [
        ({'volume': 1000, 'null_rate': 0.005, 'violation_rate': 0.03,
          'anomaly_rate': 0.04, 'avg_anomaly_score': 0.38, 'delta_score': 0.10}, 0),
        ({'volume': 950, 'null_rate': 0.02, 'violation_rate': 0.05,
          'anomaly_rate': 0.09, 'avg_anomaly_score': 0.52, 'delta_score': 0.12}, 1),
        ({'volume': 850, 'null_rate': 0.04, 'violation_rate': 0.08,
          'anomaly_rate': 0.15, 'avg_anomaly_score': 0.58, 'delta_score': 0.15}, 2),
        ({'volume': 750, 'null_rate': 0.08, 'violation_rate': 0.15,
          'anomaly_rate': 0.25, 'avg_anomaly_score': 0.65, 'delta_score': 0.25}, 2),
        ({'volume': 550, 'null_rate': 0.15, 'violation_rate': 0.30,
          'anomaly_rate': 0.55, 'avg_anomaly_score': 0.88, 'delta_score': 0.45}, 3),
        ({'volume': 1100, 'null_rate': 0.01, 'violation_rate': 0.04,
          'anomaly_rate': 0.08, 'avg_anomaly_score': 0.48, 'delta_score': 0.10}, 1),
        ({'volume': 1000, 'null_rate': 0.03, 'violation_rate': 0.06,
          'anomaly_rate': 0.11, 'avg_anomaly_score': 0.53, 'delta_score': 0.14}, 1),
        ({'volume': 900, 'null_rate': 0.01, 'violation_rate': 0.04,
          'anomaly_rate': 0.07, 'avg_anomaly_score': 0.45, 'delta_score': 0.08}, 0),
    ]

    agree_count = 0
    for meta, expected_fb in test_cases:
        meter_pred, meter_conf = meter_strategy(model, scaler, feature_cols, meta)
        fb_pred = fallback_strategy(
            meta['anomaly_rate'], meta['null_rate'],
            meta['violation_rate'], meta['avg_anomaly_score']
        )
        agree = meter_pred == fb_pred
        if agree:
            agree_count += 1
        mark = "AGREE" if agree else "DIFF"
        print(f"  [{mark}] METER={meter_pred:20s} FB={fb_pred:20s} | {meta}")

    agreement_rate = agree_count / len(test_cases) * 100
    print(f"\n  Agreement rate: {agree_count}/{len(test_cases)} ({agreement_rate:.0f}%)")
    return agreement_rate >= 60


def test_iec_integration():
    """Check that model files exist and are loadable by IECOperator."""
    print("\n" + "=" * 60)
    print("Test 4: IECOperator integration check")
    print("=" * 60)

    all_exist = True
    for path, name in [(MODEL_PATH, 'meter_hypernetwork.pkl'),
                        (SCALER_PATH, 'meter_scaler.pkl'),
                        (METADATA_PATH, 'meter_metadata.json')]:
        exists = path.exists()
        print(f"  {'[OK]' if exists else '[MISSING]'} {name} ({path})")
        if not exists:
            all_exist = False

    if all_exist:
        # Try loading
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            print(f"\n  Model loaded OK")
            print(f"  Architecture: {metadata.get('architecture')}")
            print(f"  Test accuracy: {metadata.get('test_accuracy')}")
            print(f"  Test F1: {metadata.get('test_f1_weighted')}")
            print(f"  Features: {metadata.get('input_features')}")
            return True
        except Exception as e:
            print(f"\n  ERROR loading model: {e}")
            return False
    return False


def test_boundary_conditions(model, scaler, feature_cols):
    """Test boundary conditions between strategy classes."""
    print("\n" + "=" * 60)
    print("Test 5: Boundary conditions")
    print("=" * 60)

    boundaries = [
        # Anomaly rate boundaries
        ({'volume': 1000, 'null_rate': 0.01, 'violation_rate': 0.05,
          'anomaly_rate': 0.10, 'avg_anomaly_score': 0.50, 'delta_score': 0.15},
         "anomaly_rate=0.10 (threshold boundary)"),
        ({'volume': 1000, 'null_rate': 0.01, 'violation_rate': 0.05,
          'anomaly_rate': 0.15, 'avg_anomaly_score': 0.60, 'delta_score': 0.20},
         "anomaly_rate=0.15 (retrain boundary)"),
        ({'volume': 1000, 'null_rate': 0.01, 'violation_rate': 0.05,
          'anomaly_rate': 0.25, 'avg_anomaly_score': 0.70, 'delta_score': 0.25},
         "anomaly_rate=0.25 (switch boundary)"),
        # Delta score boundaries
        ({'volume': 1000, 'null_rate': 0.01, 'violation_rate': 0.05,
          'anomaly_rate': 0.12, 'avg_anomaly_score': 0.55, 'delta_score': 0.10},
         "delta_score=0.10 (low delta)"),
        ({'volume': 1000, 'null_rate': 0.01, 'violation_rate': 0.20,
          'anomaly_rate': 0.20, 'avg_anomaly_score': 0.60, 'delta_score': 0.50},
         "delta_score=0.50 (high delta)"),
        # Null rate boundary
        ({'volume': 1000, 'null_rate': 0.05, 'violation_rate': 0.05,
          'anomaly_rate': 0.10, 'avg_anomaly_score': 0.55, 'delta_score': 0.15},
         "null_rate=0.05 (boundary)"),
        ({'volume': 1000, 'null_rate': 0.10, 'violation_rate': 0.05,
          'anomaly_rate': 0.10, 'avg_anomaly_score': 0.55, 'delta_score': 0.15},
         "null_rate=0.10 (high null)"),
    ]

    for meta, desc in boundaries:
        pred_name, conf = meter_strategy(model, scaler, feature_cols, meta)
        print(f"  {desc:50s} -> {pred_name} (conf={conf:.2f})")


def main():
    print("=" * 60)
    print("METER Model Verification (Step 5)")
    print("=" * 60)

    # Load model
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run: python src/ml/train_meter.py first")
        return 1

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    feature_cols = metadata.get('input_features',
        ['volume', 'null_rate', 'violation_rate', 'avg_anomaly_score',
         'anomaly_rate', 'delta_score'])

    print(f"\nModel: {metadata.get('architecture')}")
    print(f"Test accuracy: {metadata.get('test_accuracy')}")
    print(f"Test F1: {metadata.get('test_f1_weighted')}")

    results = {}
    results['all_classes'] = test_all_classes(model, scaler, feature_cols)
    test_edge_cases(model, scaler, feature_cols)
    results['fallback_agreement'] = test_meter_vs_fallback(model, scaler, feature_cols)
    results['iec_integration'] = test_iec_integration()
    test_boundary_conditions(model, scaler, feature_cols)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    print("=" * 60)

    if all(results.values()):
        print("\nAll tests PASSED. METER model is ready for production.")
        return 0
    else:
        print("\nSome tests FAILED. Review model quality.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
