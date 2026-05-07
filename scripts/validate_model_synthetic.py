#!/usr/bin/env python3
"""
Synthetic Validation - Go/No-Go Gate for Model.
Spec: Lines 3468-3523 (FPR < 5%, Recall > 75%)

CRITICAL: This is a Go/No-Go validation gate.
Model must pass before deployment.

Success Criteria:
- Recall (TPR) ≥ 0.75 (detect at least 75% of anomalies)
- False Positive Rate < 0.05 (at most 5% false alarms on clean data)

Usage:
  python scripts/validate_model_synthetic.py
"""

import argparse
import sys
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.vectorizer import FeatureVectorizer


def get_context_key(record: dict, neighborhood_mapping: dict) -> str:
    """Generate 4D context key for threshold lookup.

    Args:
        record: Trip record
        neighborhood_mapping: Zone to neighborhood mapping

    Returns:
        Context key: "trip_type_time_daytype_neighborhood"
    """
    try:
        from datetime import datetime

        # Parse datetime
        pickup_dt = record.get('tpep_pickup_datetime')
        if isinstance(pickup_dt, str):
            pickup_dt = datetime.fromisoformat(pickup_dt)

        # Trip type (based on distance)
        distance = record.get('trip_distance', 0)
        if distance < 2:
            trip_type = 'short'
        elif distance < 10:
            trip_type = 'medium'
        else:
            trip_type = 'long'

        # Time window
        hour = pickup_dt.hour
        if 6 <= hour < 10:
            time_window = 'morning_rush'
        elif 17 <= hour < 20:
            time_window = 'evening_rush'
        elif 22 <= hour or hour < 6:
            time_window = 'night'
        else:
            time_window = 'midday'

        # Day type
        day_type = 'weekend' if pickup_dt.weekday() >= 5 else 'weekday'

        # Neighborhood
        zone_id = str(record.get('PULocationID', 0))
        neighborhood = neighborhood_mapping.get('mapping', {}).get(zone_id, 'unknown')

        return f"{trip_type}_{time_window}_{day_type}_{neighborhood}"

    except Exception as e:
        return "unknown_unknown_unknown_unknown"


def validate_on_synthetic(
    model_path: str,
    data_path: str,
    labels_path: str,
    thresholds_path: str,
    neighborhood_path: str,
    scaler_path: str
):
    """Validate model on synthetic anomalies.

    CRITICAL: This is a Go/No-Go gate. Model must achieve:
    - Recall ≥ 0.75 (detect 75%+ of anomalies)
    - FPR < 0.05 (at most 5% false alarms)

    Args:
        model_path: Path to trained iForest model
        data_path: Path to data with synthetic anomalies
        labels_path: Path to anomaly labels CSV
        thresholds_path: Path to context thresholds JSON
        neighborhood_path: Path to neighborhood mapping JSON
        scaler_path: Path to fitted StandardScaler

    Returns:
        True if PASS, False if FAIL
    """
    print("="*60)
    print("SYNTHETIC VALIDATION - Go/No-Go Gate")
    print("="*60)

    # 1. Load artifacts
    print("\n1. Loading artifacts...")

    print(f"   - Model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"   - Scaler: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print(f"   - Thresholds: {thresholds_path}")
    with open(thresholds_path) as f:
        thresholds_data = json.load(f)

    print(f"   - Neighborhood mapping: {neighborhood_path}")
    with open(neighborhood_path) as f:
        neighborhood_mapping = json.load(f)

    print("   ✓ All artifacts loaded")

    # 2. Load data and labels
    print(f"\n2. Loading data...")
    print(f"   - Data: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"   - Labels: {labels_path}")
    labels_df = pd.read_csv(labels_path)

    print(f"   ✓ Data: {len(df):,} records")
    print(f"   ✓ Labels: {len(labels_df):,} anomalies")

    # 3. Score all records
    print(f"\n3. Scoring all records...")
    vectorizer = FeatureVectorizer()

    predictions = []
    ground_truth = []

    for idx, row in df.iterrows():
        # Get ground truth
        is_anomaly_true = labels_df[labels_df.index == idx]['is_anomaly'].values
        if len(is_anomaly_true) > 0:
            ground_truth.append(int(is_anomaly_true[0]))
        else:
            ground_truth.append(0)  # Clean record

        # Vectorize and score
        try:
            features = vectorizer.transform(row.to_dict())
            features_scaled = scaler.transform([features])[0]
            feature_dict = {i: float(v) for i, v in enumerate(features_scaled)}

            score = model.score_one(feature_dict)

            # Get context-specific threshold
            context_key = get_context_key(row.to_dict(), neighborhood_mapping)
            threshold = thresholds_data['thresholds'].get(
                context_key,
                thresholds_data.get('global_threshold', 0.5)
            )

            # Predict
            is_anomaly_pred = int(score > threshold)
            predictions.append(is_anomaly_pred)

        except Exception as e:
            # On error, predict clean
            predictions.append(0)

        # Progress
        if (idx + 1) % 500000 == 0:
            print(f"   Scored: {idx + 1:,} / {len(df):,}")

    print(f"   ✓ Scoring complete")

    # 4. Compute metrics
    print(f"\n4. Computing metrics...")

    y_true = np.array(ground_truth)
    y_pred = np.array(predictions)

    # Confusion matrix
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    # Metrics
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 5. Results
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print('='*60)
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:,}  |  FP: {fp:,}")
    print(f"  FN: {fn:,}  |  TN: {tn:,}")

    print(f"\nMetrics:")
    print(f"  Recall (TPR):     {recall:.3f}  (target: ≥0.75)")
    print(f"  FPR:              {fpr:.3f}  (target: <0.05)")
    print(f"  Precision:        {precision:.3f}")
    print(f"  F1 Score:         {f1:.3f}")

    print(f"\nCounts:")
    print(f"  True Anomalies:   {(y_true == 1).sum():,}")
    print(f"  True Clean:       {(y_true == 0).sum():,}")
    print(f"  Predicted Anom:   {(y_pred == 1).sum():,}")
    print(f"  Predicted Clean:  {(y_pred == 0).sum():,}")

    # 6. Go/No-Go decision
    print(f"\n{'='*60}")
    print("GO/NO-GO DECISION")
    print('='*60)

    recall_pass = recall >= 0.75
    fpr_pass = fpr < 0.05

    print(f"\n  Recall ≥ 0.75:    {'✅ PASS' if recall_pass else '❌ FAIL'} ({recall:.3f})")
    print(f"  FPR < 0.05:       {'✅ PASS' if fpr_pass else '❌ FAIL'} ({fpr:.3f})")

    if recall_pass and fpr_pass:
        print(f"\n🎉 ✅ VALIDATION PASSED - Model ready for deployment")
        print('='*60)
        return True
    else:
        print(f"\n⚠️  ❌ VALIDATION FAILED - Retrain required")
        print('='*60)
        if not recall_pass:
            print(f"   Issue: Recall too low ({recall:.3f} < 0.75)")
            print("   Suggestion: Increase n_trees or adjust thresholds")
        if not fpr_pass:
            print(f"   Issue: FPR too high ({fpr:.3f} ≥ 0.05)")
            print("   Suggestion: Increase thresholds per context")
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate iForest on synthetic anomalies')
    parser.add_argument(
        '--model',
        type=str,
        default='models/iforest_model.pkl',
        help='Path to trained model'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/clean/jan_2024_with_50k_anomalies.parquet',
        help='Path to data with synthetic anomalies'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default='data/clean/anomaly_labels.csv',
        help='Path to anomaly labels'
    )
    parser.add_argument(
        '--thresholds',
        type=str,
        default='models/context_thresholds.json',
        help='Path to context thresholds'
    )
    parser.add_argument(
        '--neighborhood',
        type=str,
        default='models/neighborhood_mapping.json',
        help='Path to neighborhood mapping'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default='models/scaler.pkl',
        help='Path to fitted StandardScaler'
    )

    args = parser.parse_args()

    # Validate all files exist
    for name, path in [
        ('Model', args.model),
        ('Data', args.data),
        ('Labels', args.labels),
        ('Thresholds', args.thresholds),
        ('Neighborhood', args.neighborhood),
        ('Scaler', args.scaler)
    ]:
        if not Path(path).exists():
            print(f"❌ Error: {name} not found: {path}")
            return 1

    # Run validation
    try:
        passed = validate_on_synthetic(
            model_path=args.model,
            data_path=args.data,
            labels_path=args.labels,
            thresholds_path=args.thresholds,
            neighborhood_path=args.neighborhood,
            scaler_path=args.scaler
        )

        return 0 if passed else 1

    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
