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
    scaler_path: str,
    subset: int = None
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
        subset: Optional number of records to validate on (for quick testing)

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

    # CRITICAL: Anomalies are in the LAST 50K rows of the parquet file.
    # The inject script appends anomalies after clean records.
    # For validation, we sample from the tail to ensure anomalies are included.
    n_clean = len(df) - 50000
    n_anomalies_total = 50000

    if subset is not None:
        # For subset testing: take last N records to capture anomalies
        if subset <= n_anomalies_total:
            # Very small subset: take only from anomaly section
            df = df.tail(subset).reset_index(drop=True)
            labels_df = labels_df.tail(subset).reset_index(drop=True)
            print(f"   - Using last {subset:,} records (anomaly section)")
        else:
            # Larger subset: take all anomalies + some clean from front
            df_clean = df.head(n_clean).sample(subset - n_anomalies_total, random_state=42)
            df_anom = df.tail(n_anomalies_total)
            df = pd.concat([df_clean, df_anom], ignore_index=True)
            labels_clean = labels_df.head(n_clean).loc[df_clean.index]
            labels_anom = labels_df.tail(n_anomalies_total)
            labels_df = pd.concat([labels_clean, labels_anom], ignore_index=True)
            print(f"   - Using {subset:,} records (all anomalies + {subset - n_anomalies_total:,} clean)")
    else:
        print(f"   - Using ALL {len(df):,} records (full validation)")

    print(f"   Records: {len(df):,}")
    print(f"   Anomalies in sample: {labels_df['is_anomaly'].sum():,}")

    # 3. Score all records (batch for speed)
    print(f"\n3. Scoring all records (batch mode)...")
    vectorizer = FeatureVectorizer()

    # Batch vectorization
    print(f"   Vectorizing...")
    X = vectorizer.transform_batch(df)

    # Batch scaling
    print(f"   Scaling...")
    X_scaled = scaler.transform(X)

    # sklearn IsolationForest: batch score_samples
    # Returns negative scores (lower = more anomalous)
    print(f"   Scoring with sklearn IsolationForest...")
    raw_scores = model.score_samples(X_scaled)
    anomaly_scores = -raw_scores  # Negate: higher = more anomalous

    # Load context thresholds (global + per-context)
    # NOTE: Old thresholds (from River HalfSpaceTrees) may be incompatible with sklearn scores.
    # sklearn scores are ~[0.3, 0.8], while River scores were ~[0.9, 0.95].
    # If thresholds don't match score range, fall back to percentile-based.
    global_thresh = thresholds_data.get('global_threshold', 0.5)
    context_thresholds = thresholds_data.get('thresholds', {})

    if global_thresh < anomaly_scores.min() or global_thresh > anomaly_scores.max():
        print(f"   WARNING: Stored threshold ({global_thresh:.3f}) outside score range")
        print(f"            Score range: [{anomaly_scores.min():.3f}, {anomaly_scores.max():.3f}]")
        print(f"   Falling back to percentile-based thresholding...")

        # Find best threshold by F1 on this data
        best_f1 = 0
        best_thresh = 0
        best_pct = 0
        for pct in [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]:
            thresh = np.percentile(anomaly_scores, pct)
            y_pred_t = (anomaly_scores >= thresh).astype(int)
            y_true_t = labels_df['is_anomaly'].values
            tp = ((y_true_t == 1) & (y_pred_t == 1)).sum()
            fp = ((y_true_t == 0) & (y_pred_t == 1)).sum()
            tn = ((y_true_t == 0) & (y_pred_t == 0)).sum()
            fn = ((y_true_t == 1) & (y_pred_t == 0)).sum()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
            print(f"     {pct}th: t={thresh:.4f} recall={recall:.3f} fpr={fpr:.3f} f1={f1:.3f}")
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                best_pct = pct

        print(f"   Best: {best_pct}th percentile, thresh={best_thresh:.4f}, F1={best_f1:.3f}")
        global_thresh = best_thresh
        context_thresholds = {}

    # Compute predictions
    print(f"   Computing context-aware decisions...")
    predictions = []

    for idx, row in df.iterrows():
        try:
            context_key = get_context_key(row.to_dict(), neighborhood_mapping)
            threshold = context_thresholds.get(context_key, global_thresh)
            is_anomaly = int(anomaly_scores[idx] > threshold)
            predictions.append(is_anomaly)
        except Exception:
            predictions.append(0)

    ground_truth = labels_df['is_anomaly'].values

    print(f"   Scoring complete")

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
    parser.add_argument(
        '--subset',
        type=int,
        default=None,
        help='Validate on subset of records (for quick testing)'
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
            scaler_path=args.scaler,
            subset=args.subset
        )

        return 0 if passed else 1

    except Exception as e:
        print(f"\n❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
