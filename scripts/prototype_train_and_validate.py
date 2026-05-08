#!/usr/bin/env python3
"""
Prototype End-to-End: Train iForest with Ratio Features + Validate

Combines:
1. Enhanced features (ratio features)
2. Layer 2 clean data (99.5K)
3. Extreme synthetic anomalies (5K)
4. Quick validation

Compare with baseline v3 model.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, '.')

from river.anomaly import HalfSpaceTrees


class EnhancedFeatureVectorizer:
    """
    Enhanced vectorizer with ratio features.

    Original 15D:
    - Raw (5D): distance, duration, fare, passenger, total
    - Derived (4D): speed, fare_per_mile, fare_per_minute, fare_per_passenger
    - Temporal (6D): hour, day_of_week, is_weekend, is_rush_hour, is_night, month

    NEW Ratio Features (+6D) = 21D total:
    - fare_per_mile_ratio (vs baseline)
    - fare_per_minute_ratio (vs baseline)
    - implied_speed_ratio (vs baseline)
    - passenger_distance_ratio
    - fare_distance_product
    - duration_distance_ratio
    """

    # Baseline values from clean data analysis
    BASELINE = {
        'fare_per_mile': 2.5,
        'fare_per_minute': 0.67,
        'implied_speed': 12.0,
    }

    def transform(self, record):
        """
        Extract enhanced 21D feature vector.

        Returns:
            np.array of shape (21,)
        """
        eps = 1e-6

        # Parse datetime
        pickup = record.get('tpep_pickup_datetime')
        dropoff = record.get('tpep_dropoff_datetime')

        if isinstance(pickup, str):
            pickup = pd.to_datetime(pickup)
        if isinstance(dropoff, str):
            dropoff = pd.to_datetime(dropoff)

        duration_seconds = (dropoff - pickup).total_seconds()
        duration_minutes = duration_seconds / 60
        duration_hours = duration_seconds / 3600

        # Raw features
        distance = float(record.get('trip_distance', 0))
        fare = float(record.get('fare_amount', 0))
        passengers = float(record.get('passenger_count', 1))
        total = float(record.get('total_amount', 0))

        # Derived features
        speed = distance / (duration_hours + eps)
        fare_per_mile = fare / (distance + eps)
        fare_per_minute = fare / (duration_minutes + eps)
        fare_per_passenger = fare / (passengers + eps)

        # Temporal features
        hour = pickup.hour
        day_of_week = pickup.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        is_rush_hour = 1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0
        is_night = 1 if (hour < 6 or hour > 22) else 0
        month = pickup.month

        # NEW: Ratio features (key innovation!)
        fare_per_mile_ratio = fare_per_mile / (self.BASELINE['fare_per_mile'] + eps)
        fare_per_minute_ratio = fare_per_minute / (self.BASELINE['fare_per_minute'] + eps)
        implied_speed_ratio = speed / (self.BASELINE['implied_speed'] + eps)

        passenger_distance_ratio = passengers / (distance + eps)
        fare_distance_product = fare * distance  # Interaction term
        duration_distance_ratio = duration_minutes / (distance + eps)

        # Assemble 21D vector
        features = np.array([
            # Raw (5)
            distance, duration_minutes, fare, passengers, total,
            # Derived (4)
            speed, fare_per_mile, fare_per_minute, fare_per_passenger,
            # Temporal (6)
            hour, day_of_week, is_weekend, is_rush_hour, is_night, month,
            # Ratio features (6) - NEW!
            fare_per_mile_ratio,
            fare_per_minute_ratio,
            implied_speed_ratio,
            passenger_distance_ratio,
            fare_distance_product,
            duration_distance_ratio,
        ])

        return features


def train_prototype_model():
    """Train iForest on Layer 2 clean data with enhanced features."""
    print("="*70)
    print("PROTOTYPE TRAINING - Enhanced Features")
    print("="*70)

    # Load clean data (from Layer 2)
    clean_path = 'data/clean/prototype_layer2_clean.parquet'
    print(f"\n1. Loading clean data: {clean_path}")
    df_clean = pd.read_parquet(clean_path)
    print(f"   ✓ {len(df_clean):,} clean records")

    # Vectorize
    print(f"\n2. Vectorizing with enhanced features (21D)...")
    vectorizer = EnhancedFeatureVectorizer()

    X = []
    for idx, row in df_clean.iterrows():
        try:
            features = vectorizer.transform(row.to_dict())
            X.append(features)
        except Exception as e:
            if idx % 10000 == 0:
                print(f"   Warning: Failed at {idx}: {e}")

        if (idx + 1) % 25000 == 0:
            print(f"   Vectorized: {idx+1:,} / {len(df_clean):,}")

    X = np.array(X)
    print(f"   ✓ Shape: {X.shape}")

    # Scale
    print(f"\n3. Fitting StandardScaler...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   ✓ Scaled (mean≈0, std≈1)")

    # Train model
    print(f"\n4. Training HalfSpaceTrees (prototype config)...")
    print(f"   Config: n_trees=200, height=10, window=512")

    model = HalfSpaceTrees(
        n_trees=200,
        height=10,
        window_size=512,
        seed=42
    )

    for i, features in enumerate(X_scaled):
        feature_dict = {idx: float(val) for idx, val in enumerate(features)}
        model.learn_one(feature_dict)

        if (i + 1) % 25000 == 0:
            print(f"   Trained: {i+1:,} / {len(X):,}")

    print(f"   ✓ Training complete")

    # Save artifacts
    print(f"\n5. Saving artifacts...")

    Path('models/prototype').mkdir(parents=True, exist_ok=True)

    with open('models/prototype/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/prototype/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/prototype/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"   ✓ Saved to models/prototype/")

    return model, scaler, vectorizer


def validate_prototype():
    """Validate on extreme synthetic anomalies."""
    print("\n" + "="*70)
    print("PROTOTYPE VALIDATION - Extreme Anomalies")
    print("="*70)

    # Load artifacts
    print(f"\n1. Loading artifacts...")
    with open('models/prototype/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/prototype/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/prototype/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"   ✓ Loaded model, scaler, vectorizer")

    # Load data
    print(f"\n2. Loading test data...")
    df = pd.read_parquet('data/clean/prototype_with_extreme_anomalies.parquet')
    labels = pd.read_csv('data/clean/prototype_anomaly_labels.csv')
    print(f"   Total: {len(df):,}")
    print(f"   Clean: {(labels['is_anomaly']==0).sum():,}")
    print(f"   Anomalies: {(labels['is_anomaly']==1).sum():,}")

    # Score
    print(f"\n3. Scoring all records...")
    scores = []

    for idx, row in df.iterrows():
        try:
            features = vectorizer.transform(row.to_dict())
            features_scaled = scaler.transform([features])[0]
            feature_dict = {i: float(v) for i, v in enumerate(features_scaled)}
            score = model.score_one(feature_dict)
            scores.append(score)
        except:
            scores.append(0.0)

        if (idx + 1) % 25000 == 0:
            print(f"   Scored: {idx+1:,} / {len(df):,}")

    scores = np.array(scores)
    print(f"   ✓ Scored {len(scores):,} records")

    # Compute threshold (95th percentile of clean scores)
    print(f"\n4. Computing threshold...")
    clean_scores = scores[labels['is_anomaly'] == 0]
    threshold = np.percentile(clean_scores, 95)
    print(f"   95th percentile threshold: {threshold:.6f}")

    # Predict
    print(f"\n5. Computing metrics...")
    predictions = (scores > threshold).astype(int)
    y_true = labels['is_anomaly'].values

    tp = ((y_true == 1) & (predictions == 1)).sum()
    fp = ((y_true == 0) & (predictions == 1)).sum()
    tn = ((y_true == 0) & (predictions == 0)).sum()
    fn = ((y_true == 1) & (predictions == 0)).sum()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print results
    print("\n" + "="*70)
    print("PROTOTYPE RESULTS")
    print("="*70)

    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:,}  |  FP: {fp:,}")
    print(f"  FN: {fn:,}  |  TN: {tn:,}")

    print(f"\nMetrics:")
    print(f"  Recall (TPR):    {recall:.1%}  (target: ≥75%)")
    print(f"  FPR:             {fpr:.1%}  (target: <5%)")
    print(f"  Precision:       {precision:.1%}")
    print(f"  F1 Score:        {f1:.3f}")

    print(f"\nGo/No-Go:")
    recall_pass = "✅" if recall >= 0.75 else "❌"
    fpr_pass = "✅" if fpr < 0.05 else "❌"
    print(f"  Recall ≥ 75%:    {recall_pass} ({recall:.1%})")
    print(f"  FPR < 5%:        {fpr_pass} ({fpr:.1%})")

    overall_pass = recall >= 0.75 and fpr < 0.05
    if overall_pass:
        print(f"\n🎉 ✅ PROTOTYPE PASSED!")
    else:
        print(f"\n⚠️  ❌ PROTOTYPE NEEDS TUNING")

    return {
        'recall': recall,
        'fpr': fpr,
        'precision': precision,
        'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def main():
    # Train
    print("STEP 1: TRAINING")
    model, scaler, vectorizer = train_prototype_model()

    # Validate
    print("\n\nSTEP 2: VALIDATION")
    results = validate_prototype()

    print("\n" + "="*70)
    print("✅ PROTOTYPE COMPLETE")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. If passed: Scale to full dataset")
    print(f"2. If failed: Tune threshold or add more features")

    return results


if __name__ == '__main__':
    main()
