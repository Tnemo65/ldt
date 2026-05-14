#!/usr/bin/env python3
"""
CA-DQStream ML Model Training Script.

Trains an IsolationForest anomaly detection model on synthetic NYC taxi data,
registers it in MLflow, and uploads the artifact to MinIO.

Pipeline:
  1. Generate or load synthetic historical NYC taxi data
  2. Apply Layer 1 filters (schema + physical impossibility) to get clean baseline
  3. Compute 21D ratio features for IsolationForest
  4. Train IsolationForest on clean data
  5. Compute per-cluster thresholds (baseline stats)
  6. Register in MLflow with params, metrics, and artifacts
  7. Upload to MinIO: cadqstream-checkpoints/ml-models/anomaly_detector_<version>.pkl

Usage:
  python train_model.py --version v1 --n-samples 100000
  python train_model.py --load-from-minio s3://nyc-taxi-raw/training_data.parquet

Environment Variables:
  MLFLOW_TRACKING_URI    MLflow server URL (default: http://mlflow:5000)
  MINIO_ENDPOINT         MinIO endpoint (default: minio:9000)
  MINIO_ACCESS_KEY       MinIO access key (default: minioadmin)
  MINIO_SECRET_KEY       MinIO secret key (default: minioadmin123)
  MODEL_BUCKET           MinIO bucket for models (default: cadqstream-checkpoints)
"""

import os
import sys
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Cloud / storage
import mlflow
from mlflow.tracking import MlflowClient
import boto3
from botocore.config import Config


# ─── Constants ─────────────────────────────────────────────────────────────────

PAYMENT_TYPES = [1, 2, 3, 4, 5, 6]
TRIP_PROFILES = [
    ("short",   0.5,  3.0,   5.0,  20.0,   300, 1800),
    ("medium",  2.0, 10.0,  15.0,  60.0,   900, 3600),
    ("long",    8.0, 30.0,  40.0, 200.0,  1800, 5400),
    ("airport", 15.0, 35.0,  50.0, 150.0,  2400, 5400),
]
NYC_ZONE_MIN, NYC_ZONE_MAX = 1, 263

FEATURE_NAMES = [
    'fare_per_mile', 'tip_pct', 'tolls_per_fare', 'extra_pct',
    'duration_miles_per_hour', 'fare_per_duration',
    'fare_z', 'distance_z', 'passenger_z',
    'pickup_zone', 'dropoff_zone',
    'payment_type', 'is_round_trip',
    'long_trip', 'short_trip', 'airport_trip',
    'rush_hour', 'late_night', 'weekend',
    'manhattan_trip', 'borough_cross',
]


# ─── Data Generation ────────────────────────────────────────────────────────────

def generate_synthetic_data(n_samples=100_000, seed=42):
    """Generate synthetic NYC taxi trip data with realistic distributions."""
    rng = np.random.default_rng(seed)
    records = []

    for i in range(n_samples):
        profile = TRIP_PROFILES[rng.integers(0, len(TRIP_PROFILES))]
        name, d_min, d_max, f_min, f_max, du_min, du_max = profile

        distance = rng.uniform(d_min, d_max)
        fare = rng.uniform(f_min, f_max)
        dur_sec = rng.uniform(du_min, du_max)
        dur_h = dur_sec / 3600.0

        extra = rng.uniform(0, 3.5)
        mta_tax = 0.5
        tip = fare * rng.uniform(0, 0.25) if rng.random() > 0.3 else 0.0
        tolls = rng.uniform(0, 20) if rng.random() > 0.9 else 0.0
        imp_surcharge = 1.0
        cong_surcharge = 2.5
        airport_fee = 2.5 if name == "airport" else 0.0
        total = fare + extra + mta_tax + tip + tolls + imp_surcharge + cong_surcharge + airport_fee

        hour = rng.integers(0, 24)
        day = rng.integers(1, 29)

        records.append({
            'VendorID': rng.choice([1, 2]),
            'tpep_pickup_datetime': f"2024-01-{day:02d}T{hour:02d}:30:00",
            'passenger_count': rng.choice([1, 2, 3, 4, 5, 6]),
            'trip_distance': round(distance, 2),
            'PULocationID': rng.integers(1, NYC_ZONE_MAX + 1),
            'DOLocationID': rng.integers(1, NYC_ZONE_MAX + 1),
            'payment_type': rng.choice(PAYMENT_TYPES),
            'fare_amount': round(fare, 2),
            'total_amount': round(total, 2),
            'trip_duration': dur_h,
        })

    return pd.DataFrame(records)


def layer1_filter(df):
    """Apply Layer 1 schema + physical impossibility filters.

    Returns a boolean mask of clean records.
    """
    mask = pd.Series(True, index=df.index)

    # Required fields not null
    for col in ['trip_distance', 'fare_amount', 'PULocationID', 'DOLocationID', 'passenger_count']:
        mask &= df[col].notna()

    # Zone range
    mask &= (df['PULocationID'] >= NYC_ZONE_MIN) & (df['PULocationID'] <= NYC_ZONE_MAX)
    mask &= (df['DOLocationID'] >= NYC_ZONE_MIN) & (df['DOLocationID'] <= NYC_ZONE_MAX)

    # Physical impossibility filters
    mask &= df['passenger_count'] >= 1
    mask &= df['trip_distance'] >= 0
    mask &= df['fare_amount'] >= 0
    mask &= df['trip_duration'] > 0

    return mask


# ─── Feature Engineering ────────────────────────────────────────────────────────

def engineer_features(df):
    """Compute 21 ratio/demographic features for IsolationForest."""
    features = pd.DataFrame(index=df.index)

    # Handle division by zero
    eps = 1e-6

    features['fare_per_mile'] = df['fare_amount'] / (df['trip_distance'] + eps)
    features['tip_pct'] = 0.0  # Will be computed if tip column exists
    features['tolls_per_fare'] = 0.0  # No tolls in simplified data
    features['extra_pct'] = df['fare_amount'] / (df['fare_amount'] + eps)

    duration_h = np.maximum(df['trip_duration'], eps / 3600)
    features['duration_miles_per_hour'] = df['trip_distance'] / duration_h
    features['fare_per_duration'] = df['fare_amount'] / duration_h

    # Z-score features (relative to dataset)
    features['fare_z'] = (df['fare_amount'] - df['fare_amount'].mean()) / (df['fare_amount'].std() + eps)
    features['distance_z'] = (df['trip_distance'] - df['trip_distance'].mean()) / (df['trip_distance'].std() + eps)
    features['passenger_z'] = (df['passenger_count'] - df['passenger_count'].mean()) / (df['passenger_count'].std() + eps)

    features['pickup_zone'] = df['PULocationID'] / NYC_ZONE_MAX
    features['dropoff_zone'] = df['DOLocationID'] / NYC_ZONE_MAX
    features['payment_type'] = df['payment_type'] / 6.0

    features['is_round_trip'] = (df['PULocationID'] == df['DOLocationID']).astype(float)
    features['long_trip'] = (df['trip_distance'] > 10.0).astype(float)
    features['short_trip'] = (df['trip_distance'] < 1.0).astype(float)

    pu_zone = df['PULocationID']
    features['airport_trip'] = pu_zone.isin([132, 138]).astype(float)
    features['manhattan_trip'] = (pu_zone <= 50).astype(float)
    features['borough_cross'] = (
        (pu_zone <= 50) & (df['DOLocationID'] > 50) |
        (pu_zone > 50) & (df['DOLocationID'] <= 50)
    ).astype(float)

    features['rush_hour'] = df['tpep_pickup_datetime'].apply(
        lambda x: 1.0 if (7 <= int(x[11:13]) <= 9 or 16 <= int(x[11:13]) <= 19) else 0.0
    )
    features['late_night'] = df['tpep_pickup_datetime'].apply(
        lambda x: 1.0 if (0 <= int(x[11:13]) <= 5) else 0.0
    )
    features['weekend'] = df['tpep_pickup_datetime'].apply(
        lambda x: 1.0 if pd.Timestamp(x).dayofweek >= 5 else 0.0
    )

    # Clip extreme z-scores
    for col in ['fare_z', 'distance_z', 'passenger_z']:
        features[col] = features[col].clip(-5, 5)

    return features[FEATURE_NAMES]


# ─── Model Training ─────────────────────────────────────────────────────────────

def train_isolation_forest(X_train, X_test, contamination=0.05, n_estimators=200, random_state=42):
    """Train IsolationForest on clean training data."""
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=min(256, len(X_train)),
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def evaluate_model(model, X_test):
    """Evaluate model on test set, computing precision/recall/f1."""
    preds = model.predict(X_test)
    labels = (preds == -1).astype(int)  # -1 = anomaly, 1 = normal

    # Compute metrics relative to contamination
    precision = precision_score(labels, np.zeros_like(labels), zero_division=0)
    recall = recall_score(labels, np.zeros_like(labels), zero_division=0)
    f1 = f1_score(labels, np.zeros_like(labels), zero_division=0)

    # Since we trained on clean data, we can't compute true anomalies here
    # In production, you'd have labelled anomaly data
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_anomalies': int((preds == -1).sum()),
        'n_normal': int((preds == 1).sum()),
        'contamination': float((preds == -1).sum() / len(preds)),
    }


def compute_thresholds(model, X_train, scaler):
    """Compute per-feature baseline thresholds for cluster analysis."""
    X_scaled = scaler.transform(X_train)
    return {
        'fare_per_mile': {'mean': float(X_train['fare_per_mile'].mean()), 'std': float(X_train['fare_per_mile'].std())},
        'fare_z': {'mean': float(X_train['fare_z'].mean()), 'std': float(X_train['fare_z'].std())},
        'distance_z': {'mean': float(X_train['distance_z'].mean()), 'std': float(X_train['distance_z'].std())},
        'duration_miles_per_hour': {'mean': float(X_train['duration_miles_per_hour'].mean()),
                                     'std': float(X_train['duration_miles_per_hour'].std())},
    }


# ─── MinIO / S3 Upload ────────────────────────────────────────────────────────

def get_minio_client():
    """Create a MinIO / S3-compatible client."""
    endpoint = os.getenv('MINIO_ENDPOINT', 'minio:9000')
    access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
    secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')

    client = boto3.client(
        's3',
        endpoint_url=f'http://{endpoint}',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4'),
    )
    return client


def upload_to_minio(local_path, bucket, key):
    """Upload a local file to MinIO."""
    client = get_minio_client()
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)

    client.upload_file(local_path, bucket, key)
    print(f"[upload] Uploaded {local_path} -> s3://{bucket}/{key}")


# ─── MLflow Integration ─────────────────────────────────────────────────────────

def setup_mlflow(tracking_uri):
    """Configure MLflow tracking server."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('cadqstream-anomaly-detection')

    client = MlflowClient(tracking_uri)
    return client


# ─── Main Training Pipeline ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train CA-DQStream anomaly detection model')
    parser.add_argument('--version', default='v1', help='Model version (default: v1)')
    parser.add_argument('--n-samples', type=int, default=100_000,
                        help='Number of synthetic samples (default: 100000)')
    parser.add_argument('--contamination', type=float, default=0.05,
                        help='IsolationForest contamination (default: 0.05)')
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='IsolationForest n_estimators (default: 200)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--model-bucket', default='cadqstream-checkpoints',
                        help='MinIO model bucket (default: cadqstream-checkpoints)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set fraction (default: 0.2)')
    parser.add_argument('--no-upload', action='store_true',
                        help='Skip MinIO upload')
    parser.add_argument('--no-register', action='store_true',
                        help='Skip MLflow registration')

    args = parser.parse_args()

    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    model_name = f'anomaly_detector_{args.version}'

    print(f"[train] === CA-DQStream Model Training ===")
    print(f"[train] Version: {model_name}")
    print(f"[train] Samples: {args.n_samples:,}")
    print(f"[train] Contamination: {args.contamination}")
    print(f"[train] MLflow: {tracking_uri}")

    # ─── Step 1: Generate data ───────────────────────────────────────────────
    print(f"\n[train] Step 1: Generating {args.n_samples:,} synthetic trips...")
    df = generate_synthetic_data(n_samples=args.n_samples, seed=args.seed)
    print(f"[train] Generated {len(df):,} records")

    # ─── Step 2: Layer 1 filter ──────────────────────────────────────────────
    print(f"\n[train] Step 2: Applying Layer 1 filters...")
    clean_mask = layer1_filter(df)
    df_clean = df[clean_mask].copy()
    print(f"[train] Clean records: {len(df_clean):,} / {len(df):,} "
          f"({len(df_clean)/len(df):.1%})")

    # ─── Step 3: Feature engineering ─────────────────────────────────────────
    print(f"\n[train] Step 3: Engineering {len(FEATURE_NAMES)} features...")
    X = engineer_features(df_clean)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    print(f"[train] Feature matrix: {X.shape}")

    # ─── Step 4: Split ──────────────────────────────────────────────────────
    print(f"\n[train] Step 4: Train/test split ({args.test_size:.0%} test)...")
    X_train, X_test = train_test_split(X, test_size=args.test_size, random_state=args.seed)
    print(f"[train] Train: {len(X_train):,}, Test: {len(X_test):,}")

    # ─── Step 5: Scale ───────────────────────────────────────────────────────
    print(f"\n[train] Step 5: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ─── Step 6: Train ──────────────────────────────────────────────────────
    print(f"\n[train] Step 6: Training IsolationForest (n_estimators={args.n_estimators})...")
    model = train_isolation_forest(
        X_train_scaled, X_test_scaled,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )
    print(f"[train] Model trained successfully")

    # ─── Step 7: Evaluate ─────────────────────────────────────────────────
    print(f"\n[train] Step 7: Evaluating model...")
    metrics = evaluate_model(model, X_test_scaled)
    print(f"[train] Anomalies detected: {metrics['n_anomalies']:,} "
          f"({metrics['contamination']:.2%} of test set)")

    # ─── Step 8: Compute thresholds ────────────────────────────────────────
    print(f"\n[train] Step 8: Computing baseline thresholds...")
    thresholds = compute_thresholds(model, X_train, scaler)

    # ─── Step 9: Save artifacts locally ───────────────────────────────────
    print(f"\n[train] Step 9: Saving artifacts...")
    artifact_dir = '/tmp/cadqstream-models'
    os.makedirs(artifact_dir, exist_ok=True)

    model_path = f'{artifact_dir}/anomaly_detector_{args.version}.pkl'
    scaler_path = f'{artifact_dir}/anomaly_scaler_{args.version}.pkl'
    threshold_path = f'{artifact_dir}/thresholds_{args.version}.json'
    feature_path = f'{artifact_dir}/feature_names_{args.version}.txt'

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(threshold_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    with open(feature_path, 'w') as f:
        f.write('\n'.join(FEATURE_NAMES))

    print(f"[train] Saved: {model_path}")
    print(f"[train] Saved: {scaler_path}")
    print(f"[train] Saved: {threshold_path}")

    # ─── Step 10: Register in MLflow ───────────────────────────────────────
    if not args.no_register:
        print(f"\n[train] Step 10: Registering model in MLflow...")
        setup_mlflow(tracking_uri)

        with mlflow.start_run(run_name=f'train-{model_name}') as run:
            run_id = run.info.run_id

            # Log parameters
            mlflow.log_param('model_version', args.version)
            mlflow.log_param('model_type', 'IsolationForest')
            mlflow.log_param('contamination', args.contamination)
            mlflow.log_param('n_estimators', args.n_estimators)
            mlflow.log_param('n_features', len(FEATURE_NAMES))
            mlflow.log_param('n_training_samples', len(X_train))
            mlflow.log_param('n_clean_samples', len(df_clean))
            mlflow.log_param('random_state', args.seed)
            mlflow.log_param('features', ','.join(FEATURE_NAMES[:5]) + '...')

            # Log metrics
            mlflow.log_metric('n_anomalies', metrics['n_anomalies'])
            mlflow.log_metric('contamination_actual', metrics['contamination'])
            mlflow.log_metric('n_normal', metrics['n_normal'])
            mlflow.log_metric('n_clean_records', len(df_clean))
            mlflow.log_metric('clean_rate', len(df_clean) / len(df))

            # Log baseline threshold stats
            for feat, stats in thresholds.items():
                mlflow.log_metric(f'threshold_{feat}_mean', stats['mean'])
                mlflow.log_metric(f'threshold_{feat}_std', stats['std'])

            # Log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(scaler_path)
            mlflow.log_artifact(threshold_path)
            mlflow.log_artifact(feature_path)

            print(f"[train] MLflow run: {run_id}")

        # Register as model version
        try:
            model_uri = f'runs:/{run_id}/anomaly_detector_{args.version}.pkl'
            mv = mlflow.register_model(model_uri, model_name)
            print(f"[train] Registered model: {model_name}:{mv.version}")
        except Exception as e:
            print(f"[train] Note: Could not register model in MLflow registry: {e}")

    # ─── Step 11: Upload to MinIO ───────────────────────────────────────────
    if not args.no_upload:
        print(f"\n[train] Step 11: Uploading to MinIO (bucket: {args.model_bucket})...")
        try:
            upload_to_minio(model_path, args.model_bucket, f'{model_name}.pkl')
            upload_to_minio(scaler_path, args.model_bucket, f'anomaly_scaler_{args.version}.pkl')
            upload_to_minio(threshold_path, args.model_bucket, f'thresholds_{args.version}.json')
            print(f"[train] All artifacts uploaded to MinIO")
        except Exception as e:
            print(f"[train] WARNING: MinIO upload failed: {e}")
            print(f"[train] Artifacts saved locally at: {artifact_dir}")

    print(f"\n[train] === Training Complete ===")
    print(f"[train] Model: {model_path}")
    print(f"[train] MLflow: {tracking_uri}")
    print(f"[train] MinIO: s3://{args.model_bucket}/{model_name}.pkl")
    print(f"[train] Run: python src/flink_job_complete.py (set MODEL_VERSION={args.version})")


if __name__ == '__main__':
    main()
