#!/usr/bin/env python3
"""
CA-DQStream Model Loader - Loads trained model from MinIO and publishes to Kafka.

This script:
1. Downloads model artifacts from MinIO (ml-models bucket)
2. Reads the trained IsolationForest model, scaler, thresholds, and feature names
3. Creates a JSON message with base64-encoded model bytes
4. Publishes to Kafka topic 'if-model-updates' (compacted)
5. Flink's BroadcastStateLoader picks it up and distributes to all operators

Usage:
  python load_model_to_broadcast.py --version v1 --bootstrap kafka:9092

Environment:
  MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY (default: minio:9000 / minioadmin / minioadmin123)
  MODEL_BUCKET (default: ml-models)

MinIO artifact keys (from train_model.py):
  anomaly_detector_<version>.pkl    - trained IsolationForest model
  anomaly_scaler_<version>.pkl     - fitted StandardScaler
  thresholds_<version>.json         - per-feature baseline thresholds
  feature_names_<version>.txt       - ordered list of 21 feature names
"""

import argparse
import base64
import json
import os
import sys
import tempfile
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


def download_from_minio(bucket, key, endpoint, access_key, secret_key):
    """Download a file from MinIO to a temporary file."""
    from minio import Minio
    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False
    )
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(key)[1]
    )
    client.fget_object(bucket, key, tmp.name)
    tmp.close()
    return tmp.name


def main():
    parser = argparse.ArgumentParser(
        description='Load model from MinIO and publish to Kafka for Flink BroadcastState'
    )
    parser.add_argument(
        '--version', default='v1',
        help='Model version (default: v1)'
    )
    parser.add_argument(
        '--bootstrap', default='kafka:9092',
        help='Kafka bootstrap servers (default: kafka:9092)'
    )
    parser.add_argument(
        '--topic', default='if-model-updates',
        help='Kafka topic (default: if-model-updates)'
    )
    args = parser.parse_args()

    # MinIO config (must match train_model.py)
    endpoint = os.getenv('MINIO_ENDPOINT', 'minio:9000')
    access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
    secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
    bucket = os.getenv('MODEL_BUCKET', 'ml-models')

    # Artifact keys MUST match train_model.py upload keys
    model_key = f'anomaly_detector_{args.version}.pkl'
    scaler_key = f'anomaly_scaler_{args.version}.pkl'
    thresholds_key = f'thresholds_{args.version}.json'
    feature_key = f'feature_names_{args.version}.txt'

    print(f"[ModelLoader] Downloading model artifacts from MinIO ({endpoint}/{bucket})...")
    print(f"[ModelLoader] Version: {args.version}")

    # Download all artifacts
    model_path = download_from_minio(bucket, model_key, endpoint, access_key, secret_key)
    scaler_path = download_from_minio(bucket, scaler_key, endpoint, access_key, secret_key)
    thresholds_path = download_from_minio(bucket, thresholds_key, endpoint, access_key, secret_key)
    feature_path = download_from_minio(bucket, feature_key, endpoint, access_key, secret_key)

    # Read model and scaler bytes
    with open(model_path, 'rb') as f:
        model_bytes = f.read()
    with open(scaler_path, 'rb') as f:
        scaler_bytes = f.read()
    with open(feature_path, 'rb') as f:
        feature_names = [fn.strip() for fn in f.read().decode('utf-8').splitlines() if fn.strip()]
    with open(thresholds_path, 'rb') as f:
        thresholds_data = json.load(f)

    # Build message matching BroadcastStateLoaderFunction.process_broadcast_element
    # Key names must match what the Flink operator expects:
    #   model_bytes, scaler_bytes, thresholds_json
    msg = {
        'version': args.version,
        'timestamp': datetime.utcnow().isoformat(),
        'model_bytes': base64.b64encode(model_bytes).decode('utf-8'),
        'scaler_bytes': base64.b64encode(scaler_bytes).decode('utf-8'),
        'thresholds_json': json.dumps(thresholds_data),
        'feature_names': feature_names,
    }

    # Cleanup temp files
    for p in [model_path, scaler_path, thresholds_path, feature_path]:
        try:
            os.unlink(p)
        except Exception:
            pass

    # Publish to Kafka
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable

    try:
        producer = KafkaProducer(
            bootstrap_servers=args.bootstrap,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3,
        )
    except NoBrokersAvailable:
        print(f"[ModelLoader] ERROR: Cannot connect to Kafka at {args.bootstrap}")
        print("[ModelLoader] Make sure Kafka is running and accessible.")
        sys.exit(1)

    print(f"[ModelLoader] Publishing to Kafka topic '{args.topic}'...")
    future = producer.send(args.topic, value=msg)
    record_metadata = future.get(timeout=10)
    producer.flush()
    producer.close()

    print(f"[ModelLoader] SUCCESS!")
    print(f"  Topic     : {record_metadata.topic}")
    print(f"  Partition : {record_metadata.partition}")
    print(f"  Offset    : {record_metadata.offset}")
    print(f"  Version   : {args.version}")
    print(f"  Model     : {len(model_bytes):,} bytes")
    print(f"  Scaler    : {len(scaler_bytes):,} bytes")
    print(f"  Features  : {len(feature_names)}")
    print(f"  Thresholds: {list(thresholds_data.keys())}")


if __name__ == '__main__':
    main()
