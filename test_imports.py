#!/usr/bin/env python3
"""Test all imports for E2E pipeline."""
import os, sys, traceback
sys.path.insert(0, '/opt/flink/e2e')
sys.path.insert(0, '/opt/flink/e2e/src')
os.environ['KAFKA_BOOTSTRAP_SERVERS'] = 'kafka:9092'
os.environ['POSTGRES_HOST'] = 'postgres'
os.environ['POSTGRES_PORT'] = '5432'
os.environ['POSTGRES_DB'] = 'dq_pipeline'
os.environ['POSTGRES_USER'] = 'cadqstream'
os.environ['POSTGRES_PASSWORD'] = 'cadqstream123'

print("=" * 60)
print("Testing E2E Pipeline Imports")
print("=" * 60)

try:
    print("[1] PyFlink...")
    from pyflink.datastream import StreamExecutionEnvironment
    print("  OK")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

try:
    print("[2] key_generator...")
    from src.operators.key_generator import generate_trip_id
    print("  OK")
except Exception as e:
    print(f"  FAIL: {e}")

try:
    print("[3] deduplicator...")
    from src.operators.deduplicator import DeduplicatorFunction
    print("  OK")
except Exception as e:
    print(f"  FAIL: {e}")

try:
    print("[4] schema_validator...")
    from src.operators.schema_validator import SchemaValidator
    print("  OK")
except Exception as e:
    print(f"  FAIL: {e}")

try:
    print("[5] canary_rules...")
    from src.operators.canary_rules import CanaryRulesValidator, ViolationFilter
    print("  OK")
except Exception as e:
    print(f"  FAIL: {e}")

try:
    print("[6] if_scoring_operator...")
    from src.operators.if_scoring_operator import IFScoringOperator, get_context_key
    print("  OK")
except Exception as e:
    print(f"  FAIL: {e}")

try:
    print("[7] vectorizer...")
    from src.features.vectorizer import FeatureVectorizer
    print("  OK")
except Exception as e:
    print(f"  FAIL: {e}")

try:
    print("[8] ML models...")
    import pickle, json
    model = pickle.load(open('/opt/flink/e2e/models/iforest_model.pkl', 'rb'))
    print(f"  Model: {model.n_estimators} trees")
    scaler = pickle.load(open('/opt/flink/e2e/models/scaler.pkl', 'rb'))
    print("  Scaler OK")
    with open('/opt/flink/e2e/models/context_thresholds.json') as f:
        t = json.load(f)
    print(f"  Thresholds: global={t.get('global_threshold')}")
    vec = FeatureVectorizer()
    print("  Vectorizer OK")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

try:
    print("[9] Test sample record...")
    sample = {
        'VendorID': 1, 'tpep_pickup_datetime': '2024-01-01 00:00:00',
        'tpep_dropoff_datetime': '2024-01-01 00:30:00',
        'passenger_count': 1.0, 'trip_distance': 5.0,
        'PULocationID': 148, 'DOLocationID': 141,
        'fare_amount': 25.0, 'total_amount': 35.0,
        'trip_duration': 0.5, 'speed_mph': 10.0,
    }
    trip_id = generate_trip_id(sample)
    print(f"  trip_id: {trip_id}")
    ctx = get_context_key(sample)
    print(f"  context_key: {ctx}")
    feats = vec.transform(sample)
    print(f"  features: {len(feats)}D")
    print("  OK")
except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()

print("=" * 60)
print("Import Test Complete")
print("=" * 60)
