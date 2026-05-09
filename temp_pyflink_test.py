#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, '/tmp')
sys.path.insert(0, '/tmp/src')

print("Testing PyFlink imports...")
from pyflink.datastream import StreamExecutionEnvironment
print("StreamExecutionEnvironment: OK")

env = StreamExecutionEnvironment.get_execution_environment()
print(f"Environment: {env}")
print(f"Parallelism: {env.get_parallelism()}")

# Test key imports
print("\nTesting project module imports...")
from src.operators.key_generator import generate_trip_id
print("key_generator: OK")

from src.features.vectorizer import FeatureVectorizer
print("vectorizer: OK")

from src.operators.schema_validator import SchemaValidator
print("schema_validator: OK")

from src.sinks.postgres_sink import record_to_raw_trips_row
print("postgres_sink: OK")

# Test model loading
print("\nTesting model loading...")
import pickle
import json
with open('/tmp/models/iforest_model.pkl', 'rb') as f:
    model = pickle.load(f)
print(f"Model: {type(model).__name__}")
print(f"n_trees: {model.n_estimators}")

with open('/tmp/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print(f"Scaler: {type(scaler).__name__}")

with open('/tmp/models/context_thresholds.json', 'r') as f:
    thresh = json.load(f)
print(f"Thresholds: {thresh['version']}, {len(thresh['thresholds'])} contexts")

print("\n=== All PyFlink imports OK ===")
