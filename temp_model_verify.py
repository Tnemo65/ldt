#!/usr/bin/env python3
import sys
sys.path.insert(0, '/tmp')
sys.path.insert(0, '/tmp/src')

import pickle
import numpy as np
import json

print("=" * 60)
print("=== MODEL VERIFICATION ===")
print("=" * 60)

# Test IsolationForest model
try:
    with open('/tmp/models/iforest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"Model type: {type(model).__module__}.{type(model).__name__}")
    print(f"  n_trees: {model.n_trees}")
    print(f"  window_size: {model.window_size}")
    print(f"  seed: {model.seed}")

    # Test scoring
    test_vec = np.random.randn(21).astype(np.float64)
    score = model.score_one(test_vec)
    print(f"  score_one (random): {score:.4f}")

    # Score with more extreme values
    extreme_vec = np.array([100, 500, 100, 10, 10, 100, 50, 50, 5, 5, 15, 3, 0, 1, 0], dtype=np.float64)
    score2 = model.score_one(extreme_vec)
    print(f"  score_one (extreme): {score2:.4f}")
except Exception as e:
    print(f"Model load/score FAIL: {e}")

print()

# Test scaler
try:
    with open('/tmp/models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler: {type(scaler).__module__}.{type(scaler).__name__}")
    test = np.random.randn(21).reshape(1, -1)
    scaled = scaler.transform(test)
    print(f"  transform OK: shape={scaled.shape}")
except Exception as e:
    print(f"Scaler FAIL: {e}")

print()

# Test thresholds
try:
    with open('/tmp/models/context_thresholds.json', 'r') as f:
        thresholds = json.load(f)
    print(f"Thresholds: version={thresholds['version']}, percentile={thresholds['percentile']}")
    print(f"  Global threshold: {thresholds['global_threshold']:.4f}")
    print(f"  Context keys: {len(thresholds['thresholds'])}")
except Exception as e:
    print(f"Thresholds FAIL: {e}")

print()

# Test neighborhood mapping
try:
    with open('/tmp/models/neighborhood_mapping.json', 'r') as f:
        nbr = json.load(f)
    print(f"Neighborhood mapping: {len(nbr)} zones")
except Exception as e:
    print(f"Neighborhood mapping FAIL: {e}")

print()
print("=" * 60)
print("=== VECTORIZER TEST ===")
print("=" * 60)

try:
    from src.features.vectorizer import FeatureVectorizer
    vec = FeatureVectorizer()
    test_record = {
        'trip_distance': 3.5,
        'tpep_pickup_datetime': '2024-01-15T14:30:00',
        'fare_amount': 15.0,
        'total_amount': 20.0,
        'passenger_count': 2,
        'PULocationID': 230,
        'DOLocationID': 230,
        'trip_duration': 900,
    }
    features = vec.transform(test_record)
    print(f"Vectorizer output: shape={features.shape}")
    print(f"  First 5: {features[:5]}")
except Exception as e:
    print(f"Vectorizer FAIL: {e}")

print()
print("=" * 60)
print("=== ALL CHECKS DONE ===")
print("=" * 60)
