#!/usr/bin/env python3
import sys
sys.path.insert(0, '/tmp')
sys.path.insert(0, '/tmp/src')

# Test importing project modules
try:
    from src.features.vectorizer import FeatureVectorizer
    print("FeatureVectorizer: OK")
except Exception as e:
    print(f"FeatureVectorizer: FAIL - {e}")

try:
    from src.operators.key_generator import generate_trip_id
    print("key_generator: OK")
except Exception as e:
    print(f"key_generator: FAIL - {e}")

try:
    from src.operators.canary_rules import CanaryRules
    print("canary_rules: OK")
except Exception as e:
    print(f"canary_rules: FAIL - {e}")

try:
    from src.operators.if_scoring_operator import get_context_key
    print("if_scoring_operator: OK")
except Exception as e:
    print(f"if_scoring_operator: FAIL - {e}")

try:
    import sklearn
    import pandas
    import numpy
    import joblib
    import mmh3
    print(f"ML libs: sklearn={sklearn.__version__}, pandas={pandas.__version__}, numpy={numpy.__version__}, joblib={joblib.__version__}")
except Exception as e:
    print(f"ML libs: FAIL - {e}")

# Test basic IsolationForest load
try:
    import pickle
    with open('/tmp/models/iforest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"IsolationForest loaded: {type(model)}")
    print(f"n_trees: {model.n_estimators}, max_samples: {model.max_samples}")
except Exception as e:
    print(f"IsolationForest load: FAIL - {e}")

# Test scaler
try:
    import pickle
    with open('/tmp/models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded: {type(scaler)}")
except Exception as e:
    print(f"Scaler load: FAIL - {e}")

# Test thresholds
try:
    import json
    with open('/tmp/models/context_thresholds.json', 'r') as f:
        thresholds = json.load(f)
    print(f"Thresholds loaded: version={thresholds['version']}, percentile={thresholds['percentile']}, n_keys={len(thresholds['thresholds'])}")
except Exception as e:
    print(f"Thresholds load: FAIL - {e}")

print("\n=== All checks complete ===")
