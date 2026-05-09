import sys, pickle, json
sys.path.insert(0, '/tmp')
sys.path.insert(0, '/tmp/src')

from src.operators.if_scoring_operator import get_context_key
from src.features.vectorizer import FeatureVectorizer

m = pickle.load(open('/tmp/models/iforest_model.pkl', 'rb'))
s = pickle.load(open('/tmp/models/scaler.pkl', 'rb'))
t = json.load(open('/tmp/models/context_thresholds.json'))
v = FeatureVectorizer()

print(f'Model: {m.n_estimators} trees')
print(f'Thresholds: {len(t.get("thresholds", {}))} contexts')
print(f'Global threshold: {t.get("global_threshold", 0):.4f}')

# Test scoring
test_record = {
    'trip_distance': 5.0,
    'fare_amount': 15.0,
    'total_amount': 20.0,
    'passenger_count': 2,
    'tpep_pickup_datetime': '2024-01-15T10:30:00',
    'tpep_dropoff_datetime': '2024-01-15T10:45:00',
    'PULocationID': 100,
    'DOLocationID': 150,
}
features = v.transform(test_record)
features_scaled = s.transform([features])[0]
raw_score = m.score_samples(features_scaled.reshape(1, -1))[0]
anomaly_score = -raw_score
ctx = get_context_key(test_record)
thresh = t.get('thresholds', {}).get(ctx, t.get('global_threshold', 0.5))
print(f'Test score: {anomaly_score:.4f} (threshold: {thresh:.4f})')
print(f'Context: {ctx}')
print('ML scoring OK')
