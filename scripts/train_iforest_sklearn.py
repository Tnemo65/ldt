#!/usr/bin/env python3
"""
Train sklearn IsolationForest + compute context-aware thresholds.
Replaces existing HalfSpaceTrees model with sklearn IsolationForest.
"""
import sys
import pickle
import json
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import parallel_backend

sys.path.insert(0, 'c:/proj/ldt')
from src.features.vectorizer import FeatureVectorizer

print("=" * 70)
print("TRAINING sklearn IsolationForest + Context Thresholds")
print("=" * 70)

N_TREES = 200
MAX_SAMPLES = 256
CONTAMINATION = 0.001
SCALER_PATH = 'c:/proj/ldt/models/scaler.pkl'
BASELINE_PATH = 'c:/proj/ldt/data/clean/jan_2024_clean_baseline.parquet'
SAMPLE_SIZE = 500000
OUTPUT_DIR = Path('c:/proj/ldt/models')

# 1. Load scaler
print("\n[1] Loading scaler...")
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
print(f"    Scaler: {type(scaler).__name__}")

# 2. Load and sample data for training
print(f"\n[2] Loading data (sample={SAMPLE_SIZE:,})...")
t0 = time.time()
df = pd.read_parquet(BASELINE_PATH)
print(f"    Total: {len(df):,} records")
df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
print(f"    Sampled: {len(df_sample):,} records in {time.time()-t0:.1f}s")

# 3. Vectorize + Scale training sample
print(f"\n[3] Vectorizing + scaling {len(df_sample):,} records...")
t0 = time.time()
vectorizer = FeatureVectorizer()
X_sample = vectorizer.transform_batch(df_sample)
X_sample_scaled = scaler.transform(X_sample)
print(f"    Done: {X_sample_scaled.shape} in {time.time()-t0:.1f}s")
del df_sample, X_sample
gc.collect()

# 4. Train sklearn IsolationForest
print(f"\n[4] Training sklearn IsolationForest (n_trees={N_TREES}, max_samples={MAX_SAMPLES})...")
from sklearn.ensemble import IsolationForest
t0 = time.time()
model = IsolationForest(
    n_estimators=N_TREES,
    max_samples=MAX_SAMPLES,
    contamination=CONTAMINATION,
    max_features=1.0,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
with parallel_backend('threading', n_jobs=-1):
    model.fit(X_sample_scaled)
train_time = time.time() - t0
print(f"    Training complete in {train_time:.1f}s ({len(X_sample_scaled)/train_time:.0f} rec/s)")

# Quick validation
scores_sample = model.score_samples(X_sample_scaled[:1000])
print(f"    Score range: [{scores_sample.min():.4f}, {scores_sample.max():.4f}]")
del X_sample_scaled
gc.collect()

# 5. Save sklearn model (backup old first)
print(f"\n[5] Saving sklearn model...")
backup_path = OUTPUT_DIR / 'iforest_model_backup.pkl'
original_path = OUTPUT_DIR / 'iforest_model.pkl'
with open(original_path, 'rb') as f:
    old_bytes = f.read()
with open(backup_path, 'wb') as f:
    f.write(old_bytes)
print(f"    Backed up old model to: {backup_path}")

with open(original_path, 'wb') as f:
    pickle.dump(model, f)
size_mb = original_path.stat().st_size / 1e6
print(f"    Saved sklearn model: {original_path} ({size_mb:.1f} MB)")
print(f"    Model: n_estimators={model.n_estimators}, max_samples={model.max_samples}")

# 6. Compute context-aware thresholds from FULL baseline
print(f"\n[6] Computing context-aware thresholds from FULL baseline ({len(pd.read_parquet(BASELINE_PATH)):,} records)...")
t0 = time.time()
df_full = pd.read_parquet(BASELINE_PATH)
print(f"    Loaded full baseline: {len(df_full):,} records")

# Vectorize all
X_full = vectorizer.transform_batch(df_full)
X_full_scaled = scaler.transform(X_full)
del df_full
gc.collect()

# Score all records (negate sklearn scores: higher = more anomalous)
print(f"    Scoring all records...")
raw_scores = model.score_samples(X_full_scaled)  # sklearn: lower = more anomalous
anomaly_scores = -raw_scores  # now higher = more anomalous
del X_full_scaled
gc.collect()

# Compute context keys (vectorized)
print(f"    Computing context keys...")
df_full = pd.read_parquet(BASELINE_PATH)
pickup_times = pd.to_datetime(df_full['tpep_pickup_datetime'])
distances = df_full['trip_distance'].fillna(0).values
zone_ids = df_full['PULocationID'].fillna(0).values.astype(int)
hours = pickup_times.dt.hour.values
weekdays = pickup_times.dt.weekday.values

# Vectorized context key computation
trip_types = np.where(distances < 2, 'short',
             np.where(distances < 10, 'medium', 'long'))

time_windows = np.where((hours >= 6) & (hours < 10), 'morning_rush',
               np.where((hours >= 17) & (hours < 20), 'evening_rush',
               np.where((hours >= 22) | (hours < 6), 'night', 'midday')))

day_types = np.where(weekdays >= 5, 'weekend', 'weekday')

neighborhoods = np.where(zone_ids <= 50, 'manhattan',
               np.where(zone_ids <= 100, 'brooklyn',
               np.where(zone_ids <= 150, 'queens',
               np.where(zone_ids <= 200, 'bronx',
               np.where(np.isin(zone_ids, [132, 138]), 'airport', 'staten_island')))))

context_keys = [f"{t}_{w}_{d}_{n}" for t,w,d,n in zip(trip_types, time_windows, day_types, neighborhoods)]
del df_full
gc.collect()

print(f"    Building context buckets...")
from collections import defaultdict
context_buckets = defaultdict(list)
for i, ctx_key in enumerate(context_keys):
    context_buckets[ctx_key].append(anomaly_scores[i])
    if (i + 1) % 500000 == 0:
        print(f"      Processed {i+1:,} records...")

print(f"    Built {len(context_buckets)} context buckets")

# Compute 95th and 98th percentile thresholds
print(f"\n[7] Computing thresholds (95th + 98th percentile)...")
thresholds_95 = {}
thresholds_98 = {}
for ctx_key, scores_list in context_buckets.items():
    thresholds_95[ctx_key] = float(np.percentile(scores_list, 95))
    thresholds_98[ctx_key] = float(np.percentile(scores_list, 98))

global_thresh_95 = float(np.percentile(anomaly_scores, 95))
global_thresh_98 = float(np.percentile(anomaly_scores, 98))
print(f"    Global thresholds: 95th={global_thresh_95:.4f}, 98th={global_thresh_98:.4f}")

# Save thresholds v1 (95th)
thresh_v1 = {
    'version': '1.0',
    'percentile': 95,
    'global_threshold': global_thresh_95,
    'thresholds': thresholds_95,
    'model_type': 'sklearn_IsolationForest',
    'n_trees': N_TREES,
    'max_samples': MAX_SAMPLES,
    'training_sample': SAMPLE_SIZE,
    'n_contexts': len(thresholds_95),
}
v1_path = OUTPUT_DIR / 'context_thresholds.json'
with open(v1_path, 'w') as f:
    json.dump(thresh_v1, f, indent=2)
print(f"    Saved v1: {v1_path}")

# Save thresholds v2 (98th)
thresh_v2 = {
    'version': '2.0',
    'percentile': 98,
    'old_percentile': 95,
    'global_threshold': global_thresh_98,
    'thresholds': thresholds_98,
    'model_type': 'sklearn_IsolationForest',
    'n_trees': N_TREES,
    'max_samples': MAX_SAMPLES,
    'training_sample': SAMPLE_SIZE,
    'n_contexts': len(thresholds_98),
}
v2_path = OUTPUT_DIR / 'context_thresholds_v2.json'
with open(v2_path, 'w') as f:
    json.dump(thresh_v2, f, indent=2)
print(f"    Saved v2: {v2_path}")

print(f"\n{'=' * 70}")
print(f"COMPLETE!")
print(f"  Model: {original_path}")
print(f"  Config: n_estimators={N_TREES}, max_samples={MAX_SAMPLES}")
print(f"  Training time: {train_time:.1f}s on {SAMPLE_SIZE:,} records")
print(f"  Contexts: {len(thresholds_95)} buckets")
print(f"  Global thresh (95th): {global_thresh_95:.4f}")
print(f"  Global thresh (98th): {global_thresh_98:.4f}")
print(f"  Backup: {backup_path}")
print(f"{'=' * 70}")
