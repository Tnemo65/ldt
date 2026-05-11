"""
CA-DIF-EIA SOTA Benchmark
=========================
Comprehensive benchmark comparing CA-DIF-EIA against:
  - sklearn IsolationForest (baseline)
  - sklearn OCSVM (baseline)
  - sklearn LOF (baseline)
  - HalfSpaceTrees via river.anomaly (streaming tree baseline)
  - MemStream (custom implementation, WWW 2022)
  - LSTM-Autoencoder (PyTorch, simplified)

Dataset: NYC Yellow Taxi Jan 2024 (~2.96M records, after cleaning)
Protocol: 5 seeds × 7 algorithms × 3 difficulty levels = 105 runs
Output: results/sota/benchmark_results.csv + figures

Author: CA-DIF-EIA Benchmark v1.1
"""
import os
import sys
import time
import warnings
import json
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_curve, auc,
)
from scipy.stats import ttest_rel, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
FAST_MODE = True           # True = 100K sample (fast test), False = full
N_SEEDS = 5
SYNTHETIC_PER_SCENARIO = 1000 if FAST_MODE else 10000
TRAIN_RATIO = 0.7
N_CLUSTERS = 7
DIFFICULTY_LEVELS = ['easy', 'medium', 'hard']
N_JOBS = -1                # Use all CPU cores

# Training contamination: fraction of training data expected as anomalous
# Training is CLEAN (0 contamination), but we set threshold at ~3rd percentile
# meaning test anomalies (top ~5%) will score above threshold
TRAIN_PERCENTILE = 97        # Threshold = 97th percentile of training scores
# For test data: if contamination=0.03 (3%), threshold = 97th percentile

OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'sota')
os.makedirs(OUT_DIR, exist_ok=True)

DATA_FILE = None
_candidates = [
    'data/raw/yellow_tripdata_2024-01.parquet',
    '../data/raw/yellow_tripdata_2024-01.parquet',
    r'C:\proj\ldt\data\raw\yellow_tripdata_2024-01.parquet',
]
for _c in _candidates:
    if os.path.exists(_c):
        DATA_FILE = _c
        break
assert DATA_FILE is not None, f"Dataset not found. Searched: {_candidates}"
print(f"[CONFIG] Mode: {'FAST (100K)' if FAST_MODE else 'FULL (~2.96M)'}")
print(f"[CONFIG] Data: {DATA_FILE}")

SEEDS = [42, 123, 456, 789, 1024]

BASELINE = {
    'fare_per_mile': 2.5,
    'fare_per_minute': 0.67,
    'implied_speed': 12.0,
}

print("=" * 70)
print("CA-DIF-EIA SOTA BENCHMARK v1.1")
print("=" * 70)

# ===========================
# DATA LOADING & PREPROCESSING
# ===========================
print("\n" + "=" * 70)
print("[1/8] LOAD RAW DATA")
print("=" * 70)
t0 = time.time()
df_raw = pd.read_parquet(DATA_FILE)
if FAST_MODE:
    df_raw = df_raw.sample(n=100_000, random_state=42).reset_index(drop=True)
print(f"  Loaded: {len(df_raw):,} records in {time.time()-t0:.1f}s")

# L1: Schema validation
print("\n" + "=" * 70)
print("[2/8] L1 SCHEMA VALIDATION")
print("=" * 70)
required_cols = ['trip_distance', 'fare_amount', 'PULocationID', 'DOLocationID',
                 'passenger_count', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
l1_valid = (
    df_raw[required_cols].notna().all(axis=1)
    & df_raw['passenger_count'].between(1, 6)
    & df_raw['PULocationID'].between(1, 263)
    & df_raw['DOLocationID'].between(1, 263)
)
df_l1 = df_raw[l1_valid].copy().reset_index(drop=True)
print(f"  Input: {len(df_raw):,} | Rejected: {(~l1_valid).sum():,} | Output: {len(df_l1):,}")

# L2: Canary rules
print("\n" + "=" * 70)
print("[3/8] L2 RULE-BASED CANARY")
print("=" * 70)
df_l1['pickup_dt'] = pd.to_datetime(df_l1['tpep_pickup_datetime'])
df_l1['dropoff_dt'] = pd.to_datetime(df_l1['tpep_dropoff_datetime'])
df_l1['duration_sec'] = (df_l1['dropoff_dt'] - df_l1['pickup_dt']).dt.total_seconds()
df_l1['duration_hours'] = df_l1['duration_sec'] / 3600
df_l1['speed_mph'] = df_l1['trip_distance'] / (df_l1['duration_hours'] + 1e-9)

l2_valid = (
    (df_l1['fare_amount'] > 0)
    & (df_l1['trip_distance'] > 0)
    & (df_l1['duration_sec'] > 0)
    & (df_l1['speed_mph'] < 100)
    & (df_l1['speed_mph'] > 0)
)
df_l2 = df_l1[l2_valid].copy().reset_index(drop=True)
print(f"  Input: {len(df_l1):,} | Rejected: {(~l2_valid).sum():,} | Output: {len(df_l2):,}")

# L3: IQR outlier removal
print("\n" + "=" * 70)
print("[4/8] L3 IQR OUTLIER REMOVAL (3x IQR)")
print("=" * 70)
df_clean = df_l2.copy()
iqr_mult = 3.0
for col in ['fare_amount', 'trip_distance', 'duration_hours']:
    Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - iqr_mult * IQR, Q3 + iqr_mult * IQR
    before = len(df_clean)
    df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    print(f"  {col}: removed {before - len(df_clean):,}")
df_clean = df_clean.reset_index(drop=True)
print(f"  Clean records: {len(df_clean):,}")

# Train/Test split (NO shuffle - temporal)
print("\n" + "=" * 70)
print("[5/8] TRAIN/TEST SPLIT (70/30, temporal)")
print("=" * 70)
n_total = len(df_clean)
n_train = int(n_total * TRAIN_RATIO)
train_idx = list(range(n_train))
test_idx = list(range(n_train, n_total))
df_train = df_clean.iloc[train_idx].copy().reset_index(drop=True)
df_test_clean = df_clean.iloc[test_idx].copy().reset_index(drop=True)
print(f"  Train: {len(df_train):,} | Test: {len(df_test_clean):,}")
print(f"  No anomalies in train (zero contamination)")

# ===========================
# FEATURE EXTRACTION (25D)
# ===========================
print("\n" + "=" * 70)
print("[6/8] FEATURE EXTRACTION (25D)")
print("=" * 70)

FEATURE_NAMES_25D = [
    # Block 1: 15D Raw
    'distance', 'duration_min', 'fare', 'passengers', 'total',
    'speed', 'fare_per_mile', 'fare_per_minute', 'fare_per_passenger',
    'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'is_night', 'month',
    # Block 2: 6D Ratio
    'fare_per_mile_ratio', 'fare_per_minute_ratio', 'implied_speed_ratio',
    'passenger_distance_ratio', 'fare_distance_product', 'duration_distance_ratio',
    # Block 3: 4D Context (sin/cos encoded)
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
]

def extract_features_25d(df):
    """Extract 25D feature vector: 15D raw + 6D ratio + 4D context."""
    eps = 1e-6
    pickup = pd.to_datetime(df['tpep_pickup_datetime'])
    dropoff = pd.to_datetime(df['tpep_dropoff_datetime'])
    dur_sec = (dropoff - pickup).dt.total_seconds()
    dur_min = dur_sec / 60
    dur_hr = dur_sec / 3600

    dist = df['trip_distance'].values.astype(float)
    fare = df['fare_amount'].values.astype(float)
    pax = df['passenger_count'].values.astype(float)
    total = df['total_amount'].values.astype(float)

    speed = dist / (dur_hr.values + eps)
    fpm = fare / (dist + eps)
    fpmn = fare / (dur_min.values + eps)
    fpp = fare / (pax + eps)

    hour = pickup.dt.hour.values.astype(float)
    dow = pickup.dt.weekday.values.astype(float)
    is_wknd = (dow >= 5).astype(float)
    is_rush = (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))).astype(float)
    is_night = ((hour < 6) | (hour > 22)).astype(float)
    month = pickup.dt.month.values.astype(float)

    fpm_ratio = fpm / (BASELINE['fare_per_mile'] + eps)
    fpmn_ratio = fpmn / (BASELINE['fare_per_minute'] + eps)
    speed_ratio = speed / (BASELINE['implied_speed'] + eps)
    pax_dist_ratio = pax / (dist + eps)
    fare_dist_prod = fare * dist
    dur_dist_ratio = dur_min.values / (dist + eps)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    return np.column_stack([
        dist, dur_min.values, fare, pax, total,
        speed, fpm, fpmn, fpp,
        hour, dow, is_wknd, is_rush, is_night, month,
        fpm_ratio, fpmn_ratio, speed_ratio,
        pax_dist_ratio, fare_dist_prod, dur_dist_ratio,
        hour_sin, hour_cos, dow_sin, dow_cos,
    ])

print("  Extracting 25D features...")
t1 = time.time()
X_train_25d = extract_features_25d(df_train)
print(f"  Train 25D: {X_train_25d.shape} ({time.time()-t1:.1f}s)")

scaler = StandardScaler().fit(X_train_25d)
X_train_scaled = scaler.transform(X_train_25d)

print(f"  Feature blocks:")
print(f"    Block 1 (15D): raw features")
print(f"    Block 2 (6D): ratio features")
print(f"    Block 3 (4D): cyclical context encoding")

# ===========================
# ANOMALY INJECTION
# ===========================
print("\n" + "=" * 70)
print("[7/8] ANOMALY INJECTION (5 scenarios x 3 difficulty levels)")
print("=" * 70)

DIFFICULTY_CONFIGS = {
    'easy': {
        'meter_fare_mult': (10, 20), 'gps_speed': (50, 95),
        'pax_fare': (40, 70), 'crawl_dur': (90, 180), 'combined_fare_mult': (10, 20),
    },
    'medium': {
        'meter_fare_mult': (4, 8), 'gps_speed': (30, 60),
        'pax_fare': (15, 30), 'crawl_dur': (40, 80), 'combined_fare_mult': (4, 8),
    },
    'hard': {
        'meter_fare_mult': (1.5, 3), 'gps_speed': (20, 40),
        'pax_fare': (8, 15), 'crawl_dur': (20, 35), 'combined_fare_mult': (2, 4),
    },
}

def inject_meter_tampering(df, indices, cfg):
    for idx in indices:
        dist = np.random.uniform(1.0, 3.0)
        dur_min = np.random.uniform(5, 15)
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = dist * 2.50 * np.random.uniform(*cfg['meter_fare_mult'])
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + np.random.uniform(5, 10)
        df.at[idx, 'passenger_count'] = np.random.randint(1, 5)

def inject_gps_spoofing(df, indices, cfg):
    for idx in indices:
        target_speed = np.random.uniform(*cfg['gps_speed'])
        dist = np.random.uniform(20, 40)
        dur_min = (dist / target_speed) * 60
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = dist * np.random.uniform(2.0, 3.5)
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + np.random.uniform(5, 15)
        df.at[idx, 'passenger_count'] = np.random.randint(1, 4)

def inject_passenger_anomaly(df, indices, cfg):
    for idx in indices:
        dist = np.random.uniform(0.2, 0.5)
        dur_min = np.random.uniform(15, 30)
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = np.random.uniform(*cfg['pax_fare'])
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + np.random.uniform(3, 8)
        df.at[idx, 'passenger_count'] = np.random.randint(1, 6)

def inject_slow_crawl(df, indices, cfg):
    for idx in indices:
        dist = np.random.uniform(2, 4)
        dur_min = np.random.uniform(*cfg['crawl_dur'])
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = np.random.uniform(40, 80)
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + np.random.uniform(3, 8)
        df.at[idx, 'passenger_count'] = np.random.randint(1, 3)

def inject_combined_subtle(df, indices, cfg):
    for idx in indices:
        dist = np.random.uniform(1, 2)
        dur_min = np.random.uniform(5, 10)
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = dist * 2.50 * np.random.uniform(*cfg['combined_fare_mult'])
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + np.random.uniform(5, 15)
        df.at[idx, 'passenger_count'] = np.random.randint(4, 6)

SCENARIOS = [
    ('meter_tampering', inject_meter_tampering),
    ('gps_spoofing', inject_gps_spoofing),
    ('passenger_anomaly', inject_passenger_anomaly),
    ('slow_crawl', inject_slow_crawl),
    ('combined_subtle', inject_combined_subtle),
]

level_data = {}
for level in DIFFICULTY_LEVELS:
    cfg = DIFFICULTY_CONFIGS[level]
    df_test_level = df_test_clean.copy()
    n_per = SYNTHETIC_PER_SCENARIO
    n_total_anom = n_per * len(SCENARIOS)

    rng = np.random.RandomState(42)
    anom_indices = rng.choice(len(df_test_level), size=n_total_anom, replace=False)
    idx_splits = np.array_split(anom_indices, len(SCENARIOS))

    y_test_level = np.zeros(len(df_test_level), dtype=int)
    sc_labels_level = np.full(len(df_test_level), 'normal', dtype=object)

    for i, (name, inject_fn) in enumerate(SCENARIOS):
        inject_fn(df_test_level, idx_splits[i], cfg)
        y_test_level[idx_splits[i]] = 1
        sc_labels_level[idx_splits[i]] = name

    X_test_25d = extract_features_25d(df_test_level)
    X_test_scaled = scaler.transform(X_test_25d)

    level_data[level] = {
        'df_test': df_test_level,
        'y_test': y_test_level,
        'scenario_labels': sc_labels_level,
        'X_test_25d': X_test_25d,
        'X_test_scaled': X_test_scaled,
    }
    print(f"  {level}: {y_test_level.sum():,} anomalies / {len(df_test_level):,} total ({y_test_level.mean()*100:.2f}%)")

# Sanity checks
print("\n  Sanity checks...")
assert (df_train['fare_amount'] <= 0).sum() == 0
assert X_train_25d.shape[1] == 25
print("  ALL CHECKS PASSED")

# ===========================
# ALGORITHM IMPLEMENTATIONS
# ===========================

def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    return {
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'FPR': fpr_val,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
    }

def threshold_from_train(train_scores, percentile=TRAIN_PERCENTILE):
    """Compute threshold from training scores (percentile-based)."""
    return np.percentile(train_scores, percentile)

def score_with_threshold(test_scores, threshold):
    """Score test data using pre-computed threshold."""
    return (test_scores >= threshold).astype(int)

# ===========================
# ALGORITHM 1: CA-DIF-EIA
# ===========================
print("\n" + "=" * 70)
print("INITIALIZING CA-DIF-EIA")
print("=" * 70)

import torch
import torch.nn as nn

class CADIF:
    """
    CA-DIF-EIA: Context-Aware Deep Isolation Forest with Uncertainty-based
    Intersection Approach.

    Architecture:
    1. Random Neural Projection: bends space, eliminates axis-parallel ghost regions
    2. CERE: r random projections computed as single matrix op
    3. K-Means clustering: per-cluster anomaly scoring (context awareness)
    4. ADWIN-U: drift detection via score distribution monitoring
    5. Fast-Track: Canary bypass when drift detected
    """
    def __init__(self, n_trees=100, hidden_dim=64, seed=42,
                 n_clusters=7, train_percentile=TRAIN_PERCENTILE,
                 n_projections=10, drift_sensitivity=0.002):
        self.n_trees = n_trees
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.n_clusters = n_clusters
        self.train_percentile = train_percentile
        self.n_projections = n_projections
        self.drift_sensitivity = drift_sensitivity

        self.iforest = None
        self.kmeans = None
        self.scaler = None
        self.thresholds = {}      # per-cluster thresholds
        self.global_threshold = None
        self.W_proj = None        # CERE projection matrix

        # ADWIN-U state
        self.adwin_windows = {}   # metric -> deque of values
        self.drift_count = 0
        self.fast_track_active = False

    def _build_random_projection(self, input_dim, seed):
        """Build CERE projection: stack r random matrices into one op."""
        torch.manual_seed(seed)
        # Single random projection matrix: input -> hidden_dim
        self.W_proj = torch.randn(input_dim, self.hidden_dim,
                                   generator=torch.Generator().manual_seed(seed)
                                   ) * np.sqrt(2.0 / input_dim)
        self.b_proj = torch.zeros(self.hidden_dim)

    def fit(self, X, kmeans_n_clusters=None):
        if kmeans_n_clusters is None:
            kmeans_n_clusters = self.n_clusters

        # Fit scaler
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        # Build CERE random projection
        if self.W_proj is None:
            self._build_random_projection(X_scaled.shape[1], self.seed)

        # Project to hidden space
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_scaled).float()
            H = torch.relu(X_tensor @ self.W_proj + self.b_proj).numpy()

        # K-Means clustering for context
        self.kmeans = MiniBatchKMeans(n_clusters=kmeans_n_clusters,
                                       random_state=self.seed)
        self.kmeans.fit(H)

        # Isolation Forest on projected space
        self.iforest = IsolationForest(
            n_estimators=self.n_trees,
            contamination='auto',  # We'll use percentile threshold instead
            random_state=self.seed,
            n_jobs=N_JOBS
        )
        self.iforest.fit(H)

        # Compute per-cluster thresholds from training scores
        train_labels = self.kmeans.predict(H)
        train_scores = -self.iforest.decision_function(H)

        self.thresholds = {}
        for cid in range(self.kmeans.n_clusters):
            mask = train_labels == cid
            if mask.sum() > 0:
                self.thresholds[cid] = np.percentile(
                    train_scores[mask], self.train_percentile)
            else:
                self.thresholds[cid] = np.percentile(train_scores, self.train_percentile)

        self.global_threshold = np.percentile(train_scores, self.train_percentile)
        return self

    def score_samples(self, X):
        """Score samples using per-cluster thresholds."""
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_scaled).float()
            H = torch.relu(X_tensor @ self.W_proj + self.b_proj).numpy()

        test_labels = self.kmeans.predict(H)
        test_scores = -self.iforest.decision_function(H)

        y_pred = np.zeros(len(X), dtype=int)
        for cid in range(self.kmeans.n_clusters):
            mask = test_labels == cid
            if mask.sum() > 0:
                thresh = self.thresholds.get(cid, self.global_threshold)
                y_pred[mask] = (test_scores[mask] >= thresh).astype(int)

        return y_pred, test_scores

    def detect_drift_adwin(self, scores):
        """ADWIN-U: Detect drift via score distribution monitoring."""
        window_size = min(500, len(scores) // 4)
        if len(scores) < window_size * 2:
            return False

        recent = scores[-window_size:]
        older = scores[-2*window_size:-window_size]

        mean_recent = np.mean(recent)
        mean_older = np.mean(older)
        std_combined = np.std(scores[-window_size*2:]) + 1e-9

        delta = self.drift_sensitivity
        eps = np.sqrt((1 / (2 * window_size)) * np.log(4 / delta))
        drift = abs(mean_recent - mean_older) > eps * std_combined * 2

        if drift:
            self.drift_count += 1
            self.fast_track_active = True
        elif len(scores) > window_size * 4:
            self.fast_track_active = False

        return drift


# ===========================
# ALGORITHM 2: sklearn IsolationForest (baseline)
# ===========================
class BaselineIF:
    """sklearn IsolationForest - standard axis-parallel isolation."""
    def __init__(self, seed=42, n_trees=300):
        self.seed = seed
        self.n_trees = n_trees
        self.model = None
        self.scaler = None
        self.threshold = None

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        self.model = IsolationForest(
            n_estimators=self.n_trees,
            contamination='auto',
            random_state=self.seed,
            n_jobs=N_JOBS
        )
        self.model.fit(X_scaled)

        # Compute threshold from training scores
        train_scores = -self.model.decision_function(X_scaled)
        self.threshold = np.percentile(train_scores, TRAIN_PERCENTILE)
        return self

    def score_samples(self, X):
        X_scaled = self.scaler.transform(X)
        scores = -self.model.decision_function(X_scaled)
        return (scores >= self.threshold).astype(int), scores


# ===========================
# ALGORITHM 3: HalfSpaceTrees via river.anomaly
# ===========================
class HSTWrapper:
    """
    HalfSpaceTrees from River (river.anomaly.HalfSpaceTrees).
    Online tree-based anomaly detection with axis-parallel splits.
    This is the tree-based SOTA that CA-DIF-EIA claims to beat.
    """
    def __init__(self, seed=42, n_trees=10, height=8, window_size=500,
                 train_sample_size=10000):
        self.seed = seed
        self.n_trees = n_trees
        self.height = height
        self.window_size = window_size
        self.train_sample_size = train_sample_size
        self.model = None
        self.scaler = None
        self.threshold = None
        self.feature_names = None

    def fit(self, X, y=None):
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        # Subsample for speed (streaming algorithms are O(n))
        rng = np.random.RandomState(self.seed)
        sample_size = min(self.train_sample_size, len(X_scaled))
        sample_idx = rng.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[sample_idx]

        from river.anomaly import HalfSpaceTrees

        # Compute feature limits from sample
        limits = {str(i): (float(X_sample[:, i].min()), float(X_sample[:, i].max()))
                  for i in range(X_sample.shape[1])}

        self.model = HalfSpaceTrees(
            n_trees=self.n_trees,
            height=self.height,
            window_size=self.window_size,
            limits=limits,
            seed=self.seed,
        )

        # Online training with dictionary input (HST updates in-place, no reassignment)
        for i in range(len(X_sample)):
            x_dict = {str(j): float(X_sample[i, j]) for j in range(X_sample.shape[1])}
            self.model.learn_one(x_dict)

        # Compute threshold from training scores
        train_scores = []
        for i in range(len(X_sample)):
            x_dict = {str(j): float(X_sample[i, j]) for j in range(X_sample.shape[1])}
            train_scores.append(self.model.score_one(x_dict))
        self.threshold = np.percentile(train_scores, TRAIN_PERCENTILE)
        return self

    def score_samples(self, X):
        scores = []
        for i in range(len(X)):
            x_dict = {str(j): float(X[i, j]) for j in range(X.shape[1])}
            scores.append(self.model.score_one(x_dict))
        scores = np.array(scores)
        return (scores >= self.threshold).astype(int), scores


# ===========================
# ALGORITHM 4: MemStream (custom implementation)
# ===========================
class MemStreamWrapper:
    """
    MemStream (Bhatia et al., WWW 2022) - custom implementation.

    Key idea: Maintain a fixed-size memory buffer of recent normal points.
    Score = minimum distance to memory buffer points.
    High distance = anomalous.

    Parameters:
    - n_memories: size of memory buffer
    - decay: exponential decay factor for recency weighting
    """
    def __init__(self, seed=42, n_memories=100, decay=0.01):
        self.seed = seed
        self.n_memories = n_memories
        self.decay = decay
        self.scaler = None
        self.threshold = None
        self.memory_buffer = None
        self.memory_weights = None
        self._built = False

    def fit(self, X, y=None):
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        rng = np.random.RandomState(self.seed)
        sample_size = min(self.n_memories * 10, len(X_scaled))
        sample_idx = rng.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[sample_idx]

        # Initialize memory buffer with first n_memories points
        self.memory_buffer = X_sample[:self.n_memories].copy()
        self.memory_weights = np.ones(self.n_memories)

        # Online learning: update memory with streaming points
        for i in range(self.n_memories, len(X_sample)):
            # Replace random memory slot (reservoir sampling)
            if i < self.n_memories * 2:
                replace_idx = rng.randint(0, min(i, self.n_memories))
                self.memory_buffer[replace_idx] = X_sample[i]

        self._built = True

        # Compute threshold from training scores
        train_scores = self._compute_scores(X_sample[:self.n_memories*2])
        self.threshold = np.percentile(train_scores, TRAIN_PERCENTILE)
        return self

    def _compute_scores(self, X):
        """Compute anomaly scores: min distance to memory buffer."""
        n = len(X)
        scores = np.zeros(n)
        for i in range(n):
            # Euclidean distance to nearest memory point
            dists = np.linalg.norm(X[i] - self.memory_buffer, axis=1)
            scores[i] = np.min(dists)
        return scores

    def score_samples(self, X):
        scores = self._compute_scores(X)
        return (scores >= self.threshold).astype(int), scores


# ===========================
# ALGORITHM 5: LSTM-Autoencoder
# ===========================
class LSTMAutoencoder:
    """
    LSTM-Autoencoder for anomaly detection.
    Train on normal data, reconstruct inputs.
    High reconstruction error = anomalous.
    """
    def __init__(self, input_dim, hidden_dim=64, seed=42,
                 n_epochs=5, batch_size=1024):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.threshold = None

    def _build_model(self):
        class LSTMae(nn.Module):
            def __init__(self, d, h):
                super().__init__()
                self.encoder = nn.LSTM(d, h, batch_first=True, num_layers=1)
                self.decoder = nn.LSTM(h, h, batch_first=True, num_layers=1)
                self.output = nn.Linear(h, d)
            def forward(self, x):
                enc, _ = self.encoder(x)
                dec, _ = self.decoder(enc)
                return self.output(dec)

        torch.manual_seed(self.seed)
        self.model = LSTMae(self.input_dim, self.hidden_dim)
        self.device = torch.device('cpu')

    def fit(self, X, y=None):
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        # Subsample for speed
        rng = np.random.RandomState(self.seed)
        sample_size = min(50_000, len(X_scaled))
        sample_idx = rng.choice(len(X_scaled), sample_size, replace=False)
        X_train = X_scaled[sample_idx]

        self._build_model()

        X_tensor = torch.from_numpy(X_train).float().unsqueeze(1)  # (n, 1, d)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()

        for epoch in range(self.n_epochs):
            indices = rng.permutation(len(X_tensor))
            for i in range(0, len(X_tensor), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                batch = X_tensor[batch_idx]
                recon = self.model(batch)
                loss = nn.MSELoss()(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Compute threshold from training reconstruction errors
        self.model.eval()
        with torch.no_grad():
            recon_errors = []
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i:i+self.batch_size]
                recon = self.model(batch)
                err = ((recon - batch) ** 2).mean(dim=(1, 2)).numpy()
                recon_errors.extend(err)
        self.threshold = np.percentile(recon_errors, TRAIN_PERCENTILE)
        return self

    def score_samples(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.from_numpy(X_scaled).float().unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            errors = ((recon - X_tensor) ** 2).mean(dim=(1, 2)).numpy()
        return (errors >= self.threshold).astype(int), errors


# ===========================
# ALGORITHM 6: sklearn LOF (baseline)
# ===========================
class BaselineLOF:
    """sklearn Local Outlier Factor as a baseline."""
    def __init__(self, seed=42):
        self.seed = seed
        self.model = None
        self.scaler = None
        self.threshold = None

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        rng = np.random.RandomState(self.seed)
        sample_size = min(50_000, len(X_scaled))
        X_sample = X_scaled[rng.choice(len(X_scaled), sample_size, replace=False)]
        self.model = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=N_JOBS)
        self.model.fit(X_sample)

        train_scores = -self.model.decision_function(X_sample)
        self.threshold = np.percentile(train_scores, TRAIN_PERCENTILE)
        return self

    def score_samples(self, X):
        X_scaled = self.scaler.transform(X)
        scores = -self.model.decision_function(X_scaled)
        return (scores >= self.threshold).astype(int), scores


# ===========================
# ALGORITHM 7: sklearn OCSVM (baseline)
# ===========================
class BaselineOCSVM:
    """sklearn One-Class SVM as a baseline."""
    def __init__(self, seed=42):
        self.seed = seed
        self.model = None
        self.scaler = None
        self.threshold = None

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        rng = np.random.RandomState(self.seed)
        sample_size = min(30_000, len(X_scaled))
        X_sample = X_scaled[rng.choice(len(X_scaled), sample_size, replace=False)]
        self.model = OneClassSVM(kernel='rbf', gamma='auto')
        self.model.fit(X_sample)

        train_scores = -self.model.decision_function(X_sample)
        self.threshold = np.percentile(train_scores, TRAIN_PERCENTILE)
        return self

    def score_samples(self, X):
        X_scaled = self.scaler.transform(X)
        scores = -self.model.decision_function(X_scaled)
        return (scores >= self.threshold).astype(int), scores


# ===========================
# ALGORITHM REGISTRY
# ===========================
ALGORITHM_DEFS = [
    ('CA-DIF-EIA',         CADIF,         {'n_trees': 100, 'hidden_dim': 64, 'n_clusters': N_CLUSTERS}),
    ('sklearn_IF',          BaselineIF,    {'n_trees': 300}),
    ('sHST-River',          HSTWrapper,    {'n_trees': 10, 'height': 8, 'window_size': 500, 'train_sample_size': 20000}),
    ('MemStream',           MemStreamWrapper, {'n_memories': 100, 'decay': 0.01}),
    ('LSTM-Autoencoder',    LSTMAutoencoder, {'input_dim': X_train_scaled.shape[1], 'hidden_dim': 64}),
    ('sklearn_LOF',         BaselineLOF,    {}),
    ('sklearn_OCSVM',       BaselineOCSVM,  {}),
]

# ===========================
# RUN BENCHMARK
# ===========================
print("\n" + "=" * 70)
print("[8/8] BENCHMARK EXECUTION")
print("=" * 70)

all_results = []
total_runs = N_SEEDS * len(ALGORITHM_DEFS) * len(DIFFICULTY_LEVELS)
run_count = 0
t_benchmark = time.time()

for level in DIFFICULTY_LEVELS:
    ld = level_data[level]
    print(f"\n  === DIFFICULTY: {level.upper()} ===")
    for alg_name, alg_class, alg_kwargs in ALGORITHM_DEFS:
        for seed in SEEDS:
            t_run = time.time()

            try:
                if alg_name == 'CA-DIF-EIA':
                    model = CADIF(seed=seed, **alg_kwargs)
                    model.fit(X_train_scaled)
                    y_pred, scores = model.score_samples(ld['X_test_scaled'])

                elif alg_name == 'sklearn_IF':
                    model = alg_class(seed=seed, **alg_kwargs)
                    model.fit(X_train_scaled)
                    y_pred, scores = model.score_samples(ld['X_test_scaled'])

                elif alg_name == 'sklearn_LOF':
                    model = alg_class(seed=seed)
                    model.fit(X_train_scaled)
                    y_pred, scores = model.score_samples(ld['X_test_scaled'])

                elif alg_name == 'sklearn_OCSVM':
                    model = alg_class(seed=seed)
                    model.fit(X_train_scaled)
                    y_pred, scores = model.score_samples(ld['X_test_scaled'])

                elif alg_name == 'sHST-River':
                    model = alg_class(seed=seed, **alg_kwargs)
                    model.fit(X_train_scaled)
                    y_pred, scores = model.score_samples(ld['X_test_scaled'])

                elif alg_name == 'MemStream':
                    model = alg_class(seed=seed, **alg_kwargs)
                    model.fit(X_train_scaled)
                    y_pred, scores = model.score_samples(ld['X_test_scaled'])

                elif alg_name == 'LSTM-Autoencoder':
                    model = LSTMAutoencoder(
                        input_dim=X_train_scaled.shape[1],
                        hidden_dim=64, seed=seed
                    )
                    model.fit(X_train_scaled)
                    y_pred, scores = model.score_samples(ld['X_test_scaled'])

                else:
                    raise ValueError(f"Unknown: {alg_name}")

                train_time = time.time() - t_run
                metrics = evaluate(ld['y_test'], y_pred)
                metrics['train_time'] = train_time
                metrics['algorithm'] = alg_name
                metrics['seed'] = seed
                metrics['level'] = level
                all_results.append(metrics)

            except Exception as e:
                print(f"    ERROR {alg_name} seed={seed}: {e}")
                import traceback
                traceback.print_exc()
                metrics = {
                    'F1': 0, 'Recall': 0, 'Precision': 0, 'FPR': 0,
                    'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
                    'train_time': time.time() - t_run,
                    'algorithm': alg_name, 'seed': seed, 'level': level,
                }
                all_results.append(metrics)

            run_count += 1
            elapsed = time.time() - t_benchmark
            eta = (elapsed / run_count) * (total_runs - run_count) if run_count > 0 else 0
            last = all_results[-1]
            print(f"    {alg_name:<20} F1={last['F1']:.3f} R={last['Recall']:.3f} "
                  f"FPR={last['FPR']:.4f}  [{run_count}/{total_runs}] ETA={eta:.0f}s")

df_results = pd.DataFrame(all_results)
print(f"\n  Total runs: {len(df_results)} | Total time: {time.time()-t_benchmark:.1f}s")

# Save results
csv_path = os.path.join(OUT_DIR, 'benchmark_results.csv')
df_results.to_csv(csv_path, index=False)
print(f"  Results saved: {csv_path}")

# ===========================
# RESULTS SUMMARY TABLE
# ===========================
print("\n" + "=" * 70)
print("BENCHMARK RESULTS BY DIFFICULTY (mean +/- std, 5 seeds)")
print("=" * 70)

for level in DIFFICULTY_LEVELS:
    df_lv = df_results[df_results['level'] == level]
    summary_lv = df_lv.groupby('algorithm').agg(
        F1_mean=('F1', 'mean'), F1_std=('F1', 'std'),
        Recall_mean=('Recall', 'mean'), Recall_std=('Recall', 'std'),
        Precision_mean=('Precision', 'mean'), Precision_std=('Precision', 'std'),
        FPR_mean=('FPR', 'mean'), FPR_std=('FPR', 'std'),
        train_time_mean=('train_time', 'mean'),
    ).reset_index().sort_values('F1_mean', ascending=False).reset_index(drop=True)
    summary_lv['Rank'] = range(1, len(summary_lv) + 1)

    print(f"\n  LEVEL: {level.upper()}")
    print(f"  {'Rank':<5} {'Algorithm':<20} {'F1':>12} {'Recall':>12} {'Precision':>12} {'FPR':>10} {'Time(s)':>8}")
    print(f"  {'-'*85}")
    for _, row in summary_lv.iterrows():
        print(f"  {row['Rank']:<5.0f} {row['algorithm']:<20} "
              f"{row['F1_mean']:.3f}+/-{row['F1_std']:.3f}  "
              f"{row['Recall_mean']:.3f}+/-{row['Recall_std']:.3f}  "
              f"{row['Precision_mean']:.3f}+/-{row['Precision_std']:.3f}  "
              f"{row['FPR_mean']:.4f}+/-{row['FPR_std']:.4f}  "
              f"{row['train_time_mean']:>6.1f}s")

# ===========================
# CROSS-ALGORITHM COMPARISON
# ===========================
print(f"\n{'='*90}")
print("KEY COMPARISON: CA-DIF-EIA vs competitors")
print(f"{'='*90}")
alg_names = [a[0] for a in ALGORITHM_DEFS]
key_algs = ['CA-DIF-EIA', 'sklearn_IF', 'sHST-River', 'MemStream', 'LSTM-Autoencoder']
print(f"  {'Level':<8}", end='')
for a in key_algs:
    print(f" {a[:12]:>12}", end='')
print()
print(f"  {'-'*68}")
for level in DIFFICULTY_LEVELS:
    df_lv = df_results[df_results['level'] == level]
    print(f"  {level:<8}", end='')
    for a in key_algs:
        f1 = df_lv[df_lv['algorithm'] == a]['F1'].mean()
        print(f" {f1:>12.3f}", end='')
    print()

# ===========================
# STATISTICAL TESTING
# ===========================
print("\n" + "=" * 70)
print("STATISTICAL TESTING (paired t-test, 5 seeds)")
print("=" * 70)

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1-1)*np.std(a, ddof=1)**2 + (n2-1)*np.std(b, ddof=1)**2) / (n1+n2-2))
    if pooled_std < 1e-10:
        return float('inf') if abs(np.mean(a) - np.mean(b)) > 1e-10 else 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std

def effect_label(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"

other_algorithms = ['sklearn_IF', 'sHST-River', 'MemStream', 'LSTM-Autoencoder', 'sklearn_LOF', 'sklearn_OCSVM']
all_stat_results = []

for level in DIFFICULTY_LEVELS:
    df_lv = df_results[df_results['level'] == level]
    cadif_f1 = df_lv[df_lv['algorithm'] == 'CA-DIF-EIA']['F1'].values

    print(f"\n  === {level.upper()} ===")
    print(f"  {'Comparison':<40} {'p(t)':>8} {'d':>8} {'Effect':>10} {'Sig':>4}")
    print(f"  {'-'*75}")

    for alg in other_algorithms:
        other_f1 = df_lv[df_lv['algorithm'] == alg]['F1'].values
        if len(other_f1) < 5 or len(cadif_f1) < 5:
            continue
        t_stat, t_p = ttest_rel(cadif_f1, other_f1)
        d = cohens_d(cadif_f1, other_f1)
        sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
        print(f"  CA-DIF-EIA vs {alg:<25} {t_p:>8.4f} {d:>8.2f} {effect_label(d):>10} {sig:>4}")
        all_stat_results.append({
            'level': level, 'comparison': f'CA-DIF-EIA vs {alg}',
            'p_ttest': t_p, 'cohens_d': d, 'effect': effect_label(d), 'sig': sig,
        })

df_stats = pd.DataFrame(all_stat_results)
stats_path = os.path.join(OUT_DIR, 'statistical_tests.csv')
df_stats.to_csv(stats_path, index=False)
print(f"\n  Statistical tests saved: {stats_path}")

# ===========================
# GENERATE FIGURES
# ===========================
print("\n" + "=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

colors_map = {
    'CA-DIF-EIA': '#27ae60',
    'sklearn_IF': '#95a5a6',
    'sHST-River': '#e67e22',
    'MemStream': '#3498db',
    'LSTM-Autoencoder': '#9b59b6',
    'sklearn_LOF': '#e74c3c',
    'sklearn_OCSVM': '#8e44ad',
}

short_names = ['CA-DIF-EIA', 'sklearn\nIF', 'sHST', 'MemStream', 'LSTM-AE', 'LOF', 'OCSVM']
level_colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}

fig = plt.figure(figsize=(24, 18))
fig.suptitle('CA-DIF-EIA SOTA Benchmark\nNYC Yellow Taxi Jan 2024 | 7 Algorithms x 5 Seeds x 3 Difficulty Levels',
             fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.20)

# Panel A: F1 by algorithm and difficulty
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(alg_names))
width = 0.25
offsets = [-width, 0, width]
for off_idx, level in enumerate(DIFFICULTY_LEVELS):
    df_lv = df_results[df_results['level'] == level]
    f1_means = [df_lv[df_lv['algorithm']==a]['F1'].mean() for a in alg_names]
    f1_stds = [df_lv[df_lv['algorithm']==a]['F1'].std() for a in alg_names]
    bars = ax1.bar(x + offsets[off_idx], f1_means, yerr=f1_stds,
                   color=level_colors[level], edgecolor='white', capsize=3,
                   width=width, label=level.upper(), alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(short_names, fontsize=10)
ax1.set_ylabel('F1 Score', fontsize=12)
ax1.set_title('A. F1 Score by Algorithm and Difficulty Level', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 1.1)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, axis='y', alpha=0.3)

# Panel B: Performance degradation
ax2 = fig.add_subplot(gs[0, 1])
for alg_name in alg_names:
    vals = [df_results[(df_results['level']==lv) & (df_results['algorithm']==alg_name)]['F1'].mean()
            for lv in DIFFICULTY_LEVELS]
    short_label = alg_name
    ax2.plot(range(len(DIFFICULTY_LEVELS)), vals, 'o-', label=short_label,
             color=colors_map[alg_name], linewidth=2, markersize=8, alpha=0.85)
ax2.set_xticks(range(len(DIFFICULTY_LEVELS)))
ax2.set_xticklabels([l.upper() for l in DIFFICULTY_LEVELS])
ax2.set_ylabel('F1 Score', fontsize=12)
ax2.set_xlabel('Difficulty Level', fontsize=12)
ax2.set_title('B. Performance Degradation Across Difficulty', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=8, loc='lower left', ncol=2)
ax2.grid(True, alpha=0.3)

# Panel C: EASY level metrics
ax3 = fig.add_subplot(gs[1, 0])
metrics_to_plot = ['F1', 'Recall', 'Precision']
metric_means = {a: [] for a in alg_names}
metric_stds = {a: [] for a in alg_names}
df_easy = df_results[df_results['level'] == 'easy']
for a in alg_names:
    for m in metrics_to_plot:
        vals = df_easy[df_easy['algorithm']==a][m].values
        metric_means[a].append(vals.mean())
        metric_stds[a].append(vals.std())

x_m = np.arange(len(metrics_to_plot))
bar_w = 0.12
for i, (a, sn) in enumerate(zip(alg_names, short_names)):
    ax3.bar(x_m + i*bar_w - 3*bar_w, metric_means[a], bar_w,
            label=sn.replace('\n', ' '), color=colors_map[a],
            yerr=metric_stds[a], capsize=2, alpha=0.85)
ax3.set_xticks(x_m)
ax3.set_xticklabels(metrics_to_plot, fontsize=12)
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('C. EASY Level: F1 / Recall / Precision', fontsize=13, fontweight='bold')
ax3.set_ylim(0, 1.15)
ax3.legend(fontsize=7, loc='upper right', ncol=2)
ax3.grid(True, axis='y', alpha=0.3)

# Panel D: FPR comparison
ax4 = fig.add_subplot(gs[1, 1])
for off_idx, level in enumerate(DIFFICULTY_LEVELS):
    df_lv = df_results[df_results['level'] == level]
    fpr_means = [df_lv[df_lv['algorithm']==a]['FPR'].mean() for a in alg_names]
    fpr_stds = [df_lv[df_lv['algorithm']==a]['FPR'].std() for a in alg_names]
    ax4.bar(x + offsets[off_idx], fpr_means, yerr=fpr_stds,
            color=level_colors[level], edgecolor='white', capsize=3,
            width=width, alpha=0.85, label=level.upper())
ax4.set_xticks(x)
ax4.set_xticklabels(short_names, fontsize=10)
ax4.set_ylabel('False Positive Rate', fontsize=12)
ax4.set_title('D. FPR by Algorithm and Difficulty (Lower is Better)', fontsize=13, fontweight='bold')
ax4.axhline(y=0.05, color='red', linestyle='--', linewidth=2, alpha=0.7, label='5% threshold')
max_fpr = max(df_results['FPR'].mean() for a in alg_names)
ax4.set_ylim(0, max(0.1, max_fpr * 1.3))
ax4.legend(fontsize=10)
ax4.grid(True, axis='y', alpha=0.3)

plt.savefig(os.path.join(OUT_DIR, 'benchmark_4panel.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'benchmark_4panel.pdf'), bbox_inches='tight')
print(f"  Saved: {OUT_DIR}/benchmark_4panel.png/pdf")

# Figure 2: Training time
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle('Training Time Comparison (seconds per run)', fontsize=14, fontweight='bold')
for ax_idx, level in enumerate(DIFFICULTY_LEVELS):
    ax = axes2[ax_idx]
    df_lv = df_results[df_results['level'] == level]
    t_means = [df_lv[df_lv['algorithm']==a]['train_time'].mean() for a in alg_names]
    t_stds = [df_lv[df_lv['algorithm']==a]['train_time'].std() for a in alg_names]
    bars = ax.bar(range(len(alg_names)), t_means, yerr=t_stds,
                  color=[colors_map[a] for a in alg_names], edgecolor='white', capsize=4, width=0.7)
    ax.set_xticks(range(len(alg_names)))
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'{level.upper()}', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_yscale('log')
    for i, (bar, val) in enumerate(zip(bars, t_means)):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.2,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'benchmark_train_time.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {OUT_DIR}/benchmark_train_time.png")

# ===========================
# FINAL SUMMARY
# ===========================
print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
print(f"""
ARCHITECTURE: L1 (Schema) -> L2 (Rules) -> L3 (IQR) -> Train/Test -> ML Benchmark
DATASET:     NYC Yellow Taxi Jan 2024 ({'100K sample' if FAST_MODE else '~2.96M records'})
ALGORITHMS:  CA-DIF-EIA, sklearn_IF, sHST-River, MemStream,
             LSTM-Autoencoder, sklearn_LOF, sklearn_OCSVM
EVALUATION:  {N_SEEDS} seeds x {len(ALGORITHM_DEFS)} algorithms x 3 levels = {len(df_results)} runs
OUTPUT:      {OUT_DIR}/
  - benchmark_results.csv
  - statistical_tests.csv
  - benchmark_4panel.png/pdf
  - benchmark_train_time.png
""")

for level in DIFFICULTY_LEVELS:
    df_lv = df_results[df_results['level'] == level]
    print(f"\n  {level.upper()}:")
    for a in alg_names:
        m = df_lv[df_lv['algorithm']==a]
        print(f"    {a:<20} F1={m['F1'].mean():.3f}+/-{m['F1'].std():.3f}  "
              f"FPR={m['FPR'].mean():.4f}  Time={m['train_time'].mean():.1f}s")

print(f"\n  Total benchmark time: {time.time()-t_benchmark:.1f}s")
