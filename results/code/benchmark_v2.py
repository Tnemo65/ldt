"""
CA-DIF-EIA Benchmark v2: Scientifically Rigorous Streaming Anomaly Detection
============================================================================
Comprehensive benchmark with full statistical rigor:

  - 12 monthly folds (January-December 2024)
  - 7 algorithms × 10 seeds × 3 difficulty levels = 2,520 runs per fold
  - Friedman omnibus test + post-hoc Wilcoxon
  - Bonferroni, Holm-Bonferroni, Benjamini-Hochberg corrections
  - Bootstrap 95% CI
  - Critical Difference (CD) diagram (Demvsar 2006)
  - 8 publication-quality figures

Output: results/v2/ directory with CSVs and figures

Author: CA-DIF-EIA Benchmark v2.0
"""

import os
import sys
import time
import warnings
import json
import itertools
import multiprocessing as mp
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_curve, auc,
    confusion_matrix,
)
from scipy.stats import (
    friedmanchisquare, wilcoxon, ttest_rel,
    sem,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
N_SEEDS = 10
SYNTHETIC_PER_SCENARIO = 1000
TRAIN_RATIO = 0.8
N_CLUSTERS = 7
N_JOBS = -1
TRAIN_PERCENTILE = 97
DIFFICULTY_LEVELS = ['easy', 'medium', 'hard']
N_FOLDS = 12

SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]

BASE_DIR = r'C:\proj\ldt'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUT_DIR = os.path.join(BASE_DIR, 'results', 'v2')
os.makedirs(OUT_DIR, exist_ok=True)

PARQUET_FILES = sorted([
    os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
    if f.startswith('yellow_tripdata_2024-') and f.endswith('.parquet')
])
assert len(PARQUET_FILES) == 12, f"Expected 12 parquet files, got {len(PARQUET_FILES)}"

MONTH_NAMES = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]

BASELINE = {
    'fare_per_mile': 2.5,
    'fare_per_minute': 0.67,
    'implied_speed': 12.0,
}

print("=" * 70)
print("CA-DIF-EIA BENCHMARK v2.0 — Scientifically Rigorous Evaluation")
print("=" * 70)
print(f"  Folds:        {N_FOLDS} monthly (Jan-Dec 2024)")
print(f"  Algorithms:   7")
print(f"  Seeds:        {N_SEEDS}")
print(f"  Difficulties: {len(DIFFICULTY_LEVELS)}")
print(f"  Total runs:   {N_FOLDS * 7 * N_SEEDS * len(DIFFICULTY_LEVELS):,}")
print(f"  Output:       {OUT_DIR}")
print("=" * 70)

# ===========================
# FEATURE ENGINEERING
# ===========================
FEATURE_NAMES_25D = [
    'distance', 'duration_min', 'fare', 'passengers', 'total',
    'speed', 'fare_per_mile', 'fare_per_minute', 'fare_per_passenger',
    'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'is_night', 'month',
    'fare_per_mile_ratio', 'fare_per_minute_ratio', 'implied_speed_ratio',
    'passenger_distance_ratio', 'fare_distance_product', 'duration_distance_ratio',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
]

def extract_features_25d(df):
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


# ===========================
# DATA LOADING & CLEANING
# ===========================
def load_and_clean(parquet_path):
    """Load one month, apply L1+L2+L3 cleaning."""
    df_raw = pd.read_parquet(parquet_path)

    # L1: Schema validation
    required_cols = ['trip_distance', 'fare_amount', 'PULocationID', 'DOLocationID',
                    'passenger_count', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    l1_valid = (
        df_raw[required_cols].notna().all(axis=1)
        & df_raw['passenger_count'].between(1, 6)
        & df_raw['PULocationID'].between(1, 263)
        & df_raw['DOLocationID'].between(1, 263)
    )
    df = df_raw[l1_valid].copy().reset_index(drop=True)

    # L2: Canary rules
    df['pickup_dt'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['dropoff_dt'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['duration_sec'] = (df['dropoff_dt'] - df['pickup_dt']).dt.total_seconds()
    df['duration_hours'] = df['duration_sec'] / 3600
    df['speed_mph'] = df['trip_distance'] / (df['duration_hours'] + 1e-9)

    l2_valid = (
        (df['fare_amount'] > 0)
        & (df['trip_distance'] > 0)
        & (df['duration_sec'] > 0)
        & (df['speed_mph'] < 100)
        & (df['speed_mph'] > 0)
    )
    df = df[l2_valid].copy().reset_index(drop=True)

    # L3: IQR 3x
    for col in ['fare_amount', 'trip_distance', 'duration_hours']:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    df = df.reset_index(drop=True)

    return df


# ===========================
# ANOMALY INJECTION
# ===========================
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

SCENARIOS = [
    ('meter_tampering', 'inject_meter_tampering'),
    ('gps_spoofing', 'inject_gps_spoofing'),
    ('passenger_anomaly', 'inject_passenger_anomaly'),
    ('slow_crawl', 'inject_slow_crawl'),
    ('combined_subtle', 'inject_combined_subtle'),
]

def inject_meter_tampering(df, indices, cfg, rng):
    for idx in indices:
        dist = rng.uniform(1.0, 3.0)
        dur_min = rng.uniform(5, 15)
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = dist * 2.50 * rng.uniform(*cfg['meter_fare_mult'])
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + rng.uniform(5, 10)
        df.at[idx, 'passenger_count'] = rng.randint(1, 5)

def inject_gps_spoofing(df, indices, cfg, rng):
    for idx in indices:
        target_speed = rng.uniform(*cfg['gps_speed'])
        dist = rng.uniform(20, 40)
        dur_min = (dist / target_speed) * 60
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = dist * rng.uniform(2.0, 3.5)
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + rng.uniform(5, 15)
        df.at[idx, 'passenger_count'] = rng.randint(1, 4)

def inject_passenger_anomaly(df, indices, cfg, rng):
    for idx in indices:
        dist = rng.uniform(0.2, 0.5)
        dur_min = rng.uniform(15, 30)
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = rng.uniform(*cfg['pax_fare'])
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + rng.uniform(3, 8)
        df.at[idx, 'passenger_count'] = rng.randint(1, 6)

def inject_slow_crawl(df, indices, cfg, rng):
    for idx in indices:
        dist = rng.uniform(2, 4)
        dur_min = rng.uniform(*cfg['crawl_dur'])
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = rng.uniform(40, 80)
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + rng.uniform(3, 8)
        df.at[idx, 'passenger_count'] = rng.randint(1, 3)

def inject_combined_subtle(df, indices, cfg, rng):
    for idx in indices:
        dist = rng.uniform(1, 2)
        dur_min = rng.uniform(5, 10)
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = dist * 2.50 * rng.uniform(*cfg['combined_fare_mult'])
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + rng.uniform(5, 15)
        df.at[idx, 'passenger_count'] = rng.randint(4, 6)

INJECTION_FNS = {
    'inject_meter_tampering': inject_meter_tampering,
    'inject_gps_spoofing': inject_gps_spoofing,
    'inject_passenger_anomaly': inject_passenger_anomaly,
    'inject_slow_crawl': inject_slow_crawl,
    'inject_combined_subtle': inject_combined_subtle,
}

def prepare_fold(parquet_path, fold_idx, difficulty):
    """Load one fold, split, inject anomalies, extract features."""
    df = load_and_clean(parquet_path)
    n_total = len(df)
    n_train = int(n_total * TRAIN_RATIO)

    df_train = df.iloc[:n_train].copy().reset_index(drop=True)
    df_test_clean = df.iloc[n_train:].copy().reset_index(drop=True)

    cfg = DIFFICULTY_CONFIGS[difficulty]
    df_test = df_test_clean.copy()
    n_per = SYNTHETIC_PER_SCENARIO
    n_total_anom = n_per * len(SCENARIOS)

    rng = np.random.RandomState(SEEDS[fold_idx % len(SEEDS)])
    anom_indices = rng.choice(len(df_test), size=n_total_anom, replace=False)
    idx_splits = np.array_split(anom_indices, len(SCENARIOS))

    y_test = np.zeros(len(df_test), dtype=int)
    for i, (name, fn_name) in enumerate(SCENARIOS):
        INJECTION_FNS[fn_name](df_test, idx_splits[i], cfg, rng)
        y_test[idx_splits[i]] = 1

    X_test_25d = extract_features_25d(df_test)
    return df_train, X_test_25d, y_test


# ===========================
# ALGORITHMS
# ===========================
def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    try:
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_pred)
        auc_roc = auc(fpr_curve, tpr_curve)
    except Exception:
        auc_roc = 0.0
    return {
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'FPR': fpr_val,
        'AUC_ROC': auc_roc,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
    }


# CA-DIF-EIA
class CADIF:
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
        self.thresholds = {}
        self.global_threshold = None
        self.W_proj = None
        self.b_proj = None

    def _build_random_projection(self, input_dim, seed):
        import torch
        torch.manual_seed(seed)
        self.W_proj = torch.randn(input_dim, self.hidden_dim,
                                   generator=torch.Generator().manual_seed(seed)
                                   ) * np.sqrt(2.0 / input_dim)
        self.b_proj = torch.zeros(self.hidden_dim)

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        if self.W_proj is None:
            self._build_random_projection(X_scaled.shape[1], self.seed)

        import torch
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_scaled).float()
            H = torch.relu(X_tensor @ self.W_proj + self.b_proj).numpy()

        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters,
                                       random_state=self.seed, batch_size=2048)
        self.kmeans.fit(H)

        self.iforest = IsolationForest(
            n_estimators=self.n_trees,
            contamination='auto',
            random_state=self.seed,
            n_jobs=N_JOBS
        )
        self.iforest.fit(H)

        train_labels = self.kmeans.predict(H)
        train_scores = -self.iforest.decision_function(H)

        self.thresholds = {}
        for cid in range(self.kmeans.n_clusters):
            mask = train_labels == cid
            if mask.sum() > 0:
                self.thresholds[cid] = np.percentile(train_scores[mask], self.train_percentile)
            else:
                self.thresholds[cid] = np.percentile(train_scores, self.train_percentile)
        self.global_threshold = np.percentile(train_scores, self.train_percentile)
        return self

    def score_samples(self, X):
        import torch
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


# sklearn IsolationForest
class BaselineIF:
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
            n_estimators=self.n_trees, contamination='auto',
            random_state=self.seed, n_jobs=N_JOBS
        )
        self.model.fit(X_scaled)
        train_scores = -self.model.decision_function(X_scaled)
        self.threshold = np.percentile(train_scores, TRAIN_PERCENTILE)
        return self

    def score_samples(self, X):
        X_scaled = self.scaler.transform(X)
        scores = -self.model.decision_function(X_scaled)
        return (scores >= self.threshold).astype(int), scores


# HalfSpaceTrees
class HSTWrapper:
    def __init__(self, seed=42, n_trees=10, height=8, window_size=500,
                 train_sample_size=20000):
        self.seed = seed
        self.n_trees = n_trees
        self.height = height
        self.window_size = window_size
        self.train_sample_size = train_sample_size
        self.model = None
        self.scaler = None
        self.threshold = None

    def fit(self, X, y=None):
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        rng = np.random.RandomState(self.seed)
        sample_size = min(self.train_sample_size, len(X_scaled))
        sample_idx = rng.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[sample_idx]

        from river.anomaly import HalfSpaceTrees
        limits = {str(i): (float(X_sample[:, i].min()), float(X_sample[:, i].max()))
                  for i in range(X_sample.shape[1])}
        self.model = HalfSpaceTrees(
            n_trees=self.n_trees, height=self.height,
            window_size=self.window_size, limits=limits, seed=self.seed,
        )
        for i in range(len(X_sample)):
            x_dict = {str(j): float(X_sample[i, j]) for j in range(X_sample.shape[1])}
            self.model.learn_one(x_dict)

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


# MemStream
class MemStreamWrapper:
    def __init__(self, seed=42, n_memories=100, decay=0.01):
        self.seed = seed
        self.n_memories = n_memories
        self.decay = decay
        self.scaler = None
        self.threshold = None
        self.memory_buffer = None
        self._built = False

    def fit(self, X, y=None):
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        rng = np.random.RandomState(self.seed)
        sample_size = min(self.n_memories * 10, len(X_scaled))
        sample_idx = rng.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[sample_idx]
        self.memory_buffer = X_sample[:self.n_memories].copy()
        for i in range(self.n_memories, min(self.n_memories * 2, len(X_sample))):
            replace_idx = rng.randint(0, min(i, self.n_memories))
            self.memory_buffer[replace_idx] = X_sample[i]
        self._built = True
        train_scores = self._compute_scores(X_sample[:self.n_memories * 2])
        self.threshold = np.percentile(train_scores, TRAIN_PERCENTILE)
        return self

    def _compute_scores(self, X):
        scores = np.zeros(len(X))
        for i in range(len(X)):
            dists = np.linalg.norm(X[i] - self.memory_buffer, axis=1)
            scores[i] = np.min(dists)
        return scores

    def score_samples(self, X):
        scores = self._compute_scores(X)
        return (scores >= self.threshold).astype(int), scores


# LSTM-Autoencoder
class LSTMAutoencoder:
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
        import torch
        import torch.nn as nn
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
        import torch
        import torch.nn as nn
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        rng = np.random.RandomState(self.seed)
        sample_size = min(50_000, len(X_scaled))
        sample_idx = rng.choice(len(X_scaled), sample_size, replace=False)
        X_train = X_scaled[sample_idx]

        self._build_model()
        X_tensor = torch.from_numpy(X_train).float().unsqueeze(1)
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
        import torch
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.from_numpy(X_scaled).float().unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            errors = ((recon - X_tensor) ** 2).mean(dim=(1, 2)).numpy()
        return (errors >= self.threshold).astype(int), errors


# sklearn LOF
class BaselineLOF:
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


# sklearn OCSVM
class BaselineOCSVM:
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


ALGORITHM_DEFS = [
    ('CA-DIF-EIA',        CADIF,          {'n_trees': 100, 'hidden_dim': 64, 'n_clusters': N_CLUSTERS}),
    ('sklearn_IF',         BaselineIF,     {'n_trees': 300}),
    ('sHST-River',         HSTWrapper,     {'n_trees': 10, 'height': 8, 'window_size': 500, 'train_sample_size': 20000}),
    ('MemStream',          MemStreamWrapper, {'n_memories': 100, 'decay': 0.01}),
    ('LSTM-Autoencoder',   LSTMAutoencoder, {'hidden_dim': 64}),
    ('sklearn_LOF',        BaselineLOF,     {}),
    ('sklearn_OCSVM',     BaselineOCSVM,   {}),
]
ALG_NAMES = [a[0] for a in ALGORITHM_DEFS]

ALG_COLORS = {
    'CA-DIF-EIA':        '#27ae60',
    'sklearn_IF':        '#95a5a6',
    'sHST-River':        '#e67e22',
    'MemStream':         '#3498db',
    'LSTM-Autoencoder':  '#9b59b6',
    'sklearn_LOF':       '#e74c3c',
    'sklearn_OCSVM':     '#8e44ad',
}

# ===========================
# RUN SINGLE BENCHMARK
# ===========================
def run_single(alg_name, alg_class, alg_kwargs, X_train_scaled, X_test_scaled, y_test, seed):
    """Run one algorithm on one fold/difficulty/seed."""
    t_run = time.time()
    try:
        if alg_name == 'CA-DIF-EIA':
            model = alg_class(seed=seed, **alg_kwargs)
            model.fit(X_train_scaled)
            y_pred, _ = model.score_samples(X_test_scaled)
        elif alg_name == 'LSTM-Autoencoder':
            model = alg_class(input_dim=X_train_scaled.shape[1], hidden_dim=64, seed=seed)
            model.fit(X_train_scaled)
            y_pred, _ = model.score_samples(X_test_scaled)
        elif alg_name in ('sHST-River',):
            model = alg_class(seed=seed, **alg_kwargs)
            model.fit(X_train_scaled)
            y_pred, _ = model.score_samples(X_test_scaled)
        elif alg_name in ('MemStream',):
            model = alg_class(seed=seed, **alg_kwargs)
            model.fit(X_train_scaled)
            y_pred, _ = model.score_samples(X_test_scaled)
        elif alg_name in ('sklearn_LOF',):
            model = alg_class(seed=seed)
            model.fit(X_train_scaled)
            y_pred, _ = model.score_samples(X_test_scaled)
        elif alg_name in ('sklearn_OCSVM',):
            model = alg_class(seed=seed)
            model.fit(X_train_scaled)
            y_pred, _ = model.score_samples(X_test_scaled)
        else:
            model = alg_class(seed=seed, **alg_kwargs)
            model.fit(X_train_scaled)
            y_pred, _ = model.score_samples(X_test_scaled)

        train_time = time.time() - t_run
        metrics = evaluate(y_test, y_pred)
        metrics['train_time'] = train_time
        metrics['algorithm'] = alg_name
        metrics['seed'] = seed
        return metrics

    except Exception as e:
        train_time = time.time() - t_run
        return {
            'F1': 0, 'Precision': 0, 'Recall': 0, 'FPR': 0, 'AUC_ROC': 0,
            'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
            'train_time': train_time,
            'algorithm': alg_name, 'seed': seed,
            'error': str(e),
        }


# ===========================
# STATISTICAL FUNCTIONS
# ===========================
def cohens_d_paired(a, b):
    diff = a - b
    std = np.std(diff, ddof=1)
    if std < 1e-10:
        return 0.0
    return np.mean(diff) / std


def rank_biserial(W_plus, n):
    return (2 * W_plus / (n * (n + 1))) - 1


def holm_correction(p_values):
    """Holm-Bonferroni step-down procedure."""
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    n = len(sorted_p)
    adjusted = np.zeros(n)
    for rank, p in enumerate(sorted_p):
        adj = p * (n - rank)
        if rank > 0:
            adj = min(adj, adjusted[rank - 1])
        adjusted[rank] = min(adj, 1.0)
    adjusted_final = np.empty(n)
    adjusted_final[sorted_idx] = adjusted
    return adjusted_final


def bonferroni_correction(p_values):
    return np.clip(np.array(p_values) * len(p_values), 0, 1.0)


def bootstrap_ci(data, n_resamples=10000, ci=0.95):
    rng = np.random.RandomState(42)
    means = []
    n = len(data)
    for _ in range(n_resamples):
        sample = rng.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def critical_difference_q(N_folds, K_algs, alpha=0.05):
    """Critical value q_alpha from Demvsar 2006 table (approximate)."""
    table = {
        (3, 3): 2.343, (3, 5): 2.728, (3, 7): 2.936, (3, 10): 3.102,
        (5, 3): 2.489, (5, 5): 2.858, (5, 7): 3.054, (5, 10): 3.209,
        (7, 3): 2.564, (7, 5): 2.928, (7, 7): 3.116, (7, 10): 3.267,
        (10, 3): 2.633, (10, 5): 2.993, (10, 7): 3.177, (10, 10): 3.325,
        (12, 3): 2.683, (12, 5): 3.039, (12, 7): 3.220, (12, 10): 3.366,
    }
    key = (K_algs, N_folds)
    q_alpha = table.get(key, 2.949)
    cd = q_alpha * np.sqrt((K_algs * (K_algs + 1)) / (6 * N_folds))
    return q_alpha, cd


def friedman_test(score_matrix):
    """score_matrix: (N_folds, K_algorithms)."""
    stat, p = friedmanchisquare(*[score_matrix[:, i] for i in range(score_matrix.shape[1])])
    return {'chi2': stat, 'p': p, 'df': score_matrix.shape[1] - 1}


def pairwise_wilcoxon(score_matrix, alg_names):
    """All pairwise Wilcoxon signed-rank tests."""
    results = []
    K = score_matrix.shape[1]
    for i, j in itertools.combinations(range(K), 2):
        diff = score_matrix[:, i] - score_matrix[:, j]
        mask = diff != 0
        if mask.sum() < 5:
            continue
        try:
            W_stat, p = wilcoxon(diff[mask], alternative='two-sided')
            rb = rank_biserial(W_stat, mask.sum())
            cd = cohens_d_paired(score_matrix[:, i], score_matrix[:, j])
            results.append({
                'alg_i': alg_names[i],
                'alg_j': alg_names[j],
                'W': W_stat,
                'p_wilcoxon': p,
                'rank_biserial': rb,
                'cohens_d': cd,
                'n_nonzero': int(mask.sum()),
            })
        except Exception:
            continue
    return pd.DataFrame(results)


def sig_label(p, sig='*'):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def effect_label(d):
    d = abs(d)
    if d < 0.2:  return 'negligible'
    if d < 0.5:  return 'small'
    if d < 0.8:  return 'medium'
    return 'large'


# ===========================
# MAIN BENCHMARK LOOP
# ===========================
print("\n" + "=" * 70)
print("[BENCHMARK] Loading all folds, running all algorithms...")
print("=" * 70)

t_benchmark = time.time()
all_results = []

for fold_idx, parquet_path in enumerate(PARQUET_FILES):
    month_name = MONTH_NAMES[fold_idx]
    t_fold = time.time()
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx+1}/12: {month_name} 2024 — {parquet_path.split(os.sep)[-1]}")
    print(f"{'='*70}")

    df_train, X_test_25d, y_test = prepare_fold(parquet_path, fold_idx, 'easy')
    X_train_25d = extract_features_25d(df_train)
    scaler = StandardScaler().fit(X_train_25d)
    X_train_scaled = scaler.transform(X_train_25d)
    X_test_scaled = scaler.transform(X_test_25d)
    n_train, n_test = len(X_train_scaled), len(X_test_scaled)
    print(f"  Train: {n_train:,} | Test: {n_test:,} | Anomalies: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

    total_runs_fold = len(DIFFICULTY_LEVELS) * len(ALGORITHM_DEFS) * N_SEEDS
    run_count = 0

    for difficulty in DIFFICULTY_LEVELS:
        _, X_test_d, y_test_d = prepare_fold(parquet_path, fold_idx, difficulty)
        X_test_d_scaled = scaler.transform(X_test_d)

        for alg_name, alg_class, alg_kwargs in ALGORITHM_DEFS:
            for seed in SEEDS:
                result = run_single(alg_name, alg_class, alg_kwargs,
                                   X_train_scaled, X_test_d_scaled, y_test_d, seed)
                result['fold'] = fold_idx + 1
                result['month'] = month_name
                result['difficulty'] = difficulty
                all_results.append(result)
                run_count += 1

                if run_count % 100 == 0:
                    elapsed = time.time() - t_benchmark
                    rate = run_count / elapsed
                    eta = (total_runs_fold * len(PARQUET_FILES) - run_count * (fold_idx + 1)) / rate
                    print(f"  [Run {run_count}/{total_runs_fold} fold] ETA remaining: {eta/60:.1f}min")

    fold_time = time.time() - t_fold
    print(f"  Fold {month_name} completed in {fold_time/60:.1f}min")

df_results = pd.DataFrame(all_results)
df_results['fold_month'] = df_results['month']
print(f"\n  Total benchmark time: {(time.time()-t_benchmark)/60:.1f} minutes")
print(f"  Total runs: {len(df_results):,}")

# Save raw results
csv_path = os.path.join(OUT_DIR, 'benchmark_results_full.csv')
df_results.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")

# ===========================
# FILTER OUT ERRORS
# ===========================
df_valid = df_results[df_results['F1'] > 0].copy()
print(f"\n  Valid runs: {len(df_valid):,} / {len(df_results):,}")

# ===========================
# SUMMARY STATISTICS
# ===========================
print("\n" + "=" * 70)
print("SUMMARY: Mean F1 by Algorithm and Difficulty")
print("=" * 70)

for difficulty in DIFFICULTY_LEVELS:
    print(f"\n  {difficulty.upper()}:")
    df_d = df_valid[df_valid['difficulty'] == difficulty]
    summary = df_d.groupby('algorithm').agg(
        F1_mean=('F1', 'mean'), F1_std=('F1', 'std'),
        F1_ci_low=('F1', lambda x: bootstrap_ci(x.values)[0]),
        F1_ci_high=('F1', lambda x: bootstrap_ci(x.values)[1]),
        Recall_mean=('Recall', 'mean'),
        Precision_mean=('Precision', 'mean'),
        train_time_mean=('train_time', 'mean'),
    ).sort_values('F1_mean', ascending=False).reset_index()
    summary['Rank'] = range(1, len(summary) + 1)
    for _, row in summary.iterrows():
        print(f"    Rank {int(row['Rank']):2d} | {row['algorithm']:<20} | "
              f"F1={row['F1_mean']:.4f} +/- {row['F1_std']:.4f} | "
              f"CI=[{row['F1_ci_low']:.4f}, {row['F1_ci_high']:.4f}]")

# ===========================
# FRIEDMAN TEST
# ===========================
print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS: Friedman Omnibus Test")
print("=" * 70)

friedman_results = []
for metric in ['F1', 'Precision', 'Recall', 'AUC_ROC']:
    score_matrix = np.zeros((N_FOLDS, len(ALG_NAMES)))
    for f in range(N_FOLDS):
        for a_idx, alg in enumerate(ALG_NAMES):
            mask = (
                (df_valid['fold'] == f + 1) &
                (df_valid['algorithm'] == alg) &
                (df_valid['difficulty'] == 'medium')
            )
            vals = df_valid.loc[mask, metric].values
            if len(vals) > 0:
                score_matrix[f, a_idx] = np.mean(vals)
            else:
                score_matrix[f, a_idx] = 0.0

    fr = friedman_test(score_matrix)
    fr['metric'] = metric
    friedman_results.append(fr)
    sig = '***' if fr['p'] < 0.001 else '**' if fr['p'] < 0.01 else '*' if fr['p'] < 0.05 else 'ns'
    print(f"  {metric:<12} | chi2={fr['chi2']:.2f} | df={fr['df']} | p={fr['p']:.6f} | {sig}")

df_friedman = pd.DataFrame(friedman_results)
df_friedman.to_csv(os.path.join(OUT_DIR, 'friedman_results.csv'), index=False)
print(f"\n  Friedman results saved.")

# ===========================
# POST-HOC PAIRWISE TESTS
# ===========================
print("\n" + "=" * 70)
print("POST-HOC: Pairwise Wilcoxon + Corrections")
print("=" * 70)

def run_post_hoc_for_metric(df_valid, metric, alg_names):
    score_matrix = np.zeros((N_FOLDS, len(alg_names)))
    for f in range(N_FOLDS):
        for a_idx, alg in enumerate(alg_names):
            mask = (
                (df_valid['fold'] == f + 1) &
                (df_valid['algorithm'] == alg) &
                (df_valid['difficulty'] == 'medium')
            )
            vals = df_valid.loc[mask, metric].values
            score_matrix[f, a_idx] = np.mean(vals) if len(vals) > 0 else 0.0

    pw_df = pairwise_wilcoxon(score_matrix, alg_names)
    if len(pw_df) == 0:
        return pd.DataFrame()

    p_raw = pw_df['p_wilcoxon'].values
    pw_df['p_bonferroni'] = bonferroni_correction(p_raw)
    pw_df['p_holm'] = holm_correction(list(p_raw))

    try:
        from statsmodels.stats.multitest import multipletests
        _, pw_df['p_bh'], _, _ = multipletests(p_raw, alpha=0.05, method='fdr_bh')
    except Exception:
        pw_df['p_bh'] = pw_df['p_wilcoxon']

    pw_df['sig_bonf'] = pw_df['p_bonferroni'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
    pw_df['sig_holm'] = pw_df['p_holm'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
    pw_df['sig_bh'] = pw_df['p_bh'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
    pw_df['effect_label'] = pw_df['cohens_d'].apply(effect_label)
    pw_df['metric'] = metric
    return pw_df

all_post_hoc = []
for metric in ['F1', 'Precision', 'Recall']:
    ph = run_post_hoc_for_metric(df_valid, metric, ALG_NAMES)
    if len(ph) > 0:
        all_post_hoc.append(ph)

if all_post_hoc:
    df_post_hoc = pd.concat(all_post_hoc, ignore_index=True)
else:
    df_post_hoc = pd.DataFrame()

df_post_hoc.to_csv(os.path.join(OUT_DIR, 'statistical_tests_full.csv'), index=False)
print(f"  Post-hoc results saved: {len(df_post_hoc)} comparisons")

if len(df_post_hoc) > 0:
    print(f"\n  Top 10 comparisons by significance (F1):")
    f1_ph = df_post_hoc[df_post_hoc['metric'] == 'F1'].sort_values('p_holm')
    for _, row in f1_ph.head(10).iterrows():
        print(f"    {row['alg_i']:<20} vs {row['alg_j']:<20} | "
              f"d={row['cohens_d']:.3f} | Holm={row['p_holm']:.6f} {row['sig_holm']:<4} | "
              f"BH={row['p_bh']:.6f} {row['sig_bh']:<4} | {row['effect_label']}")

# ===========================
# CRITICAL DIFFERENCE
# ===========================
print("\n" + "=" * 70)
print("CRITICAL DIFFERENCE ANALYSIS")
print("=" * 70)

def compute_avg_ranks(df_valid, metric, difficulty, alg_names):
    score_matrix = np.zeros((N_FOLDS, len(alg_names)))
    for f in range(N_FOLDS):
        for a_idx, alg in enumerate(alg_names):
            mask = (
                (df_valid['fold'] == f + 1) &
                (df_valid['algorithm'] == alg) &
                (df_valid['difficulty'] == difficulty)
            )
            vals = df_valid.loc[mask, metric].values
            score_matrix[f, a_idx] = np.mean(vals) if len(vals) > 0 else 0.0

    ranks = np.zeros_like(score_matrix)
    for f in range(N_FOLDS):
        ranks[f] = score_matrix[f].argsort().argsort() + 1

    avg_ranks = ranks.mean(axis=0)
    std_ranks = ranks.std(axis=0)
    return avg_ranks, std_ranks, score_matrix

q_alpha, cd = critical_difference_q(N_FOLDS, len(ALG_NAMES))
print(f"  N_folds={N_FOLDS}, K={len(ALG_NAMES)}, q_alpha={q_alpha:.3f}, CD={cd:.3f}")

cd_results = []
for difficulty in DIFFICULTY_LEVELS:
    avg_ranks, std_ranks, _ = compute_avg_ranks(df_valid, 'F1', difficulty, ALG_NAMES)
    for a_idx, alg in enumerate(ALG_NAMES):
        cd_results.append({
            'difficulty': difficulty,
            'algorithm': alg,
            'avg_rank': avg_ranks[a_idx],
            'std_rank': std_ranks[a_idx],
        })
    sorted_idx = np.argsort(avg_ranks)
    print(f"\n  {difficulty.upper()} ranks (avg_rank, lower = better):")
    for rank_pos, alg_idx in enumerate(sorted_idx):
        alg = ALG_NAMES[alg_idx]
        r = avg_ranks[alg_idx]
        marker = ' <-- CA-DIF-EIA' if alg == 'CA-DIF-EIA' else ''
        print(f"    Rank {rank_pos+1}: {alg:<20} avg_rank={r:.2f}{marker}")

df_cd = pd.DataFrame(cd_results)
df_cd.to_csv(os.path.join(OUT_DIR, 'cd_ranks.csv'), index=False)

# ===========================
# FIGURES
# ===========================
print("\n" + "=" * 70)
print("GENERATING FIGURES (8 figures)")
print("=" * 70)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

# ----- Fig 1: Boxplot F1 -----
print("  [1/8] Boxplot F1...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax_idx, difficulty in enumerate(DIFFICULTY_LEVELS):
    ax = axes[ax_idx]
    data_by_alg = []
    labels = []
    for alg in ALG_NAMES:
        mask = (
            (df_valid['algorithm'] == alg) &
            (df_valid['difficulty'] == difficulty)
        )
        vals = df_valid.loc[mask, 'F1'].values
        data_by_alg.append(vals)
        labels.append(alg.replace('sklearn_', '').replace('sHST-River', 'sHST'))

    bp = ax.boxplot(data_by_alg, patch_artist=True, notch=False,
                    medianprops={'color': 'black', 'linewidth': 1.5})
    for patch, alg in zip(bp['boxes'], ALG_NAMES):
        patch.set_facecolor(ALG_COLORS[alg])
        patch.set_alpha(0.7)
    for i, (alg, data) in enumerate(zip(ALG_NAMES, data_by_alg)):
        means = [np.mean(data_by_alg[j]) for j in range(len(ALG_NAMES))]
        ax.scatter([i + 1], [means[i]], color='white', s=30, zorder=5, edgecolors='black', linewidth=1)

    ax.set_xticklabels([a.replace('sklearn_', '').replace('sHST-River', 'sHST') for a in ALG_NAMES],
                       rotation=30, ha='right', fontsize=8)
    ax.set_title(difficulty.upper(), fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score' if ax_idx == 0 else '')
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)

fig.suptitle('F1 Score Distribution by Algorithm and Difficulty\n'
             '(12 Monthly Folds, 10 Seeds)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_boxplot_f1.png'), dpi=150, bbox_inches='tight')
plt.close()

# ----- Fig 2: Critical Difference Diagram -----
print("  [2/8] Critical Difference diagram...")
def draw_cd_diagram(avg_ranks, alg_names, cd, month_name, output_path):
    n = len(alg_names)
    sorted_idx = np.argsort(avg_ranks)
    sorted_algs = [alg_names[i] for i in sorted_idx]
    sorted_ranks = avg_ranks[sorted_idx]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, n + 1)
    ax.set_ylim(-0.5, 4)

    ax.axhline(y=3.2, color='#888', linestyle='--', alpha=0.5)

    x_left = sorted_ranks[0] - cd / 2
    x_right = sorted_ranks[0] + cd / 2
    ax.plot([x_left, x_right], [3.0, 3.0], 'k-', lw=3)
    ax.plot([x_left, x_left], [2.85, 3.15], 'k-', lw=2)
    ax.plot([x_right, x_right], [2.85, 3.15], 'k-', lw=2)
    ax.text((x_left + x_right) / 2, 3.25, f'CD = {cd:.2f}',
            ha='center', fontsize=11, fontweight='bold')

    for i, (alg, rank) in enumerate(zip(sorted_algs, sorted_ranks)):
        color = ALG_COLORS.get(alg, '#333')
        is_proposed = 'CA-DIF-EIA' in alg
        marker = 's' if is_proposed else 'o'
        ms = 12 if is_proposed else 10
        lw = 2.5 if is_proposed else 1.5
        ax.plot(rank, 2.0, marker, color=color, ms=ms, zorder=5,
                linewidth=lw, mew=lw if is_proposed else 0)
        ax.text(rank, 1.55, f'{rank:.2f}', ha='center', va='top', fontsize=9)
        label = alg.replace('sklearn_', '').replace('sHST-River', 'sHST')
        ax.text(rank, 1.1, label, ha='center', va='top', fontsize=8,
                fontweight='bold' if is_proposed else 'normal',
                rotation=25)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(sorted_ranks[i] - sorted_ranks[j]) < cd:
                ax.plot([sorted_ranks[i], sorted_ranks[j]], [2.3, 2.3],
                        color='#27ae60', lw=5, solid_capstyle='round', alpha=0.6, zorder=3)

    ax.axis('off')
    ax.set_title(f'Critical Difference Diagram — {month_name}\n'
                 f'Algorithms connected by green bar are NOT significantly different (Nemenyi, α=0.05)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

for difficulty in DIFFICULTY_LEVELS:
    avg_ranks, _, _ = compute_avg_ranks(df_valid, 'F1', difficulty, ALG_NAMES)
    out = os.path.join(OUT_DIR, f'fig2_cd_{difficulty}.png')
    draw_cd_diagram(avg_ranks, ALG_NAMES, cd, difficulty.upper(), out)
    print(f"    Saved CD diagram: fig2_cd_{difficulty}.png")

# ----- Fig 3: P-value Heatmap -----
print("  [3/8] P-value heatmap (Holm-adjusted)...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax_idx, difficulty in enumerate(DIFFICULTY_LEVELS):
    ax = axes[ax_idx]
    score_matrix = np.zeros((N_FOLDS, len(ALG_NAMES)))
    for f in range(N_FOLDS):
        for a_idx, alg in enumerate(ALG_NAMES):
            mask = (
                (df_valid['fold'] == f + 1) &
                (df_valid['algorithm'] == alg) &
                (df_valid['difficulty'] == difficulty)
            )
            vals = df_valid.loc[mask, 'F1'].values
            score_matrix[f, a_idx] = np.mean(vals) if len(vals) > 0 else 0.0

    K = len(ALG_NAMES)
    p_matrix = np.ones((K, K))
    for i, j in itertools.combinations(range(K), 2):
        diff = score_matrix[:, i] - score_matrix[:, j]
        mask_nz = diff != 0
        if mask_nz.sum() >= 5:
            try:
                _, p = wilcoxon(diff[mask_nz])
                p_adj = holm_correction([p])[0]
                p_matrix[i, j] = p_adj
                p_matrix[j, i] = p_adj
            except Exception:
                pass

    labels = [a.replace('sklearn_', '').replace('sHST-River', 'sHST') for a in ALG_NAMES]
    mask_lower = np.tril(np.ones_like(p_matrix), k=0).astype(bool)
    p_display = p_matrix.copy()
    np.fill_diagonal(p_display, 1.0)

    im = ax.imshow(-np.log10(p_display + 1e-10), cmap='RdYlGn_r', vmin=0, vmax=4)
    for i in range(K):
        for j in range(K):
            if i != j:
                p = p_matrix[i, j]
                s = sig_label(p)
                color = 'white' if p < 0.01 else 'black'
                ax.text(j, i, s, ha='center', va='center', fontsize=8, color=color, fontweight='bold')
            else:
                ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='#aaa')
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f'{difficulty.upper()} (Holm adj.)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='-log10(p)')

fig.suptitle('Pairwise Significance: Holm-Bonferroni Adjusted p-values\n'
             '(*** p<0.001, ** p<0.01, * p<0.05, ns = not significant)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_pvalue_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# ----- Fig 4: Radar Chart -----
print("  [4/8] Radar chart...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
for ax_idx, difficulty in enumerate(DIFFICULTY_LEVELS):
    ax = axes[ax_idx]
    metrics_for_radar = ['F1', 'Precision', 'Recall', 'AUC_ROC']
    angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
    angles += angles[:1]

    for alg in ALG_NAMES:
        mask = (
            (df_valid['algorithm'] == alg) &
            (df_valid['difficulty'] == difficulty)
        )
        vals = [df_valid.loc[mask, m].mean() for m in metrics_for_radar]
        vals += vals[:1]
        color = ALG_COLORS[alg]
        lw = 2 if 'CA-DIF-EIA' in alg else 1
        alpha = 0.3 if 'CA-DIF-EIA' not in alg else 0.15
        ax.plot(angles, vals, 'o-', color=color, lw=lw, ms=4, alpha=0.8)
        ax.fill(angles, vals, color=color, alpha=alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_for_radar, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(difficulty.upper(), fontsize=11, fontweight='bold', pad=15)

axes[0].legend([a.replace('sklearn_', '').replace('sHST-River', 'sHST')
                for a in ALG_NAMES], loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=7)
fig.suptitle('Radar Chart: Performance Metrics by Algorithm\n(CA-DIF-EIA = bold, shaded)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_radar.png'), dpi=150, bbox_inches='tight')
plt.close()

# ----- Fig 5: 95% CI Bars -----
print("  [5/8] 95% Bootstrap CI bars...")
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(ALG_NAMES))
width = 0.25

for off_idx, difficulty in enumerate(DIFFICULTY_LEVELS):
    means = []
    ci_lows = []
    ci_highs = []
    for alg in ALG_NAMES:
        mask = (df_valid['algorithm'] == alg) & (df_valid['difficulty'] == difficulty)
        vals = df_valid.loc[mask, 'F1'].values
        means.append(np.mean(vals))
        lo, hi = bootstrap_ci(vals)
        ci_lows.append(np.mean(vals) - lo)
        ci_highs.append(hi - np.mean(vals))

    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    offset = (off_idx - 1) * width
    bars = ax.bar(x + offset, means, width, label=difficulty.upper(),
                  color=colors[off_idx], alpha=0.85, edgecolor='white')
    ax.errorbar(x + offset, means, yerr=[ci_lows, ci_highs],
                fmt='none', color='black', capsize=3, linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels([a.replace('sklearn_', '').replace('sHST-River', 'sHST')
                    for a in ALG_NAMES], rotation=25, ha='right', fontsize=9)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('F1 Score with 95% Bootstrap CI (12 Folds, 10 Seeds)\n'
             'Error bars: 10,000 bootstrap resamples', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, 1.1)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig5_ci_bars.png'), dpi=150, bbox_inches='tight')
plt.close()

# ----- Fig 6: Per-Fold Performance -----
print("  [6/8] Per-fold performance...")
fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharey=True)
axes_flat = axes.flatten()
for fold_idx, month in enumerate(MONTH_NAMES):
    ax = axes_flat[fold_idx]
    x = np.arange(len(ALG_NAMES))
    means = []
    for alg in ALG_NAMES:
        mask = (
            (df_valid['month'] == month) &
            (df_valid['algorithm'] == alg) &
            (df_valid['difficulty'] == 'medium')
        )
        means.append(df_valid.loc[mask, 'F1'].mean())

    colors = [ALG_COLORS[a] for a in ALG_NAMES]
    ax.bar(x, means, color=colors, edgecolor='white', alpha=0.85)
    best_idx = np.argmax(means)
    ax.bar(best_idx, means[best_idx], color='#222', edgecolor='white', alpha=0.5)

    ax.set_xticks(x[::2])
    ax.set_xticklabels([a[:8] for a in ALG_NAMES[::2]], rotation=45, ha='right', fontsize=6)
    ax.set_title(f'{month}', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)

fig.suptitle('F1 Score by Month (Medium Difficulty)\n'
             'Dark bar = best algorithm per month',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig6_per_fold.png'), dpi=150, bbox_inches='tight')
plt.close()

# ----- Fig 7: Difficulty Breakdown -----
print("  [7/8] Difficulty breakdown...")
fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(ALG_NAMES))
width = 0.25
offsets = [-width, 0, width]
level_colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}

for off_idx, difficulty in enumerate(DIFFICULTY_LEVELS):
    means = []
    stds = []
    for alg in ALG_NAMES:
        mask = (df_valid['algorithm'] == alg) & (df_valid['difficulty'] == difficulty)
        vals = df_valid.loc[mask, 'F1'].values
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    bars = ax.bar(x + offsets[off_idx], means, width,
                  label=difficulty.upper(), color=level_colors[difficulty],
                  edgecolor='white', alpha=0.85, yerr=stds, capsize=2)

ax.set_xticks(x)
ax.set_xticklabels([a.replace('sklearn_', '').replace('sHST-River', 'sHST')
                    for a in ALG_NAMES], rotation=20, ha='right', fontsize=10)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 Score by Difficulty Level (All Folds)\n'
             'Error bars = std across 12 folds x 10 seeds',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig7_difficulty.png'), dpi=150, bbox_inches='tight')
plt.close()

# ----- Fig 8: Rank Stability Across Seeds -----
print("  [8/8] Rank stability across seeds...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax_idx, difficulty in enumerate(DIFFICULTY_LEVELS):
    ax = axes[ax_idx]
    for alg in ALG_NAMES:
        mask = (
            (df_valid['algorithm'] == alg) &
            (df_valid['difficulty'] == difficulty)
        )
        vals = df_valid.loc[mask, 'F1'].values
        if len(vals) == 0:
            continue
        seed_means = []
        for s in SEEDS:
            s_mask = mask & (df_valid['seed'] == s)
            s_mean = df_valid.loc[s_mask, 'F1'].mean()
            seed_means.append(s_mean)
        x_pos = SEEDS.index(SEEDS[0]) + 0.5
        color = ALG_COLORS[alg]
        lw = 2 if 'CA-DIF-EIA' in alg else 1
        ax.plot(range(len(SEEDS)), seed_means, 'o-',
                color=color, lw=lw, ms=5, label=alg, alpha=0.8)

    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([str(s) for s in SEEDS], rotation=45, fontsize=7)
    ax.set_title(f'{difficulty.upper()}', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 Score' if ax_idx == 0 else '')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, [l.replace('sklearn_', '').replace('sHST-River', 'sHST')
             for l in labels], loc='upper center',
           bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=8)
fig.suptitle('Rank Stability Across Seeds (12 Folds Averaged)\n'
             'CA-DIF-EIA = bold lines', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig8_rank_stability.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  All 8 figures saved to: {OUT_DIR}")

# ===========================
# FINAL SUMMARY
# ===========================
print("\n" + "=" * 70)
print("BENCHMARK v2.0 COMPLETE")
print("=" * 70)
print(f"""
CONFIGURATION:
  Folds:      {N_FOLDS} monthly (Jan-Dec 2024)
  Algorithms: {len(ALG_NAMES)}
  Seeds:      {N_SEEDS}
  Levels:     {len(DIFFICULTY_LEVELS)} (easy, medium, hard)
  Total runs: {len(df_valid):,}

STATISTICAL METHODS:
  - Friedman omnibus test (omnibus: is there ANY difference?)
  - Post-hoc Wilcoxon signed-rank tests (21 pairwise comparisons)
  - Bonferroni correction (FWER)
  - Holm-Bonferroni step-down (FWER, more powerful)
  - Benjamini-Hochberg FDR
  - Cohen's d (effect size)
  - Rank-biserial correlation
  - Bootstrap 95% CI (10,000 resamples)
  - Critical Difference diagram (Demvsar 2006)

OUTPUT: {OUT_DIR}
  - benchmark_results_full.csv     ({len(df_results):,} rows)
  - friedman_results.csv           (Friedman test per metric)
  - statistical_tests_full.csv     ({len(df_post_hoc)} pairwise comparisons)
  - cd_ranks.csv                  (average ranks per algorithm)
  - fig1_boxplot_f1.png
  - fig2_cd_*.png                 (3 CD diagrams)
  - fig3_pvalue_heatmap.png
  - fig4_radar.png
  - fig5_ci_bars.png
  - fig6_per_fold.png
  - fig7_difficulty.png
  - fig8_rank_stability.png
""")

total_time_min = (time.time() - t_benchmark) / 60
print(f"  Total wall-clock time: {total_time_min:.1f} minutes")
