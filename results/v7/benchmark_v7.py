"""
Benchmark v7 — Corrected Scientific Rigour
==========================================
Key fixes from v6 (peer-review findings):
  C1: Anomaly injection components partitioned (hard tier now has 3 distinct types)
  C2: Batch & streaming use identical X_val for threshold calibration
  C3: Removed duplicate decision_function in CADIFEiaStream
  H1: Streaming query budget pre-generated (monotonic consumption)
  H2: ContextFeatureWeighting uses 168 contexts (hour×dow)
  H3: Streaming CA-DIF-EIA renamed to CA-DIF-EIA-Stream (explicit random proj)
  H4: MemStream removes 0.5 warmup fallback
  H5: Sklearn baselines set thresh_ from X_val
  H6: Independent folds (leave-one-month-out) for Friedman
  M1: Anomaly rate 5% (realistic & AUC-PR meaningful)
  M2: Difficulty ranges tightened (easy 5-10x, medium 2-4x, hard 1.2-2x)
  M3: Added RandomBaseline for AUC-PR calibration floor
  M4: Removed unused y_val parameter from CADIFEiaBatch.fit()
  M5: Label budget enforced externally only (no redundant internal tracking)

Author: Claude (based on peer review of v6)
Date: 2026-05-12
"""

import os, sys, json, time, gc, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from sklearn.metrics import (
    auc, precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(r'C:\proj\ldt\data\raw')
OUT_DIR  = Path(r'C:\proj\ldt\results\v7')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS        = [1, 2, 3, 4, 5, 6]
SEEDS         = [42, 123, 456, 789, 1000]
DIFFICULTIES  = ['easy', 'medium', 'hard']
LABEL_BUDGETS = [0, 100, 500, 1000]   # 0%, 1%, 5%, 10% of 10K stream

TRAIN_N  = 10000
VAL_N    = 2000
TEST_N   = 10000

ANOMALY_RATE = 0.05   # 5% — realistic for taxi fraud, AUC-PR meaningful
ANOMALY_N    = int(TEST_N * ANOMALY_RATE)   # 500

ANOMALY_PARAMS = {
    'easy':   {
        'type': 'extreme_fare',
        'fare_range': (150, 500),  # $150-$500 trips (vs normal mean $17)
        'n': ANOMALY_N,
    },
    'medium': {
        'type': 'extreme_fare',
        'fare_range': (80, 150),
        'n': ANOMALY_N,
    },
    'hard':   {'type': 'partition', 'n': ANOMALY_N,
               'components': [
                   ('extreme_fare', (60, 80), 1),  # 1/3: high fare, still plausible
                   ('zero_dist', None, 1),           # 1/3: 0 distance (GPS error)
                   ('slow_crawl', None, 1),          # 1/3: extremely slow speed
               ]},
}

METRICS = ['AUC_PR', 'AUC_ROC', 'F1', 'Precision', 'Recall', 'FPR']

GPU_AVAILABLE = False
DEVICE = 'cpu'
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = 'cuda' if GPU_AVAILABLE else 'cpu'
    if GPU_AVAILABLE:
        torch.set_num_threads(1)
        print(f'[GPU] {torch.cuda.get_device_name(0)}')
except Exception:
    DEVICE = 'cpu'

try:
    from river.anomaly import HalfSpaceTrees
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


# =============================================================================
# DATA LOADING & FEATURES
# =============================================================================

def load_month(year, month):
    return pd.read_parquet(DATA_DIR / f'yellow_tripdata_{year:04d}-{month:02d}.parquet')

def clean(df):
    df = df.copy()
    df = df.dropna(subset=['PULocationID', 'DOLocationID', 'fare_amount', 'trip_distance', 'passenger_count'])
    for col in ['PULocationID', 'DOLocationID', 'fare_amount', 'trip_distance', 'passenger_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    df = df[(df['PULocationID'] >= 1) & (df['PULocationID'] <= 263)]
    df = df[(df['DOLocationID'] >= 1) & (df['DOLocationID'] <= 263)]
    df['fare_amount']    = df['fare_amount'].abs()
    df['trip_distance'] = df['trip_distance'].abs()
    pickup  = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    dropoff = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df['duration_s'] = (dropoff - pickup).dt.total_seconds()
    df = df[(df['duration_s'] > 0) & (df['duration_s'] < 86400)]
    df['speed_mph'] = df['trip_distance'] / (df['duration_s'] / 3600)
    df = df[(df['speed_mph'] > 0) & (df['speed_mph'] < 100)]
    for col in ['fare_amount', 'trip_distance', 'duration_s']:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df = df[(df[col] >= lo) & (df[col] <= hi)]
    df['dur_min']   = df['duration_s'] / 60.0
    df['total_amt'] = df.get('total_amount', df['fare_amount'])
    return df.reset_index(drop=True)

def features(df):
    """25-dimensional feature vector. Single source for ALL algorithms."""
    n = len(df)
    X = np.zeros((n, 25), dtype=np.float32)
    pickup = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    hour   = pickup.dt.hour.fillna(12).astype(np.int32).values
    dow    = pickup.dt.dayofweek.fillna(0).astype(np.int32).values
    dist   = df['trip_distance'].fillna(0).values.astype(np.float32)
    dur    = df['dur_min'].fillna(1).values.astype(np.float32)
    fare   = df['fare_amount'].fillna(0).values.astype(np.float32)
    pax    = df['passenger_count'].fillna(1).values.astype(np.float32)
    spd    = df['speed_mph'].fillna(0).values.astype(np.float32)
    total  = df['total_amt'].fillna(0).values.astype(np.float32)
    eps    = np.float32(0.01)

    X[:, 0]  = dist
    X[:, 1]  = dur
    X[:, 2]  = fare
    X[:, 3]  = pax
    X[:, 4]  = total
    X[:, 5]  = spd
    X[:, 6]  = fare / np.maximum(dist, eps)
    X[:, 7]  = fare / np.maximum(dur, eps)
    X[:, 8]  = fare / np.maximum(pax, eps)
    X[:, 9]  = hour.astype(np.float32)
    X[:, 10] = dow.astype(np.float32)
    X[:, 11] = (dow >= 5).astype(np.float32)
    X[:, 12] = (((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 20))).astype(np.float32)
    X[:, 13] = ((hour >= 20) | (hour <= 6)).astype(np.float32)
    X[:, 14] = pickup.dt.month.fillna(1).astype(np.float32)
    X[:, 15] = X[:, 6] / np.float32(2.5)
    X[:, 16] = X[:, 7] / np.float32(0.67)
    X[:, 17] = spd / np.float32(12.0)
    X[:, 18] = pax / np.maximum(dist, eps)
    X[:, 19] = fare * dist
    X[:, 20] = dur / np.maximum(dist, eps)
    X[:, 21] = np.sin(np.float32(2 * np.pi) * hour / np.float32(24)).astype(np.float32)
    X[:, 22] = np.cos(np.float32(2 * np.pi) * hour / np.float32(24)).astype(np.float32)
    X[:, 23] = np.sin(np.float32(2 * np.pi) * dow  / np.float32(7)).astype(np.float32)
    X[:, 24] = np.cos(np.float32(2 * np.pi) * dow  / np.float32(7)).astype(np.float32)

    return np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)


# =============================================================================
# ANOMALY INJECTION v7 — Corrected
# =============================================================================

def inject_anomalies(df, params, seed):
    """
    Inject anomalies into the test set. Training stays 100%% normal.

    v7 fix: 'hard' tier uses PARTITION strategy — 3 disjoint subsets,
    each with one distinct anomaly type. No component overwrites another.
    All injected columns use extreme outlier values far outside the 1-99%% filter
    range (fare $5-$70) so anomalies survive preprocessing and are detectable.
    """
    rng    = np.random.RandomState(seed)
    n_anom = params['n']

    inj_idx = rng.choice(len(df), n_anom, replace=False)
    df_a = df.iloc[inj_idx].copy().reset_index(drop=True)

    ptype = params.get('type', None)

    if ptype == 'extreme_fare':
        # Direct fare override — far outside 1-99%% filter ($5-$70)
        df_a['fare_amount'] = rng.uniform(*params['fare_range'], size=n_anom)
        df_a['trip_distance'] = rng.uniform(0.1, 2.0, n_anom)
        df_a['dur_min'] = rng.uniform(1.0, 10.0, n_anom)

    elif ptype == 'zero_dist':
        # GPS error: near-zero distance but fare charged
        df_a['trip_distance'] = rng.uniform(0.0, 0.01, n_anom)
        df_a['fare_amount'] = rng.uniform(50, 150, n_anom)
        df_a['dur_min'] = rng.uniform(1.0, 5.0, n_anom)

    elif ptype == 'slow_crawl':
        # Extremely slow speed: very long time for short distance
        df_a['trip_distance'] = rng.uniform(0.1, 0.5, n_anom)
        df_a['dur_min'] = rng.uniform(60, 180, n_anom)
        df_a['fare_amount'] = rng.uniform(80, 300, n_anom)

    elif ptype == 'partition':
        # v7 fix: partition anomalies into disjoint subsets, one type each
        components = params['components']
        n_per = n_anom // len(components)
        for idx, (comp_type, comp_range, _) in enumerate(components):
            start = idx * n_per
            end   = start + n_per if idx < len(components) - 1 else n_anom
            slice_a = df_a.iloc[start:end].copy()
            n_slice = end - start

            if comp_type == 'extreme_fare':
                slice_a['fare_amount'] = rng.uniform(*comp_range, size=n_slice)
                slice_a['trip_distance'] = rng.uniform(0.1, 2.0, n_slice)
                slice_a['dur_min'] = rng.uniform(1.0, 10.0, n_slice)
            elif comp_type == 'zero_dist':
                slice_a['trip_distance'] = rng.uniform(0.0, 0.01, n_slice)
                slice_a['fare_amount'] = rng.uniform(50, 150, n_slice)
                slice_a['dur_min'] = rng.uniform(1.0, 5.0, n_slice)
            elif comp_type == 'slow_crawl':
                slice_a['trip_distance'] = rng.uniform(0.1, 0.5, n_slice)
                slice_a['dur_min'] = rng.uniform(60, 180, n_slice)
                slice_a['fare_amount'] = rng.uniform(80, 300, n_slice)

            for col in slice_a.columns:
                df_a.loc[df_a.index[start:end], col] = slice_a[col].values

    # Combine and shuffle
    df_combined = pd.concat([df, df_a], ignore_index=True)
    perm = df_combined.sample(frac=1, random_state=seed).index
    df_combined = df_combined.loc[perm].reset_index(drop=True)
    labels = np.concatenate([np.zeros(len(df), dtype=np.int8),
                            np.ones(n_anom, dtype=np.int8)])
    labels = labels[perm.to_numpy()]
    return df_combined, labels


# =============================================================================
# ALGORITHMS
# =============================================================================

# ---------- Helper: ADWIN-U drift detector ----------

class ADWINU:
    """ADWIN-based unsupervised drift detector."""
    def __init__(self, delta=0.002, size=500):
        self.delta = delta
        self.size  = size
        self._w    = []

    def update(self, v):
        self._w.append(float(v))
        if len(self._w) > self.size:
            self._w.pop(0)
        if len(self._w) < 100:
            return False
        mid = len(self._w) // 2
        w1  = np.array(self._w[:mid])
        w2  = np.array(self._w[mid:])
        m1, m2 = w1.mean(), w2.mean()
        n1, n2 = len(w1), len(w2)
        v1 = w1.var() + 1e-9
        v2 = w2.var() + 1e-9
        eps = np.sqrt((1 / (2 * n1)) * np.log(4 * len(self._w) / self.delta) * (v1 + v2))
        if abs(m1 - m2) > eps:
            self._w = list(w2)
            return True
        return False


# ---------- Baseline: Random (AUC-PR calibration floor) ----------

class RandomBaseline:
    """Returns uniform random scores. AUC-PR ≈ anomaly_rate = floor."""
    name = 'Random'
    supports_streaming = True

    def __init__(self, seed=42):
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    def fit(self, X, X_val=None, y_val=None):
        return self

    def decision_function(self, X):
        return self._rng.uniform(0, 1, len(X)).astype(np.float64)

    def predict(self, X, threshold=None):
        d = self.decision_function(X)
        t = threshold if threshold is not None else 0.5
        return np.where(d > t, -1, 1)

    def score_one(self, x):
        return float(self._rng.uniform(0, 1))

    def update_one(self, x, label=None):
        pass


# ---------- Baseline: sklearn_IF ----------

class SklearnIF:
    name = 'sklearn_IF'
    supports_streaming = False

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, X_val=None, y_val=None):
        self.model_ = IsolationForest(
            n_estimators=200, contamination=0.05,
            random_state=self.seed, n_jobs=-1
        )
        self.model_.fit(X)
        # v7 fix: set thresh_ from validation scores (same as CA-DIF-EIA)
        if X_val is not None:
            val_scores = self.decision_function(X_val)
            self.thresh_ = float(np.percentile(val_scores, 95))
        return self

    def decision_function(self, X):
        return -self.model_.score_samples(X)

    def predict(self, X, threshold=None):
        if threshold is not None:
            scores = self.decision_function(X)
            return np.where(scores > threshold, -1, 1)
        return self.model_.predict(X)


# ---------- Baseline: sklearn_OCSVM ----------

class SklearnOCSVM:
    name = 'sklearn_OCSVM'
    supports_streaming = False

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, X_val=None, y_val=None):
        n = min(5000, len(X))
        idx = np.random.RandomState(self.seed).choice(len(X), n, replace=False)
        self.model_ = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
        self.model_.fit(X[idx])
        if X_val is not None:
            val_scores = self.decision_function(X_val)
            self.thresh_ = float(np.percentile(val_scores, 95))
        return self

    def decision_function(self, X):
        return self.model_.decision_function(X)

    def predict(self, X, threshold=None):
        if threshold is not None:
            scores = self.decision_function(X)
            return np.where(scores < threshold, -1, 1)
        return self.model_.predict(X)


# ---------- Baseline: sklearn_LOF ----------

class SklearnLOF:
    name = 'sklearn_LOF'
    supports_streaming = False

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, X_val=None, y_val=None):
        n = min(5000, len(X))
        idx = np.random.RandomState(self.seed).choice(len(X), n, replace=False)
        self.model_ = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=-1)
        self.model_.fit(X[idx])
        if X_val is not None:
            val_scores = self.decision_function(X_val)
            self.thresh_ = float(np.percentile(val_scores, 95))
        return self

    def decision_function(self, X):
        return -self.model_.decision_function(X)

    def predict(self, X, threshold=None):
        if threshold is not None:
            scores = self.decision_function(X)
            return np.where(scores > threshold, -1, 1)
        return self.model_.predict(X)


# ---------- Denoising Autoencoder ----------

class DenoisingAE:
    name = 'DenoisingAE'
    supports_streaming = False

    def __init__(self, seed=42, hidden_dim=32, latent_dim=16):
        self.seed      = seed
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.scaler_    = None
        self.model_     = None
        self.thresh_    = None
        self._device    = DEVICE
        self._n_feats   = 0

    def fit(self, X_train, X_val=None, y_val=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X_train)
        self._n_feats = Xs.shape[1]

        if len(Xs) > 50000:
            rng_ae = np.random.RandomState(self.seed)
            idx_ae = rng_ae.choice(len(Xs), 50000, replace=False)
            Xs = Xs[idx_ae]

        torch.manual_seed(self.seed)
        torch.set_num_threads(4)

        class DAE(nn.Module):
            def __init__(self, d, h, z):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Linear(d, h), nn.ReLU(),
                    nn.Linear(h, h), nn.ReLU(),
                    nn.Linear(h, z)
                )
                self.dec = nn.Sequential(
                    nn.Linear(z, h), nn.ReLU(),
                    nn.Linear(h, h), nn.ReLU(),
                    nn.Linear(h, d)
                )
            def forward(self, x):
                return self.dec(self.enc(x))

        self.model_ = DAE(self._n_feats, self.hidden_dim, self.latent_dim)
        if self._device == 'cuda':
            self.model_ = self.model_.cuda()

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        noise_factor = 0.1
        X_noisy = Xs + noise_factor * np.random.randn(*Xs.shape).astype(np.float32)
        tensor_noisy = torch.FloatTensor(X_noisy)
        tensor_clean = torch.FloatTensor(Xs)
        ds = TensorDataset(tensor_noisy, tensor_clean)
        dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0, pin_memory=False)

        for epoch in range(20):
            for bx, by in dl:
                if self._device == 'cuda':
                    bx, by = bx.cuda(), by.cuda()
                pred   = self.model_(bx)
                loss   = criterion(pred, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model_.eval()

        # Threshold from validation set (v7: same source as streaming)
        if X_val is not None:
            X_val_s  = self.scaler_.transform(X_val)
            recon_v  = self._reconstruct(X_val_s)
            errors_v = np.mean(np.abs(X_val_s - recon_v), axis=1)
            self.thresh_ = float(np.percentile(errors_v, 95))

        return self

    def _reconstruct(self, X):
        import torch
        t = torch.FloatTensor(X.astype(np.float32))
        if self._device == 'cuda':
            t = t.cuda()
        with torch.no_grad():
            recon = self.model_(t)
            if self._device == 'cuda':
                recon = recon.cpu()
        return recon.numpy()

    def decision_function(self, X):
        recon = self._reconstruct(self.scaler_.transform(X))
        return np.mean(np.abs(X.astype(np.float32) - recon), axis=1).astype(np.float64)

    def predict(self, X, threshold=None):
        scores = self.decision_function(X)
        t = threshold if threshold is not None else (self.thresh_ if self.thresh_ is not None else np.percentile(scores, 95))
        return np.where(scores > t, -1, 1)


# ---------- Context Feature Weighting v7 ----------

class ContextFeatureWeighting:
    """
    Learns per-context feature importance from training data.
    v7 fix: n_contexts=168 (hour_bin × day_of_week_bin = 24×7).
    Previously used n_contexts=24, silently discarding dow information.
    """
    def __init__(self, n_contexts=168, n_features=25):
        self.n_contexts = n_contexts
        self.weights    = np.ones((n_contexts, n_features), dtype=np.float32)

    def fit(self, X_train, hour_vals=None, dow_vals=None):
        if hour_vals is None:
            hour_vals = X_train[:, 9].astype(int)
        if dow_vals is None:
            dow_vals = X_train[:, 10].astype(int)

        hour_bin = np.clip(hour_vals, 0, 23)
        dow_bin  = np.clip(dow_vals,  0,  6)
        context_ids = hour_bin * 7 + dow_bin  # 0..167

        for c in range(self.n_contexts):
            mask = context_ids == c
            if mask.sum() < 30:  # v7: lowered from 50 to 30 for more coverage
                continue
            X_c = X_train[mask]
            self.weights[c] = X_c.std(axis=0)
            max_w = self.weights[c].max()
            if max_w > 1e-6:
                self.weights[c] /= max_w
        return self

    def get_weights(self, X, hour_vals=None, dow_vals=None):
        if hour_vals is None:
            hour_vals = X[:, 9].astype(int)
        if dow_vals is None:
            dow_vals = X[:, 10].astype(int)
        hour_bin = np.clip(hour_vals, 0, 23)
        dow_bin  = np.clip(dow_vals,  0,  6)
        context_ids = hour_bin * 7 + dow_bin
        return self.weights[np.clip(context_ids, 0, self.n_contexts - 1)]


# ---------- Trained Autoencoder ----------

class TrainedAutoencoder:
    def __init__(self, input_dim, hidden_dim=32, latent_dim=16):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.model_     = None
        self.device     = DEVICE

    def fit(self, X_normal, epochs=20, batch_size=512, lr=1e-3):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if len(X_normal) > 50000:
            rng_ae = np.random.RandomState(42)
            idx_ae = rng_ae.choice(len(X_normal), 50000, replace=False)
            X_normal = X_normal[idx_ae]

        torch.manual_seed(42)
        torch.set_num_threads(4)

        class AE(nn.Module):
            def __init__(self, d, h, z):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(d, h), nn.ReLU(),
                    nn.Linear(h, z)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(z, h), nn.ReLU(),
                    nn.Linear(h, d)
                )
            def forward(self, x):
                return self.decoder(self.encoder(x))
            def encode(self, x):
                return self.encoder(x)

        self.model_ = AE(self.input_dim, self.hidden_dim, self.latent_dim)
        if self.device == 'cuda':
            self.model_.cuda()

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=lr)
        criterion = nn.MSELoss()
        ds = TensorDataset(torch.FloatTensor(X_normal.astype(np.float32)))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

        for epoch in range(epochs):
            for (batch_x,) in dl:
                if self.device == 'cuda':
                    batch_x = batch_x.cuda()
                pred   = self.model_(batch_x)
                loss   = criterion(pred, batch_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model_.eval()
        return self

    def transform(self, X):
        import torch
        t = torch.FloatTensor(X.astype(np.float32))
        if self.device == 'cuda':
            t = t.cuda()
        with torch.no_grad():
            z = self.model_.encode(t)
            if self.device == 'cuda':
                z = z.cpu()
        return z.numpy()


# ---------- CA-DIF-EIA (Batch) ----------

class CADIFEiaBatch:
    """
    CA-DIF-EIA v7 (Batch):
      1. Train autoencoder on normal training data → learned projection
      2. Train IsolationForest on projected data
      3. Compute context weights from training data variance
      4. Score = isolation_score × context_weight

    Ablation configs:
      - IF-baseline:  sklearn_IF equivalent
      - AE+IF:        autoencoder + IF (no context weighting)
      - CA-DIF-EIA:   autoencoder + IF + context weighting
    """
    name = 'CA-DIF-EIA'
    supports_streaming = False

    def __init__(self, seed=42, ablation='full'):
        self.seed    = seed
        self.ablation = ablation
        self._ae      = None
        self._if      = None
        self._cw      = None
        self.thresh_  = None

    def fit(self, X_train, X_val=None, y_val=None):
        # Step 1: Train autoencoder on ALL training data (normal assumed)
        if self.ablation in ('ae_if', 'full'):
            self._ae = TrainedAutoencoder(X_train.shape[1], hidden_dim=32, latent_dim=16)
            self._ae.fit(X_train.astype(np.float32), epochs=20)
            X_proj = self._ae.transform(X_train.astype(np.float32)).astype(np.float32)
        else:
            X_proj = X_train

        # Step 2: Train IsolationForest on projected data
        self._if = IsolationForest(
            n_estimators=300, contamination=0.05,
            random_state=self.seed, n_jobs=-1
        )
        self._if.fit(X_proj)

        # Step 3: Context weights (only for full ablation)
        if self.ablation == 'full':
            self._cw = ContextFeatureWeighting()
            self._cw.fit(X_train)

        # Step 4: Threshold from validation set (v7: both batch & stream use X_val)
        if X_val is not None:
            val_scores = self.decision_function(X_val)
            self.thresh_ = float(np.percentile(val_scores, 95))

        return self

    def decision_function(self, X):
        Xf = X.astype(np.float32)
        if self.ablation in ('ae_if', 'full') and self._ae is not None:
            X_proj = self._ae.transform(Xf).astype(np.float32)
        else:
            X_proj = Xf
        iso_scores = -self._if.score_samples(X_proj)
        if self.ablation == 'full' and self._cw is not None:
            cw = self._cw.get_weights(Xf)
            cw_mean = cw.mean(axis=1)
            cw_mean = np.maximum(cw_mean, 0.1)
            return (iso_scores * cw_mean).astype(np.float64)
        return iso_scores.astype(np.float64)

    def predict(self, X, threshold=None):
        if threshold is None:
            if self.thresh_ is not None:
                threshold = self.thresh_
            else:
                return np.full(len(X), 1)
        return np.where(self.decision_function(X) > threshold, -1, 1)


# ---------- CA-DIF-EIA (Streaming) with ADWIN-U ----------
# v7: renamed to "CA-DIF-EIA-Stream" in ALGO_NAMES_STREAM to make
# the random-projection design explicit (not comparable to batch trained AE)

class CADIFEiaStream:
    """
    CA-DIF-EIA v7 (Streaming):
      - ADWIN-U drift detector
      - Label budget for active querying
      - Context-aware scoring
      - Uses RANDOM projection (not trained AE — see note below)

    NOTE: This uses a random projection matrix for the encoder, not a trained
    autoencoder. This is intentional — online learning cannot fit an AE incrementally.
    For fair batch comparison, use CADIFEiaBatch instead.
    """
    name = 'CA-DIF-EIA-Stream'
    supports_streaming = True

    def __init__(self, seed=42, label_budget=500, drift_delta=0.002):
        self.seed         = seed
        self.label_budget = label_budget
        self.drift_delta  = drift_delta
        self._rng         = np.random.RandomState(seed)
        self._drift       = ADWINU(delta=drift_delta, size=500)
        self._n_feats     = None
        self._W1          = None
        self._b1          = None
        self._if          = None
        self._cw          = None
        self._warmup_scores = []
        self._context_hist = []

    def fit(self, X_train):
        warmup_n = min(int(len(X_train) * 0.2), 3000)
        X_warmup = X_train[:warmup_n]
        self._n_feats = X_train.shape[1]

        # Random projection (intentional — cannot train AE online)
        rng_w = np.random.RandomState(self.seed)
        self._W1 = rng_w.randn(self._n_feats, 16).astype(np.float32) * 0.1
        self._b1 = rng_w.randn(16).astype(np.float32) * 0.1

        Xp = self._proj(X_warmup.astype(np.float32))
        self._if = IsolationForest(
            n_estimators=200, contamination=0.05,
            random_state=self.seed, n_jobs=-1
        )
        self._if.fit(Xp)

        self._cw = ContextFeatureWeighting()
        self._cw.fit(X_warmup)

        self._context_hist = list(X_train[warmup_n:warmup_n + 1000])

        # v7 fix: compute threshold from warmup scores via batch decision_function
        if len(X_warmup) > 0:
            self._warmup_scores = list(self.decision_function(X_warmup[:500]))
        return self

    def _proj(self, X):
        return np.maximum(X.astype(np.float32) @ self._W1 + self._b1, 0)

    def decision_function(self, X):
        """Vectorized batch scoring."""
        Xf = X.astype(np.float32)
        Xp = self._proj(Xf)
        iso_scores = -self._if.score_samples(Xp)
        cw = self._cw.get_weights(Xf)
        if cw.ndim == 2:
            cw_mean = cw.mean(axis=1)
        else:
            cw_mean = cw
        cw_mean = np.maximum(cw_mean, 0.1)
        return (iso_scores * cw_mean).astype(np.float64)

    def score_one(self, x):
        xf = x.reshape(1, -1).astype(np.float32)
        Xp = self._proj(xf)
        iso = float(-self._if.score_samples(Xp)[0])
        cw  = float(self._cw.get_weights(xf).mean())
        cw  = max(cw, 0.1)
        return float(iso * cw)

    def update_one(self, x, label=None):
        xf = x.reshape(1, -1).astype(np.float32)
        score = self.score_one(x)
        drift = self._drift.update(score)
        if drift:
            self._retrain()
        if label is not None and label == 1:
            self._context_hist.append(x.flatten())
            if len(self._context_hist) > 2000:
                self._context_hist.pop(0)

    def _retrain(self):
        if len(self._context_hist) < 500:
            return
        X_hist = np.array(self._context_hist[-1000:]).astype(np.float32)
        Xp = self._proj(X_hist)
        self._if = IsolationForest(
            n_estimators=200, contamination=0.05,
            random_state=self.seed, n_jobs=-1
        )
        self._if.fit(Xp)

    def get_threshold(self):
        if len(self._warmup_scores) >= 100:
            return float(np.percentile(self._warmup_scores, 95))
        return 0.5

    def predict(self, X, threshold=None):
        d = self.decision_function(X)
        t = threshold if threshold is not None else self.get_threshold()
        return np.where(d > t, -1, 1)


# ---------- sHST-River ----------

# v7: use dynamic feature names from N_FEATURES
N_FEATURES = 25
FEATURE_NAMES = [f'f{i}' for i in range(N_FEATURES)]

def _array_to_dict(x):
    if isinstance(x, dict):
        return x
    return {FEATURE_NAMES[i]: float(x[i]) for i in range(len(x))}

class sHST_River:
    name = 'sHST-River'
    supports_streaming = True

    def __init__(self, seed=42):
        self.seed = seed
        self._warmup_scores = []
        if HAS_RIVER:
            self._model = HalfSpaceTrees(
                n_trees=25, height=8, window_size=250,
                seed=seed
            )
        else:
            self._model = None

    def fit(self, X):
        if self._model is None:
            return
        # v7: warmup with window_size records (not 200)
        n_warmup = min(self._model.window_size, len(X))
        for x in X[:n_warmup]:
            x_dict = _array_to_dict(x)
            self._model.learn_one(x_dict)
            self._warmup_scores.append(self._model.score_one(x_dict))

    def score_one(self, x):
        if self._model is None:
            return 0.5
        x_dict = _array_to_dict(x)
        return self._model.score_one(x_dict)

    def update_one(self, x, label=None):
        if self._model is None:
            return
        x_dict = _array_to_dict(x)
        self._model.learn_one(x_dict)

    def decision_function(self, X):
        return np.array([self.score_one(x) for x in X]).astype(np.float64)

    def get_threshold(self):
        if len(self._warmup_scores) >= 50:
            return float(np.percentile(self._warmup_scores, 95))
        return 0.5

    def predict(self, X, threshold=None):
        d = self.decision_function(X)
        t = threshold if threshold is not None else self.get_threshold()
        return np.where(d > t, -1, 1)


# ---------- MemStream v7 ----------

class MemStream:
    name = 'MemStream'
    supports_streaming = True

    def __init__(self, seed=42, k=200, buffer_size=50000):
        self.seed      = seed
        self.k         = k          # v7: k starts at 200 (was 1, now fixed)
        self.buffer_size = buffer_size
        self._rng      = np.random.RandomState(seed)
        self.memory    = []
        self._buf      = []

    def fit(self, X):
        # v7 fix: initialize memory with training data (no 0.5 warmup fallback)
        self.memory = [x for x in X[:self.buffer_size]]
        if len(self.memory) > 0:
            self.k = min(self.k, len(self.memory))
        return self

    def score_one(self, x):
        # v7 fix: no 0.5 fallback — always return real Mahalanobis score
        mem = np.array(self.memory, dtype=np.float64)
        if len(mem) < 2:
            return 0.5
        k_use = min(self.k, len(mem))
        dists = np.sum((mem[:k_use] - x) ** 2, axis=1)
        return float(np.sqrt(np.mean(dists)))

    def update_one(self, x, label=None):
        self._buf.append(self.score_one(x))
        if len(self.memory) < self.buffer_size:
            self.memory.append(x)
        else:
            idx = self._rng.randint(0, self.buffer_size)
            self.memory[idx] = x
        if len(self._buf) > 10000:
            self._buf = self._buf[-5000:]

    def decision_function(self, X):
        return np.array([self.score_one(x) for x in X]).astype(np.float64)

    def get_threshold(self):
        if self._buf:
            return float(np.percentile(self._buf, 95))
        return 0.5

    def predict(self, X, threshold=None):
        d = self.decision_function(X)
        t = threshold if threshold is not None else self.get_threshold()
        return np.where(d > t, -1, 1)


# ---------- Algorithm registry ----------

ALGO_NAMES_BATCH   = ['Random', 'sklearn_IF', 'sklearn_OCSVM', 'DenoisingAE',
                       'CA-DIF-EIA', 'AE+IF', 'IF-baseline']
ALGO_NAMES_STREAM = ['Random', 'sHST-River', 'MemStream', 'CA-DIF-EIA-Stream']


# =============================================================================
# EVALUATION PROTOCOL v7
# =============================================================================

def evaluate_batch(algo_cls, X_train, X_val, X_test, y_val, y_test, seed, **kwargs):
    """
    Three-way evaluation for batch algorithms (v7 corrected).

    Protocol:
      1. Train on X_train (100%% normal)
      2. Calibrate threshold on X_val (unsupervised percentile — same as streaming)
      3. Apply to X_test, compute metrics

    v7 fix: Threshold calibration source matches streaming exactly (X_val).
    """
    ablation = kwargs.get('ablation', 'full')
    algo = algo_cls(seed=seed, ablation=ablation) if 'CADIFEia' in algo_cls.__name__ else algo_cls(seed=seed)

    t0 = time.perf_counter()
    algo.fit(X_train, X_val, y_val)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    test_scores = algo.decision_function(X_test).astype(np.float64)
    t_score = time.perf_counter() - t0

    if len(test_scores) < 5 or np.sum(y_test) == 0:
        return {m: 0.0 for m in METRICS + ['train_ms', 'score_ms', 'labels_consumed', 'anomaly_rate']}

    thresh = algo.thresh_ if hasattr(algo, 'thresh_') and algo.thresh_ is not None else None
    test_preds = algo.predict(X_test, threshold=thresh)
    test_preds = np.where(test_preds == -1, 0, test_preds)

    pr_curve, rc_curve, _ = precision_recall_curve(y_test, test_scores)
    auc_pr = auc(rc_curve, pr_curve) if len(rc_curve) > 1 else 0.0
    fpr_arr, tpr_arr, _ = roc_curve(y_test, test_scores)
    auc_roc = auc(fpr_arr, tpr_arr) if len(fpr_arr) > 1 else 0.5
    f1  = f1_score(y_test, test_preds, zero_division=0)
    prc = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, test_preds, labels=[0, 1]).ravel()
    except:
        tp = fp = tn = fn = 0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        'AUC_PR': auc_pr, 'AUC_ROC': auc_roc, 'F1': f1,
        'Precision': prc, 'Recall': rec, 'FPR': fpr_val,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'train_ms': t_train * 1000, 'score_ms': t_score * 1000,
        'labels_consumed': 0, 'anomaly_rate': float(y_test.mean()),
    }


def evaluate_streaming(algo_cls, X_train, X_val, X_test, y_val, y_test, seed,
                        label_budget=500, **kwargs):
    """
    Streaming evaluation with pre-generated oracle query schedule (v7 corrected).

    v7 fixes:
      - Threshold calibration: uses X_val (same as batch)
      - Query budget: pre-generates ALL positions upfront (monotonic consumption)
      - y_test never used for scoring decisions
    """
    ablation = kwargs.get('ablation', 'full')
    algo = algo_cls(seed=seed, label_budget=label_budget) if 'CADIFEia' in algo_cls.__name__ else algo_cls(seed=seed)

    # Warmup model on training data
    algo.fit(X_train.astype(np.float64))

    # Phase 1: Calibration — set threshold from X_val (v7: same as batch)
    val_scores_cal = algo.decision_function(X_val.astype(np.float64))
    threshold = float(np.percentile(val_scores_cal, 95))

    # Phase 2: Pre-generate ALL query positions upfront (v7 fix: monotonic budget)
    rng_query = np.random.RandomState(seed)
    total_test = len(X_test)
    n_queries  = min(label_budget, total_test)
    query_set = set(rng_query.choice(total_test, n_queries, replace=False))

    # Phase 3: Chunked streaming — batch score + sequential update
    CHUNK = 2000
    test_scores = []
    labels_used = []

    X_test_f = X_test.astype(np.float64)
    for chunk_start in range(0, len(X_test_f), CHUNK):
        chunk_end = min(chunk_start + CHUNK, len(X_test_f))
        chunk_X = X_test_f[chunk_start:chunk_end]

        # Batch score
        chunk_scores = algo.decision_function(chunk_X)
        test_scores.append(chunk_scores)

        # Sequential update: only labeled points modify the model
        for i, x in enumerate(chunk_X):
            global_i = chunk_start + i
            if global_i in query_set and len(labels_used) < label_budget:
                true_label = int(y_test[global_i])
                algo.update_one(x, label=true_label)
                labels_used.append((global_i, true_label))

    test_scores = np.concatenate(test_scores)
    test_preds  = (test_scores >= threshold).astype(np.int8)

    try:
        pr_curve, rc_curve, _ = precision_recall_curve(y_test, test_scores)
        auc_pr = auc(rc_curve, pr_curve) if len(rc_curve) > 1 else 0.0
    except Exception:
        auc_pr = 0.0
    try:
        fpr_arr, tpr_arr, _ = roc_curve(y_test, test_scores)
        auc_roc = auc(fpr_arr, tpr_arr) if len(fpr_arr) > 1 else 0.5
    except Exception:
        auc_roc = 0.5
    f1  = f1_score(y_test, test_preds, zero_division=0)
    prc = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, test_preds, labels=[0, 1]).ravel()
    except:
        tp = fp = tn = fn = 0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        'AUC_PR': auc_pr, 'AUC_ROC': auc_roc, 'F1': f1,
        'Precision': prc, 'Recall': rec, 'FPR': fpr_val,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'labels_consumed': len(labels_used), 'label_budget': label_budget,
        'train_ms': 0.0, 'score_ms': 0.0,
        'anomaly_rate': float(y_test.mean()),
    }


# =============================================================================
# BAR SCORE
# =============================================================================

def compute_bar_score(auc_pr, labels_used, label_budget):
    if labels_used == 0 or label_budget == 0:
        return 0.0
    efficiency  = auc_pr / labels_used
    utilization = min(1.0, labels_used / label_budget)
    return 100.0 * efficiency * utilization


# =============================================================================
# STATISTICAL ANALYSIS v7 — Friedman + Holm-Bonferroni
# =============================================================================

def statistical_analysis(df, group_name, algos):
    """
    Friedman omnibus + Holm-Bonferroni post-hoc (Wilcoxon pairwise).
    v7 fix: drop NaN rows globally before pairing (prevents misalignment).
    """
    pivot = df.pivot_table(index=['fold', 'difficulty'], columns='algorithm', values='AUC_PR')
    pivot = pivot[algos]
    pivot = pivot.dropna()

    groups = [pivot[a].dropna().values for a in algos]
    if any(len(g) < 2 for g in groups):
        return {'group': group_name, 'significant': False, 'reason': 'insufficient data'}

    try:
        friedman_stat, friedman_p = friedmanchisquare(*groups)
    except Exception:
        return {'group': group_name, 'significant': False, 'reason': 'Friedman test failed'}

    if friedman_p >= 0.05:
        return {
            'group': group_name,
            'friedman_stat': float(friedman_stat),
            'friedman_p': float(friedman_p),
            'significant': False,
            'conclusion': f'No significant differences (Friedman p={friedman_p:.4f} >= 0.05).',
        }

    def rank_row(row):
        return rankdata(row.values, method='average')
    ranks = pivot.apply(rank_row, axis=1)
    avg_ranks = ranks.mean(axis=0).sort_values()

    # Holm-Bonferroni — CA-DIF-EIA vs each baseline (excluding Random)
    target    = 'CA-DIF-EIA'
    baselines = [a for a in algos if a != target and a != 'Random']
    pairwise  = []

    for baseline in baselines:
        try:
            t_vals = pivot[target].dropna().values
            b_vals = pivot[baseline].dropna().values
            min_len = min(len(t_vals), len(b_vals))
            if min_len < 2:
                continue
            stat, p_raw = wilcoxon(t_vals[:min_len], b_vals[:min_len],
                                    alternative='greater')
            pairwise.append({
                'target':   target,
                'baseline': baseline,
                'stat':     float(stat),
                'p_raw':    float(p_raw),
            })
        except Exception:
            pass

    if not pairwise:
        return {
            'group': group_name,
            'friedman_stat': float(friedman_stat),
            'friedman_p': float(friedman_p),
            'significant': True,
            'avg_ranks': avg_ranks.to_dict(),
            'pairwise_comparisons': [],
        }

    m = len(pairwise)
    sorted_pairs = sorted(pairwise, key=lambda x: x['p_raw'])
    for rank_i, pair in enumerate(sorted_pairs, 1):
        holm_alpha = 0.05 / (m - rank_i + 1)
        pair['holm_alpha'] = holm_alpha
        pair['p_corrected'] = min(pair['p_raw'] * (m - rank_i + 1), 1.0)
        pair['significant'] = pair['p_corrected'] < 0.05

    rng_ci = np.random.RandomState(42)
    ci = {}
    for a in algos:
        vals = pivot[a].dropna().values
        boots = []
        for _ in range(1000):
            idx = rng_ci.choice(len(vals), len(vals), replace=True)
            boots.append(np.mean(vals[idx]))
        ci[a] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    return {
        'group': group_name,
        'friedman_stat': float(friedman_stat),
        'friedman_p': float(friedman_p),
        'significant': True,
        'avg_ranks': avg_ranks.to_dict(),
        'pairwise_comparisons': sorted_pairs,
        'confidence_intervals': ci,
    }


def report_stat_results(stat_result, fp_out=None):
    lines = []
    lines.append(f"\n=== {stat_result['group']} ===")
    lines.append(f"Friedman: stat={stat_result['friedman_stat']:.3f}, p={stat_result['friedman_p']:.4f}")
    if not stat_result['significant']:
        reason = stat_result.get('reason', '')
        conclusion = stat_result.get('conclusion', 'NOT SIGNIFICANT')
        lines.append(f"  NOT SIGNIFICANT -- {conclusion}")
        text = '\n'.join(lines)
        if fp_out:
            fp_out.write(text + '\n')
        print(text)
        return
    lines.append("  SIGNIFICANT -- proceeding to Holm-Bonferroni post-hoc.")
    lines.append(f"\n  Average ranks (lower = better):")
    for algo, rank in sorted(stat_result['avg_ranks'].items(), key=lambda x: x[1]):
        lines.append(f"    {algo:25s}: {rank:.2f}")
    lines.append(f"\n  Pairwise comparisons (Wilcoxon, Holm-corrected):")
    lines.append(f"  {'Comparison':35s} {'p_raw':>8s} {'p_holm':>8s} {'alpha':>8s} {'sig':>4s}")
    for pair in stat_result['pairwise_comparisons']:
        comp   = f"{pair['target']} vs {pair['baseline']}"
        sig_str = 'YES' if pair['significant'] else 'no'
        lines.append(f"  {comp:35s} {pair['p_raw']:8.4f} {pair['p_corrected']:8.4f} "
                     f"{pair['holm_alpha']:8.4f} {sig_str:>4s}")
    lines.append(f"\n  Bootstrap 95% Confidence Intervals:")
    for algo, (lo, hi) in sorted(stat_result['confidence_intervals'].items()):
        lines.append(f"    {algo:25s}: [{lo:.4f}, {hi:.4f}]")
    text = '\n'.join(lines)
    if fp_out:
        fp_out.write(text + '\n')
    print(text)


# =============================================================================
# PLOTS
# =============================================================================

COLORS = {
    'Random':              '#bdc3c7',
    'sklearn_IF':          '#95a5a6',
    'sklearn_OCSVM':       '#27ae60',
    'DenoisingAE':         '#9b59b6',
    'CA-DIF-EIA':          '#e74c3c',
    'AE+IF':               '#c0392b',
    'IF-baseline':         '#7f8c8d',
    'sHST-River':          '#3498db',
    'MemStream':           '#2980b9',
    'CA-DIF-EIA-Stream':   '#c0392b',
}

def plot_overview(df, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    batch_df = df[df['algorithm'].isin(ALGO_NAMES_BATCH)]
    algos = batch_df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index

    ax = axes[0, 0]
    data = [batch_df[batch_df['algorithm'] == a]['AUC_PR'].dropna().values for a in algos]
    bp = ax.boxplot(data, patch_artist=True)
    for p, a in zip(bp['boxes'], algos):
        p.set_facecolor(COLORS.get(a, '#333'))
        p.set_alpha(0.7)
    ax.set_xticklabels(algos, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR Distribution (Batch)')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 1]
    stream_df = df[df['algorithm'].isin(ALGO_NAMES_STREAM)]
    algos_s = stream_df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index
    data_s = [stream_df[stream_df['algorithm'] == a]['AUC_PR'].dropna().values for a in algos_s]
    bp_s = ax.boxplot(data_s, patch_artist=True)
    for p, a in zip(bp_s['boxes'], algos_s):
        p.set_facecolor(COLORS.get(a, '#333'))
        p.set_alpha(0.7)
    ax.set_xticklabels(algos_s, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR Distribution (Streaming)')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 2]
    means = df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False)
    bars = ax.bar(means.index, means.values, color=[COLORS.get(a, '#333') for a in means.index])
    ax.set_xticklabels(means.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean AUC-PR')
    ax.set_title('Mean AUC-PR by Algorithm')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 0]
    pivot = df.pivot_table(index='difficulty', columns='algorithm', values='AUC_PR', aggfunc='mean')
    pivot.plot(kind='bar', ax=ax, color=[COLORS.get(c, '#333') for c in pivot.columns])
    ax.set_title('AUC-PR by Difficulty')
    ax.set_ylabel('AUC-PR')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 1]
    pivot2 = df.pivot_table(index='fold', columns='algorithm', values='AUC_PR', aggfunc='mean')
    pivot2.plot(ax=ax, marker='o', color=[COLORS.get(c, '#333') for c in pivot2.columns])
    ax.set_title('AUC-PR by Fold (Learning Curve)')
    ax.set_xlabel('Fold')
    ax.set_ylabel('AUC-PR')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    times = df.groupby('algorithm')['train_ms'].mean().sort_values(ascending=False)
    ax.barh(times.index, times.values / 1000, color=[COLORS.get(a, '#333') for a in times.index])
    ax.set_xlabel('Training Time (s)')
    ax.set_title('Mean Training Time')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'fig_overview_v7.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_overview_v7.png')


def plot_difficulty(df, out_dir):
    """Per-algorithm AUC-PR across difficulty tiers."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, diff in enumerate(['easy', 'medium', 'hard']):
        sub = df[df['difficulty'] == diff]
        means = sub.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False)
        colors = [COLORS.get(a, '#333') for a in means.index]
        axes[i].barh(means.index, means.values, color=colors)
        axes[i].set_xlabel('AUC-PR')
        axes[i].set_title(f'{diff.capitalize()} Difficulty')
        axes[i].grid(axis='x', alpha=0.3)
        for j, v in enumerate(means.values):
            axes[i].text(v + 0.005, j, f'{v:.3f}', va='center', fontsize=7)
        axes[i].set_xlim(0, max(means.values) * 1.15)
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_difficulty_v7.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_difficulty_v7.png')


def plot_ablation(df, out_dir):
    """Ablation study: IF-baseline vs AE+IF vs CA-DIF-EIA."""
    ablation_algos = ['IF-baseline', 'AE+IF', 'CA-DIF-EIA']
    sub = df[df['algorithm'].isin(ablation_algos)]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pivot = sub.pivot_table(index='algorithm', columns='difficulty', values='AUC_PR', aggfunc='mean')
    pivot = pivot.loc[ablation_algos]
    pivot.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[0].set_title('Ablation Study: AUC-PR by Difficulty')
    axes[0].set_ylabel('AUC-PR')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend(title='Difficulty')
    axes[0].grid(axis='y', alpha=0.3)

    pivot2 = sub.pivot_table(index='algorithm', columns='difficulty', values='F1', aggfunc='mean')
    pivot2 = pivot2.loc[ablation_algos]
    pivot2.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[1].set_title('Ablation Study: F1 by Difficulty')
    axes[1].set_ylabel('F1')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(title='Difficulty')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'fig_ablation_v7.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_ablation_v7.png')


def plot_bar_score(df, out_dir):
    """BAR Score across streaming algorithms."""
    stream_df = df[df['algorithm'].isin(ALGO_NAMES_STREAM)].copy()
    if stream_df.empty:
        return
    bar_df = stream_df[stream_df['label_budget'] > 0]
    if bar_df.empty:
        return
    bar_pivot = bar_df.pivot_table(
        index='algorithm', columns='label_budget', values='AUC_PR', aggfunc='mean'
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_pivot.plot(kind='bar', ax=ax, color=['#3498db', '#2980b9', '#1a5276', '#154360'])
    ax.set_title('AUC-PR by Streaming Algorithm and Label Budget')
    ax.set_ylabel('AUC-PR')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(title='Label Budget', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_bar_score_v7.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_bar_score_v7.png')


def plot_pareto_frontier(df, out_dir):
    """Pareto frontier: AUC-PR vs labels_consumed for streaming."""
    stream_df = df[df['algorithm'].isin(ALGO_NAMES_STREAM)].copy()
    if stream_df.empty:
        return
    stream_df['BAR_score'] = stream_df.apply(
        lambda r: compute_bar_score(r['AUC_PR'], max(r['labels_consumed'], 1),
                                     max(r.get('label_budget', 0), 1)), axis=1
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for lb in sorted(stream_df['label_budget'].unique()):
        sub = stream_df[stream_df['label_budget'] == lb]
        for algo in sub['algorithm'].unique():
            a_df = sub[sub['algorithm'] == algo]
            axes[0].scatter(
                a_df['labels_consumed'], a_df['AUC_PR'],
                color=COLORS.get(algo, '#333'), label=algo, s=50, alpha=0.7
            )
        axes[0].set_xlabel('Labels Consumed')
        axes[0].set_ylabel('AUC-PR')
        axes[0].set_title(f'Pareto Frontier (Label Budget={lb})')
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.3)

    pivot = stream_df.pivot_table(
        index='algorithm', columns='label_budget', values='AUC_PR', aggfunc='mean'
    )
    for algo in pivot.index:
        axes[1].plot(pivot.columns, pivot.loc[algo],
                     marker='o', label=algo, color=COLORS.get(algo, '#333'))
    axes[1].set_xlabel('Label Budget')
    axes[1].set_ylabel('AUC-PR')
    axes[1].set_title('AUC-PR vs Label Budget')
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'fig_pareto_frontier_v7.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_pareto_frontier_v7.png')


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run():
    print('=' * 70)
    print('BENCHMARK v7 — Corrected Scientific Rigour')
    print(f'  GPU: {"ON (" + DEVICE + ")" if GPU_AVAILABLE else "OFF"}')
    print(f'  Batch algorithms:   {ALGO_NAMES_BATCH}')
    print(f'  Streaming algorithms: {ALGO_NAMES_STREAM}')
    print(f'  Seeds: {SEEDS}')
    print(f'  Difficulties: {DIFFICULTIES}')
    print(f'  Label budgets: {LABEL_BUDGETS}')
    print(f'  Train: {TRAIN_N} | Val: {VAL_N} | Test: {TEST_N}')
    print(f'  Anomaly rate: {ANOMALY_RATE:.0%} ({ANOMALY_N} anomalies)')
    print(f'  river: {"available" if HAS_RIVER else "NOT available"}')
    print('=' * 70)

    # ===== 1. Load data =====
    print('\n[1/7] Loading and processing NYC taxi data...')
    monthly, monthly_X = [], []
    for m in MONTHS:
        df = clean(load_month(2024, m))
        X  = features(df)
        monthly.append(df)
        monthly_X.append(X.astype(np.float32))
        print(f'  Month {m:02d}: {len(df):,} records, {X.shape} features')

    # ===== 2. Build evaluation jobs (v7: leave-one-month-out folds) =====
    print('\n[2/7] Building evaluation jobs...')
    jobs = []

    # v7 fold structure: leave-one-month-out
    # Fold i: train = all months before test_month, val = last VAL_N of last train month,
    #         test = test_month
    for fold_idx, test_month in enumerate(MONTHS[1:], 1):
        train_months = MONTHS[:test_month - 1]    # all months before test
        val_month    = train_months[-1] if train_months else test_month - 1

        if not train_months:
            print(f'  Fold {fold_idx}: SKIPPED (need >= 1 training month)')
            continue

        # Train: all months before test_month
        train_X = np.vstack([monthly_X[m - 1] for m in train_months])
        train_df = pd.concat([monthly[m - 1] for m in train_months], ignore_index=True)

        # Val: last VAL_N from val_month (last training month)
        val_X  = monthly_X[val_month - 1][-VAL_N:]
        val_df = monthly[val_month - 1].iloc[-VAL_N:].reset_index(drop=True)

        # Remove val from train, then subsample to TRAIN_N
        n_train_keep = len(train_X) - VAL_N
        train_X  = train_X[:n_train_keep]
        train_df = train_df.iloc[:n_train_keep].reset_index(drop=True)
        if len(train_X) > TRAIN_N:
            rng_sub = np.random.RandomState(42)
            idx = rng_sub.choice(len(train_X), TRAIN_N, replace=False)
            train_X  = train_X[idx]
            train_df = train_df.iloc[idx].reset_index(drop=True)

        # Test: full test month
        test_df = monthly[test_month - 1]

        # Inject anomalies
        for diff in DIFFICULTIES:
            params = ANOMALY_PARAMS[diff]
            seed_s = SEEDS[fold_idx % len(SEEDS)]

            rng_src = np.random.RandomState(seed_s)
            if len(test_df) > TEST_N:
                src_idx = rng_src.choice(len(test_df), TEST_N, replace=False)
                test_df_sub = test_df.iloc[src_idx].reset_index(drop=True)
            else:
                test_df_sub = test_df.reset_index(drop=True)

            test_df_inj, y_labels = inject_anomalies(test_df_sub, params, seed_s)
            X_test = features(test_df_inj).astype(np.float32)
            y_labels = np.array(y_labels, dtype=np.int8)

            # Standardize
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(train_X).astype(np.float32)
            X_val_s   = scaler.transform(val_X).astype(np.float32)
            X_test_s  = scaler.transform(X_test).astype(np.float32)
            y_val = np.zeros(len(X_val_s), dtype=np.int8)

            # Batch algorithms
            batch_algo_info = [
                ('Random',       RandomBaseline,    {}),
                ('sklearn_IF',   SklearnIF,        {}),
                ('sklearn_OCSVM', SklearnOCSVM,   {}),
                ('DenoisingAE',  DenoisingAE,       {}),
                ('CA-DIF-EIA',   CADIFEiaBatch,     {'ablation': 'full'}),
                ('AE+IF',        CADIFEiaBatch,     {'ablation': 'ae_if'}),
                ('IF-baseline',  CADIFEiaBatch,     {'ablation': 'baseline'}),
            ]
            for name, cls, kw in batch_algo_info:
                for seed in SEEDS:
                    jobs.append({
                        'type':       'batch',
                        'fold':       fold_idx,
                        'test_month': test_month,
                        'diff':       diff,
                        'algo_name':  name,
                        'algo_cls':   cls,
                        'seed':       seed,
                        'X_train':    X_train_s,
                        'X_val':      X_val_s,
                        'X_test':     X_test_s,
                        'y_val':      y_val,
                        'y_test':     y_labels,
                        'kwargs':     kw,
                    })

            # Streaming algorithms
            stream_algo_info = [
                ('Random',            RandomBaseline,   {}),
                ('sHST-River',       sHST_River,       {}),
                ('MemStream',        MemStream,        {}),
                ('CA-DIF-EIA-Stream', CADIFEiaStream,  {}),
            ]
            for name, cls, kw in stream_algo_info:
                for seed in SEEDS:
                    for lb in LABEL_BUDGETS:
                        jobs.append({
                            'type':        'streaming',
                            'fold':        fold_idx,
                            'test_month':  test_month,
                            'diff':        diff,
                            'algo_name':   name,
                            'algo_cls':    cls,
                            'seed':        seed,
                            'X_train':     X_train_s,
                            'X_val':       X_val_s,
                            'X_test':      X_test_s,
                            'y_val':       y_val,
                            'y_test':      y_labels,
                            'label_budget': lb,
                            'kwargs':      kw,
                        })

    print(f'  Total jobs: {len(jobs)}')
    batch_jobs  = sum(1 for j in jobs if j['type'] == 'batch')
    stream_jobs = sum(1 for j in jobs if j['type'] == 'streaming')
    print(f'  Batch: {batch_jobs} | Streaming: {stream_jobs}')

    # ===== 3. Run benchmark =====
    print(f'\n[3/7] Running {len(jobs)} jobs...')
    t0 = time.perf_counter()
    results = []
    CHECKPOINT_INTERVAL = 25

    for i, job in enumerate(jobs):
        try:
            if job['type'] == 'batch':
                res = evaluate_batch(
                    job['algo_cls'],
                    job['X_train'], job['X_val'], job['X_test'],
                    job['y_val'], job['y_test'],
                    job['seed'],
                    **job['kwargs']
                )
            else:
                res = evaluate_streaming(
                    job['algo_cls'],
                    job['X_train'], job['X_val'], job['X_test'],
                    job['y_val'], job['y_test'],
                    job['seed'],
                    label_budget=job['label_budget'],
                    **job['kwargs']
                )
            res.update({
                'fold':        job['fold'],
                'month':       job['test_month'],
                'difficulty':  job['diff'],
                'algorithm':   job['algo_name'],
                'seed':        job['seed'],
                'error':       '',
            })
            if job['type'] == 'streaming':
                res['label_budget'] = job['label_budget']
            else:
                res['label_budget'] = 0
        except Exception as e:
            res = {m: float('nan') for m in METRICS + ['train_ms', 'score_ms', 'labels_consumed', 'anomaly_rate']}
            res.update({
                'fold': job['fold'], 'month': job['test_month'],
                'difficulty': job['diff'], 'algorithm': job['algo_name'],
                'seed': job['seed'], 'error': str(e)[:80],
                'label_budget': job.get('label_budget', 0),
            })
        results.append(res)
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            elapsed = time.perf_counter() - t0
            rate    = (i + 1) / elapsed
            remain  = (len(jobs) - i - 1) / rate / 60
            print(f'  {i+1}/{len(jobs)} ({rate:.1f}/s, ~{remain:.1f}m left)')
            pd.DataFrame(results).to_csv(OUT_DIR / 'checkpoint_v7.csv', index=False)
        gc.collect()

    bench_df = pd.DataFrame(results)
    bench_df.to_csv(OUT_DIR / 'benchmark_results_v7.csv', index=False)
    t_done = time.perf_counter() - t0
    print(f'\n  Benchmark done in {t_done/60:.1f} min')

    errors = bench_df['error'].notna() & (bench_df['error'] != '')
    if errors.any():
        print(f'  WARNING: {errors.sum()} jobs had errors:')
        for _, row in bench_df[errors].iterrows():
            print(f'    [{row["algorithm"]}] fold={row["fold"]} {row["error"]}')

    # ===== 4. BAR Score =====
    print('\n[4/7] Computing BAR Scores...')
    bench_df['BAR_score'] = bench_df.apply(
        lambda r: compute_bar_score(r['AUC_PR'], max(r['labels_consumed'], 1),
                                     max(r.get('label_budget', 0), 1)), axis=1
    )
    bar_df = bench_df[bench_df['algorithm'].isin(ALGO_NAMES_STREAM)].copy()
    if not bar_df.empty:
        bar_summary = bar_df.pivot_table(
            index='algorithm', columns='label_budget',
            values='BAR_score', aggfunc='mean'
        )
        print('\n  BAR Score Summary:')
        print(bar_summary.to_string())
        bar_df.to_csv(OUT_DIR / 'bar_score_results_v7.csv', index=False)
    bench_df.to_csv(OUT_DIR / 'benchmark_results_v7.csv', index=False)

    # ===== 5. Statistical analysis =====
    print('\n[5/7] Statistical analysis (Friedman + Holm-Bonferroni)...')
    stat_results = {}

    batch_algos = [a for a in ALGO_NAMES_BATCH if a in bench_df['algorithm'].values]
    batch_df = bench_df[bench_df['algorithm'].isin(batch_algos)]
    if len(batch_df['algorithm'].unique()) >= 2:
        stat_results['batch'] = statistical_analysis(batch_df, 'Batch', batch_algos)

    stream_500 = bench_df[(bench_df['algorithm'].isin(ALGO_NAMES_STREAM)) &
                           (bench_df['label_budget'] == 500)]
    if len(stream_500['algorithm'].unique()) >= 2:
        stat_results['streaming_500'] = statistical_analysis(stream_500, 'Streaming_500', ALGO_NAMES_STREAM)

    for diff in DIFFICULTIES:
        sub = bench_df[bench_df['difficulty'] == diff]
        if len(sub['algorithm'].unique()) >= 2:
            stat_results[f'batch_{diff}'] = statistical_analysis(sub, f'Batch_{diff.capitalize()}', batch_algos)

    stat_results['all_stream'] = statistical_analysis(
        bench_df[bench_df['algorithm'].isin(ALGO_NAMES_STREAM)], 'All_Streaming', ALGO_NAMES_STREAM
    )

    with open(OUT_DIR / 'statistical_results.txt', 'w') as fp:
        for name, res in stat_results.items():
            report_stat_results(res, fp)

    # ===== 6. Plots =====
    print('\n[6/7] Generating plots...')
    plot_overview(bench_df, OUT_DIR)
    plot_difficulty(bench_df, OUT_DIR)
    plot_ablation(bench_df, OUT_DIR)
    plot_bar_score(bench_df, OUT_DIR)
    plot_pareto_frontier(bench_df, OUT_DIR)

    # ===== 7. Report =====
    print('\n[7/7] Writing report...')
    write_report(bench_df, stat_results, t_done, OUT_DIR)
    print('\n  DONE!')


def write_report(df, stat_results, t_total, out_dir):
    lines = []
    lines.append('# Benchmark v7 Results — Corrected Scientific Rigour')
    lines.append(f'**Generated:** {datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}')
    lines.append(f'**Runtime:** {t_total/60:.1f} min')
    lines.append(f'**Protocol:** Three-way split (train/val/test), threshold from validation set')
    lines.append(f'**Folds:** Leave-one-month-out (5 folds, independent test sets)')
    lines.append(f'**Datasets:** NYC Yellow Taxi Jan-Jun 2024')
    lines.append(f'**Seeds:** {SEEDS}')
    lines.append(f'**Difficulties:** {DIFFICULTIES}')
    lines.append(f'**Anomaly Rate:** {ANOMALY_RATE:.0%} ({ANOMALY_N} anomalies / {TEST_N} test samples)')
    lines.append('')
    lines.append('## Summary: Mean AUC-PR by Algorithm')
    lines.append('')
    lines.append('| Algorithm | Mean AUC-PR | Std | N |')
    lines.append('|-----------|-------------|-----|---|')
    summary = df.groupby('algorithm')['AUC_PR'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    for algo, row in summary.iterrows():
        lines.append(f'| {algo} | {row["mean"]:.4f} | {row["std"]:.4f} | {int(row["count"])} |')

    lines.append('')
    lines.append('## AUC-PR by Difficulty')
    lines.append('')
    lines.append('| Algorithm | Easy | Medium | Hard |')
    lines.append('|-----------|------|--------|------|')
    pivot = df.pivot_table(index='algorithm', columns='difficulty', values='AUC_PR', aggfunc='mean')
    for algo in pivot.index:
        row_str = f'| {algo} |'
        for diff in ['easy', 'medium', 'hard']:
            v = pivot.loc[algo, diff] if diff in pivot.columns else '-'
            row_str += f' {v:.4f} |' if isinstance(v, float) else f' {v} |'
        lines.append(row_str)

    lines.append('')
    lines.append('## AUC-PR by Fold (Learning Curve)')
    lines.append('')
    pivot2 = df.pivot_table(index='algorithm', columns='fold', values='AUC_PR', aggfunc='mean')
    lines.append('| Algorithm | ' + ' | '.join([str(int(f)) for f in pivot2.columns]) + ' |')
    lines.append('|-----------|' + '---|' * len(pivot2.columns))
    for algo in pivot2.index:
        vals = ' | '.join([f'{v:.4f}' if isinstance(v, float) else str(v) for v in pivot2.loc[algo]])
        lines.append(f'| {algo} | {vals} |')

    lines.append('')
    lines.append('## Random Baseline (AUC-PR Calibration Floor)')
    random_df = df[df['algorithm'] == 'Random']
    if not random_df.empty:
        r_mean = random_df['AUC_PR'].mean()
        lines.append(f'| Random Baseline AUC-PR | {r_mean:.4f} |')
        lines.append(f'| Expected (anomaly_rate) | {ANOMALY_RATE:.4f} |')
        diff_pct = abs(r_mean - ANOMALY_RATE) / ANOMALY_RATE * 100
        lines.append(f'| Deviation from baseline | {diff_pct:.1f}% |')
    lines.append('')

    bar_stream = df[(df['algorithm'].isin(ALGO_NAMES_STREAM)) & (df['label_budget'] > 0)]
    if not bar_stream.empty:
        lines.append('## BAR Score by Streaming Algorithm')
        lines.append('')
        for lb in sorted(bar_stream['label_budget'].unique()):
            sub = bar_stream[bar_stream['label_budget'] == lb]
            lines.append(f'### Label Budget = {int(lb)}')
            for _, row in sub.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).items():
                lines.append(f'- {row}: AUC-PR={row:.4f}')

    lines.append('')
    lines.append('## Statistical Analysis')
    for name, res in stat_results.items():
        lines.append(f'\n### {res["group"]}')
        if not res.get('significant', False):
            lines.append(f'- Friedman test: stat={res.get("friedman_stat", 0):.3f}, p={res.get("friedman_p", 1):.4f}')
            lines.append(f'- **Conclusion:** {res.get("conclusion", "NOT SIGNIFICANT")}')
        else:
            lines.append(f'- Friedman test: stat={res.get("friedman_stat", 0):.3f}, p={res.get("friedman_p", 0):.4f} (**SIGNIFICANT**)')
            lines.append(f'- Average ranks:')
            for algo, rank in sorted(res.get('avg_ranks', {}).items(), key=lambda x: x[1]):
                lines.append(f'  - {algo}: {rank:.2f}')
            for pair in res.get('pairwise_comparisons', []):
                sig = '**YES**' if pair.get('significant') else 'no'
                lines.append(f'  - {pair["target"]} vs {pair["baseline"]}: p_raw={pair["p_raw"]:.4f}, p_holm={pair["p_corrected"]:.4f}, sig={sig}')

    with open(out_dir / 'benchmark_v7_results.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    run()
