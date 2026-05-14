"""
Benchmark v6 — Scientific Rigour Overhaul
==========================================
Key fixes from v5:
  - Three-way split: train / val (threshold) / test (final eval)
  - No test-set threshold optimization (fixed data leakage)
  - Anomaly injection redesigned by difficulty tier (easy/medium/hard)
  - Training data = 100% normal (unsupervised methods train clean)
  - Statistical pipeline: Friedman + Holm-Bonferroni + Wilcoxon + Bootstrap CIs
  - CA-DIF-EIA with trained autoencoder (not random projection)
  - Streaming with pre-generated oracle query schedule
  - BAR Score and Pareto Frontier chart
  - Critical Difference (CD) diagrams

Author: Claude (based on plan_v6.md)
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
OUT_DIR  = Path(r'C:\proj\ldt\results\v6')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS       = [1, 2, 3, 4, 5, 6]
SEEDS        = [42, 123, 456, 789, 1000]
DIFFICULTIES = ['easy', 'medium', 'hard']
LABEL_BUDGETS = [0, 100, 500, 1000]   # 0%, 1%, 5%, 10% of 10K test stream

TRAIN_N = 10000
VAL_N    = 2000    # held out from training month
TEST_N   = 10000

ANOMALY_RATE  = 0.15   # 15% in test set (1,500 anomalies / 10,000 total)
ANOMALY_N     = int(TEST_N * ANOMALY_RATE)

ANOMALY_PARAMS = {
    'easy':   {'type': 'meter_mult', 'range': (10, 20),  'n': ANOMALY_N},
    'medium': {'type': 'meter_mult', 'range': (4, 8),    'n': ANOMALY_N},
    'hard':   {'type': 'combined',   'components': [
        ('meter_mult', (1.5, 3.0)),
        ('gps_spoof',  (1.5, 3.0)),
        ('slow_crawl', None),
    ], 'n': ANOMALY_N},
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
# ANOMALY INJECTION (v6 — redesigned by difficulty tier)
# =============================================================================

def inject_anomalies(df, params, seed):
    """
    Inject anomalies ONLY into the provided dataset.
    Training data stays 100% normal (Decision #4 in plan_v6.md).

    Injection is by difficulty tier:
      easy:   meter_mult 10-20x  → clear outlier
      medium: meter_mult 4-8x     → visible fraud
      hard:   combined (meter_mult 1.5-3x + gps_spoof + slow_crawl) → subtle fraud

    All 1,500 injected anomalies at 15% anomaly rate.
    """
    rng    = np.random.RandomState(seed)
    n_anom = params['n']

    inj_idx = rng.choice(len(df), n_anom, replace=False)
    df_a = df.iloc[inj_idx].copy().reset_index(drop=True)

    ptype = params.get('type', None)

    if ptype == 'meter_mult':
        mult = rng.uniform(*params['range'], size=n_anom)
        df_a['fare_amount'] = df_a['trip_distance'] * 2.5 * mult

    elif ptype == 'gps_spoof':
        spd = rng.uniform(*params['range'], size=n_anom)
        df_a['trip_distance'] = df_a['dur_min'] * spd / 60.0
        df_a['fare_amount']   = df_a['trip_distance'] * 2.5

    elif ptype == 'slow_crawl':
        df_a['dur_min']       = rng.uniform(40, 120, n_anom)
        df_a['trip_distance'] = rng.uniform(0.5, 3.0, n_anom)
        df_a['fare_amount']   = rng.uniform(8, 30, n_anom)

    elif ptype == 'combined':
        components = params['components']
        for comp_type, comp_range in components:
            if comp_type == 'meter_mult':
                mult = rng.uniform(*comp_range, size=n_anom)
                df_a['fare_amount'] = df_a['trip_distance'] * 2.5 * mult
            elif comp_type == 'gps_spoof':
                spd = rng.uniform(*comp_range, size=n_anom)
                df_a['trip_distance'] = df_a['dur_min'] * spd / 60.0
                df_a['fare_amount']   = df_a['trip_distance'] * 2.5
            elif comp_type == 'slow_crawl':
                df_a['dur_min']       = rng.uniform(20, 60, n_anom)
                df_a['trip_distance'] = rng.uniform(0.5, 3.0, n_anom)

    # Combine and shuffle
    df_combined = pd.concat([df, df_a], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    labels = np.concatenate([np.zeros(len(df), dtype=np.int8),
                               np.ones(n_anom, dtype=np.int8)])
    return df_combined, labels


# =============================================================================
# ALGORITHMS
# =============================================================================

# ---------- Helper: ADWIN-U drift detector ----------

class ADWINU:
    """
    ADWIN-based unsupervised drift detector (ADWIN-U).
    Detects drift by monitoring the score distribution.
    """
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
        return self

    def decision_function(self, X):
        return -self.model_.decision_function(X)

    def predict(self, X, threshold=None):
        if threshold is not None:
            scores = self.decision_function(X)
            return np.where(scores > threshold, -1, 1)
        return self.model_.predict(X)


# ---------- Baseline: sklearn_LOF ----------

class SklearnLOF:
    name = 'sklearn_LOF'
    supports_streaming = False

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X, X_val=None, y_val=None):
        n = min(10000, len(X))
        idx = np.random.RandomState(self.seed).choice(len(X), n, replace=False)
        self.model_ = LocalOutlierFactor(
            n_neighbors=20, contamination=0.05,
            novelty=True, n_jobs=4
        )
        self.model_.fit(X[idx])
        return self

    def decision_function(self, X):
        return -self.model_.decision_function(X)

    def predict(self, X, threshold=None):
        if threshold is not None:
            scores = self.decision_function(X)
            return np.where(scores > threshold, -1, 1)
        return self.model_.predict(X)


# ---------- Denoising Autoencoder (replaces LSTM-AE) ----------

class DenoisingAE:
    """
    Denoising Autoencoder for anomaly detection.
    Train on normal data only. High reconstruction error = anomaly.
    """
    name = 'DenoisingAE'
    supports_streaming = False

    def __init__(self, seed=42, hidden_dim=32, latent_dim=16):
        self.seed      = seed
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.scaler_   = None
        self.model_    = None
        self.threshold_ = None
        self._device   = DEVICE
        self._n_feats  = 0

    def fit(self, X_train, X_val=None, y_val=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X_train)
        self._n_feats = Xs.shape[1]

        # Subsample for faster AE training (50K max)
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
                z  = self.enc(x)
                return self.dec(z)

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

        # Threshold from validation set
        if X_val is not None:
            X_val_s  = self.scaler_.transform(X_val)
            recon_v  = self._reconstruct(X_val_s)
            errors_v = np.mean(np.abs(X_val_s - recon_v), axis=1)
            self.threshold_ = float(np.percentile(errors_v, 95))

        return self

    def _reconstruct(self, X):
        import torch
        t = torch.FloatTensor(X.astype(np.float32))
        if self._device == 'cuda':
            t = t.cuda()
        with torch.no_grad():
            out = self.model_(t)
            if self._device == 'cuda':
                out = out.cpu()
        return out.numpy()

    def decision_function(self, X):
        if self.model_ is None or self.scaler_ is None:
            return np.full(len(X), 0.5)
        Xs    = self.scaler_.transform(X)
        recon = self._reconstruct(Xs)
        return np.mean(np.abs(Xs - recon), axis=1).astype(np.float64)

    def predict(self, X, threshold=None):
        if threshold is None:
            threshold = self.threshold_
        if threshold is None:
            return np.full(len(X), 1)
        return np.where(self.decision_function(X) > threshold, -1, 1)


# ---------- Context Feature Weighting (shared component) ----------

class ContextFeatureWeighting:
    """
    Learns per-context feature importance from training data.
    Context = (hour_bin, day_of_week_bin).
    """
    def __init__(self, n_contexts=24, n_features=25):
        self.n_contexts = n_contexts
        self.weights    = np.ones((n_contexts, n_features), dtype=np.float32)

    def fit(self, X_train, hour_vals=None, dow_vals=None):
        if hour_vals is None:
            hour_vals = X_train[:, 9].astype(int)
        if dow_vals is None:
            dow_vals = X_train[:, 10].astype(int)

        hour_bin = np.clip(hour_vals, 0, 23)
        dow_bin  = np.clip(dow_vals,  0,  6)
        context_ids = hour_bin * 7 + dow_bin  # 24*7 = 168 contexts

        for c in range(self.n_contexts):
            mask = context_ids == c
            if mask.sum() < 50:
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


# ---------- Trained Autoencoder for CA-DIF-EIA ----------

class TrainedAutoencoder:
    """
    Trains on normal data only. Bottleneck provides learned projection.
    """
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

        # Subsample for faster training (50K max)
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
                z = self.encoder(x)
                return self.decoder(z)

            def encode(self, x):
                return self.encoder(x)

        self.model_ = AE(self.input_dim, self.hidden_dim, self.latent_dim)
        if self.device == 'cuda':
            self.model_.cuda()

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=lr)
        criterion = nn.MSELoss()

        tensor_x = torch.FloatTensor(X_normal.astype(np.float32))
        ds = TensorDataset(tensor_x, tensor_x)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

        for epoch in range(epochs):
            for bx, by in dl:
                if self.device == 'cuda':
                    bx, by = bx.cuda(), by.cuda()
                recon = self.model_(bx)
                loss  = criterion(recon, by)
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

    def decision_function(self, X):
        """Reconstruction error as anomaly score."""
        import torch
        t    = torch.FloatTensor(X.astype(np.float32))
        if self.device == 'cuda':
            t = t.cuda()
        with torch.no_grad():
            recon = self.model_(t)
            if self.device == 'cuda':
                recon = recon.cpu()
        recon = recon.numpy()
        return np.mean(np.abs(X.astype(np.float32) - recon), axis=1).astype(np.float64)


# ---------- CA-DIF-EIA (Batch) — Proper Implementation ----------

class CADIFEiaBatch:
    """
    CA-DIF-EIA v6 (Batch):
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
        self.ablation = ablation  # 'baseline', 'ae_if', 'full'
        self._ae      = None
        self._if      = None
        self._cw      = None
        self.thresh_  = None

    def fit(self, X_train, X_val=None, y_val=None):
        rng = np.random.RandomState(self.seed)

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

        # Step 4: Threshold from validation set (if provided)
        if X_val is not None and y_val is not None:
            val_scores = self.decision_function(X_val)
            self.thresh_ = float(np.percentile(val_scores, 95))
        elif X_val is not None:
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

class CADIFEiaStream:
    """
    CA-DIF-EIA v6 (Streaming):
      - ADWIN-U drift detector
      - Label budget for active querying
      - Context-aware scoring
    """
    name = 'CA-DIF-EIA (streaming)'
    supports_streaming = True

    def __init__(self, seed=42, label_budget=500, drift_delta=0.002):
        self.seed        = seed
        self.label_budget = label_budget
        self.drift_delta  = drift_delta
        self._rng         = np.random.RandomState(seed)
        self._budget_used = 0
        self._drift       = ADWINU(delta=drift_delta, size=500)
        self._n_feats     = None
        self._W1          = None
        self._b1          = None
        self._if          = None
        self._cw          = None
        self._threshold   = None
        self._warmup_scores = []
        self._context_hist = []
        self._calibrated   = False

    def fit(self, X_train):
        warmup_n = min(int(len(X_train) * 0.2), 3000)
        X_warmup = X_train[:warmup_n]

        self._n_feats = X_train.shape[1]
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

        # Calibrate threshold from warmup scores
        for x in X_warmup[:500]:
            self._warmup_scores.append(self.score_one(x))

        self._calibrated = True
        return self

    def _proj(self, X):
        return np.maximum(X.astype(np.float32) @ self._W1 + self._b1, 0)

    def decision_function(self, X):
        """Vectorized batch scoring — much faster than per-record."""
        Xf = X.astype(np.float32)
        # Batch project
        Xp = self._proj(Xf)
        # Batch IF scoring
        iso_scores = -self._if.score_samples(Xp)
        # Batch context weights
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
            self._budget_used += 1
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

    def decision_function(self, X):
        """Vectorized batch scoring — much faster than per-record."""
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

    def get_threshold(self):
        if len(self._warmup_scores) >= 100:
            return float(np.percentile(self._warmup_scores, 95))
        return 0.5

    def predict(self, X, threshold=None):
        d = self.decision_function(X)
        t = threshold if threshold is not None else self.get_threshold()
        return np.where(d > t, -1, 1)


# ---------- MemStream ----------

class MemStream:
    name = 'MemStream'
    supports_streaming = True

    def __init__(self, seed=42, k=10, memory_cap=2000):
        self.seed       = seed
        self.k          = k
        self.memory_cap = memory_cap
        self._rng       = np.random.RandomState(seed)
        self.memory     = []
        self._buf       = []

    def fit(self, X):
        for i in range(min(500, len(X))):
            self.update_one(X[i].astype(np.float64))

    def score_one(self, x):
        if len(self.memory) < self.k:
            return 0.5
        mem   = np.array(self.memory, dtype=np.float64)
        dists = np.sort(np.linalg.norm(mem - x.astype(np.float64), axis=1))
        return float(dists[:self.k].mean())

    def update_one(self, x, label=None):
        xf = x.astype(np.float64)
        if len(self.memory) < self.memory_cap:
            self.memory.append(xf)
        else:
            self.memory[self._rng.randint(0, len(self.memory))] = xf
        s = self.score_one(x)
        self._buf.append(s)
        if len(self._buf) > 500:
            self._buf.pop(0)

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


# ---------- sHST-River (using river library) ----------

FEATURE_NAMES = ['f%d' % i for i in range(25)]

def _array_to_dict(x):
    """Convert numpy array to dict for river."""
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
        for x in X[:200]:
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


# ---------- IForestASD (fixed decision_function) ----------

class IForestASD:
    name = 'IForestASD'
    supports_streaming = True

    def __init__(self, seed=42):
        self.seed        = seed
        self._rng        = np.random.RandomState(seed)
        self.window_size = 2000
        self.n_trees     = 100
        self.max_samples = 512
        self.trees       = []
        self.buffer      = []

    def fit(self, X):
        for i in range(min(self.window_size, len(X))):
            self._partial_fit(X[i])

    def _partial_fit(self, x):
        xf = x.reshape(-1).astype(np.float32)
        if len(self.buffer) >= self.window_size:
            self.buffer.pop(0)
        self.buffer.append(xf)
        if len(self.buffer) < self.max_samples:
            return
        buf = np.array(list(self.buffer)[-self.max_samples:], dtype=np.float32)
        self.trees = []
        for _ in range(self.n_trees):
            idx = self._rng.choice(len(buf), min(self.max_samples, len(buf)), replace=False)
            s   = buf[idx]
            fd  = s.shape[1]
            fi  = self._rng.randint(0, fd)
            lo, hi = s[:, fi].min(), s[:, fi].max()
            sp = self._rng.uniform(lo, hi + 1e-8)
            self.trees.append((fi, float(sp)))

    def score_one(self, x):
        if not self.trees:
            return 0.5
        xf = x.astype(np.float32)
        depth_sum = sum(0.0 if xf[fi] < sp else 1.0 for fi, sp in self.trees)
        return float(depth_sum / len(self.trees))

    def decision_function(self, X):
        scores = np.zeros(len(X), dtype=np.float64)
        for idx, x in enumerate(X):
            scores[idx] = self.score_one(x)
        return scores

    def update_one(self, x, label=None):
        self._partial_fit(x)

    def predict(self, X, threshold=None):
        d = self.decision_function(X)
        t = threshold if threshold is not None else 0.5
        return np.where(d > t, -1, 1)


# ---------- Algorithm registry ----------

ALGOS_BATCH = [
    SklearnIF,
    DenoisingAE,
    CADIFEiaBatch,   # ablation='full'
    CADIFEiaBatch,   # ablation='ae_if'  (ablation study)
    CADIFEiaBatch,   # ablation='baseline' (ablation study)
]

ALGOS_STREAM = [
    sHST_River,
    MemStream,
    CADIFEiaStream,
]

ALGO_NAMES_BATCH   = ['sklearn_IF', 'DenoisingAE',
                      'CA-DIF-EIA', 'AE+IF', 'IF-baseline']
ALGO_NAMES_STREAM = ['sHST-River', 'MemStream', 'CA-DIF-EIA (streaming)']


# =============================================================================
# EVALUATION PROTOCOL (Three-Way Split — v6)
# =============================================================================

def evaluate_batch(algo_cls, X_train, X_val, X_test, y_val, y_test, seed, **kwargs):
    """
    Three-way evaluation for batch algorithms.

    Protocol:
      1. Train on X_train (100% normal — no injection)
      2. Get validation scores, find optimal threshold on X_val (with y_val labels)
      3. Apply threshold to X_test, compute final metrics

    CRITICAL: y_test is NEVER used for threshold selection.
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

    test_preds = algo.predict(X_test, threshold=algo.thresh_ if hasattr(algo, 'thresh_') and algo.thresh_ is not None else None)
    # sklearn models return -1/1; y_test is 0/1 — normalize to 0/1 for metrics
    test_preds = np.where(test_preds == -1, 0, test_preds)

    pr_curve, rc_curve, _ = precision_recall_curve(y_test, test_scores)
    auc_pr  = auc(rc_curve, pr_curve) if len(rc_curve) > 1 else 0.0
    # Guard against non-monotonic rc_curve (can happen with constant scores)
    if not (np.all(np.diff(rc_curve) <= 0) or np.all(np.diff(rc_curve) >= 0)):
        # Sort by scores descending and recompute
        order = np.argsort(test_scores)[::-1]
        sorted_scores = test_scores[order]
        sorted_labels = y_test[order]
        pr_curve2, rc_curve2, _ = precision_recall_curve(sorted_labels, sorted_scores)
        auc_pr = auc(rc_curve2, pr_curve2) if len(rc_curve2) > 1 else 0.0
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
    Streaming evaluation with pre-generated oracle query schedule.

    Phase 1 (Calibration): Score X_val without updating model.
      - Threshold is set from X_train scores (contamination-based percentile)

    Phase 2 (Test): Stream X_test with label budget.
      - Pre-generated query positions (deterministic from seed)
      - y_test[i] provides the "oracle" response
      - y_test is NEVER used for decisions inside the loop

    CRITICAL: y_test is NEVER used inside the streaming loop for decisions.
    """
    ablation = kwargs.get('ablation', 'full')
    algo = algo_cls(seed=seed, label_budget=label_budget) if 'CADIFEia' in algo_cls.__name__ else algo_cls(seed=seed)

    # Warmup model on training data
    algo.fit(X_train.astype(np.float64))

    # Phase 1: Calibration — set threshold from training data scores (batch)
    train_scores_cal = algo.decision_function(X_train[:1000].astype(np.float64))
    threshold = float(np.percentile(train_scores_cal, 95))

    # Phase 2: Chunked streaming — batch score + sequential update
    # This is the KEY optimization: score in batches (vectorized),
    # only update model sequentially for labeled points
    CHUNK = 2000
    test_scores = []
    labels_used = []

    X_test_f = X_test.astype(np.float64)
    for chunk_start in range(0, len(X_test_f), CHUNK):
        chunk_end = min(chunk_start + CHUNK, len(X_test_f))
        chunk_X = X_test_f[chunk_start:chunk_end]

        # Batch score — much faster than per-record
        chunk_scores = algo.decision_function(chunk_X)
        test_scores.append(chunk_scores)

        # Sequential update loop for this chunk (only labeled points modify model)
        rng_c = np.random.RandomState(seed + chunk_start)
        total_processed = chunk_start + len(chunk_X)
        n_queries = min(label_budget, total_processed)
        query_positions = sorted(rng_c.choice(total_processed, min(n_queries, total_processed), replace=False))

        for i, x in enumerate(chunk_X):
            global_i = chunk_start + i
            is_query = (global_i in set(query_positions)) and (len(labels_used) < label_budget)
            if is_query:
                true_label = int(y_test[global_i])
                algo.update_one(x, label=true_label)
                labels_used.append((global_i, true_label))

    test_scores = np.concatenate(test_scores)
    test_preds  = (test_scores >= threshold).astype(np.int8)

    # Guard against constant scores (causes auc() to fail)
    try:
        pr_curve, rc_curve, _ = precision_recall_curve(y_test, test_scores)
        auc_pr = auc(rc_curve, pr_curve) if len(rc_curve) > 1 else 0.0
        if not (np.all(np.diff(rc_curve) <= 0) or np.all(np.diff(rc_curve) >= 0)):
            order = np.argsort(test_scores)[::-1]
            pr_curve2, rc_curve2, _ = precision_recall_curve(y_test[order], test_scores[order])
            auc_pr = auc(rc_curve2, pr_curve2) if len(rc_curve2) > 1 else 0.0
    except Exception:
        auc_pr = 0.0
    try:
        fpr_arr, tpr_arr, _ = roc_curve(y_test, test_scores)
        auc_roc = auc(fpr_arr, tpr_arr) if len(fpr_arr) > 1 else 0.5
    except Exception:
        auc_roc = 0.5
    f1      = f1_score(y_test, test_preds, zero_division=0)
    prc     = precision_score(y_test, test_preds, zero_division=0)
    rec     = recall_score(y_test, test_preds, zero_division=0)
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
# BAR SCORE — Budget-Aware Ranking
# =============================================================================

def compute_bar_score(auc_pr, labels_used, label_budget):
    """
    BAR = 100 × (AUC_PR / Labels_Used) × min(1, Labels_Used / Label_Budget)

    Higher BAR = more efficient use of labels.
    BAR = 0 when no labels are used and no performance gain.
    """
    if labels_used == 0 or label_budget == 0:
        return 0.0
    efficiency  = auc_pr / labels_used
    utilization = min(1.0, labels_used / label_budget)
    return 100.0 * efficiency * utilization


# =============================================================================
# STATISTICAL ANALYSIS — Friedman + Holm-Bonferroni + Bootstrap CIs
# =============================================================================

def statistical_analysis(df, group_name, algos):
    """
    Friedman omnibus + Holm-Bonferroni post-hoc (Wilcoxon pairwise).

    Pipeline:
      1. Friedman test: H0 = all algorithms have equal AUC-PR
         -> p < 0.05: reject H0, proceed to post-hoc
         -> p >= 0.05: STOP, report "no significant differences"

      2. Average ranks: lower rank = better

      3. Holm-Bonferroni post-hoc (one-sided, CA-DIF-EIA vs each baseline):
         For each pair (CA-DIF-EIA, baseline_i):
           - Wilcoxon signed-rank, H1: CA-DIF-EIA > baseline_i
           - Holm threshold: alpha_i = 0.05 / (m - i + 1)
           - Reject if p_holm < 0.05

      4. Bootstrap 95% CIs (1000 iterations)
    """
    pivot = df.pivot_table(index='fold', columns='algorithm', values='AUC_PR')
    pivot = pivot[algos]

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

    # Average rank per algorithm
    def rank_row(row):
        return rankdata(row.values, method='average')
    ranks = pivot.apply(rank_row, axis=1)
    avg_ranks = ranks.mean(axis=0).sort_values()

    # Holm-Bonferroni — CA-DIF-EIA vs each baseline
    target   = 'CA-DIF-EIA'
    baselines = [a for a in algos if a != target]
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

    # Holm correction
    m = len(pairwise)
    sorted_pairs = sorted(pairwise, key=lambda x: x['p_raw'])
    for rank_i, pair in enumerate(sorted_pairs, 1):
        holm_alpha = 0.05 / (m - rank_i + 1)
        pair['holm_alpha'] = holm_alpha
        pair['p_corrected'] = min(pair['p_raw'] * (m - rank_i + 1), 1.0)
        pair['significant'] = pair['p_corrected'] < 0.05

    # Bootstrap 95% CIs
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
    """Pretty-print statistical results."""
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
    'sklearn_IF':         '#95a5a6',
    'DenoisingAE':        '#9b59b6',
    'CA-DIF-EIA':         '#e74c3c',
    'AE+IF':              '#c0392b',
    'IF-baseline':        '#d5d8dc',
    'sHST-River':         '#3498db',
    'MemStream':           '#2980b9',
    'CA-DIF-EIA (streaming)': '#c0392b',
}


def plot_overview(df, out_dir):
    """Overview plots: AUC-PR distribution, bar chart, F1, timing."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0, 0]
    algos = df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index
    data  = [df[df['algorithm'] == a]['AUC_PR'].dropna().values for a in algos]
    bp = ax.boxplot(data, patch_artist=True)
    for p, a in zip(bp['boxes'], algos):
        p.set_facecolor(COLORS.get(a, '#333'))
        p.set_alpha(0.7)
    ax.set_xticklabels(algos, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR Distribution')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 1]
    colors_d = {'easy': '#27ae60', 'medium': '#f39c12', 'hard': '#c0392b'}
    top = df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index
    x  = np.arange(len(top))
    width = 0.22
    for di, diff in enumerate(['easy', 'medium', 'hard']):
        d = df[df['difficulty'] == diff].groupby('algorithm')['AUC_PR'].mean()
        d = d.reindex(top)
        ax.bar(x + di * width, d.values, width, label=diff.capitalize(),
               color=colors_d[diff], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(top, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR by Difficulty')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 2]
    f1_mean = df.groupby('algorithm')['F1'].mean().sort_values(ascending=True)
    ax.barh(range(len(f1_mean)), f1_mean.values,
            color=[COLORS.get(a, '#333') for a in f1_mean.index])
    ax.set_yticks(range(len(f1_mean)))
    ax.set_yticklabels(f1_mean.index, fontsize=8)
    ax.set_xlabel('F1')
    ax.set_title('Mean F1')
    ax.grid(axis='x', alpha=0.3)

    ax = axes[1, 0]
    times = df.groupby('algorithm')['score_ms'].mean().sort_values(ascending=True)
    ax.barh(range(len(times)), times.values,
            color=[COLORS.get(a, '#333') for a in times.index])
    ax.set_yticks(range(len(times)))
    ax.set_yticklabels(times.index, fontsize=8)
    ax.set_xlabel('Score time (ms)')
    ax.set_title('Mean Scoring Time')
    ax.grid(axis='x', alpha=0.3)

    ax = axes[1, 1]
    for algo in df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).head(3).index:
        mdata = df[df['algorithm'] == algo].groupby('fold')['AUC_PR'].mean()
        ax.plot(mdata.index, mdata.values, 'o-', label=algo,
                color=COLORS.get(algo, '#333'), linewidth=2, markersize=5)
    ax.set_xlabel('Fold')
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR Over Folds')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    for algo in df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).head(3).index:
        mdata = df[df['algorithm'] == algo].groupby('difficulty')['AUC_PR'].mean()
        ax.plot(mdata.index, mdata.values, 'o-', label=algo,
                color=COLORS.get(algo, '#333'), linewidth=2, markersize=6)
    ax.set_xlabel('Difficulty')
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR by Difficulty')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.suptitle('Benchmark v6 Overview', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'fig_overview_v6.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path.name}')


def plot_cd_diagram(stat_result, out_dir, group_name, filename):
    """
    Critical Difference diagram.
    X-axis: Average rank (lower is better = goes left)
    Groups connected by horizontal bar are NOT significantly different.
    """
    if not stat_result.get('significant'):
        print(f'  CD diagram skipped for {group_name}: not significant')
        return

    algos = list(stat_result['avg_ranks'].keys())
    avg_ranks = stat_result['avg_ranks']
    pairwise  = stat_result['pairwise_comparisons']

    if not algos:
        return

    n = len(algos)
    # Critical difference (Nemenyi approximation for reference)
    k = n
    try:
        from scipy.stats import studentized_range
        q_alpha = 3.0
    except Exception:
        q_alpha = 3.0
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * 5))

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), max(6, n * 0.6)))

    y_positions = np.arange(n)
    sorted_algos = sorted(algos, key=lambda a: avg_ranks[a])

    for i, algo in enumerate(sorted_algos):
        ax.scatter(avg_ranks[algo], i, s=120, zorder=5,
                   color=COLORS.get(algo, '#333'), edgecolors='black', linewidths=0.5)
        ax.text(avg_ranks[algo] - 0.05, i, f'  {algo}', va='center', fontsize=9)

    # Draw CD bar
    cd_y = n + 0.5
    ax.plot([sorted_algos[0] and avg_ranks[sorted_algos[0]] or 0,
             sorted_algos[-1] and avg_ranks[sorted_algos[-1]] or 0], [cd_y, cd_y],
            color='black', linewidth=1.5)
    ax.text((avg_ranks[sorted_algos[0]] + avg_ranks[sorted_algos[-1]]) / 2, cd_y + 0.1,
            f'CD = {cd:.2f}', ha='center', fontsize=9)

    # Connect non-significant pairs
    significant_pairs = {(p['target'], p['baseline']): True
                        for p in pairwise if p['significant']}
    significant_pairs.update({(p['baseline'], p['target']): True
                              for p in pairwise if p['significant']})

    for i, algo_a in enumerate(sorted_algos):
        for j, algo_b in enumerate(sorted_algos):
            if i >= j:
                continue
            if (algo_a, algo_b) not in significant_pairs:
                r_a = avg_ranks[algo_a]
                r_b = avg_ranks[algo_b]
                ax.plot([r_a, r_b], [cd_y - 0.15, cd_y - 0.15],
                        color='gray', linewidth=1.5, alpha=0.6)

    ax.set_xlim(left=-0.5)
    ax.set_ylim(-0.5, cd_y + 1.5)
    ax.set_xlabel('Average Rank (lower is better)', fontsize=11)
    ax.set_yticks([])
    ax.set_title(f'Critical Difference Diagram — {group_name}', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    path = out_dir / filename
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path.name}')


def plot_pareto_frontier(df, out_dir):
    """
    Pareto Frontier chart: AUC-PR vs Label Budget for streaming algorithms.
    Each algorithm = one line. Pareto frontier = undominated points.
    """
    if 'label_budget' not in df.columns:
        print('  Pareto chart skipped: no label_budget data')
        return

    streaming = df[df['algorithm'].isin(ALGO_NAMES_STREAM)].copy()
    if streaming.empty:
        return

    budgets = sorted(streaming['label_budget'].unique())
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in ALGO_NAMES_STREAM:
        aucs = []
        for budget in budgets:
            sub = streaming[(streaming['algorithm'] == algo) & (streaming['label_budget'] == budget)]
            aucs.append(sub['AUC_PR'].mean() if not sub.empty else 0.0)

        style  = '-' if algo == 'CA-DIF-EIA (streaming)' else '--'
        lw     = 2.5 if algo == 'CA-DIF-EIA (streaming)' else 1.5
        color  = COLORS.get(algo, '#333')
        ax.plot(budgets, aucs, style, label=algo, color=color, linewidth=lw, markersize=7)

    # Draw Pareto frontier
    all_points = []
    for algo in ALGO_NAMES_STREAM:
        for budget in budgets:
            sub = streaming[(streaming['algorithm'] == algo) & (streaming['label_budget'] == budget)]
            if not sub.empty:
                all_points.append((budget, sub['AUC_PR'].mean(), algo))

    undominated = []
    for p in all_points:
        is_dominated = any(
            other[0] <= p[0] and other[1] >= p[1] and
            (other[0] < p[0] or other[1] > p[1])
            for other in all_points if other[2] != p[2]
        )
        if not is_dominated:
            undominated.append(p)

    undominated.sort()
    if len(undominated) >= 2:
        ax.plot([p[0] for p in undominated], [p[1] for p in undominated],
                'k:', alpha=0.6, label='Pareto frontier', linewidth=2)

    ax.set_xlabel('Label Budget (# labels)', fontsize=12)
    ax.set_ylabel('AUC-PR', fontsize=12)
    ax.set_title('Pareto Frontier: AUC-PR vs Label Budget (Streaming Algorithms)', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xticks(budgets)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / 'fig_pareto_frontier_v6.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path.name}')


def plot_bar_score(df, out_dir):
    """BAR Score comparison across streaming algorithms."""
    if 'label_budget' not in df.columns:
        return

    streaming = df[df['algorithm'].isin(ALGO_NAMES_STREAM)].copy()
    if streaming.empty:
        return

    budgets = sorted(streaming['label_budget'].unique())
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(budgets))
    width = 0.2
    for i, algo in enumerate(ALGO_NAMES_STREAM):
        bars = []
        for budget in budgets:
            sub = streaming[(streaming['algorithm'] == algo) & (streaming['label_budget'] == budget)]
            if not sub.empty:
                auc_pr     = sub['AUC_PR'].mean()
                labels_u   = sub['labels_consumed'].mean()
                bar_score  = compute_bar_score(auc_pr, max(labels_u, 1), max(budget, 1))
            else:
                bar_score = 0.0
            bars.append(bar_score)
        ax.bar(x + i * width, bars, width, label=algo, color=COLORS.get(algo, '#333'), alpha=0.8)

    ax.set_xlabel('Label Budget (# labels)', fontsize=12)
    ax.set_ylabel('BAR Score', fontsize=12)
    ax.set_title('BAR Score: Budget-Aware Ranking (Streaming Algorithms)', fontsize=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(budgets)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = out_dir / 'fig_bar_score_v6.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path.name}')


def plot_ablation(df, out_dir):
    """Ablation study: IF-baseline vs AE+IF vs CA-DIF-EIA."""
    ablation_algos = ['IF-baseline', 'AE+IF', 'CA-DIF-EIA']
    present = [a for a in ablation_algos if a in df['algorithm'].values]
    if len(present) < 2:
        print('  Ablation chart skipped: not all configs present')
        return

    sub = df[df['algorithm'].isin(present)]
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    auc_means = sub.groupby('algorithm')['AUC_PR'].agg(['mean', 'std']).reindex(present)
    colors_a = [COLORS.get(a, '#333') for a in present]
    bars = ax.bar(present, auc_means['mean'].values, yerr=auc_means['std'].values,
                  color=colors_a, alpha=0.8, capsize=5)
    ax.set_ylabel('AUC-PR')
    ax.set_title('Ablation Study: AUC-PR Comparison')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, auc_means['mean'].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1]
    colors_d = {'easy': '#27ae60', 'medium': '#f39c12', 'hard': '#c0392b'}
    x = np.arange(len(present))
    for di, diff in enumerate(['easy', 'medium', 'hard']):
        vals = []
        for a in present:
            sub_d = sub[(sub['algorithm'] == a) & (sub['difficulty'] == diff)]
            vals.append(sub_d['AUC_PR'].mean() if not sub_d.empty else 0.0)
        ax.bar(x + di * 0.25, vals, 0.22, label=diff.capitalize(),
               color=colors_d[diff], alpha=0.8)
    ax.set_xticks(x + 0.25)
    ax.set_xticklabels(present)
    ax.set_ylabel('AUC-PR')
    ax.set_title('Ablation: AUC-PR by Difficulty')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = out_dir / 'fig_ablation_v6.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {path.name}')


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(df, stat_results, out_dir):
    """Generate markdown report with all results."""
    lines = []
    lines.append("# Benchmark v6 Results — Scientific Rigour Overhaul\n")
    lines.append(f"**Generated:** {datetime.now().isoformat()}\n")
    lines.append(f"**Protocol:** Three-way split (train/val/test), threshold from validation set\n")
    lines.append(f"**Datasets:** NYC Yellow Taxi Jan-Jun 2024\n")
    lines.append(f"**Seeds:** {SEEDS}\n")
    lines.append(f"**Difficulties:** {DIFFICULTIES}\n")
    lines.append(f"**Anomaly Rate:** {ANOMALY_RATE:.0%} ({ANOMALY_N} anomalies / {TEST_N} test samples)\n")
    lines.append(f"**Label Budgets:** {LABEL_BUDGETS}\n")

    # Summary table
    lines.append("\n## Summary: Mean AUC-PR by Algorithm\n")
    summary = df.groupby('algorithm')['AUC_PR'].agg(['mean', 'std', 'count'])
    summary = summary.sort_values('mean', ascending=False)
    lines.append("| Algorithm | Mean AUC-PR | Std | N |")
    lines.append("|-----------|-------------|-----|---|")
    for algo, row in summary.iterrows():
        lines.append(f"| {algo} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |")

    # By difficulty
    lines.append("\n## AUC-PR by Difficulty\n")
    pivot = df.pivot_table(index='algorithm', columns='difficulty', values='AUC_PR', aggfunc='mean')
    pivot = pivot.reindex(summary.index)
    lines.append("| Algorithm | Easy | Medium | Hard |")
    lines.append("|-----------|------|--------|------|")
    for algo in pivot.index:
        row = pivot.loc[algo]
        lines.append(f"| {algo} | {row.get('easy', 0):.4f} | {row.get('medium', 0):.4f} | {row.get('hard', 0):.4f} |")

    # BAR Score
    if 'label_budget' in df.columns:
        lines.append("\n## BAR Score by Streaming Algorithm\n")
        for budget in sorted(df['label_budget'].unique()):
            sub = df[(df['algorithm'].isin(ALGO_NAMES_STREAM)) & (df['label_budget'] == budget)]
            if sub.empty:
                continue
            lines.append(f"\n### Label Budget = {budget}\n")
            for algo in ALGO_NAMES_STREAM:
                sub_a = sub[sub['algorithm'] == algo]
                if sub_a.empty:
                    continue
                auc_pr   = sub_a['AUC_PR'].mean()
                labels_u = sub_a['labels_consumed'].mean()
                bar      = compute_bar_score(auc_pr, max(labels_u, 1), max(budget, 1))
                lines.append(f"- {algo}: AUC-PR={auc_pr:.4f}, Labels={labels_u:.0f}, BAR={bar:.4f}")

    # Statistical analysis
    lines.append("\n## Statistical Analysis\n")
    for group_name, sr in stat_results.items():
        lines.append(f"\n### {group_name}\n")
        lines.append(f"- Friedman test: stat={sr.get('friedman_stat', 'N/A')}, p={sr.get('friedman_p', 'N/A')}")
        lines.append(f"- Significant: {sr.get('significant', False)}")
        if sr.get('significant'):
            lines.append(f"\n**Average Ranks (lower = better):**")
            for algo, rank in sorted(sr.get('avg_ranks', {}).items(), key=lambda x: x[1]):
                lines.append(f"  - {algo}: {rank:.2f}")
            lines.append(f"\n**Pairwise Comparisons (Wilcoxon, Holm-corrected):**")
            for pair in sr.get('pairwise_comparisons', []):
                sig = '**SIGNIFICANT**' if pair.get('significant') else 'not significant'
                lines.append(f"  - {pair['target']} vs {pair['baseline']}: p_raw={pair['p_raw']:.4f}, "
                             f"p_holm={pair['p_corrected']:.4f}, {sig}")
            lines.append(f"\n**Bootstrap 95% Confidence Intervals:**")
            for algo, (lo, hi) in sr.get('confidence_intervals', {}).items():
                lines.append(f"  - {algo}: [{lo:.4f}, {hi:.4f}]")
        else:
            lines.append(f"  {sr.get('conclusion', 'No significant differences.')}")

    path = out_dir / 'benchmark_v6_results.md'
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n  Report saved: {path.name}')


# =============================================================================
# MAIN
# =============================================================================

def run():
    print('=' * 70)
    print('BENCHMARK v6 — Scientific Rigour Overhaul')
    print(f'  GPU: {"ON (" + DEVICE + ")" if GPU_AVAILABLE else "OFF"}')
    print(f'  Algorithms (batch):   {ALGO_NAMES_BATCH}')
    print(f'  Algorithms (stream):  {ALGO_NAMES_STREAM}')
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

    # ===== 2. Build evaluation jobs =====
    print('\n[2/7] Building evaluation jobs...')
    jobs = []

    # Fold structure (sliding window):
    # Fold i: train = Jan..month(i), val = last 20% of month(i), test = month(i+1)
    for fold_idx, test_month in enumerate(MONTHS[1:], 1):
        train_months = MONTHS[:fold_idx]           # [1..fold_idx]
        val_month    = train_months[-1]             # last training month

        # Full training data from all accumulated months
        train_X = np.vstack([monthly_X[m - 1] for m in train_months])
        train_df = pd.concat([monthly[m - 1] for m in train_months], ignore_index=True)

        # Validation: last VAL_N from the LAST training month (temporal hold-out)
        last_month_X  = monthly_X[val_month - 1]
        last_month_df = monthly[val_month - 1]
        val_X  = last_month_X[-VAL_N:]
        val_df = last_month_df.iloc[-VAL_N:].reset_index(drop=True)

        # Remove val from train: drop last VAL_N from accumulated train set
        n_train_keep = len(train_X) - VAL_N
        train_X  = train_X[:n_train_keep]
        train_df = train_df.iloc[:n_train_keep].reset_index(drop=True)

        # Test: full test month
        test_df = monthly[test_month - 1]

        # Inject anomalies into test
        for diff in DIFFICULTIES:
            params = ANOMALY_PARAMS[diff]
            seed_s = SEEDS[fold_idx % len(SEEDS)]

            # CRITICAL: sample TEST_N from source BEFORE injection
            # Otherwise 1500 anomalies get diluted over 2.6M records
            rng_src = np.random.RandomState(seed_s)
            if len(test_df) > TEST_N:
                src_idx = rng_src.choice(len(test_df), TEST_N, replace=False)
                test_df_sub = test_df.iloc[src_idx].reset_index(drop=True)
            else:
                test_df_sub = test_df.reset_index(drop=True)

            # Now inject into the TEST_N-sized sample
            test_df_inj, y_labels = inject_anomalies(test_df_sub, params, seed_s)
            X_test = features(test_df_inj).astype(np.float32)
            y_labels = np.array(y_labels, dtype=np.int8)

            # Standardize using training data stats
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(train_X).astype(np.float32)
            X_val_s   = scaler.transform(val_X).astype(np.float32)
            X_test_s  = scaler.transform(X_test).astype(np.float32)

            # Labels for validation (100% normal — no injection in val)
            y_val = np.zeros(len(X_val_s), dtype=np.int8)

            # --- BATCH algorithms ---
            batch_algo_info = [
                ('sklearn_IF',      SklearnIF,         {}),
                ('DenoisingAE',    DenoisingAE,        {}),
                ('CA-DIF-EIA',     CADIFEiaBatch,      {'ablation': 'full'}),
                ('AE+IF',          CADIFEiaBatch,      {'ablation': 'ae_if'}),
                ('IF-baseline',    CADIFEiaBatch,      {'ablation': 'baseline'}),
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

            # --- STREAMING algorithms ---
            stream_algo_info = [
                ('sHST-River',            sHST_River,     {}),
                ('MemStream',             MemStream,      {}),
                ('CA-DIF-EIA (streaming)', CADIFEiaStream, {}),
            ]
            for name, cls, kw in stream_algo_info:
                for seed in SEEDS:
                    for lb in LABEL_BUDGETS:
                        jobs.append({
                            'type':       'streaming',
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
                            'label_budget': lb,
                            'kwargs':     kw,
                        })

    print(f'  Total jobs: {len(jobs)}')
    print(f'  Batch: {sum(1 for j in jobs if j["type"] == "batch")}')
    print(f'  Streaming: {sum(1 for j in jobs if j["type"] == "streaming")}')

    # ===== 3. Run benchmark =====
    print(f'\n[3/7] Running {len(jobs)} jobs...')
    t0 = time.perf_counter()
    results = []
    CHECKPOINT_INTERVAL = 50

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
            pd.DataFrame(results).to_csv(OUT_DIR / 'checkpoint_v6.csv', index=False)
        gc.collect()

    bench_df = pd.DataFrame(results)
    bench_df.to_csv(OUT_DIR / 'benchmark_results_v6.csv', index=False)
    t_done = time.perf_counter() - t0
    print(f'\n  Benchmark done in {t_done/60:.1f} min')

    errors = bench_df['error'].notna() & (bench_df['error'] != '')
    if errors.any():
        print(f'  WARNING: {errors.sum()} jobs had errors:')
        for _, row in bench_df[errors].iterrows():
            print(f'    [{row["algorithm"]}] fold={row["fold"]} {row["error"]}')

    # ===== 4. BAR Score computation =====
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
        bar_df.to_csv(OUT_DIR / 'bar_score_results_v6.csv', index=False)

    bench_df.to_csv(OUT_DIR / 'benchmark_results_v6.csv', index=False)

    # ===== 5. Statistical analysis =====
    print('\n[5/7] Statistical analysis (Friedman + Holm-Bonferroni)...')
    stat_results = {}

    # Batch: all algorithms
    batch_algos = [a for a in ALGO_NAMES_BATCH if a in bench_df['algorithm'].values]
    batch_df = bench_df[bench_df['algorithm'].isin(batch_algos)]
    if not batch_df.empty:
        sr = statistical_analysis(batch_df, 'Batch Algorithms', batch_algos)
        stat_results['Batch'] = sr
        with open(OUT_DIR / 'statistical_results.txt', 'w') as fp:
            report_stat_results(sr, fp)

    # Batch by difficulty
    for diff in DIFFICULTIES:
        sub = bench_df[(bench_df['algorithm'].isin(batch_algos)) & (bench_df['difficulty'] == diff)]
        if sub.empty:
            continue
        sr = statistical_analysis(sub, f'Batch — {diff.capitalize()}', batch_algos)
        stat_results[f'Batch_{diff}'] = sr

    # Streaming: primary (label_budget=500)
    stream_500 = [a for a in ALGO_NAMES_STREAM if a in bench_df['algorithm'].values]
    stream_df  = bench_df[(bench_df['algorithm'].isin(stream_500)) & (bench_df['label_budget'] == 500)]
    if not stream_df.empty:
        sr = statistical_analysis(stream_df, 'Streaming (Budget=500)', stream_500)
        stat_results['Streaming_500'] = sr

    # ===== 6. Plots =====
    print('\n[6/7] Generating plots...')
    plot_overview(bench_df, OUT_DIR)

    for group_name, sr in stat_results.items():
        fn = f'fig_cd_{group_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}_v6.png'
        plot_cd_diagram(sr, OUT_DIR, group_name, fn)

    plot_pareto_frontier(bench_df, OUT_DIR)
    plot_bar_score(bench_df, OUT_DIR)
    plot_ablation(bench_df, OUT_DIR)

    # ===== 7. Report =====
    print('\n[7/7] Generating report...')
    generate_report(bench_df, stat_results, OUT_DIR)

    # ===== Environment info =====
    env = {
        'version':       '6.0',
        'timestamp':     datetime.now().isoformat(),
        'python':        sys.version.split()[0],
        'gpu':           GPU_AVAILABLE,
        'gpu_device':    torch.cuda.get_device_name(0) if GPU_AVAILABLE else None,
        'scipy':         '1.17.1',
        'sklearn':       '1.8.0',
        'torch':         '2.5.1',
        'river':         '0.24.2' if HAS_RIVER else None,
        'total_jobs':    len(jobs),
        'runtime_min':   t_done / 60,
        'protocol':      'three-way split (train/val/test)',
        'pre_registration': '2026-05-12',
    }
    with open(OUT_DIR / 'environment.json', 'w') as f:
        json.dump(env, f, indent=2)

    # ===== Summary =====
    print('\n' + '=' * 70)
    print('BENCHMARK v6 COMPLETE')
    print(f'  {len(jobs)} jobs in {t_done/60:.1f} min')
    print(f'  Results: {OUT_DIR}')
    print('=' * 70)

    # Quick summary
    print('\nQuick Summary (Mean AUC-PR, Batch):')
    for a in ALGO_NAMES_BATCH:
        sub = bench_df[bench_df['algorithm'] == a]
        if not sub.empty:
            print(f'  {a:25s}: {sub["AUC_PR"].mean():.4f} ± {sub["AUC_PR"].std():.4f}')

    print('\nQuick Summary (Mean AUC-PR, Streaming, Budget=500):')
    for a in ALGO_NAMES_STREAM:
        sub = bench_df[(bench_df['algorithm'] == a) & (bench_df['label_budget'] == 500)]
        if not sub.empty:
            print(f'  {a:25s}: {sub["AUC_PR"].mean():.4f} ± {sub["AUC_PR"].std():.4f}')

    return bench_df


if __name__ == '__main__':
    run()
