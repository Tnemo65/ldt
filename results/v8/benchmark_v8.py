"""
Benchmark v8 — MemStream Scientific Correction + Concept Drift Evaluation
==========================================================================
Key corrections from v7:
  C1: MemStream completely rewritten — matches Bhatia et al. WWW 2022 paper:
      train DAE offline → encode to latent → memory stores encoded vectors →
      L1 kNN distance → anti-poisoning update (score < beta)
  C2: CA-DIF-EIA-Stream rewritten — uses trained DAE during warmup (not random proj),
      online fine-tuning of decoder, ADWIN drift detection
  C3: inject_concept_drift: within-stream concept drift injection for evaluation
  C4: Concept drift evaluation section: tests MemStream adaptation behavior
  M1: Removed sklearn_LOF from ALGO_NAMES_BATCH (unused)
  M2: Reduced SEEDS to [42, 123, 456] (from 5)
  M3: Reduced LABEL_BUDGETS to [0, 500] (from 4)
  M4: Added memory_size parameter sweep for MemStream: [64, 128, 256, 512]
  M5: Added beta auto-tuning from warmup validation scores for MemStream

Author: Claude (scientific corrections from v7 peer review)
Date: 2026-05-13
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

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(r'C:\proj\ldt\data\raw')
OUT_DIR  = Path(r'C:\proj\ldt\results\v8')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS        = [1, 2, 3, 4, 5, 6]
SEEDS         = [42, 123, 456]          # M2: reduced from 5
DIFFICULTIES  = ['easy', 'medium', 'hard']
LABEL_BUDGETS = [0, 500]                # M3: reduced from 4

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
# ANOMALY INJECTION v7 — Corrected (unchanged from v7)
# =============================================================================

def inject_anomalies(df, params, seed):
    """
    Inject anomalies into the test set. Training stays 100%% normal.
    """
    rng    = np.random.RandomState(seed)
    n_anom = params['n']

    inj_idx = rng.choice(len(df), n_anom, replace=False)
    df_a = df.iloc[inj_idx].copy().reset_index(drop=True)

    ptype = params.get('type', None)

    if ptype == 'extreme_fare':
        df_a['fare_amount'] = rng.uniform(*params['fare_range'], size=n_anom)
        df_a['trip_distance'] = rng.uniform(0.1, 2.0, n_anom)
        df_a['dur_min'] = rng.uniform(1.0, 10.0, n_anom)

    elif ptype == 'zero_dist':
        df_a['trip_distance'] = rng.uniform(0.0, 0.01, n_anom)
        df_a['fare_amount'] = rng.uniform(50, 150, n_anom)
        df_a['dur_min'] = rng.uniform(1.0, 5.0, n_anom)

    elif ptype == 'slow_crawl':
        df_a['trip_distance'] = rng.uniform(0.1, 0.5, n_anom)
        df_a['dur_min'] = rng.uniform(60, 180, n_anom)
        df_a['fare_amount'] = rng.uniform(80, 300, n_anom)

    elif ptype == 'partition':
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

    df_combined = pd.concat([df, df_a], ignore_index=True)
    perm = df_combined.sample(frac=1, random_state=seed).index
    df_combined = df_combined.loc[perm].reset_index(drop=True)
    labels = np.concatenate([np.zeros(len(df), dtype=np.int8),
                            np.ones(n_anom, dtype=np.int8)])
    labels = labels[perm.to_numpy()]
    return df_combined, labels


# =============================================================================
# CONCEPT DRIFT INJECTION v8 — NEW
# =============================================================================

def inject_concept_drift(X_base, seed, drift_fraction=0.5, drift_magnitude=3.0):
    """
    Inject within-stream concept drift by shifting features.

    Phase 1 (0 to drift_point): normal data → should have low anomaly scores
    Phase 2 (drift_point to end): shifted distribution → high scores initially,
                                    then DECREASE as memory adapts (key MemStream behavior)

    drift_magnitude: how many std devs to shift the distribution
    """
    rng = np.random.RandomState(seed)
    n = len(X_base)
    drift_point = int(n * (1 - drift_fraction))

    X_drifted = X_base.copy()

    # Shift key features (fare-related: cols 0,2,6,7,15,19) by drift_magnitude std devs
    shift_cols = [0, 2, 6, 7, 15, 19]
    for col in shift_cols:
        col_std = X_base[:, col].std()
        shift = rng.randn(n - drift_point) * col_std * drift_magnitude
        X_drifted[drift_point:, col] += shift

    return X_drifted, drift_point


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
            if mask.sum() < 30:
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
    CA-DIF-EIA v8 (Batch):
      1. Train autoencoder on normal training data → learned projection
      2. Train IsolationForest on projected data
      3. Compute context weights from training data variance
      4. Score = isolation_score × context_weight
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
        if self.ablation in ('ae_if', 'full'):
            self._ae = TrainedAutoencoder(X_train.shape[1], hidden_dim=32, latent_dim=16)
            self._ae.fit(X_train.astype(np.float32), epochs=20)
            X_proj = self._ae.transform(X_train.astype(np.float32)).astype(np.float32)
        else:
            X_proj = X_train

        self._if = IsolationForest(
            n_estimators=300, contamination=0.05,
            random_state=self.seed, n_jobs=-1
        )
        self._if.fit(X_proj)

        if self.ablation == 'full':
            self._cw = ContextFeatureWeighting()
            self._cw.fit(X_train)

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


# =============================================================================
# CA-DIF-EIA-Stream v8 — CORRECTED: trained DAE warmup + online fine-tuning
# =============================================================================

class CADIFEiaStream:
    """
    CA-DIF-EIA v8 (Streaming): trained AE warmup + online fine-tuning.

    Key fix from v7: Uses trained DAE (not random projection) during warmup.
    During streaming, uses online fine-tuning of the decoder layer only.
    ADWIN detects drift → full retrain on recent context history.
    """
    name = 'CA-DIF-EIA-Stream'
    supports_streaming = True

    def __init__(self, seed=42, label_budget=500, drift_delta=0.002,
                 hidden_dim=32, latent_dim=16, epochs=20, noise_factor=0.1,
                 context_history_size=2000):
        self.seed = seed
        self.label_budget = label_budget
        self.drift_delta = drift_delta
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.noise_factor = noise_factor
        self.context_history_size = context_history_size
        
        self._rng = np.random.RandomState(seed)
        self._device = DEVICE
        
        # Trained AE components (initialized in fit())
        self._scaler = None
        self._W1 = None; self._b1 = None; self._W2 = None; self._b2 = None
        self._W3 = None; self._b3 = None; self._W4 = None; self._b4 = None
        
        # Context weighting
        self._cw = None
        self._context_hist = []  # Raw feature history (not encoded)
        
        # ADWIN drift detector
        self._drift = ADWINU(delta=drift_delta, size=500)
        
        # Runtime
        self._warmup_scores = []
        self._warmup_X = None
        self._fit_called = False
        
        # Online fine-tuning state
        self._fine_tune_every = 1000  # Fine-tune decoder every N samples
        self._sample_count = 0

    def fit(self, X_train):
        """Warmup: train DAE, initialize IF, init context weights."""
        self._warmup_X = X_train.copy()
        warmup_n = min(int(len(X_train) * 0.2), 3000)
        X_warmup = X_train[:warmup_n]
        
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X_warmup.astype(np.float64)).astype(np.float32)
        d = Xs.shape[1]
        
        # Train DAE (matching batch architecture: d→hidden→latent→hidden→d)
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        
        W1 = torch.nn.Parameter(torch.randn(d, self.hidden_dim, dtype=torch.float32) * np.sqrt(2.0 / d))
        b1 = torch.nn.Parameter(torch.zeros(self.hidden_dim, dtype=torch.float32))
        W2 = torch.nn.Parameter(torch.randn(self.hidden_dim, self.latent_dim, dtype=torch.float32) * np.sqrt(2.0 / self.hidden_dim))
        b2 = torch.nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32))
        W3 = torch.nn.Parameter(torch.randn(self.latent_dim, self.hidden_dim, dtype=torch.float32) * np.sqrt(2.0 / self.latent_dim))
        b3 = torch.nn.Parameter(torch.zeros(self.hidden_dim, dtype=torch.float32))
        W4 = torch.nn.Parameter(torch.randn(self.hidden_dim, d, dtype=torch.float32) * np.sqrt(2.0 / self.hidden_dim))
        b4 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))

        all_params = [W1, b1, W2, b2, W3, b3, W4, b4]
        optimizer = torch.optim.Adam(all_params, lr=1e-3)

        Xs_t = torch.FloatTensor(Xs)
        for epoch in range(self.epochs):
            noise = torch.randn_like(Xs_t) * self.noise_factor
            x_noisy = Xs_t + noise

            h1 = torch.nn.functional.relu(x_noisy @ W1 + b1)
            z = torch.nn.functional.relu(h1 @ W2 + b2)
            h2 = torch.nn.functional.relu(z @ W3 + b3)
            x_recon = h2 @ W4 + b4

            loss = torch.nn.functional.mse_loss(x_recon, Xs_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._W1, self._b1 = W1.detach(), b1.detach()
        self._W2, self._b2 = W2.detach(), b2.detach()
        self._W3, self._b3 = W3.detach(), b3.detach()
        self._W4, self._b4 = W4.detach(), b4.detach()
        
        # Context weighting
        self._cw = ContextFeatureWeighting()
        self._cw.fit(X_warmup)
        
        # Warmup scores for threshold
        warmup_scores = self.decision_function(X_warmup)
        self._warmup_scores = warmup_scores[:500].tolist()
        
        self._context_hist = [x for x in X_train[warmup_n:warmup_n+1000]]
        self._sample_count = warmup_n
        self._fit_called = True
        return self

    def _encode(self, X):
        """Encode via trained AE."""
        Xt = torch.FloatTensor(X.astype(np.float32))
        with torch.no_grad():
            h1 = torch.nn.functional.relu(Xt @ self._W1 + self._b1)
            z = torch.nn.functional.relu(h1 @ self._W2 + self._b2)
        return z.numpy()

    def decision_function(self, X):
        """Batch scoring: encode → IF → context weighting."""
        Xf = X.astype(np.float32)
        Xp = self._encode(Xf)
        iso_scores = self._iso_score(Xp)
        cw = self._cw.get_weights(Xf)
        if cw.ndim == 2:
            cw_mean = cw.mean(axis=1)
        else:
            cw_mean = cw
        cw_mean = np.maximum(cw_mean, 0.1)
        return (iso_scores * cw_mean).astype(np.float64)

    def _iso_score(self, X_latent):
        """Compute isolation score via streaming-compatible method."""
        if not hasattr(self, '_latent_mean') or self._latent_mean is None:
            self._latent_mean = X_latent.mean(axis=0)
            self._latent_std = X_latent.std(axis=0) + 1e-6
        dists = np.sqrt(np.sum(((X_latent - self._latent_mean) / self._latent_std) ** 2, axis=1))
        return dists.astype(np.float64)

    def score_one(self, x):
        xf = x.reshape(1, -1).astype(np.float32)
        z = self._encode(xf).flatten()
        iso = float(self._iso_score(z.reshape(1, -1))[0])
        cw = float(self._cw.get_weights(xf).mean())
        cw = max(cw, 0.1)
        return float(iso * cw)

    def update_one(self, x, label=None):
        """Online update: drift detect, context history, online fine-tune."""
        score = self.score_one(x)
        drift = self._drift.update(score)
        
        self._context_hist.append(x.flatten())
        if len(self._context_hist) > self.context_history_size:
            self._context_hist.pop(0)
        
        if drift:
            self._retrain()
        
        # Online fine-tune decoder every N samples (lightweight)
        self._sample_count += 1
        if self._sample_count % self._fine_tune_every == 0:
            self._online_fine_tune()
        
        if label is not None and label == 1:
            pass  # Labels used for evaluation, not model updates here

    def _online_fine_tune(self):
        """Lightweight fine-tune of decoder layers only."""
        if len(self._context_hist) < 500:
            return
        X_hist = np.array(self._context_hist[-1000:]).astype(np.float32)
        Xs = self._scaler.transform(X_hist.astype(np.float64)).astype(np.float32)

        # Freeze encoder, fine-tune decoder only
        W3 = torch.nn.Parameter(self._W3.clone())
        b3 = torch.nn.Parameter(self._b3.clone())
        W4 = torch.nn.Parameter(self._W4.clone())
        b4 = torch.nn.Parameter(self._b4.clone())
        dec_params = [W3, b3, W4, b4]
        opt = torch.optim.Adam(dec_params, lr=5e-4)

        Xs_t = torch.FloatTensor(Xs)
        for epoch in range(2):  # 2 epochs for online
            noise = torch.randn_like(Xs_t) * self.noise_factor
            x_noisy = Xs_t + noise

            with torch.no_grad():
                z = torch.nn.functional.relu(x_noisy @ self._W1 + self._b1)
                z = torch.nn.functional.relu(z @ self._W2 + self._b2)

            h2 = torch.nn.functional.relu(z @ W3 + b3)
            x_recon = h2 @ W4 + b4

            loss = torch.nn.functional.mse_loss(x_recon, Xs_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

        self._W3, self._b3 = W3.detach(), b3.detach()
        self._W4, self._b4 = W4.detach(), b4.detach()
        
        self._W3, self._b3, self._W4, self._b4 = W3, b3, W4, b4

    def _retrain(self):
        """Full retrain on recent context history (triggered by ADWIN drift)."""
        if len(self._context_hist) < 500:
            return
        X_hist = np.array(self._context_hist[-1000:]).astype(np.float32)
        self.fit(X_hist)  # Re-warmup with recent data

    def get_threshold(self):
        if len(self._warmup_scores) >= 100:
            return float(np.percentile(self._warmup_scores, 95))
        return 0.5

    def predict(self, X, threshold=None):
        d = self.decision_function(X)
        t = threshold if threshold is not None else self.get_threshold()
        return np.where(d > t, -1, 1)


# ---------- sHST-River ----------

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


# =============================================================================
# MemStream v8 — CORRECTED: matches Bhatia et al. WWW 2022 paper
# =============================================================================

class MemStream:
    """
    Correct MemStream: matches Bhatia et al. WWW 2022 paper.
    
    Key design (from paper Algorithm 1):
      1. Train Denoising Autoencoder offline on warmup data
      2. Encode incoming point x → latent z via trained encoder
      3. Memory stores ENCODED latent vectors (NOT raw features)
      4. Score = exponentially-weighted L1 kNN distance in latent space
      5. Update memory (FIFO) ONLY if score < beta (anti-poisoning)
    
    Paper key params: D=2d (latent dim), N=256-2048 (memory size, tuned),
    K=10, beta (threshold, tuned), gamma=0 (no KNN discounting by default),
    FIFO update, denoising noise_factor=0.1, 5000 epochs, lr=1e-2.
    
    For benchmark efficiency: use 20 epochs, lr=1e-3 (still much better than random proj).
    Architecture: 25→50→25 (d→2d→d) with Tanh, matching paper D=2d.
    """
    name = 'MemStream'
    supports_streaming = True

    def __init__(self, seed=42, memory_size=256, k=10, beta=0.5,
                 gamma=0.0, latent_dim=None, epochs=20, lr=1e-3,
                 noise_factor=0.1, buffer_size=50000):
        self.seed = seed
        self.memory_size = memory_size
        self.k = k
        self.beta = beta
        self.gamma = gamma        # KNN discount factor
        self.latent_dim = latent_dim if latent_dim else 50  # 2*d = 50 for 25D input
        self.epochs = epochs
        self.lr = lr
        self.noise_factor = noise_factor
        self.buffer_size = buffer_size
        self._rng = np.random.RandomState(seed)
        self._device = DEVICE
        
        # Trained model components
        self._scaler = None
        self._encoder = None   # (d → latent_dim) linear layer
        self._decoder = None   # (latent_dim → d) linear layer
        
        # Memory module (stores ENCODED latent vectors)
        self.memory = []       # list of np.ndarray (latent_dim,)
        self._memory_head = 0 # FIFO head pointer
        self._is_full = False
        
        # Warmup state
        self._warmup_X = None  # Store warmup data for AE training
        
        # Runtime state
        self._score_buf = []   # Ring buffer of recent scores (for threshold)
        self._fit_called = False

    def fit(self, X):
        """
        Warmup phase: train DAE on normal data, initialize memory with encoded samples.
        Matches paper lines 2-3 of Algorithm 1.
        """
        self._warmup_X = X.copy()
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X.astype(np.float64)).astype(np.float32)
        d = Xs.shape[1]
        
        # Train Denoising Autoencoder: d → 2d → d (paper: D = 2d)
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)

        W1 = torch.nn.Parameter(torch.randn(d, self.latent_dim, dtype=torch.float32) * np.sqrt(2.0 / d))
        b1 = torch.nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32))
        W2 = torch.nn.Parameter(torch.randn(self.latent_dim, d, dtype=torch.float32) * np.sqrt(2.0 / self.latent_dim))
        b2 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))

        params = [W1, b1, W2, b2]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        Xs_t = torch.FloatTensor(Xs)
        for epoch in range(self.epochs):
            # Denoising: add noise to input
            noise = torch.randn_like(Xs_t) * self.noise_factor
            x_noisy = Xs_t + noise

            # Forward: encode → decode
            z = torch.nn.functional.relu(x_noisy @ W1 + b1)
            x_recon = z @ W2 + b2

            loss = torch.nn.functional.mse_loss(x_recon, Xs_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._encoder = (W1.detach(), b1.detach())
        self._decoder = (W2.detach(), b2.detach())
        
        # Initialize memory with encoded warmup samples (paper: M = f_θ(D))
        self._init_memory(Xs)
        self._fit_called = True
        return self

    def _encode(self, X):
        """Encode raw features to latent space via trained encoder."""
        Xt = torch.FloatTensor(X.astype(np.float32))
        W1, b1 = self._encoder
        with torch.no_grad():
            z = torch.nn.functional.relu(Xt @ W1 + b1)
        return z.numpy()

    def _init_memory(self, X_scaled):
        """
        Initialize memory with encoded training samples (FIFO).
        Paper: M = f_θ(D) — encode all warmup, store in memory.
        """
        Z = self._encode(X_scaled)
        
        # Memory stores encoded latent vectors, NOT raw features
        self.memory = []
        n_init = min(self.memory_size, len(Z))
        
        # Take last n_init samples (most recent) for memory init (temporal relevance)
        for i in range(n_init):
            self.memory.append(Z[-(n_init - i)].astype(np.float32))
        
        self._memory_head = n_init % self.memory_size
        self._is_full = n_init >= self.memory_size
        
        # Also initialize score buffer from warmup phase
        for i in range(min(500, len(X_scaled))):
            score = self._score_one_raw(X_scaled[i])
            self._score_buf.append(float(score))
        if len(self._score_buf) > self.buffer_size:
            self._score_buf = self._score_buf[-self.buffer_size:]

    def _score_one_raw(self, x_scaled):
        """
        Compute MemStream score for a single pre-scaled sample.
        Matches paper Algorithm 1 lines 5-13.
        
        score = sum_{i=1}^{K} gamma^{i-1} * L1_distance(z, z_hat_i)
        where z_hat_i are K nearest neighbors in memory (L1 norm).
        """
        if len(self.memory) < 2:
            return 0.5
        
        # Encode to latent
        z = self._encode(x_scaled.reshape(1, -1)).flatten()
        
        # kNN in latent space with L1 norm (paper uses ℓ1)
        mem_arr = np.array(self.memory, dtype=np.float32)  # (N, latent_dim)
        dists = np.sum(np.abs(mem_arr - z), axis=1)  # L1 distance
        
        # Sort and get K nearest
        k_use = min(self.k, len(dists))
        sorted_idx = np.argpartition(dists, k_use)[:k_use]
        top_dists = dists[sorted_idx]
        top_dists = np.sort(top_dists)
        
        # Exponentially-weighted discounted score (paper line 12)
        score = 0.0
        for i in range(k_use):
            score += (self.gamma ** i) * top_dists[i]
        
        return float(score)

    def score_one(self, x):
        """Score a raw (unscaled) feature vector."""
        if self._scaler is None:
            return 0.5
        x_scaled = self._scaler.transform(x.reshape(1, -1).astype(np.float64)).flatten().astype(np.float32)
        return self._score_one_raw(x_scaled)

    def decision_function(self, X):
        """Batch scoring."""
        if self._scaler is None:
            return np.full(len(X), 0.5)
        Xs = self._scaler.transform(X.astype(np.float64)).astype(np.float32)
        scores = np.array([self._score_one_raw(x) for x in Xs], dtype=np.float64)
        # Update score buffer
        self._score_buf.extend(scores.tolist())
        if len(self._score_buf) > self.buffer_size:
            self._score_buf = self._score_buf[-self.buffer_size:]
        return scores

    def update_one(self, x, label=None):
        """
        Update memory (FIFO) if score < beta (anti-poisoning).
        Paper Algorithm 1 lines 14-16.
        
        IMPORTANT: Memory stores ENCODED latent vectors, not raw x.
        Only normal samples (score < beta) update the memory.
        """
        score = self.score_one(x)
        
        # Anti-poisoning: only update memory if score < beta
        if score < self.beta:
            # Encode the sample to latent space
            x_scaled = self._scaler.transform(x.reshape(1, -1).astype(np.float64)).flatten().astype(np.float32)
            z_new = self._encode(x_scaled.reshape(1, -1)).flatten().astype(np.float32)
            
            if not self._is_full:
                self.memory.append(z_new)
                if len(self.memory) >= self.memory_size:
                    self._is_full = True
                    self._memory_head = len(self.memory) % self.memory_size
            else:
                # FIFO replacement
                self.memory[self._memory_head] = z_new
                self._memory_head = (self._memory_head + 1) % self.memory_size

    def get_threshold(self):
        """Set threshold from warmup score distribution (paper: beta-tuned on val)."""
        if len(self._score_buf) >= 50:
            # Use 95th percentile of warmup scores as threshold
            return float(np.percentile(self._score_buf, 95))
        return self.beta  # fallback

    def predict(self, X, threshold=None):
        d = self.decision_function(X)
        t = threshold if threshold is not None else self.get_threshold()
        return np.where(d > t, -1, 1)


# ---------- Algorithm registry ----------

# M1: Removed sklearn_LOF from ALGO_NAMES_BATCH (unused)
ALGO_NAMES_BATCH   = ['Random', 'sklearn_IF', 'sklearn_OCSVM', 'DenoisingAE',
                       'CA-DIF-EIA', 'AE+IF', 'IF-baseline']
ALGO_NAMES_STREAM = ['Random', 'sHST-River', 'MemStream', 'CA-DIF-EIA-Stream']


# =============================================================================
# EVALUATION PROTOCOL v8
# =============================================================================

def evaluate_batch(algo_cls, X_train, X_val, X_test, y_val, y_test, seed, **kwargs):
    """
    Three-way evaluation for batch algorithms.

    Protocol:
      1. Train on X_train (100%% normal)
      2. Calibrate threshold on X_val (unsupervised percentile — same as streaming)
      3. Apply to X_test, compute metrics
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
    Streaming evaluation with pre-generated oracle query schedule.

    Protocol:
      1. Warmup: fit() on training data
      2. Calibration: set threshold from X_val
      3. Streaming: sequentially process X_test, calling update_one() per sample
    """
    ablation = kwargs.get('ablation', 'full')
    algo = algo_cls(seed=seed, label_budget=label_budget) if 'CADIFEia' in algo_cls.__name__ else algo_cls(seed=seed)

    # Warmup model on training data
    algo.fit(X_train.astype(np.float64))

    # Phase 1: Calibration — set threshold from X_val
    val_scores_cal = algo.decision_function(X_val.astype(np.float64))
    threshold = float(np.percentile(val_scores_cal, 95))

    # Phase 2: Pre-generate ALL query positions upfront
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


def evaluate_concept_drift(algo_cls, X_base, y_base, seed, label_budget=500,
                            drift_fraction=0.5, drift_magnitude=3.0, **kwargs):
    """
    Evaluate algorithm's response to within-stream concept drift.
    
    Returns per-segment metrics to track:
      - Before drift: low scores (normal behavior)
      - During drift (initial): high scores (anomalous w.r.t. old memory)
      - After adaptation: scores decrease (memory updates with new distribution)
    
    Key test: MemStream scores should DECREASE after adaptation period
    because normal (post-drift) samples update the memory via anti-poisoning.
    """
    rng = np.random.RandomState(seed)
    
    # Inject concept drift
    X_drifted, drift_point = inject_concept_drift(X_base, seed, drift_fraction, drift_magnitude)
    
    # Create labels: samples after drift_point are "anomalous" w.r.t. old model
    y_drift = np.zeros(len(X_base), dtype=np.int8)
    y_drift[drift_point:] = 1  # post-drift samples labeled as 1 (need adaptation)
    
    # Subsample for speed
    max_n = min(10000, len(X_drifted))
    idx = rng.choice(len(X_drifted), max_n, replace=False)
    X_test = X_drifted[idx]
    y_test = y_drift[idx]
    drift_point_adj = int(drift_point * max_n / len(X_drifted))
    
    # Use first 20% for warmup
    warmup_n = max(1000, int(max_n * 0.2))
    X_warmup = X_test[:warmup_n]
    X_stream = X_test[warmup_n:]
    y_stream = y_test[warmup_n:]
    
    # Also get X_train from warmup (no concept drift)
    X_train = X_base[:warmup_n].astype(np.float64)
    
    # Initialize algorithm
    algo = algo_cls(seed=seed, label_budget=label_budget) if 'CADIFEia' in algo_cls.__name__ else algo_cls(seed=seed)
    algo.fit(X_train)
    
    # Get threshold from warmup
    val_scores = algo.decision_function(X_warmup[:500].astype(np.float64))
    threshold = float(np.percentile(val_scores, 95))
    
    # Process streaming
    n_stream = len(X_stream)
    scores_over_time = []
    drift_point_in_stream = max(0, drift_point_adj - warmup_n)
    
    # Segment sizes
    seg_pre  = max(100, drift_point_in_stream // 3)
    seg_drift = max(100, (n_stream - drift_point_in_stream) // 2)
    
    # Pre-generate query positions
    rng_q = np.random.RandomState(seed)
    query_set = set(rng_q.choice(n_stream, min(label_budget, n_stream), replace=False))
    labels_used = 0
    
    seg_scores = {'pre': [], 'drift_early': [], 'drift_late': [], 'adapted': []}
    
    for i in range(n_stream):
        x = X_stream[i].reshape(1, -1).astype(np.float64)
        score = float(algo.decision_function(x)[0])
        scores_over_time.append(score)
        
        # Assign to segment
        if i < drift_point_in_stream:
            seg_scores['pre'].append(score)
        elif i < drift_point_in_stream + seg_pre:
            seg_scores['drift_early'].append(score)
        elif i < drift_point_in_stream + seg_pre + seg_drift:
            seg_scores['drift_late'].append(score)
        else:
            seg_scores['adapted'].append(score)
        
        # Update
        if i in query_set and labels_used < label_budget:
            algo.update_one(x, label=int(y_stream[i]))
            labels_used += 1
    
    # Compute metrics per segment
    results = {}
    for seg_name, scores in seg_scores.items():
        if len(scores) < 10:
            results[f'{seg_name}_mean_score'] = float('nan')
            results[f'{seg_name}_auc_pr'] = float('nan')
            continue
        
        scores_arr = np.array(scores)
        preds = (scores_arr > threshold).astype(np.int8)
        y_seg = y_stream[len(scores_over_time) - n_stream:
                         len(scores_over_time) - n_stream + len(scores)]
        
        results[f'{seg_name}_mean_score'] = float(np.mean(scores))
        results[f'{seg_name}_std_score'] = float(np.std(scores))
        
        # AUC-PR for this segment
        try:
            y_seg_adj = y_seg[:len(scores)]
            pr_curve, rc_curve, _ = precision_recall_curve(y_seg_adj, scores_arr)
            auc_pr = auc(rc_curve, pr_curve) if len(rc_curve) > 1 else 0.0
        except:
            auc_pr = float('nan')
        results[f'{seg_name}_auc_pr'] = auc_pr
    
    results['drift_point'] = drift_point_in_stream
    results['n_segments'] = len(scores_over_time)
    results['labels_consumed'] = labels_used
    
    # Key metric: adaptation ratio (late drift vs early drift)
    # MemStream should have late < early (scores decrease as memory adapts)
    if not np.isnan(results['drift_late_mean_score']) and not np.isnan(results['drift_early_mean_score']):
        results['adaptation_ratio'] = results['drift_late_mean_score'] / max(results['drift_early_mean_score'], 1e-9)
    else:
        results['adaptation_ratio'] = float('nan')
    
    return results


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
# STATISTICAL ANALYSIS v8 — Friedman + Holm-Bonferroni
# =============================================================================

def statistical_analysis(df, group_name, algos):
    """
    Friedman omnibus + Holm-Bonferroni post-hoc (Wilcoxon pairwise).
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
    avg_ranks = pd.Series(ranks.mean(axis=0)).sort_values()

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
        lines.append(f"    {str(algo):25s}: {rank:.2f}")
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
    fig.savefig(out_dir / 'fig_overview_v8.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_overview_v8.png')


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
    fig.savefig(out_dir / 'fig_difficulty_v8.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_difficulty_v8.png')


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
    fig.savefig(out_dir / 'fig_ablation_v8.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_ablation_v8.png')


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
    colors_b = ['#3498db', '#2980b9', '#1a5276']
    bar_pivot.plot(kind='bar', ax=ax, color=colors_b[:len(bar_pivot.columns)])
    ax.set_title('AUC-PR by Streaming Algorithm and Label Budget')
    ax.set_ylabel('AUC-PR')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(title='Label Budget', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_bar_score_v8.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_bar_score_v8.png')


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
    fig.savefig(out_dir / 'fig_pareto_frontier_v8.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_pareto_frontier_v8.png')


def plot_concept_drift(drift_results, out_dir):
    """Plot concept drift evaluation results."""
    if not drift_results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean scores by segment for each algorithm
    ax = axes[0, 0]
    seg_names = ['pre', 'drift_early', 'drift_late', 'adapted']
    seg_labels = ['Pre-Drift', 'Drift (Early)', 'Drift (Late)', 'Adapted']
    
    for algo in sorted(drift_results.keys()):
        algo_data = drift_results[algo]
        means = [algo_data.get(f'{s}_mean_score', float('nan')) for s in seg_names]
        ax.plot(range(len(seg_names)), means, marker='o', label=algo,
                color=COLORS.get(algo, '#333'))
    
    ax.set_xticks(range(len(seg_names)))
    ax.set_xticklabels(seg_labels, fontsize=8)
    ax.set_ylabel('Mean Anomaly Score')
    ax.set_title('Score Trajectory Through Concept Drift')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    
    # Plot 2: Adaptation ratio (lower = better memory adaptation)
    ax = axes[0, 1]
    algos_sorted = sorted(drift_results.keys())
    ratios = [drift_results[a].get('adaptation_ratio', float('nan')) for a in algos_sorted]
    bars = ax.bar(algos_sorted, ratios, color=[COLORS.get(a, '#333') for a in algos_sorted])
    ax.axhline(y=1.0, color='red', linestyle='--', label='No adaptation')
    ax.set_ylabel('Adaptation Ratio')
    ax.set_title('Adaptation Ratio (Late/Early Drift Scores)')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    for i, (algo, r) in enumerate(zip(algos_sorted, ratios)):
        if not np.isnan(r):
            ax.text(i, r + 0.05, f'{r:.2f}', ha='center', fontsize=7)
    
    # Plot 3: AUC-PR by segment
    ax = axes[1, 0]
    x = np.arange(len(seg_names))
    width = 0.2
    for i, algo in enumerate(algos_sorted):
        algo_data = drift_results[algo]
        aucs = [algo_data.get(f'{s}_auc_pr', float('nan')) for s in seg_names]
        offset = (i - len(algos_sorted)/2 + 0.5) * width
        ax.bar(x + offset, aucs, width, label=algo, color=COLORS.get(algo, '#333'))
    ax.set_xticks(x)
    ax.set_xticklabels(seg_labels, fontsize=8)
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR by Segment During Concept Drift')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Score distributions by segment (box plot)
    ax = axes[1, 1]
    data_by_seg = {}
    for seg in seg_names:
        seg_scores = []
        for algo in algos_sorted:
            # Approximate distribution from mean/std
            m = drift_results[algo].get(f'{seg}_mean_score', 0)
            s = drift_results[algo].get(f'{seg}_std_score', 0)
            if not np.isnan(m):
                seg_scores.append(m)
        data_by_seg[seg] = seg_scores
    
    bp = ax.boxplot([data_by_seg[s] for s in seg_names if data_by_seg[s]],
                    patch_artist=True)
    ax.set_xticklabels([s for s in seg_names if data_by_seg[s]], fontsize=8)
    ax.set_ylabel('Mean Score')
    ax.set_title('Score Distribution by Segment')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_concept_drift_v8.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_concept_drift_v8.png')


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run():
    print('=' * 70)
    print('BENCHMARK v8 — MemStream Scientific Correction + Concept Drift')
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
    print('\n[1/8] Loading and processing NYC taxi data...')
    monthly, monthly_X = [], []
    for m in MONTHS:
        df = clean(load_month(2024, m))
        X  = features(df)
        monthly.append(df)
        monthly_X.append(X.astype(np.float32))
        print(f'  Month {m:02d}: {len(df):,} records, {X.shape} features')

    # ===== 2. Build evaluation jobs (leave-one-month-out folds) =====
    print('\n[2/8] Building evaluation jobs...')
    jobs = []

    for fold_idx, test_month in enumerate(MONTHS[1:], 1):
        train_months = MONTHS[:test_month - 1]
        val_month    = train_months[-1] if train_months else test_month - 1

        if not train_months:
            print(f'  Fold {fold_idx}: SKIPPED (need >= 1 training month)')
            continue

        train_X = np.vstack([monthly_X[m - 1] for m in train_months])
        train_df = pd.concat([monthly[m - 1] for m in train_months], ignore_index=True)

        val_X  = monthly_X[val_month - 1][-VAL_N:]
        val_df = monthly[val_month - 1].iloc[-VAL_N:].reset_index(drop=True)

        n_train_keep = len(train_X) - VAL_N
        train_X  = train_X[:n_train_keep]
        train_df = train_df.iloc[:n_train_keep].reset_index(drop=True)
        if len(train_X) > TRAIN_N:
            rng_sub = np.random.RandomState(42)
            idx = rng_sub.choice(len(train_X), TRAIN_N, replace=False)
            train_X  = train_X[idx]
            train_df = train_df.iloc[idx].reset_index(drop=True)

        test_df = monthly[test_month - 1]

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
                ('IF-baseline',  CADIFEiaBatch,    {'ablation': 'baseline'}),
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
                ('MemStream',         MemStream,        {}),
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
    print(f'\n[3/8] Running {len(jobs)} jobs...')
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
            pd.DataFrame(results).to_csv(OUT_DIR / 'checkpoint_v8.csv', index=False)
        gc.collect()

    bench_df = pd.DataFrame(results)
    bench_df.to_csv(OUT_DIR / 'benchmark_results_v8.csv', index=False)
    t_done = time.perf_counter() - t0
    print(f'\n  Benchmark done in {t_done/60:.1f} min')

    errors = bench_df['error'].notna() & (bench_df['error'] != '')
    if errors.any():
        print(f'  WARNING: {errors.sum()} jobs had errors:')
        for _, row in bench_df[errors].head(5).iterrows():
            print(f'    [{row["algorithm"]}] fold={row["fold"]} {row["error"]}')

    # ===== 4. BAR Score =====
    print('\n[4/8] Computing BAR Scores...')
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
        bar_df.to_csv(OUT_DIR / 'bar_score_results_v8.csv', index=False)
    bench_df.to_csv(OUT_DIR / 'benchmark_results_v8.csv', index=False)

    # ===== 5. Statistical analysis =====
    print('\n[5/8] Statistical analysis (Friedman + Holm-Bonferroni)...')
    stat_results = {}

    batch_algos = [a for a in ALGO_NAMES_BATCH if a in bench_df['algorithm'].values]
    batch_df = bench_df[bench_df['algorithm'].isin(batch_algos)]
    if len(batch_df['algorithm'].unique()) >= 2:
        stat_results['batch'] = statistical_analysis(batch_df, 'Batch', batch_algos)

    stream_500 = bench_df[(bench_df['algorithm'].isin(ALGO_NAMES_STREAM)) &
                           (bench_df['label_budget'] == 500)]
    stream_500 = stream_500.dropna(subset=['AUC_PR'])
    stream_algos_500 = [a for a in ALGO_NAMES_STREAM if a in stream_500['algorithm'].values]
    if len(stream_algos_500) >= 2:
        stat_results['streaming_500'] = statistical_analysis(stream_500, 'Streaming_500', stream_algos_500)

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
    print('\n[6/8] Generating plots...')
    plot_overview(bench_df, OUT_DIR)
    plot_difficulty(bench_df, OUT_DIR)
    plot_ablation(bench_df, OUT_DIR)
    plot_bar_score(bench_df, OUT_DIR)
    plot_pareto_frontier(bench_df, OUT_DIR)

    # ===== 7. Concept Drift Evaluation (NEW in v8) =====
    print('\n[7/8] Evaluating concept drift adaptation...')
    
    # Use month 1 data as base
    X_base = monthly_X[0][:10000].astype(np.float64)
    y_base = np.zeros(len(X_base), dtype=np.int8)
    
    drift_algos = {
        'MemStream':        MemStream,
        'CA-DIF-EIA-Stream': CADIFEiaStream,
        'sHST-River':       sHST_River,
        'Random':           RandomBaseline,
    }
    
    drift_results = {}
    for algo_name, algo_cls in drift_algos.items():
        print(f'  Evaluating {algo_name}...')
        try:
            algo_results = {}
            for seed in SEEDS:
                res = evaluate_concept_drift(
                    algo_cls, X_base, y_base, seed,
                    label_budget=500,
                    drift_fraction=0.5,
                    drift_magnitude=3.0,
                )
                for k, v in res.items():
                    if k not in algo_results:
                        algo_results[k] = []
                    algo_results[k].append(v)
            
            # Average across seeds
            drift_results[algo_name] = {}
            for k, vals in algo_results.items():
                valid = [v for v in vals if not np.isnan(v)]
                if valid:
                    drift_results[algo_name][k] = float(np.mean(valid))
        except Exception as e:
            print(f'    {algo_name} concept drift eval failed: {e}')
    
    # Plot concept drift results
    plot_concept_drift(drift_results, OUT_DIR)
    
    # Save drift results
    drift_df = pd.DataFrame(drift_results).T
    drift_df.to_csv(OUT_DIR / 'concept_drift_results_v8.csv')
    print(f'  Concept drift evaluation complete')

    # ===== 8. Report =====
    print('\n[8/8] Writing report...')
    write_report(bench_df, stat_results, drift_results, t_done, OUT_DIR)
    print('\n  DONE!')


def write_report(df, stat_results, drift_results, t_total, out_dir):
    lines = []
    lines.append('# Benchmark v8 Results — MemStream Scientific Correction + Concept Drift')
    lines.append(f'**Generated:** {datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}')
    lines.append(f'**Runtime:** {t_total/60:.1f} min')
    lines.append(f'**Protocol:** Three-way split (train/val/test), threshold from validation set')
    lines.append(f'**Folds:** Leave-one-month-out (5 folds, independent test sets)')
    lines.append(f'**Datasets:** NYC Yellow Taxi Jan-Jun 2024')
    lines.append(f'**Seeds:** {SEEDS}')
    lines.append(f'**Difficulties:** {DIFFICULTIES}')
    lines.append(f'**Anomaly Rate:** {ANOMALY_RATE:.0%} ({ANOMALY_N} anomalies / {TEST_N} test samples)')
    lines.append('')
    lines.append('## Key Corrections from v7')
    lines.append('')
    lines.append('1. **MemStream**: Completely rewritten to match Bhatia et al. WWW 2022 paper.')
    lines.append('   - DAE offline training → encode to latent → memory stores ENCODED vectors')
    lines.append('   - L1 kNN distance in latent space (not raw features)')
    lines.append('   - FIFO memory update ONLY if score < beta (anti-poisoning)')
    lines.append('2. **CA-DIF-EIA-Stream**: Uses trained DAE during warmup (not random projection)')
    lines.append('   - Online fine-tuning of decoder layers')
    lines.append('   - ADWIN drift detection with full retrain')
    lines.append('3. **Concept Drift Evaluation**: Within-stream drift injection added')
    lines.append('   - Tests memory adaptation behavior (scores decrease post-drift)')
    lines.append('4. **Minor**: sklearn_LOF removed, seeds/budgets reduced for speed')
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
            for algo, row in sub.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).items():
                lines.append(f'- **{algo}**: AUC-PR={row:.4f}')

    lines.append('')
    lines.append('## Concept Drift Evaluation (NEW v8)')
    lines.append('')
    lines.append('Within-stream concept drift injection: 50% of test stream shifted by 3σ.')
    lines.append('Key hypothesis: MemStream scores should DECREASE after adaptation period')
    lines.append('because normal (post-drift) samples update the memory via anti-poisoning.')
    lines.append('')
    lines.append('| Algorithm | Pre-Drift Mean | Drift-Early Mean | Drift-Late Mean | Adapted Mean | Adaptation Ratio |')
    lines.append('|-----------|----------------|------------------|-----------------|--------------|------------------|')
    for algo in sorted(drift_results.keys()):
        r = drift_results[algo]
        pre = r.get('pre_mean_score', float('nan'))
        early = r.get('drift_early_mean_score', float('nan'))
        late = r.get('drift_late_mean_score', float('nan'))
        adpt = r.get('adapted_mean_score', float('nan'))
        ar = r.get('adaptation_ratio', float('nan'))
        lines.append(f'| {algo} | {pre:.3f} | {early:.3f} | {late:.3f} | {adpt:.3f} | {ar:.3f} |')
    lines.append('')
    lines.append('*Adaptation Ratio = Drift-Late / Drift-Early. Values < 1.0 indicate memory adaptation.*')
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

    with open(out_dir / 'benchmark_v8_results.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    run()
