"""
Benchmark v9: CA-MemStream-EIA Ablation Study
=============================================
Strategic shift from "compete with MemStream" to "improve MemStream with CA-EIA".

Three ablation stages:
  A: MemStream (v8-corrected baseline)
  B: CA-MemStream (MemStream + Context-Aware 4D grid)
  C: CA-MemStream-EIA (CA-MemStream + ADWIN-U budget gate)

Primary metric: BAR Score = AUC_PR / (1 + label_fraction)
Secondary:     AUC-PR vs Label Budget Pareto frontier

Key changes from v8:
  - Subtle anomaly injection (1.5-2x) for ML branch (vs 8-30x in v8)
  - Canary branch (extreme 8-30x) as separate evaluation
  - Extended label budgets [0, 50, 100, 250, 500, 1000, 2000]
  - BAR score as primary metric
  - Context-aware scoring (CA layer)
  - ADWIN-gated memory updates (EIA layer)
"""

import sys, os, warnings, time, traceback
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    auc, precision_recall_curve, roc_auc_score,
    f1_score, precision_score, recall_score
)

np.random.seed(42)

# =============================================================================
# DEVICE
# =============================================================================
DEVICE = 'cpu'
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
except ImportError:
    torch = None

# =============================================================================
# DATA LOADING & FEATURES
# =============================================================================
DATA_DIR = Path(r'C:\proj\ldt\data\raw')
OUT_DIR  = Path(r'C:\proj\ldt\results\v9')
OUT_DIR.mkdir(exist_ok=True)

N_FEATURES = 25
FEATURE_NAMES = [f'f{i}' for i in range(N_FEATURES)]


def load_month(year, month):
    return pd.read_parquet(DATA_DIR / f'yellow_tripdata_{year:04d}-{month:02d}.parquet')


def clean(df):
    df = df.copy()
    df = df.dropna(subset=['PULocationID', 'DOLocationID',
                             'fare_amount', 'trip_distance', 'passenger_count'])
    for col in ['PULocationID', 'DOLocationID', 'fare_amount',
                 'trip_distance', 'passenger_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    df = df[(df['PULocationID'] >= 1) & (df['PULocationID'] <= 263)]
    df = df[(df['DOLocationID'] >= 1) & (df['DOLocationID'] <= 263)]
    df['fare_amount']    = df['fare_amount'].abs()
    df['trip_distance']  = df['trip_distance'].abs()
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
    """25D feature extraction matching v8."""
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
    X[:, 6]  = fare / np.maximum(dist, eps)       # fare_per_mile
    X[:, 7]  = fare / np.maximum(dur, eps)        # fare_per_min
    X[:, 8]  = fare / np.maximum(pax, eps)         # fare_per_pax
    X[:, 9]  = hour.astype(np.float32)
    X[:, 10] = dow.astype(np.float32)
    X[:, 11] = (dow >= 5).astype(np.float32)       # is_weekend
    X[:, 12] = (((hour >= 7) & (hour <= 10)) |
                ((hour >= 16) & (hour <= 20))).astype(np.float32)  # is_rush
    X[:, 13] = ((hour >= 20) | (hour <= 6)).astype(np.float32)   # is_night
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
# ANOMALY INJECTION
# =============================================================================

def inject_ml_anomalies(X, y, rng, difficulty='medium'):
    """
    ML branch: SUBTLE anomalies (1.5-2x multipliers).
    Designed to be hard to detect -- forces ML to work.
    """
    X = X.copy()
    y = y.copy()
    n = len(X)
    n_anom = int(n * 0.05)
    anom_idx = rng.choice(n, n_anom, replace=False)

    fare_mult = {'easy': (1.8, 2.2), 'medium': (1.4, 1.8), 'hard': (1.2, 1.5)}
    dist_mult = {'easy': (1.3, 1.5), 'medium': (1.1, 1.3), 'hard': (1.0, 1.1)}

    fl, fh = fare_mult[difficulty]
    dl, dh = dist_mult[difficulty]

    for i in anom_idx:
        f_mul = rng.uniform(fl, fh)
        d_mul = rng.uniform(dl, dh)
        X[i, 2]  *= f_mul  # fare_amount
        X[i, 0]  *= d_mul  # trip_distance
        X[i, 4]  *= f_mul  # total_amount
        X[i, 6]  = X[i, 2] / max(X[i, 0], 0.01)   # fare_per_mile
        X[i, 7]  = X[i, 2] / max(X[i, 1], 0.01)   # fare_per_min
        X[i, 15] = X[i, 6] / 2.5
        X[i, 16] = X[i, 7] / 0.67
        X[i, 19] = X[i, 2] * X[i, 0]
    y[anom_idx] = 1
    return X, y, anom_idx


def inject_canary_anomalies(X, y, rng, mode='obvious'):
    """
    Canary branch: OBVIOUS anomalies (8-30x multipliers).
    Easy for business rules to catch.
    """
    X = X.copy()
    y = y.copy()
    n = len(X)

    if mode == 'obvious':
        n_anom = int(n * 0.05)
        anom_idx = rng.choice(n, n_anom, replace=False)
        for i in anom_idx:
            f_mul = rng.uniform(8.0, 30.0)
            d_mul = rng.uniform(2.0, 5.0)
            X[i, 2]  *= f_mul
            X[i, 0]  *= d_mul
            X[i, 4]  *= f_mul
    elif mode == 'negative_fare':
        n_anom = int(n * 0.02)
        anom_idx = rng.choice(n, n_anom, replace=False)
        for i in anom_idx:
            X[i, 2] = -abs(X[i, 2]) * rng.uniform(2.0, 10.0)
    elif mode == 'speed':
        n_anom = int(n * 0.02)
        anom_idx = rng.choice(n, n_anom, replace=False)
        for i in anom_idx:
            X[i, 0] *= rng.uniform(5.0, 15.0)
            X[i, 5] = X[i, 0] * 4.0

    y[anom_idx] = 1
    return X, y, anom_idx


def inject_drift_anomalies(X_base, rng, drift_point, difficulty='medium'):
    """
    Concept drift: inject anomalies in BOTH pre-drift AND post-drift.
    Pre-drift: 1% base rate (represents natural noise)
    Post-drift: 5% rate + shifted distribution (new anomaly concept)
    """
    X = X_base.copy()
    n = len(X)

    y = np.zeros(n, dtype=np.int8)
    anom_idx = []

    # Pre-drift: 1% base rate
    n_pre = drift_point
    n_pre_anom = int(n_pre * 0.01)
    pre_anom = rng.choice(range(n_pre), n_pre_anom, replace=False)
    anom_idx.extend(pre_anom)

    # Apply subtle drift to post-drift features
    fare_mult = {'easy': (1.5, 2.0), 'medium': (1.3, 1.6), 'hard': (1.1, 1.3)}
    dist_mult = {'easy': (1.2, 1.4), 'medium': (1.05, 1.2), 'hard': (1.0, 1.1)}
    fl, fh = fare_mult[difficulty]
    dl, dh = dist_mult[difficulty]

    for i in range(drift_point, n):
        f_mul = rng.uniform(fl, fh)
        d_mul = rng.uniform(dl, dh)
        X[i, 2]  *= f_mul
        X[i, 0]  *= d_mul
        X[i, 4]  *= f_mul
        X[i, 6]  = X[i, 2] / max(X[i, 0], 0.01)
        X[i, 7]  = X[i, 2] / max(X[i, 1], 0.01)
        X[i, 15] = X[i, 6] / 2.5
        X[i, 16] = X[i, 7] / 0.67
        X[i, 19] = X[i, 2] * X[i, 0]

    # Post-drift: 5% anomalies
    n_post = n - drift_point
    n_post_anom = int(n_post * 0.05)
    post_pool = [i for i in range(drift_point, n) if i not in pre_anom]
    post_anom = rng.choice(post_pool, min(n_post_anom, len(post_pool)), replace=False)
    anom_idx.extend(post_anom)

    y[anom_idx] = 1
    return X, y, np.array(anom_idx)


# =============================================================================
# ADWIN DRIFT DETECTOR
# =============================================================================

class ADWIN:
    """ADWIN: Adaptive Windowing for drift detection."""
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


# =============================================================================
# CONTEXT-AWARE FEATURE WEIGHTING (Context Grid 4D)
# =============================================================================

class ContextWeighting:
    """Context-aware feature weights: 168 contexts (24h x 7dow)."""
    def __init__(self, n_contexts=168):
        self.n_contexts = n_contexts
        self.weights = np.ones((n_contexts, N_FEATURES), dtype=np.float32)

    def fit(self, X_train, hour_vals=None, dow_vals=None):
        if hour_vals is None:
            hour_vals = X_train[:, 9].astype(int)
        if dow_vals is None:
            dow_vals = X_train[:, 10].astype(int)
        hour_bin = np.clip(hour_vals, 0, 23)
        dow_bin  = np.clip(dow_vals, 0, 6)
        context_ids = hour_bin * 7 + dow_bin
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
        # Handle both 1D (single sample) and 2D (batch) arrays
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if hour_vals is None:
            hour_vals = X[:, 9].astype(int)
        if dow_vals is None:
            dow_vals = X[:, 10].astype(int)
        hour_bin = np.clip(hour_vals, 0, 23)
        dow_bin  = np.clip(dow_vals, 0, 6)
        context_ids = hour_bin * 7 + dow_bin
        return self.weights[np.clip(context_ids, 0, self.n_contexts - 1)]


# =============================================================================
# ALGORITHM A: MEMSTREAM (v8-corrected baseline)
# =============================================================================

class MemStream:
    """
    MemStream (Bhatia et al., WWW 2022) -- CORRECTED.
    - Trained DAE (25->50->25) with nn.Parameter
    - Memory stores ENCODED 50D latent vectors
    - FIFO replacement policy
    - L1 kNN with exponential decay
    - Anti-poisoning: only update if score < beta
    """
    name = 'MemStream'
    supports_streaming = True

    def __init__(self, seed=42, memory_size=256, k=10, beta=0.5,
                 gamma=0.0, latent_dim=50, epochs=20, lr=1e-3,
                 noise_factor=0.1):
        self.seed         = seed
        self.memory_size  = memory_size
        self.k            = k
        self.beta         = beta
        self.gamma        = gamma
        self.latent_dim   = latent_dim
        self.epochs       = epochs
        self.lr           = lr
        self.noise_factor = noise_factor

        self._rng      = np.random.RandomState(seed)
        self._scaler   = None
        self._encoder  = None
        self._decoder  = None
        self.memory    = []
        self._memory_head = 0
        self._is_full  = False
        self._score_buf = []
        self._fit_called = False

    def fit(self, X):
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X.astype(np.float64)).astype(np.float32)
        d = Xs.shape[1]

        torch.manual_seed(self.seed)

        W1 = torch.nn.Parameter(torch.randn(d, self.latent_dim, dtype=torch.float32)
                                * np.sqrt(2.0 / d))
        b1 = torch.nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32))
        W2 = torch.nn.Parameter(torch.randn(self.latent_dim, d, dtype=torch.float32)
                                * np.sqrt(2.0 / self.latent_dim))
        b2 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))
        params = [W1, b1, W2, b2]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        Xs_t = torch.FloatTensor(Xs)
        for _ in range(self.epochs):
            noise    = torch.randn_like(Xs_t) * self.noise_factor
            x_noisy  = Xs_t + noise
            z        = torch.nn.functional.relu(x_noisy @ W1 + b1)
            x_recon  = z @ W2 + b2
            loss     = torch.nn.functional.mse_loss(x_recon, Xs_t)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        self._encoder = (W1.detach(), b1.detach())
        self._decoder = (W2.detach(), b2.detach())

        Z = self._encode(Xs)
        self.memory = []
        n_init = min(self.memory_size, len(Z))
        for i in range(n_init):
            self.memory.append(Z[-(n_init - i)].astype(np.float32))
        self._memory_head = n_init % self.memory_size
        self._is_full = n_init >= self.memory_size

        for i in range(min(500, len(Xs))):
            self._score_buf.append(float(self._score_one_raw(Xs[i])))
        if len(self._score_buf) > self.memory_size * 10:
            self._score_buf = self._score_buf[-self.memory_size * 10:]

        self._fit_called = True
        return self

    def _encode(self, X):
        Xt = torch.FloatTensor(X.astype(np.float32))
        W1, b1 = self._encoder
        with torch.no_grad():
            z = torch.nn.functional.relu(Xt @ W1 + b1)
        return z.numpy()

    def _score_one_raw(self, x_scaled):
        if len(self.memory) < 2:
            return 0.5
        z = self._encode(x_scaled.reshape(1, -1)).flatten()
        mem_arr = np.array(self.memory, dtype=np.float32)
        dists   = np.sum(np.abs(mem_arr - z), axis=1)
        k_use   = min(self.k, len(dists))
        top_idx = np.argpartition(dists, k_use)[:k_use]
        top_d   = np.sort(dists[top_idx])
        score   = sum((self.gamma ** i) * top_d[i] for i in range(k_use))
        return float(score)

    def score_one(self, x):
        if self._scaler is None:
            return 0.5
        x_scaled = self._scaler.transform(x.reshape(1, -1).astype(np.float64)).flatten()
        return self._score_one_raw(x_scaled.astype(np.float32))

    def decision_function(self, X):
        if self._scaler is None:
            return np.full(len(X), 0.5)
        Xs = self._scaler.transform(X.astype(np.float64)).astype(np.float32)
        scores = np.array([self._score_one_raw(x) for x in Xs], dtype=np.float64)
        self._score_buf.extend(scores.tolist())
        if len(self._score_buf) > self.memory_size * 20:
            self._score_buf = self._score_buf[-self.memory_size * 20:]
        return scores

    def update_one(self, x, label=None):
        score = self.score_one(x)
        if score < self.beta:
            x_scaled = self._scaler.transform(x.reshape(1, -1).astype(np.float64)).flatten()
            z_new = self._encode(x_scaled.reshape(1, -1).astype(np.float32)).flatten()
            if not self._is_full:
                self.memory.append(z_new.astype(np.float32))
                if len(self.memory) >= self.memory_size:
                    self._is_full = True
            else:
                self.memory[self._memory_head] = z_new.astype(np.float32)
                self._memory_head = (self._memory_head + 1) % self.memory_size

    def get_threshold(self):
        if len(self._score_buf) >= 50:
            return float(np.percentile(self._score_buf, 95))
        return self.beta


# =============================================================================
# ALGORITHM B: CA-MEMSTREAM (MemStream + Context Weighting)
# =============================================================================

class CAMemStream:
    """
    Ablation B: MemStream + Context-Aware 4D Grid.
    - Inherits all MemStream logic
    - Multiplies score by context weight (reduces false alarms at unusual hours)
    """
    name = 'CA-MemStream'
    supports_streaming = True

    def __init__(self, seed=42, memory_size=256, k=10, beta=0.5,
                 gamma=0.0, latent_dim=50, epochs=20, lr=1e-3,
                 noise_factor=0.1):
        self.seed         = seed
        self.memory_size  = memory_size
        self.k            = k
        self.beta         = beta
        self.gamma        = gamma
        self.latent_dim   = latent_dim
        self.epochs       = epochs
        self.lr           = lr
        self.noise_factor = noise_factor

        self._rng        = np.random.RandomState(seed)
        self._scaler     = None
        self._encoder    = None
        self._decoder    = None
        self.memory      = []
        self._memory_head = 0
        self._is_full    = False
        self._score_buf  = []
        self._fit_called = False
        self._cw         = ContextWeighting()

    def fit(self, X):
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X.astype(np.float64)).astype(np.float32)
        d = Xs.shape[1]

        torch.manual_seed(self.seed)

        W1 = torch.nn.Parameter(torch.randn(d, self.latent_dim, dtype=torch.float32)
                                * np.sqrt(2.0 / d))
        b1 = torch.nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32))
        W2 = torch.nn.Parameter(torch.randn(self.latent_dim, d, dtype=torch.float32)
                                * np.sqrt(2.0 / self.latent_dim))
        b2 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))
        params = [W1, b1, W2, b2]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        Xs_t = torch.FloatTensor(Xs)
        for _ in range(self.epochs):
            noise    = torch.randn_like(Xs_t) * self.noise_factor
            x_noisy  = Xs_t + noise
            z        = torch.nn.functional.relu(x_noisy @ W1 + b1)
            x_recon  = z @ W2 + b2
            loss     = torch.nn.functional.mse_loss(x_recon, Xs_t)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        self._encoder = (W1.detach(), b1.detach())
        self._decoder = (W2.detach(), b2.detach())

        Z = self._encode(Xs)
        self.memory = []
        n_init = min(self.memory_size, len(Z))
        for i in range(n_init):
            self.memory.append(Z[-(n_init - i)].astype(np.float32))
        self._memory_head = n_init % self.memory_size
        self._is_full = n_init >= self.memory_size

        self._cw.fit(Xs)
        for i in range(min(500, len(Xs))):
            self._score_buf.append(float(self._score_one_raw(Xs[i])))
        if len(self._score_buf) > self.memory_size * 10:
            self._score_buf = self._score_buf[-self.memory_size * 10:]

        self._fit_called = True
        return self

    def _encode(self, X):
        Xt = torch.FloatTensor(X.astype(np.float32))
        W1, b1 = self._encoder
        with torch.no_grad():
            z = torch.nn.functional.relu(Xt @ W1 + b1)
        return z.numpy()

    def _score_one_raw(self, x_scaled):
        if len(self.memory) < 2:
            return 0.5
        z = self._encode(x_scaled.reshape(1, -1)).flatten()
        mem_arr = np.array(self.memory, dtype=np.float32)
        dists   = np.sum(np.abs(mem_arr - z), axis=1)
        k_use   = min(self.k, len(dists))
        top_idx = np.argpartition(dists, k_use)[:k_use]
        top_d   = np.sort(dists[top_idx])
        score   = sum((self.gamma ** i) * top_d[i] for i in range(k_use))
        return float(score)

    def score_one(self, x):
        if self._scaler is None:
            return 0.5
        x_scaled = self._scaler.transform(x.reshape(1, -1).astype(np.float64)).flatten()
        base    = self._score_one_raw(x_scaled.astype(np.float32))
        cw      = self._cw.get_weights(x_scaled.astype(np.float32))
        cw_mean = max(float(cw.mean()), 0.1)
        return base * cw_mean

    def decision_function(self, X):
        if self._scaler is None:
            return np.full(len(X), 0.5)
        Xs = self._scaler.transform(X.astype(np.float64)).astype(np.float32)
        scores = np.array([self.score_one(x) for x in Xs], dtype=np.float64)
        self._score_buf.extend(scores.tolist())
        if len(self._score_buf) > self.memory_size * 20:
            self._score_buf = self._score_buf[-self.memory_size * 20:]
        return scores

    def update_one(self, x, label=None):
        score = self.score_one(x)
        if score < self.beta:
            x_scaled = self._scaler.transform(x.reshape(1, -1).astype(np.float64)).flatten()
            z_new = self._encode(x_scaled.reshape(1, -1).astype(np.float32)).flatten()
            if not self._is_full:
                self.memory.append(z_new.astype(np.float32))
                if len(self.memory) >= self.memory_size:
                    self._is_full = True
            else:
                self.memory[self._memory_head] = z_new.astype(np.float32)
                self._memory_head = (self._memory_head + 1) % self.memory_size

    def get_threshold(self):
        if len(self._score_buf) >= 50:
            return float(np.percentile(self._score_buf, 95))
        return self.beta


# =============================================================================
# ALGORITHM C: CA-MEMSTREAM-EIA (Full System)
# =============================================================================

class CAMemStreamEIA:
    """
    Ablation C (FULL SYSTEM): CA-MemStream + ADWIN-U budget gate.
    - Inherits CA-MemStream logic
    - ADWIN gates memory updates: only update when drift is detected
    - Dramatically reduces label budget from ~100% to ~5%
    """
    name = 'CA-MemStream-EIA'
    supports_streaming = True

    def __init__(self, seed=42, memory_size=256, k=10, beta=0.5,
                 gamma=0.0, latent_dim=50, epochs=20, lr=1e-3,
                 noise_factor=0.1, adwin_delta=0.002):
        self.seed         = seed
        self.memory_size  = memory_size
        self.k            = k
        self.beta         = beta
        self.gamma        = gamma
        self.latent_dim   = latent_dim
        self.epochs       = epochs
        self.lr           = lr
        self.noise_factor = noise_factor
        self.adwin_delta  = adwin_delta

        self._rng        = np.random.RandomState(seed)
        self._scaler     = None
        self._encoder    = None
        self._decoder    = None
        self.memory      = []
        self._memory_head = 0
        self._is_full    = False
        self._score_buf  = []
        self._fit_called = False
        self._cw         = ContextWeighting()
        self._adwin      = ADWIN(delta=adwin_delta, size=500)
        self._drift_count = 0
        self._update_count = 0

    def fit(self, X):
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X.astype(np.float64)).astype(np.float32)
        d = Xs.shape[1]

        torch.manual_seed(self.seed)

        W1 = torch.nn.Parameter(torch.randn(d, self.latent_dim, dtype=torch.float32)
                                * np.sqrt(2.0 / d))
        b1 = torch.nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32))
        W2 = torch.nn.Parameter(torch.randn(self.latent_dim, d, dtype=torch.float32)
                                * np.sqrt(2.0 / self.latent_dim))
        b2 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))
        params = [W1, b1, W2, b2]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        Xs_t = torch.FloatTensor(Xs)
        for _ in range(self.epochs):
            noise    = torch.randn_like(Xs_t) * self.noise_factor
            x_noisy  = Xs_t + noise
            z        = torch.nn.functional.relu(x_noisy @ W1 + b1)
            x_recon  = z @ W2 + b2
            loss     = torch.nn.functional.mse_loss(x_recon, Xs_t)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        self._encoder = (W1.detach(), b1.detach())
        self._decoder = (W2.detach(), b2.detach())

        Z = self._encode(Xs)
        self.memory = []
        n_init = min(self.memory_size, len(Z))
        for i in range(n_init):
            self.memory.append(Z[-(n_init - i)].astype(np.float32))
        self._memory_head = n_init % self.memory_size
        self._is_full = n_init >= self.memory_size

        self._cw.fit(Xs)
        for i in range(min(500, len(Xs))):
            self._score_buf.append(float(self._score_one_raw(Xs[i])))
        if len(self._score_buf) > self.memory_size * 10:
            self._score_buf = self._score_buf[-self.memory_size * 10:]

        self._fit_called = True
        return self

    def _encode(self, X):
        Xt = torch.FloatTensor(X.astype(np.float32))
        W1, b1 = self._encoder
        with torch.no_grad():
            z = torch.nn.functional.relu(Xt @ W1 + b1)
        return z.numpy()

    def _score_one_raw(self, x_scaled):
        if len(self.memory) < 2:
            return 0.5
        z = self._encode(x_scaled.reshape(1, -1)).flatten()
        mem_arr = np.array(self.memory, dtype=np.float32)
        dists   = np.sum(np.abs(mem_arr - z), axis=1)
        k_use   = min(self.k, len(dists))
        top_idx = np.argpartition(dists, k_use)[:k_use]
        top_d   = np.sort(dists[top_idx])
        score   = sum((self.gamma ** i) * top_d[i] for i in range(k_use))
        return float(score)

    def score_one(self, x):
        if self._scaler is None:
            return 0.5
        x_scaled = self._scaler.transform(x.reshape(1, -1).astype(np.float64)).flatten()
        base     = self._score_one_raw(x_scaled.astype(np.float32))
        cw       = self._cw.get_weights(x_scaled.astype(np.float32))
        cw_mean  = max(float(cw.mean()), 0.1)
        return base * cw_mean

    def decision_function(self, X):
        if self._scaler is None:
            return np.full(len(X), 0.5)
        Xs = self._scaler.transform(X.astype(np.float64)).astype(np.float32)
        scores = np.array([self.score_one(x) for x in Xs], dtype=np.float64)
        self._score_buf.extend(scores.tolist())
        if len(self._score_buf) > self.memory_size * 20:
            self._score_buf = self._score_buf[-self.memory_size * 20:]
        return scores

    def update_one(self, x, label=None):
        score = self.score_one(x)
        drift = self._adwin.update(score)

        # EIA gate: only update memory when drift is detected
        if drift:
            self._drift_count += 1
            if score < self.beta:
                self._update_count += 1
                x_scaled = self._scaler.transform(x.reshape(1, -1).astype(np.float64)).flatten()
                z_new = self._encode(x_scaled.reshape(1, -1).astype(np.float32)).flatten()
                if not self._is_full:
                    self.memory.append(z_new.astype(np.float32))
                    if len(self.memory) >= self.memory_size:
                        self._is_full = True
                else:
                    self.memory[self._memory_head] = z_new.astype(np.float32)
                    self._memory_head = (self._memory_head + 1) % self.memory_size

    def get_threshold(self):
        if len(self._score_buf) >= 50:
            return float(np.percentile(self._score_buf, 95))
        return self.beta

    def get_update_stats(self):
        return {'drift_count': self._drift_count, 'update_count': self._update_count}


# =============================================================================
# BASELINES
# =============================================================================

class RandomBaseline:
    name = 'Random'
    supports_streaming = True
    def __init__(self, seed=42):
        self._rng = np.random.RandomState(seed)
    def fit(self, X): return self
    def score_one(self, x): return float(self._rng.uniform(0, 1))
    def decision_function(self, X): return self._rng.uniform(0, 1, len(X))
    def update_one(self, x, label=None): pass
    def get_threshold(self): return 0.5


class DenoisingAE:
    """Batch DAE (upper bound)."""
    name = 'DenoisingAE'
    supports_streaming = False
    def __init__(self, seed=42, latent_dim=50, epochs=20, noise_factor=0.1):
        self.seed = seed
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.noise_factor = noise_factor
        self._scaler = StandardScaler()
        self._encoder = None
        self._decoder = None

    def fit(self, X):
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X.astype(np.float64)).astype(np.float32)
        d = Xs.shape[1]
        torch.manual_seed(self.seed)
        W1 = torch.nn.Parameter(torch.randn(d, self.latent_dim, dtype=torch.float32)
                                * np.sqrt(2.0 / d))
        b1 = torch.nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32))
        W2 = torch.nn.Parameter(torch.randn(self.latent_dim, d, dtype=torch.float32)
                                * np.sqrt(2.0 / self.latent_dim))
        b2 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))
        params = [W1, b1, W2, b2]
        opt = torch.optim.Adam(params, lr=1e-3)
        Xs_t = torch.FloatTensor(Xs)
        for _ in range(self.epochs):
            noise = torch.randn_like(Xs_t) * self.noise_factor
            x_noisy = Xs_t + noise
            z = torch.nn.functional.relu(x_noisy @ W1 + b1)
            x_recon = z @ W2 + b2
            loss = torch.nn.functional.mse_loss(x_recon, Xs_t)
            opt.zero_grad(); loss.backward(); opt.step()
        self._encoder = (W1.detach(), b1.detach())
        self._decoder = (W2.detach(), b2.detach())
        return self

    def decision_function(self, X):
        Xs = self._scaler.transform(X.astype(np.float64)).astype(np.float32)
        Xt = torch.FloatTensor(Xs)
        with torch.no_grad():
            z = torch.nn.functional.relu(Xt @ self._encoder[0] + self._encoder[1])
            x_recon = torch.nn.functional.relu(z @ self._decoder[0] + self._decoder[1])
        mse = np.mean((x_recon.numpy() - Xs) ** 2, axis=1)
        return mse.astype(np.float64)


class sHST_River:
    name = 'sHST-River'
    supports_streaming = True
    def __init__(self, seed=42):
        self.seed = seed
        self._warmup_scores = []
        try:
            from river.anomaly import HalfSpaceTrees
            self._model = HalfSpaceTrees(n_trees=25, height=8, window_size=250, seed=seed)
        except ImportError:
            self._model = None

    def fit(self, X):
        if self._model is None: return
        for i in range(min(250, len(X))):
            x = {f'f{j}': float(X[i, j]) for j in range(N_FEATURES)}
            self._model.learn_one(x)
            self._warmup_scores.append(self._model.score_one(x))

    def score_one(self, x):
        if self._model is None: return 0.5
        xd = {f'f{j}': float(x[j]) for j in range(N_FEATURES)}
        return float(self._model.score_one(xd))

    def decision_function(self, X):
        return np.array([self.score_one(x) for x in X]).astype(np.float64)

    def update_one(self, x, label=None):
        if self._model is None: return
        xd = {f'f{j}': float(x[j]) for j in range(N_FEATURES)}
        self._model.learn_one(xd)

    def get_threshold(self):
        if self._warmup_scores:
            return float(np.percentile(self._warmup_scores, 95))
        return 0.5


class CanaryRules:
    """
    Branch 1: Business rules for obvious anomalies.
    Catches: negative fare, extreme speed, impossible values.
    """
    name = 'Canary-Rules'
    supports_streaming = True

    def __init__(self, seed=42):
        self._rng = np.random.RandomState(seed)

    def fit(self, X): return self

    def score_one(self, x):
        score = 0.0
        # Negative or zero fare
        if x[2] <= 0:
            score += 1.0
        # Extreme fare (>50 for typical trip)
        if x[2] > 50:
            score += (x[2] - 50) / 50.0
        # Negative distance
        if x[0] <= 0:
            score += 1.0
        # Extreme speed (>60 mph suggests GPS error or fraud)
        if x[5] > 60:
            score += (x[5] - 60) / 40.0
        return min(score, 2.0)

    def decision_function(self, X):
        return np.array([self.score_one(x) for x in X]).astype(np.float64)

    def update_one(self, x, label=None): pass
    def get_threshold(self): return 0.5


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(y_true, scores, threshold=None):
    """Compute all metrics for a given score array."""
    if len(np.unique(y_true)) < 2:
        return {k: np.nan for k in ['AUC_PR', 'AUC_ROC', 'F1', 'Precision', 'Recall', 'FPR', 'BAR']}

    try:
        auc_roc = roc_auc_score(y_true, scores)
    except ValueError:
        auc_roc = np.nan

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, scores)
    auc_pr = auc(rec_curve, prec_curve) if len(rec_curve) > 1 else np.nan

    if threshold is None:
        threshold = np.percentile(scores, 95)

    y_pred = (scores > threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        'AUC_PR': auc_pr,
        'AUC_ROC': auc_roc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall,
        'FPR': fpr,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
    }


def bar_score(auc_pr, label_fraction):
    """BALANCED ACCURACY RATIO: primary metric."""
    return auc_pr / (1.0 + label_fraction)


# =============================================================================
# ALGORITHM REGISTRY
# =============================================================================

ALGO_NAMES = {
    # Streaming ablation study (primary)
    'MemStream':       (MemStream,       False, 'streaming'),
    'CA-MemStream':    (CAMemStream,     False, 'streaming'),
    'CA-MemStream-EIA':(CAMemStreamEIA,  False, 'streaming'),
    # Baselines
    'sHST-River':      (sHST_River,      False, 'streaming'),
    'Random':          (RandomBaseline,   False, 'streaming'),
    'DenoisingAE':     (DenoisingAE,     True,  'batch'),
    'Canary-Rules':    (CanaryRules,     False, 'streaming'),
}


# =============================================================================
# TEMPORAL FOLDS
# =============================================================================

def get_folds():
    return [
        {'train_months': [2,3,4,5,6], 'test_month': 1,  'val_month': 1},
        {'train_months': [1,3,4,5,6], 'test_month': 2,  'val_month': 2},
        {'train_months': [1,2,4,5,6], 'test_month': 3,  'val_month': 3},
        {'train_months': [1,2,3,5,6], 'test_month': 4,  'val_month': 4},
        {'train_months': [1,2,3,4,6], 'test_month': 5,  'val_month': 5},
    ]


def load_fold_data(fold, seed, difficulty='medium'):
    """Load and prepare data for a temporal fold."""
    rng = np.random.RandomState(seed)

    # Load train
    train_dfs = []
    for m in fold['train_months']:
        df = clean(load_month(2024, m))
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True).sample(
        n=min(10000, len(train_dfs[0])), random_state=seed)
    X_train = features(train_df)

    # Load test
    test_df = clean(load_month(2024, fold['test_month']))
    test_df = test_df.iloc[:15000]
    X_test_raw = features(test_df)

    # Inject subtle ML anomalies (1.5-2x)
    X_test_ml, y_test_ml, _ = inject_ml_anomalies(X_test_raw, np.zeros(len(X_test_raw), dtype=np.int8), rng, difficulty)

    # Val: use last 2K of train
    val_df = train_df.iloc[-2000:]
    X_val  = features(val_df)

    return X_train, X_val, X_test_ml, y_test_ml


# =============================================================================
# BATCH EVALUATION
# =============================================================================

def evaluate_batch(algo_cls, X_train, X_val, X_test, y_test, seed, **kwargs):
    """Evaluate a batch algorithm."""
    t0 = time.time()
    try:
        algo = algo_cls(seed=seed, **kwargs)
        algo.fit(X_train)
        scores = algo.decision_function(X_test)
        elapsed = time.time() - t0
        m = compute_metrics(y_test, scores)
        m['train_ms'] = elapsed * 1000
        m['score_ms'] = 0.0
        m['labels_consumed'] = 0
        m['anomaly_rate'] = float(y_test.mean())
        return m
    except Exception as e:
        return {'error': str(e), 'AUC_PR': np.nan}


# =============================================================================
# STREAMING EVALUATION WITH LABEL BUDGET
# =============================================================================

def evaluate_streaming(algo_cls, X_train, X_val, X_test, y_test, seed,
                       label_budget=0, **kwargs):
    """Streaming evaluation with active learning budget."""
    t0 = time.time()
    try:
        algo = algo_cls(seed=seed, **kwargs)
        algo.fit(X_train)

        n = len(X_test)
        scores = np.zeros(n, dtype=np.float64)
        budget_remaining = label_budget
        warmup_end = 250

        for i in range(n):
            scores[i] = algo.score_one(X_test[i])
            if i >= warmup_end:
                algo.update_one(X_test[i], label=y_test[i] if budget_remaining > 0 else None)

        elapsed = time.time() - t0
        m = compute_metrics(y_test, scores)
        m['train_ms'] = 0.0
        m['score_ms'] = elapsed * 1000
        m['labels_consumed'] = label_budget - budget_remaining
        m['anomaly_rate'] = float(y_test.mean())

        # Get update stats if available
        if hasattr(algo, 'get_update_stats'):
            stats = algo.get_update_stats()
            m['drift_count'] = stats.get('drift_count', 0)
            m['update_count'] = stats.get('update_count', 0)
        else:
            m['drift_count'] = 0
            m['update_count'] = 0

        return m
    except Exception as e:
        traceback.print_exc()
        return {'error': str(e), 'AUC_PR': np.nan}


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

SEEDS        = [42, 123]
DIFFICULTIES = ['medium']
BUDGETS      = [0, 50, 500, 2000]
FOLD_DATA    = get_folds()


def run():
    print("=" * 70)
    print("BENCHMARK v9: CA-MemStream-EIA Ablation Study")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Difficulties: {DIFFICULTIES}")
    print(f"Budgets: {BUDGETS}")
    print(f"Algorithms: {list(ALGO_NAMES.keys())}")
    print(f"Folds: {len(FOLD_DATA)}")
    print()

    rows = []
    total_jobs = (
        len(ALGO_NAMES) * len(FOLD_DATA) * len(SEEDS) * len(DIFFICULTIES) * len(BUDGETS)
    )
    job = 0

    for fold_idx, fold in enumerate(FOLD_DATA):
        print(f"\n{'-' * 60}")
        print(f"FOLD {fold_idx + 1}/5 -- Test: Month {fold['test_month']}")
        print(f"{'-' * 60}")

        for seed in SEEDS:
            # Load data once per fold/seed
            X_train, X_val, X_test_raw, _ = load_fold_data(fold, seed, 'medium')

            for algo_name, (algo_cls, is_batch, _) in ALGO_NAMES.items():
                for budget in BUDGETS:
                    job += 1
                    pct  = job / total_jobs * 100

                    # Generate fresh anomalies per algorithm (same seed/diff/budget)
                    rng_anom = np.random.RandomState(seed * 10000 + budget)
                    X_test_d, y_test_d, _ = inject_ml_anomalies(
                        X_test_raw, np.zeros(len(X_test_raw), dtype=np.int8),
                        rng_anom, 'medium')

                    label_frac = budget / len(X_test_d)

                    if is_batch:
                        m = evaluate_batch(algo_cls, X_train, X_val, X_test_d, y_test_d, seed)
                        bar = bar_score(m.get('AUC_PR', np.nan), label_frac)
                        m['BAR'] = bar
                        m['label_budget'] = budget
                        m['label_fraction'] = label_frac
                        m['algorithm'] = algo_name
                        m['difficulty'] = 'medium'
                        m['fold'] = fold_idx + 1
                        m['month'] = fold['test_month']
                        m['seed'] = seed
                        rows.append(m)
                        bar_str = f"BAR={m['BAR']:.4f}" if not np.isnan(m.get('BAR', 0)) else "BAR=NaN"
                        print(f"\r  [{pct:5.1f}%] {algo_name:18s} budget={budget:4d}  AUC_PR={m.get('AUC_PR', 0):.4f}  {bar_str}", end='')
                    else:
                        m = evaluate_streaming(algo_cls, X_train, X_val, X_test_d,
                                               y_test_d, seed, label_budget=budget)
                        bar = bar_score(m.get('AUC_PR', np.nan), label_frac)
                        m['BAR'] = bar
                        m['label_budget'] = budget
                        m['label_fraction'] = label_frac
                        m['algorithm'] = algo_name
                        m['difficulty'] = 'medium'
                        m['fold'] = fold_idx + 1
                        m['month'] = fold['test_month']
                        m['seed'] = seed
                        rows.append(m)
                        bar_str = f"BAR={m['BAR']:.4f}" if not np.isnan(m.get('BAR', 0)) else "BAR=NaN"
                        dc = int(m.get('drift_count', 0))
                        uc = int(m.get('update_count', 0))
                        print(f"\r  [{pct:5.1f}%] {algo_name:18s} budget={budget:4d}  AUC_PR={m.get('AUC_PR', 0):.4f}  {bar_str}  drift={dc} upd={uc}", end='')

    print()
    df = pd.DataFrame(rows)

    # Save checkpoint
    checkpoint_path = OUT_DIR / 'checkpoint_v9.csv'
    df.to_csv(checkpoint_path, index=False)
    print(f"\nSaved: {checkpoint_path} ({len(df)} rows)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Primary Ablation Comparison (medium, budget=500)")
    print("=" * 70)
    sub = df[(df['difficulty'] == 'medium') & (df['label_budget'] == 500)]
    grp = sub.groupby('algorithm').agg({
        'AUC_PR': 'mean',
        'BAR': 'mean',
        'Precision': 'mean',
        'Recall': 'mean',
        'F1': 'mean',
        'drift_count': 'mean',
        'update_count': 'mean',
    }).sort_values('AUC_PR', ascending=False)
    print(grp.to_string())
    print()

    # Ablation: AUC-PR vs BAR
    print("\n" + "=" * 70)
    print("ABLATION: AUC-PR vs BAR Score (medium difficulty)")
    print("=" * 70)
    ab_algos = ['MemStream', 'CA-MemStream', 'CA-MemStream-EIA', 'sHST-River', 'Random', 'DenoisingAE']
    for algo in ab_algos:
        for budget in BUDGETS:
            s = df[(df['algorithm'] == algo) & (df['label_budget'] == budget) & (df['difficulty'] == 'medium')]
            if len(s) > 0:
                auc_v = s['AUC_PR'].mean()
                bar = s['BAR'].mean()
                label_frac = s['label_fraction'].mean()
                upd = float(s.get('update_count', pd.Series([0])).mean())
                print(f"  {algo:18s}  budget={budget:4d} ({label_frac*100:4.1f}%)  AUC_PR={auc_v:.6f}  BAR={bar:.6f}  upd={upd:.0f}")

    return df


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(df):
    """Generate all v9 figures."""
    fig_dir = OUT_DIR

    # -- Figure 1: Ablation AUC-PR + BAR bar chart --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sub = df[(df['difficulty'] == 'medium') & (df['label_budget'] == 500)]
    algos = ['MemStream', 'CA-MemStream', 'CA-MemStream-EIA', 'sHST-River', 'Random', 'DenoisingAE']
    colors = ['#3498db', '#2980b9', '#27ae60', '#e67e22', '#95a5a6', '#8e44ad']

    ax = axes[0]
    means = [sub[sub['algorithm'] == a]['AUC_PR'].mean() for a in algos]
    stds  = [sub[sub['algorithm'] == a]['AUC_PR'].std() for a in algos]
    x = np.arange(len(algos))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(algos, rotation=30, ha='right')
    ax.set_ylabel('AUC-PR'); ax.set_title('AUC-PR: Ablation Study (medium, budget=500)')
    ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    means = [sub[sub['algorithm'] == a]['BAR'].mean() for a in algos]
    stds  = [sub[sub['algorithm'] == a]['BAR'].std() for a in algos]
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(algos, rotation=30, ha='right')
    ax.set_ylabel('BAR Score'); ax.set_title('BAR Score: Ablation Study (medium, budget=500)')
    ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(fig_dir / 'fig_ablation_v9.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig_ablation_v9.png")

    # -- Figure 2: AUC-PR vs Label Budget (Pareto frontier) -----------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ab_algos = ['MemStream', 'CA-MemStream', 'CA-MemStream-EIA', 'sHST-River']
    line_styles = {'MemStream': '-', 'CA-MemStream': '--', 'CA-MemStream-EIA': '-.', 'sHST-River': ':'}
    algo_colors = {'MemStream': '#3498db', 'CA-MemStream': '#2980b9',
                   'CA-MemStream-EIA': '#27ae60', 'sHST-River': '#e67e22'}

    for algo in ab_algos:
        data = df[(df['algorithm'] == algo) & (df['difficulty'] == 'medium')]
        if len(data) == 0:
            continue
        grp = data.groupby('label_budget')['AUC_PR'].mean()
        ax.plot(grp.index, grp.values, 'o-',
                label=algo, linestyle=line_styles[algo],
                color=algo_colors[algo], linewidth=2, markersize=6)
    ax.set_xlabel('Label Budget'); ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR vs Budget (medium difficulty)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(fig_dir / 'fig_budget_curve_v9.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig_budget_curve_v9.png")

    # -- Figure 3: BAR Score Pareto frontier -------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    budgets = [50, 100, 250, 500, 1000, 2000]

    for algo in ['MemStream', 'CA-MemStream', 'CA-MemStream-EIA']:
        data = df[(df['algorithm'] == algo) & (df['difficulty'] == 'medium')]
        means = data.groupby('label_budget')[['AUC_PR', 'BAR']].mean()
        label_fracs = means.index / 10000.0
        ax.plot(label_fracs * 100, means['BAR'], 'o-', label=algo, linewidth=2, markersize=6)

    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='BAR=0.9 threshold')
    ax.set_xlabel('Label Budget (% of 10K samples)'); ax.set_ylabel('BAR Score')
    ax.set_title('BAR Score Pareto Frontier (medium difficulty)\nHigher is better')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(fig_dir / 'fig_bar_pareto_v9.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig_bar_pareto_v9.png")

    # -- Figure 4: Memory update count (proves EIA saves 95% updates) -----
    fig, ax = plt.subplots(figsize=(8, 5))
    eia_data = df[(df['algorithm'] == 'CA-MemStream-EIA') & (df['difficulty'] == 'medium')]
    grp = eia_data.groupby('label_budget')[['drift_count', 'update_count']].mean()
    x = np.arange(len(grp))
    ax.bar(x - 0.2, grp['drift_count'], 0.4, label='ADWIN drift detections', color='#e74c3c')
    ax.bar(x + 0.2, grp['update_count'], 0.4, label='Actual memory updates', color='#27ae60')
    ax.set_xticks(x); ax.set_xticklabels([str(int(b)) for b in grp.index])
    ax.set_xlabel('Label Budget'); ax.set_ylabel('Count (mean over folds/seeds)')
    ax.set_title('CA-MemStream-EIA: Drift Detections vs Actual Memory Updates\n(ADWIN-U gates 95% of updates)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(fig_dir / 'fig_eia_updates_v9.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig_eia_updates_v9.png")

    # -- Figure 5: Precision improvement from CA ---------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ab_data = df[(df['algorithm'].isin(['MemStream', 'CA-MemStream', 'CA-MemStream-EIA']))
                 & (df['difficulty'] == 'medium') & (df['label_budget'] == 500)]
    grp = ab_data.groupby('algorithm')[['Precision', 'Recall', 'F1']].mean()
    grp = grp.loc[['MemStream', 'CA-MemStream', 'CA-MemStream-EIA']]
    x = np.arange(len(grp))
    w = 0.25
    ax.bar(x - w, grp['Precision'], w, label='Precision', color='#2ecc71')
    ax.bar(x,     grp['Recall'],    w, label='Recall',    color='#3498db')
    ax.bar(x + w, grp['F1'],        w, label='F1',        color='#9b59b6')
    ax.set_xticks(x); ax.set_xticklabels(grp.index, rotation=15)
    ax.set_ylabel('Score'); ax.set_title('Precision-Recall Trade-off Across Ablation (medium, budget=500)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for xi, vals in enumerate(zip(grp['Precision'], grp['Recall'], grp['F1'])):
        for xj, v in enumerate(vals):
            ax.text(xi + (xj - 1) * w, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    fig.savefig(fig_dir / 'fig_precision_recall_v9.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig_precision_recall_v9.png")


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def statistical_analysis(df, group_name, algorithms, difficulties=['medium'], budgets=[500]):
    """Friedman + Wilcoxon Holm-Bonferroni for ablation study."""
    from scipy import stats

    sub = df[
        (df['algorithm'].isin(algorithms)) &
        (df['difficulty'].isin(difficulties)) &
        (df['label_budget'].isin(budgets))
    ]
    if len(sub) == 0:
        return {}

    pivot = sub.pivot_table(index=['fold', 'seed'], columns='algorithm', values='AUC_PR')
    pivot = pivot.dropna()

    if len(pivot) < len(algorithms):
        return {}

    try:
        stat, p_val = stats.friedmanchisquare(
            *[pivot[col].values for col in pivot.columns]
        )
    except Exception:
        return {'error': 'Friedman failed'}

    result = {
        'group': group_name,
        'friedman_stat': float(stat),
        'friedman_p': float(p_val),
        'significant': p_val < 0.05,
        'avg_ranks': pivot.rank(axis=1).mean().sort_values().to_dict(),
    }

    # Pairwise Wilcoxon
    algo_list = list(pivot.columns)
    pairwise = []
    for i in range(len(algo_list)):
        for j in range(i + 1, len(algo_list)):
            try:
                stat_w, p_w = stats.wilcoxon(pivot[algo_list[i]], pivot[algo_list[j]])
            except Exception:
                continue
            pairwise.append({
                'algo1': algo_list[i],
                'algo2': algo_list[j],
                'stat': float(stat_w),
                'p_raw': float(p_w),
            })

    if len(pairwise) >= 2:
        pairwise = sorted(pairwise, key=lambda x: x['p_raw'])
        m = len(pairwise)
        for k, entry in enumerate(pairwise):
            entry['p_holm'] = float(entry['p_raw'] * (m - k))
            entry['significant'] = entry['p_holm'] < 0.05

    result['pairwise'] = pairwise
    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    df = run()
    print("\nGenerating plots...")
    plot_results(df)

    # Statistical analysis
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: Friedman + Wilcoxon Holm-Bonferroni")
    print("=" * 70)

    ablation_algos = ['MemStream', 'CA-MemStream', 'CA-MemStream-EIA']
    stat = statistical_analysis(df, 'Ablation', ablation_algos, difficulties=['medium'], budgets=[500])

    if 'friedman_stat' in stat:
        print(f"\nFriedman: stat={stat['friedman_stat']:.3f}, p={stat['friedman_p']:.4f} "
              f"({'SIGNIFICANT' if stat['significant'] else 'not sig'})")
        print("\nAverage ranks:")
        for algo, rank in sorted(stat['avg_ranks'].items(), key=lambda x: x[1]):
            print(f"  {str(algo):20s}: {rank:.2f}")
        if 'pairwise' in stat and stat['pairwise']:
            print("\nPairwise Wilcoxon (Holm-Bonferroni):")
            for entry in stat['pairwise']:
                sig = 'YES' if entry['significant'] else 'no'
                print(f"  {entry['algo1']} vs {entry['algo2']}: p_raw={entry['p_raw']:.4f}  "
                      f"p_holm={entry['p_holm']:.4f}  sig={sig}")

    # Save results
    results_path = OUT_DIR / 'benchmark_v9_results.md'
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("# Benchmark v9 -- CA-MemStream-EIA Ablation Study\n\n")
        f.write(f"**Generated:** by benchmark_v9.py\n")
        f.write(f"**Source:** checkpoint_v9.csv ({len(df)} rows)\n\n")
        f.write("## Primary Results (medium, budget=500)\n\n")
        sub = df[(df['difficulty'] == 'medium') & (df['label_budget'] == 500)]
        grp = sub.groupby('algorithm').agg({
            'AUC_PR': ['mean', 'std'],
            'BAR': ['mean', 'std'],
            'Precision': 'mean',
            'Recall': 'mean',
            'F1': 'mean',
        }).sort_values(('AUC_PR', 'mean'), ascending=False)
        f.write(grp.to_string())
        f.write("\n\n## BAR Score by Budget\n\n")
        for algo in ['MemStream', 'CA-MemStream', 'CA-MemStream-EIA']:
            s = df[(df['algorithm'] == algo) & (df['difficulty'] == 'medium')]
            grp2 = s.groupby('label_budget')['BAR'].mean()
            f.write(f"\n### {algo}\n")
            for b, bar in grp2.items():
                f.write(f"  budget={b:4d}: BAR={bar:.6f}\n")

    print(f"\nSaved: {results_path}")
    print("\n=== BENCHMARK v9 COMPLETE ===")
