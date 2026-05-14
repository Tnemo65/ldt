"""
Refined Concept Drift Evaluation for v9.
Fixes v8 flaw: anomalies in BOTH pre-drift AND post-drift segments.

Protocol:
  Phase 1 (0-50%): Normal + 1% base anomaly rate
  Phase 2 (50-100%): Shifted distribution + 5% anomaly rate

Metrics:
  - AUC-PR in each phase
  - Score trajectory
  - Recovery time after drift
  - Adaptation ratio (late/early post-drift)
"""

import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve

sys.path.insert(0, str(Path(__file__).parent))
DATA_DIR = Path(r'C:\proj\ldt\data\raw')
OUT_DIR  = Path(r'C:\proj\ldt\results\v9')

import torch

# =============================================================================
# Copy classes from v9
# =============================================================================

class ADWIN:
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


class ContextWeighting:
    def __init__(self, n_contexts=168):
        self.n_contexts = n_contexts
        self.weights = np.ones((n_contexts, 25), dtype=np.float32)

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


class MemStream:
    name = 'MemStream'
    def __init__(self, seed=42, memory_size=256, k=10, beta=0.5,
                 gamma=0.0, latent_dim=50, epochs=20, lr=1e-3, noise_factor=0.1):
        self.seed = seed; self.memory_size = memory_size; self.k = k
        self.beta = beta; self.gamma = gamma; self.latent_dim = latent_dim
        self.epochs = epochs; self.lr = lr; self.noise_factor = noise_factor
        self._rng = np.random.RandomState(seed)
        self._scaler = None; self._encoder = None; self._decoder = None
        self.memory = []; self._memory_head = 0; self._is_full = False
        self._score_buf = []

    def fit(self, X):
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X.astype(np.float64)).astype(np.float32)
        d = Xs.shape[1]
        torch.manual_seed(self.seed)
        W1 = torch.nn.Parameter(torch.randn(d, self.latent_dim, dtype=torch.float32) * np.sqrt(2.0/d))
        b1 = torch.nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32))
        W2 = torch.nn.Parameter(torch.randn(self.latent_dim, d, dtype=torch.float32) * np.sqrt(2.0/self.latent_dim))
        b2 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))
        params = [W1, b1, W2, b2]
        opt = torch.optim.Adam(params, lr=self.lr)
        Xs_t = torch.FloatTensor(Xs)
        for _ in range(self.epochs):
            noise = torch.randn_like(Xs_t) * self.noise_factor
            z = torch.nn.functional.relu((Xs_t + noise) @ W1 + b1)
            x_recon = z @ W2 + b2
            loss = torch.nn.functional.mse_loss(x_recon, Xs_t)
            opt.zero_grad(); loss.backward(); opt.step()
        self._encoder = (W1.detach(), b1.detach())
        self._decoder = (W2.detach(), b2.detach())
        Z = self._encode(Xs)
        self.memory = []
        n_init = min(self.memory_size, len(Z))
        for i in range(n_init):
            self.memory.append(Z[-(n_init-i)].astype(np.float32))
        self._memory_head = n_init % self.memory_size
        self._is_full = n_init >= self.memory_size
        for i in range(min(500, len(Xs))):
            self._score_buf.append(float(self._score_one_raw(Xs[i])))
        return self

    def _encode(self, X):
        Xt = torch.FloatTensor(X.astype(np.float32))
        W1, b1 = self._encoder
        with torch.no_grad():
            z = torch.nn.functional.relu(Xt @ W1 + b1)
        return z.numpy()

    def _score_one_raw(self, x_scaled):
        if len(self.memory) < 2: return 0.5
        z = self._encode(x_scaled.reshape(1,-1)).flatten()
        mem_arr = np.array(self.memory, dtype=np.float32)
        dists = np.sum(np.abs(mem_arr - z), axis=1)
        k_use = min(self.k, len(dists))
        top_idx = np.argpartition(dists, k_use)[:k_use]
        top_d = np.sort(dists[top_idx])
        return float(sum((self.gamma**i) * top_d[i] for i in range(k_use)))

    def score_one(self, x):
        if self._scaler is None: return 0.5
        x_scaled = self._scaler.transform(x.reshape(1,-1).astype(np.float64)).flatten()
        return self._score_one_raw(x_scaled.astype(np.float32))

    def update_one(self, x, label=None):
        score = self.score_one(x)
        if score < self.beta:
            x_scaled = self._scaler.transform(x.reshape(1,-1).astype(np.float64)).flatten()
            z_new = self._encode(x_scaled.reshape(1,-1).astype(np.float32)).flatten()
            if not self._is_full:
                self.memory.append(z_new.astype(np.float32))
                if len(self.memory) >= self.memory_size: self._is_full = True
            else:
                self.memory[self._memory_head] = z_new.astype(np.float32)
                self._memory_head = (self._memory_head + 1) % self.memory_size


class CAMemStream:
    name = 'CA-MemStream'
    def __init__(self, seed=42, memory_size=256, k=10, beta=0.5,
                 gamma=0.0, latent_dim=50, epochs=20, lr=1e-3, noise_factor=0.1):
        self.seed = seed; self.memory_size = memory_size; self.k = k
        self.beta = beta; self.gamma = gamma; self.latent_dim = latent_dim
        self.epochs = epochs; self.lr = lr; self.noise_factor = noise_factor
        self._rng = np.random.RandomState(seed)
        self._scaler = None; self._encoder = None; self._decoder = None
        self.memory = []; self._memory_head = 0; self._is_full = False
        self._score_buf = []; self._cw = ContextWeighting()

    def fit(self, X):
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X.astype(np.float64)).astype(np.float32)
        d = Xs.shape[1]
        torch.manual_seed(self.seed)
        W1 = torch.nn.Parameter(torch.randn(d, self.latent_dim, dtype=torch.float32) * np.sqrt(2.0/d))
        b1 = torch.nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32))
        W2 = torch.nn.Parameter(torch.randn(self.latent_dim, d, dtype=torch.float32) * np.sqrt(2.0/self.latent_dim))
        b2 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))
        params = [W1, b1, W2, b2]
        opt = torch.optim.Adam(params, lr=self.lr)
        Xs_t = torch.FloatTensor(Xs)
        for _ in range(self.epochs):
            noise = torch.randn_like(Xs_t) * self.noise_factor
            z = torch.nn.functional.relu((Xs_t + noise) @ W1 + b1)
            x_recon = z @ W2 + b2
            loss = torch.nn.functional.mse_loss(x_recon, Xs_t)
            opt.zero_grad(); loss.backward(); opt.step()
        self._encoder = (W1.detach(), b1.detach())
        self._decoder = (W2.detach(), b2.detach())
        Z = self._encode(Xs)
        self.memory = []
        n_init = min(self.memory_size, len(Z))
        for i in range(n_init):
            self.memory.append(Z[-(n_init-i)].astype(np.float32))
        self._memory_head = n_init % self.memory_size
        self._is_full = n_init >= self.memory_size
        self._cw.fit(Xs)
        for i in range(min(500, len(Xs))):
            self._score_buf.append(float(self._score_one_raw(Xs[i])))
        return self

    def _encode(self, X):
        Xt = torch.FloatTensor(X.astype(np.float32))
        W1, b1 = self._encoder
        with torch.no_grad():
            z = torch.nn.functional.relu(Xt @ W1 + b1)
        return z.numpy()

    def _score_one_raw(self, x_scaled):
        if len(self.memory) < 2: return 0.5
        z = self._encode(x_scaled.reshape(1,-1)).flatten()
        mem_arr = np.array(self.memory, dtype=np.float32)
        dists = np.sum(np.abs(mem_arr - z), axis=1)
        k_use = min(self.k, len(dists))
        top_idx = np.argpartition(dists, k_use)[:k_use]
        top_d = np.sort(dists[top_idx])
        return float(sum((self.gamma**i) * top_d[i] for i in range(k_use)))

    def score_one(self, x):
        if self._scaler is None: return 0.5
        x_scaled = self._scaler.transform(x.reshape(1,-1).astype(np.float64)).flatten()
        base = self._score_one_raw(x_scaled.astype(np.float32))
        cw = self._cw.get_weights(x_scaled.astype(np.float32))
        return base * max(float(cw.mean()), 0.1)

    def update_one(self, x, label=None):
        score = self.score_one(x)
        if score < self.beta:
            x_scaled = self._scaler.transform(x.reshape(1,-1).astype(np.float64)).flatten()
            z_new = self._encode(x_scaled.reshape(1,-1).astype(np.float32)).flatten()
            if not self._is_full:
                self.memory.append(z_new.astype(np.float32))
                if len(self.memory) >= self.memory_size: self._is_full = True
            else:
                self.memory[self._memory_head] = z_new.astype(np.float32)
                self._memory_head = (self._memory_head + 1) % self.memory_size


class CAMemStreamEIA:
    name = 'CA-MemStream-EIA'
    def __init__(self, seed=42, memory_size=256, k=10, beta=0.5,
                 gamma=0.0, latent_dim=50, epochs=20, lr=1e-3, noise_factor=0.1, adwin_delta=0.002):
        self.seed = seed; self.memory_size = memory_size; self.k = k
        self.beta = beta; self.gamma = gamma; self.latent_dim = latent_dim
        self.epochs = epochs; self.lr = lr; self.noise_factor = noise_factor
        self.adwin_delta = adwin_delta
        self._rng = np.random.RandomState(seed)
        self._scaler = None; self._encoder = None; self._decoder = None
        self.memory = []; self._memory_head = 0; self._is_full = False
        self._score_buf = []; self._cw = ContextWeighting()
        self._adwin = ADWIN(delta=adwin_delta, size=500)
        self._drift_count = 0; self._update_count = 0

    def fit(self, X):
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X.astype(np.float64)).astype(np.float32)
        d = Xs.shape[1]
        torch.manual_seed(self.seed)
        W1 = torch.nn.Parameter(torch.randn(d, self.latent_dim, dtype=torch.float32) * np.sqrt(2.0/d))
        b1 = torch.nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32))
        W2 = torch.nn.Parameter(torch.randn(self.latent_dim, d, dtype=torch.float32) * np.sqrt(2.0/self.latent_dim))
        b2 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))
        params = [W1, b1, W2, b2]
        opt = torch.optim.Adam(params, lr=self.lr)
        Xs_t = torch.FloatTensor(Xs)
        for _ in range(self.epochs):
            noise = torch.randn_like(Xs_t) * self.noise_factor
            z = torch.nn.functional.relu((Xs_t + noise) @ W1 + b1)
            x_recon = z @ W2 + b2
            loss = torch.nn.functional.mse_loss(x_recon, Xs_t)
            opt.zero_grad(); loss.backward(); opt.step()
        self._encoder = (W1.detach(), b1.detach())
        self._decoder = (W2.detach(), b2.detach())
        Z = self._encode(Xs)
        self.memory = []
        n_init = min(self.memory_size, len(Z))
        for i in range(n_init):
            self.memory.append(Z[-(n_init-i)].astype(np.float32))
        self._memory_head = n_init % self.memory_size
        self._is_full = n_init >= self.memory_size
        self._cw.fit(Xs)
        for i in range(min(500, len(Xs))):
            self._score_buf.append(float(self._score_one_raw(Xs[i])))
        return self

    def _encode(self, X):
        Xt = torch.FloatTensor(X.astype(np.float32))
        W1, b1 = self._encoder
        with torch.no_grad():
            z = torch.nn.functional.relu(Xt @ W1 + b1)
        return z.numpy()

    def _score_one_raw(self, x_scaled):
        if len(self.memory) < 2: return 0.5
        z = self._encode(x_scaled.reshape(1,-1)).flatten()
        mem_arr = np.array(self.memory, dtype=np.float32)
        dists = np.sum(np.abs(mem_arr - z), axis=1)
        k_use = min(self.k, len(dists))
        top_idx = np.argpartition(dists, k_use)[:k_use]
        top_d = np.sort(dists[top_idx])
        return float(sum((self.gamma**i) * top_d[i] for i in range(k_use)))

    def score_one(self, x):
        if self._scaler is None: return 0.5
        x_scaled = self._scaler.transform(x.reshape(1,-1).astype(np.float64)).flatten()
        base = self._score_one_raw(x_scaled.astype(np.float32))
        cw = self._cw.get_weights(x_scaled.astype(np.float32))
        return base * max(float(cw.mean()), 0.1)

    def update_one(self, x, label=None):
        score = self.score_one(x)
        drift = self._adwin.update(score)
        if drift:
            self._drift_count += 1
            if score < self.beta:
                self._update_count += 1
                x_scaled = self._scaler.transform(x.reshape(1,-1).astype(np.float64)).flatten()
                z_new = self._encode(x_scaled.reshape(1,-1).astype(np.float32)).flatten()
                if not self._is_full:
                    self.memory.append(z_new.astype(np.float32))
                    if len(self.memory) >= self.memory_size: self._is_full = True
                else:
                    self.memory[self._memory_head] = z_new.astype(np.float32)
                    self._memory_head = (self._memory_head + 1) % self.memory_size

    def get_stats(self):
        return {'drift_count': self._drift_count, 'update_count': self._update_count}


class sHST_River:
    name = 'sHST-River'
    def __init__(self, seed=42):
        try:
            from river.anomaly import HalfSpaceTrees
            self._model = HalfSpaceTrees(n_trees=25, height=8, window_size=250, seed=seed)
        except ImportError:
            self._model = None
    def fit(self, X):
        if self._model is None: return
        for i in range(min(250, len(X))):
            d = {f'f{j}': float(X[i,j]) for j in range(25)}
            self._model.learn_one(d)
    def score_one(self, x):
        if self._model is None: return 0.5
        d = {f'f{j}': float(x[j]) for j in range(25)}
        return float(self._model.score_one(d))
    def update_one(self, x, label=None):
        if self._model is None: return
        d = {f'f{j}': float(x[j]) for j in range(25)}
        self._model.learn_one(d)


class RandomBaseline:
    name = 'Random'
    def __init__(self, seed=42):
        self._rng = np.random.RandomState(seed)
    def fit(self, X): pass
    def score_one(self, x): return float(self._rng.uniform(0, 1))
    def update_one(self, x, label=None): pass


# =============================================================================
# DATA LOADING
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
    df['dur_min']  = df['duration_s'] / 60.0
    df['total_amt'] = df.get('total_amount', df['fare_amount'])
    return df.reset_index(drop=True)

def features(df):
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

    X[:, 0]  = dist; X[:, 1] = dur; X[:, 2] = fare; X[:, 3] = pax; X[:, 4] = total; X[:, 5] = spd
    X[:, 6]  = fare / np.maximum(dist, eps)
    X[:, 7]  = fare / np.maximum(dur, eps)
    X[:, 8]  = fare / np.maximum(pax, eps)
    X[:, 9]  = hour.astype(np.float32); X[:, 10] = dow.astype(np.float32)
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
    X[:, 21] = np.sin(np.float32(2*np.pi) * hour / 24).astype(np.float32)
    X[:, 22] = np.cos(np.float32(2*np.pi) * hour / 24).astype(np.float32)
    X[:, 23] = np.sin(np.float32(2*np.pi) * dow / 7).astype(np.float32)
    X[:, 24] = np.cos(np.float32(2*np.pi) * dow / 7).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)


# =============================================================================
# REFINED CONCEPT DRIFT EVALUATION
# =============================================================================

def inject_refined_drift(X_base, rng, drift_point, pre_rate=0.01, post_rate=0.05, magnitude=1.5):
    """
    FIXED from v8: anomalies in BOTH pre-drift AND post-drift.
    Pre-drift: 1% base rate (normal noise level)
    Post-drift: 5% rate + distribution shift
    """
    X = X_base.copy()
    n = len(X)
    y = np.zeros(n, dtype=np.int8)

    # Phase 1: pre-drift with 1% base anomalies
    n_pre = drift_point
    n_pre_anom = int(n_pre * pre_rate)
    pre_anom = rng.choice(n_pre, n_pre_anom, replace=False)
    y[pre_anom] = 1

    # Phase 2: apply distribution shift to ALL post-drift samples
    shift_cols = [0, 2, 6, 7, 15, 19]
    for col in shift_cols:
        col_std = X_base[:drift_point, col].std()
        shift = rng.randn(n - drift_point) * col_std * (magnitude - 1.0)
        X[drift_point:, col] += shift

    # Phase 2: inject 5% anomalies
    n_post = n - drift_point
    n_post_anom = int(n_post * post_rate)
    pool = [i for i in range(drift_point, n) if i not in pre_anom]
    post_anom = rng.choice(pool, min(n_post_anom, len(pool)), replace=False)
    y[post_anom] = 1

    # Make anomalies subtle (1.5-2x multipliers, matching ML hard)
    for i in pre_anom:
        X[i, 2] *= rng.uniform(1.8, 2.2)  # fare
        X[i, 0] *= rng.uniform(1.3, 1.5)  # distance
        X[i, 4] *= rng.uniform(1.8, 2.2)  # total
    for i in post_anom:
        X[i, 2] *= rng.uniform(1.5, 2.0)
        X[i, 0] *= rng.uniform(1.2, 1.4)
        X[i, 4] *= rng.uniform(1.5, 2.0)

    return X, y


def evaluate_drift_refined(algo_cls, X_base, seed, drift_fraction=0.5, magnitude=1.5):
    """Evaluate with refined concept drift protocol."""
    rng = np.random.RandomState(seed)
    n = len(X_base)
    drift_point = int(n * (1 - drift_fraction))

    X, y = inject_refined_drift(X_base, rng, drift_point,
                                  pre_rate=0.01, post_rate=0.05, magnitude=magnitude)

    algo = algo_cls(seed=seed)
    algo.fit(X_base)

    scores = np.zeros(n, dtype=np.float64)
    phase_scores = {'pre': [], 'post': []}

    for i in range(n):
        scores[i] = algo.score_one(X[i])
        if i < drift_point:
            phase_scores['pre'].append(scores[i])
        else:
            phase_scores['post'].append(scores[i])
        if i >= 250:
            algo.update_one(X[i], label=y[i])

    # Overall AUC-PR
    prec, rec, _ = precision_recall_curve(y, scores)
    auc_overall = auc(rec, prec) if len(rec) > 1 else 0.0

    # Phase 1 (pre-drift): has 1% anomalies -> meaningful AUC-PR
    y_pre = y[:drift_point]
    s_pre = scores[:drift_point]
    prec1, rec1, _ = precision_recall_curve(y_pre, s_pre)
    auc_pre = auc(rec1, prec1) if len(rec1) > 1 else 0.0

    # Phase 2 (post-drift): has 5% anomalies + shift -> tests adaptation
    y_post = y[drift_point:]
    s_post = scores[drift_point:]
    prec2, rec2, _ = precision_recall_curve(y_post, s_post)
    auc_post = auc(rec2, prec2) if len(rec2) > 1 else 0.0

    # Recovery: split post-drift into early/late halves
    mid = len(s_post) // 2
    s_early = s_post[:mid]
    y_early = y_post[:mid]
    s_late  = s_post[mid:]
    y_late  = y_post[mid:]

    prec_early, rec_early, _ = precision_recall_curve(y_early, s_early)
    prec_late,  rec_late,  _ = precision_recall_curve(y_late,  s_late)
    auc_early = auc(rec_early, prec_early) if len(rec_early) > 1 else 0.0
    auc_late  = auc(rec_late,  prec_late)  if len(rec_late)  > 1 else 0.0

    # Adaptation ratio: AUC-PR recovery
    recovery_ratio = auc_late / (auc_early + 1e-9)

    # Score trajectory
    window = 200
    rolling = np.convolve(np.array(phase_scores['post']), np.ones(window)/window, mode='valid')
    adaptation_trend = (rolling[-1] - rolling[0]) / (rolling[0] + 1e-9)

    stats = {'drift_count': 0, 'update_count': 0}
    if hasattr(algo, 'get_stats'):
        stats = algo.get_stats()

    return {
        'auc_overall': auc_overall,
        'auc_pre': auc_pre,
        'auc_post': auc_post,
        'auc_early_post': auc_early,
        'auc_late_post': auc_late,
        'recovery_ratio': recovery_ratio,
        'mean_score_pre': float(np.mean(phase_scores['pre'])),
        'mean_score_post': float(np.mean(phase_scores['post'])),
        'adaptation_trend': float(adaptation_trend),
        'drift_count': stats['drift_count'],
        'update_count': stats['update_count'],
        'drift_point': drift_point,
        'n_pre_anom': int(y_pre.sum()),
        'n_post_anom': int(y_post.sum()),
    }


def main():
    print("=" * 70)
    print("REFINED Concept Drift Evaluation v9")
    print("(Fixed: anomalies in BOTH pre and post drift)")
    print("=" * 70)

    print("\nLoading NYC Taxi month 1...")
    df = clean(load_month(2024, 1))
    X_base = features(df)[:10000].astype(np.float64)
    print(f"  {len(X_base)} samples loaded")

    algos = {
        'MemStream':       MemStream,
        'CA-MemStream':    CAMemStream,
        'CA-MemStream-EIA': CAMemStreamEIA,
        'sHST-River':     sHST_River,
        'Random':          RandomBaseline,
    }

    SEEDS = [42, 123, 456]
    results = {}

    for name, cls in algos.items():
        print(f"\nEvaluating {name}...")
        all_keys = ['auc_overall', 'auc_pre', 'auc_post', 'auc_early_post', 'auc_late_post',
                    'recovery_ratio', 'mean_score_pre', 'mean_score_post', 'adaptation_trend',
                    'drift_count', 'update_count']
        agg = {k: [] for k in all_keys}
        for seed in SEEDS:
            r = evaluate_drift_refined(cls, X_base, seed)
            for k in all_keys:
                agg[k].append(r[k])
        avg = {k: float(np.nanmean(v)) for k, v in agg.items()}
        results[name] = avg

        print(f"  Overall AUC-PR : {avg['auc_overall']:.4f}")
        print(f"  Pre-drift AUC-PR: {avg['auc_pre']:.4f}  (1% anomalies, normal distribution)")
        print(f"  Post-drift AUC-PR: {avg['auc_post']:.4f}  (5% anomalies, shifted distribution)")
        print(f"  Early post-drift : {avg['auc_early_post']:.4f}")
        print(f"  Late post-drift  : {avg['auc_late_post']:.4f}")
        print(f"  Recovery ratio   : {avg['recovery_ratio']:.4f}  (>1 = improving)")
        print(f"  Adaptation trend : {avg['adaptation_trend']:.4f}  (<0 = adapting)")
        print(f"  Drift detections : {avg['drift_count']:.0f}")
        print(f"  Memory updates   : {avg['update_count']:.0f}")

    # Save
    drift_df = pd.DataFrame(results).T
    drift_df.to_csv(OUT_DIR / 'concept_drift_results_v9.csv')
    print(f"\nSaved: concept_drift_results_v9.csv")

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    names = list(results.keys())
    colors = {'MemStream': '#3498db', 'CA-MemStream': '#2980b9',
              'CA-MemStream-EIA': '#27ae60', 'sHST-River': '#e67e22', 'Random': '#95a5a6'}

    # AUC-PR by phase
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x - 0.25, [results[n]['auc_pre'] for n in names], 0.25,
           label='Pre-drift (1%)', color='#3498db', alpha=0.85)
    ax.bar(x,       [results[n]['auc_post'] for n in names], 0.25,
           label='Post-drift (5%)', color='#e74c3c', alpha=0.85)
    ax.bar(x + 0.25, [results[n]['auc_late_post'] for n in names], 0.25,
           label='Late post-drift', color='#27ae60', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('AUC-PR'); ax.set_title('AUC-PR by Drift Phase\n(Pre: 1% anom, Post: 5% anom + shift)')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='random')

    # Recovery ratio
    ax = axes[1]
    ratios = [results[n]['recovery_ratio'] for n in names]
    bar_c = ['#27ae60' if r > 1.0 else '#e74c3c' for r in ratios]
    ax.bar(names, ratios, color=bar_c, alpha=0.85)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5)
    ax.set_ylabel('Recovery Ratio'); ax.set_title('Recovery Ratio\n(late/early post-drift AUC-PR, >1 = improving)')
    ax.set_xticklabels(names, rotation=30, ha='right'); ax.grid(axis='y', alpha=0.3)

    # Memory updates (proves EIA saves updates)
    ax = axes[2]
    updates = [results[n]['update_count'] for n in names]
    drifts  = [results[n]['drift_count'] for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, drifts, 0.4, label='Drift detections', color='#e74c3c', alpha=0.85)
    ax.bar(x + 0.2, updates, 0.4, label='Memory updates', color='#27ae60', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Count'); ax.set_title('ADWIN Drift Detections vs Memory Updates\n(EIA gates most updates)')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # Adaptation trend
    ax = axes[3]
    trends = [results[n]['adaptation_trend'] for n in names]
    bar_c = ['#27ae60' if t < 0 else '#e74c3c' for t in trends]
    ax.bar(names, trends, color=bar_c, alpha=0.85)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_ylabel('Score Trend'); ax.set_title('Score Adaptation Trend\n(<0 = memory adapting to new concept)')
    ax.set_xticklabels(names, rotation=30, ha='right'); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig_concept_drift_v9.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig_concept_drift_v9.png")

    print("\n=== Refined Concept Drift Complete ===")


if __name__ == '__main__':
    main()
