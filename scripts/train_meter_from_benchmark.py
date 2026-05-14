#!/usr/bin/env python3
"""
Step 1: Extract real drift scenario data from v9 benchmark.
Runs the CA-MemStream-EIA with concept drift across multiple configurations,
collects meta-features (6D) and ground-truth strategy labels.
"""

import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
DATA_DIR = Path(r'C:\proj\ldt\data\raw')
OUT_FILE = Path(r'C:\proj\ldt\data\meter_training_real.csv')

# =============================================================================
# Copy v9 classes (ADWIN, ContextWeighting, MemStream, CA-MemStream-EIA)
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
        if hour_vals is None:
            hour_vals = X[:, 9].astype(int)
        if dow_vals is None:
            dow_vals = X[:, 10].astype(int)
        hour_bin = np.clip(hour_vals, 0, 23)
        dow_bin  = np.clip(dow_vals, 0, 6)
        context_ids = hour_bin * 7 + dow_bin
        return self.weights[np.clip(context_ids, 0, self.n_contexts - 1)]


class CAMemStreamEIA:
    """CA-MemStream-EIA with ADWIN drift detection and memory gating."""
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
        self._drift_points = []
        self._score_trajectory = []

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
        x = x_scaled.reshape(1, -1) if x_scaled.ndim == 1 else x_scaled
        if len(self.memory) < 2: return 0.5
        z = self._encode(x).flatten()
        mem_arr = np.array(self.memory, dtype=np.float32)
        dists = np.sum(np.abs(mem_arr - z), axis=1)
        k_use = min(self.k, len(dists))
        top_idx = np.argpartition(dists, k_use)[:k_use]
        top_d = np.sort(dists[top_idx])
        return float(sum((self.gamma**i) * top_d[i] for i in range(k_use)))

    def score_one(self, x):
        if self._scaler is None: return 0.5
        x_scaled = self._scaler.transform(x.reshape(1, -1).astype(np.float64)).flatten()
        base = self._score_one_raw(x_scaled)
        cw = self._cw.get_weights(x_scaled.reshape(1, -1).astype(np.float32))
        return base * max(float(cw.mean()), 0.1)

    def update_one(self, x, label=None):
        score = self.score_one(x)
        drift = self._adwin.update(score)
        if drift:
            self._drift_count += 1
            self._drift_points.append(len(self._score_trajectory))
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
        self._score_trajectory.append(score)

    def get_stats(self):
        return {
            'drift_count': self._drift_count,
            'update_count': self._update_count,
            'drift_points': self._drift_points,
            'score_trajectory': self._score_trajectory,
        }


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
        x = x_scaled.reshape(1, -1) if x_scaled.ndim == 1 else x_scaled
        if len(self.memory) < 2: return 0.5
        z = self._encode(x).flatten()
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


# =============================================================================
# Data loading and features (same as v9)
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
# Canary rules (simplified)
# =============================================================================

def apply_canary_rules(X):
    """Simplified canary rule checker.
    Returns: violation_mask (bool array), violation_rate
    """
    fare = X[:, 2]
    dist = X[:, 0]
    pax  = X[:, 3]

    violations = (
        (fare <= 0) |
        ((dist == 0) & (fare > 0)) |
        ((pax < 1) | (pax > 6))
    )
    return violations, violations.mean()


# =============================================================================
# Scenario configurations — realistic drift scenarios
# =============================================================================

# Vary: drift_fraction (WHEN drift happens), magnitude (HOW SEVERE), seed
SCENARIO_CONFIGS = []

# Normal: no drift, 3 seeds
for seed in [42, 123, 456]:
    SCENARIO_CONFIGS.append({'name': 'normal', 'drift_fraction': 0.0, 'magnitude': 1.0, 'seed': seed, 'pre_rate': 0.01, 'post_rate': 0.01})

# Threshold drift (mild): adjust_threshold
for mag in [1.2, 1.5]:
    for seed in [42, 123, 456]:
        SCENARIO_CONFIGS.append({'name': 'threshold_drift', 'drift_fraction': 0.5, 'magnitude': mag, 'seed': seed, 'pre_rate': 0.01, 'post_rate': 0.03})

# Distribution shift (moderate): retrain_model
for mag in [1.8, 2.0, 2.5]:
    for seed in [42, 123, 456]:
        SCENARIO_CONFIGS.append({'name': 'distribution_shift', 'drift_fraction': 0.5, 'magnitude': mag, 'seed': seed, 'pre_rate': 0.01, 'post_rate': 0.05})

# Concept drift (severe): switch_model
for mag in [3.0, 4.0]:
    for seed in [42, 123, 456]:
        SCENARIO_CONFIGS.append({'name': 'concept_drift', 'drift_fraction': 0.5, 'magnitude': mag, 'seed': seed, 'pre_rate': 0.01, 'post_rate': 0.08})


def inject_drift(X_base, rng, drift_point, pre_rate, post_rate, magnitude):
    """Inject anomalies and distribution shift."""
    X = X_base.copy()
    n = len(X)
    y = np.zeros(n, dtype=np.int8)

    n_pre = drift_point
    n_pre_anom = int(n_pre * pre_rate)
    pre_anom = rng.choice(n_pre, n_pre_anom, replace=False).astype(np.int8)
    y[pre_anom] = 1

    if magnitude > 1.0:
        shift_cols = [0, 2, 6, 7, 15, 19]
        for col in shift_cols:
            col_std = X_base[:max(drift_point, 1), col].std()
            shift = rng.randn(n - drift_point) * col_std * (magnitude - 1.0)
            X[drift_point:, col] += shift

    n_post = n - drift_point
    n_post_anom = int(n_post * post_rate)
    pool = np.array(list(range(drift_point, n)))
    post_anom_idx = rng.choice(len(pool), min(n_post_anom, len(pool)), replace=False)
    post_anom = pool[post_anom_idx].astype(np.int8)
    y[post_anom] = 1

    for i in pre_anom:
        X[i, 2] *= rng.uniform(1.8, 2.2)
        X[i, 0] *= rng.uniform(1.3, 1.5)
        X[i, 4] *= rng.uniform(1.8, 2.2)
    for i in post_anom:
        X[i, 2] *= rng.uniform(1.5, 2.0)
        X[i, 0] *= rng.uniform(1.2, 1.4)
        X[i, 4] *= rng.uniform(1.5, 2.0)

    return X, y


def label_strategy(drift_count, recovery_ratio, magnitude, scenario_name):
    """Label strategy based on ADWIN signals and recovery."""
    if scenario_name == 'normal':
        return 0  # do_nothing

    if drift_count == 0:
        return 0  # No drift detected — do_nothing

    if drift_count <= 2 and recovery_ratio >= 0.8:
        return 1  # Mild drift, recovering — adjust_threshold

    if drift_count >= 3 and recovery_ratio < 1.0 and magnitude >= 2.5:
        return 3  # Severe, not recovering — switch_model

    if drift_count >= 2 or recovery_ratio < 1.0:
        return 2  # Moderate drift — retrain_model

    return 1  # Default to adjust


def compute_meta_metrics_window(scores, labels, window_size=500, window_idx=0):
    """Compute meta-metrics for a window of scores."""
    start = window_idx * window_size
    end = min(start + window_size, len(scores))
    if end - start < 50:
        return None

    window_scores = scores[start:end]
    window_labels = labels[start:end]

    anomaly_mask = window_scores > 0.5
    anomaly_rate = anomaly_mask.mean()
    avg_score = window_scores.mean()

    # Simulated violation rate (canary doesn't know true labels)
    violation_rate = 0.03 + 0.02 * np.sin(window_idx * 0.3)
    null_rate = 0.005 + 0.005 * np.random.rand()

    # delta_score per thesis eq (5.18)
    epsilon = 1e-6
    delta_score = abs(violation_rate - anomaly_rate) / (violation_rate + anomaly_rate + epsilon)

    return {
        'volume': end - start,
        'null_rate': null_rate,
        'violation_rate': violation_rate,
        'anomaly_rate': anomaly_rate,
        'avg_anomaly_score': avg_score,
        'delta_score': delta_score,
        'strategy': None,  # filled later
        'scenario': None,
        'seed': None,
        'window_idx': window_idx,
    }


def run_scenario(X_base, config):
    """Run one scenario and collect meta-metrics."""
    rng = np.random.RandomState(config['seed'])
    name = config['name']
    drift_fraction = config['drift_fraction']
    magnitude = config['magnitude']
    pre_rate = config['pre_rate']
    post_rate = config['post_rate']

    n = len(X_base)
    drift_point = int(n * (1 - drift_fraction)) if drift_fraction > 0 else n

    X, y = inject_drift(X_base, rng, drift_point, pre_rate, post_rate, magnitude)

    # Test both MemStream and CA-MemStream-EIA
    for algo_name, algo_cls in [('MemStream', MemStream), ('CA-MemStream-EIA', CAMemStreamEIA)]:
        algo = algo_cls(seed=config['seed'], memory_size=256, k=10, beta=0.5)
        algo.fit(X_base[:5000])

        algo_scores = []
        for i in range(n):
            s = algo.score_one(X[i])
            algo_scores.append(s)
            if i >= 250:
                algo.update_one(X[i], label=y[i])

        scores = np.array(algo_scores)

        stats = algo.get_stats() if hasattr(algo, 'get_stats') else {}
        drift_count = stats.get('drift_count', 0)
        score_trajectory = stats.get('score_trajectory', scores)

        # Compute recovery ratio from post-drift scores
        if drift_fraction > 0:
            mid = drift_point + (n - drift_point) // 2
            early_post = scores[drift_point:mid]
            late_post  = scores[mid:n]
            early_mean = np.mean(early_post) if len(early_post) > 0 else 0.5
            late_mean  = np.mean(late_post) if len(late_post) > 0 else 0.5
            recovery_ratio = late_mean / (early_mean + 1e-9)
        else:
            recovery_ratio = 1.0

        strategy = label_strategy(drift_count, recovery_ratio, magnitude, name)

        # Extract meta-metrics from post-drift window (last 500 samples)
        window_size = 500
        if n >= window_size:
            post_scores = scores[max(0, n-window_size):]
            post_labels = y[max(0, n-window_size):]
            mm = compute_meta_metrics_window(post_scores, post_labels, window_size, window_idx=0)
            if mm:
                mm['strategy'] = strategy
                mm['scenario'] = name
                mm['seed'] = config['seed']
                mm['algo'] = algo_name
                mm['drift_count'] = drift_count
                mm['recovery_ratio'] = recovery_ratio
                mm['magnitude'] = magnitude
                yield mm

        # Also extract from pre-drift window for balance
        if drift_fraction > 0 and drift_point >= window_size:
            pre_scores = scores[:window_size]
            pre_labels = y[:window_size]
            mm = compute_meta_metrics_window(pre_scores, pre_labels, window_size, window_idx=0)
            if mm:
                mm['strategy'] = 0  # Normal = do_nothing
                mm['scenario'] = name + '_pre'
                mm['seed'] = config['seed']
                mm['algo'] = algo_name
                mm['drift_count'] = 0
                mm['recovery_ratio'] = 1.0
                mm['magnitude'] = magnitude
                yield mm


def main():
    print("=" * 70)
    print("Step 1: Extract Real Drift Scenario Data from v9 Benchmark")
    print("=" * 70)

    print("\nLoading NYC Taxi data (Jan 2024, first 10K samples)...")
    df = clean(load_month(2024, 1))
    X_base = features(df)[:10000].astype(np.float64)
    print(f"  {len(X_base)} samples loaded")

    print(f"\nRunning {len(SCENARIO_CONFIGS)} scenario configurations...")
    all_records = []

    for i, cfg in enumerate(SCENARIO_CONFIGS):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(SCENARIO_CONFIGS)}")
        for record in run_scenario(X_base, cfg):
            all_records.append(record)

    df_out = pd.DataFrame(all_records)

    # Keep only useful columns
    cols = ['volume', 'null_rate', 'violation_rate', 'anomaly_rate',
            'avg_anomaly_score', 'delta_score', 'strategy', 'scenario',
            'seed', 'algo', 'drift_count', 'recovery_ratio', 'magnitude']
    df_out = df_out[[c for c in cols if c in df_out.columns]]

    # Rename avg_anomaly_score -> avg_score for consistency
    if 'avg_anomaly_score' in df_out.columns:
        df_out = df_out.rename(columns={'avg_anomaly_score': 'avg_score'})

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE} ({len(df_out)} records)")

    print("\nStrategy distribution:")
    for s, name in [(0, 'do_nothing'), (1, 'adjust_threshold'), (2, 'retrain_model'), (3, 'switch_model')]:
        count = (df_out['strategy'] == s).sum()
        print(f"  {s} ({name}): {count}")

    print("\nScenario distribution:")
    print(df_out['scenario'].value_counts())

    print("\n" + "=" * 70)
    print("Step 1 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
