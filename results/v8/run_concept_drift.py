"""Run just the concept drift evaluation from v8."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add v8 module
sys.path.insert(0, str(Path(__file__).parent))

# Import data loading from v8
DATA_DIR = Path(r'C:\proj\ldt\data\raw')
OUT_DIR = Path(r'C:\proj\ldt\results\v8')

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

# Copy necessary classes from v8
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, precision_recall_curve

DEVICE = 'cpu'
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except:
    DEVICE = 'cpu'

class ADWINU:
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

class RandomBaseline:
    def __init__(self, seed=42):
        self._rng = np.random.RandomState(seed)
    def fit(self, X): return self
    def score_one(self, x): return float(self._rng.uniform(0, 1))
    def update_one(self, x, label=None): pass

try:
    from river.anomaly import HalfSpaceTrees
    class sHST_River:
        name = 'sHST-River'
        def __init__(self, seed=42):
            self._model = HalfSpaceTrees(n_trees=25, height=8, window_size=250, seed=seed)
        def fit(self, X):
            for x in X[:250]:
                self._model.learn_one({f'f{i}': float(x[i]) for i in range(25)})
        def score_one(self, x):
            return self._model.score_one({f'f{i}': float(x[i]) for i in range(25)})
        def update_one(self, x, label=None):
            self._model.learn_one({f'f{i}': float(x[i]) for i in range(25)})
except:
    class sHST_River:
        name = 'sHST-River'
        def __init__(self, seed=42): pass
        def fit(self, X): return self
        def score_one(self, x): return 0.5
        def update_one(self, x, label=None): pass

# CORRECTED MemStream
class MemStream:
    name = 'MemStream'
    def __init__(self, seed=42, memory_size=256, k=10, beta=0.5, gamma=0.0, latent_dim=50, epochs=20, lr=1e-3, noise_factor=0.1):
        self.seed = seed; self.memory_size = memory_size; self.k = k; self.beta = beta
        self.gamma = gamma; self.latent_dim = latent_dim; self.epochs = epochs
        self.lr = lr; self.noise_factor = noise_factor
        self._rng = np.random.RandomState(seed)
        self._scaler = None; self._encoder = None; self._decoder = None
        self.memory = []; self._memory_head = 0; self._is_full = False
        self._score_buf = []

    def fit(self, X):
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X.astype(np.float64)).astype(np.float32)
        d = Xs.shape[1]
        torch.manual_seed(self.seed); torch.set_num_threads(4)
        W1 = torch.nn.Parameter(torch.randn(d, self.latent_dim, dtype=torch.float32) * np.sqrt(2.0/d))
        b1 = torch.nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32))
        W2 = torch.nn.Parameter(torch.randn(self.latent_dim, d, dtype=torch.float32) * np.sqrt(2.0/self.latent_dim))
        b2 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))
        params = [W1, b1, W2, b2]
        opt = torch.optim.Adam(params, lr=self.lr)
        Xs_t = torch.FloatTensor(Xs)
        for _ in range(self.epochs):
            noise = torch.randn_like(Xs_t) * self.noise_factor
            x_noisy = Xs_t + noise
            z = torch.nn.functional.relu(x_noisy @ W1 + b1)
            x_recon = z @ W2 + b2
            loss = torch.nn.functional.mse_loss(x_recon, Xs_t)
            opt.zero_grad(); loss.backward(); opt.step()
        self._encoder = (W1.detach(), b1.detach()); self._decoder = (W2.detach(), b2.detach())
        # Init memory
        Z = self._encode(Xs)
        self.memory = []
        n_init = min(self.memory_size, len(Z))
        for i in range(n_init):
            self.memory.append(Z[-(n_init-i)].astype(np.float32))
        self._memory_head = n_init % self.memory_size
        self._is_full = n_init >= self.memory_size
        # Score buffer
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
        sorted_idx = np.argpartition(dists, k_use)[:k_use]
        top_dists = np.sort(dists[sorted_idx])
        score = sum((self.gamma**i) * top_dists[i] for i in range(k_use))
        return float(score)

    def score_one(self, x):
        if self._scaler is None: return 0.5
        x_scaled = self._scaler.transform(x.reshape(1,-1).astype(np.float64)).flatten().astype(np.float32)
        return self._score_one_raw(x_scaled)

    def update_one(self, x, label=None):
        score = self.score_one(x)
        self._score_buf.append(float(score))
        if len(self._score_buf) > 10000: self._score_buf = self._score_buf[-5000:]
        if score < self.beta:
            x_scaled = self._scaler.transform(x.reshape(1,-1).astype(np.float64)).flatten().astype(np.float32)
            z_new = self._encode(x_scaled.reshape(1,-1)).flatten().astype(np.float32)
            if not self._is_full:
                self.memory.append(z_new)
                if len(self.memory) >= self.memory_size: self._is_full = True
            else:
                self.memory[self._memory_head] = z_new
                self._memory_head = (self._memory_head + 1) % self.memory_size


class ContextFeatureWeighting:
    def __init__(self, n_contexts=168, n_features=25):
        self.n_contexts = n_contexts
        self.weights = np.ones((n_contexts, n_features), dtype=np.float32)
    def fit(self, X_train, hour_vals=None, dow_vals=None):
        if hour_vals is None: hour_vals = X_train[:, 9].astype(int)
        if dow_vals is None: dow_vals = X_train[:, 10].astype(int)
        hour_bin = np.clip(hour_vals, 0, 23); dow_bin = np.clip(dow_vals, 0, 6)
        context_ids = hour_bin * 7 + dow_bin
        for c in range(self.n_contexts):
            mask = context_ids == c
            if mask.sum() < 30: continue
            X_c = X_train[mask]; self.weights[c] = X_c.std(axis=0)
            max_w = self.weights[c].max()
            if max_w > 1e-6: self.weights[c] /= max_w
        return self
    def get_weights(self, X, hour_vals=None, dow_vals=None):
        if hour_vals is None: hour_vals = X[:, 9].astype(int)
        if dow_vals is None: dow_vals = X[:, 10].astype(int)
        hour_bin = np.clip(hour_vals, 0, 23); dow_bin = np.clip(dow_vals, 0, 6)
        context_ids = hour_bin * 7 + dow_bin
        return self.weights[np.clip(context_ids, 0, self.n_contexts-1)]


class CADIFEiaStream:
    name = 'CA-DIF-EIA-Stream'
    def __init__(self, seed=42, label_budget=500, drift_delta=0.002, hidden_dim=32, latent_dim=16, epochs=20, noise_factor=0.1, context_history_size=2000):
        self.seed = seed; self.label_budget = label_budget; self.drift_delta = drift_delta
        self.hidden_dim = hidden_dim; self.latent_dim = latent_dim; self.epochs = epochs
        self.noise_factor = noise_factor; self.context_history_size = context_history_size
        self._rng = np.random.RandomState(seed)
        self._scaler = None; self._W1 = self._b1 = self._W2 = self._b2 = None
        self._W3 = self._b3 = self._W4 = self._b4 = None
        self._cw = None; self._context_hist = []
        self._drift = ADWINU(delta=drift_delta, size=500)
        self._warmup_scores = []; self._sample_count = 0

    def fit(self, X_train):
        warmup_n = min(int(len(X_train)*0.2), 3000)
        X_warmup = X_train[:warmup_n]
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X_warmup.astype(np.float64)).astype(np.float32)
        d = Xs.shape[1]
        torch.manual_seed(self.seed); torch.set_num_threads(4)
        W1 = torch.nn.Parameter(torch.randn(d, self.hidden_dim, dtype=torch.float32)*np.sqrt(2.0/d))
        b1 = torch.nn.Parameter(torch.zeros(self.hidden_dim, dtype=torch.float32))
        W2 = torch.nn.Parameter(torch.randn(self.hidden_dim, self.latent_dim, dtype=torch.float32)*np.sqrt(2.0/self.hidden_dim))
        b2 = torch.nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32))
        W3 = torch.nn.Parameter(torch.randn(self.latent_dim, self.hidden_dim, dtype=torch.float32)*np.sqrt(2.0/self.latent_dim))
        b3 = torch.nn.Parameter(torch.zeros(self.hidden_dim, dtype=torch.float32))
        W4 = torch.nn.Parameter(torch.randn(self.hidden_dim, d, dtype=torch.float32)*np.sqrt(2.0/self.hidden_dim))
        b4 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))
        all_params = [W1,b1,W2,b2,W3,b3,W4,b4]
        opt = torch.optim.Adam(all_params, lr=1e-3)
        Xs_t = torch.FloatTensor(Xs)
        for _ in range(self.epochs):
            noise = torch.randn_like(Xs_t) * self.noise_factor
            x_noisy = Xs_t + noise
            h1 = torch.nn.functional.relu(x_noisy @ W1 + b1)
            z = torch.nn.functional.relu(h1 @ W2 + b2)
            h2 = torch.nn.functional.relu(z @ W3 + b3)
            x_recon = h2 @ W4 + b4
            loss = torch.nn.functional.mse_loss(x_recon, Xs_t)
            opt.zero_grad(); loss.backward(); opt.step()
        self._W1, self._b1 = W1.detach(), b1.detach()
        self._W2, self._b2 = W2.detach(), b2.detach()
        self._W3, self._b3 = W3.detach(), b3.detach()
        self._W4, self._b4 = W4.detach(), b4.detach()
        self._cw = ContextFeatureWeighting(); self._cw.fit(X_warmup)
        self._warmup_scores = self.decision_function(X_warmup)[:500].tolist()
        self._context_hist = [x for x in X_train[warmup_n:warmup_n+1000]]
        self._sample_count = warmup_n
        return self

    def _encode(self, X):
        Xt = torch.FloatTensor(X.astype(np.float32))
        with torch.no_grad():
            h1 = torch.nn.functional.relu(Xt @ self._W1 + self._b1)
            z = torch.nn.functional.relu(h1 @ self._W2 + self._b2)
        return z.numpy()

    def decision_function(self, X):
        Xf = X.astype(np.float32)
        Xp = self._encode(Xf)
        if not hasattr(self, '_latent_mean') or self._latent_mean is None:
            self._latent_mean = Xp.mean(axis=0)
            self._latent_std = Xp.std(axis=0) + 1e-6
        dists = np.sqrt(np.sum(((Xp - self._latent_mean)/self._latent_std)**2, axis=1))
        cw = self._cw.get_weights(Xf)
        cw_mean = cw.mean(axis=1) if cw.ndim == 2 else cw
        cw_mean = np.maximum(cw_mean, 0.1)
        return (dists * cw_mean).astype(np.float64)

    def score_one(self, x):
        xf = x.reshape(1,-1).astype(np.float32)
        z = self._encode(xf).flatten()
        iso = float(self._iso_score(z.reshape(1,-1))[0])
        cw = float(self._cw.get_weights(xf).mean()); cw = max(cw, 0.1)
        return float(iso * cw)

    def _iso_score(self, X_latent):
        if not hasattr(self, '_latent_mean') or self._latent_mean is None:
            self._latent_mean = X_latent.mean(axis=0)
            self._latent_std = X_latent.std(axis=0) + 1e-6
        return np.sqrt(np.sum(((X_latent - self._latent_mean)/self._latent_std)**2, axis=1)).astype(np.float64)

    def update_one(self, x, label=None):
        score = self.score_one(x)
        self._drift.update(score)
        self._context_hist.append(x.flatten())
        if len(self._context_hist) > self.context_history_size: self._context_hist.pop(0)


def inject_concept_drift(X_base, seed, drift_fraction=0.5, drift_magnitude=3.0):
    rng = np.random.RandomState(seed)
    n = len(X_base)
    drift_point = int(n * (1 - drift_fraction))
    X_drifted = X_base.copy()
    shift_cols = [0, 2, 6, 7, 15, 19]
    for col in shift_cols:
        col_std = X_base[:, col].std()
        shift = rng.randn(n - drift_point) * col_std * drift_magnitude
        X_drifted[drift_point:, col] += shift
    return X_drifted, drift_point


def evaluate_concept_drift(algo_cls, X_base, y_base, seed, label_budget=500, drift_fraction=0.5, drift_magnitude=3.0):
    X_drifted, drift_point = inject_concept_drift(X_base, seed, drift_fraction, drift_magnitude)
    n = len(X_base)
    y_drifted = y_base.copy()

    # Inject anomalies in the drifted portion
    rng = np.random.RandomState(seed)
    n_anom = int(n * 0.05)
    anom_idx = rng.choice(range(drift_point, n), n_anom, replace=False)
    y_drifted[anom_idx] = 1

    algo = algo_cls(seed=seed)
    algo.fit(X_base)

    # Track scores over time
    scores = []
    segment_scores = {'pre_drift': [], 'post_drift': []}
    for i in range(n):
        x = X_drifted[i]
        score = algo.score_one(x)
        scores.append(float(score))
        if i < drift_point:
            segment_scores['pre_drift'].append(score)
        else:
            segment_scores['post_drift'].append(score)
        if i >= 250:  # Start updating after warmup
            algo.update_one(x, label=y_drifted[i] if i in anom_idx else 0)

    # Compute metrics
    pr_curve, rc_curve, _ = precision_recall_curve(y_drifted, scores)
    auc_pr = auc(rc_curve, pr_curve) if len(rc_curve) > 1 else 0.0

    # Segment AUC-PR
    y_pre = y_drifted[:drift_point]; y_post = y_drifted[drift_point:]
    s_pre = scores[:drift_point]; s_post = scores[drift_point:]
    p1, r1, _ = precision_recall_curve(y_pre, s_pre)
    p2, r2, _ = precision_recall_curve(y_post, s_post)
    auc_pre = auc(r1, p1) if len(r1) > 1 else 0.0
    auc_post = auc(r2, p2) if len(r2) > 1 else 0.0

    # Score trajectory: early vs late post-drift
    post_scores = np.array(segment_scores['post_drift'])
    early_post = np.mean(post_scores[:len(post_scores)//2]) if len(post_scores) > 0 else 0
    late_post = np.mean(post_scores[len(post_scores)//2:]) if len(post_scores) > 0 else 0
    adaptation_ratio = late_post / (early_post + 1e-9)

    # Mean score in each segment
    mean_pre = np.mean(segment_scores['pre_drift']) if segment_scores['pre_drift'] else 0
    mean_post = np.mean(segment_scores['post_drift']) if segment_scores['post_drift'] else 0

    return {
        'auc_pr_overall': auc_pr,
        'auc_pr_pre': auc_pre,
        'auc_pr_post': auc_post,
        'mean_score_pre': mean_pre,
        'mean_score_post': mean_post,
        'early_post_score': early_post,
        'late_post_score': late_post,
        'adaptation_ratio': adaptation_ratio,
        'drift_point': drift_point,
    }


def main():
    print('='*60)
    print('Concept Drift Evaluation')
    print('='*60)

    # Load data
    print('\nLoading month 1 data...')
    df = clean(load_month(2024, 1))
    X_base = features(df)[:10000].astype(np.float64)
    y_base = np.zeros(len(X_base), dtype=np.int8)
    print(f'  Loaded {len(X_base)} samples')

    SEEDS = [42, 123, 456]
    drift_algos = {
        'MemStream': MemStream,
        'CA-DIF-EIA-Stream': CADIFEiaStream,
        'sHST-River': sHST_River,
        'Random': RandomBaseline,
    }

    all_results = {}
    for algo_name, algo_cls in drift_algos.items():
        print(f'\nEvaluating {algo_name}...')
        try:
            algo_results = {k: [] for k in ['auc_pr_overall', 'auc_pr_pre', 'auc_pr_post',
                                            'mean_score_pre', 'mean_score_post',
                                            'adaptation_ratio', 'early_post_score', 'late_post_score']}
            for seed in SEEDS:
                res = evaluate_concept_drift(algo_cls, X_base, y_base, seed)
                for k, v in res.items():
                    if k in algo_results:
                        algo_results[k].append(v)
            avg = {k: float(np.nanmean(v)) for k, v in algo_results.items()}
            all_results[algo_name] = avg
            print(f'  Overall AUC-PR: {avg["auc_pr_overall"]:.4f}')
            print(f'  Pre-drift AUC-PR: {avg["auc_pr_pre"]:.4f}')
            print(f'  Post-drift AUC-PR: {avg["auc_pr_post"]:.4f}')
            print(f'  Adaptation ratio: {avg["adaptation_ratio"]:.4f}  (<1 = adapting)')
            print(f'  Mean score pre: {avg["mean_score_pre"]:.4f}')
            print(f'  Mean score post: {avg["mean_score_post"]:.4f}')
        except Exception as e:
            print(f'  FAILED: {e}')

    # Save results
    drift_df = pd.DataFrame(all_results).T
    drift_df.to_csv(OUT_DIR / 'concept_drift_results_v8.csv')
    print(f'\nSaved concept_drift_results_v8.csv')

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = list(all_results.keys())
    colors = {'MemStream': '#2980b9', 'CA-DIF-EIA-Stream': '#c0392b', 'sHST-River': '#3498db', 'Random': '#bdc3c7'}

    # AUC-PR comparison
    ax = axes[0]
    overall = [all_results[n]['auc_pr_overall'] for n in names]
    pre = [all_results[n]['auc_pr_pre'] for n in names]
    post = [all_results[n]['auc_pr_post'] for n in names]
    x = np.arange(len(names))
    w = 0.25
    ax.bar(x - w, overall, w, label='Overall', color=['#2ecc71'])
    ax.bar(x, pre, w, label='Pre-drift', color=['#3498db'])
    ax.bar(x + w, post, w, label='Post-drift', color=['#e74c3c'])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('AUC-PR'); ax.set_title('AUC-PR by Segment'); ax.legend(); ax.grid(axis='y', alpha=0.3)

    # Mean scores
    ax = axes[1]
    pre_s = [all_results[n]['mean_score_pre'] for n in names]
    post_s = [all_results[n]['mean_score_post'] for n in names]
    ax.bar(x - w/2, pre_s, w, label='Pre-drift', color='#3498db')
    ax.bar(x + w/2, post_s, w, label='Post-drift', color='#e74c3c')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Mean Score'); ax.set_title('Mean Anomaly Score by Segment'); ax.legend(); ax.grid(axis='y', alpha=0.3)

    # Adaptation ratio
    ax = axes[2]
    ratios = [all_results[n]['adaptation_ratio'] for n in names]
    bar_colors = ['#27ae60' if r < 1 else '#e74c3c' for r in ratios]
    ax.bar(names, ratios, color=bar_colors)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='No adaptation')
    ax.set_ylabel('Adaptation Ratio'); ax.set_title('Adaptation Ratio (late/early post-drift)\n<1 = adapting to new distribution')
    ax.set_xticklabels(names, rotation=30, ha='right'); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig_concept_drift_v8.png', dpi=150)
    plt.close()
    print(f'Saved fig_concept_drift_v8.png')

    print('\n=== Summary ===')
    print('Adaptation ratio: <1 means scores decreased over time (memory adapting)')
    for name in names:
        r = all_results[name]['adaptation_ratio']
        status = 'ADAPTING' if r < 0.95 else 'STABLE' if r < 1.05 else 'SCORES RISING'
        print(f'  {name:25s}: ratio={r:.4f} -> {status}')


if __name__ == '__main__':
    main()
