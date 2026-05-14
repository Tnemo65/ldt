#!/usr/bin/env python3
"""
Benchmark v5 — Streaming Anomaly Detection (Optimized for Speed)
============================================================
- 6 months only (Jan-Jun 2024)
- 3 seeds
- 3 difficulties
- Subsampled: 10k train, 10k test
- Sequential execution (no multiprocessing crash risk)
- Checkpoint save every 50 jobs
"""

import os, sys, json, time, gc
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (
    auc, precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

warnings = __import__('warnings')
warnings.filterwarnings('ignore')

DATA_DIR = Path(r'C:\proj\ldt\data\raw')
OUT_DIR  = Path(r'C:\proj\ldt\results\v5')
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS      = [1, 2, 3, 4, 5, 6]
SEEDS       = [42, 123, 456]
DIFFICULTIES = ['easy', 'medium', 'hard']
ANOMALY_PARAMS = {
    'easy':   {'meter_mult': (10, 20), 'speed': (50, 95),   'pax_fare': (40, 70),  'crawl_dur': (90, 180)},
    'medium': {'meter_mult': (4, 8),   'speed': (30, 60),  'pax_fare': (15, 30),  'crawl_dur': (40, 80)},
    'hard':   {'meter_mult': (1.5, 3),'speed': (20, 40),  'pax_fare': (8, 15),   'crawl_dur': (20, 35)},
}
METRICS = ['AUC_PR', 'AUC_ROC', 'F1', 'Precision', 'Recall', 'FPR']
TRAIN_N = 10000
TEST_N  = 10000

GPU_AVAILABLE = False
DEVICE = 'cpu'
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = 'cuda' if GPU_AVAILABLE else 'cpu'
    if GPU_AVAILABLE:
        torch.set_num_threads(1)
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except Exception:
    DEVICE = 'cpu'


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
    n = len(df)
    X = np.zeros((n, 25), dtype=np.float32)
    pickup = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    hour   = pickup.dt.hour.fillna(12).astype(np.int32).values
    dow    = pickup.dt.dayofweek.fillna(0).astype(np.int32).values
    dist  = df['trip_distance'].fillna(0).values.astype(np.float32)
    dur   = df['dur_min'].fillna(1).values.astype(np.float32)
    fare  = df['fare_amount'].fillna(0).values.astype(np.float32)
    pax   = df['passenger_count'].fillna(1).values.astype(np.float32)
    spd   = df['speed_mph'].fillna(0).values.astype(np.float32)
    total = df['total_amt'].fillna(0).values.astype(np.float32)
    eps = np.float32(0.01)
    X[:, 0] = dist; X[:, 1] = dur; X[:, 2] = fare; X[:, 3] = pax
    X[:, 4] = total; X[:, 5] = spd
    X[:, 6] = fare / np.maximum(dist, eps); X[:, 7] = fare / np.maximum(dur, eps)
    X[:, 8] = fare / np.maximum(pax, eps)
    X[:, 9]  = hour.astype(np.float32); X[:, 10] = dow.astype(np.float32)
    X[:, 11] = (dow >= 5).astype(np.float32)
    X[:, 12] = (((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 20))).astype(np.float32)
    X[:, 13] = ((hour >= 20) | (hour <= 6)).astype(np.float32)
    X[:, 14] = pickup.dt.month.fillna(1).astype(np.float32)
    X[:, 15] = X[:, 6] / np.float32(2.5); X[:, 16] = X[:, 7] / np.float32(0.67)
    X[:, 17] = spd / np.float32(12.0); X[:, 18] = pax / np.maximum(dist, eps)
    X[:, 19] = fare * dist; X[:, 20] = dur / np.maximum(dist, eps)
    X[:, 21] = np.sin(np.float32(2*np.pi) * hour / np.float32(24)).astype(np.float32)
    X[:, 22] = np.cos(np.float32(2*np.pi) * hour / np.float32(24)).astype(np.float32)
    X[:, 23] = np.sin(np.float32(2*np.pi) * dow  / np.float32(7)).astype(np.float32)
    X[:, 24] = np.cos(np.float32(2*np.pi) * dow  / np.float32(7)).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)


# =============================================================================
# ANOMALY INJECTION
# =============================================================================

def inject(df, n_per, diff, seed):
    rng = np.random.RandomState(seed)
    df  = df.copy().reset_index(drop=True)
    labels = np.zeros(len(df), dtype=np.int32)
    p = ANOMALY_PARAMS[diff]
    for sname in ['meter_tampering', 'gps_spoofing', 'passenger_anomaly', 'slow_crawl', 'combined_subtle']:
        recs = []
        for i in range(n_per):
            r = df.iloc[i % len(df)].copy()
            if sname == 'meter_tampering':
                r['fare_amount'] = r['trip_distance'] * 2.5 * rng.uniform(*p['meter_mult'])
            elif sname == 'gps_spoofing':
                sp = rng.uniform(*p['speed'])
                r['trip_distance'] = r['dur_min'] * sp / 60.0
                r['fare_amount']   = r['trip_distance'] * 2.5
            elif sname == 'passenger_anomaly':
                r['trip_distance'] = rng.uniform(0.1, 1.5)
                r['fare_amount']   = rng.uniform(*p['pax_fare'])
            elif sname == 'slow_crawl':
                r['dur_min']       = rng.uniform(*p['crawl_dur'])
                r['fare_amount']   = rng.uniform(*p['pax_fare'])
                r['trip_distance'] = rng.uniform(0.5, 3.0)
            elif sname == 'combined_subtle':
                mult = rng.uniform(1.2, 2.0)
                r['fare_amount']   = r['fare_amount'] * mult
                r['trip_distance'] = r['trip_distance'] * rng.uniform(0.8, 1.2)
                r['dur_min']       = r['dur_min'] * rng.uniform(0.9, 1.1)
            recs.append(r)
        anom  = pd.DataFrame(recs)
        df    = pd.concat([df, anom], ignore_index=True)
        labels = np.append(labels, np.ones(len(recs), dtype=np.int32))
    return df, labels


# =============================================================================
# ALGORITHMS
# =============================================================================

class SklearnIF:
    name = 'sklearn_IF'
    supports_streaming = False
    def __init__(self, seed=42): self.seed = seed
    def fit(self, X):
        self.model_ = IsolationForest(n_estimators=200, contamination=0.05, random_state=self.seed, n_jobs=-1)
        self.model_.fit(X)
        raw = -self.model_.score_samples(X)
        self.thresh_ = float(np.percentile(raw, 97))
    def decision_function(self, X): return -self.model_.score_samples(X)
    def predict(self, X): return np.where(self.decision_function(X) > self.thresh_, -1, 1)

class SklearnOCSVM:
    name = 'sklearn_OCSVM'
    supports_streaming = False
    def __init__(self, seed=42): self.seed = seed
    def fit(self, X):
        n = min(5000, len(X))
        idx = np.random.RandomState(self.seed).choice(len(X), n, replace=False)
        self.model_ = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
        self.model_.fit(X[idx])
        raw = -self.model_.decision_function(X[idx])
        self.thresh_ = float(np.percentile(raw, 97))
    def decision_function(self, X): return -self.model_.decision_function(X)
    def predict(self, X): return np.where(self.decision_function(X) > self.thresh_, -1, 1)

class SklearnLOF:
    name = 'sklearn_LOF'
    supports_streaming = False
    def __init__(self, seed=42): self.seed = seed
    def fit(self, X):
        n = min(10000, len(X))
        idx = np.random.RandomState(self.seed).choice(len(X), n, replace=False)
        self.model_ = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=4)
        self.model_.fit(X[idx])
        raw = -self.model_.decision_function(X[idx])
        self.thresh_ = float(np.percentile(raw, 97))
    def decision_function(self, X): return -self.model_.decision_function(X)
    def predict(self, X): return np.where(self.decision_function(X) > self.thresh_, -1, 1)

class LSTMAE:
    name = 'LSTM-AE'
    supports_streaming = False
    def __init__(self, seed=42):
        self.seed = seed; self.hidden_dim = 64; self.threshold = 1.0
        self.scaler_ = None; self.model_ = None; self._device = DEVICE; self._n_features = 0
    def fit(self, X):
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)
        self._n_features = Xs.shape[1]
        self._train(Xs)
    def _train(self, Xs):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        seq = torch.FloatTensor(Xs.reshape(-1, 1, self._n_features))
        class AE(nn.Module):
            def __init__(self, d, h):
                super().__init__()
                self.enc = nn.LSTM(d, h, batch_first=True)
                self.dec = nn.LSTM(h, d, batch_first=True)
            def forward(self, x):
                _, (h, _) = self.enc(x)
                dec, _ = self.dec(h.permute(1, 0, 2).repeat(1, 1, 1))
                return dec
        self.model_ = AE(self._n_features, self.hidden_dim)
        if self._device == 'cuda': self.model_ = self.model_.cuda()
        opt = torch.optim.Adam(self.model_.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        ds = TensorDataset(seq, seq)
        dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0, pin_memory=False)
        for epoch in range(10):
            for bx, by in dl:
                if self._device == 'cuda': bx, by = bx.cuda(), by.cuda()
                pred = self.model_(bx); loss = loss_fn(pred, by)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            if self._device == 'cuda': preds = self.model_(seq.cuda()).cpu().numpy()
            else: preds = self.model_(seq).numpy()
        errors = np.mean(np.abs(seq.numpy() - preds), axis=(1, 2))
        self.threshold = float(np.percentile(errors, 97))
    def decision_function(self, X):
        if self.model_ is None or self.scaler_ is None: return np.full(len(X), 0.5)
        Xs = self.scaler_.transform(X)
        seq = torch.FloatTensor(Xs.reshape(-1, 1, self._n_features))
        with torch.no_grad():
            if self._device == 'cuda': preds = self.model_(seq.cuda()).cpu().numpy()
            else: preds = self.model_(seq).numpy()
        return np.mean(np.abs(seq.numpy() - preds), axis=(1, 2)).astype(np.float64)
    def predict(self, X): return np.where(self.decision_function(X) > self.threshold, -1, 1)

class CADIFEia:
    name = 'CA-DIF-EIA'
    supports_streaming = False
    def __init__(self, seed=42): self.seed = seed
    def fit(self, X):
        self._n_features = X.shape[1]
        self._rng = np.random.RandomState(self.seed)
        W1 = self._rng.randn(self._n_features, 32).astype(np.float32) * 0.1
        b1 = self._rng.randn(32).astype(np.float32) * 0.1
        W2 = self._rng.randn(32, 16).astype(np.float32) * 0.1
        b2 = self._rng.randn(16).astype(np.float32) * 0.1
        def proj(Xp):
            return np.maximum(np.maximum(Xp @ W1 + b1, 0) @ W2 + b2, 0)
        Xp = proj(X.astype(np.float32))
        self.if_ = IsolationForest(n_estimators=200, contamination=0.05, random_state=self.seed, n_jobs=-1)
        self.if_.fit(Xp)
        raw = -self.if_.score_samples(Xp)
        self.thresh_ = float(np.percentile(raw, 97))
        self._W1 = W1; self._b1 = b1; self._W2 = W2; self._b2 = b2
        n = min(3000, len(X))
        idx = self._rng.choice(len(X), n, replace=False)
        scores = -self.if_.score_samples(proj(X[idx]))
        weights = np.zeros(self._n_features, dtype=np.float32)
        for f in range(self._n_features):
            fvar = X[idx, f].std()
            if fvar > 1e-6:
                corr = abs(np.corrcoef(X[idx, f], scores)[0, 1])
                weights[f] = corr if not np.isnan(corr) else 0.1
            else: weights[f] = 0.1
        weights = weights / (weights.max() + 1e-6) * 2.0 + 0.5
        self._w = float(weights.mean())
    def decision_function(self, X):
        Xf = X.astype(np.float32)
        Xp = np.maximum(np.maximum(Xf @ self._W1 + self._b1, 0) @ self._W2 + self._b2, 0)
        return (-self.if_.score_samples(Xp) * self._w).astype(np.float64)
    def predict(self, X): return np.where(self.decision_function(X) > self.thresh_, -1, 1)

class sHST_River:
    name = 'sHST-River'
    supports_streaming = True
    def __init__(self, seed=42):
        self.seed = seed; self._rng = np.random.RandomState(seed)
        self.n_trees = 25; self.depth = 8
        self.split_pts = self._rng.uniform(-3, 3, size=(self.n_trees, self.depth)).astype(np.float32)
        self.tree_f = self._rng.randint(0, 25, size=(self.n_trees, self.depth))
        self.window = []
    def fit(self, X):
        for i in range(min(200, len(X))):
            self.update_one(X[i].astype(np.float64))
    def score_one(self, x):
        if len(self.window) < 5: return 0.5
        xf = x.astype(np.float64)
        scores = np.zeros(self.n_trees)
        for t in range(self.n_trees):
            tf = self.tree_f[t]; sp = self.split_pts[t]
            depths = np.where(xf[tf] > sp, 1, 0)
            scores[t] = np.mean(depths)
        return float(np.mean(scores))
    def decision_function(self, X):
        return np.array([self.score_one(x) for x in X]).astype(np.float64)
    def update_one(self, x, label=None):
        xf = x.astype(np.float64)
        if len(self.window) >= 250: self.window.pop(0)
        self.window.append(xf)
        self._thresh = float(np.percentile([self.score_one(w) for w in self.window[-250:]], 95)) if len(self.window) >= 10 else 0.5
    def predict(self, X):
        d = self.decision_function(X)
        t = getattr(self, '_thresh', 0.5)
        return np.where(d > t, -1, 1)

class MemStream:
    name = 'MemStream'
    supports_streaming = True
    def __init__(self, seed=42):
        self.seed = seed; self._rng = np.random.RandomState(seed)
        self.memory = []; self.k = 10; self._buf = []
    def fit(self, X):
        for i in range(min(500, len(X))): self.update_one(X[i].astype(np.float64))
    def score_one(self, x):
        if len(self.memory) < self.k: return 0.5
        mem = np.array(self.memory, dtype=np.float64)
        dists = np.sort(np.linalg.norm(mem - x.astype(np.float64), axis=1))
        return float(dists[:self.k].mean())
    def update_one(self, x, label=None):
        xf = x.astype(np.float64)
        if len(self.memory) < 500: self.memory.append(xf)
        s = self.score_one(x)
        self._buf.append(s)
        if len(self._buf) > 500: self._buf.pop(0)
    def decision_function(self, X):
        return np.array([self.score_one(x) for x in X]).astype(np.float64)
    def predict(self, X):
        d = self.decision_function(X)
        t = float(np.percentile(self._buf, 95)) if self._buf else 0.5
        return np.where(d > t, -1, 1)

class IForestASD:
    name = 'IForestASD'
    supports_streaming = True
    def __init__(self, seed=42):
        self.seed = seed; self._rng = np.random.RandomState(seed)
        self.window_size = 500; self.n_trees = 50; self.max_samples = 256
        self.trees = []; self.buffer = []
    def fit(self, X):
        for i in range(min(self.window_size, len(X))): self._partial_fit(X[i])
    def _partial_fit(self, x):
        xf = x.reshape(-1).astype(np.float32)
        if len(self.buffer) >= self.window_size: self.buffer.pop(0)
        self.buffer.append(xf)
        if len(self.buffer) < self.max_samples: return
        buf = np.array(list(self.buffer)[-self.max_samples:], dtype=np.float32)
        self.trees = []
        for _ in range(self.n_trees):
            idx = self._rng.choice(len(buf), min(self.max_samples, len(buf)), replace=False)
            s = buf[idx]; fd = s.shape[1]; fi = self._rng.randint(0, fd)
            lo, hi = s[:, fi].min(), s[:, fi].max()
            sp = self._rng.uniform(lo, hi + 1e-8)
            self.trees.append((fi, float(sp)))
    def score_one(self, x):
        if not self.trees: return 0.5
        xf = x.astype(np.float32)
        depth_sum = sum(0.0 if xf[fi] < sp else 1.0 for fi, sp in self.trees)
        return float(depth_sum / len(self.trees))
    def decision_function(self, X):
        return np.array([self.score_one(x) for x in X]).astype(np.float64)
    def update_one(self, x, label=None): self._partial_fit(x)
    def predict(self, X):
        d = self.decision_function(X)
        return np.where(d > 0.5, -1, 1)

class CADIFEiaStream:
    name = 'CA-DIF-EIA (streaming)'
    supports_streaming = True
    def __init__(self, seed=42):
        self.seed = seed; self._rng = np.random.RandomState(seed)
        self._drift = _ADWIN(delta=0.002)
    def fit(self, X):
        self._n_features = X.shape[1]
        W1 = self._rng.randn(self._n_features, 32).astype(np.float32) * 0.1
        b1 = self._rng.randn(32).astype(np.float32) * 0.1
        W2 = self._rng.randn(32, 16).astype(np.float32) * 0.1
        b2 = self._rng.randn(16).astype(np.float32) * 0.1
        self._W1 = W1; self._b1 = b1; self._W2 = W2; self._b2 = b2
        warmup = min(int(len(X) * 0.2), 3000)
        Xp = self._proj(X[:warmup])
        self.if_ = IsolationForest(n_estimators=200, contamination=0.05, random_state=self.seed, n_jobs=-1)
        self.if_.fit(Xp)
        self._w = 1.5
    def _proj(self, X):
        Xf = X.astype(np.float32)
        return np.maximum(np.maximum(Xf @ self._W1 + self._b1, 0) @ self._W2 + self._b2, 0)
    def score_one(self, x):
        xf = x.reshape(1, -1).astype(np.float32)
        return float(-self.if_.score_samples(self._proj(xf))[0] * self._w)
    def update_one(self, x, label=None):
        xf = x.reshape(1, -1).astype(np.float32)
        score = float(-self.if_.score_samples(self._proj(xf))[0])
        if self._drift.update(score):
            Xp = self._proj(xf)
            self.if_ = IsolationForest(n_estimators=200, contamination=0.05, random_state=self.seed, n_jobs=-1)
            self.if_.fit(Xp)
    def decision_function(self, X):
        Xp = self._proj(X.astype(np.float32))
        return (-self.if_.score_samples(Xp) * self._w).astype(np.float64)
    def predict(self, X):
        d = self.decision_function(X)
        return np.where(d > np.percentile(d, 95), -1, 1)

class _ADWIN:
    def __init__(self, delta=0.002, size=500):
        self.delta = delta; self.size = size; self._w = []
    def update(self, v):
        self._w.append(v)
        if len(self._w) > self.size: self._w.pop(0)
        if len(self._w) < 100: return False
        mid = len(self._w) // 2
        w1, w2 = np.array(self._w[:mid]), np.array(self._w[mid:])
        m1, m2 = w1.mean(), w2.mean()
        n1, n2 = len(w1), len(w2)
        v1, v2 = w1.var() + 1e-9, w2.var() + 1e-9
        eps = np.sqrt((1/(2*n1)) * np.log(4*len(self._w)/self.delta) * (v1 + v2))
        if abs(m1 - m2) > eps:
            self._w = list(w2)
            return True
        return False


ALGOS = [SklearnIF, SklearnOCSVM, SklearnLOF, LSTMAE, CADIFEia, sHST_River, MemStream, IForestASD, CADIFEiaStream]
ALGO_NAMES = [a.name for a in ALGOS]
ALGO_MAP = {a.name: a for a in ALGOS}


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(algo_cls, X_train, X_test, y_test, seed):
    algo = algo_cls(seed=seed)
    t0 = time.perf_counter()
    algo.fit(X_train)
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = algo.decision_function(X_test).astype(np.float64)
    t_score = time.perf_counter() - t0

    if len(scores) < 5 or np.sum(y_test) == 0:
        return {m: 0.0 for m in METRICS + ['train_ms', 'score_ms', 'labels_consumed']}

    thresholds = np.percentile(scores, np.arange(80, 100, 0.5))
    best_f1, best_t = 0.0, float(np.percentile(scores, 97))
    for t in thresholds:
        preds = (scores >= t).astype(int)
        if preds.sum() == 0: continue
        f1 = f1_score(y_test, preds, zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t

    preds = (scores >= best_t).astype(int)
    pr_curve, rc_curve, _ = precision_recall_curve(y_test, scores)
    auc_pr  = auc(rc_curve, pr_curve) if len(rc_curve) > 1 else 0.0
    fpr_arr, tpr_arr, _ = roc_curve(y_test, scores)
    auc_roc = auc(fpr_arr, tpr_arr) if len(fpr_arr) > 1 else 0.5
    f1  = f1_score(y_test, preds, zero_division=0)
    prc = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    except: tp = fp = tn = fn = 0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        'AUC_PR': auc_pr, 'AUC_ROC': auc_roc, 'F1': f1, 'Precision': prc,
        'Recall': rec, 'FPR': fpr_val, 'TP': int(tp), 'FP': int(fp),
        'TN': int(tn), 'FN': int(fn), 'optimal_threshold': best_t,
        'train_ms': t_train * 1000, 'score_ms': t_score * 1000, 'labels_consumed': 0,
    }


# =============================================================================
# MAIN
# =============================================================================

def run():
    print('=' * 60)
    print('BENCHMARK v5 (6 months, optimized)')
    print(f'  Algorithms: {ALGO_NAMES}')
    print(f'  Seeds: {SEEDS}, Difficulties: {DIFFICULTIES}')
    print(f'  Train: {TRAIN_N}/sample, Test: {TEST_N}/sample')
    print(f'  GPU: {"ON" if GPU_AVAILABLE else "OFF"}')
    print('=' * 60)

    # Load 6 months
    print('\n[1/6] Loading months...')
    monthly, monthly_X = [], []
    for m in MONTHS:
        df = clean(load_month(2024, m))
        X  = features(df)
        monthly.append(df); monthly_X.append(X.astype(np.float32))
        print(f'  Month {m:02d}: {len(df):,} records')

    # Build jobs
    print('\n[2/6] Building jobs...')
    jobs = []
    for fold_idx, test_month in enumerate(MONTHS[1:], 1):
        train_X = np.vstack([monthly_X[i] for i in range(fold_idx)])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_X).astype(np.float32)
        if len(X_train) > TRAIN_N:
            idx = np.random.RandomState(42).choice(len(X_train), TRAIN_N, replace=False)
            X_train = X_train[idx]

        X_test_raw = monthly_X[test_month - 1]
        for diff in DIFFICULTIES:
            seed_idx = (fold_idx * 3 + DIFFICULTIES.index(diff)) % len(SEEDS)
            df_inj, y_labels = inject(monthly[test_month - 1], 500, diff, SEEDS[seed_idx])
            X_inj = scaler.transform(features(df_inj)).astype(np.float32)
            y_labels = np.array(y_labels, dtype=np.int32)
            if len(X_inj) > TEST_N:
                idx = np.random.RandomState(SEEDS[seed_idx]).choice(len(X_inj), TEST_N, replace=False)
                X_inj = X_inj[idx]; y_labels = y_labels[idx]

            for algo in ALGOS:
                for seed in SEEDS:
                    jobs.append({
                        'fold': fold_idx, 'test_month': test_month,
                        'diff': diff, 'algo_name': algo.name,
                        'seed': seed, 'X_train': X_train,
                        'X_test': X_inj, 'y_test': y_labels,
                    })
    print(f'  Total jobs: {len(jobs)}')

    # Run
    print(f'\n[3/6] Running {len(jobs)} jobs...')
    t0 = time.perf_counter()
    results = []
    for i, job in enumerate(jobs):
        try:
            res = evaluate(ALGO_MAP[job['algo_name']], job['X_train'], job['X_test'], job['y_test'], job['seed'])
            res.update({
                'fold': job['fold'], 'month': job['test_month'],
                'difficulty': job['diff'], 'algorithm': job['algo_name'],
                'seed': job['seed'], 'error': '',
            })
        except Exception as e:
            res = {m: float('nan') for m in METRICS + ['train_ms', 'score_ms', 'labels_consumed']}
            res.update({'fold': job['fold'], 'month': job['test_month'],
                        'difficulty': job['diff'], 'algorithm': job['algo_name'],
                        'seed': job['seed'], 'error': str(e)[:80]})
        results.append(res)
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            remain = (len(jobs) - i - 1) / rate / 60
            print(f'  {i+1}/{len(jobs)} ({rate:.1f}/s, ~{remain:.1f}m left)')
            pd.DataFrame(results).to_csv(OUT_DIR / 'checkpoint_v5.csv', index=False)
        gc.collect()

    bench_df = pd.DataFrame(results)
    bench_df.to_csv(OUT_DIR / 'benchmark_results_v5.csv', index=False)
    t_done = time.perf_counter() - t0
    print(f'\n  Done in {t_done/60:.1f} min')

    # Stats
    print('\n[4/6] Statistical analysis...')
    for group_name, algos in [('Batch', ['sklearn_IF', 'sklearn_OCSVM', 'sklearn_LOF', 'LSTM-AE', 'CA-DIF-EIA']),
                               ('Streaming', ['sHST-River', 'MemStream', 'IForestASD', 'CA-DIF-EIA (streaming)'])]:
        gdf = bench_df[bench_df['algorithm'].isin(algos)]
        if gdf.empty: continue
        print(f'\n  [{group_name}]')
        top = gdf.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False)
        for a, v in top.items(): print(f'    {a:25s}: {v:.4f}')

    # Plots
    print('\n[5/6] Generating plots...')
    _plot_all(bench_df, OUT_DIR)

    # Env
    env = {'version': '5.0', 'timestamp': datetime.now().isoformat(),
            'python': sys.version, 'gpu': GPU_AVAILABLE,
            'gpu_device': torch.cuda.get_device_name(0) if GPU_AVAILABLE else None,
            'total_jobs': len(jobs), 'runtime_min': t_done/60,
            'batch_algos': ['sklearn_IF', 'sklearn_OCSVM', 'sklearn_LOF', 'LSTM-AE', 'CA-DIF-EIA'],
            'stream_algos': ['sHST-River', 'MemStream', 'IForestASD', 'CA-DIF-EIA (streaming)']}
    with open(OUT_DIR / 'environment.json', 'w') as f: json.dump(env, f, indent=2)

    print('\n' + '=' * 60)
    print('BENCHMARK v5 COMPLETE')
    print(f'  {len(jobs)} jobs in {t_done/60:.1f} min')
    print('=' * 60)
    return bench_df


# =============================================================================
# PLOTS
# =============================================================================

COLORS = {
    'sklearn_IF': '#95a5a6', 'sklearn_OCSVM': '#bdc3c7', 'sklearn_LOF': '#7f8c8d',
    'sHST-River': '#3498db', 'MemStream': '#2980b9', 'IForestASD': '#e67e22',
    'LSTM-AE': '#9b59b6', 'CA-DIF-EIA': '#e74c3c', 'CA-DIF-EIA (streaming)': '#c0392b',
}

def _plot_all(df, out):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0, 0]
    algos = df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index
    bp = ax.boxplot([df[df['algorithm']==a]['AUC_PR'].dropna().values for a in algos], patch_artist=True)
    for p, a in zip(bp['boxes'], algos): p.set_facecolor(COLORS.get(a, '#333')); p.set_alpha(0.7)
    ax.set_xticklabels(algos, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('AUC-PR'); ax.set_title('AUC-PR Distribution'); ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 1]
    colors_d = {'easy': '#27ae60', 'medium': '#f39c12', 'hard': '#c0392b'}
    top = df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).head(6).index
    x = np.arange(len(top))
    for di, diff in enumerate(['easy', 'medium', 'hard']):
        d = df[df['difficulty']==diff].groupby('algorithm')['AUC_PR'].mean()
        d = d.reindex(top)
        ax.bar(x + di*0.25, d.values, 0.22, label=diff.capitalize(), color=colors_d[diff], alpha=0.8)
    ax.set_xticks(x + 0.25); ax.set_xticklabels(top, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('AUC-PR'); ax.set_title('AUC-PR by Difficulty'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 2]
    f1_mean = df.groupby('algorithm')['F1'].mean().sort_values(ascending=True)
    ax.barh(range(len(f1_mean)), f1_mean.values, color=[COLORS.get(a,'#333') for a in f1_mean.index])
    ax.set_yticks(range(len(f1_mean))); ax.set_yticklabels(f1_mean.index, fontsize=8)
    ax.set_xlabel('F1'); ax.set_title('Mean F1'); ax.grid(axis='x', alpha=0.3)

    ax = axes[1, 0]
    times = df.groupby('algorithm')['score_ms'].mean().sort_values(ascending=True)
    ax.barh(range(len(times)), times.values, color=[COLORS.get(a,'#333') for a in times.index])
    ax.set_yticks(range(len(times))); ax.set_yticklabels(times.index, fontsize=8)
    ax.set_xlabel('Score time (ms)'); ax.set_title('Mean Scoring Time'); ax.grid(axis='x', alpha=0.3)

    ax = axes[1, 1]
    for algo in df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).head(3).index:
        mdata = df[df['algorithm']==algo].groupby('fold')['AUC_PR'].mean()
        ax.plot(mdata.index, mdata.values, 'o-', label=algo, color=COLORS.get(algo,'#333'), linewidth=2, markersize=5)
    ax.set_xlabel('Fold'); ax.set_ylabel('AUC-PR'); ax.set_title('AUC-PR Over Folds'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    for algo in df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).head(3).index:
        mdata = df[df['algorithm']==algo].groupby('difficulty')['AUC_PR'].mean()
        ax.plot(mdata.index, mdata.values, 'o-', label=algo, color=COLORS.get(algo,'#333'), linewidth=2, markersize=6)
    ax.set_xlabel('Difficulty'); ax.set_ylabel('AUC-PR'); ax.set_title('AUC-PR by Difficulty'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle('Benchmark v5 Overview', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out / 'fig_overview_v5.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig_overview_v5.png')


if __name__ == '__main__':
    run()
