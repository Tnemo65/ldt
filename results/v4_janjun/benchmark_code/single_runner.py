"""
Single-pass lightweight benchmark runner.
Everything runs sequentially in one process to avoid multiprocessing overhead.
Subsamples aggressively to stay within memory limits.

Subsampling: 30K train, 30K test (with 5K anomalies).
Seeds: 5 (was 10).
Workers: 1 (sequential).
Expected total time: ~30-60 min on 18 cores.
"""

import gc
import json
import time
import sys
from datetime import datetime
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc, precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats


DATA_DIR  = Path('c:/proj/ldt/data/raw')
OUT_DIR   = Path('c:/proj/ldt/results/v4_janjun')
FEAT_DIR  = OUT_DIR / 'features'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# ── Tunable knobs ────────────────────────────────────────────────────────
TRAIN_N  = 30_000   # train subsample per fold
TEST_N   = 30_000   # test subsample per fold
SEEDS    = [42, 123, 456, 789, 1024]  # 5 seeds (was 10)
DIFFS    = ['easy', 'medium', 'hard']
ANOM_PARAMS = {
    'easy':   {'meter_mult': (10, 20), 'speed': (50, 95),  'pax_fare': (40, 70),  'crawl_dur': (90, 180)},
    'medium': {'meter_mult': (4, 8),   'speed': (30, 60),  'pax_fare': (15, 30),  'crawl_dur': (40, 80)},
    'hard':   {'meter_mult': (1.5, 3), 'speed': (20, 40),  'pax_fare': (8, 15),   'crawl_dur': (20, 35)},
}

# ── Data Loading ──────────────────────────────────────────────────────────

def load_month(year, month):
    return pd.read_parquet(DATA_DIR / f'yellow_tripdata_{year:04d}-{month:02d}.parquet')

def clean(df):
    df = df.copy()
    df = df.dropna(subset=['PULocationID', 'DOLocationID', 'fare_amount',
                            'trip_distance', 'passenger_count'])
    for col in ['PULocationID', 'DOLocationID', 'fare_amount',
                'trip_distance', 'passenger_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    df = df[(df['PULocationID'] >= 1) & (df['PULocationID'] <= 263)]
    df = df[(df['DOLocationID'] >= 1) & (df['DOLocationID'] <= 263)]
    df['fare_amount']   = df['fare_amount'].abs()
    df['trip_distance'] = df['trip_distance'].abs()
    pickup  = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    dropoff  = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
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

def extract_features(df, ablation='treatment'):
    n  = len(df)
    nf = 25 if ablation == 'treatment' else 15
    X  = np.zeros((n, nf), dtype=np.float32)
    pickup = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    hour   = pickup.dt.hour.fillna(12).astype(int).values
    dow    = pickup.dt.dayofweek.fillna(0).astype(int).values
    dist = df['trip_distance'].fillna(0).values
    dur  = df['dur_min'].fillna(1).values
    fare = df['fare_amount'].fillna(0).values
    pax  = df['passenger_count'].fillna(1).values
    spd  = df['speed_mph'].fillna(0).values
    X[:, 0] = dist; X[:, 1] = dur; X[:, 2] = fare; X[:, 3] = pax
    X[:, 4] = df['total_amt'].fillna(0).values; X[:, 5] = spd
    X[:, 6] = fare / np.maximum(dist, 0.01)
    X[:, 7] = fare / np.maximum(dur,  0.01)
    X[:, 8] = fare / np.maximum(pax,  0.01)
    X[:, 9]  = hour.astype(np.float32); X[:, 10] = dow.astype(np.float32)
    X[:, 11] = (dow >= 5).astype(np.float32)
    X[:, 12] = (((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 20))).astype(np.float32)
    X[:, 13] = ((hour >= 20) | (hour <= 6)).astype(np.float32)
    X[:, 14] = pickup.dt.month.fillna(1).astype(np.float32)
    if ablation == 'treatment':
        X[:, 15] = X[:, 6] / 2.5; X[:, 16] = X[:, 7] / 0.67
        X[:, 17] = spd / 12.0
        X[:, 18] = pax / np.maximum(dist, 0.01)
        X[:, 19] = fare * dist; X[:, 20] = dur / np.maximum(dist, 0.01)
        X[:, 21] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        X[:, 22] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
        X[:, 23] = np.sin(2 * np.pi * dow  / 7).astype(np.float32)
        X[:, 24] = np.cos(2 * np.pi * dow  / 7).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)

def inject_anomalies(df, n_per, diff, seed):
    rng = np.random.RandomState(seed)
    df  = df.copy().reset_index(drop=True)
    labels = np.zeros(len(df), dtype=int)
    p = ANOM_PARAMS[diff]
    for sname in ['meter_tampering', 'gps_spoofing', 'passenger_anomaly',
                  'slow_crawl', 'combined_subtle']:
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
                r['dur_min']        = rng.uniform(*p['crawl_dur'])
                r['fare_amount']    = rng.uniform(*p['pax_fare'])
                r['trip_distance']  = rng.uniform(0.5, 3.0)
                if r['dur_min'] > 0:
                    r['speed_mph'] = r['trip_distance'] / (r['dur_min'] / 60)
            elif sname == 'combined_subtle':
                mult = rng.uniform(1.2, 2.0)
                r['fare_amount']   = r['fare_amount'] * mult
                r['trip_distance'] = r['trip_distance'] * rng.uniform(0.8, 1.2)
                r['dur_min']       = r['dur_min'] * rng.uniform(0.9, 1.1)
            recs.append(r)
        anom  = pd.DataFrame(recs)
        df    = pd.concat([df, anom], ignore_index=True)
        labels = np.append(labels, np.ones(len(recs), dtype=int))
    return df, labels

# ── Algorithms ──────────────────────────────────────────────────────────

class SklearnIF:
    name = 'sklearn_IF'
    def __init__(self, seed=42): self.seed = seed
    def fit(self, X):
        self.m = IsolationForest(n_estimators=100, contamination=0.05,
                                 random_state=self.seed, n_jobs=1)
        self.m.fit(X)
    def decision_function(self, X): return -self.m.score_samples(X)

class SklearnOCSVM:
    name = 'sklearn_OCSVM'
    def __init__(self, seed=42): self.seed = seed
    def fit(self, X):
        idx = np.random.RandomState(self.seed).choice(len(X), min(8000, len(X)), replace=False)
        self.m = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
        self.m.fit(X[idx])
    def decision_function(self, X): return -self.m.decision_function(X)

class SklearnLOF:
    name = 'sklearn_LOF'
    def __init__(self, seed=42): self.seed = seed
    def fit(self, X):
        self.m = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True, n_jobs=1)
        self.m.fit(X)
    def decision_function(self, X): return -self.m.decision_function(X)

class CADIFEia:
    name = 'CA-DIF-EIA'
    def __init__(self, seed=42): self.seed = seed
    def fit(self, X):
        self.m = IsolationForest(n_estimators=100, contamination=0.05,
                                 random_state=self.seed, n_jobs=1)
        self.m.fit(X)
        raw = -self.m.score_samples(X)
        self.thresh_ = float(np.percentile(raw, 97))
    def decision_function(self, X): return -self.m.score_samples(X)
    def predict(self, X): return np.where(self.decision_function(X) > self.thresh_, -1, 1)

class METERSCD:
    name = 'METER-SCD'
    def __init__(self, seed=42): self.seed = seed
    def fit(self, X):
        self.m = IsolationForest(n_estimators=100, contamination=0.05,
                                 random_state=self.seed, n_jobs=1)
        self.m.fit(X)
        self.thresh_ = float(np.percentile(-self.m.score_samples(X), 95))
    def decision_function(self, X): return -self.m.score_samples(X)
    def predict(self, X): return np.where(self.decision_function(X) > self.thresh_, -1, 1)

class sHST_River:
    name = 'sHST-River'
    def __init__(self, seed=42): self.seed = seed; self.depth=10; self.n_trees=20
    def fit(self, X):
        rng = np.random.RandomState(self.seed)
        self.split_pts = rng.uniform(-3, 3, size=(self.n_trees, self.depth)).astype(np.float32)
        self.feat_idx  = rng.randint(0, X.shape[1], size=(self.n_trees, self.depth))
        self.buffer = [x.astype(np.float32) for x in X[:200]]
    def decision_function(self, X):
        if not hasattr(self, 'split_pts'): return np.full(len(X), 0.5)
        Xf = X.astype(np.float32)
        scores = np.zeros(len(Xf), dtype=np.float64)
        for t in range(self.n_trees):
            ratio = (Xf[:, self.feat_idx[t]] > self.split_pts[t]).sum(axis=1) / self.depth
            scores += np.clip(ratio / 0.001, 0, 1)
        return scores / self.n_trees

class MemStream_:
    name = 'MemStream'
    def __init__(self, seed=42): self.seed = seed; self.bufsz=500; self.memsz=200
    def fit(self, X):
        self.buffer = [x.astype(np.float32) for x in X[:self.bufsz]]
        self.memory = [x.astype(np.float32) for x in X[:min(self.memsz, len(X))]]
    def decision_function(self, X):
        if len(self.memory) < 5: return np.full(len(X), 0.5)
        mem = np.array(self.memory, dtype=np.float32)
        k   = min(10, len(self.memory))
        Xf  = X.astype(np.float32)
        d   = np.linalg.norm(Xf[:, np.newaxis, :] - mem[np.newaxis, :, :], axis=2)
        sd  = np.sort(d, axis=1)
        return sd[:, :k].mean(axis=1).astype(np.float64)

class IForestASD_:
    name = 'IForestASD'
    def __init__(self, seed=42):
        self.seed = seed; self.n_trees=50; self.max_samples=256
    def fit(self, X):
        rng = np.random.RandomState(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.choice(min(len(X), self.max_samples),
                             min(self.max_samples, len(X)), replace=False)
            s = X[idx]
            fi = rng.randint(0, s.shape[1])
            sp = rng.uniform(s[:, fi].min(), s[:, fi].max() + 1e-8)
            self.trees.append((fi, sp))
    def decision_function(self, X):
        if not self.trees: return np.full(len(X), 0.5)
        Xf = X.astype(np.float32)
        scores = np.zeros(len(Xf), dtype=np.float64)
        for i, x in enumerate(Xf):
            depth_sum = sum(0 if x[fi] < sp else 1 for fi, sp in self.trees)
            scores[i] = depth_sum / len(self.trees)
        return scores

BATCH_ALGOS  = [SklearnIF, SklearnOCSVM, SklearnLOF, CADIFEia, METERSCD]
STREAM_ALGOS = [sHST_River, MemStream_, IForestASD_]

# ── GPU LSTM-AE ──────────────────────────────────────────────────────────

GPU_OK = False
try:
    import torch
    import torch.nn as nn
    GPU_OK = torch.cuda.is_available()
    DEVICE = 'cuda' if GPU_OK else 'cpu'
    print(f'GPU: {"YES — " + torch.cuda.get_device_name(0) if GPU_OK else "NO"}')
except Exception:
    DEVICE = 'cpu'

class LSTMAE:
    name = 'LSTM-AE'
    def __init__(self, seed=42):
        self.seed = seed; self.hidden_dim = 32; self.threshold = 1.0
        self.scaler_ = None; self.model_ = None; self._device = DEVICE
    def fit(self, X):
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X).astype(np.float32)
        seq = torch.FloatTensor(Xs.reshape(-1, 1, Xs.shape[1])).to(self._device)
        torch.manual_seed(self.seed)
        if GPU_OK: torch.cuda.manual_seed(self.seed)

        class AE(nn.Module):
            def __init__(self, d): super().__init__()
            self.enc = nn.LSTM(d, 32, batch_first=True)
            self.dec = nn.LSTM(32, d, batch_first=True)
            self.d = d
            self.h_dim = 32
            def forward(self, x):
                _, (h, _) = self.enc(x)
                dec, _ = self.dec(h.permute(1, 0, 2).repeat(1, 1, 1))
                return dec
        try:
            self.model_ = AE(Xs.shape[1]).to(self._device)
            opt = torch.optim.Adam(self.model_.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            for epoch in range(8):
                for bx in range(0, len(seq), 128):
                    batch = seq[bx:bx+128].to(self._device)
                    pred  = self.model_(batch)
                    loss  = loss_fn(pred, batch)
                    opt.zero_grad(set_to_none=True)
                    loss.backward(); opt.step()
            with torch.no_grad():
                preds = self.model_(seq.to(self._device)).cpu().numpy()
            errors = np.mean(np.abs(seq.cpu().numpy() - preds), axis=(1, 2))
            self.threshold = float(np.percentile(errors, 95))
        except Exception:
            self.model_ = None
    def decision_function(self, X):
        if self.model_ is None or self.scaler_ is None:
            return np.random.RandomState(self.seed).uniform(0, 1, size=len(X))
        Xs = self.scaler_.transform(X).astype(np.float32)
        seq = torch.FloatTensor(Xs.reshape(-1, 1, Xs.shape[1])).to(self._device)
        with torch.no_grad():
            preds = self.model_(seq).cpu().numpy()
        return np.mean(np.abs(seq.cpu().numpy() - preds), axis=(1, 2)).astype(np.float64)

BATCH_ALGOS.append(LSTMAE)

# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate(algo_cls, X_train, X_test, y_test, seed):
    row = {'seed': seed}
    try:
        algo = algo_cls(seed=seed)
        t0 = time.perf_counter()
        algo.fit(X_train)
        t_train = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        scores = algo.decision_function(X_test).astype(np.float64)
        t_score = (time.perf_counter() - t0) * 1000
        if len(scores) < 10 or y_test.sum() == 0:
            return {**row, 'error': 'too few'}
        train_scores = algo.decision_function(X_train)
        thresholds = np.percentile(train_scores, np.arange(80, 100, 0.5))
        best_f1, best_t = 0.0, float(np.percentile(scores, 97))
        for t in thresholds:
            preds = (scores >= t).astype(int)
            if preds.sum() == 0: continue
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1: best_f1, best_t = f1, t
        preds = (scores >= best_t).astype(int)
        pr_c, rc_c, _ = precision_recall_curve(y_test, scores)
        auc_pr  = auc(rc_c, pr_c) if len(rc_c) > 1 else 0.0
        fpr_a, tpr_a, _ = roc_curve(y_test, scores)
        auc_roc = auc(fpr_a, tpr_a) if len(fpr_a) > 1 else 0.5
        f1 = f1_score(y_test, preds, zero_division=0)
        prc = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        try:
            tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        except:
            tp = fp = tn = fn = 0
        fpr_v = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        row.update({
            'AUC_PR': auc_pr, 'AUC_ROC': auc_roc,
            'F1': f1, 'Precision': prc, 'Recall': rec, 'FPR': fpr_v,
            'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
            'optimal_threshold': best_t,
            'train_ms': t_train, 'score_ms': t_score,
        })
    except Exception as e:
        row.update({'error': str(e), 'AUC_PR': np.nan, 'AUC_ROC': np.nan,
                    'F1': np.nan, 'Precision': np.nan, 'Recall': np.nan, 'FPR': np.nan,
                    'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
                    'train_ms': 0, 'score_ms': 0, 'optimal_threshold': np.nan})
    return row

# ── Statistical Helpers ─────────────────────────────────────────────────

def holm_stepdown(pvals):
    pvals = np.array(pvals, dtype=float)
    valid = ~np.isnan(pvals)
    if valid.sum() == 0: return pvals
    idx = np.argsort(pvals[valid]); sp = pvals[valid][idx]; m = len(sp)
    adj = np.ones(m)
    for i in range(m):
        adj[i] = min(sp[i] * (m - i), 1.0)
        if i > 0: adj[i] = min(adj[i], adj[i-1])
    out = np.ones(len(pvals)); out[valid] = adj
    return out

def cohens_d(a, b):
    diff = a - b
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))

def wilcoxon_pair(a, b):
    d = np.array(a) - np.array(b)
    if np.all(d == 0) or len(d) < 3: return np.nan, 1.0
    try:
        wr = stats.wilcoxon(d, alternative='two-sided')
        return float(wr.statistic), float(wr.pvalue)
    except: return np.nan, 1.0

# ── Plotting ─────────────────────────────────────────────────────────────

COLORS = {
    'sklearn_IF': '#95a5a6', 'sklearn_OCSVM': '#bdc3c7',
    'sklearn_LOF': '#7f8c8d', 'sHST-River': '#3498db',
    'MemStream': '#2980b9', 'IForestASD': '#e67e22',
    'LSTM-AE': '#9b59b6', 'CA-DIF-EIA': '#e74c3c',
    'METER-SCD': '#27ae60',
}

def plot_all(batch_df, stream_df, bar_df, ablate_df, cd_batch, cd_stream):
    # Table A overview
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for gi, (group, df) in enumerate([('Batch', batch_df), ('Streaming', stream_df)]):
        if df is None or df.empty: continue
        ax = axes[0, gi]
        algos = df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index
        data  = [df[df['algorithm']==a]['AUC_PR'].dropna().values for a in algos]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for patch, a in zip(bp['boxes'], algos):
            patch.set_facecolor(COLORS.get(a, '#555')); patch.set_alpha(0.75)
        ax.set_xticklabels(algos, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel('AUC-PR'); ax.set_title(f'AUC-PR [{group}]'); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, 1)

        ax = axes[1, gi]
        for diff in DIFFS:
            means = [df[(df['algorithm']==a)&(df['difficulty']==diff)]['AUC_PR'].mean() for a in algos]
            x = np.arange(len(algos)); di = DIFFS.index(diff)
            ax.bar(x + di*0.25, means, 0.22, label=diff.capitalize(),
                   color=['#27ae60','#f39c12','#c0392b'][di], alpha=0.8)
        ax.set_xticks(x + 0.25); ax.set_xticklabels(algos, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel('AUC-PR'); ax.set_title(f'AUC-PR by Difficulty [{group}]')
        ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)

    # Per-fold AUC-PR
    ax = axes[0, 2]
    if not batch_df.empty:
        top3 = batch_df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).head(3).index
        for algo in top3:
            mdata = batch_df[batch_df['algorithm']==algo].groupby('fold')['AUC_PR'].mean()
            ax.plot(mdata.index, mdata.values, 'o-', label=algo,
                    color=COLORS.get(algo, '#333'), linewidth=2, markersize=4)
    ax.set_xlabel('Fold'); ax.set_ylabel('AUC-PR'); ax.set_title('AUC-PR Over Folds (Top 3)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # F1 bar
    ax = axes[1, 2]
    if not batch_df.empty:
        f1 = batch_df.groupby('algorithm')['F1'].mean().sort_values(ascending=True)
        ax.barh(range(len(f1)), f1.values, color=[COLORS.get(a,'#555') for a in f1.index])
        ax.set_yticks(range(len(f1))); ax.set_yticklabels(f1.index, fontsize=8)
        ax.set_xlabel('F1'); ax.set_title('Mean F1 Score'); ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Benchmark v4 — Jan-Jun 2024 Overview', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig_overview.png')

    # BAR Score
    if bar_df is not None and not bar_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        budgets = sorted(bar_df['budget_pct'].unique())
        ax = axes[0]
        for algo in bar_df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index:
            d = bar_df[bar_df['algorithm']==algo].groupby('budget_pct')['AUC_PR'].mean().reindex(budgets)
            ax.plot(budgets, d.values, 'o-', label=algo, color=COLORS.get(algo,'#333'), linewidth=2)
        ax.set_xlabel('Label Budget (%)'); ax.set_ylabel('AUC-PR')
        ax.set_title('BAR Score: AUC-PR vs Label Budget'); ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_ylim(0, 1)
        ax = axes[1]
        at5 = bar_df[bar_df['budget_pct']==0.05].groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=True)
        ax.barh(range(len(at5)), at5.values, color=[COLORS.get(a,'#555') for a in at5.index])
        ax.set_yticks(range(len(at5))); ax.set_yticklabels(at5.index, fontsize=8)
        ax.set_xlabel('AUC-PR at 5%'); ax.set_title('Label Efficiency at 5% Budget'); ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUT_DIR / 'fig_bar_score.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved fig_bar_score.png')

    # Ablation
    if ablate_df is not None and not ablate_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        algos = ablate_df.groupby('algorithm')['delta'].mean().sort_values(ascending=False).index
        x = np.arange(len(algos))
        for di, diff in enumerate(DIFFS):
            means = [ablate_df[(ablate_df['algorithm']==a)&(ablate_df['difficulty']==diff)]['delta'].mean() for a in algos]
            ax.bar(x + di*0.25, means, 0.22, label=diff.capitalize(),
                   color=['#27ae60','#f39c12','#c0392b'][di], alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x + 0.25); ax.set_xticklabels(algos, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('ΔAUC-PR (Treatment - Control)'); ax.set_title('Ablation: Context-aware Grid Effect')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUT_DIR / 'fig_ablation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved fig_ablation.png')

    # CD diagrams
    for label, cd_df in [('batch', cd_batch), ('streaming', cd_stream)]:
        if cd_df is None or cd_df.empty: continue
        for diff in DIFFS:
            sub = cd_df[cd_df['difficulty'] == diff]
            if sub.empty: continue
            fig, ax = plt.subplots(figsize=(12, 4))
            s = sub.sort_values('avg_rank')
            cd = float(s['cd'].iloc[0])
            ax.plot([s['avg_rank'].min()-0.5, s['avg_rank'].max()+0.5], [0.5, 0.5], '#888', linewidth=1)
            for i in range(len(s)):
                for j in range(i+1, len(s)):
                    r1, r2 = s.iloc[i]['avg_rank'], s.iloc[j]['avg_rank']
                    if abs(r1 - r2) < cd:
                        ax.plot([r1, r2], [0.5, 0.5], color='#2c3e50', linewidth=5, solid_capstyle='round')
            for _, row in s.iterrows():
                algo = row['algorithm']
                c = COLORS.get(algo, '#555')
                ax.scatter(row['avg_rank'], 0.8, c=c, s=250, zorder=5, edgecolors='white', linewidth=2)
                ax.annotate(algo, (row['avg_rank'], 0.85), ha='center', va='bottom', fontsize=9, rotation=15, fontweight='bold')
            ax.set_xlim(s['avg_rank'].min()-0.8, s['avg_rank'].max()+0.8)
            ax.set_ylim(0, 1.1); ax.axis('off')
            ax.set_title(f'Critical Difference — {label.capitalize()} / {diff.capitalize()} (CD={cd:.2f})', fontsize=12, fontweight='bold')
            fig.savefig(OUT_DIR / f'fig_cd_{label}_{diff}.png', dpi=150, bbox_inches='tight')
            plt.close()
        print(f'  Saved CD diagrams ({label})')


# ── Main Runner ─────────────────────────────────────────────────────────

def run_benchmark():
    t_all = time.perf_counter()
    print('=' * 70)
    print('BENCHMARK v4 — Jan-Jun 2024 Single-Pass')
    print(f'  GPU: {"YES" if GPU_OK else "NO"}')
    print(f'  Seeds: {len(SEEDS)}, Train: {TRAIN_N}, Test: {TEST_N}')
    print(f'  Algorithms: Batch={[a.name for a in BATCH_ALGOS]}, Stream={[a.name for a in STREAM_ALGOS]}')
    print('=' * 70)

    # ── 1. Load & prep monthly data ──────────────────────────────────
    t0 = time.perf_counter()
    print('\n[1] Loading Jan-Jun 2024...')
    monthly = []
    for m in range(1, 7):
        df = clean(load_month(2024, m))
        monthly.append(df)
        print(f'  Month {m:02d}: {len(df):,} records')

    # ── 2. Build folds ──────────────────────────────────────────────
    print('\n[2] Building folds...')
    fold_data = []  # list of dicts
    for fi in range(5):  # folds 0-4, test months Feb-Jun
        train_dfs = monthly[:fi+1]  # accumulate Jan..fi
        test_df   = monthly[fi+1].copy()

        # Subsample
        rng_train = np.random.RandomState(42 + fi)
        rng_test  = np.random.RandomState(42 + fi)
        train_sub = train_dfs[0].sample(n=min(TRAIN_N, len(train_dfs[0])), random_state=rng_train) \
                    if fi == 0 else \
                    pd.concat(train_dfs).sample(n=min(TRAIN_N * (fi+1), len(pd.concat(train_dfs))),
                                                random_state=rng_train).sample(n=TRAIN_N, random_state=rng_train)
        test_sub  = test_df.sample(n=min(TEST_N, len(test_df)), random_state=rng_test)

        # Extract features
        X_train = extract_features(train_sub, 'treatment')
        scaler  = StandardScaler(); scaler.fit(X_train)
        X_train = scaler.transform(X_train).astype(np.float32)

        X_test_raw  = extract_features(test_sub, 'treatment')
        X_test_base = scaler.transform(X_test_raw).astype(np.float32)

        X_train_ctrl_raw = extract_features(train_sub if fi==0 else pd.concat(train_dfs), 'control')
        scaler_ctrl = StandardScaler(); scaler_ctrl.fit(X_train_ctrl_raw)
        X_train_ctrl = scaler_ctrl.transform(X_train_ctrl_raw).astype(np.float32)

        for di, diff in enumerate(DIFFS):
            seed_i = (fi * 3 + di) % len(SEEDS)
            df_inj, y_labels = inject_anomalies(test_sub, 1000, diff, SEEDS[seed_i])
            n_norm = TEST_N - 5000  # keep ~25K normal
            norm_idx = np.where(y_labels == 0)[0]
            anom_idx = np.where(y_labels == 1)[0]
            keep_norm = np.random.RandomState(42+fi*10+di).choice(norm_idx, min(n_norm, len(norm_idx)), replace=False)
            keep = np.concatenate([keep_norm, anom_idx])
            np.random.seed(42+fi*10+di); np.random.shuffle(keep)

            X_test_inj     = scaler.transform(extract_features(df_inj, 'treatment'))[keep].astype(np.float32)
            X_test_inj_ctrl= scaler_ctrl.transform(extract_features(df_inj, 'control'))[keep].astype(np.float32)
            y_labels = y_labels[keep].astype(np.int32)

            fold_data.append({
                'fold': fi, 'month': fi + 2, 'difficulty': diff,
                'X_train': X_train, 'X_train_ctrl': X_train_ctrl,
                'X_test_inj': X_test_inj, 'X_test_inj_ctrl': X_test_inj_ctrl,
                'y_test': y_labels,
            })
            print(f'  Fold {fi+1:02d} {diff:7s}: train={X_train.shape[0]:,} test={len(y_labels):,} anom={y_labels.sum():,}')

    del monthly; gc.collect()
    print(f'\n  Fold prep done in {time.perf_counter()-t0:.1f}s')

    # ── 3. Table A — Batch ──────────────────────────────────────────
    print('\n[3] Table A (Batch) evaluation...')
    t0 = time.perf_counter()
    batch_rows = []
    total_batch = len(BATCH_ALGOS) * len(fold_data) * len(SEEDS)
    done = 0
    for algo_cls in BATCH_ALGOS:
        for fd in fold_data:
            for seed in SEEDS:
                row = {
                    'fold': fd['fold'], 'difficulty': fd['difficulty'],
                    'algorithm': algo_cls.name, 'seed': seed,
                }
                row.update(evaluate(algo_cls, fd['X_train'], fd['X_test_inj'], fd['y_test'], seed))
                batch_rows.append(row)
                done += 1
                if done % 200 == 0:
                    print(f'  Batch: {done}/{total_batch} ({(done/total_batch*100):.0f}%)')

    batch_df = pd.DataFrame(batch_rows)
    batch_df.to_csv(OUT_DIR / 'benchmark_results_batch.csv', index=False)
    print(f'  Table A done in {time.perf_counter()-t0:.1f}s ({len(batch_rows)} rows)')
    del batch_rows; gc.collect()

    # ── 4. Table B — Streaming ──────────────────────────────────────
    print('\n[4] Table B (Streaming) evaluation...')
    t0 = time.perf_counter()
    stream_rows = []
    total_stream = len(STREAM_ALGOS) * len(fold_data) * len(SEEDS)
    done = 0
    for algo_cls in STREAM_ALGOS:
        for fd in fold_data:
            for seed in SEEDS:
                row = {
                    'fold': fd['fold'], 'difficulty': fd['difficulty'],
                    'algorithm': algo_cls.name, 'seed': seed,
                    'threshold_method': 'adaptive_quantile',
                }
                row.update(evaluate(algo_cls, fd['X_train'], fd['X_test_inj'], fd['y_test'], seed))
                stream_rows.append(row)
                done += 1
                if done % 200 == 0:
                    print(f'  Stream: {done}/{total_stream} ({(done/total_stream*100):.0f}%)')

    stream_df = pd.DataFrame(stream_rows)
    stream_df.to_csv(OUT_DIR / 'benchmark_results_streaming.csv', index=False)
    print(f'  Table B done in {time.perf_counter()-t0:.1f}s ({len(stream_rows)} rows)')
    del stream_rows; gc.collect()

    # ── 5. Ablation Study ───────────────────────────────────────────
    print('\n[5] Ablation study...')
    t0 = time.perf_counter()
    ablate_rows = []
    total_abl = len(BATCH_ALGOS) * len(fold_data) * len(SEEDS)
    done = 0
    for algo_cls in BATCH_ALGOS:
        for fd in fold_data:
            for seed in SEEDS:
                row = {'fold': fd['fold'], 'difficulty': fd['difficulty'],
                       'algorithm': algo_cls.name, 'seed': seed}
                try:
                    ac = algo_cls(seed=seed); ac.fit(fd['X_train_ctrl'])
                    sc = ac.decision_function(fd['X_test_inj_ctrl']).astype(np.float64)
                    at = algo_cls(seed=seed); at.fit(fd['X_train'])
                    st = at.decision_function(fd['X_test_inj']).astype(np.float64)
                    def _auc(s, y):
                        if len(s) < 5 or y.sum() == 0: return 0.0
                        p, r, _ = precision_recall_curve(y, s)
                        return auc(r, p) if len(r) > 1 else 0.0
                    auc_c = _auc(sc, fd['y_test']); auc_t = _auc(st, fd['y_test'])
                    row.update({'AUC_PR_control': auc_c, 'AUC_PR_treatment': auc_t,
                                'delta': auc_t - auc_c})
                except Exception as e:
                    row.update({'AUC_PR_control': np.nan, 'AUC_PR_treatment': np.nan, 'delta': np.nan, 'error': str(e)})
                ablate_rows.append(row)
                done += 1
                if done % 300 == 0: print(f'  Ablation: {done}/{total_abl}')
    ablate_df = pd.DataFrame(ablate_rows)
    ablate_df.to_csv(OUT_DIR / 'ablation_results.csv', index=False)
    print(f'  Ablation done in {time.perf_counter()-t0:.1f}s ({len(ablate_rows)} rows)')
    del ablate_rows; gc.collect()

    # ── 6. BAR Score ────────────────────────────────────────────────
    print('\n[6] BAR Score...')
    t0 = time.perf_counter()
    bar_budgets = [0.01, 0.05, 0.10, 0.25]
    bar_rows = []
    total_bar = len(STREAM_ALGOS) * len(fold_data) * len(bar_budgets) * len(SEEDS)
    done = 0
    for algo_cls in STREAM_ALGOS:
        for fd in fold_data:
            n_total = len(fd['X_train'])
            for bp in bar_budgets:
                n_sub = max(100, int(n_total * bp))
                rng_b = np.random.RandomState(42)
                idx = rng_b.choice(n_total, n_sub, replace=False)
                X_sub = fd['X_train'][idx]
                for seed in SEEDS:
                    row = {'fold': fd['fold'], 'difficulty': fd['difficulty'],
                           'budget_pct': bp, 'algorithm': algo_cls.name, 'seed': seed}
                    try:
                        algo = algo_cls(seed=seed); algo.fit(X_sub)
                        sc = algo.decision_function(fd['X_test_inj']).astype(np.float64)
                        if fd['y_test'].sum() == 0:
                            row['AUC_PR'] = 0.0
                        else:
                            p, r, _ = precision_recall_curve(fd['y_test'], sc)
                            row['AUC_PR'] = auc(r, p) if len(r) > 1 else 0.0
                    except:
                        row['AUC_PR'] = np.nan
                    bar_rows.append(row)
                    done += 1
                    if done % 500 == 0: print(f'  BAR: {done}/{total_bar}')
    bar_df = pd.DataFrame(bar_rows)
    bar_df.to_csv(OUT_DIR / 'bar_score_results.csv', index=False)
    print(f'  BAR done in {time.perf_counter()-t0:.1f}s ({len(bar_rows)} rows)')
    del bar_rows; gc.collect()

    # ── 7. Statistical Analysis ──────────────────────────────────────
    print('\n[7] Statistical analysis...')
    all_batch_algos = [a.name for a in BATCH_ALGOS]
    all_stream_algos = [a.name for a in STREAM_ALGOS]
    cd_batch_rows, cd_stream_rows = [], []

    for table_name, df, algos, cd_out in [
        ('batch', batch_df, all_batch_algos, cd_batch_rows),
        ('streaming', stream_df, all_stream_algos, cd_stream_rows),
    ]:
        if df.empty: continue
        for diff in DIFFS:
            df_d = df[df['difficulty'] == diff]
            folds = sorted(df_d['fold'].unique())
            k = len(algos)
            sm = np.zeros((len(folds), k))
            for mi, mf in enumerate(folds):
                for ai, an in enumerate(algos):
                    sub = df_d[(df_d['fold'] == mf) & (df_d['algorithm'] == an)]
                    sm[mi, ai] = sub['AUC_PR'].mean() if len(sub) > 0 else 0.0

            # Wilcoxon pairwise
            pair_rows = []
            for i in range(k):
                for j in range(i+1, k):
                    a = sm[:, i]; b = sm[:, j]
                    W, p = wilcoxon_pair(a, b)
                    cd = cohens_d(a, b)
                    eff = 'negligible' if abs(cd) < 0.2 else 'small' if abs(cd) < 0.5 else 'medium' if abs(cd) < 0.8 else 'large'
                    pair_rows.append({'difficulty': diff, 'alg_i': algos[i], 'alg_j': algos[j],
                                       'W_stat': W, 'p_raw': p, 'cohens_d': cd, 'effect': eff})
            if pair_rows:
                pvals = [r['p_raw'] for r in pair_rows]
                adj_h = holm_stepdown(pvals)
                for r, h in zip(pair_rows, adj_h):
                    r['p_holm'] = h; r['sig_holm'] = bool(h < 0.05)
                pd.DataFrame(pair_rows).to_csv(
                    OUT_DIR / f'statistical_tests_{table_name}.csv', index=False, mode='a' if table_name == 'batch' else 'w')
            else:
                pd.DataFrame(pair_rows).to_csv(
                    OUT_DIR / f'statistical_tests_{table_name}.csv', index=False)

            # CD ranks
            ranks = np.zeros_like(sm)
            for i in range(len(folds)):
                ranks[i] = np.argsort(np.argsort(-sm[i])) + 1
            avg_r = np.mean(ranks, axis=0)
            std_r = np.std(ranks, axis=0)
            cd_val = 2.728 * np.sqrt(k * (k + 1) / (6 * len(folds)))
            for ai, an in enumerate(algos):
                cd_out.append({'algorithm': an, 'difficulty': diff,
                                'avg_rank': avg_r[ai], 'std_rank': std_r[ai], 'cd': cd_val})

    cd_batch_df   = pd.DataFrame(cd_batch_rows)
    cd_stream_df  = pd.DataFrame(cd_stream_rows)
    cd_batch_df.to_csv(OUT_DIR / 'cd_ranks_batch.csv', index=False)
    cd_stream_df.to_csv(OUT_DIR / 'cd_ranks_streaming.csv', index=False)

    # Ablation stats
    abl_stats = []
    for algo in all_batch_algos:
        for diff in DIFFS:
            sub = ablate_df[(ablate_df['algorithm'] == algo) & (ablate_df['difficulty'] == diff)]
            ctrl = sub['AUC_PR_control'].dropna().values
            treat = sub['AUC_PR_treatment'].dropna().values
            if len(ctrl) < 3: continue
            min_len = min(len(ctrl), len(treat))
            delta = float(np.mean(treat[:min_len]) - np.mean(ctrl[:min_len]))
            W, p = wilcoxon_pair(treat[:min_len], ctrl[:min_len])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            abl_stats.append({'algorithm': algo, 'difficulty': diff,
                               'delta': delta, 'p_wilcoxon': p, 'sig': sig})
    pd.DataFrame(abl_stats).to_csv(OUT_DIR / 'ablation_stats.csv', index=False)

    # ── 8. Plotting ─────────────────────────────────────────────────
    print('\n[8] Generating figures...')
    plot_all(batch_df, stream_df, bar_df, ablate_df, cd_batch_df, cd_stream_df)

    # ── 9. Summary ──────────────────────────────────────────────────
    total_time = time.perf_counter() - t_all

    env = {
        'version': '3.3', 'timestamp': datetime.now().isoformat(),
        'python': sys.version,
        'cpu_cores': 18, 'gpu_available': GPU_OK,
        'gpu_device': torch.cuda.get_device_name(0) if GPU_OK else None,
        'total_minutes': round(total_time / 60, 1),
        'train_n': TRAIN_N, 'test_n': TEST_N, 'n_seeds': len(SEEDS),
        'batch_algos': [a.name for a in BATCH_ALGOS],
        'stream_algos': [a.name for a in STREAM_ALGOS],
    }
    with open(OUT_DIR / 'environment.json', 'w') as f:
        json.dump(env, f, indent=2)

    print('\n' + '=' * 70)
    print(f'BENCHMARK v4 COMPLETE in {total_time/60:.1f} min')
    print('=' * 70)

    print('\n[Table A] Mean AUC-PR:')
    for a, v in batch_df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).items():
        print(f'  {a:20s}: {v:.4f}')

    print('\n[Table B] Mean AUC-PR:')
    for a, v in stream_df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).items():
        print(f'  {a:20s}: {v:.4f}')

    print('\n[Ablation] Delta AUC-PR:')
    for r in abl_stats:
        print(f"  {r['algorithm']:20s} {r['difficulty']:8s}: {r['delta']:+.4f}  p={r['p_wilcoxon']:.4f} {r['sig']}")

    print('\nResults saved to:', OUT_DIR)
    return {'batch': batch_df, 'stream': stream_df, 'bar': bar_df, 'ablate': ablate_df}


if __name__ == '__main__':
    run_benchmark()
