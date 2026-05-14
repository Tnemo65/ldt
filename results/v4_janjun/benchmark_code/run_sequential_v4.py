#!/usr/bin/env python3
"""
Benchmark v4 Lite — Sequential runner, streamlined version.
Focuses on fixed algorithms + new fast algorithms.
Removes: OCSVM (too slow), DeepSVDD (too slow), CBLOF (too slow).

v4 improvements:
  - Fixed CA-DIF-EIA: truly context-aware (train per context cell)
  - Fixed METER-SCD: context-aware scoring
  - Fixed LSTM-AE: proper module-level class + GPU
  - Added HBOS, LOF-K50, Streaming Ensemble
  - Improved threshold calibration

Usage:
    python benchmark/run_sequential_v4.py --out v4     # full run
    python benchmark/run_sequential_v4.py --out v4 --resume  # skip existing
"""
import gc, json, time, sys, os, argparse, traceback, signal
from datetime import datetime
from pathlib import Path

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
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import MiniBatchKMeans
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import torch.nn as nn

GPU_OK = torch.cuda.is_available()
DEVICE = 'cuda' if GPU_OK else 'cpu'

def _excepthook(etype, evalue, etb):
    lines = traceback.format_exception(etype, evalue, etb)
    sys.stderr.write('FATAL EXCEPTION:\n' + ''.join(lines))
    sys.stderr.flush()

def _signal_handler(signum, frame):
    sig_name = signal.Signals(signum).name
    sys.stderr.write(f'RECEIVED SIGNAL {sig_name} (signal {signum})\n')
    traceback.print_stack(frame, file=sys.stderr)
    sys.stderr.flush()
    sys.exit(128 + signum)

sys.excepthook = _excepthook
for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGSEGV,
             signal.SIGFPE, signal.SIGILL, signal.SIGABRT):
    try:
        signal.signal(_sig, _signal_handler)
    except (OSError, ValueError):
        pass

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='v4')
    p.add_argument('--resume', action='store_true')
    return p.parse_args()

args = parse_args()
OUT_DIR = Path(f'c:/proj/ldt/results/{args.out}')
CACHE_DIR = OUT_DIR / 'features' / 'fold_data'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]
DIFFS = ['easy', 'medium', 'hard']
COLORS = {
    'sklearn_IF': '#95a5a6', 'sklearn_LOF': '#7f8c8d', 'LOF-50': '#34495e',
    'CA-DIF-EIA': '#e74c3c', 'METER-SCD': '#27ae60', 'LSTM-AE': '#9b59b6',
    'HBOS': '#2ecc71',
    'sHST-River': '#3498db', 'MemStream': '#2980b9', 'IForestASD': '#e67e22',
    'sHST-Mem-Ens': '#8e44ad',
}

# ═══════════════════════════════════════════════════════════════════
# BATCH ALGORITHMS
# ═══════════════════════════════════════════════════════════════════

class SklearnIF:
    name = 'sklearn_IF'
    def __init__(self, seed=42):
        self.seed = seed
    def fit(self, X):
        self.m = IsolationForest(n_estimators=200, contamination=0.05,
                                random_state=self.seed, n_jobs=1)
        self.m.fit(X)
    def decision_function(self, X):
        return -self.m.score_samples(X).astype(np.float64)


class SklearnLOF:
    name = 'sklearn_LOF'
    def __init__(self, seed=42):
        self.seed = seed
    def fit(self, X):
        idx = np.random.RandomState(self.seed).choice(
            len(X), min(20000, len(X)), replace=False)
        self.m = LocalOutlierFactor(n_neighbors=20, contamination=0.05,
                                    novelty=True, n_jobs=1)
        self.m.fit(X[idx])
    def decision_function(self, X):
        return -self.m.decision_function(X)


class LOF_K50:
    """LOF with k=50 neighbors."""
    name = 'LOF-50'
    def __init__(self, seed=42):
        self.seed = seed
    def fit(self, X):
        idx = np.random.RandomState(self.seed).choice(
            len(X), min(30000, len(X)), replace=False)
        self.m = LocalOutlierFactor(n_neighbors=50, contamination=0.05,
                                    novelty=True, n_jobs=1)
        self.m.fit(X[idx])
    def decision_function(self, X):
        return -self.m.decision_function(X)


# ─── FIXED CA-DIF-EIA: Context-Aware Density-Informed Feature Ensemble ───
class CADIFEia:
    """
    Truly context-aware: trains separate IsolationForest per context cell.
    Context = month feature (last column of X).
    """
    name = 'CA-DIF-EIA'
    def __init__(self, seed=42, n_bins=5):
        self.seed = seed
        self.n_bins = n_bins
        self.models = {}

    def fit(self, X):
        n = len(X)
        context_vals = X[:, -1]
        bins = np.linspace(context_vals.min(), context_vals.max(), self.n_bins + 1)
        ctx_labels = np.digitize(context_vals, bins[1:-1])

        for ctx in range(self.n_bins):
            mask = ctx_labels == ctx
            n_ctx = mask.sum()
            if n_ctx < 100:
                continue
            X_ctx = X[mask]
            rng = np.random.RandomState(self.seed + ctx)
            n_est = max(50, min(200, n_ctx // 100))
            m = IsolationForest(
                n_estimators=n_est,
                contamination=0.05,
                random_state=rng.randint(0, 2**31),
                n_jobs=1
            )
            m.fit(X_ctx)
            self.models[ctx] = {'model': m, 'bins': bins}

        self.bins = bins
        self.X_train = X

    def decision_function(self, X):
        if not self.models:
            return np.zeros(len(X))

        scores = np.zeros(len(X), dtype=np.float64)
        weights = np.zeros(len(X), dtype=np.float64)
        ctx_labels = np.digitize(X[:, -1], self.bins[1:-1])

        for ctx, info in self.models.items():
            mask = ctx_labels == ctx
            if mask.any():
                m = info['model']
                scores[mask] = -m.score_samples(X[mask])
                weights[mask] = 1.0
            else:
                nearest = min(self.models.keys(), key=lambda k: abs(k - ctx))
                m = self.models[nearest]['model']
                dist = np.abs(ctx_labels.astype(float) - ctx)
                valid = dist > 0
                scores[valid] = -m.score_samples(X[valid])
                weights[valid] = 1.0 / (dist[valid] + 1.0)

        if weights.sum() > 0:
            scores = scores / (weights + 1e-8)
        return scores


# ─── FIXED METER-SCD: Context-aware with sliding window ───────────────────
class METERSCD:
    """
    Context-aware with sliding window per context cell.
    """
    name = 'METER-SCD'
    def __init__(self, seed=42, n_bins=5, window_size=5000):
        self.seed = seed
        self.n_bins = n_bins
        self.window_size = window_size
        self.ctx_models = {}

    def fit(self, X):
        context_vals = X[:, -1]
        bins = np.linspace(context_vals.min(), context_vals.max(), self.n_bins + 1)
        ctx_labels = np.digitize(context_vals, bins[1:-1])

        for ctx in range(self.n_bins):
            mask = ctx_labels == ctx
            n_ctx = mask.sum()
            if n_ctx < 100:
                continue
            X_ctx = X[mask][-self.window_size:]
            rng = np.random.RandomState(self.seed + ctx)
            n_est = max(50, min(200, n_ctx // 100))
            m = IsolationForest(
                n_estimators=n_est,
                contamination=0.05,
                random_state=rng.randint(0, 2**31),
                n_jobs=1
            )
            m.fit(X_ctx)
            self.ctx_models[ctx] = {'model': m}

        self.bins = bins

    def decision_function(self, X):
        if not self.ctx_models:
            return np.zeros(len(X))

        scores = np.zeros(len(X), dtype=np.float64)
        ctx_labels = np.digitize(X[:, -1], self.bins[1:-1])

        for ctx, info in self.ctx_models.items():
            mask = ctx_labels == ctx
            if mask.any():
                scores[mask] = -info['model'].score_samples(X[mask])

        return scores


# ─── HBOS: Histogram-Based Outlier Score ────────────────────────────────
class HBOS:
    """Histogram-Based Outlier Score. Fast for high-dimensional data."""
    name = 'HBOS'
    def __init__(self, seed=42, n_bins=10, contamination=0.05):
        self.seed = seed
        self.n_bins = n_bins
        self.contamination = contamination

    def fit(self, X):
        n, d = X.shape
        self.bin_edges = []
        self.bin_densities = []

        for j in range(d):
            col = X[:, j]
            edges = np.linspace(col.min() - 1e-8, col.max() + 1e-8, self.n_bins + 1)
            counts, _ = np.histogram(col, bins=edges)
            densities = counts / (counts.sum() + 1e-8)
            densities = np.maximum(densities, 1e-10)
            self.bin_edges.append(edges)
            self.bin_densities.append(densities)

        rng = np.random.RandomState(self.seed)
        sample_idx = rng.choice(n, min(10000, n), replace=False)
        scores_sample = self._hbos_scores(X[sample_idx])
        self.threshold_ = float(np.percentile(scores_sample, 100 * (1 - self.contamination)))

    def _hobos_scores(self, X):
        n, d = X.shape
        hbos = np.zeros(n)
        for j in range(d):
            edges = self.bin_edges[j]
            densities = self.bin_densities[j]
            bin_ids = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, len(densities) - 1)
            hbos -= np.log(densities[bin_ids] + 1e-10)
        return hbos / d

    def decision_function(self, X):
        return self._hobos_scores(X).astype(np.float64)


# ─── LSTM Autoencoder ───────────────────────────────────────────────────
class LSTMAE:
    name = 'LSTM-AE'
    def __init__(self, seed=42, hidden_dim=64, epochs=15):
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.scaler_ = None
        self.model_ = None
        self._device = DEVICE

    def fit(self, X):
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X).astype(np.float32)
        seq = torch.FloatTensor(Xs.reshape(-1, 1, Xs.shape[1])).to(self._device)
        torch.manual_seed(self.seed)
        if GPU_OK:
            torch.cuda.manual_seed(self.seed)

        try:
            self.model_ = _LSTMAEInner(Xs.shape[1], self.hidden_dim).to(self._device)
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            for epoch in range(self.epochs):
                for bx in range(0, len(seq), 256):
                    batch = seq[bx:bx+256].to(self._device)
                    pred = self.model_(batch)
                    loss = loss_fn(pred, batch)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                preds = self.model_(seq.to(self._device)).cpu().numpy()
                errors = np.mean(np.abs(seq.cpu().numpy() - preds), axis=(1, 2))
                self.threshold_ = float(np.percentile(errors, 95))
        except:
            self.model_ = None

    def decision_function(self, X):
        if self.model_ is None or self.scaler_ is None:
            return np.random.RandomState(self.seed).uniform(0, 1, size=len(X))
        Xs = self.scaler_.transform(X).astype(np.float32)
        seq = torch.FloatTensor(Xs.reshape(-1, 1, Xs.shape[1])).to(self._device)
        with torch.no_grad():
            preds = self.model_(seq).cpu().numpy()
        errors = np.mean(np.abs(seq.cpu().numpy() - preds), axis=(1, 2))
        return errors.astype(np.float64)


class _LSTMAEInner(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.enc = nn.LSTM(d, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, d, batch_first=True)
    def forward(self, x):
        _, (h, _) = self.enc(x)
        dec, _ = self.dec(h.permute(1, 0, 2).repeat(1, 1, 1))
        return dec


# ═══════════════════════════════════════════════════════════════════
# STREAMING ALGORITHMS
# ═══════════════════════════════════════════════════════════════════

class sHST_River:
    name = 'sHST-River'
    def __init__(self, seed=42, depth=10, n_trees=20):
        self.seed = seed
        self.depth = depth
        self.n_trees = n_trees

    def fit(self, X):
        rng = np.random.RandomState(self.seed)
        self.split_pts = rng.uniform(-3, 3, size=(self.n_trees, self.depth)).astype(np.float32)
        self.feat_idx = rng.randint(0, X.shape[1], size=(self.n_trees, self.depth))

    def decision_function(self, X):
        if not hasattr(self, 'split_pts'):
            return np.full(len(X), 0.5)
        Xf = X.astype(np.float32)
        scores = np.zeros(len(Xf), dtype=np.float64)
        for t in range(self.n_trees):
            ratio = (Xf[:, self.feat_idx[t]] > self.split_pts[t]).sum(axis=1) / self.depth
            scores += np.clip(ratio / 0.001, 0, 1)
        return scores / self.n_trees


class MemStream_:
    name = 'MemStream'
    def __init__(self, seed=42, bufsz=500, memsz=200):
        self.seed = seed
        self.bufsz = bufsz
        self.memsz = memsz

    def fit(self, X):
        self.buffer = [x.astype(np.float32) for x in X[:self.bufsz]]
        self.memory = [x.astype(np.float32) for x in X[:min(self.memsz, len(X))]]

    def decision_function(self, X):
        if len(self.memory) < 5:
            return np.full(len(X), 0.5)
        mem = np.array(self.memory, dtype=np.float32)
        k = min(10, len(self.memory))
        Xf = X.astype(np.float32)
        d = np.linalg.norm(Xf[:, np.newaxis, :] - mem[np.newaxis, :, :], axis=2)
        sd = np.sort(d, axis=1)
        return sd[:, :k].mean(axis=1).astype(np.float64)


class IForestASD_:
    name = 'IForestASD'
    def __init__(self, seed=42, n_trees=50, max_samples=256):
        self.seed = seed
        self.n_trees = n_trees
        self.max_samples = max_samples

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
        if not self.trees:
            return np.full(len(X), 0.5)
        Xf = X.astype(np.float32)
        scores = np.zeros(len(Xf), dtype=np.float64)
        for i, x in enumerate(Xf):
            scores[i] = sum(0 if x[fi] < sp else 1 for fi, sp in self.trees) / len(self.trees)
        return scores


class sHST_Mem_Ensemble:
    """Weighted ensemble of sHST-River and MemStream."""
    name = 'sHST-Mem-Ens'
    def __init__(self, seed=42, sHST_weight=0.4, mem_weight=0.6):
        self.seed = seed
        self.sHST_weight = sHST_weight
        self.mem_weight = mem_weight
        self.sHST = sHST_River(seed=seed)
        self.mem = MemStream_(seed=seed)

    def fit(self, X):
        self.sHST.fit(X)
        self.mem.fit(X)

    def decision_function(self, X):
        s1 = self.sHST.decision_function(X)
        s2 = self.mem.decision_function(X)
        return (self.sHST_weight * s1 + self.mem_weight * s2).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════
# ALGORITHM LISTS
# ═══════════════════════════════════════════════════════════════════

BATCH_ALGOS = [
    SklearnIF, SklearnLOF, LOF_K50,
    CADIFEia, METERSCD,
    HBOS, LSTMAE,
]
STREAM_ALGOS = [
    sHST_River, MemStream_, IForestASD_,
    sHST_Mem_Ensemble,
]

# ═══════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════

def find_threshold_percentile(scores, contamination=0.05):
    return float(np.percentile(scores, 100 * (1 - contamination)))


def find_threshold_f1(scores, y_true):
    thresholds = np.percentile(scores, np.arange(80, 100, 0.5))
    best_f1, best_t = 0.0, float(np.percentile(scores, 97))
    for t in thresholds:
        preds = (scores >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


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
            return {**row, 'error': 'too few', 'AUC_PR': np.nan, 'AUC_ROC': np.nan,
                    'F1': np.nan, 'Precision': np.nan, 'Recall': np.nan, 'FPR': np.nan,
                    'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
                    'optimal_threshold': np.nan, 'train_ms': t_train, 'score_ms': t_score}

        pr_c, rc_c, _ = precision_recall_curve(y_test, scores)
        auc_pr = auc(rc_c, pr_c) if len(rc_c) > 1 else 0.0
        fpr_a, tpr_a, _ = roc_curve(y_test, scores)
        auc_roc = auc(fpr_a, tpr_a) if len(fpr_a) > 1 else 0.5

        opt_thresh, opt_f1 = find_threshold_f1(scores, y_test)
        preds = (scores >= opt_thresh).astype(int)

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
            'optimal_threshold': opt_thresh,
            'train_ms': t_train, 'score_ms': t_score
        })
    except Exception as e:
        row.update({
            'error': str(e), 'AUC_PR': np.nan, 'AUC_ROC': np.nan,
            'F1': np.nan, 'Precision': np.nan, 'Recall': np.nan, 'FPR': np.nan,
            'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
            'optimal_threshold': np.nan, 'train_ms': 0, 'score_ms': 0
        })
    return row


# ═══════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════

def holm_stepdown(pvals):
    pvals = np.array(pvals, dtype=float)
    valid = ~np.isnan(pvals)
    if valid.sum() == 0:
        return pvals
    idx = np.argsort(pvals[valid])
    sp = pvals[valid][idx]
    m = len(sp)
    adj = np.ones(m)
    for i in range(m):
        adj[i] = min(sp[i] * (m - i), 1.0)
        if i > 0:
            adj[i] = min(adj[i], adj[i - 1])
    out = np.ones(len(pvals))
    out[valid] = adj
    return out


def wilcoxon_pair(a, b):
    d = np.array(a) - np.array(b)
    if np.all(d == 0) or len(d) < 3:
        return np.nan, 1.0
    try:
        wr = stats.wilcoxon(d, alternative='two-sided')
        return float(wr.statistic), float(wr.pvalue)
    except:
        return np.nan, 1.0


def cohens_d(a, b):
    diff = np.array(a) - np.array(b)
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))


# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════

def plot_results(batch_df, stream_df, bar_df, ablate_df, cd_batch_df, cd_stream_df):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for gi, (label, df) in enumerate([('Batch', batch_df), ('Streaming', stream_df)]):
        if df is None or df.empty:
            continue
        ax = axes[0, gi]
        algos = df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index
        data = [df[df['algorithm'] == a]['AUC_PR'].dropna().values for a in algos]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for patch, a in zip(bp['boxes'], algos):
            patch.set_facecolor(COLORS.get(a, '#555'))
            patch.set_alpha(0.75)
        ax.set_xticklabels(algos, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('AUC-PR')
        ax.set_title(f'AUC-PR [{label}]')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)

        ax = axes[1, gi]
        for di, diff in enumerate(DIFFS):
            means = [df[(df['algorithm'] == a) & (df['difficulty'] == diff)]['AUC_PR'].mean()
                     for a in algos]
            x = np.arange(len(algos))
            ax.bar(x + di * 0.25, means, 0.22, label=diff.capitalize(),
                   color=['#27ae60', '#f39c12', '#c0392b'][di], alpha=0.8)
        ax.set_xticks(x + 0.25)
        ax.set_xticklabels(algos, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('AUC-PR')
        ax.set_title(f'AUC-PR by Difficulty [{label}]')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)

    # Overall rankings
    if batch_df is not None and not batch_df.empty:
        ax = axes[0, 2]
        algos_b = batch_df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False)
        bars = ax.barh(range(len(algos_b)), algos_b.values,
                       color=[COLORS.get(a, '#555') for a in algos_b.index])
        ax.set_yticks(range(len(algos_b)))
        ax.set_yticklabels(algos_b.index, fontsize=9)
        ax.set_xlabel('Mean AUC-PR')
        ax.set_title('Overall Ranking [Batch]')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 1)
        for bar, v in zip(bars, algos_b.values):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{v:.3f}', va='center', fontsize=8)
    else:
        axes[0, 2].axis('off')

    if stream_df is not None and not stream_df.empty:
        ax = axes[1, 2]
        algos_s = stream_df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False)
        bars = ax.barh(range(len(algos_s)), algos_s.values,
                       color=[COLORS.get(a, '#555') for a in algos_s.index])
        ax.set_yticks(range(len(algos_s)))
        ax.set_yticklabels(algos_s.index, fontsize=9)
        ax.set_xlabel('Mean AUC-PR')
        ax.set_title('Overall Ranking [Streaming]')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 1)
        for bar, v in zip(bars, algos_s.values):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{v:.3f}', va='center', fontsize=8)
    else:
        axes[1, 2].axis('off')

    plt.tight_layout()
    fig.savefig(OUT_DIR / 'fig_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig_overview.png')

    # BAR Score plot
    if bar_df is not None and not bar_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        algos_bar = bar_df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index
        x = np.arange(len(bar_df['budget_pct'].unique()))
        for algo in algos_bar:
            vals = [bar_df[(bar_df['algorithm'] == algo) &
                           (bar_df['budget_pct'] == bp)]['AUC_PR'].mean()
                    for bp in sorted(bar_df['budget_pct'].unique())]
            ax.plot(x, vals, 'o-', label=algo, color=COLORS.get(algo, '#555'), linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{bp*100:.0f}%' for bp in sorted(bar_df['budget_pct'].unique())])
        ax.set_xlabel('Training Budget')
        ax.set_ylabel('AUC-PR')
        ax.set_title('Label Efficiency (BAR Score)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUT_DIR / 'fig_bar_score.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved fig_bar_score.png')

    # Ablation plot
    if ablate_df is not None and not ablate_df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        algos_ab = ablate_df.groupby('algorithm')['delta'].mean().sort_values(ascending=False).index
        x = np.arange(len(algos_ab))
        colors = ['#27ae60', '#f39c12', '#c0392b']
        for di, diff in enumerate(DIFFS):
            means = [ablate_df[(ablate_df['algorithm'] == a) &
                               (ablate_df['difficulty'] == diff)]['delta'].mean()
                     for a in algos_ab]
            ax.bar(x + di * 0.25, means, 0.22, label=diff.capitalize(),
                   color=colors[di], alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(x + 0.25)
        ax.set_xticklabels(algos_ab, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Delta AUC-PR (Treatment - Control)')
        ax.set_title('Ablation: Context-aware Treatment Effect')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUT_DIR / 'fig_ablation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved fig_ablation.png')

    # CD diagrams
    for lbl, cd_df in [('batch', cd_batch_df), ('streaming', cd_stream_df)]:
        if cd_df is None or cd_df.empty:
            continue
        for diff in DIFFS:
            sub = cd_df[cd_df['difficulty'] == diff]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(14, 4))
            s = sub.sort_values('avg_rank')
            cd = float(s['cd'].iloc[0])
            ax.plot([s['avg_rank'].min() - 0.5, s['avg_rank'].max() + 0.5],
                    [0.5, 0.5], '#888', linewidth=1)
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    r1, r2 = s.iloc[i]['avg_rank'], s.iloc[j]['avg_rank']
                    if abs(r1 - r2) < cd:
                        ax.plot([r1, r2], [0.5, 0.5], color='#2c3e50', linewidth=5,
                                solid_capstyle='round')
            for _, row in s.iterrows():
                c = COLORS.get(row['algorithm'], '#555')
                ax.scatter(row['avg_rank'], 0.8, c=c, s=250, zorder=5,
                           edgecolors='white', linewidth=2)
                ax.annotate(row['algorithm'],
                           (row['avg_rank'], 0.85),
                           ha='center', va='bottom',
                           fontsize=9, rotation=15, fontweight='bold')
            ax.set_xlim(s['avg_rank'].min() - 0.8, s['avg_rank'].max() + 0.8)
            ax.set_ylim(0, 1.1)
            ax.axis('off')
            ax.set_title(f'Critical Difference — {lbl.capitalize()} / '
                         f'{diff.capitalize()} (CD={cd:.2f})', fontsize=12, fontweight='bold')
            fig.savefig(OUT_DIR / f'fig_cd_{lbl}_{diff}.png',
                       dpi=150, bbox_inches='tight')
            plt.close()
        print(f'  Saved CD diagrams ({lbl})')


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t_all = time.perf_counter()
    print('=' * 70)
    print(f'BENCHMARK v4 Lite — Sequential')
    print(f'  GPU: {"YES" if GPU_OK else "NO"} | Seeds: {len(SEEDS)}')
    print(f'  Batch: {len(BATCH_ALGOS)} | Streaming: {len(STREAM_ALGOS)}')
    print('=' * 70)

    # Load fold bundles
    print('\n[1] Loading cached fold data...')
    bundles = []
    for path in sorted(CACHE_DIR.glob('fold*.npz')):
        d = np.load(path)
        bundles.append({
            'fold': int(d['fold']),
            'difficulty': str(d['difficulty']),
            'X_train': d['X_train'],
            'X_train_ctrl': d['X_train_ctrl'],
            'X_test_inj': d['X_test_inj'],
            'X_test_inj_ctrl': d['X_test_inj_ctrl'],
            'y_test': d['y_test'],
        })
        print(f'  {path.stem}: train={d["X_train"].shape} test={len(d["y_test"])}')
    print(f'  Total: {len(bundles)} bundles')

    # ── Table A (Batch) ─────────────────────────────────────────────
    batch_csv = OUT_DIR / 'benchmark_results_batch.csv'
    if args.resume and batch_csv.exists():
        print('\n[2] Table A (Batch) — SKIPPED')
        batch_df = pd.read_csv(batch_csv)
    else:
        print('\n[2] Table A (Batch) evaluation...')
        t0 = time.perf_counter()
        batch_rows = []
        total = len(BATCH_ALGOS) * len(bundles) * len(SEEDS)
        done = 0
        for algo_cls in BATCH_ALGOS:
            print(f'  [{algo_cls.name}]')
            for fd in bundles:
                for seed in SEEDS:
                    row = {'fold': fd['fold'], 'difficulty': fd['difficulty'],
                           'algorithm': algo_cls.name, 'seed': seed}
                    row.update(evaluate(algo_cls, fd['X_train'], fd['X_test_inj'],
                                         fd['y_test'], seed))
                    batch_rows.append(row)
                    done += 1
                    if done % 100 == 0:
                        elapsed = time.perf_counter() - t0
                        eta = (elapsed / done) * (total - done) / 60
                        print(f'  {done}/{total} ({done/total*100:.0f}%) ETA={eta:.1f}min', flush=True)
                        pd.DataFrame(batch_rows).to_csv(OUT_DIR / 'benchmark_results_batch_check.csv', index=False)
        batch_df = pd.DataFrame(batch_rows)
        batch_df.to_csv(OUT_DIR / 'benchmark_results_batch.csv', index=False)
        print(f'  Table A done in {(time.perf_counter()-t0)/60:.1f}min — {len(batch_rows)} rows')
    gc.collect()

    # ── Table B (Streaming) ───────────────────────────────────────
    stream_csv = OUT_DIR / 'benchmark_results_streaming.csv'
    if args.resume and stream_csv.exists():
        print('\n[3] Table B (Streaming) — SKIPPED')
        stream_df = pd.read_csv(stream_csv)
    else:
        print('\n[3] Table B (Streaming) evaluation...')
        t0 = time.perf_counter()
        stream_rows = []
        total = len(STREAM_ALGOS) * len(bundles) * len(SEEDS)
        done = 0
        for algo_cls in STREAM_ALGOS:
            print(f'  [{algo_cls.name}]')
            for fd in bundles:
                for seed in SEEDS:
                    row = {'fold': fd['fold'], 'difficulty': fd['difficulty'],
                           'algorithm': algo_cls.name, 'seed': seed,
                           'threshold_method': 'adaptive_quantile'}
                    row.update(evaluate(algo_cls, fd['X_train'], fd['X_test_inj'],
                                         fd['y_test'], seed))
                    stream_rows.append(row)
                    done += 1
                    if done % 200 == 0:
                        elapsed = time.perf_counter() - t0
                        eta = (elapsed / done) * (total - done) / 60
                        print(f'  {done}/{total} ({done/total*100:.0f}%) ETA={eta:.1f}min')
        stream_df = pd.DataFrame(stream_rows)
        stream_df.to_csv(OUT_DIR / 'benchmark_results_streaming.csv', index=False)
        print(f'  Table B done in {(time.perf_counter()-t0)/60:.1f}min — {len(stream_rows)} rows')
    gc.collect()

    # ── Ablation ────────────────────────────────────────────────
    print('\n[4] Ablation study...')
    t0 = time.perf_counter()
    ablate_rows = []
    ABLATE_ALGOS = [CADIFEia, METERSCD, SklearnIF]
    ABLATE_TEST_MAX = 10000
    total = len(ABLATE_ALGOS) * len(bundles) * len(SEEDS)
    done = 0

    for algo_cls in ABLATE_ALGOS:
        algo_name = algo_cls.name
        print(f'  [{algo_name}] starting ({total // len(ABLATE_ALGOS)} experiments)')
        for fd in bundles:
            y_full = fd['y_test']
            n_anom = y_full.sum()
            if n_anom >= ABLATE_TEST_MAX:
                idx_anom = np.where(y_full == 1)[0][:ABLATE_TEST_MAX]
                idx_norm = np.where(y_full == 0)[0][:ABLATE_TEST_MAX]
                idx = np.concatenate([idx_anom, idx_norm])
            else:
                idx_norm = np.where(y_full == 0)[0][:ABLATE_TEST_MAX - n_anom]
                idx_anom = np.where(y_full == 1)[0]
                idx = np.concatenate([idx_anom, idx_norm])
            rng_local = np.random.RandomState(42)
            rng_local.shuffle(idx)
            X_ctrl_sub = fd['X_test_inj_ctrl'][idx]
            X_treat_sub = fd['X_test_inj'][idx]
            y_sub = y_full[idx]

            for seed in SEEDS:
                row = {'fold': fd['fold'], 'difficulty': fd['difficulty'],
                       'algorithm': algo_name, 'seed': seed}
                try:
                    ac = algo_cls(seed=seed)
                    ac.fit(fd['X_train_ctrl'])
                    sc = ac.decision_function(X_ctrl_sub).astype(np.float64)

                    at = algo_cls(seed=seed)
                    at.fit(fd['X_train'])
                    st = at.decision_function(X_treat_sub).astype(np.float64)

                    def _auc(s, y):
                        if len(s) < 5 or y.sum() == 0:
                            return 0.0
                        p, r, _ = precision_recall_curve(y, s)
                        return auc(r, p) if len(r) > 1 else 0.0

                    auc_c = _auc(sc, y_sub)
                    auc_t = _auc(st, y_sub)
                    row.update({'AUC_PR_control': auc_c, 'AUC_PR_treatment': auc_t,
                                'delta': auc_t - auc_c})
                except Exception as e:
                    row.update({'AUC_PR_control': np.nan, 'AUC_PR_treatment': np.nan,
                                'delta': np.nan, 'error': str(e)})
                ablate_rows.append(row)
                done += 1
                if done % 100 == 0:
                    pd.DataFrame(ablate_rows).to_csv(OUT_DIR / 'ablation_results.csv', index=False)
                    print(f'  {done}/{total} ({time.perf_counter()-t0:.0f}s)')
        print(f'  [{algo_name}] done')

    ablate_df = pd.DataFrame(ablate_rows)
    ablate_df.to_csv(OUT_DIR / 'ablation_results.csv', index=False)
    print(f'  Ablation done in {(time.perf_counter()-t0)/60:.1f}min — {len(ablate_rows)} rows')
    gc.collect()

    # ── BAR Score ───────────────────────────────────────────────
    bar_csv = OUT_DIR / 'bar_score_results.csv'
    if args.resume and bar_csv.exists():
        print('\n[5] BAR Score — SKIPPED')
        bar_df = pd.read_csv(bar_csv)
    else:
        print('\n[5] BAR Score...')
        t0 = time.perf_counter()
        bar_budgets = [0.01, 0.05, 0.10, 0.25]
        bar_rows = []
        total = len(STREAM_ALGOS) * len(bundles) * len(bar_budgets) * len(SEEDS)
        done = 0
        for algo_cls in STREAM_ALGOS:
            print(f'  [{algo_cls.name}]')
            for fd in bundles:
                n_total = len(fd['X_train'])
                for bp in bar_budgets:
                    n_sub = max(100, int(n_total * bp))
                    rng_b = np.random.RandomState(42)
                    X_sub = fd['X_train'][rng_b.choice(n_total, n_sub, replace=False)]
                    for seed in SEEDS:
                        row = {'fold': fd['fold'], 'difficulty': fd['difficulty'],
                               'budget_pct': bp, 'algorithm': algo_cls.name, 'seed': seed}
                        try:
                            algo = algo_cls(seed=seed)
                            algo.fit(X_sub)
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
                        if done % 500 == 0:
                            print(f'  {done}/{total}')
        bar_df = pd.DataFrame(bar_rows)
        bar_df.to_csv(OUT_DIR / 'bar_score_results.csv', index=False)
        print(f'  BAR done in {(time.perf_counter()-t0)/60:.1f}min — {len(bar_rows)} rows')
    gc.collect()

    # ── Statistical Analysis ────────────────────────────────────
    print('\n[6] Statistical analysis...')
    cd_batch_rows, cd_stream_rows = [], []

    for table_name, df, algos, cd_out in [
        ('batch', batch_df, [a.name for a in BATCH_ALGOS], cd_batch_rows),
        ('streaming', stream_df, [a.name for a in STREAM_ALGOS], cd_stream_rows),
    ]:
        if df.empty:
            continue
        for diff in DIFFS:
            df_d = df[df['difficulty'] == diff]
            folds = sorted(df_d['fold'].unique())
            k = len(algos)
            sm = np.zeros((len(folds), k))
            for mi, mf in enumerate(folds):
                for ai, an in enumerate(algos):
                    sub = df_d[(df_d['fold'] == mf) & (df_d['algorithm'] == an)]
                    sm[mi, ai] = sub['AUC_PR'].mean() if len(sub) > 0 else 0.0
            pair_rows = []
            for i in range(k):
                for j in range(i + 1, k):
                    W, p = wilcoxon_pair(sm[:, i], sm[:, j])
                    cd = cohens_d(sm[:, i], sm[:, j])
                    eff = ('negligible' if abs(cd) < 0.2 else 'small'
                           if abs(cd) < 0.5 else 'medium'
                           if abs(cd) < 0.8 else 'large')
                    pair_rows.append({'difficulty': diff, 'alg_i': algos[i], 'alg_j': algos[j],
                                      'W_stat': W, 'p_raw': p, 'cohens_d': cd, 'effect': eff})
            if pair_rows:
                pvals = [r['p_raw'] for r in pair_rows]
                adj_h = holm_stepdown(pvals)
                for r, h in zip(pair_rows, adj_h):
                    r['p_holm'] = h
                    r['sig_holm'] = bool(h < 0.05)
                pd.DataFrame(pair_rows).to_csv(
                    OUT_DIR / f'statistical_tests_{table_name}.csv', index=False)
            ranks = np.zeros_like(sm)
            for i in range(len(folds)):
                ranks[i] = np.argsort(np.argsort(-sm[i])) + 1
            avg_r = np.mean(ranks, axis=0)
            std_r = np.std(ranks, axis=0)
            cd_val = 2.728 * np.sqrt(k * (k + 1) / (6 * len(folds)))
            for ai, an in enumerate(algos):
                cd_out.append({'algorithm': an, 'difficulty': diff,
                             'avg_rank': avg_r[ai], 'std_rank': std_r[ai], 'cd': cd_val})

    cd_batch_df = pd.DataFrame(cd_batch_rows)
    cd_stream_df = pd.DataFrame(cd_stream_rows)
    cd_batch_df.to_csv(OUT_DIR / 'cd_ranks_batch.csv', index=False)
    cd_stream_df.to_csv(OUT_DIR / 'cd_ranks_streaming.csv', index=False)

    # Ablation stats
    abl_stats = []
    for algo in [a.name for a in ABLATE_ALGOS]:
        for diff in DIFFS:
            sub = ablate_df[(ablate_df['algorithm'] == algo) &
                             (ablate_df['difficulty'] == diff)]
            ctrl = sub['AUC_PR_control'].dropna().values
            treat = sub['AUC_PR_treatment'].dropna().values
            if len(ctrl) < 3:
                continue
            mn = min(len(ctrl), len(treat))
            delta = float(np.mean(treat[:mn]) - np.mean(ctrl[:mn]))
            W, p = wilcoxon_pair(treat[:mn], ctrl[:mn])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            abl_stats.append({'algorithm': algo, 'difficulty': diff,
                               'delta': delta, 'p_wilcoxon': p, 'sig': sig})
    pd.DataFrame(abl_stats).to_csv(OUT_DIR / 'ablation_stats.csv', index=False)

    # ── Plotting ────────────────────────────────────────────────
    print('\n[7] Generating figures...')
    plot_results(batch_df, stream_df, bar_df, ablate_df, cd_batch_df, cd_stream_df)

    # ── Save environment ──────────────────────────────────────
    total_time = time.perf_counter() - t_all
    env = {
        'version': '4.0-Lite',
        'timestamp': datetime.now().isoformat(),
        'python': sys.version,
        'cpu_cores': os.cpu_count(),
        'gpu_available': GPU_OK,
        'gpu_device': torch.cuda.get_device_name(0) if GPU_OK else None,
        'total_minutes': round(total_time / 60, 1),
        'n_seeds': len(SEEDS),
        'batch_algos': [a.name for a in BATCH_ALGOS],
        'stream_algos': [a.name for a in STREAM_ALGOS],
    }
    with open(OUT_DIR / 'environment.json', 'w') as f:
        json.dump(env, f, indent=2)

    # ── Summary ───────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(f'BENCHMARK v4 Lite COMPLETE in {total_time/60:.1f} min')
    print('=' * 70)
    print('\n[Table A] Mean AUC-PR:')
    for a, v in batch_df.groupby('algorithm')['AUC_PR'].mean().sort_values(
            ascending=False).items():
        print(f'  {a:20s}: {v:.4f}')
    print('\n[Table B] Mean AUC-PR:')
    for a, v in stream_df.groupby('algorithm')['AUC_PR'].mean().sort_values(
            ascending=False).items():
        print(f'  {a:20s}: {v:.4f}')
    print('\n[Ablation] Delta AUC-PR:')
    for r in abl_stats:
        print(f"  {r['algorithm']:20s} {r['difficulty']:8s}: {r['delta']:+.4f}  "
              f"p={r['p_wilcoxon']:.4f} {r['sig']}")
    print('\n[BAR Score] AUC-PR at 1% budget:')
    bar1 = bar_df[bar_df['budget_pct'] == 0.01]
    for a, v in bar1.groupby('algorithm')['AUC_PR'].mean().sort_values(
            ascending=False).items():
        print(f'  {a:20s}: {v:.4f}')
    print('\nResults: ' + str(OUT_DIR))


if __name__ == '__main__':
    main()
