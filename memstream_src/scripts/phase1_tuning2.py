#!/usr/bin/env python3
"""Optimized Phase 1: Focus on MemStream_, HBOS, CADIFEia. Minimal IF configs."""
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score
)
from sklearn.ensemble import IsolationForest

CACHE_DIR = Path('c:/proj/ldt/results/v5_clean/fold_cache')
OUT_DIR = Path('c:/proj/ldt/results/tuning_v1')
OUT_DIR.mkdir(exist_ok=True)

DIFFICULTIES = ['easy', 'medium', 'hard']
ALL_FOLDS = list(range(1, 12))
TUNING_SEED = 42


def load_fold_data(fold_num, difficulty):
    fold_dir = CACHE_DIR / f'fold_{fold_num:02d}'
    diff_dir = fold_dir / difficulty
    X_train = np.load(fold_dir / 'X_train.npy').astype(np.float32)
    X_test = np.load(diff_dir / 'X_test.npy').astype(np.float32)
    y_labels = np.load(diff_dir / 'y_labels.npy')
    scaler_mean = np.load(fold_dir / 'scaler_mean.npy')
    scaler_scale = np.load(fold_dir / 'scaler_scale.npy')
    train_std = X_train.std(axis=0)
    bad_features = np.where(train_std < 1e-4)[0]
    if len(bad_features) > 0:
        good_mask = train_std >= 1e-4
        safe_scale = max(np.median(scaler_scale[good_mask]), 1.0) if good_mask.sum() > 0 else 1.0
        for f in bad_features:
            old_scale = scaler_scale[f]
            raw_train = X_train[:, f] * old_scale + scaler_mean[f]
            raw_test = X_test[:, f] * old_scale + scaler_mean[f]
            X_train[:, f] = (raw_train - scaler_mean[f]) / safe_scale
            X_test[:, f] = (raw_test - scaler_mean[f]) / safe_scale
            scaler_scale[f] = safe_scale
    return {
        'fold': fold_num, 'difficulty': difficulty,
        'X_train': X_train, 'X_test': X_test,
        'y_labels': y_labels, 'n_anomalies': int(y_labels.sum()),
    }


class MS:
    name = 'MemStream_'
    def __init__(self, seed=42, bufsz=500, memsz=200, k=10):
        self.seed = seed; self.bufsz = bufsz; self.memsz = memsz; self.k = k
    def fit(self, X):
        self.memory = [x.astype(np.float32) for x in X[:min(self.memsz, len(X))]]
    def decision_function(self, X):
        if len(self.memory) < 5:
            return np.full(len(X), 0.5)
        mem = np.array(self.memory, dtype=np.float32)
        k = min(self.k, len(self.memory))
        Xf = X.astype(np.float32)
        d = np.linalg.norm(Xf[:, np.newaxis, :] - mem[np.newaxis, :, :], axis=2)
        sd = np.sort(d, axis=1)
        return sd[:, :k].mean(axis=1).astype(np.float64)


class HBOS_:
    name = 'HBOS'
    def __init__(self, seed=42, n_bins=10):
        self.seed = seed; self.n_bins = n_bins
    def fit(self, X):
        n, d = X.shape
        self.bin_edges = []; self.bin_densities = []
        for j in range(d):
            col = X[:, j]
            edges = np.linspace(col.min() - 1e-8, col.max() + 1e-8, self.n_bins + 1)
            counts, _ = np.histogram(col, bins=edges)
            densities = np.maximum(counts / (counts.sum() + 1e-8), 1e-10)
            self.bin_edges.append(edges)
            self.bin_densities.append(densities)
    def decision_function(self, X):
        n, d = X.shape
        hbos = np.zeros(n)
        for j in range(d):
            edges = self.bin_edges[j]; densities = self.bin_densities[j]
            bin_ids = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, len(densities) - 1)
            hbos -= np.log(densities[bin_ids] + 1e-10)
        return (hbos / d).astype(np.float64)


class IF_:
    name = 'sklearn_IF'
    def __init__(self, seed=42, n_estimators=200, max_features=1.0):
        self.seed = seed; self.n_estimators = n_estimators; self.max_features = max_features
    def fit(self, X):
        self.m = IsolationForest(n_estimators=self.n_estimators,
            max_samples=1.0, max_features=self.max_features,
            contamination=0.05, random_state=self.seed, n_jobs=-1)
        self.m.fit(X)
    def decision_function(self, X):
        return -self.m.score_samples(X).astype(np.float64)


class CADIFEia_:
    name = 'CADIFEia'
    def __init__(self, seed=42, n_bins=5, n_estimators=100):
        self.seed = seed; self.n_bins = n_bins; self.n_estimators = n_estimators; self.models = {}
    def fit(self, X):
        context_vals = X[:, -1]
        bins = np.linspace(context_vals.min(), context_vals.max(), self.n_bins + 1)
        ctx_labels = np.digitize(context_vals, bins[1:-1])
        for ctx in range(self.n_bins):
            mask = ctx_labels == ctx
            n_ctx = mask.sum()
            if n_ctx < 100:
                continue
            rng = np.random.RandomState(self.seed + ctx)
            m = IsolationForest(n_estimators=max(50, min(self.n_estimators, n_ctx // 100)),
                contamination=0.05, random_state=rng.randint(0, 2**31), n_jobs=1)
            m.fit(X[mask])
            self.models[ctx] = {'model': m, 'bins': bins}
        self.bins = bins
    def decision_function(self, X):
        if not self.models:
            return np.zeros(len(X))
        scores = np.zeros(len(X), dtype=np.float64)
        ctx_labels = np.digitize(X[:, -1], self.bins[1:-1])
        for ctx, info in self.models.items():
            mask = ctx_labels == ctx
            if mask.any():
                scores[mask] = -info['model'].score_samples(X[mask])
            else:
                nearest = min(self.models.keys(), key=lambda k: abs(k - ctx))
                scores[mask] = -self.models[nearest]['model'].score_samples(X[mask])
        return scores


def compute_auc(y_true, scores):
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan, np.nan
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_roc = auc(fpr, tpr) if len(fpr) > 1 else 0.5
    precision, recall, _ = precision_recall_curve(y_true, scores)
    auc_pr = auc(recall, precision) if len(recall) > 1 else 0.0
    return auc_roc, auc_pr


def find_optimal_f1(scores, y_true):
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


def evaluate(algo_cls, params, data, seed):
    row = {'fold': data['fold'], 'difficulty': data['difficulty'],
           'seed': seed, 'algorithm': algo_cls.name}
    t0 = time.perf_counter()
    try:
        algo = algo_cls(**params)
        algo.fit(data['X_train'])
        t_train = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = algo.decision_function(data['X_test']).astype(np.float64)
        t_score = time.perf_counter() - t0
        y_true = data['y_labels']
        auc_roc, auc_pr = compute_auc(y_true, scores)
        opt_t, opt_f1 = find_optimal_f1(scores, y_true)
        preds = (scores >= opt_t).astype(int)
        row.update({
            'AUC_ROC': auc_roc, 'AUC_PR': auc_pr, 'F1': opt_f1,
            'Precision': precision_score(y_true, preds, zero_division=0),
            'Recall': recall_score(y_true, preds, zero_division=0),
            'optimal_threshold': opt_t,
            'train_ms': t_train * 1000, 'score_ms': t_score * 1000,
            'n_anomalies': data['n_anomalies'],
            'params': str(params),
        })
    except Exception as e:
        row.update({'error': str(e), 'AUC_ROC': np.nan, 'AUC_PR': np.nan,
                    'F1': np.nan, 'Precision': np.nan, 'Recall': np.nan,
                    'train_ms': 0, 'score_ms': 0, 'params': str(params)})
    return row


# OPTIMIZED CONFIGURATIONS (reduced from 50 to 28)
CONFIGS = []

# MemStream_: fewer configs, focused on key params
# Key insight: k=5 beats k=10 consistently (from partial results)
for bs in [200, 500, 1000]:
    for ms in [200, 500, 1000]:
        for k in [5, 10, 20]:
            CONFIGS.append({
                'cls': MS,
                'params': {'bufsz': bs, 'memsz': ms, 'k': k},
                'label': f'MS_b{bs}_m{ms}_k{k}',
                'group': 'MemStream_',
            })

# HBOS: n_bins - same 5 configs
for nb in [5, 10, 20, 50, 100]:
    CONFIGS.append({
        'cls': HBOS_,
        'params': {'n_bins': nb},
        'label': f'HBOS_b{nb}',
        'group': 'HBOS',
    })

# sklearn_IF: minimal (only 3 key configs)
for ne in [100, 200]:
    CONFIGS.append({
        'cls': IF_,
        'params': {'n_estimators': ne, 'max_features': 1.0},
        'label': f'IF_n{ne}',
        'group': 'sklearn_IF',
    })

# CADIFEia: same 6 configs
for nb in [3, 5, 10]:
    for ne in [50, 200]:
        CONFIGS.append({
            'cls': CADIFEia_,
            'params': {'n_bins': nb, 'n_estimators': ne},
            'label': f'CADIFEia_b{nb}_n{ne}',
            'group': 'CADIFEia',
        })

print(f"Total configs: {len(CONFIGS)}", flush=True)

log_path = OUT_DIR / 'phase1_log2.txt'
def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, 'a') as f:
        f.write(line + '\n')

# Check for existing results
results_file = OUT_DIR / 'phase1_results2.csv'
existing_keys = set()
if results_file.exists():
    existing = pd.read_csv(results_file)
    existing_keys = set(zip(existing['fold'], existing['difficulty'], existing['label']))
    log(f"Resuming: {len(existing_keys)} done")

rows = []
total = len(CONFIGS) * len(ALL_FOLDS) * len(DIFFICULTIES)
count = 0
t_start = time.time()

for fold in ALL_FOLDS:
    for diff in DIFFICULTIES:
        data = load_fold_data(fold, diff)

        for cfg in CONFIGS:
            key = (fold, diff, cfg['label'])
            count += 1
            if key in existing_keys:
                continue

            row = evaluate(cfg['cls'], cfg['params'], data, TUNING_SEED)
            row['label'] = cfg['label']
            row['group'] = cfg['group']
            rows.append(row)

            if len(rows) % 50 == 0:
                elapsed = time.time() - t_start
                rate = len(rows) / max(elapsed, 0.1)
                remaining = (total - len(rows) - len(existing_keys)) / rate / 60
                log(f"  {len(rows)+len(existing_keys)}/{total} ({count}/{total}) - "
                    f"{elapsed/60:.1f}min elapsed, ~{remaining:.1f}min remaining")

        # Save incrementally
        df = pd.DataFrame(rows)
        df.to_csv(results_file, mode='a' if results_file.exists() else 'w',
                  index=False, header=not results_file.exists())
        rows = []

        del data
        gc.collect()

        log(f"Fold {fold} {diff}: DONE")

elapsed = time.time() - t_start
log(f"Phase 1 DONE: {len(existing_keys) + len(rows)} rows in {elapsed/60:.1f} min")
