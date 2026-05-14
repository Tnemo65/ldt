#!/usr/bin/env python3
"""Minimal reproduction of the benchmark loop."""
import sys, os, time, gc
from pathlib import Path

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans

CACHE_DIR = Path('c:/proj/ldt/results/v5_clean/fold_cache')
DIFFICULTIES = ['easy', 'medium', 'hard']
ALL_FOLDS = [1, 2]  # just 2 folds

TUNING_SEED = 42
np.random.seed(TUNING_SEED)

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
    return X_train, X_test, y_labels

class MS_knn:
    name = 'MS_knn'
    def __init__(self, bufsz=500, memsz=200, k=10, strategy='first', use_weighted=False):
        self.bufsz = bufsz; self.memsz = memsz; self.k = k
        self.strategy = strategy; self.use_weighted = use_weighted
    def fit(self, X):
        n = len(X); buf = X[:self.bufsz]
        if self.strategy == 'first':
            self.memory = [x.astype(np.float32) for x in buf[:min(self.memsz, len(buf))]]
        elif self.strategy == 'diverse':
            pool = buf.astype(np.float32)
            self.memory = []
            idx = np.random.randint(len(pool))
            self.memory.append(pool[idx])
            for _ in range(min(self.memsz - 1, len(pool) - 1)):
                dists = np.array([np.min([np.linalg.norm(p - m) for m in self.memory]) for p in pool])
                probs = dists / dists.sum()
                idx = np.random.choice(len(pool), p=probs)
                if np.linalg.norm(pool[idx] - self.memory[-1]) > 1e-6:
                    self.memory.append(pool[idx])
        elif self.strategy == 'kmeans':
            km = MiniBatchKMeans(n_clusters=min(self.memsz, len(buf)), random_state=42, n_init=2)
            km.fit(buf)
            self.memory = [c.astype(np.float32) for c in km.cluster_centers_]
    def decision_function(self, X):
        if len(self.memory) < 5:
            return np.full(len(X), 0.5)
        mem = np.array(self.memory, dtype=np.float32)
        k = min(self.k, len(self.memory))
        Xf = X.astype(np.float32)
        if self.use_weighted:
            d = np.linalg.norm(Xf[:, np.newaxis, :] - mem[np.newaxis, :, :], axis=2)
            sd = np.sort(d, axis=1)
            weights = 1.0 / (sd[:, :k] + 1e-8)
            weighted_scores = (weights * sd[:, :k]).sum(axis=1) / weights.sum(axis=1)
            return weighted_scores.astype(np.float64)
        else:
            d = np.linalg.norm(Xf[:, np.newaxis, :] - mem[np.newaxis, :, :], axis=2)
            sd = np.sort(d, axis=1)
            return sd[:, :k].mean(axis=1).astype(np.float64)

CONFIGS = [
    {'bufsz': 500, 'memsz': 200, 'k': 5, 'strategy': 'first', 'use_weighted': False, 'label': 'MS_k5'},
    {'bufsz': 500, 'memsz': 200, 'k': 5, 'strategy': 'first', 'use_weighted': True, 'label': 'MS_k5_wt'},
]

results = []

for fold in ALL_FOLDS:
    for diff in DIFFICULTIES:
        print(f"FOLD={fold} DIFF={diff}: Loading data...", flush=True)
        X_train, X_test, y_labels = load_fold_data(fold, diff)
        print(f"  y_labels sum: {y_labels.sum()}", flush=True)

        for cfg in CONFIGS:
            label = cfg['label']
            t0 = time.perf_counter()
            try:
                algo = MS_knn(**{k: v for k, v in cfg.items() if k != 'label'})
                algo.fit(X_train)
                scores = algo.decision_function(X_test)
                t_score = time.perf_counter() - t0

                auc_roc, auc_pr = None, None
                if y_labels.sum() == 0 or y_labels.sum() == len(y_labels):
                    auc_roc = np.nan; auc_pr = np.nan
                else:
                    fpr, tpr, _ = roc_curve(y_labels, scores)
                    auc_roc = auc(fpr, tpr) if len(fpr) > 1 else 0.5
                    precision, recall, _ = precision_recall_curve(y_labels, scores)
                    auc_pr = auc(recall, precision) if len(recall) > 1 else 0.0

                print(f"  {label}: AUC_ROC={auc_roc} AUC_PR={auc_pr}", flush=True)
                results.append({'fold': fold, 'diff': diff, 'label': label,
                               'AUC_ROC': auc_roc, 'AUC_PR': auc_pr})

            except Exception as e:
                import traceback
                print(f"  {label}: EXCEPTION {e}: {traceback.format_exc()[:200]}", flush=True)
                results.append({'fold': fold, 'diff': diff, 'label': label,
                               'AUC_ROC': np.nan, 'AUC_PR': np.nan})

        del X_train, X_test, y_labels
        gc.collect()

print("\nFINAL RESULTS:")
for r in results:
    print(f"  {r['fold']} {r['diff']} {r['label']}: ROC={r['AUC_ROC']} PR={r['AUC_PR']}")
