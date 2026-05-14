#!/usr/bin/env python3
"""
Hyperparameter Tuning Benchmark - Optimized for Feasible Runtime.

Phase 1: All 11 folds x 1 seed x all configs (fast algorithms only)
Phase 2: Analyze and select best configs
Phase 3: Full validation of best configs (11 folds x 5 seeds x 3 difficulties)
CA-MemStream: Separate run after Phase 3

Runtime estimate:
  - Phase 1: ~120 configs * 33 fold-diff = ~2 hours (MemStream+HBOS+IF+Ensemble)
  - Phase 3: ~10 configs * 33 folds * 5 seeds = ~3 hours
  - Total: ~5 hours
"""
import gc
import os
import sys
import time
import importlib.util
from datetime import datetime
from pathlib import Path

# CRITICAL: Unbuffered output for real-time monitoring
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (
    auc, precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score
)
from sklearn.ensemble import IsolationForest

import torch

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

GPU_OK = torch.cuda.is_available()
DEVICE = 'cuda' if GPU_OK else 'cpu'
CACHE_DIR = Path('c:/proj/ldt/results/v5_clean/fold_cache')
OUT_DIR = Path('c:/proj/ldt/results/tuning_v1')
OUT_DIR.mkdir(exist_ok=True)

DIFFICULTIES = ['easy', 'medium', 'hard']
ALL_FOLDS = list(range(1, 12))

# ═══════════════════════════════════════════════════════════════════════════
# CORE LOADING
# ═══════════════════════════════════════════════════════════════════════════

MemStreamCore = None
MemStreamConfig = None
set_determinism = None

def _load_core():
    global MemStreamCore, MemStreamConfig, set_determinism
    src_dir = Path('c:/proj/ldt/memstream_src')

    core_path = src_dir / 'core/memstream_core.py'
    if core_path.exists():
        spec = importlib.util.spec_from_file_location("memstream_core", core_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        MemStreamCore = mod.MemStreamCore
        MemStreamConfig = mod.MemStreamConfig
        set_determinism = mod.set_determinism

_load_core()
print(f"Core loaded. GPU: {GPU_OK}, Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════

class TunedMemStream:
    """kNN-based MemStream with tunable buffer, memory, and k."""
    name = 'MemStream_'

    def __init__(self, seed=42, bufsz=500, memsz=200, k=10):
        self.seed = seed
        self.bufsz = bufsz
        self.memsz = memsz
        self.k = k

    def fit(self, X):
        self.buffer = [x.astype(np.float32) for x in X[:self.bufsz]]
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


class TunedHBOS:
    """HBOS with tunable bin count."""
    name = 'HBOS'

    def __init__(self, seed=42, n_bins=10):
        self.seed = seed
        self.n_bins = n_bins

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

    def decision_function(self, X):
        n, d = X.shape
        hbos = np.zeros(n)
        for j in range(d):
            edges = self.bin_edges[j]
            densities = self.bin_densities[j]
            bin_ids = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, len(densities) - 1)
            hbos -= np.log(densities[bin_ids] + 1e-10)
        return (hbos / d).astype(np.float64)


class TunedIF:
    """sklearn IsolationForest with tunable parameters."""
    name = 'sklearn_IF'

    def __init__(self, seed=42, n_estimators=200, max_samples=1.0,
                 max_features=1.0, contamination=0.05):
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.contamination = contamination

    def fit(self, X):
        self.m = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=-1
        )
        self.m.fit(X)

    def decision_function(self, X):
        return -self.m.score_samples(X).astype(np.float64)


class TunedCADIFEia:
    """CADIFEia with tunable n_bins and context resolution."""
    name = 'CADIFEia'

    def __init__(self, seed=42, n_bins=5, n_estimators=100):
        self.seed = seed
        self.n_bins = n_bins
        self.n_estimators = n_estimators
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
            m = IsolationForest(
                n_estimators=max(50, min(self.n_estimators, n_ctx // 100)),
                contamination=0.05,
                random_state=rng.randint(0, 2**31),
                n_jobs=1
            )
            m.fit(X_ctx)
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


class ScoreAvgEnsemble:
    """Weighted average of normalized scores from multiple algorithms."""
    name = 'ScoreAvgEnsemble'

    def __init__(self, components):
        self.components = components

    def fit(self, X):
        self.models = {}
        for (cls, params) in self.components:
            m = cls(**params)
            m.fit(X)
            self.models[cls.name] = m

    def decision_function(self, X):
        n = len(X)
        weighted_sum = np.zeros(n)
        total_w = 0.0
        for name, m in self.models.items():
            scores = m.decision_function(X)
            mn, mx = scores.min(), scores.max()
            norm = (scores - mn) / (mx - mn) if mx > mn else np.full(n, 0.5)
            w = m.weight if hasattr(m, 'weight') else 1.0
            weighted_sum += w * norm
            total_w += w
        return (weighted_sum / total_w).astype(np.float64)


class RankAvgEnsemble:
    """Rank-based ensemble of multiple algorithms."""
    name = 'RankAvgEnsemble'

    def __init__(self, components):
        self.components = components

    def fit(self, X):
        self.models = {}
        for (cls, params) in self.components:
            m = cls(**params)
            m.fit(X)
            self.models[cls.name] = m

    def decision_function(self, X):
        n = len(X)
        rank_sum = np.zeros(n)
        for name, m in self.models.items():
            scores = m.decision_function(X)
            ranks = np.argsort(np.argsort(scores)) / max(n - 1, 1)
            rank_sum += ranks
        return (rank_sum / len(self.models)).astype(np.float64)


class TuningCA_MemStream:
    """CA-MemStream with tunable warmup parameters."""
    name = 'CA_MemStream'

    def __init__(self, seed=42, warmup_epochs=100, memory_len=100, batch_size=256):
        self.seed = seed
        self.warmup_epochs = warmup_epochs
        self.memory_len = memory_len
        self.batch_size = batch_size
        self._ms_core = None
        self._threshold = None

    def _create_core(self):
        cfg = MemStreamConfig()
        cfg.seed = self.seed
        cfg.warmup_epochs = self.warmup_epochs
        cfg.warmup_batch_size = self.batch_size
        cfg.memory_len = self.memory_len
        cfg.warmup_early_stop_patience = 20
        return MemStreamCore(cfg=cfg, device=DEVICE)

    def fit(self, X, X_calib=None):
        if MemStreamCore is None:
            return
        set_determinism(self.seed)
        torch.manual_seed(self.seed)
        if GPU_OK:
            torch.cuda.manual_seed(self.seed)

        n = len(X)
        n_warmup = int(n * 0.75)
        X_warmup = X[:n_warmup].astype(np.float32)

        self._ms_core = self._create_core()
        self._ms_core.warmup(X_warmup, epochs=self.warmup_epochs,
                              batch_size=self.batch_size, verbose=False)
        self._ms_core.set_beta(0.5)

        if X_calib is not None:
            X_cal = X_calib.astype(np.float32)
        else:
            X_cal = X[n_warmup:].astype(np.float32)
        if len(X_cal) > 0:
            cal_scores = self._ms_core.score_batch(X_cal)
            self._threshold = float(np.percentile(cal_scores, 50))
            self._ms_core.set_beta(self._threshold)
        else:
            self._threshold = 0.5

    def decision_function(self, X):
        if self._ms_core is None:
            return np.full(len(X), 0.5)
        Xf = X.astype(np.float32)
        scores = self._ms_core.score_batch(Xf)
        return scores.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

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

    n_cal = int(len(X_train) * 0.1)
    X_calibration = X_train[-n_cal:].copy()

    return {
        'fold': fold_num, 'difficulty': difficulty,
        'X_train': X_train, 'X_calibration': X_calibration,
        'X_test': X_test, 'y_labels': y_labels,
        'n_anomalies': int(y_labels.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

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
    row = {
        'fold': data['fold'], 'difficulty': data['difficulty'],
        'seed': seed, 'algorithm': algo_cls.name,
    }
    t0 = time.perf_counter()
    try:
        algo = algo_cls(**params)
        if algo_cls.name == 'CA_MemStream':
            algo.fit(data['X_train'], data.get('X_calibration'))
        else:
            algo.fit(data['X_train'])
        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        scores = algo.decision_function(data['X_test']).astype(np.float64)
        t_score = time.perf_counter() - t0

        y_true = data['y_labels']
        auc_roc, auc_pr = compute_auc(y_true, scores)
        opt_t, opt_f1 = find_optimal_f1(scores, y_true)
        preds = (scores >= opt_t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        prc = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)

        row.update({
            'AUC_ROC': auc_roc, 'AUC_PR': auc_pr,
            'F1': opt_f1, 'Precision': prc, 'Recall': rec,
            'optimal_threshold': opt_t,
            'train_ms': t_train * 1000, 'score_ms': t_score * 1000,
            'n_anomalies': data['n_anomalies'],
            'params': str(params),
        })
    except Exception as e:
        row.update({
            'error': str(e), 'AUC_ROC': np.nan, 'AUC_PR': np.nan,
            'F1': np.nan, 'Precision': np.nan, 'Recall': np.nan,
            'train_ms': 0, 'score_ms': 0, 'params': str(params),
        })
    return row


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

# Each config entry: (cls, params, label, group, weight_for_ensemble)
# weight_for_ensemble: None for non-ensemble, or weight value

CONFIGS = []

# MemStream_ configs: vary bufsz, memsz, k
ms_bufsz = [200, 500, 1000]
ms_memsz = [100, 200, 500]
ms_k = [5, 10, 20]
for bs in ms_bufsz:
    for ms in ms_memsz:
        for k in ms_k:
            CONFIGS.append({
                'cls': TunedMemStream,
                'params': {'bufsz': bs, 'memsz': ms, 'k': k},
                'label': f'MS_b{bs}_m{ms}_k{k}',
                'group': 'MemStream_',
                'weight': None,
            })

# HBOS configs: vary n_bins
for nb in [5, 10, 20, 50, 100]:
    CONFIGS.append({
        'cls': TunedHBOS,
        'params': {'n_bins': nb},
        'label': f'HBOS_b{nb}',
        'group': 'HBOS',
        'weight': None,
    })

# sklearn_IF configs: vary n_estimators, max_samples, max_features
if_params = [
    {'n_estimators': 100, 'max_samples': 1.0, 'max_features': 1.0},
    {'n_estimators': 200, 'max_samples': 1.0, 'max_features': 1.0},
    {'n_estimators': 200, 'max_samples': 0.8, 'max_features': 0.8},
    {'n_estimators': 200, 'max_samples': 0.5, 'max_features': 1.0},
    {'n_estimators': 200, 'max_samples': 0.5, 'max_features': 0.5},
    {'n_estimators': 500, 'max_samples': 1.0, 'max_features': 1.0},
    {'n_estimators': 500, 'max_samples': 0.5, 'max_features': 0.5},
    {'n_estimators': 500, 'max_samples': 0.5, 'max_features': 1.0},
]
for p in if_params:
    ne = p['n_estimators']
    ms_val = p['max_samples']
    mf_val = p['max_features']
    CONFIGS.append({
        'cls': TunedIF,
        'params': p,
        'label': f'IF_n{ne}_ms{int(ms_val*100)}_mf{int(mf_val*100)}',
        'group': 'sklearn_IF',
        'weight': None,
    })

# CADIFEia configs
for nb in [3, 5, 10]:
    for ne in [50, 200]:
        CONFIGS.append({
            'cls': TunedCADIFEia,
            'params': {'n_bins': nb, 'n_estimators': ne},
            'label': f'CADIFEia_b{nb}_n{ne}',
            'group': 'CADIFEia',
            'weight': None,
        })

# Ensemble configs: combine best MS variant with HBOS variants
# Will be determined dynamically based on Phase 1 results
ENSEMBLE_CONFIGS = []

print(f"Base configs: {len(CONFIGS)}")
print(f"  MemStream_: {sum(1 for c in CONFIGS if c['group'] == 'MemStream_')}")
print(f"  HBOS: {sum(1 for c in CONFIGS if c['group'] == 'HBOS')}")
print(f"  sklearn_IF: {sum(1 for c in CONFIGS if c['group'] == 'sklearn_IF')}")
print(f"  CADIFEia: {sum(1 for c in CONFIGS if c['group'] == 'CADIFEia')}")

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: GRID SEARCH ON ALL 11 FOLDS, 1 SEED
# ═══════════════════════════════════════════════════════════════════════════

def phase1():
    """Run grid search on all 11 folds, 1 seed."""
    print("\n" + "=" * 80)
    print("PHASE 1: GRID SEARCH (11 folds x 1 seed)")
    print("=" * 80)

    rows = []
    total = len(CONFIGS) * len(ALL_FOLDS) * len(DIFFICULTIES)
    count = 0

    t_start = time.time()

    for fold in ALL_FOLDS:
        for diff in DIFFICULTIES:
            data = load_fold_data(fold, diff)
            seed = 42

            for cfg in CONFIGS:
                count += 1
                row = evaluate(cfg['cls'], cfg['params'], data, seed)
                row['label'] = cfg['label']
                row['group'] = cfg['group']
                rows.append(row)

                if count % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = count / elapsed
                    remaining = (total - count) / rate / 60
                    print(f"  {count}/{total} ({count/total*100:.0f}%) - "
                          f"{elapsed/60:.1f}min elapsed, ~{remaining:.1f}min remaining")

            # Clear fold data from memory
            del data
            gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'phase1_raw.csv', index=False)

    elapsed = time.time() - t_start
    print(f"\nPhase 1 done: {len(df)} rows in {elapsed/60:.1f} min")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: ANALYSIS AND SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def phase2(df):
    """Analyze tuning results and select best configurations."""
    print("\n" + "=" * 80)
    print("PHASE 2: TUNING ANALYSIS")
    print("=" * 80)

    # Group analysis
    results = []
    for group in df['group'].unique():
        sub = df[df['group'] == group]
        for label in sub['label'].unique():
            lsub = sub[sub['label'] == label]
            diff_means = lsub.groupby('difficulty')['AUC_PR'].mean()
            overall_mean = lsub['AUC_PR'].mean()
            overall_std = lsub['AUC_PR'].std()
            overall_median = lsub['AUC_PR'].median()

            results.append({
                'group': group,
                'label': label,
                'params': lsub['params'].iloc[0],
                'mean_AUC_PR': overall_mean,
                'std_AUC_PR': overall_std,
                'median_AUC_PR': overall_median,
                'cv': overall_std / max(overall_mean, 1e-9) * 100,
                'n': len(lsub),
                'easy': diff_means.get('easy', np.nan),
                'medium': diff_means.get('medium', np.nan),
                'hard': diff_means.get('hard', np.nan),
            })

    res_df = pd.DataFrame(results).sort_values('mean_AUC_PR', ascending=False)
    res_df.to_csv(OUT_DIR / 'phase1_summary.csv', index=False)

    # Per-group top configs
    best_cfgs = []
    for group in res_df['group'].unique():
        top = res_df[res_df['group'] == group].nlargest(2, 'mean_AUC_PR')
        best_cfgs.extend(top.to_dict('records'))

    # Also include the current baseline configs
    baseline_labels = [
        ('MemStream_', 'MS_b500_m200_k10'),
        ('HBOS', 'HBOS_b10'),
        ('sklearn_IF', None),  # default
        ('CADIFEia', 'CADIFEia_b5_n200'),
    ]

    # Print results
    print("\n--- Top config per group ---")
    for group in ['MemStream_', 'HBOS', 'sklearn_IF', 'CADIFEia']:
        sub = res_df[res_df['group'] == group].head(3)
        for _, row in sub.iterrows():
            print(f"  {group}: mean={row['mean_AUC_PR']:.4f} "
                  f"e={row['easy']:.4f} m={row['medium']:.4f} h={row['hard']:.4f} "
                  f"params={row['params']}")

    # Print top 10 overall
    print("\n--- Top 10 overall ---")
    for i, (_, row) in enumerate(res_df.head(10).iterrows()):
        print(f"  {i+1}. [{row['group']}] {row['label']}: "
              f"AUC-PR={row['mean_AUC_PR']:.4f} cv={row['cv']:.1f}%")

    return res_df, best_cfgs


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: FULL VALIDATION OF BEST CONFIGS (5 seeds)
# ═══════════════════════════════════════════════════════════════════════════

def phase3(best_cfgs, res_df):
    """Validate best configs on ALL folds with 5 seeds."""
    print("\n" + "=" * 80)
    print("PHASE 3: FULL VALIDATION (11 folds x 5 seeds x 3 difficulties)")
    print("=" * 80)

    # Select: top 3 MemStream_, top 2 HBOS, top 2 IF, top 2 CADIFEia
    # Plus: best ensemble candidates
    selected = []
    for group in ['MemStream_', 'HBOS', 'sklearn_IF', 'CADIFEia']:
        top = res_df[res_df['group'] == group].nlargest(2, 'mean_AUC_PR')
        selected.extend(top.to_dict('records'))

    # Add 3 ensemble configs (best MS + best HBOS combinations)
    best_ms = res_df[res_df['group'] == 'MemStream_'].iloc[0]
    best_hb = res_df[res_df['group'] == 'HBOS'].iloc[0]
    best_if = res_df[res_df['group'] == 'sklearn_IF'].iloc[0]
    best_ca = res_df[res_df['group'] == 'CADIFEia'].iloc[0]

    ms_params = eval(best_ms['params'])
    hb_params = eval(best_hb['params'])
    if_params = eval(best_if['params'])
    ca_params = eval(best_ca['params'])

    # Parse MS params for ensemble
    ms_p = eval(best_ms['params'])

    for w_ms, w_hb, name in [
        (1.0, 1.0, 'Ens_MS_HBOS_1_1'),
        (2.0, 1.0, 'Ens_MS_HBOS_2_1'),
        (1.0, 2.0, 'Ens_MS_HBOS_1_2'),
    ]:
        selected.append({
            'group': 'Ensemble',
            'label': name,
            'params': str({'components': [
                (TunedMemStream, ms_p),
                (TunedHBOS, hb_params),
            ], 'weights': {'MemStream_': w_ms, 'HBOS': w_hb}}),
        })

    # 3-algorithm ensemble
    selected.append({
        'group': 'Ensemble',
        'label': 'Ens_MS_HBOS_IF',
        'params': str({'components': [
            (TunedMemStream, ms_p),
            (TunedHBOS, hb_params),
            (TunedIF, if_params),
        ], 'weights': None}),
    })

    # 4-algorithm ensemble
    selected.append({
        'group': 'Ensemble',
        'label': 'Ens_MS_HBOS_IF_CADIFEia',
        'params': str({'components': [
            (TunedMemStream, ms_p),
            (TunedHBOS, hb_params),
            (TunedIF, if_params),
            (TunedCADIFEia, ca_params),
        ], 'weights': None}),
    })

    # Rank-average ensemble
    selected.append({
        'group': 'Ensemble',
        'label': 'Ens_RankAvg',
        'params': str({'components': [
            (TunedMemStream, ms_p),
            (TunedHBOS, hb_params),
            (TunedIF, if_params),
        ]}),
    })

    print(f"Selected {len(selected)} configs for validation:")
    for s in selected:
        print(f"  [{s['group']}] {s['label']}")

    # We need to reconstruct the actual ensemble classes
    # Replace string params with actual parsed params
    final_configs = []
    for s in selected:
        try:
            params = eval(s['params'])
        except Exception:
            params = {}
        final_configs.append({
            'cls': ScoreAvgEnsemble if s['group'] == 'Ensemble' else s['group'],
            'params': params,
            'label': s['label'],
            'group': s['group'],
        })

    # Handle ensemble class construction
    def make_ensemble_cls(cfg_entry):
        """Create a ScoreAvgEnsemble or RankAvgEnsemble with proper components."""
        params = cfg_entry['params']
        components = params.get('components', [])
        weights = params.get('weights')
        is_rank = 'Rank' in cfg_entry['label']
        cls = RankAvgEnsemble if is_rank else ScoreAvgEnsemble
        # Assign weights to components
        for i, (c_cls, c_params) in enumerate(components):
            if weights and c_cls.name in weights:
                c_obj = c_cls(**c_params)
                c_obj.weight = weights[c_cls.name]
            else:
                c_obj = c_cls(**c_params)
                c_obj.weight = 1.0
            components[i] = (type('Component', (), {'cls': c_cls, 'params': c_params, 'weight': getattr(c_obj, 'weight', 1.0)}), c_params)
        return cls(components)

    rows = []
    total = len(selected) * len(ALL_FOLDS) * len(DIFFICULTIES) * len([42, 123, 456, 789, 1024])
    count = 0
    t_start = time.time()

    for fold in ALL_FOLDS:
        for diff in DIFFICULTIES:
            data = load_fold_data(fold, diff)
            for seed in [42, 123, 456, 789, 1024]:
                for s in selected:
                    count += 1
                    try:
                        params = eval(s['params']) if isinstance(s['params'], str) else s['params']
                    except Exception:
                        params = {}

                    if s['group'] == 'Ensemble':
                        # Build components from params
                        comps = params.get('components', [])
                        weights = params.get('weights')
                        is_rank = 'Rank' in s['label']

                        # Create component objects with weights
                        comp_objs = []
                        for c_cls, c_p in comps:
                            c_obj = c_cls(**c_p)
                            if weights and c_cls.name in weights:
                                c_obj.weight = weights[c_cls.name]
                            else:
                                c_obj.weight = 1.0
                            comp_objs.append((c_cls, c_p))

                        ens_cls = RankAvgEnsemble if is_rank else ScoreAvgEnsemble
                        ens = ens_cls(comp_objs)
                        row = evaluate_ens(ens, data, seed)
                    else:
                        if s['group'] == 'MemStream_':
                            row = evaluate(TunedMemStream, params, data, seed)
                        elif s['group'] == 'HBOS':
                            row = evaluate(TunedHBOS, params, data, seed)
                        elif s['group'] == 'sklearn_IF':
                            row = evaluate(TunedIF, params, data, seed)
                        elif s['group'] == 'CADIFEia':
                            row = evaluate(TunedCADIFEia, params, data, seed)
                        else:
                            continue

                    row['label'] = s['label']
                    row['group'] = s['group']
                    rows.append(row)

                    if count % 50 == 0:
                        elapsed = time.time() - t_start
                        rate = count / elapsed
                        remaining = (total - count) / rate / 60
                        print(f"  {count}/{total} ({count/total*100:.0f}%) - "
                              f"{elapsed/60:.1f}min elapsed, ~{remaining:.1f}min remaining")

                gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'phase3_raw.csv', index=False)

    elapsed = time.time() - t_start
    print(f"\nPhase 3 done: {len(df)} rows in {elapsed/60:.1f} min")
    return df


def evaluate_ens(ens, data, seed):
    """Evaluate an ensemble."""
    row = {
        'fold': data['fold'], 'difficulty': data['difficulty'],
        'seed': seed, 'algorithm': ens.name,
    }
    t0 = time.perf_counter()
    try:
        ens.fit(data['X_train'])
        t_train = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = ens.decision_function(data['X_test']).astype(np.float64)
        t_score = time.perf_counter() - t0

        y_true = data['y_labels']
        auc_roc, auc_pr = compute_auc(y_true, scores)
        opt_t, opt_f1 = find_optimal_f1(scores, y_true)
        preds = (scores >= opt_t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        prc = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)

        row.update({
            'AUC_ROC': auc_roc, 'AUC_PR': auc_pr,
            'F1': opt_f1, 'Precision': prc, 'Recall': rec,
            'optimal_threshold': opt_t,
            'train_ms': t_train * 1000, 'score_ms': t_score * 1000,
            'n_anomalies': data['n_anomalies'],
            'params': '',
        })
    except Exception as e:
        row.update({
            'error': str(e), 'AUC_ROC': np.nan, 'AUC_PR': np.nan,
            'F1': np.nan, 'Precision': np.nan, 'Recall': np.nan,
            'train_ms': 0, 'score_ms': 0, 'params': '',
        })
    return row


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def phase4(df):
    """Statistical analysis of validation results."""
    print("\n" + "=" * 80)
    print("PHASE 4: STATISTICAL ANALYSIS")
    print("=" * 80)

    # Summary by group/difficulty
    summary = df.groupby(['label', 'group', 'difficulty'])['AUC_PR'].agg(
        ['mean', 'std', 'median', 'count']
    ).reset_index()
    summary['cv'] = summary['std'] / summary['mean'].clip(lower=1e-9) * 100
    summary.to_csv(OUT_DIR / 'phase3_summary.csv', index=False)

    # Overall ranking
    overall = df.groupby(['label', 'group'])['AUC_PR'].agg(
        ['mean', 'std', 'median']
    ).reset_index()
    overall['cv'] = overall['std'] / overall['mean'].clip(lower=1e-9) * 100
    overall = overall.sort_values('mean', ascending=False)
    overall.to_csv(OUT_DIR / 'phase3_overall.csv', index=False)

    print("\n--- Overall ranking (5 seeds x 11 folds x 3 difficulties) ---")
    print(f"{'Rank':>4} {'Label':<35} {'Group':<12} {'Mean':>8} {'Std':>8} {'CV%':>7} {'Median':>8}")
    print("-" * 90)
    for i, (_, row) in enumerate(overall.iterrows()):
        print(f"{i+1:>4} {row['label']:<35} {row['group']:<12} "
              f"{row['mean']:>8.4f} {row['std']:>8.4f} {row['cv']:>7.1f}% {row['median']:>8.4f}")

    # Per-difficulty ranking
    print("\n--- Per-difficulty ranking ---")
    for diff in ['easy', 'medium', 'hard']:
        diff_df = summary[summary['difficulty'] == diff].sort_values('mean', ascending=False)
        print(f"\n{diff.upper()}:")
        for i, (_, row) in enumerate(diff_df.head(8).iterrows()):
            print(f"  {i+1}. {row['label']:<35}: {row['mean']:.4f} +/- {row['std']:.4f}")

    # Wilcoxon vs best
    print("\n--- Wilcoxon signed-rank (vs best) ---")
    best_label = overall.iloc[0]['label']
    pivot = df.pivot_table(values='AUC_PR', index=['fold', 'difficulty', 'seed'],
                            columns='label')

    algos = pivot.columns.tolist()
    wilcoxon_results = []
    for algo in algos:
        if algo == best_label:
            continue
        a = pivot[best_label].dropna().values
        b = pivot[algo].dropna().values
        valid = ~(np.isnan(a) | np.isnan(b))
        a, b = a[valid], b[valid]
        if len(a) < 3:
            continue
        try:
            stat, p = stats.wilcoxon(a, b, alternative='two-sided')
        except Exception:
            p = 1.0
        diff = a.mean() - b.mean()
        wins = ((a - b) > 0).sum()
        losses = ((a - b) < 0).sum()
        wilcoxon_results.append({
            'algo': algo, 'best': best_label, 'mean_diff': diff,
            'p_value': p, 'wins': wins, 'losses': losses,
            'win_rate': wins / max(wins + losses, 1),
        })

    wilc_df = pd.DataFrame(wilcoxon_results).sort_values('p_value')
    wilc_df.to_csv(OUT_DIR / 'wilcoxon_phase3.csv', index=False)

    for _, row in wilc_df.iterrows():
        sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else ''))
        print(f"  {row['algo']:<35} vs {best_label}: diff={row['mean_diff']:+.4f} "
              f"p={row['p_value']:.4f} wins={int(row['wins'])}/{int(row['wins']+row['losses'])} {sig}")

    return df, summary, overall, wilc_df, best_label


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def make_plots(df, overall, best_label):
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    algos = sorted(df['label'].unique())
    n = len(algos)
    cmap = plt.cm.get_cmap('tab20', n)
    color_map = dict(zip(algos, [cmap(i) for i in range(n)]))

    # 1. Overall bar chart
    fig, ax = plt.subplots(figsize=(16, 7))
    labels = overall['label'].values
    means = overall['mean'].values
    stds = overall['std'].values
    groups = overall['group'].values

    colors = [color_map.get(l, 'gray') for l in labels]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=2, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('AUC-PR', fontsize=11)
    ax.set_title('Phase 3 Validation: AUC-PR by Algorithm (mean +/- std, 11 folds x 5 seeds)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Highlight best
    best_idx = list(labels).index(best_label) if best_label in labels else 0
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(2.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig_overall_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  fig_overall_validation.png")

    # 2. Per-difficulty bar charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, diff in enumerate(['easy', 'medium', 'hard']):
        ax = axes[i]
        sub = df[df['difficulty'] == diff]
        piv = sub.groupby('label')['AUC_PR'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        colors = [color_map.get(l, 'gray') for l in piv.index]
        bars = ax.bar(range(len(piv)), piv['mean'].values, yerr=piv['std'].values,
                      color=colors, capsize=2, alpha=0.85)
        ax.set_xticks(range(len(piv)))
        ax.set_xticklabels([l[:15] for l in piv.index], rotation=45, ha='right', fontsize=6)
        ax.set_title(f'{diff.upper()}', fontsize=12)
        ax.set_ylabel('AUC-PR' if i == 0 else '')
        ax.grid(axis='y', alpha=0.3)
        bars[0].set_edgecolor('red')
        bars[0].set_linewidth(2)

    plt.suptitle('AUC-PR by Difficulty Level (11 folds x 5 seeds)', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig_by_difficulty.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  fig_by_difficulty.png")

    # 3. Heatmap
    fig, axes = plt.subplots(1, 3, figsize=(20, 10), sharey=True)
    for i, diff in enumerate(['easy', 'medium', 'hard']):
        ax = axes[i]
        sub = df[(df['difficulty'] == diff)]
        piv = sub.pivot_table(values='AUC_PR', index='fold', columns='label')
        # Select top 12 algorithms by overall mean
        top_algos = overall.head(12)['label'].tolist()
        piv = piv[top_algos]

        im = ax.imshow(piv.values, aspect='auto', cmap='YlOrRd', vmin=0)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f'Fold {int(f)}' for f in piv.index], fontsize=8)
        ax.set_xticks(range(len(top_algos)))
        ax.set_xticklabels([l[:12] for l in top_algos], rotation=45, ha='right', fontsize=7)
        ax.set_title(f'{diff.upper()}', fontsize=12)
        plt.colorbar(im, ax=ax, label='AUC-PR')

    plt.suptitle('Fold-Level AUC-PR Heatmap (Top 12 Algorithms)', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig_fold_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  fig_fold_heatmap.png")

    # 4. CV stability plot
    fig, ax = plt.subplots(figsize=(14, 6))
    cv_data = df.groupby(['label', 'group'])['AUC_PR'].agg(['mean', 'std'])
    cv_data['cv'] = cv_data['std'] / cv_data['mean'].clip(lower=1e-9) * 100
    cv_data = cv_data.reset_index().sort_values('cv')

    colors = [color_map.get(l, 'gray') for l in cv_data['label']]
    ax.barh(range(len(cv_data)), cv_data['cv'].values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(cv_data)))
    ax.set_yticklabels(cv_data['label'].values, fontsize=7)
    ax.set_xlabel('Coefficient of Variation (%)')
    ax.set_title('Algorithm Stability (Lower = More Consistent Across Folds)', fontsize=12)
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='CV=50%')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig_stability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  fig_stability.png")

    # 5. Wilcoxon significance plot
    fig, ax = plt.subplots(figsize=(14, 8))
    wilc_path = OUT_DIR / 'wilcoxon_phase3.csv'
    if wilc_path.exists():
        wilc = pd.read_csv(wilc_path)
        wilc = wilc.sort_values('mean_diff')
        y = range(len(wilc))
        colors = ['green' if d > 0 else 'red' for d in wilc['mean_diff']]
        sig = wilc['p_value'] < 0.05
        alphas = [1.0 if s else 0.4 for s in sig]
        ax.barh(y, wilc['mean_diff'].values, color=colors, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(wilc['algo'].values, fontsize=7)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('Mean AUC-PR Difference (best - algo)')
        ax.set_title(f'Wilcoxon Signed-Rank Test vs {best_label} (green = worse than best)', fontsize=11)
        # Mark significant
        for i, (_, row) in enumerate(wilc.iterrows()):
            if row['p_value'] < 0.05:
                ax.text(row['mean_diff'] + 0.002, i, f"p={row['p_value']:.4f}", fontsize=7, va='center')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'fig_wilcoxon.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  fig_wilcoxon.png")

    print(f"\nAll plots saved to {OUT_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log_path = OUT_DIR / 'run_log.txt'

    def log(msg):
        print(msg)
        ts = datetime.now().strftime('%H:%M:%S')
        with open(log_path, 'a') as f:
            f.write(f"[{ts}] {msg}\n")
        sys.stdout.flush()

    log("=" * 80)
    log("HYPERPARAMETER TUNING BENCHMARK v2")
    log("=" * 80)
    t0 = time.time()

    # Phase 1
    if (OUT_DIR / 'phase1_raw.csv').exists():
        log(f"Loading existing Phase 1 results...")
        df1 = pd.read_csv(OUT_DIR / 'phase1_raw.csv')
    else:
        df1 = phase1()

    # Phase 2
    res_df, best_cfgs = phase2(df1)

    # Phase 3
    if (OUT_DIR / 'phase3_raw.csv').exists():
        log(f"Loading existing Phase 3 results...")
        df3 = pd.read_csv(OUT_DIR / 'phase3_raw.csv')
    else:
        df3 = phase3(best_cfgs, res_df)

    # Phase 4
    df_final, summary, overall, wilc_df, best_label = phase4(df3)

    # Plots
    make_plots(df_final, overall, best_label)

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed/3600:.1f} hours")
    log(f"Results: {OUT_DIR}")
    log(f"BEST ALGORITHM: {best_label}")

    return df_final, overall, wilc_df, best_label


if __name__ == '__main__':
    main()
