#!/usr/bin/env python3
"""
Targeted Optimization Benchmark - Minimal configs, maximal insight.

Uses existing benchmark results to guide targeted experiments.
Tests: best MemStream_ k-values, ensembles, and IF tuning.
"""
import gc, os, sys, time
from datetime import datetime
from pathlib import Path

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
from sklearn.ensemble import IsolationForest

CACHE_DIR = Path('c:/proj/ldt/results/v5_clean/fold_cache')
OUT_DIR = Path('c:/proj/ldt/results/tuning_v1')
OUT_DIR.mkdir(exist_ok=True)

DIFFICULTIES = ['easy', 'medium', 'hard']
ALL_FOLDS = list(range(1, 12))
TUNING_SEED = 42

log_path = OUT_DIR / 'targeted_log.txt'
def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, 'w') as f:
        f.write(line + '\n')

log("=" * 60)
log("TARGETED OPTIMIZATION BENCHMARK")
log("=" * 60)


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


# ======================================================================
# ALGORITHMS
# ======================================================================

class MS_knn:
    """kNN-based MemStream with variable k and memory initialization."""
    name = 'MS_knn'

    def __init__(self, bufsz=500, memsz=200, k=10, strategy='first', use_weighted=False):
        self.bufsz = bufsz
        self.memsz = memsz
        self.k = k
        self.strategy = strategy  # 'first', 'diverse', 'kmeans'
        self.use_weighted = use_weighted

    def fit(self, X):
        n = len(X)
        buf = X[:self.bufsz]

        if self.strategy == 'first':
            self.memory = [x.astype(np.float32) for x in buf[:min(self.memsz, len(buf))]]
        elif self.strategy == 'diverse':
            # Sample diverse points using max-min distance
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
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=min(self.memsz, len(buf)), random_state=42, n_init=2)
            km.fit(buf)
            self.memory = [c.astype(np.float32) for c in km.cluster_centers_]

    def decision_function(self, X):
        if len(self.memory) < 5:
            return np.full(len(X), 0.5)
        mem = np.array(self.memory, dtype=np.float32)
        k = min(self.k, len(self.memory))
        Xf = X.astype(np.float32)
        d = np.linalg.norm(Xf[:, np.newaxis, :] - mem[np.newaxis, :, :], axis=2)
        sd = np.sort(d, axis=1)

        if self.use_weighted:
            weights = 1.0 / (sd[:, :k] + 1e-8)
            weighted_scores = (weights * sd[:, :k]).sum(axis=1) / weights.sum(axis=1)
            return weighted_scores.astype(np.float64)
        else:
            return sd[:, :k].mean(axis=1).astype(np.float64)


class HBOS_tuned:
    name = 'HBOS'

    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def fit(self, X):
        n, d = X.shape
        self.bin_edges = []
        self.bin_densities = []
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
            edges = self.bin_edges[j]
            densities = self.bin_densities[j]
            bin_ids = np.clip(np.digitize(X[:, j], edges[1:-1]), 0, len(densities) - 1)
            hbos -= np.log(densities[bin_ids] + 1e-10)
        return (hbos / d).astype(np.float64)


class IF_tuned:
    name = 'sklearn_IF'

    def __init__(self, n_estimators=200, max_features=1.0):
        self.n_estimators = n_estimators
        self.max_features = max_features

    def fit(self, X):
        self.m = IsolationForest(n_estimators=self.n_estimators,
            max_samples=1.0, max_features=self.max_features,
            contamination=0.05, random_state=TUNING_SEED, n_jobs=-1)
        self.m.fit(X)

    def decision_function(self, X):
        return -self.m.score_samples(X).astype(np.float64)


class ScoreAvgEnsemble:
    name = 'Ensemble'

    def __init__(self, components):
        # components: list of (cls_name, algo_instance)
        self.components = components

    def decision_function(self, X):
        n = len(X)
        weighted_sum = np.zeros(n)
        total_w = 0.0
        for name, algo, weight in self.components:
            scores = algo.decision_function(X)
            mn, mx = scores.min(), scores.max()
            norm = (scores - mn) / (mx - mn) if mx > mn else np.full(n, 0.5)
            weighted_sum += weight * norm
            total_w += weight
        return (weighted_sum / total_w).astype(np.float64)


# ======================================================================
# CONFIGS TO TEST (minimal, targeted)
# ======================================================================

# Group 1: MemStream_ kNN tuning - 9 configs
MS_CONFIGS = [
    {'bufsz': 500, 'memsz': 200, 'k': 3,  'strategy': 'first',  'use_weighted': False, 'label': 'MS_k3'},
    {'bufsz': 500, 'memsz': 200, 'k': 5,  'strategy': 'first',  'use_weighted': False, 'label': 'MS_k5'},
    {'bufsz': 500, 'memsz': 200, 'k': 7,  'strategy': 'first',  'use_weighted': False, 'label': 'MS_k7'},
    {'bufsz': 500, 'memsz': 200, 'k': 10, 'strategy': 'first',  'use_weighted': False, 'label': 'MS_k10'},
    {'bufsz': 500, 'memsz': 200, 'k': 20, 'strategy': 'first',  'use_weighted': False, 'label': 'MS_k20'},
    # Strategy variations
    {'bufsz': 500, 'memsz': 200, 'k': 5,  'strategy': 'diverse', 'use_weighted': False, 'label': 'MS_k5_diverse'},
    {'bufsz': 500, 'memsz': 200, 'k': 5,  'strategy': 'kmeans', 'use_weighted': False, 'label': 'MS_k5_kmeans'},
    # Memory size variations
    {'bufsz': 500, 'memsz': 500, 'k': 5,  'strategy': 'first',  'use_weighted': False, 'label': 'MS_k5_m500'},
    {'bufsz': 500, 'memsz': 200, 'k': 5,  'strategy': 'first',  'use_weighted': True,  'label': 'MS_k5_wt'},
]

# Group 2: HBOS tuning - 5 configs
HBOS_CONFIGS = [
    {'n_bins': 5,   'label': 'HBOS_b5'},
    {'n_bins': 10,  'label': 'HBOS_b10'},
    {'n_bins': 20,  'label': 'HBOS_b20'},
    {'n_bins': 50,  'label': 'HBOS_b50'},
    {'n_bins': 100, 'label': 'HBOS_b100'},
]

# Group 3: sklearn_IF - 3 configs
IF_CONFIGS = [
    {'n_estimators': 100, 'max_features': 1.0,  'label': 'IF_n100'},
    {'n_estimators': 200, 'max_features': 1.0,  'label': 'IF_n200'},
    {'n_estimators': 200, 'max_features': 0.5, 'label': 'IF_n200_mf50'},
]

# Group 4: Ensembles - 5 configs
ENS_CONFIGS = [
    {'label': 'Ens_MS5_HBOS10', 'components': [
        ('MS_knn', {'bufsz': 500, 'memsz': 200, 'k': 5, 'strategy': 'first', 'use_weighted': False}, 1.0),
        ('HBOS_tuned', {'n_bins': 10}, 1.0),
    ]},
    {'label': 'Ens_MS5_HBOS20', 'components': [
        ('MS_knn', {'bufsz': 500, 'memsz': 200, 'k': 5, 'strategy': 'first', 'use_weighted': False}, 1.0),
        ('HBOS_tuned', {'n_bins': 20}, 1.0),
    ]},
    {'label': 'Ens_MS5_HBOS10_2_1', 'components': [
        ('MS_knn', {'bufsz': 500, 'memsz': 200, 'k': 5, 'strategy': 'first', 'use_weighted': False}, 2.0),
        ('HBOS_tuned', {'n_bins': 10}, 1.0),
    ]},
    {'label': 'Ens_MS3algo', 'components': [
        ('MS_knn', {'bufsz': 500, 'memsz': 200, 'k': 5, 'strategy': 'first', 'use_weighted': False}, 1.0),
        ('HBOS_tuned', {'n_bins': 10}, 1.0),
        ('IF_tuned', {'n_estimators': 200, 'max_features': 1.0}, 1.0),
    ]},
    {'label': 'Ens_MS5_HBOS10_IF200', 'components': [
        ('MS_knn', {'bufsz': 500, 'memsz': 200, 'k': 5, 'strategy': 'first', 'use_weighted': False}, 1.0),
        ('HBOS_tuned', {'n_bins': 10}, 1.0),
        ('IF_tuned', {'n_estimators': 200, 'max_features': 0.5}, 1.0),
    ]},
]

ALL_CONFIGS = MS_CONFIGS + HBOS_CONFIGS + IF_CONFIGS
for ec in ENS_CONFIGS:
    ec['group'] = 'Ensemble'
ALL_CONFIGS += ENS_CONFIGS

for cfg in MS_CONFIGS:
    cfg['group'] = 'MemStream_'
for cfg in HBOS_CONFIGS:
    cfg['group'] = 'HBOS'
for cfg in IF_CONFIGS:
    cfg['group'] = 'sklearn_IF'

print(f"Total configs: {len(ALL_CONFIGS)}", flush=True)
for g in ['MemStream_', 'HBOS', 'sklearn_IF', 'Ensemble']:
    n = sum(1 for c in ALL_CONFIGS if c.get('group') == g)
    print(f"  {g}: {n}", flush=True)

# ======================================================================
# RUN BENCHMARK
# ======================================================================

results = []
total = len(ALL_CONFIGS) * len(ALL_FOLDS) * len(DIFFICULTIES)
count = 0
t_start = time.time()

for fold in ALL_FOLDS:
    for diff in DIFFICULTIES:
        X_train, X_test, y_labels = load_fold_data(fold, diff)

        for cfg in ALL_CONFIGS:
            count += 1
            label = cfg['label']
            group = cfg['group']

            t0 = time.perf_counter()
            try:
                if group == 'Ensemble':
                    # Build ensemble
                    components = []
                    for cls_name, params, weight in cfg['components']:
                        if cls_name == 'MS_knn':
                            algo = MS_knn(**params)
                        elif cls_name == 'HBOS_tuned':
                            algo = HBOS_tuned(**params)
                        elif cls_name == 'IF_tuned':
                            algo = IF_tuned(**params)
                        algo.fit(X_train)
                        components.append((cls_name, algo, weight))
                    ens = ScoreAvgEnsemble(components)
                    scores = ens.decision_function(X_test)
                else:
                    if group == 'MemStream_':
                        algo = MS_knn(**{k: v for k, v in cfg.items() if k not in ['label', 'group']})
                    elif group == 'HBOS':
                        algo = HBOS_tuned(**{k: v for k, v in cfg.items() if k not in ['label', 'group']})
                    elif group == 'sklearn_IF':
                        algo = IF_tuned(**{k: v for k, v in cfg.items() if k not in ['label', 'group']})
                    algo.fit(X_train)
                    scores = algo.decision_function(X_test)

                t_train = time.perf_counter() - t0
                t0 = time.perf_counter()
                scores = scores.astype(np.float64)
                t_score = time.perf_counter() - t0

                auc_roc, auc_pr = compute_auc(y_labels, scores)
                opt_t, opt_f1 = find_optimal_f1(scores, y_labels)
                preds = (scores >= opt_t).astype(int)
                prc = precision_score(y_labels, preds, zero_division=0)
                rec = recall_score(y_labels, preds, zero_division=0)

            except Exception as e:
                import traceback
                print(f"FATAL ERROR for {label} fold={fold} {diff}: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
                auc_roc = auc_pr = opt_f1 = prc = rec = np.nan
                t_train = t_score = 0.0
                scores = np.array([])

            results.append({
                'fold': fold, 'difficulty': diff, 'seed': TUNING_SEED,
                'label': label, 'group': group,
                'AUC_ROC': auc_roc, 'AUC_PR': auc_pr,
                'F1': opt_f1, 'Precision': prc, 'Recall': rec,
                'optimal_threshold': opt_t if not np.isnan(opt_f1) else np.nan,
                'train_ms': t_train * 1000, 'score_ms': t_score * 1000,
                'n_anomalies': int(y_labels.sum()),
            })

            if count % 50 == 0:
                elapsed = time.time() - t_start
                rate = count / max(elapsed, 0.1)
                remaining = (total - count) / rate / 60
                log(f"  {count}/{total} ({count/total*100:.0f}%) - "
                    f"{elapsed/60:.1f}min elapsed, ~{remaining:.1f}min remaining")

        # Save after each fold
        df = pd.DataFrame(results)
        df.to_csv(OUT_DIR / 'targeted_results.csv', index=False)
        del X_train, X_test, y_labels
        gc.collect()
        log(f"Fold {fold}: DONE")

elapsed = time.time() - t_start
log(f"DONE: {len(results)} rows in {elapsed/60:.1f} min")

# ======================================================================
# ANALYSIS
# ======================================================================

log("\n" + "=" * 60)
log("ANALYSIS")
log("=" * 60)

df = pd.read_csv(OUT_DIR / 'targeted_results.csv')

# Overall ranking
log("\n--- OVERALL RANKING (11 folds x 3 difficulties) ---")
overall = df.groupby(['label', 'group'])['AUC_PR'].agg(['mean', 'std', 'median']).reset_index()
overall['cv'] = overall['std'] / overall['mean'].clip(lower=1e-9) * 100
overall = overall.sort_values('mean', ascending=False)
log(f"{'Rank':>4} {'Label':<25} {'Group':<12} {'Mean':>8} {'Std':>8} {'CV%':>7}")
for i, (_, row) in enumerate(overall.iterrows()):
    log(f"{i+1:>4} {row['label']:<25} {row['group']:<12} "
        f"{row['mean']:>8.4f} {row['std']:>8.4f} {row['cv']:>7.1f}%")

# Per-difficulty
for diff in ['easy', 'medium', 'hard']:
    log(f"\n--- {diff.upper()} RANKING ---")
    sub = df[df['difficulty'] == diff]
    piv = sub.groupby(['label', 'group'])['AUC_PR'].mean().sort_values(ascending=False)
    for i, (key, val) in enumerate(piv.items()):
        if i >= 8:
            break
        label, grp = key
        log(f"  {i+1:>2}. {label:<25} [{grp}]: {val:.4f}")

# MemStream_ k analysis
log("\n--- MemStream_ k PARAMETER ANALYSIS ---")
ms_df = df[df['group'] == 'MemStream_']
for diff in ['easy', 'medium', 'hard']:
    sub = ms_df[ms_df['difficulty'] == diff]
    k_vals = {}
    for label in sub['label'].unique():
        # Extract k from label
        if '_k' in label:
            k = int(label.split('_k')[1].split('_')[0])
        else:
            k = -1
        mean_pr = sub[sub['label'] == label]['AUC_PR'].mean()
        k_vals[k] = mean_pr
    log(f"  {diff.upper()}: k={sorted(k_vals.keys())}")
    for k in sorted(k_vals.keys()):
        log(f"    k={k}: AUC-PR={k_vals[k]:.4f}")

# Ensemble analysis
log("\n--- ENSEMBLE vs SINGLE ALGORITHM ---")
ms5 = df[df['label'] == 'MS_k5']
hbos10 = df[df['label'] == 'HBOS_b10']
ens = df[df['label'] == 'Ens_MS5_HBOS10']
for diff in ['easy', 'medium', 'hard']:
    ms5_v = ms5[ms5['difficulty'] == diff]['AUC_PR'].mean()
    hb10_v = hbos10[hbos10['difficulty'] == diff]['AUC_PR'].mean()
    ens_v = ens[ens['difficulty'] == diff]['AUC_PR'].mean()
    log(f"  {diff.upper()}: MS_k5={ms5_v:.4f} HBOS_b10={hb10_v:.4f} "
        f"Ens={ens_v:.4f} best_vs_MS5={ens_v-ms5_v:+.4f}")

log("\nResults saved to " + str(OUT_DIR / 'targeted_results.csv'))
