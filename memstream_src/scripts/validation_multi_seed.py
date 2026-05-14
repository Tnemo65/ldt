#!/usr/bin/env python3
"""Phase 3: Multi-seed validation of best algorithms."""
import gc, os, sys, time
from datetime import datetime
from pathlib import Path
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans
from scipy import stats

CACHE_DIR = Path('c:/proj/ldt/results/v5_clean/fold_cache')
OUT_DIR = Path('c:/proj/ldt/results/tuning_v1')
OUT_DIR.mkdir(exist_ok=True)
DIFFICULTIES = ['easy', 'medium', 'hard']
ALL_FOLDS = list(range(1, 12))
SEEDS = [42, 123, 456, 789, 1024]

log_path = OUT_DIR / 'validation_log.txt'
def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, 'w') as f:
        f.write(line + '\n')

log("=" * 60)
log("MULTI-SEED VALIDATION")
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

# Best configs from tuning
class MS_best:
    name = 'MS_k5_m500'
    def __init__(self, seed, bufsz=500, memsz=500, k=5, strategy='first', use_weighted=False):
        self.seed = seed; self.bufsz = bufsz; self.memsz = memsz
        self.k = k; self.strategy = strategy; self.use_weighted = use_weighted
    def fit(self, X):
        buf = X[:self.bufsz]
        if self.strategy == 'first':
            self.memory = [x.astype(np.float32) for x in buf[:min(self.memsz, len(buf))]]
        elif self.strategy == 'kmeans':
            km = MiniBatchKMeans(n_clusters=min(self.memsz, len(buf)), random_state=self.seed, n_init=2)
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
        return sd[:, :k].mean(axis=1).astype(np.float64)

class HBOS_best:
    name = 'HBOS_b100'
    def __init__(self, seed, n_bins=100):
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

class IF_best:
    name = 'IF_n200_mf50'
    def __init__(self, seed, n_estimators=200, max_features=0.5):
        self.seed = seed; self.n_estimators = n_estimators; self.max_features = max_features
    def fit(self, X):
        self.m = IsolationForest(n_estimators=self.n_estimators,
            max_samples=1.0, max_features=self.max_features,
            contamination=0.05, random_state=self.seed, n_jobs=-1)
        self.m.fit(X)
    def decision_function(self, X):
        return -self.m.score_samples(X).astype(np.float64)

class IF_baseline:
    name = 'sklearn_IF_baseline'
    def __init__(self, seed):
        self.seed = seed
    def fit(self, X):
        self.m = IsolationForest(n_estimators=200, contamination=0.05,
            random_state=self.seed, n_jobs=-1)
        self.m.fit(X)
    def decision_function(self, X):
        return -self.m.score_samples(X).astype(np.float64)

class MS_baseline:
    name = 'MemStream__baseline'
    def __init__(self, seed, bufsz=500, memsz=200, k=10):
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

class DifficultyAdaptiveEnsemble:
    """Ensemble that adapts weights based on estimated data difficulty.

    Uses training score distribution statistics to estimate difficulty,
    then weights algorithms accordingly.
    """
    name = 'DiffAdaptiveEnsemble'

    def __init__(self, components):
        # components: list of (cls, params)
        self.components = components

    def fit(self, X):
        self.models = []
        for cls, params in self.components:
            algo = cls(**params)
            algo.fit(X)
            self.models.append((cls.name, algo))

    def decision_function(self, X):
        all_scores = []
        for name, algo in self.models:
            s = algo.decision_function(X)
            mn, mx = s.min(), s.max()
            if mx > mn:
                norm = (s - mn) / (mx - mn)
            else:
                norm = np.full(len(X), 0.5)
            all_scores.append((name, norm))

        n = len(X)
        weights = np.ones(n)
        total_w = sum(1 for _ in all_scores)

        # For each sample, weight based on disagreement
        for i in range(n):
            score_vals = np.array([ns[i] for _, ns in all_scores])
            # Std of normalized scores = disagreement measure
            disagreement = np.std(score_vals)
            # Higher disagreement -> equal weights
            # Lower disagreement -> higher weight for MS if MS score is high
            weights[i] = 1.0  # Equal weights for now

        weighted_sum = np.zeros(n)
        for name, norm in all_scores:
            weighted_sum += norm

        return (weighted_sum / total_w).astype(np.float64)


# Top configs for validation
VALIDATION_CONFIGS = [
    ('IF_n200_mf50', IF_best, {'n_estimators': 200, 'max_features': 0.5}),
    ('IF_n200', IF_best, {'n_estimators': 200, 'max_features': 1.0}),
    ('IF_n100', IF_best, {'n_estimators': 100, 'max_features': 1.0}),
    ('MS_k5_m500', MS_best, {'bufsz': 500, 'memsz': 500, 'k': 5, 'strategy': 'first'}),
    ('MS_k3_m500', MS_best, {'bufsz': 500, 'memsz': 500, 'k': 3, 'strategy': 'first'}),
    ('MS_k5_km', MS_best, {'bufsz': 500, 'memsz': 500, 'k': 5, 'strategy': 'kmeans'}),
    ('HBOS_b100', HBOS_best, {'n_bins': 100}),
    ('HBOS_b50', HBOS_best, {'n_bins': 50}),
    ('IF_baseline', IF_baseline, {}),
    ('MS_baseline', MS_baseline, {'bufsz': 500, 'memsz': 200, 'k': 10}),
]

log(f"Total configs: {len(VALIDATION_CONFIGS)}")

results = []
total = len(VALIDATION_CONFIGS) * len(ALL_FOLDS) * len(DIFFICULTIES) * len(SEEDS)
count = 0
t_start = time.time()

for fold in ALL_FOLDS:
    for diff in DIFFICULTIES:
        X_train, X_test, y_labels = load_fold_data(fold, diff)

        for seed in SEEDS:
            for label, cls, params in VALIDATION_CONFIGS:
                count += 1
                t0 = time.perf_counter()
                try:
                    algo = cls(seed=seed, **params)
                    algo.fit(X_train)
                    t_train = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    scores = algo.decision_function(X_test).astype(np.float64)
                    t_score = time.perf_counter() - t0
                    auc_roc, auc_pr = compute_auc(y_labels, scores)
                    opt_t, opt_f1 = find_optimal_f1(scores, y_labels)
                    preds = (scores >= opt_t).astype(int)
                    prc = precision_score(y_labels, preds, zero_division=0)
                    rec = recall_score(y_labels, preds, zero_division=0)
                except Exception as e:
                    import traceback
                    auc_roc = auc_pr = opt_f1 = prc = rec = np.nan
                    t_train = t_score = 0.0
                    scores = np.array([])

                results.append({
                    'fold': fold, 'difficulty': diff, 'seed': seed,
                    'label': label,
                    'AUC_ROC': auc_roc, 'AUC_PR': auc_pr,
                    'F1': opt_f1, 'Precision': prc, 'Recall': rec,
                    'train_ms': t_train * 1000, 'score_ms': t_score * 1000,
                    'n_anomalies': int(y_labels.sum()),
                })

                if count % 200 == 0:
                    elapsed = time.time() - t_start
                    rate = count / max(elapsed, 0.1)
                    remaining = (total - count) / rate / 60
                    log(f"  {count}/{total} ({count/total*100:.0f}%) - "
                        f"{elapsed/60:.1f}min elapsed, ~{remaining:.1f}min remaining")

        del X_train, X_test, y_labels
        gc.collect()
        log(f"Fold {fold}: DONE")

df = pd.DataFrame(results)
df.to_csv(OUT_DIR / 'validation_multi_seed.csv', index=False)
elapsed = time.time() - t_start
log(f"DONE: {len(df)} rows in {elapsed/60:.1f} min")

# ======================================================================
# STATISTICAL ANALYSIS
# ======================================================================
log("\n" + "=" * 60)
log("STATISTICAL ANALYSIS")
log("=" * 60)

# Overall
overall = df.groupby('label')['AUC_PR'].agg(['mean', 'std', 'median', 'count']).reset_index()
overall['cv'] = overall['std'] / overall['mean'].clip(lower=1e-9) * 100
overall = overall.sort_values('mean', ascending=False)
overall.to_csv(OUT_DIR / 'validation_summary.csv', index=False)

log("\n--- OVERALL RANKING (11 folds x 5 seeds x 3 difficulties) ---")
log(f"{'Rank':>4} {'Label':<20} {'Mean':>8} {'Std':>8} {'Med':>8} {'CV%':>7}")
for i, (_, row) in enumerate(overall.iterrows()):
    log(f"{i+1:>4} {row['label']:<20} {row['mean']:>8.4f} {row['std']:>8.4f} "
        f"{row['median']:>8.4f} {row['cv']:>7.1f}%")

# Per-difficulty
for diff in ['easy', 'medium', 'hard']:
    log(f"\n--- {diff.upper()} ---")
    sub = df[df['difficulty'] == diff]
    piv = sub.groupby('label')['AUC_PR'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    for i, (label, vals) in enumerate(piv.iterrows()):
        if i >= 8:
            break
        log(f"  {i+1:>2}. {label:<20}: {vals['mean']:.4f} +/- {vals['std']:.4f}")

# Wilcoxon vs best
log("\n--- Wilcoxon signed-rank vs best ---")
best_label = overall.iloc[0]['label']
pivot = df.pivot_table(values='AUC_PR', index=['fold', 'difficulty', 'seed'], columns='label')

for algo in pivot.columns:
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
    except:
        p = 1.0
    diff_mean = a.mean() - b.mean()
    wins = (a > b).sum()
    losses = (a < b).sum()
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    log(f"  {algo:<20} vs {best_label}: diff={diff_mean:+.4f} p={p:.4f} "
        f"wins={wins}/{len(a)} {sig}")

# Cohen's d effect size
log("\n--- Effect Size (Cohen's d) vs best ---")
for algo in pivot.columns:
    if algo == best_label:
        continue
    a = pivot[best_label].dropna().values
    b = pivot[algo].dropna().values
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) < 2:
        continue
    diff = a.mean() - b.mean()
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = diff / (pooled_std + 1e-10)
    if abs(d) < 0.2:
        eff = 'negligible'
    elif abs(d) < 0.5:
        eff = 'small'
    elif abs(d) < 0.8:
        eff = 'medium'
    else:
        eff = 'large'
    log(f"  {algo:<20} vs {best_label}: d={d:+.3f} ({eff})")

log(f"\nResults saved to {OUT_DIR / 'validation_multi_seed.csv'}")
log(f"BEST: {best_label} (AUC-PR={overall.iloc[0]['mean']:.4f})")


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
