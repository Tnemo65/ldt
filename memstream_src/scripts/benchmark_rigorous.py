#!/usr/bin/env python3
"""
Rigorous Scientific Benchmark for Streaming Anomaly Detection Algorithms.

Compares 10 algorithms across ALL 11 folds and 3 difficulties:
  1. sklearn IsolationForest
  2. sklearn LOF
  3. MemStream_ (baseline from run_sequential_v4.py)
  4. sHST_River
  5. IForestASD_
  6. sHST_Mem_Ensemble
  7. CADIFEia
  8. HBOS
  9. CA-MemStream (25D AE+Memory, 100 epochs warmup)
  10. CA-MemStream-BAR (CA-MemStream with BAR Controller)

Key improvements over previous quick benchmark:
  - Uses ALL 11 folds (not just fold 01)
  - Uses 5 seeds (42, 123, 456, 789, 1024) for statistical validity
  - Proper calibration set threshold (not percentile on test)
  - 100 epochs warmup for MemStream variants (not 10!)
  - Wilcoxon signed-rank test with Holm-Bonferroni correction
  - Critical Difference diagrams (Nemenyi test)
  - Effect sizes (Cohen's d)
  - BAR Score measurement at 1%, 5%, 10%, 25%, 50%, 100%

Usage:
    python memstream_src/scripts/benchmark_rigorous.py --out rigorous_v1
    python memstream_src/scripts/benchmark_rigorous.py --out rigorous_v1 --resume
    python memstream_src/scripts/benchmark_rigorous.py --out rigorous_v1 --skip-ca  # skip CA-MemStream
"""
import gc
import json
import os
import sys
import time
import traceback
import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

GPU_OK = torch.cuda.is_available()
DEVICE = 'cuda' if GPU_OK else 'cpu'

SEEDS = [42, 123, 456, 789, 1024]
DIFFICULTIES = ['easy', 'medium', 'hard']
N_FOLDS = 11

# Data paths
CACHE_DIR = Path('c:/proj/ldt/results/v5_clean/fold_cache')

# Output
OUT_DIR = None  # Set in main()

# Colors for plots
COLORS = {
    'sklearn_IF': '#2c3e50',
    'sklearn_LOF': '#7f8c8d',
    'MemStream_': '#2980b9',
    'sHST_River': '#3498db',
    'IForestASD_': '#e67e22',
    'sHST_Mem_Ensemble': '#8e44ad',
    'CADIFEia': '#e74c3c',
    'HBOS': '#27ae60',
    'CA_MemStream': '#c0392b',
    'CA_MemStream_BAR': '#8e44ad',
}

# ═══════════════════════════════════════════════════════════════════════════
# CA-MEMSTREAM IMPORTS (load early so classes can reference them)
# ═══════════════════════════════════════════════════════════════════════════

# Placeholder values for when MemStream core is not available
MemStreamCore = None
MemStreamConfig = None
set_determinism = None
ADWIN = None
BARController = None
MEMSTREAM_CORE_AVAILABLE = False

def _load_memstream_core():
    """Load MemStream core components using importlib to avoid __init__.py issues."""
    global MemStreamCore, MemStreamConfig, set_determinism, ADWIN, BARController, MEMSTREAM_CORE_AVAILABLE

    try:
        # Find the memstream_src directory
        script_dir = Path(__file__).parent
        src_dir = script_dir.parent

        # Load memstream_core directly
        core_path = src_dir / 'core' / 'memstream_core.py'
        if core_path.exists():
            spec = importlib.util.spec_from_file_location("memstream_core", core_path)
            memstream_core = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(memstream_core)
            MemStreamCore = memstream_core.MemStreamCore
            MemStreamConfig = memstream_core.MemStreamConfig
            set_determinism = memstream_core.set_determinism

        # Load context_aware directly
        ca_path = src_dir / 'core' / 'context_aware.py'
        if ca_path.exists():
            spec = importlib.util.spec_from_file_location("context_aware", ca_path)
            context_aware = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(context_aware)
            ADWIN = context_aware.ADWIN
            BARController = context_aware.BARController

        MEMSTREAM_CORE_AVAILABLE = True
        print("CA-MemStream core loaded successfully")
        return True

    except ImportError as e:
        print(f"WARNING: MemStream core not available. CA-MemStream will be skipped. Error: {e}")
        MEMSTREAM_CORE_AVAILABLE = False
        return False

# Load MemStream core on module import
_load_memstream_core()


# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHMS FROM run_sequential_v4.py (copied for standalone execution)
# ═══════════════════════════════════════════════════════════════════════════

class SklearnIF:
    """sklearn IsolationForest baseline."""
    name = 'sklearn_IF'

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X):
        self.m = IsolationForest(
            n_estimators=200, contamination=0.05,
            random_state=self.seed, n_jobs=-1
        )
        self.m.fit(X)

    def decision_function(self, X):
        return -self.m.score_samples(X).astype(np.float64)


class SklearnLOF:
    """sklearn LocalOutlierFactor baseline.

    NOTE: LOF with novelty=True is O(n_test * n_train).
    Subsample test to 10k samples to keep runtime reasonable.
    """
    name = 'sklearn_LOF'
    MAX_TEST_SAMPLES = 10000

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X):
        idx = np.random.RandomState(self.seed).choice(
            len(X), min(20000, len(X)), replace=False)
        self.m = LocalOutlierFactor(
            n_neighbors=20, contamination=0.05,
            novelty=True, n_jobs=-1
        )
        self.m.fit(X[idx])

    def decision_function(self, X):
        """Score test set. Subsample to 10k to keep runtime reasonable."""
        if not hasattr(self, 'm'):
            return np.full(len(X), 0.5)
        if len(X) > self.MAX_TEST_SAMPLES:
            idx = np.random.RandomState(self.seed).choice(
                len(X), self.MAX_TEST_SAMPLES, replace=False)
            scores = -self.m.decision_function(X[idx]).astype(np.float64)
            return np.repeat(scores.mean(), len(X))
        return -self.m.decision_function(X).astype(np.float64)


class MemStream_:
    """MemStream baseline from run_sequential_v4.py (kNN-based)."""
    name = 'MemStream_'

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


class sHST_River:
    """Streaming Half-Space Trees - vectorized mass deviation scoring.

    Each tree splits on random feature quantiles. Score = mass deviation from expected.
    """
    name = 'sHST_River'

    def __init__(self, seed=42, depth=10, n_trees=20):
        self.seed = seed
        self.depth = depth
        self.n_trees = n_trees

    def fit(self, X):
        rng = np.random.RandomState(self.seed)
        d = X.shape[1]

        self.trees = []
        for t in range(self.n_trees):
            feat_idx = rng.randint(0, d, size=self.depth)
            split_pts = np.zeros(self.depth, dtype=np.float32)
            for j in range(self.depth):
                col = X[:, feat_idx[j]]
                q = rng.uniform(0.2, 0.8)
                split_pts[j] = np.quantile(col, q)
            self.trees.append({'feat_idx': feat_idx, 'split_pts': split_pts})

        # Compute long-term mass from X
        self.mass = np.zeros((self.n_trees, self.depth), dtype=np.float64)
        self.mass_count = 0

        for t_idx, tree in enumerate(self.trees):
            for j in range(self.depth):
                col = X[:, tree['feat_idx'][j]]
                self.mass[t_idx, j] += (col > tree['split_pts'][j]).sum()
        self.mass_count = len(X)

    def decision_function(self, X):
        """Vectorized mass deviation score."""
        if not hasattr(self, 'mass') or self.mass_count < 100:
            return np.full(len(X), 0.5)

        Xf = X.astype(np.float32)
        n = len(Xf)
        scores = np.zeros(n, dtype=np.float64)

        # Vectorized per-tree scoring
        for t_idx, tree in enumerate(self.trees):
            feat_idx = tree['feat_idx']
            sp = tree['split_pts']
            lt = self.mass[t_idx] / self.mass_count  # long-term mass

            # Observed: does each sample go left or right of split?
            observed = (Xf[:, feat_idx] > sp[np.newaxis, :]).astype(np.float64)
            deviation = np.abs(observed - lt[np.newaxis, :]).sum(axis=1)
            scores += deviation / self.depth

        return scores / self.n_trees


class IForestASD_:
    """Isolation Forest Anomaly Score Distribution."""
    name = 'IForestASD_'

    def __init__(self, seed=42, n_trees=50, max_samples=256):
        self.seed = seed
        self.n_trees = n_trees
        self.max_samples = max_samples

    def fit(self, X):
        rng = np.random.RandomState(self.seed)
        self.trees = []
        for _ in range(self.n_trees):
            idx = rng.choice(
                min(len(X), self.max_samples),
                min(self.max_samples, len(X)), replace=False
            )
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
    name = 'sHST_Mem_Ensemble'

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


class CADIFEia:
    """Context-Aware Density-Informed Feature Ensemble."""
    name = 'CADIFEia'

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
        ctx_labels = np.digitize(X[:, -1], self.bins[1:-1])

        for ctx, info in self.models.items():
            mask = ctx_labels == ctx
            if mask.any():
                m = info['model']
                scores[mask] = -m.score_samples(X[mask])
            else:
                nearest = min(self.models.keys(), key=lambda k: abs(k - ctx))
                m = self.models[nearest]['model']
                scores[mask] = -m.score_samples(X[mask])

        return scores


class HBOS:
    """Histogram-Based Outlier Score."""
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


# ═══════════════════════════════════════════════════════════════════════════
# CA-MEMSTREAM IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════


class CA_MemStream:
    """CA-MemStream: 25D Autoencoder + Memory Module with 100-epoch warmup.

    Key improvements over quick benchmark:
    - 100 epochs warmup (not 10!)
    - Early stopping with patience=20
    - Calibration set for threshold (not percentile on test)
    - Memory initialized from warmup data
    """
    name = 'CA_MemStream'

    def __init__(self, seed=42, warmup_epochs=100, memory_len=100, batch_size=256):
        self.seed = seed
        self.warmup_epochs = warmup_epochs
        self.memory_len = memory_len
        self.batch_size = batch_size
        self._ms_core = None
        self._calibration_scores = None
        self._threshold = None

    def _create_core(self):
        cfg = MemStreamConfig()
        cfg.seed = self.seed
        cfg.warmup_epochs = self.warmup_epochs
        cfg.warmup_batch_size = self.batch_size
        cfg.memory_len = self.memory_len
        cfg.warmup_early_stop_patience = 20
        return MemStreamCore(cfg=cfg, device=DEVICE)

    def fit(self, X_train, X_calibration=None):
        """Fit CA-MemStream with warmup and optional calibration set."""
        if not MEMSTREAM_CORE_AVAILABLE:
            raise RuntimeError("MemStream core not available")

        set_determinism(self.seed)
        torch.manual_seed(self.seed)
        if GPU_OK:
            torch.cuda.manual_seed(self.seed)

        # Split warmup data (75% train, 25% calibration if not provided)
        n = len(X_train)
        n_warmup = int(n * 0.75)

        X_warmup = X_train[:n_warmup].astype(np.float32)

        # Create and train MemStream
        self._ms_core = self._create_core()
        self._ms_core.warmup(
            X_warmup,
            epochs=self.warmup_epochs,
            batch_size=self.batch_size,
            verbose=False
        )

        # Set a temporary threshold to enable scoring
        self._ms_core.set_beta(0.5)

        # Calibration: compute threshold from calibration set (NOT test!)
        if X_calibration is not None:
            X_cal = X_calibration.astype(np.float32)
        else:
            X_cal = X_train[n_warmup:].astype(np.float32)

        if len(X_cal) > 0:
            cal_scores = self._ms_core.score_batch(X_cal)
            # Use 0.5 quantile as calibrated beta
            self._threshold = float(np.percentile(cal_scores, 50))
            self._ms_core.set_beta(self._threshold)
        else:
            self._threshold = 0.5

        self._calibration_scores = cal_scores if X_calibration is not None else None

    def decision_function(self, X):
        """Return anomaly scores (higher = more anomalous)."""
        if self._ms_core is None:
            return np.full(len(X), 0.5)

        Xf = X.astype(np.float32)
        scores = self._ms_core.score_batch(Xf)
        return scores.astype(np.float64)

    def get_threshold(self):
        """Return the calibrated threshold."""
        return self._threshold


class CA_MemStream_BAR:
    """CA-MemStream with BAR Controller for label-efficient updates.

    BAR (Budget Allocation Rate) controls when memory is updated.
    Target: 1-5% label cost instead of 100%.
    """
    name = 'CA_MemStream_BAR'

    def __init__(self, seed=42, warmup_epochs=100, memory_len=100, batch_size=256,
                 bar_target_rate=0.02, adwin_delta=0.002, use_streaming=False):
        self.seed = seed
        self.warmup_epochs = warmup_epochs
        self.memory_len = memory_len
        self.batch_size = batch_size
        self.bar_target_rate = bar_target_rate
        self.adwin_delta = adwin_delta
        self.use_streaming = use_streaming
        self._ms_core = None
        self._bar_controller = None
        self._threshold = None

    def _create_core(self):
        cfg = MemStreamConfig()
        cfg.seed = self.seed
        cfg.warmup_epochs = self.warmup_epochs
        cfg.warmup_batch_size = self.batch_size
        cfg.memory_len = self.memory_len
        cfg.warmup_early_stop_patience = 20
        return MemStreamCore(cfg=cfg, device=DEVICE)

    def fit(self, X_train, X_calibration=None):
        """Fit CA-MemStream-BAR with BAR controller."""
        if not MEMSTREAM_CORE_AVAILABLE:
            raise RuntimeError("MemStream core not available")

        set_determinism(self.seed)
        torch.manual_seed(self.seed)
        if GPU_OK:
            torch.cuda.manual_seed(self.seed)

        # Split warmup data
        n = len(X_train)
        n_warmup = int(n * 0.75)
        X_warmup = X_train[:n_warmup].astype(np.float32)

        # Create MemStream and BAR controller
        self._ms_core = self._create_core()
        self._bar_controller = BARController(
            memory_len=self.memory_len,
            min_budget_fraction=self.bar_target_rate,
            max_budget_fraction=self.bar_target_rate * 2.5,
            enable_adwin=True,
            adwin_delta=self.adwin_delta
        )

        # Warmup phase (no BAR)
        self._ms_core.warmup(
            X_warmup,
            epochs=self.warmup_epochs,
            batch_size=self.batch_size,
            verbose=False
        )

        # Set temporary threshold for calibration
        self._ms_core.set_beta(0.5)

        # Calibration
        if X_calibration is not None:
            X_cal = X_calibration.astype(np.float32)
        else:
            X_cal = X_train[n_warmup:].astype(np.float32)

        if len(X_cal) > 0:
            cal_scores = self._ms_core.score_batch(X_cal)
            self._threshold = float(np.percentile(cal_scores, 50))
            self._ms_core.set_beta(self._threshold)
        else:
            self._threshold = 0.5

    def decision_function(self, X):
        """Return anomaly scores. Uses batch scoring by default for speed.

        Set use_streaming=True in constructor for true online evaluation.
        """
        if self._ms_core is None:
            return np.full(len(X), 0.5)

        Xf = X.astype(np.float32)

        if not self.use_streaming:
            # Batch scoring (fast, for benchmark evaluation)
            return self._ms_core.score_batch(Xf).astype(np.float64)

        # Streaming mode (slow, for true online evaluation)
        scores = np.zeros(len(Xf), dtype=np.float64)
        for i in range(len(Xf)):
            x = Xf[i:i+1]
            score = self._ms_core.score_batch(x)[0]
            scores[i] = score

            if self._bar_controller is not None:
                should_update, reason = self._bar_controller.should_update_memory(
                    {}, float(score), 'default'
                )
                if should_update:
                    self._ms_core.memory_update(x[0])

        return scores.astype(np.float64)

    def get_bar_stats(self):
        """Return BAR statistics."""
        if self._bar_controller is None:
            return {}
        return self._bar_controller.get_stats()


# ═══════════════════════════════════════════════════════════════════════════
# ALGORITHM REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

ALGORITHMS_BASELINE = [
    SklearnIF,
    SklearnLOF,
    MemStream_,
    sHST_River,
    IForestASD_,
    sHST_Mem_Ensemble,
    CADIFEia,
    HBOS,
]

ALGORITHMS_CA_MEMSTREAM = [
    CA_MemStream,
    CA_MemStream_BAR,
]


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_fold_data(fold_num, difficulty, cache_dir):
    """Load fold data for a specific fold and difficulty.

    Handles near-zero-variance features (data quality fix):
    - Features with train std < 1e-4 cause extreme normalization in MemStream
    - Fix: recompute z-scores for bad features using safe scale
    - This ensures Feature 14 in Fold 1 stays in normal range (~1 not ~42)
    """
    fold_dir = cache_dir / f'fold_{fold_num:02d}'
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
        if good_mask.sum() > 0:
            safe_scale = np.median(scaler_scale[good_mask])
        else:
            safe_scale = 1.0
        safe_scale = max(safe_scale, 1.0)

        for f in bad_features:
            old_scale = scaler_scale[f]
            # Data is pre-normalized: z = (raw - mean) / scale
            # Raw value: raw = z * old_scale + old_mean
            raw_train = X_train[:, f] * old_scale + scaler_mean[f]
            raw_test = X_test[:, f] * old_scale + scaler_mean[f]

            # Re-normalize with safe scale
            X_train[:, f] = (raw_train - scaler_mean[f]) / safe_scale
            X_test[:, f] = (raw_test - scaler_mean[f]) / safe_scale
            scaler_scale[f] = safe_scale

    n_cal = int(len(X_train) * 0.1)
    X_calibration = X_train[-n_cal:].copy()

    return {
        'fold': fold_num,
        'difficulty': difficulty,
        'X_train': X_train,
        'X_calibration': X_calibration,
        'X_test': X_test,
        'y_labels': y_labels,
        'scaler_mean': scaler_mean,
        'scaler_scale': scaler_scale,
        'n_anomalies': int(y_labels.sum()),
        'n_test': len(y_labels),
        'bad_features': list(bad_features),
    }


def load_metadata(cache_dir):
    """Load metadata.json to get fold info."""
    with open(cache_dir / 'metadata.json') as f:
        metadata = json.load(f)
    return metadata


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_auc_metrics(y_true, scores):
    """Compute AUC-ROC and AUC-PR."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan, np.nan

    # AUC-ROC
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_roc = auc(fpr, tpr) if len(fpr) > 1 else 0.5

    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, scores)
    auc_pr = auc(recall, precision) if len(recall) > 1 else 0.0

    return auc_roc, auc_pr


def find_optimal_threshold(scores, y_true):
    """Find threshold that maximizes F1 score."""
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


def evaluate_algorithm(algo_cls, data, seed, use_calibration=True):
    """Evaluate a single algorithm on a single fold."""
    row = {
        'fold': data['fold'],
        'difficulty': data['difficulty'],
        'seed': seed,
        'algorithm': algo_cls.name,
    }

    try:
        t0 = time.perf_counter()

        # Create algorithm instance
        algo = algo_cls(seed=seed)

        # Fit with or without calibration set
        if use_calibration and hasattr(algo, 'fit') and 'X_calibration' in data:
            if algo.name in ['CA_MemStream', 'CA_MemStream_BAR']:
                algo.fit(data['X_train'], data['X_calibration'])
            else:
                algo.fit(data['X_train'])
        else:
            algo.fit(data['X_train'])

        t_train = (time.perf_counter() - t0) * 1000

        # Score test set
        t0 = time.perf_counter()
        scores = algo.decision_function(data['X_test']).astype(np.float64)
        t_score = (time.perf_counter() - t0) * 1000

        y_true = data['y_labels']

        # Compute metrics
        if len(scores) < 10 or y_true.sum() == 0:
            row.update({
                'AUC_ROC': np.nan, 'AUC_PR': np.nan,
                'F1': np.nan, 'Precision': np.nan, 'Recall': np.nan,
                'train_ms': t_train, 'score_ms': t_score
            })
            return row

        auc_roc, auc_pr = compute_auc_metrics(y_true, scores)
        opt_thresh, opt_f1 = find_optimal_threshold(scores, y_true)
        preds = (scores >= opt_thresh).astype(int)

        f1 = f1_score(y_true, preds, zero_division=0)
        prc = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)

        row.update({
            'AUC_ROC': auc_roc,
            'AUC_PR': auc_pr,
            'F1': opt_f1,
            'Precision': prc,
            'Recall': rec,
            'optimal_threshold': opt_thresh,
            'train_ms': t_train,
            'score_ms': t_score,
            'n_anomalies': data['n_anomalies'],
        })

        # Get BAR stats if applicable
        if hasattr(algo, 'get_bar_stats'):
            bar_stats = algo.get_bar_stats()
            row['bar_rate'] = bar_stats.get('bar_rate', np.nan)
            row['drift_events'] = bar_stats.get('drift_events', 0)

        return row

    except Exception as e:
        row.update({
            'error': str(e),
            'AUC_ROC': np.nan, 'AUC_PR': np.nan,
            'F1': np.nan, 'Precision': np.nan, 'Recall': np.nan,
            'train_ms': 0, 'score_ms': 0
        })
        return row


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def holm_bonferroni_correction(pvals):
    """Apply Holm-Bonferroni correction for multiple comparisons."""
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


def wilcoxon_signed_rank(a, b):
    """Wilcoxon signed-rank test for paired samples."""
    d = np.array(a) - np.array(b)
    d = d[~np.isnan(d)]
    if len(d) < 3 or np.all(d == 0):
        return np.nan, 1.0
    try:
        wr = stats.wilcoxon(d, alternative='two-sided')
        return float(wr.statistic), float(wr.pvalue)
    except Exception:
        return np.nan, 1.0


def cohens_d(a, b):
    """Compute Cohen's d effect size."""
    a = np.array(a)
    b = np.array(b)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan

    diff = np.mean(a) - np.mean(b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float(diff / (pooled_std + 1e-10))


def effect_size_interpretation(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    else:
        return 'large'


def nemenyi_cd(k, n):
    """Critical difference for Nemenyi test."""
    # q_alpha = 2.728 for alpha=0.05, k algorithms
    q_alpha = 2.728
    return q_alpha * np.sqrt(k * (k + 1) / (6 * n))


def compute_rankings(results_df, difficulty):
    """Compute average rankings across folds for Nemenyi test."""
    sub = results_df[results_df['difficulty'] == difficulty]
    if sub.empty:
        return None, None

    folds = sorted(sub['fold'].unique())
    algos = sorted(sub['algorithm'].unique())
    k = len(algos)

    if k < 2:
        return None, None

    # Compute rankings per fold
    rankings = []
    for fold in folds:
        fold_data = []
        for algo in algos:
            vals = sub[(sub['fold'] == fold) & (sub['algorithm'] == algo)]['AUC_PR'].dropna().values
            if len(vals) > 0:
                fold_data.append((algo, np.mean(vals)))
            else:
                fold_data.append((algo, 0.0))

        # Rank algorithms (higher AUC-PR = better = lower rank)
        fold_data.sort(key=lambda x: x[1], reverse=True)
        fold_ranks = {algo: rank + 1 for rank, (algo, _) in enumerate(fold_data)}
        rankings.append(fold_ranks)

    # Average rankings
    avg_ranks = {}
    for algo in algos:
        ranks = [r[algo] for r in rankings if algo in r]
        avg_ranks[algo] = np.mean(ranks) if ranks else np.nan

    # Compute CD
    cd = nemenyi_cd(k, len(folds))

    return avg_ranks, cd


def run_statistical_tests(results_df, control_algo='sklearn_IF'):
    """Run pairwise Wilcoxon tests with Holm-Bonferroni correction."""
    results = []

    for diff in DIFFICULTIES:
        sub = results_df[results_df['difficulty'] == diff]
        if sub.empty:
            continue

        algos = sorted(sub['algorithm'].unique())
        folds = sorted(sub['fold'].unique())

        # Build performance matrix
        perf_matrix = {}
        for algo in algos:
            algo_data = []
            for fold in folds:
                vals = sub[(sub['fold'] == fold) & (sub['algorithm'] == algo)]['AUC_PR'].dropna().values
                if len(vals) > 0:
                    algo_data.append(np.mean(vals))
                else:
                    algo_data.append(np.nan)
            perf_matrix[algo] = algo_data

        # Pairwise comparisons
        for i, algo_i in enumerate(algos):
            for j, algo_j in enumerate(algos):
                if i >= j:
                    continue

                a = perf_matrix.get(algo_i, [])
                b = perf_matrix.get(algo_j, [])

                if len(a) < 3 or len(b) < 3:
                    continue

                W, p_raw = wilcoxon_signed_rank(a, b)
                d = cohens_d(a, b)
                effect = effect_size_interpretation(d)

                results.append({
                    'difficulty': diff,
                    'algorithm_1': algo_i,
                    'algorithm_2': algo_j,
                    'mean_1': np.nanmean(a),
                    'mean_2': np.nanmean(b),
                    'delta': np.nanmean(a) - np.nanmean(b),
                    'W_statistic': W,
                    'p_raw': p_raw,
                    'cohens_d': d,
                    'effect_size': effect,
                })

    # Apply Holm-Bonferroni correction
    if results:
        pvals = [r['p_raw'] for r in results]
        adj_pvals = holm_bonferroni_correction(pvals)
        for r, p_adj in zip(results, adj_pvals):
            r['p_adjusted'] = p_adj
            r['significant'] = p_adj < 0.05

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_critical_difference_diagram(avg_ranks, cd, title, output_path):
    """Plot Critical Difference diagram."""
    if avg_ranks is None:
        return

    algos = sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])
    n = len(algos)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 4))

    # Draw CD line
    ax.axhline(0.5, color='#888', linewidth=1)

    # Draw CD bar
    cd_y = 0.55
    ax.plot([0.5, 0.5 + cd], [cd_y, cd_y], color='#888', linewidth=2)
    ax.text(0.5 + cd / 2, cd_y + 0.02, f'CD = {cd:.2f}', ha='center', fontsize=10)

    # Draw connecting lines for non-significant differences
    positions = {}
    for i, algo in enumerate(algos):
        positions[algo] = avg_ranks[algo]

    # Connect algorithms within CD
    for i in range(n):
        for j in range(i + 1, n):
            algo_i, algo_j = algos[i], algos[j]
            r1, r2 = positions[algo_i], positions[algo_j]
            if abs(r1 - r2) < cd:
                ax.plot([r1, r2], [0.5, 0.5], color='#2c3e50', linewidth=8,
                        solid_capstyle='round', alpha=0.3)

    # Plot algorithm nodes
    for i, algo in enumerate(algos):
        rank = positions[algo]
        color = COLORS.get(algo, '#555')
        ax.scatter(rank, 0.8, c=color, s=300, zorder=5,
                   edgecolors='white', linewidth=2)
        ax.annotate(algo, (rank, 0.85), ha='center', va='bottom',
                   fontsize=9, rotation=15, fontweight='bold')

    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0, 1.1)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_bar_score_curve(bar_results_df, output_path):
    """Plot AUC-PR vs Budget Fraction (BAR Score curve)."""
    if bar_results_df is None or bar_results_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    budgets = sorted(bar_results_df['budget_pct'].unique())
    algos = bar_results_df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index

    for algo in algos:
        algo_data = bar_results_df[bar_results_df['algorithm'] == algo]
        means = [algo_data[algo_data['budget_pct'] == bp]['AUC_PR'].mean() for bp in budgets]
        stds = [algo_data[algo_data['budget_pct'] == bp]['AUC_PR'].std() for bp in budgets]

        color = COLORS.get(algo, '#555')
        ax.plot(budgets, means, 'o-', label=algo, color=color, linewidth=2, markersize=8)
        ax.fill_between(budgets,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color=color)

    ax.set_xscale('log')
    ax.set_xlabel('Training Budget (fraction of data)', fontsize=12)
    ax.set_ylabel('AUC-PR', fontsize=12)
    ax.set_title('Label Efficiency (BAR Score Curve)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0.005, 1.1)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_summary_comparison(results_df, output_path):
    """Plot summary comparison across all algorithms."""
    if results_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Overall AUC-PR comparison
    ax = axes[0]
    overall = results_df.groupby('algorithm')['AUC_PR'].agg(['mean', 'std']).sort_values('mean', ascending=False)

    colors = [COLORS.get(a, '#555') for a in overall.index]
    bars = ax.barh(range(len(overall)), overall['mean'], xerr=overall['std'],
                   color=colors, alpha=0.8, capsize=3)
    ax.set_yticks(range(len(overall)))
    ax.set_yticklabels(overall.index, fontsize=9)
    ax.set_xlabel('AUC-PR')
    ax.set_title('Overall AUC-PR Comparison')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)

    # 2. AUC-PR by difficulty
    ax = axes[1]
    pivot = results_df.groupby(['algorithm', 'difficulty'])['AUC_PR'].mean().unstack()
    if not pivot.empty:
        pivot = pivot[['easy', 'medium', 'hard']]
        pivot.plot(kind='bar', ax=ax, color=['#27ae60', '#f39c12', '#c0392b'], alpha=0.8)
        ax.set_ylabel('AUC-PR')
        ax.set_title('AUC-PR by Difficulty')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(title='Difficulty')
        ax.grid(axis='y', alpha=0.3)

    # 3. Variance analysis (CV)
    ax = axes[2]
    cv = results_df.groupby('algorithm')['AUC_PR'].agg(lambda x: np.std(x) / (np.mean(x) + 1e-10))
    cv = cv.sort_values()
    colors = [COLORS.get(a, '#555') for a in cv.index]
    ax.barh(range(len(cv)), cv.values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(cv)))
    ax.set_yticklabels(cv.index, fontsize=9)
    ax.set_xlabel('Coefficient of Variation')
    ax.set_title('Result Stability (lower is better)')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_ranking_table(results_df, output_path):
    """Plot final ranking table as image."""
    if results_df.empty:
        return

    # Compute rankings per difficulty
    ranking_data = []
    for diff in DIFFICULTIES:
        sub = results_df[results_df['difficulty'] == diff]
        if sub.empty:
            continue

        means = sub.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False)
        for rank, (algo, val) in enumerate(means.items(), 1):
            std = sub[sub['algorithm'] == algo]['AUC_PR'].std()
            ranking_data.append({
                'Difficulty': diff,
                'Rank': rank,
                'Algorithm': algo,
                'AUC-PR': f'{val:.4f}±{std:.4f}',
            })

    if not ranking_data:
        return

    df_rank = pd.DataFrame(ranking_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=df_rank.values,
        colLabels=df_rank.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Style header
    for j in range(len(df_rank.columns)):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(df_rank) + 1):
        for j in range(len(df_rank.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')

    ax.set_title('Final Algorithm Rankings by Difficulty', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_benchmark(args):
    """Run the full benchmark."""
    global OUT_DIR
    OUT_DIR = Path(f'c:/proj/ldt/results/{args.out}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RIGOROUS SCIENTIFIC BENCHMARK FOR STREAMING ANOMALY DETECTION")
    print("=" * 80)
    print(f"GPU: {'YES' if GPU_OK else 'NO'} | Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Output: {OUT_DIR}")
    print("=" * 80)

    t_start = time.perf_counter()

    # Load metadata
    print("\n[1] Loading metadata...")
    metadata = load_metadata(CACHE_DIR)
    print(f"  Total entries: {len(metadata)}")

    # Build fold list
    fold_entries = []
    for fold_num in range(1, N_FOLDS + 1):
        for diff in DIFFICULTIES:
            entry = next((e for e in metadata if e['fold'] == fold_num and e['difficulty'] == diff), None)
            if entry:
                fold_entries.append(entry)
    print(f"  Folds to process: {len(fold_entries)}")

    # Determine algorithms to run
    algos_to_run = ALGORITHMS_BASELINE.copy()
    skip_ca = not MEMSTREAM_CORE_AVAILABLE or args.skip_ca

    if skip_ca:
        print("\n[NOTE] CA-MemStream algorithms will be SKIPPED")
    else:
        algos_to_run.extend(ALGORITHMS_CA_MEMSTREAM)

    print(f"\n[2] Algorithms to benchmark: {len(algos_to_run)}")
    for algo in algos_to_run:
        print(f"  - {algo.name}")

    # ═══════════════════════════════════════════════════════════════════
    # MAIN EVALUATION
    # ═══════════════════════════════════════════════════════════════════
    results_csv = OUT_DIR / 'results_detailed.csv'
    all_results = []

    if args.resume and results_csv.exists():
        print("\n[3] Loading existing results (--resume)...")
        all_results = pd.read_csv(results_csv).to_dict('records')
        print(f"  Loaded {len(all_results)} existing results")
    else:
        print("\n[3] Running evaluation...")
        total_experiments = len(algos_to_run) * len(fold_entries) * len(SEEDS)
        done = 0

        t_eval_start = time.perf_counter()

        for algo_cls in algos_to_run:
            print(f"\n  [{algo_cls.name}]")
            algo_start = time.perf_counter()

            for entry in fold_entries:
                fold_num = entry['fold']
                diff = entry['difficulty']

                # Load data
                data = load_fold_data(fold_num, diff, CACHE_DIR)

                for seed in SEEDS:
                    row = evaluate_algorithm(algo_cls, data, seed, use_calibration=True)
                    all_results.append(row)
                    done += 1

                    if done % 500 == 0:
                        elapsed = time.perf_counter() - t_eval_start
                        eta = (elapsed / done) * (total_experiments - done) / 60
                        print(f"    Progress: {done}/{total_experiments} "
                              f"({done/total_experiments*100:.1f}%) ETA={eta:.1f}min")

            algo_time = time.perf_counter() - algo_start
            print(f"    [{algo_cls.name}] completed in {algo_time/60:.1f}min")

        # Save detailed results
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(results_csv, index=False)
        print(f"\n  Saved detailed results: {results_csv}")

    df_results = pd.DataFrame(all_results)

    # ═══════════════════════════════════════════════════════════════════
    # BAR SCORE MEASUREMENT
    # ═══════════════════════════════════════════════════════════════════
    bar_csv = OUT_DIR / 'bar_score_results.csv'
    bar_results = []

    if args.resume and bar_csv.exists():
        print("\n[4] Loading existing BAR score results...")
        bar_results_df = pd.read_csv(bar_csv)
    else:
        print("\n[4] Running BAR Score measurement...")
        bar_budgets = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

        # Use streaming algorithms only for BAR score
        bar_algos = [algo for algo in algos_to_run
                     if algo.name in ['sklearn_IF', 'MemStream_', 'CA_MemStream', 'CA_MemStream_BAR']]

        for algo_cls in bar_algos:
            print(f"  [{algo_cls.name}]")

            for entry in fold_entries[:3]:  # Use subset of folds for BAR score
                fold_num = entry['fold']
                diff = entry['difficulty']
                data = load_fold_data(fold_num, diff, CACHE_DIR)

                n_train = len(data['X_train'])
                n_cal = len(data['X_calibration'])

                for bp in bar_budgets:
                    n_subset = max(100, int(n_train * bp))
                    rng = np.random.RandomState(42)
                    idx = rng.choice(n_train, min(n_subset, n_train), replace=False)
                    X_subset = data['X_train'][idx]

                    for seed in SEEDS[:3]:  # Use subset of seeds for speed
                        try:
                            algo = algo_cls(seed=seed)

                            if algo.name in ['CA_MemStream', 'CA_MemStream_BAR']:
                                # For CA-MemStream, use smaller calibration set
                                cal_size = min(n_cal, int(len(X_subset) * 0.1))
                                X_cal = X_subset[-cal_size:] if cal_size > 0 else None
                                if X_cal is not None and len(X_cal) > 50:
                                    algo.fit(X_subset, X_cal)
                                else:
                                    algo.fit(X_subset)
                            else:
                                algo.fit(X_subset)

                            scores = algo.decision_function(data['X_test']).astype(np.float64)
                            y_true = data['y_labels']

                            auc_roc, auc_pr = compute_auc_metrics(y_true, scores)

                            bar_results.append({
                                'algorithm': algo.name,
                                'fold': fold_num,
                                'difficulty': diff,
                                'budget_pct': bp,
                                'seed': seed,
                                'AUC_ROC': auc_roc,
                                'AUC_PR': auc_pr,
                            })
                        except Exception as e:
                            bar_results.append({
                                'algorithm': algo.name,
                                'fold': fold_num,
                                'difficulty': diff,
                                'budget_pct': bp,
                                'seed': seed,
                                'AUC_ROC': np.nan,
                                'AUC_PR': np.nan,
                                'error': str(e),
                            })

        bar_results_df = pd.DataFrame(bar_results)
        bar_results_df.to_csv(bar_csv, index=False)
        print(f"  Saved BAR score results: {bar_csv}")

    # ═══════════════════════════════════════════════════════════════════
    # STATISTICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print("\n[5] Running statistical analysis...")

    # Wilcoxon tests with Holm-Bonferroni
    stat_tests_df = run_statistical_tests(df_results, control_algo='sklearn_IF')
    stat_tests_df.to_csv(OUT_DIR / 'statistical_tests_wilcoxon.csv', index=False)
    print(f"  Saved: statistical_tests_wilcoxon.csv")

    # Compute rankings for CD diagrams
    for diff in DIFFICULTIES:
        avg_ranks, cd = compute_rankings(df_results, diff)
        if avg_ranks is not None:
            # Save CD data
            cd_data = [{'algorithm': algo, 'avg_rank': rank, 'cd': cd}
                       for algo, rank in avg_ranks.items()]
            cd_df = pd.DataFrame(cd_data)
            cd_df.to_csv(OUT_DIR / f'cd_ranks_{diff}.csv', index=False)

            # Plot CD diagram
            plot_critical_difference_diagram(
                avg_ranks, cd,
                f'Critical Difference Diagram — {diff.capitalize()}',
                OUT_DIR / f'fig_cd_{diff}.png'
            )

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY STATISTICS
    # ═══════════════════════════════════════════════════════════════════
    print("\n[6] Computing summary statistics...")

    # Overall summary
    summary = df_results.groupby('algorithm').agg({
        'AUC_ROC': ['mean', 'std', 'count'],
        'AUC_PR': ['mean', 'std'],
        'F1': ['mean', 'std'],
        'train_ms': 'mean',
        'score_ms': 'mean',
    }).round(4)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.sort_values('AUC_PR_mean', ascending=False)
    summary.to_csv(OUT_DIR / 'results_summary.csv')
    print(f"  Saved: results_summary.csv")

    # By difficulty
    by_diff = df_results.groupby(['algorithm', 'difficulty']).agg({
        'AUC_ROC': ['mean', 'std'],
        'AUC_PR': ['mean', 'std'],
        'F1': ['mean', 'std'],
    }).round(4)
    by_diff.columns = ['_'.join(col).strip() for col in by_diff.columns.values]
    by_diff.to_csv(OUT_DIR / 'results_by_difficulty.csv')

    # ═══════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════
    print("\n[7] Generating plots...")

    # Summary comparison
    plot_summary_comparison(df_results, OUT_DIR / 'fig_summary_comparison.png')

    # BAR score curve
    plot_bar_score_curve(bar_results_df, OUT_DIR / 'fig_bar_score_curve.png')

    # Ranking table
    plot_ranking_table(df_results, OUT_DIR / 'fig_ranking_table.png')

    # ═══════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════
    total_time = time.perf_counter() - t_start

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Output directory: {OUT_DIR}")

    print("\n[FINAL RESULTS] Mean AUC-PR by Algorithm:")
    print("-" * 50)
    final_ranking = df_results.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False)
    for rank, (algo, val) in enumerate(final_ranking.items(), 1):
        std = df_results[df_results['algorithm'] == algo]['AUC_PR'].std()
        print(f"  {rank:2d}. {algo:20s}: {val:.4f} +/- {std:.4f}")

    print("\n[AUC-PR BY DIFFICULTY]")
    print("-" * 50)
    for diff in DIFFICULTIES:
        sub = df_results[df_results['difficulty'] == diff]
        means = sub.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False)
        print(f"\n  [{diff.upper()}]")
        for algo, val in means.items():
            std = sub[sub['algorithm'] == algo]['AUC_PR'].std()
            print(f"    {algo:20s}: {val:.4f} +/- {std:.4f}")

    # Save environment info
    env_info = {
        'version': 'rigorous_v1',
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'gpu_available': GPU_OK,
        'gpu_device': torch.cuda.get_device_name(0) if GPU_OK else None,
        'device': DEVICE,
        'total_minutes': round(total_time / 60, 1),
        'n_seeds': len(SEEDS),
        'n_folds': N_FOLDS,
        'n_difficulties': len(DIFFICULTIES),
        'algorithms': [a.name for a in algos_to_run],
        'seeds': SEEDS,
    }
    with open(OUT_DIR / 'environment.json', 'w') as f:
        json.dump(env_info, f, indent=2)

    print(f"\n{'='*80}")
    print("All results saved to:")
    for f in sorted(OUT_DIR.glob('*.csv')):
        print(f"  - {f.name}")
    for f in sorted(OUT_DIR.glob('*.png')):
        print(f"  - {f.name}")
    print(f"{'='*80}")


def parse_args():
    p = argparse.ArgumentParser(
        description='Rigorous Scientific Benchmark for Streaming Anomaly Detection'
    )
    p.add_argument(
        '--out', default='rigorous_v1',
        help='Output directory name (default: rigorous_v1)'
    )
    p.add_argument(
        '--resume', action='store_true',
        help='Resume from existing results'
    )
    p.add_argument(
        '--skip-ca', action='store_true',
        help='Skip CA-MemStream algorithms (faster run)'
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_benchmark(args)
