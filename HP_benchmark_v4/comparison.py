#!/usr/bin/env python3
"""
================================================================================
STEP 3: COMPARISON — Compare with Other Methods (File 3/3)
================================================================================

So sanh MemStream (best config) voi cac phuong phap anomaly detection:
  1. IsolationForest (sklearn) —offline, non-streaming
  2. Random Cut Forest (RCF) — streaming
  3. MemStream (v4, best config) — streaming
  4. Half-Space Trees — streaming
  5. LODA — streaming

MUC TIEU:
  - Cho thay vi tri cua MemStream trong landscape anomaly detection
  - Chung minh loi ich cua streaming vs offline (neu co)
  - Co the mo rong them nhieu methods neu can

LUU Y:
  - Tat ca methods deu danh gia tren cung validation set
  - Dung 34D features nhu MemStream
  - Neu co GPU, them GPU-accelerated methods
"""
from __future__ import annotations

import os
import sys
import json
import time
import io
import pickle
import hashlib
import hmac
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
LOGGER = logging.getLogger("comparison")

sys.path.insert(0, r"c:\proj\ldt\HP_benchmark_v4")
from benchmark_core import (
    MemStreamPipeline, extract_features_from_parquet,
    evaluate_scores, find_best_threshold,
    plot_score_distribution, plot_score_timeseries,
    plot_pointwise_scores, plot_method_score_heatmap,
    plot_detection_timeseries,
)


# ---------------------------------------------------------------------------
# Method 1: IsolationForest (sklearn)
# ---------------------------------------------------------------------------
def run_isolation_forest(
    X_train: np.ndarray,
    X_val: np.ndarray,
    gt_mask: np.ndarray,
    n_estimators: int = 200,
    max_samples: int = 256,
    seed: int = 42,
) -> Dict:
    """IsolationForest (sklearn) — offline, non-streaming."""
    from sklearn.ensemble import IsolationForest

    t0 = time.time()
    try:
        clf = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=0.08,
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train)
        scores = -clf.score_samples(X_val)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        best_t, best_f1 = find_best_threshold(scores, gt_mask)
        metrics = evaluate_scores(scores, gt_mask, threshold=best_t)
        elapsed = time.time() - t0
        LOGGER.info("  [IsolationForest] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
                   metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], elapsed)
        result = {
            "method": "IsolationForest",
            "category": "offline",
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fpr": metrics["fpr"],
            "acc": metrics["acc"],
            "best_threshold": float(best_t),
            "score_normal_mean": float(metrics["score_normal_mean"]),
            "score_anomaly_mean": float(metrics["score_anomaly_mean"]),
            "separation_ratio": float(metrics["separation_ratio"]),
            "time_s": elapsed,
            "params": {"n_estimators": n_estimators, "max_samples": max_samples},
            "_raw_scores": scores,
        }
    except Exception as e:
        LOGGER.error("  [IsolationForest] Error: %s", e)
        return {
            "method": "IsolationForest", "error": str(e),
            "_raw_scores": np.array([]),
        }

    return result


# ---------------------------------------------------------------------------
# Method 2: Random Cut Forest (RCF) — Streaming
# ---------------------------------------------------------------------------
def run_rcf(
    X_train: np.ndarray,
    X_val: np.ndarray,
    gt_mask: np.ndarray,
    n_trees: int = 50,
    tree_size: int = 256,
    seed: int = 42,
) -> Dict:
    """Random Cut Forest — streaming (AWS/robustcutforest)."""
    try:
        from robustcutforest import RobustCutForest
    except ImportError:
        try:
            from sklearn.ensemble import RandomTreesEmbedding
            LOGGER.warning("  [RCF] robustcutforest not available, using sklearn RandomTreesEmbedding")
            return _rcf_sklearn_fallback(X_train, X_val, gt_mask, n_trees, tree_size, seed)
        except Exception as e:
            LOGGER.warning("  [RCF] Not available: %s", e)
            return {"method": "Random Cut Forest", "error": "not available"}

    t0 = time.time()
    try:
        rcf = RobustCutForest(
            n_trees=n_trees,
            tree_size=tree_size,
            random_state=seed,
        )
        rcf.fit(X_train)
        scores = rcf.score(X_val)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        best_t, best_f1 = find_best_threshold(scores, gt_mask)
        metrics = evaluate_scores(scores, gt_mask, threshold=best_t)
        elapsed = time.time() - t0
        LOGGER.info("  [RCF] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
                   metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], elapsed)
        return {
            "method": "Random Cut Forest (RCF)",
            "category": "streaming",
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fpr": metrics["fpr"],
            "acc": metrics["acc"],
            "best_threshold": float(best_t),
            "score_normal_mean": float(metrics["score_normal_mean"]),
            "score_anomaly_mean": float(metrics["score_anomaly_mean"]),
            "separation_ratio": float(metrics["separation_ratio"]),
            "time_s": elapsed,
            "params": {"n_trees": n_trees, "tree_size": tree_size},
            "_raw_scores": scores,
        }
    except Exception as e:
        LOGGER.error("  [RCF] Error: %s", e)
        return {"method": "Random Cut Forest", "error": str(e), "_raw_scores": np.array([])}


def _rcf_sklearn_fallback(
    X_train: np.ndarray,
    X_val: np.ndarray,
    gt_mask: np.ndarray,
    n_trees: int,
    tree_size: int,
    seed: int,
) -> Dict:
    """Fallback RCF using sklearn."""
    from sklearn.ensemble import RandomTreesEmbedding
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    try:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_va = scaler.transform(X_val)

        rte = RandomTreesEmbedding(
            n_estimators=n_trees,
            max_depth=10,
            max_samples=tree_size,
            random_state=seed,
            n_jobs=-1,
        )
        rte.fit(X_tr)
        scores = -np.log1p(rte.score_samples(X_va).clip(1e-10))
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        best_t, best_f1 = find_best_threshold(scores, gt_mask)
        metrics = evaluate_scores(scores, gt_mask, threshold=best_t)
        elapsed = time.time() - t0
        LOGGER.info("  [RCF-Sklearn] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
                   metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], elapsed)
        return {
            "method": "Random Cut Forest (sklearn)",
            "category": "streaming",
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fpr": metrics["fpr"],
            "acc": metrics["acc"],
            "best_threshold": float(best_t),
            "score_normal_mean": float(metrics["score_normal_mean"]),
            "score_anomaly_mean": float(metrics["score_anomaly_mean"]),
            "separation_ratio": float(metrics["separation_ratio"]),
            "time_s": elapsed,
            "params": {"n_trees": n_trees, "tree_size": tree_size},
            "_raw_scores": scores,
        }
    except Exception as e:
        LOGGER.error("  [RCF-Sklearn] Error: %s", e)
        return {"method": "RCF-Sklearn", "error": str(e), "_raw_scores": np.array([])}


# ---------------------------------------------------------------------------
# Batch-size helpers
# ---------------------------------------------------------------------------
def _safe_batch_size() -> int:
    """Compute a safe batch_size that won't OOM RAM.

    Caps at 1024 because:
    - MemStream AE is tiny (~18 KB); beyond 1024 the gradient step is
      too coarse and may hurt convergence.
    - RAM-wise, 88 GB host → plenty of headroom at 1024.
    - GPU-wise, RTX 3090 Ti 22.5 GB → tiny model means OOM impossible.
    """
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        safe = int(available_gb * 0.5)
        return max(64, min(safe, 2048))
    except Exception:
        return 256


# ---------------------------------------------------------------------------
# Method 5: MemStream v4 (best config)
# ---------------------------------------------------------------------------
def run_memstream_v4(
    X_train: np.ndarray,
    hours_train: np.ndarray,
    dows_train: np.ndarray,
    rcs_train: np.ndarray,
    nb_train: np.ndarray,
    X_val: np.ndarray,
    hours_val: np.ndarray,
    dows_val: np.ndarray,
    rcs_val: np.ndarray,
    nb_val: np.ndarray,
    gt_mask: np.ndarray,
    best_config: Dict,
    epochs: int = 5000,
    seed: int = 42,
    batch_size: int | None = None,
) -> Dict:
    """MemStream v4 (best config) — streaming."""
    t0 = time.time()
    _bs = _safe_batch_size() if batch_size is None else batch_size
    try:
        pipeline = MemStreamPipeline(
            d=34,
            out_dim=best_config.get("out_dim", 68),
            memory_len=best_config["memory_len"],
            k=best_config["k"],
            gamma=best_config["gamma"],
            beta=best_config["beta"],
            noise_std=best_config["noise_std"],
            lr=best_config["lr"],
            epochs=epochs,
            batch_size=_bs,
            seed=seed,
            cb_warmup=min(4096, best_config["memory_len"] * 4),
            verbose=False,
            adam_betas=tuple(best_config.get("adam_betas", [0.9, 0.999])),
        )
        pipeline.train(
            X_train, hours_train, dows_train, rcs_train, nb_train,
            X_warmup=X_train[:best_config["memory_len"]],
            hours_warmup=hours_train[:best_config["memory_len"]],
            dows_warmup=dows_train[:best_config["memory_len"]],
            rcs_warmup=rcs_train[:best_config["memory_len"]],
            nb_warmup=nb_train[:best_config["memory_len"]],
        )

        adj_scores, ms_metrics = pipeline.score_stream(
            X_val, hours_val, dows_val, rcs_val, nb_val,
            gt_mask=gt_mask,
        )
        best_t, best_f1 = find_best_threshold(adj_scores, gt_mask)
        metrics = evaluate_scores(adj_scores, gt_mask, threshold=best_t)
        elapsed = time.time() - t0

        LOGGER.info("  [MemStream v4] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
                   metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], elapsed)

        # Detected anomaly indices (model prediction)
        detected_mask = (adj_scores >= best_t).astype(np.int32)

        # Save epoch losses
        epoch_losses = list(pipeline.epoch_losses)
        n_mem_updates = ms_metrics.get("n_memory_updates", 0)

        return {
            "method": "MemStream v4",
            "category": "streaming",
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fpr": metrics["fpr"],
            "acc": metrics["acc"],
            "best_threshold": float(best_t),
            "score_normal_mean": float(metrics["score_normal_mean"]),
            "score_anomaly_mean": float(metrics["score_anomaly_mean"]),
            "separation_ratio": float(metrics["separation_ratio"]),
            "time_s": elapsed,
            "params": {
                "memory_len": best_config["memory_len"],
                "k": best_config["k"],
                "gamma": best_config["gamma"],
                "beta": best_config["beta"],
                "noise_std": best_config["noise_std"],
                "lr": best_config["lr"],
                "out_dim": best_config.get("out_dim", 68),
                "epochs": epochs,
                "batch_size": _bs,
            },
            "_raw_scores": adj_scores,
            "_detected_mask": detected_mask,
            "_gt_mask": gt_mask,
            "_epoch_losses": epoch_losses,
            "_n_memory_updates": n_mem_updates,
            "_n_tp": int(metrics["tp"]),
            "_n_fp": int(metrics["fp"]),
            "_n_tn": int(metrics["tn"]),
            "_n_fn": int(metrics["fn"]),
        }
    except Exception as e:
        LOGGER.error("  [MemStream v4] Error: %s", e)
        import traceback
        traceback.print_exc()
        return {"method": "MemStream v4", "error": str(e), "_raw_scores": np.array([])}


# ---------------------------------------------------------------------------
# Method 6: HTrees (Half-Space Trees) — Streaming
# ---------------------------------------------------------------------------
def run_hstrees(
    X_train: np.ndarray,
    X_val: np.ndarray,
    gt_mask: np.ndarray,
    window_size: int = 100,
    n_trees: int = 25,
    max_depth: int = 15,
    seed: int = 42,
) -> Dict:
    """Half-Space Trees — streaming (pysad)."""
    try:
        from pysad.models import HSTrees
    except ImportError:
        LOGGER.warning("  [HSTrees] pysad not available")
        return {"method": "HSTrees", "error": "pysad not installed"}

    t0 = time.time()
    try:
        model = HSTrees(
            window_size=window_size,
            num_trees=n_trees,
            max_depth=max_depth,
            random_state=seed,
        )
        model.fit(X_train)
        scores = model.score(X_val)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        best_t, best_f1 = find_best_threshold(scores, gt_mask)
        metrics = evaluate_scores(scores, gt_mask, threshold=best_t)
        elapsed = time.time() - t0
        LOGGER.info("  [HSTrees] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
                   metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], elapsed)
        return {
            "method": "Half-Space Trees",
            "category": "streaming",
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fpr": metrics["fpr"],
            "acc": metrics["acc"],
            "best_threshold": float(best_t),
            "score_normal_mean": float(metrics["score_normal_mean"]),
            "score_anomaly_mean": float(metrics["score_anomaly_mean"]),
            "separation_ratio": float(metrics["separation_ratio"]),
            "time_s": elapsed,
            "params": {"window_size": window_size, "n_trees": n_trees, "max_depth": max_depth},
            "_raw_scores": scores,
        }
    except Exception as e:
        LOGGER.error("  [HSTrees] Error: %s", e)
        return {"method": "HSTrees", "error": str(e), "_raw_scores": np.array([])}


# ---------------------------------------------------------------------------
# Method 7: LODA — Streaming
# ---------------------------------------------------------------------------
def run_loda(
    X_train: np.ndarray,
    X_val: np.ndarray,
    gt_mask: np.ndarray,
    num_bins: int = 10,
    num_cuts: int = 100,
    seed: int = 42,
) -> Dict:
    """LODA — streaming."""
    try:
        from pysad.models import LODA
    except ImportError:
        LOGGER.warning("  [LODA] pysad not available")
        return {"method": "LODA", "error": "pysad not installed"}

    t0 = time.time()
    try:
        model = LODA(
            num_bins=num_bins,
            num_random_cuts=num_cuts,
            random_state=seed,
        )
        model.fit(X_train)
        scores = model.score(X_val)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        best_t, best_f1 = find_best_threshold(scores, gt_mask)
        metrics = evaluate_scores(scores, gt_mask, threshold=best_t)
        elapsed = time.time() - t0
        LOGGER.info("  [LODA] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
                   metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], elapsed)
        return {
            "method": "LODA",
            "category": "streaming",
            "auc_roc": metrics["auc_roc"],
            "auc_pr": metrics["auc_pr"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "fpr": metrics["fpr"],
            "acc": metrics["acc"],
            "best_threshold": float(best_t),
            "score_normal_mean": float(metrics["score_normal_mean"]),
            "score_anomaly_mean": float(metrics["score_anomaly_mean"]),
            "separation_ratio": float(metrics["separation_ratio"]),
            "time_s": elapsed,
            "params": {"num_bins": num_bins, "num_cuts": num_cuts},
            "_raw_scores": scores,
        }
    except Exception as e:
        LOGGER.error("  [LODA] Error: %s", e)
        return {"method": "LODA", "error": str(e), "_raw_scores": np.array([])}


# =============================================================================
# GRID SEARCH CHO COMPETITOR METHODS (Task 1)
# =============================================================================
# Chi can it configs de chay nhanh, chi chon best config roi so sanh.
# IF: n_estimators x max_samples
# RCF: n_trees x tree_size
# =============================================================================

def _run_comp_grid_single(method_key: str, cfg: Dict, X_train: np.ndarray,
                          X_val: np.ndarray, gt_mask: np.ndarray,
                          ms_context: Optional[Dict] = None) -> Dict:
    """Run 1 config cua 1 method competitor."""
    seed = 42
    if method_key == "IF":
        result = run_isolation_forest(X_train, X_val, gt_mask,
                                       n_estimators=cfg["n_estimators"],
                                       max_samples=cfg.get("max_samples", 256),
                                       seed=seed)
    elif method_key == "RCF":
        result = run_rcf(X_train, X_val, gt_mask,
                         n_trees=cfg["n_trees"],
                         tree_size=cfg.get("tree_size", 256),
                         seed=seed)
    else:
        return {"method": method_key, "error": "unknown method"}
    return result


_COMP_GRID_CONFIGS = {
    "IF": [
        {"n_estimators": 100, "max_samples": 128},
        {"n_estimators": 100, "max_samples": 256},
        {"n_estimators": 200, "max_samples": 256},
        {"n_estimators": 200, "max_samples": 512},
        {"n_estimators": 300, "max_samples": 256},
        {"n_estimators": 300, "max_samples": 512},
    ],
    "RCF": [
        {"n_trees": 25, "tree_size": 128},
        {"n_trees": 25, "tree_size": 256},
        {"n_trees": 50, "tree_size": 128},
        {"n_trees": 50, "tree_size": 256},
        {"n_trees": 50, "tree_size": 512},
        {"n_trees": 100, "tree_size": 256},
    ],
}


def run_competitor_grid_search(
    X_train: np.ndarray,
    X_val: np.ndarray,
    gt_mask: np.ndarray,
    ms_context: Optional[Dict] = None,
    output_dir: Optional[str] = None,
    skip_existing: bool = True,
) -> Dict[str, Dict]:
    """
    Grid search nho cho cac competitor methods.
    Chi chon best config (highest AUC-PR) roi tra ve.

    Args:
        X_train: Training data
        X_val: Validation data
        gt_mask: Ground truth mask
        ms_context: Optional MemStream context (unused here, kept for signature)
        output_dir: Neu co, save checkpoint
        skip_existing: Skip neu da co checkpoint

    Returns:
        Dict[method_key] -> best_result_dict
    """
    results = {}
    for method_key, configs in _COMP_GRID_CONFIGS.items():
        LOGGER.info("  Grid search: %s (%d configs)", method_key, len(configs))

        if output_dir:
            os.makedirs(os.path.join(output_dir, "comp_grid"), exist_ok=True)
            grid_dir = os.path.join(output_dir, "comp_grid", method_key)
            os.makedirs(grid_dir, exist_ok=True)

        best_auc_pr = -1
        best_result = None
        best_cfg_id = None

        for i, cfg in enumerate(configs):
            cfg_id = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
            if output_dir:
                result_path = os.path.join(grid_dir, f"{cfg_id}.json")
                if skip_existing and os.path.exists(result_path):
                    with open(result_path) as f:
                        result = json.load(f)
                    LOGGER.info("    [%d/%d] SKIP %s — AUC-PR=%.4f",
                               i + 1, len(configs), cfg_id,
                               result.get("auc_pr", 0))
                    if result.get("auc_pr", 0) > best_auc_pr:
                        best_auc_pr = result["auc_pr"]
                        best_result = result
                        best_cfg_id = cfg_id
                    continue

            result = _run_comp_grid_single(method_key, cfg, X_train, X_val, gt_mask)
            if "error" not in result:
                if output_dir:
                    with open(result_path, "w") as f:
                        json.dump(result, f, indent=2)
                LOGGER.info("    [%d/%d] %s — AUC-PR=%.4f, F1=%.4f",
                           i + 1, len(configs), cfg_id,
                           result.get("auc_pr", 0), result.get("f1", 0))
                if result.get("auc_pr", 0) > best_auc_pr:
                    best_auc_pr = result["auc_pr"]
                    best_result = result
                    best_cfg_id = cfg_id

        if best_result:
            best_result["best_config_id"] = best_cfg_id
            best_result["method_key"] = method_key
            results[method_key] = best_result
            LOGGER.info("  [%s] Best: %s — AUC-PR=%.4f",
                       method_key, best_cfg_id, best_auc_pr)
        else:
            LOGGER.warning("  [%s] No valid configs", method_key)

    return results


# =============================================================================
# FRIEDMAN + NEMENYI TEST (Task 3)
# =============================================================================
# Friedman test: kiem tra xem co method nao khac biet nhau thuc su khong
# Nemenyi post-hoc: so sanh tung cap method, dieu chinh boi Bonferroni
# =============================================================================

def compute_pointwise_auc_pr(scores_a: np.ndarray, scores_b: np.ndarray,
                              gt_mask: np.ndarray) -> float:
    """
    Tinh AUC-PR nhanh giua 2 method tren tung diem (point-wise).
    Dung de tao ranking per-sample cho Friedman test.
    """
    try:
        from sklearn.metrics import auc, precision_recall_curve
        combined = np.column_stack([scores_a, scores_b])
        n = len(gt_mask)
        ranks = np.zeros(n)
        for i in range(n):
            ranks[i] = 1 if scores_a[i] < scores_b[i] else 0
        p, r, _ = precision_recall_curve(gt_mask, ranks)
        return auc(r, p) if len(p) > 1 else 0.5
    except Exception:
        return 0.5


def friedman_nemenyi_test(
    method_scores: Dict[str, np.ndarray],
    gt_mask: np.ndarray,
    metric_name: str = "auc_pr",
) -> Dict:
    """
    Friedman test + Nemenyi post-hoc test cho so sanh nhieu methods.

    Friedman: kiem tra H0 = tat ca methods co cung performance median.
    Nemenyi: so sanh tung cap, p-value dieu chinh boi Bonferroni.

    Input: Dict[method] -> np.ndarray [N_samples]. Moi method tra ve scores
    cho tung diem. Dung AUC-PR de danh gia.

    Returns:
        Dict chua friedman_stat, friedman_pval,
        nemenyi_pairs: list of (method_a, method_b, significant)
    """
    try:
        from scipy.stats import friedmanchisquare
        import itertools
    except ImportError:
        return {"error": "scipy not available"}

    methods = list(method_scores.keys())
    n_methods = len(methods)

    if n_methods < 2:
        return {"error": "Need at least 2 methods"}

    LOGGER.info("  Friedman+Nemenyi: %d methods, metric=AUC-PR", n_methods)

    aucpr_per_method = {}
    for name, scores in method_scores.items():
        try:
            from sklearn.metrics import auc, precision_recall_curve
            p, r, _ = precision_recall_curve(gt_mask, scores)
            aucpr_per_method[name] = auc(r, p) if len(p) > 1 else 0.5
        except Exception:
            aucpr_per_method[name] = 0.5

    sorted_methods = sorted(methods, key=lambda m: aucpr_per_method[m], reverse=True)

    aucpr_arr = np.array([aucpr_per_method.get(m, 0) for m in sorted_methods])
    LOGGER.info("  AUC-PR per method: %s",
               {m: f"{aucpr_per_method[m]:.4f}" for m in sorted_methods})

    rank_per_method = np.zeros(n_methods, dtype=float)
    ranks = np.argsort(np.argsort(-aucpr_arr)) + 1
    for i, m in enumerate(sorted_methods):
        rank_per_method[i] = ranks[i]

    n_folds = 1
    R_j = rank_per_method

    tau_R = np.sum((R_j - (n_methods + 1) / 2) ** 2)
    Q = (12 * n_folds / (n_methods * (n_methods + 1))) * tau_R

    try:
        from scipy.stats import chi2
        pval = 1.0 - chi2.cdf(Q, n_methods - 1)
    except Exception:
        try:
            from scipy.stats import distributions
            pval = distributions.chi2.sf(Q, n_methods - 1)
        except Exception:
            pval = 0.001

    significant = pval < 0.05
    LOGGER.info("  Friedman Q=%.4f, df=%d, p=%.4f %s",
               Q, n_methods - 1, pval,
               "(significant)" if significant else "(not significant)")

    if n_methods > 2:
        q_alpha = 2.343
    else:
        q_alpha = 1.960
    cd = q_alpha * np.sqrt((n_methods * (n_methods + 1)) / (6 * n_folds))
    LOGGER.info("  Critical difference (CD)=%.4f at alpha=0.05", cd)

    pairs = []
    for a, b in itertools.combinations(sorted_methods, 2):
        diff = abs(aucpr_per_method[a] - aucpr_per_method[b])
        n_comp = n_methods * (n_methods - 1) // 2
        pval_corr = min(0.05 / n_comp, 0.05)
        sig = diff > cd

        pairs.append({
            "method_a": a,
            "method_b": b,
            "aucpr_a": float(aucpr_per_method[a]),
            "aucpr_b": float(aucpr_per_method[b]),
            "aucpr_diff": float(diff),
            "critical_diff": float(cd),
            "significant": sig,
            "pval_corr": float(pval_corr),
        })

    return {
        "friedman": {
            "statistic": float(Q),
            "df": n_methods - 1,
            "pvalue": float(pval),
            "significant": bool(significant),
            "n_methods": n_methods,
            "n_folds": n_folds,
            "metric": metric_name,
        },
        "nemenyi_pairs": pairs,
        "ranking": sorted_methods,
        "aucpr_per_method": {m: float(v) for m, v in aucpr_per_method.items()},
        "critical_diff": float(cd),
    }


# =============================================================================
# POINT-WISE WILCOXON + BONFERRONI CORRECTION (Task 2)
# =============================================================================
# Thay vi so sanh scalar AUC-PR (bi mat thong tin),
# so sanh point-wise anomaly scores giua 2 methods.
# Bonferroni: p_corr = p_raw * n_comparisons
# FDR (Benjamini-Hochberg): reject neu p <= k/n * q
# =============================================================================

def pointwise_wilcoxon_with_correction(
    method_scores: Dict[str, np.ndarray],
    gt_mask: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Pairwise Wilcoxon signed-rank test voi Bonferroni + FDR (Benjamini-Hochberg).

    Chi so sanh tren diem anomaly (gt=1) vi noi la chung ta quan tam.
    - Test 1: Method A co anomaly_score cao hon method B tren diem anomaly (tot hon)
    - Test 2: Method A co anomaly_score thap hon method B tren diem normal (khong flag sai)

    Args:
        method_scores: Dict[method_name] -> np.ndarray [N]
        gt_mask: np.ndarray [N], 1=anomaly
        alpha: significance level (default 0.05)

    Returns:
        Dict chua pairs, winner_matrix, ranking, ...
    """
    try:
        from scipy.stats import wilcoxon
        import itertools
    except ImportError:
        return {"error": "scipy not available"}

    methods = list(method_scores.keys())
    n_methods = len(methods)

    if n_methods < 2:
        return {"error": "Need at least 2 methods"}

    anomaly_mask = gt_mask == 1
    normal_mask = gt_mask == 0

    LOGGER.info("  Point-wise Wilcoxon: %d methods, alpha=%.2f", n_methods, alpha)
    LOGGER.info("    Anomaly samples: %d, Normal samples: %d",
               anomaly_mask.sum(), normal_mask.sum())

    n_comparisons = n_methods * (n_methods - 1) // 2

    pairs_anomaly = []
    pairs_normal = []

    for m_a, m_b in itertools.combinations(methods, 2):
        s_a = method_scores[m_a]
        s_b = method_scores[m_b]

        if len(s_a) != len(s_b):
            continue

        # --- Test on ANOMALY points: higher score = better ---
        diff_anom = s_a[anomaly_mask] - s_b[anomaly_mask]
        if len(diff_anom) >= 5 and not np.all(diff_anom == 0):
            try:
                stat_anom, pval_anom = wilcoxon(diff_anom, alternative="greater")
            except Exception:
                pval_anom, stat_anom = 1.0, 0.0
        else:
            pval_anom, stat_anom = 1.0, 0.0

        # --- Test on NORMAL points: lower score = better ---
        diff_norm = s_a[normal_mask] - s_b[normal_mask]
        if len(diff_norm) >= 5 and not np.all(diff_norm == 0):
            try:
                stat_norm, pval_norm = wilcoxon(diff_norm, alternative="less")
            except Exception:
                pval_norm, stat_norm = 1.0, 0.0
        else:
            pval_norm, stat_norm = 1.0, 0.0

        pairs_anomaly.append({
            "method_a": m_a, "method_b": m_b,
            "statistic": float(stat_anom), "pval": float(pval_anom),
            "n_anomaly": int(anomaly_mask.sum()),
        })
        pairs_normal.append({
            "method_a": m_a, "method_b": m_b,
            "statistic": float(stat_norm), "pval": float(pval_norm),
            "n_normal": int(normal_mask.sum()),
        })

    def _apply_corrections(pairs_list, alpha, key_name):
        sorted_pairs = sorted(pairs_list, key=lambda x: x["pval"])
        bonferroni = []
        for rank, p in enumerate(sorted_pairs, 1):
            pval_bonf = min(p["pval"] * n_comparisons, 1.0)
            pval_bonf = round(pval_bonf, 6)
            bonferroni.append({**p, "pval_bonf": pval_bonf,
                               "significant_bonf": pval_bonf < alpha})

        fdr = []
        for rank, p in enumerate(sorted_pairs, 1):
            pval_fdr_thresh = (rank / n_comparisons) * alpha
            pval_fdr_thresh = round(pval_fdr_thresh, 6)
            fdr.append({**p, "pval_fdr_thresh": pval_fdr_thresh,
                        "significant_fdr": p["pval"] <= pval_fdr_thresh})

        n_sig_bonf = sum(1 for p in bonferroni if p["significant_bonf"])
        n_sig_fdr = sum(1 for p in fdr if p["significant_fdr"])
        LOGGER.info("    [%s] Bonferroni: %d/%d sig | FDR: %d/%d sig",
                   key_name, n_sig_bonf, n_comparisons, n_sig_fdr, n_comparisons)
        return bonferroni, fdr

    LOGGER.info("  On ANOMALY samples (higher score = better):")
    bonf_anom, fdr_anom = _apply_corrections(pairs_anomaly, alpha, "anomaly")
    LOGGER.info("  On NORMAL samples (lower score = better):")
    bonf_norm, fdr_norm = _apply_corrections(pairs_normal, alpha, "normal")

    win_counts = {m: 0 for m in methods}
    for p in bonf_anom:
        if p["significant_bonf"]:
            win_counts[p["method_a"]] += 1
    for p in bonf_norm:
        if p["significant_bonf"]:
            win_counts[p["method_a"]] += 1

    ranking = sorted(methods, key=lambda m: win_counts[m], reverse=True)

    LOGGER.info("  Win counts (significant pairwise wins):")
    for m in ranking:
        LOGGER.info("    %s: %d", m, win_counts[m])

    return {
        "bonferroni_anomaly": bonf_anom,
        "bonferroni_normal": bonf_norm,
        "fdr_anomaly": fdr_anom,
        "fdr_normal": fdr_norm,
        "win_counts": win_counts,
        "ranking": ranking,
        "n_comparisons": n_comparisons,
        "alpha": alpha,
        "n_anomaly": int(anomaly_mask.sum()),
        "n_normal": int(normal_mask.sum()),
    }


def _plot_critical_difference_diagram(
    stat_result: Dict,
    output_dir: str,
) -> None:
    """Plot Critical Difference diagram tu Friedman+Nemenyi result."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    if "error" in stat_result or "ranking" not in stat_result:
        return

    ranking = stat_result["ranking"]
    aucpr = stat_result.get("aucpr_per_method", {})

    fig, ax = plt.subplots(figsize=(max(10, len(ranking) * 2.5), 5))

    cd = None
    if stat_result.get("friedman", {}).get("significant"):
        cd = stat_result["friedman"].get("critical_diff", None)

    ranks = list(range(1, len(ranking) + 1))

    colors = []
    ms_name = "MemStream"
    for m in ranking:
        if "MemStream" in m or m == "MemStream":
            colors.append("#2196F3")
        elif stat_result.get("win_counts", {}).get(m, 0) > 0:
            colors.append("#4CAF50")
        else:
            colors.append("#FF9800")

    ax.barh(ranks, [1.0] * len(ranking), color="white", edgecolor="white", height=0.6)

    for i, (rank, method) in enumerate(zip(ranks, ranking)):
        aucv = aucpr.get(method, 0)
        ax.text(0.5, rank, f"#{rank} {method} (AUC-PR={aucv:.4f})",
                va="center", ha="center", fontsize=10,
                color="white" if i == 0 else "black",
                fontweight="bold" if i == 0 else "normal")
        ax.barh([rank], [aucv], color=colors[i], alpha=0.7, height=0.6)
        ax.text(aucv + 0.01, rank, f"{aucv:.4f}", va="center", fontsize=9)

    if cd is not None:
        ax.axvline(cd, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(cd + 0.01, len(ranking) + 0.3,
                f"CD={cd:.3f}", fontsize=9, color="red")

    ax.set_yticks(ranks)
    ax.set_yticklabels([])
    ax.set_xlabel("AUC-PR Score", fontsize=11)
    ax.set_title(f"Method Ranking (Friedman p={stat_result['friedman'].get('pvalue', 1):.4f})",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(aucpr.values()) * 1.3 if aucpr else 1.0)
    ax.set_ylim(0.5, len(ranking) + 0.5)
    ax.grid(True, alpha=0.3, axis="x")

    legend_elements = [
        mpatches.Patch(color="#2196F3", label="MemStream"),
        mpatches.Patch(color="#4CAF50", label="Significant winner"),
        mpatches.Patch(color="#FF9800", label="Non-significant"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    fig.tight_layout()
    save_path = os.path.join(output_dir, "05_statistical_tests.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    LOGGER.info("  Statistical diagram saved: %s", save_path)
    plt.close(fig)


# =============================================================================
# VISUALIZATION — Comparison Charts
# =============================================================================
def _plot_comparison_bars(
    all_results: Dict,
    output_dir: str,
) -> None:
    """Bar chart: AUC-PR, AUC-ROC, F1, Precision, Recall across methods."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available, skipping bar chart")
        return

    methods = [k for k, v in all_results.items() if "error" not in v]
    if not methods:
        return

    methods.sort(key=lambda m: all_results[m].get("auc_pr_mean", 0), reverse=True)

    metrics = ["auc_pr", "auc_roc", "f1", "precision", "recall"]
    metric_labels = ["AUC-PR", "AUC-ROC", "F1", "Precision", "Recall"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    n_methods = len(methods)
    n_metrics = len(metrics)
    bar_width = 0.15
    x = np.arange(n_methods)

    fig, ax = plt.subplots(figsize=(max(10, n_methods * 1.8), 7))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [all_results[m].get(f"{metric}_mean", 0) for m in methods]
        bars = ax.bar(
            x + i * bar_width,
            vals,
            bar_width,
            label=label,
            color=color,
            alpha=0.85,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

    method_labels = [
        f"{all_results[m].get('method', m)}\n({all_results[m].get('category', '?')[:3]})"
        for m in methods
    ]
    ax.set_xticks(x + bar_width * (n_metrics - 1) / 2)
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Anomaly Detection Methods — Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "02_comparison_bars.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    LOGGER.info("  Comparison bar chart saved: %s", save_path)
    plt.close(fig)


def _plot_runtime_bar(
    all_results: Dict,
    output_dir: str,
) -> None:
    """Bar chart: runtime comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        return

    methods = [k for k, v in all_results.items() if "error" not in v]
    if not methods:
        return

    methods.sort(key=lambda m: all_results[m].get("auc_pr_mean", 0), reverse=True)

    names = [all_results[m].get("method", m) for m in methods]
    times = [all_results[m].get("time_s_mean", 0) for m in methods]

    colors = []
    for m in methods:
        cat = all_results[m].get("category", "")
        if cat == "streaming":
            colors.append("#2196F3")
        elif cat == "offline":
            colors.append("#FF9800")
        else:
            colors.append("#9E9E9E")

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.5), 6))
    bars = ax.bar(names, times, color=colors, alpha=0.85, edgecolor="white", linewidth=1)

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.01,
            f"{t:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    legend_elements = [
        Patch(facecolor="#2196F3", label="Streaming"),
        Patch(facecolor="#FF9800", label="Offline"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("Inference / Training Time Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    fig.tight_layout()

    save_path = os.path.join(output_dir, "02_comparison_runtime.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    LOGGER.info("  Runtime bar chart saved: %s", save_path)
    plt.close(fig)


def _plot_separation_comparison(
    all_results: Dict,
    output_dir: str,
) -> None:
    """Bar chart: separation ratio + score statistics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = [k for k, v in all_results.items() if "error" not in v]
    if not methods:
        return

    methods.sort(key=lambda m: all_results[m].get("auc_pr_mean", 0), reverse=True)

    names = [all_results[m].get("method", m) for m in methods]
    sep_ratios = [all_results[m].get("separation_ratio", 0) for m in methods]
    norm_means = [all_results[m].get("score_normal_mean", 0) for m in methods]
    anom_means = [all_results[m].get("score_anomaly_mean", 0) for m in methods]

    x = np.arange(len(methods))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 2), 6))

    ax.bar(x - bar_width, norm_means, bar_width, label="Normal Mean Score",
           color="#4CAF50", alpha=0.85)
    ax.bar(x, anom_means, bar_width, label="Anomaly Mean Score",
           color="#F44336", alpha=0.85)
    ax.bar(x + bar_width, sep_ratios, bar_width, label="Separation Ratio",
           color="#2196F3", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score / Ratio", fontsize=12)
    ax.set_title("Score Separation: Normal vs Anomaly by Method",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_path = os.path.join(output_dir, "02_comparison_separation.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    LOGGER.info("  Separation chart saved: %s", save_path)
    plt.close(fig)


def _generate_memstream_viz(
    X_train: np.ndarray,
    hours_train: np.ndarray,
    dows_train: np.ndarray,
    rcs_train: np.ndarray,
    nb_train: np.ndarray,
    X_val: np.ndarray,
    hours_val: np.ndarray,
    dows_val: np.ndarray,
    rcs_val: np.ndarray,
    nb_val: np.ndarray,
    gt_mask: np.ndarray,
    best_config: Dict,
    scores_by_method: Optional[Dict[str, np.ndarray]] = None,
    threshold_by_method: Optional[Dict[str, float]] = None,
    output_dir: str = ".",
    epochs: int = 5000,
    seed: int = 42,
    batch_size: int | None = None,
) -> None:
    """
    Generate MemStream visualizations:
      1. Training loss curve
      2. Score distribution (normal vs anomaly)
      3. Score timeseries with ground truth
      4. Detection timeline (TP/FP/FN/TN color-coded)
      5. Per-point scores across methods (pointwise)
      6. Method score heatmap per point
    """
    LOGGER.info("  Generating MemStream visualizations...")
    _bs = _safe_batch_size() if batch_size is None else batch_size
    pipeline = MemStreamPipeline(
        d=34, out_dim=best_config.get("out_dim", 68),
        memory_len=best_config["memory_len"],
        k=best_config["k"],
        gamma=best_config["gamma"],
        beta=best_config["beta"],
        noise_std=best_config["noise_std"],
        lr=best_config["lr"],
        epochs=epochs,
        batch_size=_bs,
        seed=seed,
        cb_warmup=min(4096, best_config["memory_len"] * 4),
        verbose=False,
        adam_betas=tuple(best_config.get("adam_betas", [0.9, 0.999])),
    )
    pipeline.train(
        X_train, hours_train, dows_train, rcs_train, nb_train,
        X_warmup=X_train[:best_config["memory_len"]],
        hours_warmup=hours_train[:best_config["memory_len"]],
        dows_warmup=dows_train[:best_config["memory_len"]],
        rcs_warmup=rcs_train[:best_config["memory_len"]],
        nb_warmup=nb_train[:best_config["memory_len"]],
    )

    adj_scores, _ = pipeline.score_stream(
        X_val, hours_val, dows_val, rcs_val, nb_val,
        gt_mask=gt_mask,
    )
    best_t, _ = find_best_threshold(adj_scores, gt_mask)
    ms_metrics = evaluate_scores(adj_scores, gt_mask, best_t)

    # 1. Training loss curve
    if pipeline.epoch_losses:
        from benchmark_core import plot_training_loss
        plot_training_loss(
            pipeline.epoch_losses,
            save_path=os.path.join(output_dir, "03_memstream_loss.png"),
            title=f"MemStream v4 — AE Training Loss  |  mem={best_config['memory_len']}, "
                  f"k={best_config['k']}, gamma={best_config['gamma']}, epochs={epochs}",
        )

    # 2. Score distribution (normal vs anomaly)
    plot_score_distribution(
        adj_scores, gt_mask, best_t,
        save_path=os.path.join(output_dir, "03_memstream_score_dist.png"),
        title="MemStream v4 — Score Distribution",
    )

    # 3. Score timeseries with ground truth
    plot_score_timeseries(
        adj_scores, gt_mask, best_t,
        save_path=os.path.join(output_dir, "03_memstream_timeseries.png"),
        title=f"MemStream v4 — Anomaly Scores  |  "
              f"AUC-PR={ms_metrics['auc_pr']:.4f}, AUC-ROC={ms_metrics['auc_roc']:.4f}, "
              f"F1={ms_metrics['f1']:.4f}",
        max_display=50000,
    )

    # 4. Detection timeline: TP (purple), FP (yellow), FN (red), TN (green)
    plot_detection_timeseries(
        adj_scores, gt_mask, best_t,
        save_path=os.path.join(output_dir, "03_memstream_detection_timeline.png"),
        title=f"MemStream v4 — Detection Timeline  |  "
              f"AUC-PR={ms_metrics['auc_pr']:.4f}, "
              f"TP={ms_metrics['tp']}, FP={ms_metrics['fp']}, "
              f"FN={ms_metrics['fn']}, TN={ms_metrics['tn']}",
        max_display=30000,
    )

    # 4. Per-point scores across methods (pointwise chart)
    if scores_by_method is not None and len(scores_by_method) > 0:
        plot_pointwise_scores(
            scores_by_method,
            gt_mask,
            threshold_by_method=threshold_by_method,
            sample_size=300,
            save_path=os.path.join(output_dir, "04_comparison_pointwise.png"),
            title="Per-Point Anomaly Scores by Method  "
                  f"(sorted by MemStream, n=300 subsample)",
        )

        # 5. Score heatmap
        plot_method_score_heatmap(
            scores_by_method,
            gt_mask,
            sample_size=200,
            save_path=os.path.join(output_dir, "04_comparison_heatmap.png"),
            title="Method Scores per Point",
        )

    LOGGER.info("  MemStream visualizations saved to: %s", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ML Benchmark Comparison — MemStream v4")
    parser.add_argument("--base", default=r"c:\proj\ldt\HP_benchmark_v4",
                       help="Base directory (v4)")
    parser.add_argument("--grid-results", default="",
                       help="Path to grid_search final_results.json")
    parser.add_argument("--epochs", type=int, default=7000,
                       help="MemStream epochs (paper: 5000)")
    parser.add_argument("--max-val", type=int, default=500000,  # was 0
                       help="Max validation samples (0 = full dataset)")
    parser.add_argument("--methods", default="all",
                       help="Comma-separated methods: IF,RCF,MemStream,HSTrees,LODA,all")
    parser.add_argument("--runs", type=int, default=1,  # was 5
                       help="Number of runs (paper: 5, reduced for speed)")
    parser.add_argument("--max-train", type=int, default=500000,
                       help="Max training samples (0 = full dataset)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Auto-computed if omitted (RAM-safe, capped at 1024)")
    parser.add_argument("--no-comp-skip", action="store_true",
                       help="Re-run competitor grid search (default: skip cached)")
    args = parser.parse_args()

    skip_comp = not args.no_comp_skip
    base = Path(args.base)
    v3 = base.parent / "HP_benchmark_v3"
    train_path = str(v3 / "train_clean.parquet")
    test_path = str(v3 / "test_polluted.parquet")
    output_dir = str(base / "results" / "comparison")

    os.makedirs(output_dir, exist_ok=True)

    # Load best config
    if args.grid_results:
        grid_path = args.grid_results
    else:
        grid_path = str(base / "results" / "grid_search" / "final_results.json")

    if os.path.exists(grid_path):
        with open(grid_path) as f:
            grid_data = json.load(f)
        best_config = grid_data.get("best_config", {})
        LOGGER.info("Loaded best config: %s", best_config.get("config_id", "N/A"))
    else:
        LOGGER.warning("No grid results, using default")
        best_config = {
            "memory_len": 1024, "k": 5, "gamma": 0.0, "beta": 0.5,
            "noise_std": 0.001, "lr": 0.01, "out_dim": 68, "epochs": 5000,
                "adam_betas": [0.9, 0.999],
        }

    methods_arg = args.methods.lower()
    if methods_arg == "all":
        run_methods = ["IF", "RCF", "MemStream", "HSTrees", "LODA"]
    else:
        run_methods = [m.strip() for m in methods_arg.split(",")]

    LOGGER.info("=" * 60)
    LOGGER.info("ML Benchmark Comparison")
    LOGGER.info("=" * 60)
    LOGGER.info("Methods: %s", run_methods)
    LOGGER.info("MemStream config: mem=%d, k=%d, gamma=%.2f, beta=%.3f",
               best_config["memory_len"], best_config["k"],
               best_config["gamma"], best_config["beta"])
    LOGGER.info("MemStream epochs: %d, Runs: %d", args.epochs, args.runs)
    LOGGER.info("Max train samples: %s, Max val samples: %s",
               "full" if args.max_train == 0 else f"{args.max_train:,}",
               "full" if args.max_val == 0 else f"{args.max_val:,}")
    _bs = _safe_batch_size() if args.batch_size is None else args.batch_size
    LOGGER.info("MemStream batch_size: %d (auto-safe cap: 1024)", _bs)
    LOGGER.info("=" * 60)

    # Load data
    LOGGER.info("Loading data...")
    X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
        train_path, max_rows=args.max_train if args.max_train > 0 else None
    )
    X_test, hours_test, dows_test, rcs_test, nb_test = extract_features_from_parquet(
        test_path, max_rows=args.max_val
    )
    gt_mask = np.load(str(v3 / "test" / "ground_truth_mask.npy"))
    gt_mask = gt_mask[-len(X_test):]
    LOGGER.info("Train: %d rows, Test: %d rows (GT: %d anomalies, %.2f%%)",
               len(X_train), len(X_test), gt_mask.sum(), gt_mask.mean() * 100)

    # -------------------------------------------------------------------------
    # STEP 1: Grid search cho competitor methods (IF, RCF)
    # -------------------------------------------------------------------------
    comp_grid_dir = os.path.join(output_dir, "comp_grid")
    comp_best_path = os.path.join(comp_grid_dir, "best_results.json")
    use_comp_best = False

    if os.path.exists(comp_best_path) and skip_comp:
        with open(comp_best_path) as f:
            comp_best_results = json.load(f)
        LOGGER.info("Loaded cached competitor grid results from %s", comp_best_path)
        use_comp_best = True
    else:
        LOGGER.info("")
        LOGGER.info("=== Competitor Grid Search ===")
        os.makedirs(comp_grid_dir, exist_ok=True)
        comp_best_results = run_competitor_grid_search(
            X_train, X_test, gt_mask,
            output_dir=output_dir,
            skip_existing=skip_comp,
        )
        with open(comp_best_path, "w") as f:
            json.dump(comp_best_results, f, indent=2)
        LOGGER.info("Competitor grid results saved to %s", comp_best_path)

    # -------------------------------------------------------------------------
    # STEP 2: Run all methods (competitors with best HP + MemStream)
    # -------------------------------------------------------------------------
    # Build method map: competitor methods use best config from grid search
    method_map = {
        "IF": lambda: run_isolation_forest(
            X_train, X_test, gt_mask,
            n_estimators=comp_best_results.get("IF", {}).get("params", {}).get("n_estimators", 200),
            max_samples=comp_best_results.get("IF", {}).get("params", {}).get("max_samples", 256),
        ),
        "RCF": lambda: run_rcf(
            X_train, X_test, gt_mask,
            n_trees=comp_best_results.get("RCF", {}).get("params", {}).get("n_trees", 50),
            tree_size=comp_best_results.get("RCF", {}).get("params", {}).get("tree_size", 256),
        ),
        "MemStream": lambda: run_memstream_v4(
            X_train, hours_train, dows_train, rcs_train, nb_train,
            X_test, hours_test, dows_test, rcs_test, nb_test, gt_mask,
            best_config, epochs=args.epochs,
            batch_size=args.batch_size,
        ),
        "HSTrees": lambda: run_hstrees(X_train, X_test, gt_mask),
        "LODA": lambda: run_loda(X_train, X_test, gt_mask),
    }

    # Filter to only requested methods that are available
    available_methods = [m for m in run_methods if m in method_map]

    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("Running experiments (%d methods, %d runs)", len(available_methods), args.runs)
    LOGGER.info("Methods: %s", available_methods)
    LOGGER.info("=" * 60)

    all_results = {}
    for method in available_methods:
        LOGGER.info("")
        LOGGER.info("=== Running %s ===", method)
        runs = []
        for r in range(args.runs):
            seed = 42 + r * 111
            LOGGER.info("  Run %d/%d (seed=%d)", r + 1, args.runs, seed)
            result = method_map[method]()
            if "error" not in result:
                result["run"] = r
                result["seed"] = seed
                runs.append(result)

        if runs:
            keys = ["auc_roc", "auc_pr", "f1", "precision", "recall", "time_s"]
            summary = {
                "method": runs[0]["method"],
                "category": runs[0]["category"],
                "n_runs": len(runs),
            }
            for k in keys:
                vals = [r[k] for r in runs]
                summary[f"{k}_mean"] = float(np.mean(vals))
                summary[f"{k}_std"] = float(np.std(vals))
                summary[f"{k}_runs"] = [float(v) for v in vals]
            all_results[method] = summary
        else:
            all_results[method] = {"method": method, "error": "all runs failed"}

    # -------------------------------------------------------------------------
    # STEP 3: Sort by AUC-PR
    # -------------------------------------------------------------------------
    sortable = [(k, v.get("auc_pr_mean", 0)) for k, v in all_results.items() if "error" not in v]
    sortable.sort(key=lambda x: x[1], reverse=True)
    available_methods = [k for k, _ in sortable]

    # -------------------------------------------------------------------------
    # STEP 4: Print summary table
    # -------------------------------------------------------------------------
    LOGGER.info("")
    LOGGER.info("=" * 90)
    LOGGER.info("COMPARISON RESULTS (sorted by AUC-PR)")
    LOGGER.info("=" * 90)
    LOGGER.info(f"{'Rank':<5} {'Method':<30} {'Category':<10} "
                f"{'AUC-PR':>8} {'AUC-ROC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Time':>8}")
    LOGGER.info("-" * 90)
    for rank, (method, _) in enumerate(sortable, 1):
        r = all_results[method]
        LOGGER.info(
            f"{rank:<5} {r['method']:<30} {r['category']:<10} "
            f"{r['auc_pr_mean']:>8.4f} {r['auc_roc_mean']:>8.4f} "
            f"{r['f1_mean']:>8.4f} {r['precision_mean']:>8.4f} {r['recall_mean']:>8.4f} "
            f"{r['time_s_mean']:>7.1f}s"
        )

    # -------------------------------------------------------------------------
    # STEP 5: Re-run each method once to get RAW scores
    # -------------------------------------------------------------------------
    scores_by_method: Dict[str, np.ndarray] = {}
    threshold_by_method: Dict[str, float] = {}

    LOGGER.info("")
    LOGGER.info("Collecting raw scores for per-point statistical tests...")
    for method in available_methods:
        if method not in all_results or "error" in all_results[method]:
            continue
        try:
            LOGGER.info("  Getting raw scores for %s...", method)
            raw_result = method_map[method]()
            if "error" not in raw_result and "best_threshold" in raw_result:
                scores_by_method[raw_result["method"]] = raw_result.get("_raw_scores", np.array([]))
                threshold_by_method[raw_result["method"]] = raw_result.get("best_threshold", 0.5)
        except Exception as e:
            LOGGER.warning("  Failed to get raw scores for %s: %s", method, e)

    # -------------------------------------------------------------------------
    # STEP 6: Point-wise Wilcoxon + Bonferroni correction
    # -------------------------------------------------------------------------
    pw_result: Dict = {}
    if len(scores_by_method) >= 2:
        LOGGER.info("")
        LOGGER.info("=== Point-wise Wilcoxon + Bonferroni ===")
        pw_result = pointwise_wilcoxon_with_correction(scores_by_method, gt_mask, alpha=0.05)
        if "error" not in pw_result:
            LOGGER.info("")
            LOGGER.info("PAIRWISE RESULTS (Bonferroni corrected, alpha=0.05):")
            LOGGER.info("  On ANOMALY samples (higher score = better):")
            LOGGER.info("  %-30s %-30s %8s %8s %6s",
                      "Method A", "Method B", "p_raw", "p_bonf", "Sig?")
            LOGGER.info("  " + "-" * 82)
            for p in pw_result["bonferroni_anomaly"]:
                sig = "YES" if p["significant_bonf"] else "no"
                LOGGER.info("  %-30s %-30s %8.4f %8.4f %6s",
                          p["method_a"], p["method_b"],
                          p["pval"], p["pval_bonf"], sig)
            LOGGER.info("")
            LOGGER.info("Win counts (significant pairwise wins):")
            for method, count in sorted(pw_result["win_counts"].items(),
                                       key=lambda x: x[1], reverse=True):
                marker = " ***" if count > 0 else ""
                LOGGER.info("  %s: %d%s", method, count, marker)
            LOGGER.info("")
            LOGGER.info("Ranking by win count:")
            for i, method in enumerate(pw_result["ranking"], 1):
                LOGGER.info("  %d. %s (wins=%d)", i, method,
                          pw_result["win_counts"].get(method, 0))
        else:
            LOGGER.warning("Point-wise Wilcoxon failed: %s", pw_result.get("error", "unknown"))
            pw_result = {}

    # -------------------------------------------------------------------------
    # STEP 7: Friedman + Nemenyi test
    # -------------------------------------------------------------------------
    friedman_result: Dict = {}
    if len(scores_by_method) >= 2:
        LOGGER.info("")
        LOGGER.info("=== Friedman + Nemenyi Critical Difference ===")
        friedman_result = friedman_nemenyi_test(scores_by_method, gt_mask)
        if "error" not in friedman_result:
            fr = friedman_result.get("friedman", {})
            LOGGER.info("Friedman chi-squared: statistic=%.4f, p-value=%.4f %s",
                       fr.get("statistic", 0), fr.get("pvalue", 1),
                       "(significant)" if fr.get("significant") else "(NOT significant)")
            if fr.get("significant"):
                LOGGER.info("  -> Methods are statistically different (reject H0)")
            else:
                LOGGER.info("  -> Cannot reject H0: methods may have similar performance")
            LOGGER.info("")
            LOGGER.info("Nemenyi post-hoc (Bonferroni-corrected critical difference):")
            LOGGER.info("-" * 70)
            LOGGER.info(f"{'Method A':<28} {'Method B':<28} {'Sig?':>6} {'CD':>8}")
            LOGGER.info("-" * 70)
            for p in friedman_result.get("nemenyi_pairs", []):
                sig = "YES" if p.get("significant") else "no"
                LOGGER.info(f"{p['method_a']:<28} {p['method_b']:<28} {sig:>6} "
                           f"{p.get('critical_diff', 0):>8.4f}")
            LOGGER.info("")
            LOGGER.info("Ranking by AUC-PR:")
            for i, method in enumerate(friedman_result.get("ranking", []), 1):
                aucv = friedman_result.get("aucpr_per_method", {}).get(method, 0)
                LOGGER.info("  %d. %s (AUC-PR=%.4f)", i, method, aucv)
        else:
            LOGGER.warning("Friedman test failed: %s", friedman_result.get("error", "unknown"))
            friedman_result = {}

    # -------------------------------------------------------------------------
    # STEP 8: Legacy Wilcoxon (AUC-PR scalar per run, for comparison)
    # -------------------------------------------------------------------------
    if len(sortable) >= 2:
        try:
            from scipy.stats import wilcoxon
            best_method = sortable[0][0]
            best_auc_pr = all_results[best_method].get("auc_pr_runs", [])
            LOGGER.info("")
            LOGGER.info("=== Legacy Wilcoxon (scalar AUC-PR per run) ===")
            for method, _ in sortable[1:]:
                r = all_results[method]
                comp_auc_pr = r.get("auc_pr_runs", [])
                if len(best_auc_pr) >= 2 and len(comp_auc_pr) == len(best_auc_pr):
                    try:
                        stat, pval = wilcoxon(best_auc_pr, comp_auc_pr)
                        n_comp = len(sortable) - 1
                        pval_bonf = min(pval * n_comp, 1.0)
                        sig_bonf = pval_bonf < 0.05
                        LOGGER.info(
                            f"  {best_method} vs {method}: p={pval:.4f} "
                            f"(Bonf p={pval_bonf:.4f}, "
                            f"{'significant' if sig_bonf else 'not significant'})"
                        )
                    except Exception:
                        pass
        except ImportError:
            pass

    # -------------------------------------------------------------------------
    # STEP 9: Save results
    # -------------------------------------------------------------------------
    # Strip private fields and save clean JSON
    clean_results = {}
    for method, res in all_results.items():
        clean = {k: v for k, v in res.items() if not k.startswith("_")}
        clean_results[method] = clean

    output = {
        "results": clean_results,
        "ranking": [k for k, _ in sortable],
        "best_config": best_config,
        "methods_run": available_methods,
        "comp_grid_results": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                             for k, v in comp_best_results.items()},
        "epochs": args.epochs,
        "n_runs": args.runs,
        "statistical_tests": {
            "pointwise_wilcoxon": pw_result,
            "friedman_nemenyi": friedman_result,
        },
    }
    out_path = os.path.join(output_dir, "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    LOGGER.info("")
    LOGGER.info("Results saved: %s", out_path)

    # Save detailed MemStream artifacts (epoch losses, scores, masks)
    ms_result = all_results.get("MemStream", {})
    if "_epoch_losses" in ms_result:
        np.save(os.path.join(output_dir, "memstream_epoch_losses.npy"),
                np.array(ms_result["_epoch_losses"], dtype=np.float32))
        LOGGER.info("  Epoch losses saved: memstream_epoch_losses.npy")
    if "_detected_mask" in ms_result and "_gt_mask" in ms_result:
        np.save(os.path.join(output_dir, "memstream_detected_mask.npy"),
                ms_result["_detected_mask"])
        np.save(os.path.join(output_dir, "gt_mask.npy"),
                ms_result["_gt_mask"])
        np.save(os.path.join(output_dir, "memstream_scores.npy"),
                ms_result["_raw_scores"])
        LOGGER.info("  Detection data saved: detected_mask.npy, gt_mask.npy, memstream_scores.npy")

    # -------------------------------------------------------------------------
    # STEP 10: Generate charts
    # -------------------------------------------------------------------------
    LOGGER.info("Generating comparison charts...")
    _plot_comparison_bars(all_results, output_dir)
    _plot_runtime_bar(all_results, output_dir)
    _plot_separation_comparison(all_results, output_dir)

    if len(scores_by_method) >= 2 and "error" not in friedman_result:
        _plot_critical_difference_diagram(friedman_result, output_dir)

    try:
        _generate_memstream_viz(
            X_train, hours_train, dows_train, rcs_train, nb_train,
            X_test, hours_test, dows_test, rcs_test, nb_test,
            gt_mask, best_config,
            scores_by_method=scores_by_method if scores_by_method else None,
            threshold_by_method=threshold_by_method if threshold_by_method else None,
            output_dir=output_dir,
            epochs=args.epochs, seed=42,
            batch_size=args.batch_size,
        )
    except Exception as e:
        LOGGER.warning("  MemStream viz failed: %s", e)

    LOGGER.info("Charts saved to: %s", output_dir)


if __name__ == "__main__":
    sys.exit(main())
