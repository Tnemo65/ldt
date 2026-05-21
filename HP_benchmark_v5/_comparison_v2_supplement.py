#!/usr/bin/env python3
"""
_comparison_v2_supplement.py
============================
Supplement run for Comparison v2: add Half-Space Trees and LODA streaming methods.

Does NOT re-run existing methods (IF, NormalAE, MemStream, OCSVM).
Reuses existing comparison_results.json and adds new results to it.
"""
from __future__ import annotations

import os
import sys
import json
import time
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"
sys.path.insert(0, r"c:\proj\ldt\HP_benchmark_v5")

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
LOGGER = logging.getLogger("comparison_v2")


def find_best_threshold(scores, gt_mask):
    best_f1, best_t = 0, 0.0
    for t in np.linspace(scores.min(), scores.max(), 2000):
        pred = scores >= t
        tp = (pred & gt_mask).sum()
        fp = (pred & ~gt_mask).sum()
        fn = (~pred & gt_mask).sum()
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def evaluate_scores(scores, gt_mask, threshold):
    from sklearn.metrics import roc_auc_score, average_precision_score
    pred = scores >= threshold
    tp = int((pred & gt_mask).sum())
    fp = int((pred & ~gt_mask).sum())
    tn = int((~pred & ~gt_mask).sum())
    fn = int((~pred & gt_mask).sum())
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    return {
        "auc_roc": float(roc_auc_score(gt_mask, scores)),
        "auc_pr": float(average_precision_score(gt_mask, scores)),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "score_normal_mean": float(scores[~gt_mask].mean()),
        "score_anomaly_mean": float(scores[gt_mask].mean()),
        "separation_ratio": float(scores[gt_mask].mean() / (scores[~gt_mask].mean() + 1e-10)),
        "fpr": float(fp / (fp + tn + 1e-10)),
        "acc": float((tp + tn) / (len(gt_mask) + 1e-10)),
    }


def run_hstrees(X_train, X_val, gt_mask, window_size=50, n_trees=5, max_depth=5, seed=42):
    try:
        from pysad.models import HalfSpaceTrees
    except ImportError:
        return {"method": "HSTrees", "error": "pysad not installed"}

    t0 = time.time()
    try:
        fm = X_train.min(axis=0)
        fM = X_train.max(axis=0)
        model = HalfSpaceTrees(
            feature_mins=fm, feature_maxes=fM,
            window_size=window_size, num_trees=n_trees,
            max_depth=max_depth,
        )
        model.fit(X_train)
        scores = model.score(X_val)
        if scores.max() != scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        best_t, best_f1 = find_best_threshold(scores, gt_mask)
        metrics = evaluate_scores(scores, gt_mask, threshold=best_t)
        elapsed = time.time() - t0
        LOGGER.info("  [HSTrees] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
                   metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], elapsed)
        return {
            "method": "Half-Space Trees", "category": "streaming",
            "auc_roc": metrics["auc_roc"], "auc_pr": metrics["auc_pr"],
            "f1": metrics["f1"], "precision": metrics["precision"],
            "recall": metrics["recall"], "fpr": metrics["fpr"], "acc": metrics["acc"],
            "best_threshold": float(best_t),
            "score_normal_mean": float(metrics["score_normal_mean"]),
            "score_anomaly_mean": float(metrics["score_anomaly_mean"]),
            "separation_ratio": float(metrics["separation_ratio"]),
            "time_s": elapsed,
            "params": {"window_size": window_size, "n_trees": n_trees, "max_depth": max_depth},
            "_raw_scores": scores.tolist(),
        }
    except Exception as e:
        LOGGER.error("  [HSTrees] Error: %s", e)
        import traceback; traceback.print_exc()
        return {"method": "HSTrees", "error": str(e), "_raw_scores": []}


def run_inc_pca(X_train, X_val, gt_mask, n_components=15, percentile=95, seed=42):
    """
    IncrementalPCA reconstruction error — streaming anomaly detection.
    Trains on X_train, then scores X_val by reconstruction error.
    """
    from sklearn.decomposition import IncrementalPCA

    t0 = time.time()
    try:
        ipca = IncrementalPCA(n_components=n_components)
        ipca.partial_fit(X_train)
        recon = ipca.inverse_transform(ipca.transform(X_val))
        scores = np.linalg.norm(X_val - recon, axis=1).astype(np.float64)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        best_t, best_f1 = find_best_threshold(scores, gt_mask)
        metrics = evaluate_scores(scores, gt_mask, threshold=best_t)
        elapsed = time.time() - t0
        LOGGER.info("  [IncPCA] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
                   metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], elapsed)
        return {
            "method": "Inc. PCA", "category": "streaming",
            "auc_roc": metrics["auc_roc"], "auc_pr": metrics["auc_pr"],
            "f1": metrics["f1"], "precision": metrics["precision"],
            "recall": metrics["recall"], "fpr": metrics["fpr"], "acc": metrics["acc"],
            "best_threshold": float(best_t),
            "score_normal_mean": float(metrics["score_normal_mean"]),
            "score_anomaly_mean": float(metrics["score_anomaly_mean"]),
            "separation_ratio": float(metrics["separation_ratio"]),
            "time_s": elapsed,
            "params": {"n_components": n_components, "percentile": percentile},
            "_raw_scores": scores.tolist(),
        }
    except Exception as e:
        LOGGER.error("  [IncPCA] Error: %s", e)
        import traceback; traceback.print_exc()
        return {"method": "Inc. PCA", "error": str(e), "_raw_scores": []}


def run_sgd_ocsvm(X_train, X_val, gt_mask, nu=0.05, seed=42):
    """
    SGD-based One-Class SVM — streaming-friendly (mini-batch SGD).
    """
    from sklearn.linear_model import SGDOneClassSVM

    t0 = time.time()
    try:
        model = SGDOneClassSVM(nu=nu, random_state=seed, learning_rate="optimal")
        model.fit(X_train)
        scores = -model.score_samples(X_val)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        best_t, best_f1 = find_best_threshold(scores, gt_mask)
        metrics = evaluate_scores(scores, gt_mask, threshold=best_t)
        elapsed = time.time() - t0
        LOGGER.info("  [SGD-OCSVM] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
                   metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], elapsed)
        return {
            "method": "SGD One-Class SVM", "category": "streaming",
            "auc_roc": metrics["auc_roc"], "auc_pr": metrics["auc_pr"],
            "f1": metrics["f1"], "precision": metrics["precision"],
            "recall": metrics["recall"], "fpr": metrics["fpr"], "acc": metrics["acc"],
            "best_threshold": float(best_t),
            "score_normal_mean": float(metrics["score_normal_mean"]),
            "score_anomaly_mean": float(metrics["score_anomaly_mean"]),
            "separation_ratio": float(metrics["separation_ratio"]),
            "time_s": elapsed,
            "params": {"nu": nu},
            "_raw_scores": scores.tolist(),
        }
    except Exception as e:
        LOGGER.error("  [SGD-OCSVM] Error: %s", e)
        import traceback; traceback.print_exc()
        return {"method": "SGD One-Class SVM", "error": str(e), "_raw_scores": []}


def _normalize(result):
    """Ensure result has _mean/_std/_runs fields (handle both old and new formats)."""
    for k in ["auc_roc", "auc_pr", "f1", "precision", "recall", "time_s"]:
        mean_key = f"{k}_mean"
        std_key = f"{k}_std"
        runs_key = f"{k}_runs"
        if mean_key not in result:
            raw = result.get(k, 0.0)
            result[mean_key] = float(raw)
            result[std_key] = 0.0
            result[runs_key] = [float(raw)]
    return result


def save_updated_results(existing_path, results):
    sortable = [(k, v.get("auc_pr_mean", 0)) for k, v in results.items() if "error" not in v]
    sortable.sort(key=lambda x: x[1], reverse=True)
    output = {"results": {}, "ranking": [k for k, _ in sortable]}
    try:
        with open(existing_path) as f:
            existing = json.load(f)
        for k in ["config", "dataset", "timestamp"]:
            if k in existing:
                output[k] = existing[k]
    except Exception:
        pass
    for method_key, summary in results.items():
        output["results"][method_key] = summary
    with open(existing_path, "w") as f:
        json.dump(output, f, indent=2)
    LOGGER.info("Saved: %s", existing_path)

    LOGGER.info("")
    LOGGER.info("=" * 90)
    LOGGER.info("COMPARISON v2 RESULTS (sorted by AUC-PR)")
    LOGGER.info("=" * 90)
    LOGGER.info(f"{'Rank':<5} {'Method':<30} {'Cat':<10} "
                f"{'AUC-PR':>8} {'AUC-ROC':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Time':>8}")
    LOGGER.info("-" * 90)
    for rank, (method_key, _) in enumerate(sortable, 1):
        r = results[method_key]
        LOGGER.info(
            f"{rank:<5} {r['method']:<30} {r['category']:<10} "
            f"{r['auc_pr_mean']:>8.4f} {r['auc_roc_mean']:>8.4f} "
            f"{r['f1_mean']:>8.4f} {r['precision_mean']:>8.4f} {r['recall_mean']:>8.4f} "
            f"{r['time_s_mean']:>7.1f}s"
        )


def generate_charts(existing_path, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception as e:
        LOGGER.warning("Charts skipped (matplotlib unavailable): %s", e)
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(existing_path) as f:
        data = json.load(f)
    results = data.get("results", {})
    valid = {k: v for k, v in results.items() if "error" not in v and "auc_pr_mean" in v}

    if not valid:
        LOGGER.warning("No valid results for charts")
        return

    sortable = sorted(valid.items(), key=lambda x: x[1].get("auc_pr_mean", 0), reverse=True)
    methods = [k for k, _ in sortable]

    colors_map = {
        "IsolationForest": "#888888", "NormalAE": "#2196F3",
        "MemStream": "#4CAF50", "OCSVM": "#FF9800",
        "HSTrees": "#9C27B0",
        "IncPCA": "#00BCD4",
        "SGD-OCSVM": "#E91E63",
    }
    streaming_color = "#4CAF50"
    offline_color = "#2196F3"

    labels = [valid[m]["method"] for m in methods]
    pr_vals = [valid[m].get("auc_pr_mean", 0) for m in methods]
    pr_errs = [valid[m].get("auc_pr_std", 0) or 0 for m in methods]
    f1_vals = [valid[m].get("f1_mean", 0) for m in methods]
    f1_errs = [valid[m].get("f1_std", 0) or 0 for m in methods]
    time_vals = [valid[m]["time_s_mean"] for m in methods]
    sep_vals = [valid[m].get("separation_ratio", 0) for m in methods]
    all_colors = [streaming_color if valid[m].get("category") == "streaming" else offline_color for m in methods]
    n_stream = sum(1 for m in methods if valid[m].get("category") == "streaming")

    # Chart 1: 4-panel
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#f9f9f9")

    panels = [
        ("AUC-PR", pr_vals, pr_errs, 0.9),
        ("F1 Score", f1_vals, f1_errs, None),
        ("Runtime (s)", time_vals, None, None),
        ("Separation Ratio", sep_vals, None, None),
    ]
    for ax, (title, vals, errs, hline) in zip(axes.flat, panels):
        bars = ax.bar(labels, vals, color=all_colors, alpha=0.85, edgecolor="white",
                      yerr=errs, capsize=3)
        for bar, v in zip(bars, vals):
            offset = 0.015 if hline or hline is None and v > 0 else 0.01
            label = f"{v:.3f}" if v <= 1 else f"{v:.0f}x" if title == "Separation Ratio" else f"{v:.0f}s"
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                    label, ha="center", va="bottom", fontsize=8)
        if hline:
            ax.axhline(y=hline, color="red", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Score" if title != "Runtime (s)" else "Seconds")
        if title == "Runtime (s)":
            ax.set_yscale("log")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)

    stream_patch = mpatches.Patch(color=streaming_color, label="Streaming", alpha=0.85)
    offline_patch = mpatches.Patch(color=offline_color, label="Offline", alpha=0.85)
    fig.legend(handles=[stream_patch, offline_patch], loc="upper right", fontsize=10)
    fig.suptitle("Comparison v2 — MemStream vs Streaming & Offline Baselines\n"
                 "(New: Half-Space Trees, LODA)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(output_dir, "02_comparison_v2_full.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved: %s", out)

    # Chart 2: Streaming methods detail
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#f9f9f9")

    stream_idxs = [i for i, m in enumerate(methods) if valid[m].get("category") == "streaming"]
    stream_labels = [labels[i] for i in stream_idxs]
    stream_pr = [pr_vals[i] for i in stream_idxs]
    stream_pr_err = [pr_errs[i] for i in stream_idxs]
    stream_f1 = [f1_vals[i] for i in stream_idxs]
    stream_f1_err = [f1_errs[i] for i in stream_idxs]
    stream_time = [time_vals[i] for i in stream_idxs]
    stream_cols = [colors_map.get(methods[i], streaming_color) for i in stream_idxs]

    ax = axes[0]
    bars = ax.bar(stream_labels, stream_pr, color=stream_cols, alpha=0.85, edgecolor="white", yerr=stream_pr_err, capsize=4)
    for bar, v in zip(bars, stream_pr):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("AUC-PR", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score")
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    ax = axes[1]
    bars = ax.bar(stream_labels, stream_f1, color=stream_cols, alpha=0.85, edgecolor="white", yerr=stream_f1_err, capsize=4)
    for bar, v in zip(bars, stream_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("F1 Score", fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    ax = axes[2]
    bars = ax.bar(stream_labels, stream_time, color=stream_cols, alpha=0.85, edgecolor="white")
    for bar, t in zip(bars, stream_time):
        label = f"{t:.0f}s" if t >= 1 else f"{t*1000:.0f}ms"
        ax.text(bar.get_x() + bar.get_width() / 2, t + max(stream_time) * 0.02,
                label, ha="center", va="bottom", fontsize=10)
    ax.set_title("Runtime", fontsize=12, fontweight="bold")
    ax.set_ylabel("Seconds")
    ax.set_yscale("log")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("Streaming Methods — Detail Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "01_streaming_methods_v2.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved: %s", out)

    # Chart 3: AUC-PR vs F1 grouped
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#f9f9f9")
    x = np.arange(len(methods))
    w = 0.35
    bars1 = ax.bar(x - w/2, pr_vals, w, label="AUC-PR", color="#4CAF50", yerr=pr_errs, capsize=3, alpha=0.85)
    bars2 = ax.bar(x + w/2, f1_vals, w, label="F1", color="#FF9800", yerr=f1_errs, capsize=3, alpha=0.85)
    for bar, v in zip(bars1, pr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(bars2, f1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=11)
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.legend(fontsize=11)
    ax.set_title("Comparison v2 — AUC-PR vs F1", fontsize=13, fontweight="bold")
    # Shade streaming region
    ax.axvspan(-0.5, n_stream - 0.5, alpha=0.04, color=streaming_color, label="streaming")
    plt.tight_layout()
    out = os.path.join(output_dir, "02_comparison_v2_auc_f1.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved: %s", out)


def main():
    OUTPUT_DIR = r"c:\proj\ldt\HP_benchmark_v5\results\comparison"
    EXISTING = os.path.join(OUTPUT_DIR, "comparison_results.json")

    # Load existing
    existing = {}
    if os.path.exists(EXISTING):
        with open(EXISTING) as f:
            data = json.load(f)
        raw_results = data.get("results", {})
        for k, v in raw_results.items():
            existing[k] = _normalize(v)
        LOGGER.info("Loaded %d existing methods: %s", len(existing), list(existing.keys()))

    # Load data
    from benchmark_core import extract_features_from_parquet
    LOGGER.info("Loading data...")
    X_train, *_ = extract_features_from_parquet(
        r"c:\proj\ldt\HP_benchmark_v5\data\train_clean.parquet", max_rows=200000
    )
    X_test, *_ = extract_features_from_parquet(
        r"c:\proj\ldt\HP_benchmark_v5\data\test_polluted.parquet", max_rows=100000
    )
    gt_mask = np.load(r"c:\proj\ldt\HP_benchmark_v5\data\test\ground_truth_mask.npy")
    gt_mask = gt_mask[-len(X_test):]
    LOGGER.info("Train: %d, Test: %d (GT: %d anomalies, %.2f%%)",
               len(X_train), len(X_test), gt_mask.sum(), gt_mask.mean() * 100)

    new_methods = []

    LOGGER.info("")
    LOGGER.info("=== Running Half-Space Trees ===")
    r = run_hstrees(X_train, X_test, gt_mask)
    if "error" not in r:
        new_methods.append(("HSTrees", _normalize(r)))
    else:
        LOGGER.warning("HSTrees failed: %s", r.get("error"))

    LOGGER.info("")
    LOGGER.info("=== Running Incremental PCA ===")
    r = run_inc_pca(X_train, X_test, gt_mask)
    if "error" not in r:
        new_methods.append(("IncPCA", _normalize(r)))
    else:
        LOGGER.warning("IncPCA failed: %s", r.get("error"))

    LOGGER.info("")
    LOGGER.info("=== Running SGD One-Class SVM ===")
    r = run_sgd_ocsvm(X_train, X_test, gt_mask)
    if "error" not in r:
        new_methods.append(("SGD-OCSVM", _normalize(r)))
    else:
        LOGGER.warning("SGD-OCSVM failed: %s", r.get("error"))

    # Merge
    all_results = existing.copy()
    for method_key, summary in new_methods:
        summary = _normalize(summary)
        all_results[method_key] = summary
        LOGGER.info("  Added %s: AUC-PR=%.4f, F1=%.4f",
                   summary["method"], summary["auc_pr_mean"], summary["f1_mean"])

    save_updated_results(EXISTING, all_results)
    LOGGER.info("")
    LOGGER.info("Generating charts...")
    generate_charts(EXISTING, OUTPUT_DIR)
    LOGGER.info("DONE.")


if __name__ == "__main__":
    main()
