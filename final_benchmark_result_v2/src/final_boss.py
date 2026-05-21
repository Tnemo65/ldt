#!/usr/bin/env python3
"""
Part 5: Final Boss Run.
Uses the single best config from Part 1 grid search.
Produces complete output:
  - Training loss curve (chart + npy)
  - Raw anomaly scores (npy)
  - Detailed metrics (JSON)
  - Per-point CSV (index, score, gt_label, predicted_label, category)
  - Visualizations:
      detection_timeline_full.png
      score_distribution.png
      pointwise_heatmap.png
"""
from __future__ import annotations

import os, sys, json, time, csv, argparse
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src._config import (
    TRAIN_PATH, TEST_PATH, GT_MASK_PATH,
    RESULTS_DIR,
)
from src.benchmark_core import (
    extract_features_from_parquet,
    evaluate_scores, find_best_threshold,
    plot_score_distribution, plot_score_timeseries,
    plot_detection_timeseries, plot_pointwise_scores,
    MemStreamPipeline,
)

import numpy as np

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
LOGGER = logging.getLogger("final_boss")


# ---------------------------------------------------------------------------
def load_best_config():
    path = RESULTS_DIR / "grid_search" / "best_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Best config not found: {path}")
    with open(path) as f:
        cfg = json.load(f)
    LOGGER.info("Loaded best config: %s", cfg.get("config_id"))
    return cfg


def run_final_boss(base_config, max_train=500000, max_test=None):
    out_dir = RESULTS_DIR / "final_boss"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    LOGGER.info("Loading training data...")
    X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
        str(TRAIN_PATH), max_rows=max_train
    )
    LOGGER.info("Loading test data...")
    X_test, hours_test, dows_test, rcs_test, nb_test = extract_features_from_parquet(
        str(TEST_PATH), max_rows=max_test if max_test else None
    )
    gt_mask = np.load(str(GT_MASK_PATH))
    # Align GT to test data
    gt_mask = gt_mask[-len(X_test):]
    LOGGER.info("Train: %d, Test: %d (GT: %d anomalies, %.2f%%)",
               len(X_train), len(X_test), gt_mask.sum(), gt_mask.mean() * 100)

    # Extract config
    mem_len = int(base_config.get("memory_len", 1024))
    k = int(base_config.get("k", 10))
    gamma = float(base_config.get("gamma", 0.0))
    beta = float(base_config.get("beta", 0.001))
    noise_std = float(base_config.get("noise_std", 0.001))
    lr = float(base_config.get("lr", 0.01))
    out_dim = int(base_config.get("out_dim", 38))
    epochs = int(base_config.get("epochs", 5000))

    LOGGER.info("Config: memory_len=%d, k=%d, gamma=%.3f, beta=%.4f, "
                "noise_std=%.4f, lr=%.4f, out_dim=%d, epochs=%d",
                mem_len, k, gamma, beta, noise_std, lr, out_dim, epochs)

    # ── Training ──────────────────────────────────────────────────────────
    LOGGER.info("=" * 60)
    LOGGER.info("TRAINING")
    LOGGER.info("=" * 60)
    t0 = time.time()

    pipeline = MemStreamPipeline(
        d=38, out_dim=out_dim,
        memory_len=mem_len, k=k, gamma=gamma, beta=beta,
        noise_std=noise_std, lr=lr, epochs=epochs,
        batch_size=1024, seed=42,
        cb_warmup=min(4096, mem_len * 4), verbose=True,
    )

    warmup_size = min(mem_len, len(X_train))
    pipeline.train(
        X_train, hours_train, dows_train, rcs_train, nb_train,
        X_warmup=X_train[:warmup_size],
        hours_warmup=hours_train[:warmup_size],
        dows_warmup=dows_train[:warmup_size],
        rcs_warmup=rcs_train[:warmup_size],
        nb_warmup=nb_train[:warmup_size],
    )
    train_time = time.time() - t0
    LOGGER.info("Training complete: %.1fs", train_time)

    # Save epoch losses
    if hasattr(pipeline, "epoch_losses") and pipeline.epoch_losses:
        losses = np.array(pipeline.epoch_losses)
        np.save(out_dir / "epoch_losses.npy", losses)
        LOGGER.info("Saved epoch_losses.npy (%d epochs)", len(losses))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(losses, color="#2196F3", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve")
        ax.grid(True, alpha=0.3)
        # Mark early/late phases
        if len(losses) > 100:
            ax.axvline(len(losses) * 0.1, color="orange", linestyle="--",
                      alpha=0.6, label="10% mark")
            ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        LOGGER.info("Saved loss_curve.png")
    else:
        LOGGER.warning("No epoch_losses found — loss chart skipped")

    # ── Scoring ───────────────────────────────────────────────────────────
    LOGGER.info("=" * 60)
    LOGGER.info("SCORING TEST DATA")
    LOGGER.info("=" * 60)
    t1 = time.time()

    adj_scores, raw_scores = pipeline.score_stream(
        X_test, hours_test, dows_test, rcs_test, nb_test,
        gt_mask=gt_mask, update_memory=True,
    )
    score_time = time.time() - t1
    LOGGER.info("Scoring complete: %.1fs", score_time)

    # Save raw scores
    np.save(out_dir / "raw_scores.npy", adj_scores)
    LOGGER.info("Saved raw_scores.npy")

    # ── Find best threshold ───────────────────────────────────────────────
    best_t, best_f1 = find_best_threshold(adj_scores, gt_mask)
    LOGGER.info("Best threshold=%.4f (F1=%.4f)", best_t, best_f1)

    # ── Evaluation ────────────────────────────────────────────────────────
    metrics = evaluate_scores(adj_scores, gt_mask, best_t)
    pred_mask = adj_scores >= best_t

    tp_mask = pred_mask & gt_mask
    fp_mask = pred_mask & ~gt_mask
    tn_mask = ~pred_mask & ~gt_mask
    fn_mask = ~pred_mask & gt_mask

    metrics["best_threshold"] = float(best_t)
    metrics["train_time_s"] = float(train_time)
    metrics["score_time_s"] = float(score_time)
    metrics["total_time_s"] = float(train_time + score_time)
    metrics["config_id"] = base_config.get("config_id", "unknown")
    metrics["memory_len"] = mem_len
    metrics["k"] = k
    metrics["gamma"] = gamma
    metrics["beta"] = beta
    metrics["noise_std"] = noise_std
    metrics["lr"] = lr
    metrics["out_dim"] = out_dim
    metrics["epochs"] = epochs

    # Per-type breakdown
    try:
        per_type_path = RESULTS_DIR.parent / "data" / "test" / "ground_truth_per_type.json"
        with open(per_type_path) as f:
            per_type = json.load(f)
        type_breakdown = {}
        for anom_type, entries in per_type.items():
            indices = [e["index"] for e in entries if isinstance(e, dict) and "index" in e]
            if indices:
                type_gt = np.zeros(len(gt_mask), dtype=bool)
                type_gt[indices] = True
                type_pred = np.zeros(len(pred_mask), dtype=bool)
                type_pred[indices] = True
                type_tp = int((type_pred & type_gt).sum())
                type_fn = int((~type_pred & type_gt).sum())
                type_fp = int((type_pred & ~type_gt).sum())
                type_tn = int((~type_pred & ~type_gt).sum())
                type_prec = type_tp / (type_tp + type_fp + 1e-10)
                type_rec = type_tp / (type_tp + type_fn + 1e-10)
                type_f1 = 2 * type_prec * type_rec / (type_prec + type_rec + 1e-10)
                type_breakdown[anom_type] = {
                    "tp": type_tp, "fn": type_fn, "fp": type_fp, "tn": type_tn,
                    "precision": round(type_prec, 4),
                    "recall": round(type_rec, 4),
                    "f1": round(type_f1, 4),
                    "total_gt": int(type_gt.sum()),
                }
        metrics["per_type_breakdown"] = type_breakdown
    except Exception as e:
        LOGGER.warning("Could not load per-type breakdown: %s", e)

    # Save detailed metrics
    metrics_path = out_dir / "detailed_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info("Saved detailed_metrics.json")

    # ── Per-point CSV ────────────────────────────────────────────────────
    csv_path = out_dir / "per_point_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "score", "gt_label", "predicted_label", "category"])
        for i in range(len(adj_scores)):
            score = float(adj_scores[i])
            gt = int(gt_mask[i])
            pred = int(pred_mask[i])
            if gt == 0 and pred == 0:
                cat = "TN"
            elif gt == 0 and pred == 1:
                cat = "FP"
            elif gt == 1 and pred == 0:
                cat = "FN"
            else:
                cat = "TP"
            writer.writerow([i, f"{score:.6f}", gt, pred, cat])
    LOGGER.info("Saved per_point_table.csv (%d rows)", len(adj_scores))

    # ── Visualizations ──────────────────────────────────────────────────
    LOGGER.info("=" * 60)
    LOGGER.info("GENERATING VISUALIZATIONS")
    LOGGER.info("=" * 60)

    try:
        # 1. Detection timeline (2-panel: scores+threshold + detection)
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        fig.suptitle(f"Final Boss: Detection Timeline — {base_config.get('config_id', '')}",
                    fontsize=13, fontweight="bold")

        # Panel 1: scores + threshold
        ax = axes[0]
        # Subsample for visualization if too many points
        step = max(1, len(adj_scores) // 50000)
        idx_vis = np.arange(0, len(adj_scores), step)
        scores_vis = adj_scores[idx_vis]
        gt_vis = gt_mask[idx_vis]
        pred_vis = pred_mask[idx_vis]

        # Color by category
        colors = np.array(["#1565C0"] * len(idx_vis))  # default blue (normal)
        colors[gt_vis & ~pred_vis] = "#E53935"          # red = FN
        colors[~gt_vis & pred_vis] = "#FDD835"          # yellow = FP
        colors[gt_vis & pred_vis] = "#8E24AA"          # purple = TP

        ax.scatter(idx_vis, scores_vis, c=colors, s=1.0, alpha=0.4)
        ax.axhline(best_t, color="black", linestyle="--", linewidth=1.5, label=f"Threshold={best_t:.2f}")
        ax.set_ylabel("Anomaly Score")
        ax.set_title("Anomaly Scores (color: blue=TN, yellow=FP, red=FN, purple=TP)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Panel 2: detection binary signal
        ax2 = axes[1]
        ax2.fill_between(idx_vis, 0, gt_vis.astype(int), alpha=0.25, color="red", label="Ground Truth")
        ax2.fill_between(idx_vis, 0, pred_vis.astype(int), alpha=0.25, color="blue", label="Prediction")
        ax2.set_ylabel("Detection (1=Anomaly)")
        ax2.set_xlabel("Data Point Index")
        ax2.set_title("Detection Timeline")
        ax2.legend(loc="upper right")
        ax2.set_ylim(-0.1, 1.3)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "detection_timeline_full.png", dpi=150, bbox_inches="tight")
        plt.close()
        LOGGER.info("Saved detection_timeline_full.png")
    except Exception as e:
        LOGGER.warning("Could not generate detection_timeline_full.png: %s", e)
        import traceback; traceback.print_exc()

    try:
        # 2. Score distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Score Distribution — Final Boss")

        normal_scores = adj_scores[~gt_mask]
        anomaly_scores = adj_scores[gt_mask]

        ax = axes[0]
        ax.hist(normal_scores, bins=80, alpha=0.7, color="#1565C0",
                label=f"Normal (n={len(normal_scores):,})", density=True)
        ax.hist(anomaly_scores, bins=80, alpha=0.7, color="#E53935",
                label=f"Anomaly (n={len(anomaly_scores):,})", density=True)
        ax.axvline(best_t, color="black", linestyle="--", linewidth=1.5,
                   label=f"Threshold={best_t:.2f}")
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Density")
        ax.set_title("Histogram")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        box_data = [normal_scores, anomaly_scores]
        bp = ax2.boxplot(box_data, labels=["Normal", "Anomaly"],
                         patch_artist=True, notch=True)
        bp["boxes"][0].set_facecolor("#1565C0")
        bp["boxes"][1].set_facecolor("#E53935")
        ax2.set_ylabel("Anomaly Score")
        ax2.set_title("Box Plot")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "score_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()
        LOGGER.info("Saved score_distribution.png")
    except Exception as e:
        LOGGER.warning("Could not generate score_distribution.png: %s", e)
        import traceback; traceback.print_exc()

    try:
        # 3. Pointwise heatmap
        plot_pointwise_scores(adj_scores, gt_mask, pred_mask, best_t, str(out_dir / "pointwise_heatmap.png"))
        LOGGER.info("Saved pointwise_heatmap.png")
    except Exception as e:
        LOGGER.warning("Could not generate pointwise_heatmap.png: %s", e)
        import traceback; traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────────────
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("FINAL BOSS COMPLETE")
    LOGGER.info("=" * 60)
    LOGGER.info("  Config:        %s", base_config.get("config_id"))
    LOGGER.info("  AUC-ROC:      %.4f", metrics["auc_roc"])
    LOGGER.info("  AUC-PR:       %.4f", metrics["auc_pr"])
    LOGGER.info("  F1:           %.4f", metrics["f1"])
    LOGGER.info("  Precision:    %.4f", metrics["precision"])
    LOGGER.info("  Recall:       %.4f", metrics["recall"])
    LOGGER.info("  Train Time:   %.1fs", train_time)
    LOGGER.info("  Score Time:   %.1fs", score_time)
    LOGGER.info("  Output:       %s", out_dir)
    LOGGER.info("=" * 60)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final Boss — Best Config Run")
    parser.add_argument("--max-train", type=int, default=500000)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    cfg = load_best_config()
    run_final_boss(cfg, max_train=args.max_train, max_test=args.max_test)
