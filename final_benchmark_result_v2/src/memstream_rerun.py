#!/usr/bin/env python3
"""
MemStream v5 Standalone Rerun — FIX update_memory contamination

Root cause of low AUC-PR (0.743 vs expected 0.9x):
  - comparison.py calls score_stream with update_memory=True
  - beta=0.001 is so low that almost EVERY anomaly passes the memory update gate
  - Anomaly-encoded vectors pollute the kNN memory, destroying score discrimination
  - Grid search uses update_memory=False → clean memory → AUC-PR=0.916

Fix: Run comparison with update_memory=False for fair evaluation.
Also test update_memory=True to quantify the contamination effect.
"""
from __future__ import annotations

import os, sys, json, time
from pathlib import Path
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
LOGGER = logging.getLogger("ms_rerun")

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
from src.benchmark_core import (
    MemStreamPipeline, extract_features_from_parquet,
    evaluate_scores, find_best_threshold, compute_auc_roc, compute_auc_pr,
)


def run_memstream_comparison(
    X_train, hours_train, dows_train, rcs_train, nb_train,
    X_val, hours_val, dows_val, rcs_val, nb_val,
    gt_mask, best_config, epochs=1000, update_memory=True, seed=42, batch_size=None,
):
    """Run MemStream with a given update_memory setting."""
    t0 = time.time()

    if batch_size is None:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        batch_size = max(64, min(int(available_gb * 0.5), 2048))

    cfg = best_config
    pipeline = MemStreamPipeline(
        d=cfg.get("d", 38),
        out_dim=cfg.get("out_dim", 68),
        memory_len=cfg["memory_len"],
        k=cfg["k"],
        gamma=cfg.get("gamma", 0.0),
        beta=cfg.get("beta", 0.5),
        noise_std=cfg.get("noise_std", 0.001),
        lr=cfg.get("lr", 0.01),
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        cb_warmup=min(4096, cfg["memory_len"] * 4),
        verbose=False,
        adam_betas=tuple(cfg.get("adam_betas", [0.9, 0.999])),
    )

    # Training (cold-start)
    pipeline.train(
        X_train, hours_train, dows_train, rcs_train, nb_train,
        X_warmup=X_train[:cfg["memory_len"]],
        hours_warmup=hours_train[:cfg["memory_len"]],
        dows_warmup=dows_train[:cfg["memory_len"]],
        rcs_warmup=rcs_train[:cfg["memory_len"]],
        nb_warmup=nb_train[:cfg["memory_len"]],
    )

    # Scoring
    adj_scores, ms_metrics = pipeline.score_stream(
        X_val, hours_val, dows_val, rcs_val, nb_val,
        gt_mask=gt_mask,
        update_memory=update_memory,
    )

    best_t, best_f1 = find_best_threshold(adj_scores, gt_mask)
    metrics = evaluate_scores(adj_scores, gt_mask, threshold=best_t)
    elapsed = time.time() - t0

    LOGGER.info("  [update_memory=%s] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f [%.1fs]",
               update_memory, metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], elapsed)
    LOGGER.info("    Separation: normal=%.2f, anomaly=%.2f (%.2fx)",
               metrics["score_normal_mean"], metrics["score_anomaly_mean"],
               metrics["separation_ratio"])
    LOGGER.info("    Precision=%.4f, Recall=%.4f, Best threshold=%.4f",
               metrics["precision"], metrics["recall"], best_t)

    return {
        "update_memory": update_memory,
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
        "n_memory_updates": ms_metrics.get("n_memory_updates", 0),
        "params": {
            "memory_len": cfg["memory_len"],
            "k": cfg["k"],
            "gamma": cfg.get("gamma", 0.0),
            "beta": cfg.get("beta", 0.5),
            "noise_std": cfg.get("noise_std", 0.001),
            "lr": cfg.get("lr", 0.01),
            "out_dim": cfg.get("out_dim", 68),
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "_raw_scores": adj_scores.tolist(),
        "_epoch_losses": [float(x) for x in pipeline.epoch_losses],
    }


def main():
    # Load best config from grid search
    grid_path = BASE / "results" / "grid_search" / "final_results.json"
    with open(grid_path) as f:
        grid_data = json.load(f)
    best_config = grid_data.get("best_config", {})
    LOGGER.info("Loaded best config: %s", best_config.get("config_id", "N/A"))
    LOGGER.info("  mem=%d, k=%d, gamma=%.3f, beta=%.4f, out_dim=%d, epochs=%d",
               best_config["memory_len"], best_config["k"],
               best_config.get("gamma", 0), best_config.get("beta", 0.5),
               best_config.get("out_dim", 68), best_config.get("epochs", 1000))

    # Load data: use same splits as comparison (train_clean + test_polluted)
    data_dir = BASE / "data"
    train_path = str(data_dir / "train_clean.parquet")
    test_path = str(data_dir / "test_polluted.parquet")

    LOGGER.info("Loading data...")
    X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
        train_path, max_rows=500000
    )
    X_test, hours_test, dows_test, rcs_test, nb_test = extract_features_from_parquet(
        test_path, max_rows=500000
    )
    gt_mask = np.load(str(data_dir / "test" / "ground_truth_mask.npy"))
    gt_mask = gt_mask[-len(X_test):]
    LOGGER.info("Train: %d, Test: %d (GT: %d anomalies, %.2f%%)",
               len(X_train), len(X_test), gt_mask.sum(), gt_mask.mean() * 100)

    # Determine epochs: match what comparison used
    epochs = 1000  # comparison used 1000

    # Run 1: update_memory=False (like grid search, clean memory)
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("RUN 1: update_memory=False (eval mode, clean memory)")
    LOGGER.info("=" * 60)
    result_clean = run_memstream_comparison(
        X_train, hours_train, dows_train, rcs_train, nb_train,
        X_test, hours_test, dows_test, rcs_test, nb_test,
        gt_mask, best_config, epochs=epochs, update_memory=False,
    )

    # Run 2: update_memory=True (streaming, may contaminate memory)
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("RUN 2: update_memory=True (streaming, memory adaptation)")
    LOGGER.info("=" * 60)
    result_stream = run_memstream_comparison(
        X_train, hours_train, dows_train, rcs_train, nb_train,
        X_test, hours_test, dows_test, rcs_test, nb_test,
        gt_mask, best_config, epochs=epochs, update_memory=True,
    )

    # Run 3: Also test with more epochs (5000) as paper recommends
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("RUN 3: update_memory=False with 5000 epochs (paper standard)")
    LOGGER.info("=" * 60)
    result_5k = run_memstream_comparison(
        X_train, hours_train, dows_train, rcs_train, nb_train,
        X_test, hours_test, dows_test, rcs_test, nb_test,
        gt_mask, best_config, epochs=5000, update_memory=False,
    )

    # Save results
    output_dir = BASE / "results" / "comparison"
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "results": {
            "MemStream_v5_eval_mode": result_clean,
            "MemStream_v5_streaming": result_stream,
            "MemStream_v5_eval_mode_5k": result_5k,
        },
        "best_config": best_config,
        "grid_search_comparison": {
            "grid_search_auc_pr": best_config.get("auc_pr"),
            "grid_search_separation_ratio": best_config.get("separation_ratio"),
            "comparison_previous_auc_pr": 0.7434,
            "note": "Previous comparison used update_memory=True which contaminated memory"
        },
    }

    out_path = os.path.join(output_dir, "memstream_rerun_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    LOGGER.info("")
    LOGGER.info("Results saved: %s", out_path)

    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("SUMMARY")
    LOGGER.info("=" * 60)
    LOGGER.info(f"{'Mode':<30} {'AUC-PR':>8} {'AUC-ROC':>8} {'Sep Ratio':>10} {'Memory Updates':>15}")
    LOGGER.info("-" * 60)
    LOGGER.info(f"{'eval_mode (no update) [1k]':<30} {result_clean['auc_pr']:>8.4f} {result_clean['auc_roc']:>8.4f} {result_clean['separation_ratio']:>10.2f}x {result_clean['n_memory_updates']:>15}")
    LOGGER.info(f"{'streaming (update) [1k]':<30} {result_stream['auc_pr']:>8.4f} {result_stream['auc_roc']:>8.4f} {result_stream['separation_ratio']:>10.2f}x {result_stream['n_memory_updates']:>15}")
    LOGGER.info(f"{'eval_mode (no update) [5k]':<30} {result_5k['auc_pr']:>8.4f} {result_5k['auc_roc']:>8.4f} {result_5k['separation_ratio']:>10.2f}x {result_5k['n_memory_updates']:>15}")
    LOGGER.info("-" * 60)
    LOGGER.info("Grid search baseline:  AUC-PR=%.4f (update_memory=False)", best_config.get("auc_pr", 0))

    return output


if __name__ == "__main__":
    main()
