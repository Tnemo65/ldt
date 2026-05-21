#!/usr/bin/env python3
"""
Grid Search Stage 3 — Architecture variation.
Varies: out_dim, memory_len, k on top configs from Stage 2.

Usage:
    python upgrade_experiments/grid_search_s3.py [--skip] [--max-train N]
"""
from __future__ import annotations

import os, sys, json, time, argparse, logging
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

from _config import (TRAIN_PATH, VALID_PATH, GT_MASK_PATH, GRID_SEARCH_STAGE1_DIR, GRID_SEARCH_STAGE3_DIR, RESULTS_DIR)
from benchmark_core import extract_features_from_parquet

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                   stream=sys.stderr)
LOGGER = logging.getLogger("grid_s3")


def find_best_threshold(scores, gt_mask, n_steps=200):
    best_f1, best_t = 0.0, float(scores.mean())
    for t in np.linspace(scores.min(), scores.max(), n_steps):
        pred = scores >= t
        tp = (pred & gt_mask).sum()
        fp = (pred & ~gt_mask).sum()
        fn = (~pred & gt_mask).sum()
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        if f1 > best_f1:
            best_f1, best_t = float(t), float(f1)
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
        "f1": float(f1), "precision": float(prec), "recall": float(rec),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "score_normal_mean": float(scores[~gt_mask].mean()),
        "score_anomaly_mean": float(scores[gt_mask].mean()),
        "separation_ratio": float(scores[gt_mask].mean() / (scores[~gt_mask].mean() + 1e-10)),
        "fpr": float(fp / (fp + tn + 1e-10)),
        "acc": float((tp + tn) / len(gt_mask)),
    }


def run_config(X_train, hours_train, dows_train, rcs_train, nb_train,
               X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask,
               memory_len, k, gamma, beta, noise_std, lr, out_dim, epochs,
               seed=42):
    from benchmark_core import MemStreamPipeline

    pipeline = MemStreamPipeline(
        d=38, out_dim=out_dim,
        memory_len=memory_len, k=k, gamma=gamma, beta=beta,
        noise_std=noise_std, lr=lr, epochs=epochs,
        batch_size=1024, seed=seed,
        cb_warmup=min(4096, memory_len * 4), verbose=False,
    )
    pipeline.train(
        X_train, hours_train, dows_train, rcs_train, nb_train,
        X_warmup=X_train[:memory_len],
        hours_warmup=hours_train[:memory_len],
        dows_warmup=dows_train[:memory_len],
        rcs_warmup=rcs_train[:memory_len],
        nb_warmup=nb_train[:memory_len],
    )
    adj_scores, _ = pipeline.score_stream(
        X_val, hours_val, dows_val, rcs_val, nb_val,
        gt_mask=gt_mask, update_memory=True,
    )
    best_t, best_f1 = find_best_threshold(adj_scores, gt_mask)
    metrics = evaluate_scores(adj_scores, gt_mask, best_t)
    return metrics, best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", action="store_true", help="Skip existing configs")
    parser.add_argument("--max-train", type=int, default=500000)
    parser.add_argument("--max-val", type=int, default=500000)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    OUT_DIR = GRID_SEARCH_STAGE3_DIR
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load best from Stage 2, fallback to Stage 1
    s2_path = RESULTS_DIR / "grid_search_s2" / "summary.json"
    s1_path = GRID_SEARCH_STAGE1_DIR / "summary.json"

    top_configs = []
    if os.path.exists(s2_path):
        with open(s2_path) as f:
            s2_data = json.load(f)
        results = s2_data.get("results", [])
        results.sort(key=lambda x: x.get("auc_pr", 0), reverse=True)
        top_configs = results[:args.top]
        LOGGER.info("Loaded top %d from Stage 2:", len(top_configs))
        for c in top_configs:
            LOGGER.info("  %s: AUC-PR=%.4f", c.get("config_id"), c.get("auc_pr"))

    if not top_configs:
        with open(s1_path) as f:
            s1_data = json.load(f)
        results = s1_data.get("results", [])
        results.sort(key=lambda x: x.get("auc_pr", 0), reverse=True)
        top_configs = results[:args.top]
        LOGGER.info("Fell back to Stage 1 top %d:", len(top_configs))

    # Load data
    LOGGER.info("Loading data...")
    X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
        str(TRAIN_PATH), max_rows=args.max_train
    )
    X_val, hours_val, dows_val, rcs_val, nb_val = extract_features_from_parquet(
        str(VALID_PATH), max_rows=args.max_val
    )
    gt_mask = np.load(str(GT_MASK_PATH))
    gt_mask = gt_mask[-len(X_val):]
    LOGGER.info("Train: %d, Val: %d (GT: %d anomalies, %.2f%%)",
               len(X_train), len(X_val), gt_mask.sum(), gt_mask.mean() * 100)

    # Architecture grid
    arch_grid = {
        "out_dim":    [19, 38, 76],        # d/2, d, 2d
        "memory_len":  [512, 1024, 2048],    # 3 values
        "k":          [3, 5, 10],           # 3 values
    }
    n_per = (len(arch_grid["out_dim"]) * len(arch_grid["memory_len"]) *
             len(arch_grid["k"]))
    n_total = len(top_configs) * n_per
    # Total: 3 tops × 27 = 81 configs

    all_results = []
    idx = 0
    for base in top_configs:
        base_noise = base.get("noise_std", 0.001)
        base_lr = base.get("lr", 0.01)
        base_epochs = base.get("epochs", 5000)

        for od in arch_grid["out_dim"]:
            for ml in arch_grid["memory_len"]:
                for k in arch_grid["k"]:
                    idx += 1
                    cfg_id = (f"M{ml}_k{k}_od{od}_n{base_noise}_e{base_epochs}_lr{int(base_lr*1000)}")

                    result_path = os.path.join(OUT_DIR, f"{cfg_id}.json")
                    if args.skip and os.path.exists(result_path):
                        with open(result_path) as f:
                            r = json.load(f)
                        LOGGER.info("[%d/%d] SKIP %s — AUC-PR=%.4f",
                                  idx, n_total, cfg_id, r.get("auc_pr"))
                        all_results.append(r)
                        continue

                    LOGGER.info("[%d/%d] %s", idx, n_total, cfg_id)
                    t0 = time.time()
                    try:
                        metrics, best_t = run_config(
                            X_train, hours_train, dows_train, rcs_train, nb_train,
                            X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask,
                            memory_len=ml, k=k, gamma=0.0, beta=0.001,
                            noise_std=base_noise, lr=base_lr,
                            out_dim=od, epochs=base_epochs,
                        )
                        result = {
                            "config_id": cfg_id,
                            "base_id": base.get("config_id"),
                            "memory_len": ml,
                            "k": k,
                            "gamma": 0.0,
                            "beta": 0.001,
                            "noise_std": base_noise,
                            "lr": base_lr,
                            "out_dim": od,
                            "epochs": base_epochs,
                            "auc_roc": metrics["auc_roc"],
                            "auc_pr": metrics["auc_pr"],
                            "f1": metrics["f1"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "fpr": metrics["fpr"],
                            "acc": metrics["acc"],
                            "best_threshold": best_t,
                            "score_normal_mean": metrics["score_normal_mean"],
                            "score_anomaly_mean": metrics["score_anomaly_mean"],
                            "separation_ratio": metrics["separation_ratio"],
                            "train_time_s": float(time.time() - t0),
                        }
                    except Exception as e:
                        LOGGER.error("  ERROR %s: %s", cfg_id, e)
                        import traceback; traceback.print_exc()
                        result = {
                            "config_id": cfg_id, "error": str(e),
                            "out_dim": od, "memory_len": ml, "k": k,
                            "auc_pr": 0.0, "train_time_s": float(time.time() - t0),
                        }

                    with open(result_path, "w") as f:
                        json.dump(result, f, indent=2)
                    all_results.append(result)
                    LOGGER.info("  -> AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f, sep=%.2fx [%.1fs]",
                               result.get("auc_pr"), result.get("auc_roc"),
                               result.get("f1"), result.get("separation_ratio"),
                               result.get("train_time_s"))

    all_results.sort(key=lambda x: x.get("auc_pr", 0), reverse=True)

    summary = {
        "experiment": "Stage 3 Architecture (out_dim, memory_len, k)",
        "n_configs": len(all_results),
        "results": all_results,
        "best": all_results[0] if all_results else {},
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("TOP 10 Stage 3 Results (sorted by AUC-PR):")
    LOGGER.info("=" * 80)
    LOGGER.info(f"{'Rank':<5} {'ConfigID':<50} {'AUC-PR':>8} {'AUC-ROC':>8} {'F1':>8} {'Time':>8}")
    LOGGER.info("-" * 80)
    for i, r in enumerate(all_results[:10], 1):
        LOGGER.info(f"{i:<5} {r['config_id']:<50} "
                   f"{r.get('auc_pr', 0):>8.4f} {r.get('auc_roc', 0):>8.4f} "
                   f"{r.get('f1', 0):>8.4f} {r.get('train_time_s', 0):>7.1f}s")
    LOGGER.info("")
    LOGGER.info("Best: %s — AUC-PR=%.4f", all_results[0]["config_id"], all_results[0].get("auc_pr"))


if __name__ == "__main__":
    main()
