#!/usr/bin/env python3
"""
Full Data Training — Train on FULL dataset (no MAX_TRAIN limit).

Compare:
  - Train on 200K (current)
  - Train on 500K
  - Train on full dataset (no limit)

Also compare on full test set (no MAX_VAL limit).

Usage:
    python upgrade_experiments/full_data.py [--epochs N]
"""
from __future__ import annotations

import os, sys, json, time, argparse, logging
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

from _config import (TRAIN_PATH, VALID_PATH, GT_MASK_PATH, GRID_SEARCH_STAGE1_DIR, RESULTS_DIR)
from benchmark_core import extract_features_from_parquet, MemStreamPipeline

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                   stream=sys.stderr)
LOGGER = logging.getLogger("full_data")

OUT_DIR = RESULTS_DIR / "full_data"
os.makedirs(OUT_DIR, exist_ok=True)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--max-val", type=int, default=0)  # 0 = full
    args = parser.parse_args()

    # Best config from Stage 1
    with open(GRID_SEARCH_STAGE1_DIR / "summary.json") as f:
        s1 = json.load(f)
    best = s1["results"][0]
    memory_len = best["memory_len"]
    k = best["k"]
    noise_std = best.get("noise_std", 0.001)
    lr = best.get("lr", 0.01)
    epochs = args.epochs

    LOGGER.info("Best Stage 1 config: memory=%d, k=%d, noise=%.4f, lr=%.4f",
               memory_len, k, noise_std, lr)

    # Training sizes to test
    train_sizes = [
        (200000, "200K"),
        (500000, "500K"),
        (None,    "FULL"),
    ]
    # Val sizes
    val_sizes = [
        (100000, "100K"),
        (300000, "300K"),
        (None,   "FULL"),
    ]

    all_results = {}

    for train_size, train_label in train_sizes:
        LOGGER.info("")
        LOGGER.info("=" * 60)
        LOGGER.info("TRAINING SIZE: %s", train_label)
        LOGGER.info("=" * 60)

        LOGGER.info("Loading training data (max=%s)...", train_label)
        X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
            str(TRAIN_PATH),
            max_rows=train_size
        )
        LOGGER.info("  Loaded: %d rows", len(X_train))

        for val_size, val_label in val_sizes:
            LOGGER.info("")
            LOGGER.info("  Validation size: %s", val_label)
            t_val = time.time()
            X_val, hours_val, dows_val, rcs_val, nb_val = extract_features_from_parquet(
                str(VALID_PATH),
                max_rows=val_size
            )
            gt_mask = np.load(str(GT_MASK_PATH))
            gt_mask = gt_mask[-len(X_val):]
            LOGGER.info("    Loaded: %d rows, GT: %d anomalies (%.2f%%) [%.1fs]",
                       len(X_val), gt_mask.sum(), gt_mask.mean() * 100, time.time() - t_val)

            key = f"train_{train_label}_val_{val_label}"
            t0 = time.time()

            pipeline = MemStreamPipeline(
                d=38, out_dim=38, memory_len=memory_len, k=k,
                gamma=0.0, beta=0.001,
                noise_std=noise_std, lr=lr, epochs=epochs,
                batch_size=1024, seed=42,
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

            train_time = time.time() - t0
            LOGGER.info("    Training: %.1fs", train_time)

            t0 = time.time()
            adj_scores, _ = pipeline.score_stream(
                X_val, hours_val, dows_val, rcs_val, nb_val,
                gt_mask=gt_mask, update_memory=True,
            )
            score_time = time.time() - t0

            best_t, _ = find_best_threshold(adj_scores, gt_mask)
            metrics = evaluate_scores(adj_scores, gt_mask, best_t)

            LOGGER.info("    Scoring: %.1fs", score_time)
            LOGGER.info("    AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f, Recall=%.4f",
                       metrics["auc_pr"], metrics["auc_roc"], metrics["f1"], metrics["recall"])

            result = {
                "train_size": train_size or "FULL",
                "train_label": train_label,
                "val_size": val_size or "FULL",
                "val_label": val_label,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_anomaly": int(gt_mask.sum()),
                "anomaly_rate": float(gt_mask.mean()),
                "memory_len": memory_len, "k": k, "noise_std": noise_std,
                "lr": lr, "epochs": epochs,
                "train_time_s": train_time,
                "score_time_s": score_time,
                "auc_roc": metrics["auc_roc"],
                "auc_pr": metrics["auc_pr"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "separation_ratio": metrics["separation_ratio"],
                "best_threshold": best_t,
            }
            all_results[key] = result

            with open(os.path.join(OUT_DIR, f"result_{key}.json"), "w") as f:
                json.dump(result, f, indent=2)

    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 90)
    LOGGER.info("FULL DATA TRAINING SUMMARY:")
    LOGGER.info("=" * 90)
    LOGGER.info(f"{'Config':<40} {'n_train':>8} {'n_val':>8} {'n_anom':>7} "
                f"{'AUC-PR':>8} {'AUC-ROC':>8} {'F1':>8} {'Recall':>8} {'Train_s':>8}")
    LOGGER.info("-" * 90)
    for key, r in sorted(all_results.items()):
        LOGGER.info(f"{key:<40} {r['n_train']:>8} {r['n_val']:>8} "
                   f"{r['n_anomaly']:>7} "
                   f"{r['auc_pr']:>8.4f} {r['auc_roc']:>8.4f} "
                   f"{r['f1']:>8.4f} {r['recall']:>8.4f} "
                   f"{r['train_time_s']:>8.1f}")

    # Find best
    best_key = max(all_results, key=lambda k: all_results[k]["auc_pr"])
    LOGGER.info("")
    LOGGER.info("BEST: %s — AUC-PR=%.4f", best_key, all_results[best_key]["auc_pr"])

    with open(os.path.join(OUT_DIR, "full_data_results.json"), "w") as f:
        json.dump({"experiment": "Full Data Training", "results": all_results,
                   "best_key": best_key, "best": all_results[best_key],
                   "best_config": best}, f, indent=2)

    LOGGER.info("Results saved to: %s", OUT_DIR)


if __name__ == "__main__":
    main()
