#!/usr/bin/env python3
"""
Grid Search Stage 2 — Fine-tune training params.
Varies: noise_std, epochs, lr on top configs from Stage 1.

Usage:
    python upgrade_experiments/grid_search_s2.py [--skip] [--max-train N]
"""
from __future__ import annotations

import os, sys, json, time, argparse, logging
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"
sys.path.insert(0, r"c:\proj\ldt\HP_benchmark_v5")

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                   stream=sys.stderr)
LOGGER = logging.getLogger("grid_s2")

PYTHON = r"C:\Users\Administrator\Desktop\AI ComfyUI\system\python\python.exe"


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
               seed=42, cb_warmup=None):
    from benchmark_core import MemStreamPipeline

    if cb_warmup is None:
        cb_warmup = min(4096, memory_len * 4)

    pipeline = MemStreamPipeline(
        d=38, out_dim=out_dim,
        memory_len=memory_len, k=k, gamma=gamma, beta=beta,
        noise_std=noise_std, lr=lr, epochs=epochs,
        batch_size=1024, seed=seed,
        cb_warmup=cb_warmup, verbose=False,
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
    return metrics, best_t, pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", action="store_true", help="Skip existing configs")
    parser.add_argument("--max-train", type=int, default=500000)
    parser.add_argument("--max-val", type=int, default=500000)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    OUT_DIR = r"c:\proj\ldt\upgrade_experiments\grid_search_s2"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load Stage 1 results to get top configs
    s1_path = r"c:\proj\ldt\HP_benchmark_v5\results\grid_search\stage1\summary.json"
    with open(s1_path) as f:
        s1_data = json.load(f)

    # Sort by AUC-PR and take top
    s1_results = s1_data.get("results", [])
    s1_results.sort(key=lambda x: x.get("auc_pr", 0), reverse=True)
    top_configs = s1_results[:args.top]
    LOGGER.info("Stage 1 top %d configs:", len(top_configs))
    for c in top_configs:
        LOGGER.info("  %s: AUC-PR=%.4f, AUC-ROC=%.4f",
                   c.get("config_id"), c.get("auc_pr"), c.get("auc_roc"))

    # Load data
    from benchmark_core import extract_features_from_parquet
    LOGGER.info("Loading training data (max=%d)...", args.max_train)
    X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
        r"c:\proj\ldt\HP_benchmark_v5\data\train_clean.parquet", max_rows=args.max_train
    )
    LOGGER.info("Loading validation data (max=%d)...", args.max_val)
    X_val, hours_val, dows_val, rcs_val, nb_val = extract_features_from_parquet(
        r"c:\proj\ldt\HP_benchmark_v5\data\valid_polluted.parquet", max_rows=args.max_val
    )
    gt_mask = np.load(r"c:\proj\ldt\HP_benchmark_v5\data\valid\ground_truth_mask.npy")
    gt_mask = gt_mask[-len(X_val):]
    LOGGER.info("Train: %d, Val: %d (GT: %d anomalies, %.2f%%)",
               len(X_train), len(X_val), gt_mask.sum(), gt_mask.mean() * 100)

    # Fine-tune grid
    fine_grid = {
        "noise_std": [0.0005, 0.001, 0.005],   # 3 values
        "epochs":    [2000, 5000, 10000],         # 3 values
        "lr":        [0.005, 0.01],               # 2 values
    }
    n_total = len(top_configs) * len(fine_grid["noise_std"]) * len(fine_grid["epochs"]) * len(fine_grid["lr"])
    LOGGER.info("Fine grid: %d noise x %d epochs x %d lr = %d configs per top config",
               len(fine_grid["noise_std"]), len(fine_grid["epochs"]),
               len(fine_grid["lr"]), len(fine_grid["noise_std"]) * len(fine_grid["epochs"]) * len(fine_grid["lr"]))
    LOGGER.info("Total configs: %d", n_total)

    all_results = []
    idx = 0
    for base in top_configs:
        for noise in fine_grid["noise_std"]:
            for ep in fine_grid["epochs"]:
                for lr_v in fine_grid["lr"]:
                    idx += 1
                    cfg_id = (f"M{base['memory_len']}_k{base['k']}_"
                             f"n{noise}_e{ep}_lr{int(lr_v*1000)}")

                    result_path = os.path.join(OUT_DIR, f"{cfg_id}.json")
                    if args.skip and os.path.exists(result_path):
                        with open(result_path) as f:
                            r = json.load(f)
                        LOGGER.info("[%d/%d] SKIP %s — AUC-PR=%.4f", idx, n_total, cfg_id, r.get("auc_pr"))
                        all_results.append(r)
                        continue

                    LOGGER.info("[%d/%d] %s", idx, n_total, cfg_id)
                    t0 = time.time()
                    try:
                        metrics, best_t, pipeline = run_config(
                            X_train, hours_train, dows_train, rcs_train, nb_train,
                            X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask,
                            memory_len=base["memory_len"], k=base["k"],
                            gamma=base["gamma"], beta=base["beta"],
                            noise_std=noise, lr=lr_v,
                            out_dim=38,  # keep d=38 (current best practice)
                            epochs=ep,
                        )
                        result = {
                            "config_id": cfg_id,
                            "base_id": base.get("config_id"),
                            "memory_len": base["memory_len"],
                            "k": base["k"],
                            "gamma": base["gamma"],
                            "beta": base["beta"],
                            "noise_std": noise,
                            "lr": lr_v,
                            "out_dim": 38,
                            "epochs": ep,
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
                            "memory_len": base["memory_len"], "k": base["k"],
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
        "experiment": "Stage 2 Fine-tune (noise_std, epochs, lr)",
        "n_configs": len(all_results),
        "results": all_results,
        "best": all_results[0] if all_results else {},
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("TOP 10 Stage 2 Results (sorted by AUC-PR):")
    LOGGER.info("=" * 80)
    LOGGER.info(f"{'Rank':<5} {'ConfigID':<45} {'AUC-PR':>8} {'AUC-ROC':>8} {'F1':>8} {'Time':>8}")
    LOGGER.info("-" * 80)
    for i, r in enumerate(all_results[:10], 1):
        LOGGER.info(f"{i:<5} {r['config_id']:<45} "
                   f"{r.get('auc_pr', 0):>8.4f} {r.get('auc_roc', 0):>8.4f} "
                   f"{r.get('f1', 0):>8.4f} {r.get('train_time_s', 0):>7.1f}s")
    LOGGER.info("")
    LOGGER.info("Best config: %s", all_results[0]["config_id"])
    LOGGER.info("Best AUC-PR: %.4f", all_results[0].get("auc_pr"))
    LOGGER.info("Saved to: %s", OUT_DIR)


if __name__ == "__main__":
    main()
