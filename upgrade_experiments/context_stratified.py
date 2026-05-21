#!/usr/bin/env python3
"""
Context-Stratified Evaluation + ContextBeta Ablation.

Two parts:
  1. Per-context-cell and per-neighborhood breakdown of MemStream performance
  2. ContextBeta ON vs OFF ablation (ablation A in paper)

Usage:
    python upgrade_experiments/context_stratified.py [--max-train N] [--max-val N]
"""
from __future__ import annotations

import os, sys, json, time, argparse, logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"
sys.path.insert(0, r"c:\proj\ldt\HP_benchmark_v5")

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                   stream=sys.stderr)
LOGGER = logging.getLogger("context_eval")

OUT_DIR = r"c:\proj\ldt\upgrade_experiments\context_stratified"
os.makedirs(OUT_DIR, exist_ok=True)


def compute_context_cell_id(hour, dow, ratecode):
    is_special = 1 if ratecode > 1.0 else 0
    is_night = 1 if (hour >= 20 or hour < 6) else 0
    is_weekend = 1 if dow >= 5 else 0
    return (is_special << 2) | (is_night << 1) | is_weekend


CELL_NAMES = {
    0: "weekday_night",
    1: "weekday_day",
    2: "weekend_night",
    3: "weekend_day",
    4: "special_night",
    5: "special_day",
    6: "special_weekend_night",
    7: "special_weekend_day",
}

NB_NAMES = {
    0: "Manhattan", 1: "Brooklyn", 2: "Queens_lower",
    3: "Queens_upper", 4: "Bronx", 5: "StatenIsland",
    6: "EWR", 7: "JFK", 8: "NALP", 9: "Unknown",
}


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


def eval_stratified(scores, hours, dows, rcs, nb_ids, gt_mask, title=""):
    """Per-cell, per-neighborhood breakdown."""
    results = {}
    n = len(scores)

    # Global
    best_t, _ = find_best_threshold(scores, gt_mask)
    metrics = evaluate_scores(scores, gt_mask, best_t)
    results["GLOBAL"] = {
        "n": n, "n_anomaly": int(gt_mask.sum()),
        "auc_roc": metrics["auc_roc"], "auc_pr": metrics["auc_pr"],
        "f1": metrics["f1"], "precision": metrics["precision"], "recall": metrics["recall"],
        "best_threshold": best_t,
        "score_normal_mean": metrics["score_normal_mean"],
        "score_anomaly_mean": metrics["score_anomaly_mean"],
        "separation_ratio": metrics["separation_ratio"],
    }

    # Per context cell
    for ctx in range(8):
        mask = np.array([
            compute_context_cell_id(int(hours[i]), int(dows[i]), float(rcs[i])) == ctx
            for i in range(n)
        ])
        if mask.sum() < 20:
            continue
        sub_scores = scores[mask]
        sub_gt = gt_mask[mask]
        best_t_c, _ = find_best_threshold(sub_scores, sub_gt)
        m = evaluate_scores(sub_scores, sub_gt, best_t_c)
        results[f"CELL_{ctx}"] = {
            "name": CELL_NAMES.get(ctx, f"cell_{ctx}"),
            "n": int(mask.sum()), "n_anomaly": int(sub_gt.sum()),
            "auc_roc": m["auc_roc"], "auc_pr": m["auc_pr"],
            "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
            "best_threshold": best_t_c,
            "score_normal_mean": m["score_normal_mean"],
            "score_anomaly_mean": m["score_anomaly_mean"],
            "separation_ratio": m["separation_ratio"],
        }

    # Per neighborhood
    for nb in range(10):
        mask = nb_ids == nb
        if mask.sum() < 20:
            continue
        sub_scores = scores[mask]
        sub_gt = gt_mask[mask]
        best_t_c, _ = find_best_threshold(sub_scores, sub_gt)
        m = evaluate_scores(sub_scores, sub_gt, best_t_c)
        results[f"NB_{nb}"] = {
            "name": NB_NAMES.get(nb, f"nb_{nb}"),
            "n": int(mask.sum()), "n_anomaly": int(sub_gt.sum()),
            "auc_roc": m["auc_roc"], "auc_pr": m["auc_pr"],
            "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
            "best_threshold": best_t_c,
            "score_normal_mean": m["score_normal_mean"],
            "score_anomaly_mean": m["score_anomaly_mean"],
            "separation_ratio": m["separation_ratio"],
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=500000)
    parser.add_argument("--max-val", type=int, default=500000)
    parser.add_argument("--epochs", type=int, default=5000)
    args = parser.parse_args()

    from benchmark_core import MemStreamPipeline, extract_features_from_parquet

    LOGGER.info("Loading data...")
    X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
        r"c:\proj\ldt\HP_benchmark_v5\data\train_clean.parquet", max_rows=args.max_train
    )
    X_val, hours_val, dows_val, rcs_val, nb_val = extract_features_from_parquet(
        r"c:\proj\ldt\HP_benchmark_v5\data\valid_polluted.parquet", max_rows=args.max_val
    )
    gt_mask = np.load(r"c:\proj\ldt\HP_benchmark_v5\data\valid\ground_truth_mask.npy")
    gt_mask = gt_mask[-len(X_val):]
    LOGGER.info("Train: %d, Val: %d (GT: %d, %.2f%%)",
               len(X_train), len(X_val), gt_mask.sum(), gt_mask.mean() * 100)

    # Best config from Stage 1
    with open(r"c:\proj\ldt\HP_benchmark_v5\results\grid_search\stage1\summary.json") as f:
        s1 = json.load(f)
    best = s1["results"][0]
    LOGGER.info("Best Stage 1 config: %s — AUC-PR=%.4f",
               best.get("config_id"), best.get("auc_pr"))

    memory_len = best["memory_len"]
    k = best["k"]
    noise_std = best.get("noise_std", 0.001)
    lr = best.get("lr", 0.01)

    # ===================================================================
    # Part 1: ContextBeta ON (standard)
    # ===================================================================
    LOGGER.info("")
    LOGGER.info("=== Part 1: MemStream + ContextBeta ===")
    t0 = time.time()
    pipeline_cb = MemStreamPipeline(
        d=38, out_dim=38, memory_len=memory_len, k=k,
        gamma=0.0, beta=0.001,
        noise_std=noise_std, lr=lr, epochs=args.epochs,
        batch_size=1024, seed=42,
        cb_warmup=min(4096, memory_len * 4), verbose=False,
    )
    pipeline_cb.train(
        X_train, hours_train, dows_train, rcs_train, nb_train,
        X_warmup=X_train[:memory_len],
        hours_warmup=hours_train[:memory_len],
        dows_warmup=dows_train[:memory_len],
        rcs_warmup=rcs_train[:memory_len],
        nb_warmup=nb_train[:memory_len],
    )
    scores_cb, _ = pipeline_cb.score_stream(
        X_val, hours_val, dows_val, rcs_val, nb_val,
        gt_mask=gt_mask, update_memory=True,
    )
    best_t_cb, _ = find_best_threshold(scores_cb, gt_mask)
    m_cb = evaluate_scores(scores_cb, gt_mask, best_t_cb)
    LOGGER.info("  Global: AUC-PR=%.4f, F1=%.4f, Recall=%.4f [%.1fs]",
               m_cb["auc_pr"], m_cb["f1"], m_cb["recall"], time.time() - t0)

    results_cb = eval_stratified(scores_cb, hours_val, dows_val, rcs_val, nb_val, gt_mask,
                                  title="MemStream + ContextBeta")
    LOGGER.info("  Per-cell breakdown computed (%d segments)", len(results_cb))

    # ===================================================================
    # Part 2: ContextBeta OFF (global beta=0.001 for all)
    # ===================================================================
    LOGGER.info("")
    LOGGER.info("=== Part 2: MemStream + Global Beta (NO ContextBeta) ===")
    t0 = time.time()

    # Use ContextBeta with same beta for all cells (no differentiation)
    from benchmark_core import ContextBeta
    cb_global = ContextBeta(db=0.001, pct=95.0)

    # We need to run score_stream but disable ContextBeta per-cell thresholds
    # Simulate: run with a dummy ContextBeta that returns same beta everywhere
    class GlobalBetaOnly(ContextBeta):
        def beta_for(self, nb, ctx):
            return self.beta  # ignores nb and ctx

    pipeline_gb = MemStreamPipeline(
        d=38, out_dim=38, memory_len=memory_len, k=k,
        gamma=0.0, beta=0.001,
        noise_std=noise_std, lr=lr, epochs=args.epochs,
        batch_size=1024, seed=42,
        cb_warmup=min(4096, memory_len * 4), verbose=False,
    )
    pipeline_gb.train(
        X_train, hours_train, dows_train, rcs_train, nb_train,
        X_warmup=X_train[:memory_len],
        hours_warmup=hours_train[:memory_len],
        dows_warmup=dows_train[:memory_len],
        rcs_warmup=rcs_train[:memory_len],
        nb_warmup=nb_train[:memory_len],
    )
    # Force global beta for all cells
    cb_global.fit()
    pipeline_gb.cb.T[:, :] = 0.001  # same threshold for all
    pipeline_gb.cb.fitted = True

    scores_gb, _ = pipeline_gb.score_stream(
        X_val, hours_val, dows_val, rcs_val, nb_val,
        gt_mask=gt_mask, update_memory=True,
    )
    best_t_gb, _ = find_best_threshold(scores_gb, gt_mask)
    m_gb = evaluate_scores(scores_gb, gt_mask, best_t_gb)
    LOGGER.info("  Global: AUC-PR=%.4f, F1=%.4f, Recall=%.4f [%.1fs]",
               m_gb["auc_pr"], m_gb["f1"], m_gb["recall"], time.time() - t0)

    results_gb = eval_stratified(scores_gb, hours_val, dows_val, rcs_val, nb_val, gt_mask,
                                  title="MemStream + Global Beta")

    # ===================================================================
    # Part 3: Compare ContextBeta vs Global per segment
    # ===================================================================
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("ContextBeta Ablation Results:")
    LOGGER.info("=" * 80)
    LOGGER.info(f"{'Segment':<25} {'CB_AUC-PR':>10} {'GB_AUC-PR':>10} {'CB_F1':>8} {'GB_F1':>8} "
                f"{'CB_Recall':>10} {'GB_Recall':>10} {'CB_sep':>8} {'GB_sep':>8} {'n':>8}")
    LOGGER.info("-" * 80)

    comparison = {}
    all_keys = sorted(set(results_cb.keys()) | set(results_gb.keys()))
    for key in all_keys:
        if key.startswith("CELL_") or key.startswith("NB_") or key == "GLOBAL":
            r1 = results_cb.get(key, {})
            r2 = results_gb.get(key, {})
            n_seg = r1.get("n", r2.get("n", 0))
            n_anom = r1.get("n_anomaly", r2.get("n_anomaly", 0))
            seg_name = r1.get("name", r2.get("name", key))
            cb_ap = r1.get("auc_pr", 0)
            gb_ap = r2.get("auc_pr", 0)
            cb_f1 = r1.get("f1", 0)
            gb_f1 = r2.get("f1", 0)
            cb_rec = r1.get("recall", 0)
            gb_rec = r2.get("recall", 0)
            cb_sep = r1.get("separation_ratio", 0)
            gb_sep = r2.get("separation_ratio", 0)
            comparison[key] = {
                "name": seg_name, "n": n_seg, "n_anomaly": n_anom,
                "cb_auc_pr": cb_ap, "gb_auc_pr": gb_ap,
                "cb_f1": cb_f1, "gb_f1": gb_f1,
                "cb_recall": cb_rec, "gb_recall": gb_rec,
                "cb_separation": cb_sep, "gb_separation": gb_sep,
                "cb_results": r1, "gb_results": r2,
            }
            winner = "CB" if cb_ap > gb_ap else ("GB" if gb_ap > cb_ap else "TIE")
            LOGGER.info(f"{seg_name:<25} {cb_ap:>10.4f} {gb_ap:>10.4f} "
                       f"{cb_f1:>8.4f} {gb_f1:>8.4f} "
                       f"{cb_rec:>10.4f} {gb_rec:>10.4f} "
                       f"{cb_sep:>8.2f} {gb_sep:>8.2f} {n_seg:>8}")
            if winner != "TIE":
                LOGGER.info(f"  -> Winner: {winner} (+{abs(cb_ap-gb_ap):.4f})")

    # Save results
    output = {
        "experiment": "ContextBeta Ablation",
        "best_config": best,
        "epochs": args.epochs,
        "global_with_cb": results_cb,
        "global_no_cb": results_gb,
        "comparison": comparison,
        "summary": {
            "cb_global_auc_pr": m_cb["auc_pr"],
            "cb_global_f1": m_cb["f1"],
            "cb_global_recall": m_cb["recall"],
            "gb_global_auc_pr": m_gb["auc_pr"],
            "gb_global_f1": m_gb["f1"],
            "gb_global_recall": m_gb["recall"],
        }
    }
    with open(os.path.join(OUT_DIR, "context_ablation_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("SUMMARY:")
    LOGGER.info("=" * 60)
    LOGGER.info(f"  ContextBeta ON:  AUC-PR={m_cb['auc_pr']:.4f}, F1={m_cb['f1']:.4f}, Recall={m_cb['recall']:.4f}")
    LOGGER.info(f"  Global Beta:     AUC-PR={m_gb['auc_pr']:.4f}, F1={m_gb['f1']:.4f}, Recall={m_gb['recall']:.4f}")
    LOGGER.info(f"  Improvement:      AUC-PR={m_cb['auc_pr']-m_gb['auc_pr']:+.4f}, F1={m_cb['f1']-m_gb['f1']:+.4f}")
    LOGGER.info("")
    cb_wins = sum(1 for v in comparison.values() if v["cb_auc_pr"] > v["gb_auc_pr"])
    gb_wins = sum(1 for v in comparison.values() if v["gb_auc_pr"] > v["cb_auc_pr"])
    LOGGER.info(f"  Segment wins: ContextBeta={cb_wins}, Global={gb_wins}")
    LOGGER.info(f"  Results saved to: {os.path.join(OUT_DIR, 'context_ablation_results.json')}")


if __name__ == "__main__":
    main()
