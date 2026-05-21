#!/usr/bin/env python3
"""
Concept Drift Injection Experiment.

Injects synthetic anomalies at specific time windows in the test set to create
"concept drift" scenarios where the normal distribution shifts. Then compares:
  1. MemStream with streaming memory updates (adapts to drift)
  2. MemStream WITHOUT streaming updates (static memory, no adaptation)
  3. IsolationForest (offline, no adaptation)
  4. Inc. PCA (incremental, no adaptation)

Goal: Prove that streaming adaptation helps under concept drift.

Synthetic drift types:
  - Type A: Fare amounts shift +50% (price surge)
  - Type B: Trip distances shift +100% (longer routes)
  - Type C: Speed drops by 50% (traffic jam)
  - Type D: All features drift simultaneously (multi-dimensional shift)

Usage:
    python upgrade_experiments/concept_drift.py [--max-train N] [--epochs N]
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
LOGGER = logging.getLogger("concept_drift")

OUT_DIR = r"c:\proj\ldt\upgrade_experiments\concept_drift"
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


def inject_drift(X_clean, hours, dows, rcs, nb_ids,
                  drift_type="A", drift_start_pct=0.5, drift_end_pct=0.75,
                  intensity=1.0):
    """
    Inject synthetic concept drift by modifying clean test data.

    Returns: (X_drifted, gt_mask_driven)
    gt_mask = 1 for injected drift points (should be detected as anomalous vs clean baseline)
    """
    X = X_clean.copy()
    n = len(X)
    start_idx = int(n * drift_start_pct)
    end_idx = int(n * drift_end_pct)
    drift_mask = np.zeros(n, dtype=bool)
    drift_mask[start_idx:end_idx] = True

    # Feature indices (from _build_features):
    # 0: log_dist, 1: log_dur, 2: log_fare, 3: log_total, 4: log_speed
    # 7: fare_per_mi, 8: fare_per_min, 9: speed/15
    # 31: fare_per_mi_norm, 32: fare_per_min_norm, 33: speed_norm

    if drift_type == "A":
        # Type A: Price surge — fare amounts +50%
        X[drift_mask, 2] += intensity * 0.5      # log_fare
        X[drift_mask, 3] += intensity * 0.5      # log_total
        X[drift_mask, 7] += intensity * 3.0      # fare/mi
        X[drift_mask, 8] += intensity * 1.0      # fare/min
        X[drift_mask, 31] += intensity * 2.0    # fare/mi norm

    elif drift_type == "B":
        # Type B: Longer trips — distances +100%
        X[drift_mask, 0] += intensity * 1.0     # log_dist
        X[drift_mask, 1] += intensity * 0.5     # log_dur
        X[drift_mask, 13] += intensity * 1.0     # log_dist redundant
        X[drift_mask, 30] += intensity * 0.5    # log_dist norm

    elif drift_type == "C":
        # Type C: Traffic jam — speed drops by 50%
        X[drift_mask, 4] -= intensity * 0.7     # log_speed
        X[drift_mask, 9] -= intensity * 2.0    # speed/15
        X[drift_mask, 33] -= intensity * 2.0   # speed_norm
        X[drift_mask, 37] -= intensity * 2.0   # speed_norm redundant

    elif drift_type == "D":
        # Type D: Multi-dimensional shift (all features)
        X[drift_mask, 0] += intensity * 0.8    # log_dist
        X[drift_mask, 2] += intensity * 0.5    # log_fare
        X[drift_mask, 4] -= intensity * 0.5    # log_speed
        X[drift_mask, 7] += intensity * 2.0    # fare/mi
        X[drift_mask, 8] += intensity * 0.5    # fare/min
        X[drift_mask, 9] -= intensity * 1.5    # speed/15
        X[drift_mask, 31] += intensity * 1.5    # fare/mi norm
        X[drift_mask, 33] -= intensity * 1.5   # speed_norm

    return X, drift_mask


def window_eval(scores, gt_mask, n_windows=5):
    """Evaluate per time window."""
    n = len(scores)
    window_size = n // n_windows
    results = []
    for w in range(n_windows):
        s = w * window_size
        e = (w + 1) * window_size if w < n_windows - 1 else n
        sub_scores = scores[s:e]
        sub_gt = gt_mask[s:e]
        best_t, _ = find_best_threshold(sub_scores, sub_gt)
        m = evaluate_scores(sub_scores, sub_gt, best_t)
        results.append({
            "window": w + 1,
            "start": s, "end": e,
            "auc_pr": m["auc_pr"], "auc_roc": m["auc_roc"],
            "f1": m["f1"], "recall": m["recall"], "precision": m["precision"],
            "score_normal_mean": m["score_normal_mean"],
            "score_anomaly_mean": m["score_anomaly_mean"],
            "separation_ratio": m["separation_ratio"],
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=500000)
    parser.add_argument("--max-val", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=5000)
    args = parser.parse_args()

    from benchmark_core import MemStreamPipeline, extract_features_from_parquet
    from sklearn.ensemble import IsolationForest
    from sklearn.decomposition import IncrementalPCA

    LOGGER.info("Loading data...")
    X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
        r"c:\proj\ldt\HP_benchmark_v5\data\train_clean.parquet", max_rows=args.max_train
    )
    X_clean, hours_val, dows_val, rcs_val, nb_val = extract_features_from_parquet(
        r"c:\proj\ldt\HP_benchmark_v5\data\valid_polluted.parquet",
        max_rows=min(args.max_val, 100000)
    )
    gt_original = np.load(r"c:\proj\ldt\HP_benchmark_v5\data\valid\ground_truth_mask.npy")
    gt_original = gt_original[-len(X_clean):]

    # Drift types
    drift_types = ["A", "B", "C", "D"]
    drift_names = {
        "A": "Price Surge (+50% fare)",
        "B": "Longer Trips (+100% dist)",
        "C": "Traffic Jam (-50% speed)",
        "D": "Multi-dimensional Shift",
    }
    intensity_levels = [0.5, 1.0, 2.0]

    # Best config from Stage 1
    with open(r"c:\proj\ldt\HP_benchmark_v5\results\grid_search\stage1\summary.json") as f:
        s1 = json.load(f)
    best = s1["results"][0]
    memory_len = best["memory_len"]
    k = best["k"]
    noise_std = best.get("noise_std", 0.001)
    lr = best.get("lr", 0.01)

    all_results = {}

    for drift_type in drift_types:
        LOGGER.info("")
        LOGGER.info("=" * 60)
        LOGGER.info("DRIFT TYPE %s: %s", drift_type, drift_names[drift_type])
        LOGGER.info("=" * 60)

        X_drifted, drift_mask = inject_drift(
            X_clean, hours_val, dows_val, rcs_val, nb_val,
            drift_type=drift_type, drift_start_pct=0.5, drift_end_pct=0.75,
            intensity=1.0
        )

        # GT for this drift: original anomalies + drift injection
        gt_drift = gt_original.copy()
        gt_drift[drift_mask] = 1  # mark drift as anomaly

        LOGGER.info("  Drift region: indices %d-%d (%d points, %.1f%% anomaly rate)",
                   drift_mask.argmax(), drift_mask.sum(), drift_mask.sum(), drift_mask.mean() * 100)
        LOGGER.info("  Combined GT: %d anomalies (%.2f%%)",
                   gt_drift.sum(), gt_drift.mean() * 100)

        drift_results = {}

        # --- Method 1: MemStream with streaming updates ---
        LOGGER.info("")
        LOGGER.info("  [MemStream + Streaming Update]")
        t0 = time.time()
        ms_stream = MemStreamPipeline(
            d=38, out_dim=38, memory_len=memory_len, k=k,
            gamma=0.0, beta=0.001,
            noise_std=noise_std, lr=lr, epochs=args.epochs,
            batch_size=1024, seed=42,
            cb_warmup=min(4096, memory_len * 4), verbose=False,
        )
        ms_stream.train(
            X_train, hours_train, dows_train, rcs_train, nb_train,
            X_warmup=X_train[:memory_len],
            hours_warmup=hours_train[:memory_len],
            dows_warmup=dows_train[:memory_len],
            rcs_warmup=rcs_train[:memory_len],
            nb_warmup=nb_train[:memory_len],
        )
        scores_stream, _ = ms_stream.score_stream(
            X_drifted, hours_val, dows_val, rcs_val, nb_val,
            gt_mask=gt_drift, update_memory=True,
        )
        best_t, _ = find_best_threshold(scores_stream, gt_drift)
        m_stream = evaluate_scores(scores_stream, gt_drift, best_t)
        LOGGER.info("    Global: AUC-PR=%.4f, F1=%.4f, Recall=%.4f [%.1fs]",
                   m_stream["auc_pr"], m_stream["f1"], m_stream["recall"], time.time() - t0)

        # --- Method 2: MemStream WITHOUT streaming updates (static memory) ---
        LOGGER.info("  [MemStream + Static Memory (NO updates)]")
        t0 = time.time()
        ms_static = MemStreamPipeline(
            d=38, out_dim=38, memory_len=memory_len, k=k,
            gamma=0.0, beta=0.001,
            noise_std=noise_std, lr=lr, epochs=args.epochs,
            batch_size=1024, seed=42,
            cb_warmup=min(4096, memory_len * 4), verbose=False,
        )
        ms_static.train(
            X_train, hours_train, dows_train, rcs_train, nb_train,
            X_warmup=X_train[:memory_len],
            hours_warmup=hours_train[:memory_len],
            dows_warmup=dows_train[:memory_len],
            rcs_warmup=rcs_train[:memory_len],
            nb_warmup=nb_train[:memory_len],
        )
        scores_static, _ = ms_static.score_stream(
            X_drifted, hours_val, dows_val, rcs_val, nb_val,
            gt_mask=gt_drift, update_memory=False,
        )
        best_t_s, _ = find_best_threshold(scores_static, gt_drift)
        m_static = evaluate_scores(scores_static, gt_drift, best_t_s)
        LOGGER.info("    Global: AUC-PR=%.4f, F1=%.4f, Recall=%.4f [%.1fs]",
                   m_static["auc_pr"], m_static["f1"], m_static["recall"], time.time() - t0)

        # --- Method 3: IsolationForest (offline, no streaming) ---
        LOGGER.info("  [IsolationForest (offline)]")
        t0 = time.time()
        if_clean = IsolationForest(n_estimators=200, max_samples=256,
                                   contamination=0.05, random_state=42, n_jobs=-1)
        if_clean.fit(X_train)
        scores_if = -if_clean.score_samples(X_drifted)
        best_t_if, _ = find_best_threshold(scores_if, gt_drift)
        m_if = evaluate_scores(scores_if, gt_drift, best_t_if)
        LOGGER.info("    Global: AUC-PR=%.4f, F1=%.4f, Recall=%.4f [%.1fs]",
                   m_if["auc_pr"], m_if["f1"], m_if["recall"], time.time() - t0)

        # --- Method 4: IncrementalPCA (incremental, no adaptation) ---
        LOGGER.info("  [Inc. PCA (incremental)]")
        t0 = time.time()
        ipca = IncrementalPCA(n_components=15)
        ipca.partial_fit(X_train[:100000])
        recon = ipca.inverse_transform(ipca.transform(X_drifted))
        scores_ipca = np.linalg.norm(X_drifted - recon, axis=1)
        best_t_ipca, _ = find_best_threshold(scores_ipca, gt_drift)
        m_ipca = evaluate_scores(scores_ipca, gt_drift, best_t_ipca)
        LOGGER.info("    Global: AUC-PR=%.4f, F1=%.4f, Recall=%.4f [%.1fs]",
                   m_ipca["auc_pr"], m_ipca["f1"], m_ipca["recall"], time.time() - t0)

        # Window analysis (MemStream streaming vs static)
        windows_stream = window_eval(scores_stream, gt_drift, n_windows=5)
        windows_static = window_eval(scores_static, gt_drift, n_windows=5)

        # Per-window breakdown
        LOGGER.info("")
        LOGGER.info("  Per-Window AUC-PR (Streaming vs Static):")
        LOGGER.info(f"  {'Window':<8} {'Stream_PR':>10} {'Static_PR':>10} {'Diff':>8} {'Drift?':>8}")
        LOGGER.info("  " + "-" * 44)
        for w in range(5):
            is_drift = "YES" if 2 <= w <= 3 else "no"
            diff = windows_stream[w]["auc_pr"] - windows_static[w]["auc_pr"]
            LOGGER.info(f"  W{w+1:<6} {windows_stream[w]['auc_pr']:>10.4f} "
                       f"{windows_static[w]['auc_pr']:>10.4f} {diff:>+8.4f} {is_drift:>8}")

        drift_results = {
            "drift_type": drift_type,
            "drift_name": drift_names[drift_type],
            "intensity": 1.0,
            "drift_start_pct": 0.5,
            "drift_end_pct": 0.75,
            "n_drift_points": int(drift_mask.sum()),
            "n_total": len(gt_drift),
            "memstream_streaming": {
                "auc_pr": m_stream["auc_pr"], "auc_roc": m_stream["auc_roc"],
                "f1": m_stream["f1"], "precision": m_stream["precision"],
                "recall": m_stream["recall"], "separation_ratio": m_stream["separation_ratio"],
            },
            "memstream_static": {
                "auc_pr": m_static["auc_pr"], "auc_roc": m_static["auc_roc"],
                "f1": m_static["f1"], "precision": m_static["precision"],
                "recall": m_static["recall"], "separation_ratio": m_static["separation_ratio"],
            },
            "isolation_forest": {
                "auc_pr": m_if["auc_pr"], "auc_roc": m_if["auc_roc"],
                "f1": m_if["f1"], "precision": m_if["precision"],
                "recall": m_if["recall"], "separation_ratio": m_if["separation_ratio"],
            },
            "inc_pca": {
                "auc_pr": m_ipca["auc_pr"], "auc_roc": m_ipca["auc_roc"],
                "f1": m_ipca["f1"], "precision": m_ipca["precision"],
                "recall": m_ipca["recall"], "separation_ratio": m_ipca["separation_ratio"],
            },
            "windows_streaming": windows_stream,
            "windows_static": windows_static,
        }
        all_results[drift_type] = drift_results

        # Save per drift type
        with open(os.path.join(OUT_DIR, f"drift_type_{drift_type}.json"), "w") as f:
            json.dump(drift_results, f, indent=2)

    # Summary table
    LOGGER.info("")
    LOGGER.info("=" * 100)
    LOGGER.info("CONCEPT DRIFT SUMMARY:")
    LOGGER.info("=" * 100)
    LOGGER.info(f"{'Drift':<30} {'MS+Stream':>10} {'MS+Static':>10} {'IF':>10} {'PCA':>10} "
                f"{'MS_Adv':>10}")
    LOGGER.info("-" * 100)
    for dt, dr in all_results.items():
        ms_s = dr["memstream_streaming"]["auc_pr"]
        ms_st = dr["memstream_static"]["auc_pr"]
        iff = dr["isolation_forest"]["auc_pr"]
        pca = dr["inc_pca"]["auc_pr"]
        adv = ms_s - ms_st
        LOGGER.info(f"{dr['drift_name']:<30} {ms_s:>10.4f} {ms_st:>10.4f} "
                   f"{iff:>10.4f} {pca:>10.4f} {adv:>+10.4f}")

    # Save full results
    with open(os.path.join(OUT_DIR, "concept_drift_results.json"), "w") as f:
        json.dump({"experiment": "Concept Drift Injection", "drift_results": all_results,
                   "best_config": best}, f, indent=2)

    LOGGER.info("")
    LOGGER.info("Average streaming advantage (AUC-PR): %.4f",
               np.mean([r["memstream_streaming"]["auc_pr"] - r["memstream_static"]["auc_pr"]
                        for r in all_results.values()]))
    LOGGER.info("Results saved to: %s", OUT_DIR)


if __name__ == "__main__":
    main()
