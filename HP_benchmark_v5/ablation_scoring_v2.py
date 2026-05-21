"""
HP_benchmark_v5 scoring ablation v2
Use SAME data as comparison.py (train_clean.parquet + test_polluted.parquet)
Test: streaming vs non-streaming, KNN variants, scoring strategies
Goal: Achieve best possible AUC-PR.
"""
import sys, os, time, json, traceback
sys.path.insert(0, "c:\\proj\\ldt\\HP_benchmark_v5")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from benchmark_core import MemStreamPipeline, extract_features_from_parquet

DATA_DIR = "c:\\proj\\ldt\\HP_benchmark_v5\\data"
TRAIN_PATH = os.path.join(DATA_DIR, "train_clean.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test_polluted.parquet")
GT_PATH = os.path.join(DATA_DIR, "test", "ground_truth_mask.npy")

PYTHON = "C:\\Users\\Administrator\\Desktop\\AI ComfyUI\\system\\python\\python.exe"


def find_best_threshold(scores, gt):
    best_f1, best_t = 0, 0.0
    for t in np.linspace(scores.min(), scores.max(), 2000):
        pred = scores >= t
        tp = (pred & gt).sum()
        fp = (pred & ~gt).sum()
        fn = (~pred & gt).sum()
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def eval_scores(scores, gt, t):
    pred = scores >= t
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    tn = int((~pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    auc_roc = roc_auc_score(gt, scores)
    auc_pr = average_precision_score(gt, scores)
    return auc_roc, auc_pr, f1, prec, rec, tp, fp, tn, fn


def load_data(max_train=200000, max_val=300000):
    print("[DATA] Loading from comparison.py paths...")
    X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
        TRAIN_PATH, max_rows=max_train if max_train > 0 else None
    )
    X_test, hours_test, dows_test, rcs_test, nb_test = extract_features_from_parquet(
        TEST_PATH, max_rows=max_val
    )
    gt_mask = np.load(GT_PATH)
    gt_mask = gt_mask[-len(X_test):]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)} (GT: {gt_mask.sum()} anomalies, {gt_mask.mean()*100:.2f}%)")
    return X_train, hours_train, dows_train, rcs_train, nb_train, \
           X_test, hours_test, dows_test, rcs_test, nb_test, gt_mask


def run_exp(name, X_train, hours_train, dows_train, rcs_train, nb_train,
            X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask,
            best_config, epochs=5000,
            update_memory=False,
            score_fn="stream",  # "stream" or "batch"
            k=None, gamma=None, beta=None,
            batch_size=64):
    t0 = time.time()
    cfg = {**best_config}
    if k is not None: cfg["k"] = k
    if gamma is not None: cfg["gamma"] = gamma
    if beta is not None: cfg["beta"] = beta

    pipeline = MemStreamPipeline(
        d=38,
        out_dim=cfg["out_dim"],
        memory_len=cfg["memory_len"],
        k=cfg["k"],
        gamma=cfg["gamma"],
        beta=cfg["beta"],
        noise_std=cfg["noise_std"],
        lr=cfg["lr"],
        epochs=epochs,
        batch_size=batch_size,
        seed=42,
        cb_warmup=min(4096, cfg["memory_len"] * 4),
        verbose=False,
        adam_betas=tuple(cfg.get("adam_betas", [0.9, 0.999])),
    )

    pipeline.train(
        X_train, hours_train, dows_train, rcs_train, nb_train,
        X_warmup=X_train[:cfg["memory_len"]],
        hours_warmup=hours_train[:cfg["memory_len"]],
        dows_warmup=dows_train[:cfg["memory_len"]],
        rcs_warmup=rcs_train[:cfg["memory_len"]],
        nb_warmup=nb_train[:cfg["memory_len"]],
    )

    if score_fn == "stream":
        adj_scores, ms_metrics = pipeline.score_stream(
            X_val, hours_val, dows_val, rcs_val, nb_val,
            gt_mask=gt_mask, update_memory=update_memory,
        )
        scores = adj_scores
        n_updates = ms_metrics.get("n_memory_updates", 0)
    else:
        # Batch: encode all, score WITHOUT updating memory
        Xn = (X_val.astype(np.float32) - pipeline._input_mean) / (pipeline._input_std + 1e-8)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoded = []
        for s in range(0, len(Xn), 1024):
            e = min(s + 1024, len(Xn))
            with torch.no_grad():
                enc = pipeline.ae.encode(torch.from_numpy(Xn[s:e]).to(DEVICE)).cpu().numpy()
            encoded.append(enc)
        encoded = np.vstack(encoded)

        # Score using PaperScorer (CPU, no memory update)
        scores = pipeline.scorer.score_batch(X_val)
        n_updates = 0

    elapsed = time.time() - t0

    best_t, best_f1 = find_best_threshold(scores, gt_mask)
    auc_roc, auc_pr, f1, prec, rec, tp, fp, tn, fn = eval_scores(scores, gt_mask, best_t)

    sep_ratio = (scores[gt_mask].mean() / (scores[~gt_mask].mean() + 1e-10))

    print(f"  [{name:50s}] AUC-PR={auc_pr:.4f} AUC-ROC={auc_roc:.4f} "
          f"F1={f1:.4f} P={prec:.4f} R={rec:.4f} sep={sep_ratio:.2f}x "
          f"mem_upd={n_updates} [{elapsed:.1f}s]")

    return {
        "name": name,
        "auc_pr": float(auc_pr), "auc_roc": float(auc_roc), "f1": float(f1),
        "precision": float(prec), "recall": float(rec),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sep_ratio": float(sep_ratio),
        "best_threshold": float(best_t),
        "n_memory_updates": n_updates,
        "score_normal_mean": float(scores[~gt_mask].mean()),
        "score_anomaly_mean": float(scores[gt_mask].mean()),
        "score_normal_std": float(scores[~gt_mask].std()),
        "score_anomaly_std": float(scores[gt_mask].std()),
        "time_s": float(elapsed),
        "config": {k: cfg[k] for k in ["k", "gamma", "beta", "out_dim", "noise_std", "memory_len"]},
        "update_memory": update_memory,
        "score_fn": score_fn,
        "batch_size": batch_size,
    }


def main():
    os.chdir("c:\\proj\\ldt\\HP_benchmark_v5")

    # Load best config from grid search
    with open("results/grid_search/final_results.json") as f:
        gs = json.load(f)
    best = None
    for r in gs["all_results"]:
        if r.get("auc_pr", 0) > (best["auc_pr"] if best else 0):
            best = r
    print(f"[CONFIG] Best: {best['config_id']}, AUC-PR={best['auc_pr']:.4f}")
    print(f"  k={best['k']}, mem={best['memory_len']}, gamma={best['gamma']}, "
          f"beta={best['beta']}, out_dim={best['out_dim']}, noise={best['noise_std']}")

    # Load data (SAME as comparison.py)
    X_train, hours_train, dows_train, rcs_train, nb_train, \
    X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask = load_data(200000, 300000)

    results = []

    print("\n" + "=" * 120)
    print("PHASE 1: STREAMING vs NON-STREAMING (baseline comparison)")
    print("=" * 120)

    # 1. Comparison.py mode: streaming update=True
    r = run_exp("comparison_mode (stream_update=True)", X_train, hours_train, dows_train, rcs_train, nb_train,
                X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, best,
                update_memory=True, score_fn="stream")
    results.append(r)

    # 2. Non-streaming (eval mode, same as grid search)
    r = run_exp("grid_search_mode (stream_update=False)", X_train, hours_train, dows_train, rcs_train, nb_train,
                X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, best,
                update_memory=False, score_fn="stream")
    results.append(r)

    # 3. Batch scoring (no streaming updates)
    r = run_exp("batch_mode (no_stream, batch_scoring)", X_train, hours_train, dows_train, rcs_train, nb_train,
                X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, best,
                update_memory=False, score_fn="batch")
    results.append(r)

    print("\n" + "=" * 120)
    print("PHASE 2: KNN VARIANTS (non-streaming)")
    print("=" * 120)

    for k_val in [3, 5, 10, 15, 20, 30, 50]:
        r = run_exp(f"k={k_val}", X_train, hours_train, dows_train, rcs_train, nb_train,
                    X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, best,
                    update_memory=False, score_fn="stream", k=k_val)
        results.append(r)

    print("\n" + "=" * 120)
    print("PHASE 3: KNN + STREAMING (key question: does streaming help?)")
    print("=" * 120)

    for k_val in [5, 10, 15]:
        r = run_exp(f"k={k_val}+stream", X_train, hours_train, dows_train, rcs_train, nb_train,
                    X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, best,
                    update_memory=True, score_fn="stream", k=k_val)
        results.append(r)

    print("\n" + "=" * 120)
    print("PHASE 4: BATCH SIZE ABLATION")
    print("=" * 120)

    for bs in [32, 64, 128, 256, 512, 1024]:
        r = run_exp(f"batch_size={bs}", X_train, hours_train, dows_train, rcs_train, nb_train,
                    X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, best,
                    update_memory=False, score_fn="stream", batch_size=bs)
        results.append(r)

    print("\n" + "=" * 120)
    print("PHASE 5: ARCHITECTURE (out_dim)")
    print("=" * 120)

    for od in [38, 50, 68, 100, 128]:
        r = run_exp(f"out_dim={od}", X_train, hours_train, dows_train, rcs_train, nb_train,
                    X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, best,
                    update_memory=False, score_fn="stream", k=best["k"])
        results.append({**r, "config": {**r["config"], "out_dim": od}})

    print("\n" + "=" * 120)
    print("PHASE 6: BETA (memory update threshold) variants")
    print("=" * 120)

    for beta_val in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
        r = run_exp(f"beta={beta_val}+stream", X_train, hours_train, dows_train, rcs_train, nb_train,
                    X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, best,
                    update_memory=True, score_fn="stream", beta=beta_val)
        results.append(r)

    # Summary
    print("\n\n" + "=" * 120)
    print("RESULTS SUMMARY (sorted by AUC-PR)")
    print("=" * 120)
    results.sort(key=lambda x: x["auc_pr"], reverse=True)
    print(f"{'Rank':<5} {'AUC-PR':<8} {'AUC-ROC':<8} {'F1':<7} {'P':<7} {'R':<7} {'Sep':<8} {'MemUpd':<8} {'Config'}")
    print("-" * 120)
    for i, r in enumerate(results):
        cfg_str = f"k={r['config'].get('k', best['k'])},beta={r['config'].get('beta', best['beta'])},upd={r['update_memory']}"
        print(f"{i+1:<5} {r['auc_pr']:<8.4f} {r['auc_roc']:<8.4f} {r['f1']:<7.4f} "
              f"{r['precision']:<7.4f} {r['recall']:<7.4f} {r['sep_ratio']:<8.2f} "
              f"{r['n_memory_updates']:<8d} {r['name']}")
        if i == 0:
            print(f"       *** BEST: {r['name']} ***")

    # Save
    out = {"experiments": results, "best_config": best, "data_info": {
        "train_path": TRAIN_PATH, "test_path": TEST_PATH,
        "gt_path": GT_PATH, "n_train": len(X_train), "n_val": len(X_val),
        "n_anomalies": int(gt_mask.sum()), "anomaly_rate": float(gt_mask.mean())
    }}
    os.makedirs("results/ablation", exist_ok=True)
    with open("results/ablation/scoring_ablation_v2.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to results/ablation/scoring_ablation_v2.json")


if __name__ == "__main__":
    main()
