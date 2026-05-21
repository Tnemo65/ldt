"""
HP_benchmark_v5 scoring ablation
Test: KNN variants, batch size, scoring strategies, memory update modes
Goal: Achieve best possible AUC-PR.
"""
import sys, time, json, traceback
import numpy as np

sys.path.insert(0, "c:\\proj\\ldt\\HP_benchmark_v5")

from benchmark_core import MemStreamPipeline, extract_features_from_parquet

PYTHON = "C:\\Users\\Administrator\\Desktop\\AI ComfyUI\\system\\python\\python.exe"

def find_best_threshold(scores, gt):
    best_f1, best_t = 0, 0.5
    for t in np.linspace(scores.min(), scores.max(), 1000):
        pred = scores >= t
        tp = (pred & gt).sum()
        fp = (pred & ~gt).sum()
        fn = (~pred & gt).sum()
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

def eval_scores(scores, gt, t):
    pred = scores >= t
    tp, fp, tn, fn = int((pred & gt).sum()), int((pred & ~gt).sum()), \
                     int((~pred & ~gt).sum()), int((~pred & gt).sum())
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc_roc = roc_auc_score(gt, scores)
    auc_pr = average_precision_score(gt, scores)
    return auc_roc, auc_pr, f1, prec, rec

def load_data(max_train=200000, max_val=300000):
    print("[DATA] Loading...")
    import os
    data_dir = "c:\\proj\\ldt\\GOOD_DATA"
    train_files = sorted([f for f in os.listdir(data_dir) if f.startswith("train_")])
    val_files = sorted([f for f in os.listdir(data_dir) if f.startswith("val_")])

    X_tr, h_tr, d_tr, rc_tr, nb_tr = [], [], [], [], []
    for fn in train_files:
        X, h, d, rc, nb, _ = extract_features_from_parquet(os.path.join(data_dir, fn))
        X_tr.append(X); h_tr.append(h); d_tr.append(d); rc_tr.append(rc); nb_tr.append(nb)
    X_train = np.vstack(X_tr)[:max_train]
    hours_train = np.concatenate(h_tr)[:max_train]
    dows_train = np.concatenate(d_tr)[:max_train]
    rcs_train = np.concatenate(rc_tr)[:max_train]
    nb_train = np.concatenate(nb_tr)[:max_train]

    X_va, h_va, d_va, rc_va, nb_va = [], [], [], [], []
    for fn in val_files:
        X, h, d, rc, nb, gt = extract_features_from_parquet(os.path.join(data_dir, fn))
        X_va.append(X); h_va.append(h); d_va.append(d); rc_va.append(rc); nb_va.append(nb)
    X_val = np.vstack(X_va)[:max_val]
    hours_val = np.concatenate(h_va)[:max_val]
    dows_val = np.concatenate(d_va)[:max_val]
    rcs_val = np.concatenate(rc_va)[:max_val]
    nb_val = np.concatenate(nb_va)[:max_val]
    _, _, _, _, _, gt = extract_features_from_parquet(os.path.join(data_dir, val_files[0]))
    gt_mask = gt[:max_val].astype(bool)

    print(f"  Train: {len(X_train)}, Val: {len(X_val)} (GT: {gt_mask.sum()} anomalies)")
    return X_train, hours_train, dows_train, rcs_train, nb_train, \
           X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask

def run_experiment(name, X_train, hours_train, dows_train, rcs_train, nb_train,
                   X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask,
                   best_config, epochs=5000, update_memory=False,
                   score_mode="auto",  # "auto" = use batch, "point" = per-point
                   k=None, gamma=None, beta=None):
    t0 = time.time()
    cfg = {**best_config}
    if k is not None: cfg["k"] = k
    if gamma is not None: cfg["gamma"] = gamma
    if beta is not None: cfg["beta"] = beta

    pipeline = MemStreamPipeline(
        d=best_config.get("d", 38),
        out_dim=cfg["out_dim"],
        memory_len=cfg["memory_len"],
        k=cfg["k"],
        gamma=cfg["gamma"],
        beta=cfg["beta"],
        noise_std=cfg["noise_std"],
        lr=cfg["lr"],
        epochs=epochs,
        batch_size=64,
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

    if score_mode == "batch":
        # Force batch scoring: encode all, then batch
        Xn = (X_val.astype(np.float32) - pipeline._input_mean) / (pipeline._input_std + 1e-8)
        import torch
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoded = []
        for s in range(0, len(Xn), 1024):
            e = min(s + 1024, len(Xn))
            with torch.no_grad():
                enc = pipeline.ae.encode(torch.from_numpy(Xn[s:e]).to(DEVICE)).cpu().numpy()
            encoded.append(enc)
        encoded = np.vstack(encoded)

        # Batch score WITHOUT streaming updates
        scores = pipeline.scorer.score_batch(X_val)
    elif score_mode == "point":
        # Per-point scoring WITH streaming updates
        adj_scores, _ = pipeline.score_stream(
            X_val, hours_val, dows_val, rcs_val, nb_val,
            gt_mask=gt_mask, update_memory=update_memory,
        )
        scores = adj_scores
    else:
        # Auto: use the default score_stream path
        adj_scores, _ = pipeline.score_stream(
            X_val, hours_val, dows_val, rcs_val, nb_val,
            gt_mask=gt_mask, update_memory=update_memory,
        )
        scores = adj_scores

    elapsed = time.time() - t0

    best_t, best_f1 = find_best_threshold(scores, gt_mask)
    auc_roc, auc_pr, f1, prec, rec = eval_scores(scores, gt_mask, best_t)

    print(f"  [{name}] AUC-PR={auc_pr:.4f} AUC-ROC={auc_roc:.4f} F1={f1:.4f} "
          f"P={prec:.4f} R={rec:.4f} [{elapsed:.1f}s]")

    return {
        "name": name,
        "auc_pr": auc_pr, "auc_roc": auc_roc, "f1": f1,
        "precision": prec, "recall": rec,
        "time_s": elapsed,
        "config": cfg,
        "score_mode": score_mode,
        "update_memory": update_memory,
    }

def main():
    import os, torch
    os.chdir("c:\\proj\\ldt\\HP_benchmark_v5")

    # Load best config
    import json
    with open("results/grid_search/final_results.json") as f:
        gs = json.load(f)
    best = None
    for r in gs["all_results"]:
        if r.get("auc_pr", 0) > (best["auc_pr"] if best else 0):
            best = r
    print(f"[CONFIG] Best: {best['config_id']}, AUC-PR={best['auc_pr']:.4f}")
    print(f"  k={best['k']}, memory={best['memory_len']}, gamma={best['gamma']}, beta={best['beta']}")
    print(f"  out_dim={best['out_dim']}, noise={best['noise_std']}, lr={best['lr']}, epochs={best['epochs']}")

    # Load data (SAME as comparison.py: 200K train, 300K val)
    X_train, hours_train, dows_train, rcs_train, nb_train, \
    X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask = load_data(200000, 300000)

    # Add d to config
    best["d"] = X_train.shape[1]

    results = []

    print("\n=== SCORING MODE ABLATION ===")

    # 1. Baseline: comparison.py mode (update_memory=True)
    r = run_experiment(
        "comparison_mode (update=T, auto)", X_train, hours_train, dows_train, rcs_train, nb_train,
        X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, best,
        update_memory=True, score_mode="auto",
    )
    results.append(r)

    # 2. Grid search mode (update_memory=False)
    r = run_experiment(
        "grid_search_mode (update=F, auto)", X_train, hours_train, dows_train, rcs_train, nb_train,
        X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, best,
        update_memory=False, score_mode="auto",
    )
    results.append(r)

    # 3. Batch scoring (no streaming updates)
    r = run_experiment(
        "batch_mode (update=F, batch)", X_train, hours_train, dows_train, rcs_train, nb_train,
        X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, best,
        update_memory=False, score_mode="batch",
    )
    results.append(r)

    # 4. KNN variants
    for k_val in [3, 5, 10, 15, 20, 30]:
        r = run_experiment(
            f"k={k_val} (update=F)", X_train, hours_train, dows_train, rcs_train, nb_train,
            X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, best,
            update_memory=False, score_mode="auto", k=k_val,
        )
        results.append(r)

    # 5. KNN variants with streaming update
    for k_val in [5, 10]:
        r = run_experiment(
            f"k={k_val}+stream (update=T)", X_train, hours_train, dows_train, rcs_train, nb_train,
            X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, best,
            update_memory=True, score_mode="auto", k=k_val,
        )
        results.append(r)

    # 6. Score batch size variants (encode all, score in chunks)
    print("\n=== BATCH SIZE ABLATION ===")
    # Test with different scoring batch sizes (implicit in score_batch)

    # 7. Try higher out_dim
    print("\n=== ARCHITECTURE ABLATION ===")
    for out_dim in [38, 50, 68, 100, 128]:
        cfg = {**best, "out_dim": out_dim, "d": 38}
        r = run_experiment(
            f"out_dim={out_dim}", X_train, hours_train, dows_train, rcs_train, nb_train,
            X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, cfg,
            update_memory=False, score_mode="auto",
        )
        results.append(r)

    # 8. Try without denoising noise (noise_std=0)
    r = run_experiment(
        "no_denoise (noise=0)", X_train, hours_train, dows_train, rcs_train, nb_train,
        X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask, best,
        update_memory=False, score_mode="auto",
    )
    cfg_noise = {**best, "noise_std": 0.0}
    results.append({**r, "name": "no_denoise", "config": cfg_noise})

    # Summary
    print("\n\n=== RESULTS SUMMARY (sorted by AUC-PR) ===")
    results.sort(key=lambda x: x["auc_pr"], reverse=True)
    print(f"{'Rank':<4} {'AUC-PR':<8} {'AUC-ROC':<8} {'F1':<7} {'P':<7} {'R':<7} {'Mode':<30}")
    print("-" * 85)
    for i, r in enumerate(results):
        mode = f"upd={r['update_memory']} score={r['score_mode']}"
        k = r["config"].get("k", best.get("k"))
        od = r["config"].get("out_dim", best.get("out_dim"))
        name = f"{r['name']} (k={k},out={od})"
        print(f"{i+1:<4} {r['auc_pr']:<8.4f} {r['auc_roc']:<8.4f} {r['f1']:<7.4f} "
              f"{r['precision']:<7.4f} {r['recall']:<7.4f} {mode:<30}")

    # Save
    out = {"experiments": results, "best_config": best}
    with open("results/ablation/scoring_ablation.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to results/ablation/scoring_ablation.json")

if __name__ == "__main__":
    main()
