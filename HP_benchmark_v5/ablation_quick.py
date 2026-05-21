"""
Quick ablation with CORRECT data (GOOD_DATA/val_*.parquet like grid_search).
Test streaming vs non-streaming, KNN, beta variants.
"""
import sys, os, time, json
sys.path.insert(0, "c:\\proj\\ldt\\HP_benchmark_v5")

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from benchmark_core import MemStreamPipeline

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


def eval_all(scores, gt):
    best_t, _ = find_best_threshold(scores, gt)
    pred = scores >= best_t
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    tn = int((~pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    auc_roc = roc_auc_score(gt, scores)
    auc_pr = average_precision_score(gt, scores)
    sep = scores[gt].mean() / (scores[~gt].mean() + 1e-10)
    return auc_pr, auc_roc, f1, prec, rec, best_t, sep, tp, fp, tn, fn


def load_good_data(max_train=200000, max_val=300000):
    """Load same data as grid_search: GOOD_DATA/val_*.parquet."""
    data_dir = "c:\\proj\\ldt\\GOOD_DATA"
    from benchmark_core import extract_features_from_parquet

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
        X, h, d, rc, nb, _ = extract_features_from_parquet(os.path.join(data_dir, fn))
        X_va.append(X); h_va.append(h); d_va.append(d); rc_va.append(rc); nb_va.append(nb)

    X_val = np.vstack(X_va)[:max_val]
    hours_val = np.concatenate(h_va)[:max_val]
    dows_val = np.concatenate(d_va)[:max_val]
    rcs_val = np.concatenate(rc_va)[:max_val]
    nb_val = np.concatenate(nb_va)[:max_val]

    # Ground truth: last file's anomaly column
    _, _, _, _, _, gt = extract_features_from_parquet(os.path.join(data_dir, val_files[0]))
    gt_mask = gt[:max_val].astype(bool)

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Anomalies: {gt_mask.sum()} ({gt_mask.mean()*100:.2f}%)")
    return X_train, hours_train, dows_train, rcs_train, nb_train, \
           X_val, hours_val, dows_val, rcs_val, nb_val, gt_mask


def run_exp(name, X_tr, h_tr, d_tr, rc_tr, nb_tr,
            X_va, h_va, d_va, rc_va, nb_va, gt_mask,
            cfg_base, epochs=5000,
            update_mem=False, k_override=None, beta_override=None):
    t0 = time.time()
    cfg = {**cfg_base}
    if k_override is not None: cfg["k"] = k_override
    if beta_override is not None: cfg["beta"] = beta_override

    pipe = MemStreamPipeline(
        d=38,
        out_dim=cfg["out_dim"],
        memory_len=cfg["memory_len"],
        k=cfg["k"],
        gamma=cfg["gamma"],
        beta=cfg["beta"],
        noise_std=cfg["noise_std"],
        lr=cfg["lr"],
        epochs=epochs,
        batch_size=cfg.get("batch_size", 64),
        seed=42,
        cb_warmup=min(4096, cfg["memory_len"] * 4),
        verbose=False,
        adam_betas=tuple(cfg.get("adam_betas", [0.9, 0.999])),
    )
    pipe.train(
        X_tr, h_tr, d_tr, rc_tr, nb_tr,
        X_warmup=X_tr[:cfg["memory_len"]],
        hours_warmup=h_tr[:cfg["memory_len"]],
        dows_warmup=d_tr[:cfg["memory_len"]],
        rcs_warmup=rc_tr[:cfg["memory_len"]],
        nb_warmup=nb_tr[:cfg["memory_len"]],
    )

    scores, ms_info = pipe.score_stream(
        X_va, h_va, d_va, rc_va, nb_va,
        gt_mask=gt_mask, update_memory=update_mem)
    n_upd = ms_info.get("n_memory_updates", 0)

    auc_pr, auc_roc, f1, prec, rec, best_t, sep, tp, fp, tn, fn = eval_all(scores, gt_mask)
    elapsed = time.time() - t0

    print(f"  {name:<50s} PR={auc_pr:.4f} ROC={auc_roc:.4f} F1={f1:.4f} "
          f"P={prec:.4f} R={rec:.4f} sep={sep:.1f}x upd={n_upd} [{elapsed:.0f}s]")

    return {
        "name": name, "auc_pr": auc_pr, "auc_roc": auc_roc, "f1": f1,
        "prec": prec, "rec": rec, "sep": sep, "best_t": best_t,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_updates": n_upd, "time_s": elapsed,
        "cfg": {**cfg, "k": cfg["k"], "beta": cfg["beta"]},
        "update_mem": update_mem,
    }


def main():
    os.chdir("c:\\proj\\ldt\\HP_benchmark_v5")

    # Load best config from grid search
    with open("results/grid_search/final_results.json") as f:
        gs = json.load(f)
    best = max(gs["all_results"], key=lambda r: r.get("auc_pr", 0))
    print(f"\nGrid search best: {best['config_id']}, AUC-PR={best['auc_pr']:.4f}")
    print(f"  k={best['k']}, mem={best['memory_len']}, beta={best['beta']}, "
          f"out_dim={best['out_dim']}, noise={best['noise_std']}")

    # Load data (SAME as grid_search: GOOD_DATA)
    print("\nLoading data (GOOD_DATA like grid_search)...")
    X_tr, h_tr, d_tr, rc_tr, nb_tr, X_va, h_va, d_va, rc_va, nb_va, gt_mask = load_good_data(200000, 300000)

    results = []
    print("\n" + "=" * 110)
    print("PHASE 1: STREAMING vs NON-STREAMING")
    print("=" * 110)

    r = run_exp("NO_UPDATE (eval mode)", X_tr, h_tr, d_tr, rc_tr, nb_tr,
                 X_va, h_va, d_va, rc_va, nb_va, gt_mask, best, update_mem=False)
    results.append(r)

    r = run_exp("WITH_UPDATE beta=0.001 (like comparison.py)", X_tr, h_tr, d_tr, rc_tr, nb_tr,
                 X_va, h_va, d_va, rc_va, nb_va, gt_mask, best, update_mem=True)
    results.append(r)

    print("\n" + "=" * 110)
    print("PHASE 2: BETA VARIANTS (with streaming)")
    print("=" * 110)
    for beta in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
        r = run_exp(f"WITH_UPDATE beta={beta}", X_tr, h_tr, d_tr, rc_tr, nb_tr,
                     X_va, h_va, d_va, rc_va, nb_va, gt_mask, best,
                     update_mem=True, beta_override=beta)
        results.append(r)

    print("\n" + "=" * 110)
    print("PHASE 3: KNN VARIANTS (non-streaming)")
    print("=" * 110)
    for k in [3, 5, 7, 10, 15, 20, 30]:
        r = run_exp(f"NO_UPDATE k={k}", X_tr, h_tr, d_tr, rc_tr, nb_tr,
                     X_va, h_va, d_va, rc_va, nb_va, gt_mask, best,
                     update_mem=False, k_override=k)
        results.append(r)

    print("\n" + "=" * 110)
    print("PHASE 4: KNN + STREAMING (best betas)")
    print("=" * 110)
    for k in [5, 10]:
        for beta in [0.05, 0.1, 0.3, 0.5]:
            r = run_exp(f"WITH_UPDATE k={k} beta={beta}", X_tr, h_tr, d_tr, rc_tr, nb_tr,
                         X_va, h_va, d_va, rc_va, nb_va, gt_mask, best,
                         update_mem=True, k_override=k, beta_override=beta)
            results.append(r)

    # Summary
    print("\n\n" + "=" * 110)
    print("SORTED BY AUC-PR")
    print("=" * 110)
    results.sort(key=lambda x: x["auc_pr"], reverse=True)
    print(f"{'Rank':<5} {'AUC-PR':<8} {'AUC-ROC':<8} {'F1':<7} {'P':<7} {'R':<7} {'Sep':<8} {'Upd':<8} Name")
    print("-" * 110)
    for i, r in enumerate(results):
        tag = " <<<" if i == 0 else ""
        print(f"{i+1:<5} {r['auc_pr']:<8.4f} {r['auc_roc']:<8.4f} {r['f1']:<7.4f} "
              f"{r['prec']:<7.4f} {r['rec']:<7.4f} {r['sep']:<8.1f} "
              f"{r['n_updates']:<8d} {r['name']}{tag}")

    winner = results[0]
    print(f"\n*** WINNER: {winner['name']} ***")
    print(f"    AUC-PR={winner['auc_pr']:.4f} F1={winner['f1']:.4f}")
    print(f"    k={winner['cfg']['k']}, beta={winner['cfg']['beta']}, "
          f"out_dim={winner['cfg']['out_dim']}, updates={winner['n_updates']}")

    os.makedirs("results/ablation", exist_ok=True)
    with open("results/ablation/scoring_quick.json", "w") as f:
        json.dump({"results": results, "grid_best": best}, f, indent=2, default=str)
    print(f"\nSaved: results/ablation/scoring_quick.json")


if __name__ == "__main__":
    main()
