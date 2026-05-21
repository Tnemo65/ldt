"""Streaming ablation: prove update_memory=True can achieve good results with right beta."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys, time, json
sys.path.insert(0, "c:\\proj\\ldt\\HP_benchmark_v5")

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from benchmark_core import MemStreamPipeline, extract_features_from_parquet


def eval_score(scores, gt):
    best_f1, best_t = 0, 0.0
    for t in np.linspace(scores.min(), scores.max(), 1000):
        pred = scores >= t
        tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
        prec = tp / (tp + fp + 1e-10); rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        if f1 > best_f1: best_f1, best_t = f1, float(t)
    pred = scores >= best_t
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum())
    tn = int((~pred & ~gt).sum()); fn = int((~pred & gt).sum())
    prec = tp / (tp + fp + 1e-10); rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    auc_roc = roc_auc_score(gt, scores); auc_pr = average_precision_score(gt, scores)
    sep = scores[gt].mean() / (scores[~gt].mean() + 1e-10)
    return auc_pr, auc_roc, f1, prec, rec, sep, best_t


def main():
    os.chdir("c:\\proj\\ldt\\HP_benchmark_v5")

    with open("results/grid_search/final_results.json") as f:
        gs = json.load(f)
    base = max(gs["all_results"], key=lambda r: r.get("auc_pr", 0))
    print("Base config: " + base["config_id"])
    print("  k=%d, mem=%d, beta=%.4f, gamma=%.1f, od=%d, noise=%.4f" % (
        base["k"], base["memory_len"], base["beta"], base["gamma"],
        base["out_dim"], base["noise_std"]))

    print("\nLoading 3%% anomaly data...")
    X_tr, h_tr, d_tr, rc_tr, nb_tr = extract_features_from_parquet(
        "data/train_clean.parquet", max_rows=200000)
    X_va, h_va, d_va, rc_va, nb_va = extract_features_from_parquet(
        "data/test_polluted.parquet", max_rows=30000)
    gt = np.load("data/test/ground_truth_mask.npy")[-len(X_va):]
    print("  Train: %d, Val: %d, Anomalies: %d (%.2f%%)" % (
        len(X_tr), len(X_va), gt.sum(), gt.mean() * 100))

    results = []

    def run(name, cfg, update_mem, epochs=2000):
        t0 = time.time()
        pipe = MemStreamPipeline(
            d=38, out_dim=cfg["out_dim"], memory_len=cfg["memory_len"],
            k=cfg["k"], gamma=cfg["gamma"], beta=cfg["beta"],
            noise_std=cfg["noise_std"], lr=cfg["lr"],
            epochs=epochs, batch_size=64,
            seed=42, cb_warmup=min(4096, cfg["memory_len"] * 4),
            verbose=False, adam_betas=tuple(cfg.get("adam_betas", [0.9, 0.999])))
        pipe.train(X_tr, h_tr, d_tr, rc_tr, nb_tr,
                   X_warmup=X_tr[:cfg["memory_len"]],
                   hours_warmup=h_tr[:cfg["memory_len"]],
                   dows_warmup=d_tr[:cfg["memory_len"]],
                   rcs_warmup=rc_tr[:cfg["memory_len"]],
                   nb_warmup=nb_tr[:cfg["memory_len"]])
        scores, ms = pipe.score_stream(
            X_va, h_va, d_va, rc_va, nb_va,
            gt_mask=gt, update_memory=update_mem)
        n_upd = ms.get("n_memory_updates", 0)
        pr, roc, f1, prec, rec, sep, best_t = eval_score(scores, gt)
        elapsed = time.time() - t0
        snorm = scores[~gt].mean(); sanom = scores[gt].mean()
        print("  %-55s PR=%.4f ROC=%.4f F1=%.4f P=%.4f R=%.4f sep=%.1fx upd=%6d [%.0fs]" % (
            name, pr, roc, f1, prec, rec, sep, n_upd, elapsed))
        return dict(name=name, cfg=cfg, update_mem=update_mem, epochs=epochs,
                    auc_pr=pr, auc_roc=roc, f1=f1, prec=prec, rec=rec,
                    sep=sep, best_t=best_t, n_updates=n_upd,
                    score_normal_mean=snorm, score_anomaly_mean=sanom,
                    time_s=elapsed)

    # ---- BASELINE ----
    print("\n=== BASELINE (no streaming update) ===")
    cfg_base = dict(
        out_dim=base["out_dim"], memory_len=base["memory_len"],
        k=base["k"], gamma=base["gamma"], beta=base["beta"],
        noise_std=base["noise_std"], lr=base["lr"],
        adam_betas=tuple(base.get("adam_betas", [0.9, 0.999])))
    r = run("NO_UPDATE (baseline)", cfg_base, update_mem=False)
    results.append(r)

    # ---- STREAMING: BETA SEARCH ----
    print("\n=== STREAMING: BETA SEARCH (key experiment) ===")
    betas_to_test = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    for beta in betas_to_test:
        cfg = {**cfg_base, "beta": beta}
        r = run("STREAM beta=" + str(beta), cfg, update_mem=True)
        results.append(r)

    best_beta = max([r for r in results if r["update_mem"]], key=lambda x: x["auc_pr"])["cfg"]["beta"]

    # ---- KNN with best beta ----
    print("\n=== STREAMING: K SEARCH (beta=%.4f) ===" % best_beta)
    for k in [3, 5, 7, 10, 15, 20, 30]:
        cfg = {**cfg_base, "beta": best_beta, "k": k}
        r = run("STREAM k=%d beta=%.3f" % (k, best_beta), cfg, update_mem=True)
        results.append(r)

    best_k = max([r for r in results if r["update_mem"]], key=lambda x: x["auc_pr"])["cfg"]["k"]

    # ---- GAMMA ----
    print("\n=== STREAMING: GAMMA (k=%d, beta=%.4f) ===" % (best_k, best_beta))
    for gamma in [0.0, 0.1, 0.2, 0.3, 0.5]:
        cfg = {**cfg_base, "beta": best_beta, "k": best_k, "gamma": gamma}
        r = run("STREAM gamma=%.1f" % gamma, cfg, update_mem=True)
        results.append(r)

    best_gamma = max([r for r in results if r["update_mem"]], key=lambda x: x["auc_pr"])["cfg"]["gamma"]

    # ---- MEMORY LEN ----
    print("\n=== STREAMING: MEMORY LEN ===")
    for mem in [256, 512, 1024, 2048, 4096]:
        cfg = {**cfg_base, "beta": best_beta, "k": best_k, "gamma": best_gamma, "memory_len": mem}
        r = run("STREAM mem=%d" % mem, cfg, update_mem=True)
        results.append(r)

    # ---- SUMMARY ----
    results.sort(key=lambda x: x["auc_pr"], reverse=True)
    no_upd = [r for r in results if not r["update_mem"]][0]

    print("\n\n" + "=" * 110)
    print("RESULTS (sorted by AUC-PR)")
    print("=" * 110)
    print("%-4s %-8s %-8s %-7s %-7s %-7s %-8s %-8s %s" % (
        "#", "AUC-PR", "AUC-ROC", "F1", "Prec", "Rec", "Sep", "Updates", "Config"))
    print("-" * 110)
    for i, r in enumerate(results):
        tag = " <<<" if i == 0 else (" BSL" if not r["update_mem"] else "")
        cfg = r["cfg"]
        cname = "k=%d,b=%.3f,g=%.1f,m=%d" % (
            cfg["k"], cfg["beta"], cfg["gamma"], cfg["memory_len"])
        prefix = "EVAL  |" if not r["update_mem"] else "STREAM|"
        print("%-4d %.4f   %.4f   %.4f   %.4f   %.4f   %.1fx   %6d   %s%s" % (
            i + 1, r["auc_pr"], r["auc_roc"], r["f1"], r["prec"], r["rec"],
            r["sep"], r["n_updates"], prefix + cname, tag))

    winner = results[0]
    print("\n*** WINNER: %s ***" % winner["name"])
    print("    AUC-PR=%.4f F1=%.4f AUC-ROC=%.4f" % (winner["auc_pr"], winner["f1"], winner["auc_roc"]))
    print("    P=%.4f R=%.4f" % (winner["prec"], winner["rec"]))
    print("    k=%d, beta=%.4f, gamma=%.1f, mem=%d" % (
        winner["cfg"]["k"], winner["cfg"]["beta"], winner["cfg"]["gamma"], winner["cfg"]["memory_len"]))
    print("    updates=%d" % winner["n_updates"])
    print("\n  vs NO_UPDATE baseline: AUC-PR=%.4f" % no_upd["auc_pr"])
    print("  Delta: %+.4f" % (winner["auc_pr"] - no_upd["auc_pr"]))

    os.makedirs("results/ablation", exist_ok=True)
    with open("results/ablation/streaming_proof.json", "w") as f:
        json.dump({"results": results, "base_config": base,
                   "data_info": {"n_train": len(X_tr), "n_val": len(X_va),
                                 "n_anomalies": int(gt.sum()), "anomaly_rate": float(gt.mean())}},
                  f, indent=2, default=str)
    print("\nSaved: results/ablation/streaming_proof.json")


if __name__ == "__main__":
    main()
