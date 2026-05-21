"""Test streaming vs eval step by step."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys, time, json
sys.path.insert(0, "c:\\proj\\ldt\\HP_benchmark_v5")

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from benchmark_core import MemStreamPipeline, extract_features_from_parquet


def eval_score(scores, gt):
    best_f1, best_t = 0, 0.0
    for t in np.linspace(scores.min(), scores.max(), 2000):
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
    auc_pr = average_precision_score(gt, scores)
    auc_roc = roc_auc_score(gt, scores)
    sep = scores[gt].mean() / (scores[~gt].mean() + 1e-10)
    return auc_pr, auc_roc, f1, prec, rec, sep, best_t


def main():
    os.chdir("c:\\proj\\ldt\\HP_benchmark_v5")

    with open("results/grid_search/final_results.json") as f:
        gs = json.load(f)
    base = max(gs["all_results"], key=lambda r: r.get("auc_pr", 0))

    print("Loading...")
    sys.stdout.flush()
    X_tr, h_tr, d_tr, rc_tr, nb_tr = extract_features_from_parquet(
        "data/train_clean.parquet", max_rows=200000)
    X_va, h_va, d_va, rc_va, nb_va = extract_features_from_parquet(
        "data/test_polluted.parquet", max_rows=50000)
    gt = np.load("data/test/ground_truth_mask.npy")[-len(X_va):]
    print("Train=%d Val=%d Anom=%d" % (len(X_tr), len(X_va), gt.sum()))
    sys.stdout.flush()

    # Train ONCE
    print("Training...")
    sys.stdout.flush()
    t0 = time.time()
    pipe = MemStreamPipeline(
        d=38, out_dim=base["out_dim"], memory_len=base["memory_len"],
        k=base["k"], gamma=base["gamma"], beta=base["beta"],
        noise_std=base["noise_std"], lr=base["lr"],
        epochs=5000, batch_size=64, seed=42,
        cb_warmup=min(4096, base["memory_len"] * 4),
        verbose=False,
        adam_betas=tuple(base.get("adam_betas", [0.9, 0.999])))
    pipe.train(X_tr, h_tr, d_tr, rc_tr, nb_tr,
               X_warmup=X_tr[:base["memory_len"]],
               hours_warmup=h_tr[:base["memory_len"]],
               dows_warmup=d_tr[:base["memory_len"]],
               rcs_warmup=rc_tr[:base["memory_len"]],
               nb_warmup=nb_tr[:base["memory_len"]])
    print("Trained in %.0fs" % (time.time() - t0))
    sys.stdout.flush()

    # Test eval mode
    print("Testing NO_UPDATE (eval mode)...")
    sys.stdout.flush()
    scores1, ms1 = pipe.score_stream(
        X_va, h_va, d_va, rc_va, nb_va,
        gt_mask=gt, update_memory=False)
    pr1, roc1, f1_1, prec1, rec1, sep1, th1 = eval_score(scores1, gt)
    print("NO_UPDATE: PR=%.4f ROC=%.4f F1=%.4f sep=%.1fx" % (pr1, roc1, f1_1, sep1))
    sys.stdout.flush()

    # Test streaming mode
    print("Testing WITH_UPDATE (streaming)...")
    sys.stdout.flush()
    # Reset memory to trained state first
    pipe.memory._cnt = base["memory_len"]
    scores2, ms2 = pipe.score_stream(
        X_va, h_va, d_va, rc_va, nb_va,
        gt_mask=gt, update_memory=True)
    pr2, roc2, f1_2, prec2, rec2, sep2, th2 = eval_score(scores2, gt)
    upd = ms2.get("n_memory_updates", 0)
    print("WITH_UPDATE: PR=%.4f ROC=%.4f F1=%.4f sep=%.1fx upd=%d" % (pr2, roc2, f1_2, sep2, upd))
    sys.stdout.flush()

    print("\n=== SUMMARY ===")
    print("NO_UPDATE:   AUC-PR=%.4f" % pr1)
    print("WITH_UPDATE: AUC-PR=%.4f" % pr2)
    print("Delta: %+.4f" % (pr2 - pr1))
    if pr2 > pr1:
        print("STREAMING wins!")
    else:
        print("EVAL mode wins!")


if __name__ == "__main__":
    main()
