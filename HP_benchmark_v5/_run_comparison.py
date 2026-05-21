#!/usr/bin/env python3
"""Comparison: MemStream vs baselines, 5000 epochs, 200K train, save all artifacts."""
from __future__ import annotations

import os
import sys
import json
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"
sys.path.insert(0, r"c:\proj\ldt\HP_benchmark_v5")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from benchmark_core import (
    MemStreamPipeline, extract_features_from_parquet,
    evaluate_scores, find_best_threshold,
)


def eval_score(scores, gt):
    from sklearn.metrics import roc_auc_score, average_precision_score
    scores = np.asarray(scores)
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
    pred = scores >= best_t
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    tn = int((~pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    return {
        "auc_roc": float(roc_auc_score(gt, scores)),
        "auc_pr": float(average_precision_score(gt, scores)),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "score_normal_mean": float(scores[~gt].mean()),
        "score_anomaly_mean": float(scores[gt].mean()),
        "separation_ratio": float(scores[gt].mean() / (scores[~gt].mean() + 1e-10)),
        "threshold": float(best_t),
    }


def main():
    EPOCHS = 5000
    MAX_TRAIN = 200000
    MAX_VAL = 100000
    SEED = 42
    OUTPUT = "results/comparison"

    os.makedirs(OUTPUT, exist_ok=True)

    # Load best config
    grid_path = r"c:\proj\ldt\HP_benchmark_v5\results\grid_search\final_results.json"
    with open(grid_path) as f:
        grid_data = json.load(f)
    best_cfg = grid_data["best_config"]
    print(f"Loaded best config: {best_cfg.get('config_id', 'N/A')}")
    print(f"  mem={best_cfg['memory_len']}, k={best_cfg['k']}, gamma={best_cfg['gamma']}, "
          f"beta={best_cfg['beta']}, epochs={EPOCHS}")

    # Load data
    print("Loading data...")
    X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
        "data/train_clean.parquet", max_rows=MAX_TRAIN
    )
    X_test, hours_test, dows_test, rcs_test, nb_test = extract_features_from_parquet(
        "data/test_polluted.parquet", max_rows=MAX_VAL
    )
    gt_mask = np.load("data/test/ground_truth_mask.npy")[-len(X_test) :]
    print(f"Train: {len(X_train)}, Test: {len(X_test)} "
          f"(GT: {gt_mask.sum()} anomalies, {gt_mask.mean()*100:.2f}%)")

    all_results = {}

    # ===================================================================
    # METHOD 1: IsolationForest
    # ===================================================================
    print("\n[1/4] IsolationForest...")
    t0 = time.time()

    clf_if = IsolationForest(
        n_estimators=200, max_samples=512, contamination=0.08,
        random_state=SEED, n_jobs=-1
    )
    clf_if.fit(X_train)
    raw_if = -clf_if.score_samples(X_test)
    scores_if = (raw_if - raw_if.min()) / (raw_if.max() - raw_if.min() + 1e-8)
    m_if = eval_score(scores_if, gt_mask)
    m_if["method"] = "IsolationForest"
    m_if["category"] = "offline"
    m_if["time_s"] = time.time() - t0
    print(f"  IF: AUC-PR={m_if['auc_pr']:.4f} AUC-ROC={m_if['auc_roc']:.4f} "
          f"F1={m_if['f1']:.4f} [{m_if['time_s']:.1f}s]")
    all_results["IsolationForest"] = m_if
    np.save(f"{OUTPUT}/if_scores.npy", scores_if)
    np.save(f"{OUTPUT}/if_detected.npy", (scores_if >= m_if["threshold"]).astype(np.int32))

    # ===================================================================
    # METHOD 2: Normal Autoencoder (offline)
    # ===================================================================
    print("\n[2/4] Normal Autoencoder...")
    t0 = time.time()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")

    n_tr = min(len(X_train), 100000)
    idx = np.random.RandomState(SEED).choice(len(X_train), n_tr, replace=False)
    X_tr_sub = X_train[idx]

    d = X_train.shape[1]
    hidden_dims = [64, 32]

    layers = []
    in_d = d
    for h_dim in hidden_dims:
        layers.append(nn.Linear(in_d, h_dim))
        layers.append(nn.ReLU())
        in_d = h_dim
    encoder = nn.Sequential(*layers)
    decoder_layers = []
    rev_dims = list(reversed(hidden_dims))
    for i, h_dim in enumerate(rev_dims):
        decoder_layers.append(nn.Linear(h_dim, rev_dims[i + 1] if i + 1 < len(rev_dims) else d))
        if i + 1 < len(rev_dims):
            decoder_layers.append(nn.ReLU())
    decoder = nn.Sequential(*decoder_layers)

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model_ae = AE().to(device)
    optimizer_ae = optim.Adam(model_ae.parameters(), lr=0.001)
    criterion_ae = nn.MSELoss()

    X_tr_t = torch.FloatTensor(X_tr_sub).to(device)
    X_te_t = torch.FloatTensor(X_test).to(device)
    train_loader = DataLoader(TensorDataset(X_tr_t), batch_size=256, shuffle=True)

    model_ae.train()
    for epoch in range(100):
        for batch in train_loader:
            xb = batch[0]
            optimizer_ae.zero_grad()
            recon = model_ae(xb)
            loss = criterion_ae(recon, xb)
            loss.backward()
            optimizer_ae.step()
        if (epoch + 1) % 20 == 0:
            print(f"  AE ep {epoch+1}/100")

    model_ae.eval()
    with torch.no_grad():
        recon_te = model_ae(X_te_t)
        recon_errors = torch.mean((recon_te - X_te_t) ** 2, dim=1).cpu().numpy()
    scores_ae = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min() + 1e-8)
    m_ae = eval_score(scores_ae, gt_mask)
    m_ae["method"] = "Normal Autoencoder"
    m_ae["category"] = "offline"
    m_ae["time_s"] = time.time() - t0
    print(f"  AE: AUC-PR={m_ae['auc_pr']:.4f} AUC-ROC={m_ae['auc_roc']:.4f} "
          f"F1={m_ae['f1']:.4f} [{m_ae['time_s']:.1f}s]")
    all_results["NormalAE"] = m_ae
    np.save(f"{OUTPUT}/normal_ae_scores.npy", scores_ae)
    np.save(f"{OUTPUT}/normal_ae_detected.npy", (scores_ae >= m_ae["threshold"]).astype(np.int32))

    # ===================================================================
    # METHOD 3: MemStream (streaming)
    # ===================================================================
    print("\n[3/4] MemStream v5 (streaming)...")
    t0 = time.time()

    pipe_ms = MemStreamPipeline(
        d=38,
        out_dim=best_cfg.get("out_dim", 68),
        memory_len=best_cfg["memory_len"],
        k=best_cfg["k"],
        gamma=best_cfg["gamma"],
        beta=best_cfg["beta"],
        noise_std=best_cfg["noise_std"],
        lr=best_cfg["lr"],
        epochs=EPOCHS,
        batch_size=1024,
        seed=SEED,
        cb_warmup=min(4096, best_cfg["memory_len"] * 4),
        verbose=False,
        adam_betas=(0.9, 0.999),
    )
    pipe_ms.train(
        X_train,
        hours_train,
        dows_train,
        rcs_train,
        nb_train,
        X_warmup=X_train[: best_cfg["memory_len"]],
        hours_warmup=hours_train[: best_cfg["memory_len"]],
        dows_warmup=dows_train[: best_cfg["memory_len"]],
        rcs_warmup=rcs_train[: best_cfg["memory_len"]],
        nb_warmup=nb_train[: best_cfg["memory_len"]],
    )

    adj_scores_ms, ms_metrics = pipe_ms.score_stream(
        X_test,
        hours_test,
        dows_test,
        rcs_test,
        nb_test,
        gt_mask=gt_mask,
        update_memory=True,
    )

    m_ms = eval_score(adj_scores_ms, gt_mask)
    m_ms["method"] = "MemStream v5"
    m_ms["category"] = "streaming"
    m_ms["time_s"] = time.time() - t0
    m_ms["n_mem_updates"] = ms_metrics.get("n_memory_updates", 0)
    m_ms["epoch_losses"] = [float(x) for x in pipe_ms.epoch_losses]
    print(f"  MS: AUC-PR={m_ms['auc_pr']:.4f} AUC-ROC={m_ms['auc_roc']:.4f} "
          f"F1={m_ms['f1']:.4f} [{m_ms['time_s']:.1f}s]")
    all_results["MemStream"] = m_ms
    np.save(f"{OUTPUT}/memstream_scores.npy", adj_scores_ms)
    np.save(f"{OUTPUT}/memstream_detected.npy", (adj_scores_ms >= m_ms["threshold"]).astype(np.int32))
    np.save(f"{OUTPUT}/memstream_epoch_losses.npy", np.array(pipe_ms.epoch_losses, dtype=np.float32))

    # ===================================================================
    # METHOD 4: One-Class SVM (offline)
    # ===================================================================
    print("\n[4/4] One-Class SVM...")
    t0 = time.time()

    n_tr_oc = min(len(X_train), 100000)
    idx_oc = np.random.RandomState(SEED).choice(len(X_train), n_tr_oc, replace=False)
    X_tr_oc = X_train[idx_oc]

    clf_oc = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
    clf_oc.fit(X_tr_oc)
    raw_oc = clf_oc.decision_function(X_test)
    scores_oc = -(raw_oc - raw_oc.min()) / (raw_oc.max() - raw_oc.min() + 1e-8)
    m_oc = eval_score(scores_oc, gt_mask)
    m_oc["method"] = "One-Class SVM"
    m_oc["category"] = "offline"
    m_oc["time_s"] = time.time() - t0
    print(f"  OCSVM: AUC-PR={m_oc['auc_pr']:.4f} AUC-ROC={m_oc['auc_roc']:.4f} "
          f"F1={m_oc['f1']:.4f} [{m_oc['time_s']:.1f}s]")
    all_results["OCSVM"] = m_oc
    np.save(f"{OUTPUT}/ocsvm_scores.npy", scores_oc)
    np.save(f"{OUTPUT}/ocsvm_detected.npy", (scores_oc >= m_oc["threshold"]).astype(np.int32))

    # ===================================================================
    # SAVE GT MASK
    # ===================================================================
    np.save(f"{OUTPUT}/gt_mask.npy", gt_mask)

    # ===================================================================
    # RANKING TABLE
    # ===================================================================
    print()
    print("=" * 80)
    print("COMPARISON RESULTS (sorted by AUC-PR)")
    print("=" * 80)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["auc_pr"], reverse=True)
    print(f"{'Rank':<6} {'Method':<22} {'Category':<12} {'AUC-PR':>8} {'AUC-ROC':>9} "
          f"{'F1':>8} {'Prec':>8} {'Rec':>8} {'Time':>8}")
    print("-" * 80)
    for rank, (key, m) in enumerate(sorted_results, 1):
        print(f"{rank:<6} {m['method']:<22} {m['category']:<12} "
              f"{m['auc_pr']:>8.4f} {m['auc_roc']:>9.4f} {m['f1']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['time_s']:>7.1f}s")

    # ===================================================================
    # SAVE RESULTS
    # ===================================================================
    out_json = {
        "results": {k: {kk: vv for kk, vv in v.items()} for k, v in all_results.items()},
        "ranking": [k for k, _ in sorted_results],
        "best_config": best_cfg,
        "epochs": EPOCHS,
        "max_train": MAX_TRAIN,
        "max_val": MAX_VAL,
        "seed": SEED,
    }
    with open(f"{OUTPUT}/comparison_results.json", "w") as f:
        json.dump(out_json, f, indent=2, default=str)

    print()
    print(f"All artifacts saved to {OUTPUT}/:")
    print("  if_scores.npy, if_detected.npy")
    print("  normal_ae_scores.npy, normal_ae_detected.npy")
    print("  memstream_scores.npy, memstream_detected.npy, memstream_epoch_losses.npy")
    print("  ocsvm_scores.npy, ocsvm_detected.npy")
    print("  gt_mask.npy")
    print("  comparison_results.json")
    print()
    print("ALL OK - COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
