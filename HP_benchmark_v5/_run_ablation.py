#!/usr/bin/env python3
"""Ablation study: 4 setups, 5000 epochs, 200K train, save all artifacts."""
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

from benchmark_core import (
    MemStreamPipeline, extract_features_from_parquet, extract_raw_features,
    evaluate_scores, find_best_threshold,
    compute_context_cell_id, DEVICE,
    MemStreamAE, Memory, PaperScorer, ContextBeta,
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
    OUTPUT = "results/ablation"

    os.makedirs(OUTPUT, exist_ok=True)

    # Load best config
    grid_path = r"c:\proj\ldt\HP_benchmark_v5\results\grid_search\final_results.json"
    with open(grid_path) as f:
        grid_data = json.load(f)
    best_cfg = grid_data["best_config"]
    print(f"Loaded best config: {best_cfg.get('config_id', 'N/A')}")
    print(f"  mem={best_cfg['memory_len']}, k={best_cfg['k']}, gamma={best_cfg['gamma']}, "
          f"beta={best_cfg['beta']}, noise={best_cfg['noise_std']}, lr={best_cfg['lr']}, "
          f"out_dim={best_cfg.get('out_dim',68)}, epochs={EPOCHS}")

    # Load data
    print("Loading data...")
    X_train_34, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
        "data/train_clean.parquet", max_rows=MAX_TRAIN
    )
    X_train_7, _, _, _, _ = extract_raw_features("data/train_clean.parquet", max_rows=MAX_TRAIN)
    X_test_34, hours_test, dows_test, rcs_test, nb_test = extract_features_from_parquet(
        "data/test_polluted.parquet", max_rows=MAX_VAL
    )
    X_test_7, _, _, _, _ = extract_raw_features("data/test_polluted.parquet", max_rows=MAX_VAL)
    gt_mask = np.load("data/test/ground_truth_mask.npy")[-len(X_test_34) :]
    print(f"Train: {len(X_train_34)} (38D), {len(X_train_7)} (7D)")
    print(f"Test: {len(X_test_34)} / GT: {gt_mask.sum()} anomalies ({gt_mask.mean()*100:.2f}%)")

    all_results = {}
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # =====================================================================
    # SETUP A: Normal Autoencoder (no noise, no memory, raw 7D features)
    # =====================================================================
    print("\n[Setup A] Normal Autoencoder (no noise, no memory)")
    t0 = time.time()

    d7 = X_train_7.shape[1]
    out_dim7 = d7 * 2
    lr_a = best_cfg.get("lr", 0.01)

    n_init = min(best_cfg.get("memory_len", 1024), len(X_train_7))
    idx_a = np.random.permutation(len(X_train_7))[:n_init]
    X_init_a = X_train_7[idx_a].astype(np.float32)
    mean_a = X_init_a.mean(axis=0).astype(np.float32)
    std_a = np.clip(X_init_a.std(axis=0), 1e-6, None).astype(np.float32)
    Xn_a = (X_init_a - mean_a) / (std_a + 1e-8)
    Xn_a_t = torch.from_numpy(Xn_a).to(DEVICE)

    ae_a = MemStreamAE(d=d7, out_dim=out_dim7).to(DEVICE)
    opt_a = torch.optim.Adam(ae_a.parameters(), lr=lr_a, betas=(0.9, 0.999))
    crit_a = torch.nn.MSELoss()

    ae_a.train()
    for ep in range(EPOCHS):
        perm_a = torch.randperm(len(Xn_a_t), device=DEVICE)
        for i in range(0, len(Xn_a_t), 1024):
            b = Xn_a_t[perm_a[i : i + 1024]]
            loss = crit_a(ae_a(b), b)
            opt_a.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_a.parameters(), 1.0)
            opt_a.step()
        if (ep + 1) % 1000 == 0:
            print(f"  A ep {ep+1}/{EPOCHS}")

    ae_a.eval()
    Xn_val_a = (X_test_7.astype(np.float32) - mean_a) / (std_a + 1e-8)
    Xn_val_a_t = torch.from_numpy(Xn_val_a).to(DEVICE)
    with torch.no_grad():
        recon_a = ae_a(Xn_val_a_t).cpu().numpy()
    scores_a = np.mean((Xn_val_a - recon_a) ** 2, axis=1)
    scores_a = (scores_a - scores_a.min()) / (scores_a.max() - scores_a.min() + 1e-8)
    best_t_a, _ = find_best_threshold(scores_a, gt_mask)
    metrics_a = evaluate_scores(scores_a, gt_mask, threshold=best_t_a)
    elapsed_a = time.time() - t0

    print(f"  [A] AUC-PR={metrics_a['auc_pr']:.4f} AUC-ROC={metrics_a['auc_roc']:.4f} "
          f"F1={metrics_a['f1']:.4f} [{elapsed_a:.1f}s]")
    all_results["A"] = {
        "metrics": metrics_a,
        "scores": scores_a.tolist(),
        "threshold": float(best_t_a),
        "elapsed": elapsed_a,
    }

    np.save(f"{OUTPUT}/setup_A_scores.npy", scores_a)
    np.save(f"{OUTPUT}/gt_mask.npy", gt_mask)
    np.save(f"{OUTPUT}/setup_A_detected.npy", (scores_a >= best_t_a).astype(np.int32))

    # =====================================================================
    # SETUP B: Denoise Autoencoder (noise, no FE, raw 7D)
    # =====================================================================
    print("\n[Setup B] Denoise Autoencoder (raw features + noise)")
    t0 = time.time()

    noise_std_b = best_cfg.get("noise_std", 0.001)
    out_dim_b = d7 * 2

    n_init_b = min(best_cfg.get("memory_len", 1024), len(X_train_7))
    idx_b = np.random.permutation(len(X_train_7))[:n_init_b]
    X_init_b = X_train_7[idx_b].astype(np.float32)
    mean_b = X_init_b.mean(axis=0).astype(np.float32)
    std_b = np.clip(X_init_b.std(axis=0), 1e-6, None).astype(np.float32)
    Xn_b = (X_init_b - mean_b) / (std_b + 1e-8)
    Xn_b_t = torch.from_numpy(Xn_b).to(DEVICE)

    ae_b = MemStreamAE(d=d7, out_dim=out_dim_b).to(DEVICE)
    opt_b = torch.optim.Adam(ae_b.parameters(), lr=lr_a, betas=(0.9, 0.999))
    crit_b = torch.nn.MSELoss()

    ae_b.train()
    for ep in range(EPOCHS):
        perm_b = torch.randperm(len(Xn_b_t), device=DEVICE)
        for i in range(0, len(Xn_b_t), 1024):
            b = Xn_b_t[perm_b[i : i + 1024]]
            noisy_b = b + torch.randn_like(b) * noise_std_b
            loss = crit_b(ae_b(noisy_b), b)
            opt_b.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_b.parameters(), 1.0)
            opt_b.step()
        if (ep + 1) % 1000 == 0:
            print(f"  B ep {ep+1}/{EPOCHS}")

    ae_b.eval()
    memory_b = Memory(best_cfg.get("memory_len", 1024), out_dim_b)
    with torch.no_grad():
        enc_b = ae_b.encode(Xn_b_t).cpu().numpy()
        for i in range(n_init_b):
            memory_b.update(enc_b[i], enc_b[i])

    ae_state_b = {
        "input_dim": d7,
        "out_dim": out_dim_b,
        "encoder.weight": ae_b.encoder.weight.detach().cpu(),
        "encoder.bias": ae_b.encoder.bias.detach().cpu(),
        "decoder.weight": ae_b.decoder.weight.detach().cpu(),
        "decoder.bias": ae_b.decoder.bias.detach().cpu(),
    }
    scorer_b = PaperScorer(
        ae_state=ae_state_b,
        mean=mean_b,
        std=std_b,
        mem_M=memory_b.active(),
        k=best_cfg.get("k", 5),
        gamma=best_cfg.get("gamma", 0.0),
    )
    Xn_val_b = (X_test_7.astype(np.float32) - mean_b) / (std_b + 1e-8)
    scores_b = scorer_b.score_batch(Xn_val_b)
    scores_b = (scores_b - scores_b.min()) / (scores_b.max() - scores_b.min() + 1e-8)
    best_t_b, _ = find_best_threshold(scores_b, gt_mask)
    metrics_b = evaluate_scores(scores_b, gt_mask, threshold=best_t_b)
    elapsed_b = time.time() - t0

    print(f"  [B] AUC-PR={metrics_b['auc_pr']:.4f} AUC-ROC={metrics_b['auc_roc']:.4f} "
          f"F1={metrics_b['f1']:.4f} [{elapsed_b:.1f}s]")
    all_results["B"] = {
        "metrics": metrics_b,
        "scores": scores_b.tolist(),
        "threshold": float(best_t_b),
        "elapsed": elapsed_b,
    }
    np.save(f"{OUTPUT}/setup_B_scores.npy", scores_b)
    np.save(f"{OUTPUT}/setup_B_detected.npy", (scores_b >= best_t_b).astype(np.int32))

    # =====================================================================
    # SETUP C: Denoise AE + Feature Engineering (streaming memory, no CB)
    # =====================================================================
    print("\n[Setup C] Denoise AE + Feature Engineering (streaming memory)")
    t0 = time.time()

    pipe_c = MemStreamPipeline(
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
    pipe_c.train(
        X_train_34,
        hours_train,
        dows_train,
        rcs_train,
        nb_train,
        X_warmup=X_train_34[: best_cfg["memory_len"]],
        hours_warmup=hours_train[: best_cfg["memory_len"]],
        dows_warmup=dows_train[: best_cfg["memory_len"]],
        rcs_warmup=rcs_train[: best_cfg["memory_len"]],
        nb_warmup=nb_train[: best_cfg["memory_len"]],
    )

    adj_scores_c, _ = pipe_c.score_stream(
        X_test_34,
        hours_test,
        dows_test,
        rcs_test,
        nb_test,
        gt_mask=gt_mask,
        update_memory=True,
    )
    best_t_c, _ = find_best_threshold(adj_scores_c, gt_mask)
    metrics_c = evaluate_scores(adj_scores_c, gt_mask, threshold=best_t_c)
    elapsed_c = time.time() - t0
    epoch_losses_c = list(pipe_c.epoch_losses)

    print(f"  [C] AUC-PR={metrics_c['auc_pr']:.4f} AUC-ROC={metrics_c['auc_roc']:.4f} "
          f"F1={metrics_c['f1']:.4f} [{elapsed_c:.1f}s]")
    all_results["C"] = {
        "metrics": metrics_c,
        "scores": adj_scores_c.tolist(),
        "threshold": float(best_t_c),
        "elapsed": elapsed_c,
        "epoch_losses": [float(x) for x in epoch_losses_c],
    }
    np.save(f"{OUTPUT}/setup_C_scores.npy", adj_scores_c)
    np.save(f"{OUTPUT}/setup_C_detected.npy", (adj_scores_c >= best_t_c).astype(np.int32))
    np.save(f"{OUTPUT}/setup_C_epoch_losses.npy", np.array(epoch_losses_c, dtype=np.float32))

    # =====================================================================
    # SETUP D: Denoise AE + FE + ContextBeta (full MemStream)
    # =====================================================================
    print("\n[Setup D] Denoise AE + FE + ContextBeta (full MemStream)")
    t0 = time.time()

    pipe_d = MemStreamPipeline(
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
    pipe_d.train(
        X_train_34,
        hours_train,
        dows_train,
        rcs_train,
        nb_train,
        X_warmup=X_train_34[: best_cfg["memory_len"]],
        hours_warmup=hours_train[: best_cfg["memory_len"]],
        dows_warmup=dows_train[: best_cfg["memory_len"]],
        rcs_warmup=rcs_train[: best_cfg["memory_len"]],
        nb_warmup=nb_train[: best_cfg["memory_len"]],
    )

    cb_d = ContextBeta(db=best_cfg["beta"], pct=95.0)
    wn_d = min(4096, len(X_train_34))
    Xw_d = X_train_34[:wn_d]
    hw_d = hours_train[:wn_d]
    dw_d = dows_train[:wn_d]
    rcw_d = rcs_train[:wn_d]
    nbw_d = nb_train[:wn_d]
    warmup_scores_d = np.empty(wn_d, dtype=np.float64)
    for i in range(wn_d):
        warmup_scores_d[i] = pipe_d.scorer.score_point(Xw_d[i])
    for i in range(wn_d):
        ctx_d = compute_context_cell_id(int(hw_d[i]), int(dw_d[i]), float(rcw_d[i]))
        cb_d.rec(int(nbw_d[i]), ctx_d, warmup_scores_d[i])
    cb_d.fit()
    pipe_d.cb = cb_d

    adj_scores_d, ms_d = pipe_d.score_stream(
        X_test_34,
        hours_test,
        dows_test,
        rcs_test,
        nb_test,
        gt_mask=gt_mask,
        update_memory=True,
    )
    best_t_d, _ = find_best_threshold(adj_scores_d, gt_mask)
    metrics_d = evaluate_scores(adj_scores_d, gt_mask, threshold=best_t_d)
    elapsed_d = time.time() - t0
    epoch_losses_d = list(pipe_d.epoch_losses)

    print(f"  [D] AUC-PR={metrics_d['auc_pr']:.4f} AUC-ROC={metrics_d['auc_roc']:.4f} "
          f"F1={metrics_d['f1']:.4f} [{elapsed_d:.1f}s]")
    all_results["D"] = {
        "metrics": metrics_d,
        "scores": adj_scores_d.tolist(),
        "threshold": float(best_t_d),
        "elapsed": elapsed_d,
        "epoch_losses": [float(x) for x in epoch_losses_d],
        "n_mem_updates": ms_d.get("n_memory_updates", 0),
    }
    np.save(f"{OUTPUT}/setup_D_scores.npy", adj_scores_d)
    np.save(f"{OUTPUT}/setup_D_detected.npy", (adj_scores_d >= best_t_d).astype(np.int32))
    np.save(f"{OUTPUT}/setup_D_epoch_losses.npy", np.array(epoch_losses_d, dtype=np.float32))

    # =====================================================================
    # DELTA ANALYSIS & SUMMARY
    # =====================================================================
    print()
    print("=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Setup':<8} {'AUC-PR':>8} {'AUC-ROC':>9} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Sep':>8} {'Time':>8}")
    print("-" * 70)

    summary_rows = []
    for label, desc in [
        ("A", "Normal AE"),
        ("B", "Denoise AE"),
        ("C", "AE+FE+Mem"),
        ("D", "AE+FE+CB"),
    ]:
        if label in all_results:
            m = all_results[label]["metrics"]
            t = all_results[label]["elapsed"]
            print(
                f"{label:<8} {m['auc_pr']:>8.4f} {m['auc_roc']:>9.4f} {m['f1']:>8.4f} "
                f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['separation_ratio']:>8.4f} {t:>7.1f}s"
            )
            summary_rows.append(
                {
                    "setup": label,
                    "desc": desc,
                    **m,
                    "elapsed": t,
                    "threshold": all_results[label]["threshold"],
                }
            )

    print()
    print("DELTA ANALYSIS:")
    deltas = {}
    if "A" in all_results and "B" in all_results:
        d = all_results["B"]["metrics"]["auc_pr"] - all_results["A"]["metrics"]["auc_pr"]
        deltas["B_vs_A_noise"] = float(d)
        print(f"  B vs A (noise effect):              AUC-PR {d:+.4f}")
    if "B" in all_results and "C" in all_results:
        d = all_results["C"]["metrics"]["auc_pr"] - all_results["B"]["metrics"]["auc_pr"]
        deltas["C_vs_B_FE"] = float(d)
        print(f"  C vs B (FE+memory effect):          AUC-PR {d:+.4f}")
    if "C" in all_results and "D" in all_results:
        d = all_results["D"]["metrics"]["auc_pr"] - all_results["C"]["metrics"]["auc_pr"]
        deltas["D_vs_C_CB"] = float(d)
        print(f"  D vs C (ContextBeta effect):         AUC-PR {d:+.4f}")
    if "A" in all_results and "D" in all_results:
        d = all_results["D"]["metrics"]["auc_pr"] - all_results["A"]["metrics"]["auc_pr"]
        deltas["D_vs_A_total"] = float(d)
        print(f"  D vs A (total effect):              AUC-PR {d:+.4f}")

    # Save JSON summary
    output = {
        "epochs": EPOCHS,
        "max_train": MAX_TRAIN,
        "max_val": MAX_VAL,
        "seed": SEED,
        "best_config": best_cfg,
        "summary": summary_rows,
        "deltas": deltas,
    }
    with open(f"{OUTPUT}/ablation_summary.json", "w") as f:
        json.dump(output, f, indent=2)
    print()
    print(f"Artifacts saved to {OUTPUT}/:")
    print("  setup_[A-D]_scores.npy")
    print("  setup_[A-D]_detected.npy")
    print("  setup_[C-D]_epoch_losses.npy")
    print("  gt_mask.npy")
    print("  ablation_summary.json")
    print()
    print("ALL OK - ABLATION COMPLETE")


if __name__ == "__main__":
    main()
