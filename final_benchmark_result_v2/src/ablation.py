#!/usr/bin/env python3
"""
================================================================================
STEP 2: ABLATION STUDY — Analyze Component Contributions (File 2/3)
================================================================================

4 setups:
  A. Normal Autoencoder         (train AE on raw features, no memory, no denoise noise)
  B. Denoise Autoencoder         (train AE with noise, raw features, no feature engineering)
  C. Denoise AE + Feature Eng.  (train AE with noise + feature engineering, streaming memory)
  D. Denoise AE + FE + CB       (train AE with noise + FE + ContextBeta, streaming memory)

Dùng best config từ grid search làm base.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
LOGGER = logging.getLogger("ablation")

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark_core import (
    MemStreamPipeline, extract_features_from_parquet, extract_raw_features,
    evaluate_scores, find_best_threshold,
    compute_context_cell_id, DEVICE,
)


# ---------------------------------------------------------------------------
# Setup A: Normal Autoencoder (no noise, no memory, raw features)
# ---------------------------------------------------------------------------
def setup_A(
    X_train: np.ndarray,
    X_val: np.ndarray,
    gt_mask_val: np.ndarray,
    base_config: Dict,
    epochs: int = 500,
    verbose: bool = True,
) -> Tuple[Dict, Optional[object]]:
    """
    Normal Autoencoder: train on raw features, no noise injection, no memory.
    Baseline: just learn reconstruction error on training data.
    """
    if verbose:
        LOGGER.info("  [Setup A] Normal Autoencoder (no noise, no memory)")

    d = X_train.shape[1]
    out_dim = d * 2
    noise_std = 0.0  # no noise
    lr = base_config.get("lr", 0.01)
    seed = base_config.get("seed", 42)

    np.random.seed(seed)
    torch.manual_seed(seed)

    n_train = min(base_config.get("memory_len", 1024), len(X_train))
    idx = np.random.permutation(len(X_train))[:n_train]
    X_init = X_train[idx].astype(np.float32)

    mean = X_init.mean(axis=0).astype(np.float32)
    std = np.clip(X_init.std(axis=0), 1e-6, None).astype(np.float32)
    Xn = (X_init - mean) / (std + 1e-8)
    Xn_t = torch.from_numpy(Xn).to(DEVICE)

    from benchmark_core import MemStreamAE
    ae = MemStreamAE(d=d, out_dim=out_dim).to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=lr, betas=(0.9, 0.999))
    crit = torch.nn.MSELoss()

    ae.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xn_t), device=DEVICE)
        for i in range(0, len(Xn_t), 1024):
            b = Xn_t[perm[i: i + 1024]]
            loss = crit(ae(b), b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()
    ae.eval()

    # Score on test data
    Xn_val = (X_val.astype(np.float32) - mean) / (std + 1e-8)
    Xn_val_t = torch.from_numpy(Xn_val).to(DEVICE)
    with torch.no_grad():
        reconstructed = ae(Xn_val_t).cpu().numpy()
    scores = np.mean((Xn_val - reconstructed) ** 2, axis=1)

    best_t, _ = find_best_threshold(scores, gt_mask_val)
    metrics = evaluate_scores(scores, gt_mask_val, threshold=best_t)
    metrics["setup"] = "A"
    metrics["description"] = "Normal Autoencoder (no noise, no memory)"
    metrics["config"] = {"d": d, "out_dim": out_dim, "epochs": epochs, "lr": lr}
    return metrics, None


# ---------------------------------------------------------------------------
# Setup B: Denoise AE (raw features, no feature engineering)
# ---------------------------------------------------------------------------
def setup_B(
    X_train: np.ndarray,
    X_val: np.ndarray,
    gt_mask_val: np.ndarray,
    base_config: Dict,
    epochs: int = 500,
    verbose: bool = True,
) -> Tuple[Dict, Optional[object]]:
    """
    Denoise Autoencoder: train on raw features WITH noise injection.
    No feature engineering, no streaming memory.
    """
    if verbose:
        LOGGER.info("  [Setup B] Denoise Autoencoder (raw features, no FE)")

    d = X_train.shape[1]
    out_dim = d * 2
    noise_std = base_config.get("noise_std", 0.001)
    lr = base_config.get("lr", 0.01)
    k = base_config.get("k", 5)
    gamma = base_config.get("gamma", 0.0)
    memory_len = base_config.get("memory_len", 1024)
    seed = base_config.get("seed", 42)

    np.random.seed(seed)
    torch.manual_seed(seed)

    n_train = min(memory_len, len(X_train))
    idx = np.random.permutation(len(X_train))[:n_train]
    X_init = X_train[idx].astype(np.float32)

    mean = X_init.mean(axis=0).astype(np.float32)
    std = np.clip(X_init.std(axis=0), 1e-6, None).astype(np.float32)
    Xn = (X_init - mean) / (std + 1e-8)
    Xn_t = torch.from_numpy(Xn).to(DEVICE)

    from benchmark_core import MemStreamAE, Memory, PaperScorer
    ae = MemStreamAE(d=d, out_dim=out_dim).to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=lr, betas=(0.9, 0.999))
    crit = torch.nn.MSELoss()

    ae.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xn_t), device=DEVICE)
        for i in range(0, len(Xn_t), 1024):
            b = Xn_t[perm[i: i + 1024]]
            noisy_b = b + torch.randn_like(b) * noise_std
            loss = crit(ae(noisy_b), b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()
    ae.eval()

    # Init memory with encoded training samples
    memory = Memory(memory_len, out_dim)
    with torch.no_grad():
        enc = ae.encode(Xn_t).cpu().numpy()
        for i in range(n_train):
            memory.update(enc[i], enc[i])

    # Score test data using AE + kNN on memory
    ae_state = {
        "input_dim": d,
        "out_dim": out_dim,
        "encoder.weight": ae.encoder.weight.detach().cpu(),
        "encoder.bias": ae.encoder.bias.detach().cpu(),
        "decoder.weight": ae.decoder.weight.detach().cpu(),
        "decoder.bias": ae.decoder.bias.detach().cpu(),
    }
    scorer = PaperScorer(
        ae_state=ae_state,
        mean=mean,
        std=std,
        mem_M=memory.active(),
        k=k,
        gamma=gamma,
    )

    # Batch scoring
    Xn_val = (X_val.astype(np.float32) - mean) / (std + 1e-8)
    scores = scorer.score_batch(Xn_val)

    best_t, _ = find_best_threshold(scores, gt_mask_val)
    metrics = evaluate_scores(scores, gt_mask_val, threshold=best_t)
    metrics["setup"] = "B"
    metrics["description"] = "Denoise Autoencoder (raw features)"
    metrics["config"] = {"d": d, "out_dim": out_dim, "noise_std": noise_std,
                         "epochs": epochs, "lr": lr, "k": k}
    return metrics, None


# ---------------------------------------------------------------------------
# Setup C: Denoise AE + Feature Engineering (streaming memory)
# ---------------------------------------------------------------------------
def setup_C(
    X_train: np.ndarray,
    hours_train: np.ndarray,
    dows_train: np.ndarray,
    rcs_train: np.ndarray,
    nb_train: np.ndarray,
    X_val: np.ndarray,
    hours_val: np.ndarray,
    dows_val: np.ndarray,
    rcs_val: np.ndarray,
    nb_val: np.ndarray,
    gt_mask_val: np.ndarray,
    base_config: Dict,
    epochs: int = 500,
    verbose: bool = True,
) -> Tuple[Dict, MemStreamPipeline]:
    """
    Denoise AE + Feature Engineering + Streaming Memory (no ContextBeta).
    """
    if verbose:
        LOGGER.info("  [Setup C] Denoise AE + Feature Engineering (streaming memory)")

    pipeline = MemStreamPipeline(
        d=38, out_dim=base_config.get("out_dim", 38),
        memory_len=base_config["memory_len"],
        k=base_config["k"],
        gamma=base_config["gamma"],
        beta=base_config["beta"],
        noise_std=base_config["noise_std"],
        lr=base_config["lr"],
        epochs=epochs,
        batch_size=1024,
        seed=42,
        cb_warmup=min(4096, base_config["memory_len"] * 4),
        verbose=False,
    )
    pipeline.train(
        X_train, hours_train, dows_train, rcs_train, nb_train,
        X_warmup=X_train[:base_config["memory_len"]],
        hours_warmup=hours_train[:base_config["memory_len"]],
        dows_warmup=dows_train[:base_config["memory_len"]],
        rcs_warmup=rcs_train[:base_config["memory_len"]],
        nb_warmup=nb_train[:base_config["memory_len"]],
    )

    # Score with streaming memory update
    adj_scores, _ = pipeline.score_stream(
        X_val, hours_val, dows_val, rcs_val, nb_val,
        gt_mask=gt_mask_val,
        update_memory=True,
    )

    best_t, _ = find_best_threshold(adj_scores, gt_mask_val)
    metrics = evaluate_scores(adj_scores, gt_mask_val, threshold=best_t)
    metrics["setup"] = "C"
    metrics["description"] = "Denoise AE + Feature Engineering (streaming memory)"
    metrics["config"] = base_config
    return metrics, pipeline


# ---------------------------------------------------------------------------
# Setup D: Denoise AE + Feature Engineering + ContextBeta
# ---------------------------------------------------------------------------
def setup_D(
    X_train: np.ndarray,
    hours_train: np.ndarray,
    dows_train: np.ndarray,
    rcs_train: np.ndarray,
    nb_train: np.ndarray,
    X_val: np.ndarray,
    hours_val: np.ndarray,
    dows_val: np.ndarray,
    rcs_val: np.ndarray,
    nb_val: np.ndarray,
    gt_mask_val: np.ndarray,
    base_config: Dict,
    epochs: int = 500,
    verbose: bool = True,
) -> Tuple[Dict, MemStreamPipeline]:
    """
    Denoise AE + Feature Engineering + ContextBeta + Streaming Memory.
    Full MemStream pipeline with all components.
    """
    if verbose:
        LOGGER.info("  [Setup D] Denoise AE + FE + ContextBeta (streaming memory)")

    pipeline = MemStreamPipeline(
        d=38, out_dim=base_config.get("out_dim", 38),
        memory_len=base_config["memory_len"],
        k=base_config["k"],
        gamma=base_config["gamma"],
        beta=base_config["beta"],
        noise_std=base_config["noise_std"],
        lr=base_config["lr"],
        epochs=epochs,
        batch_size=1024,
        seed=42,
        cb_warmup=min(4096, base_config["memory_len"] * 4),
        verbose=False,
    )
    pipeline.train(
        X_train, hours_train, dows_train, rcs_train, nb_train,
        X_warmup=X_train[:base_config["memory_len"]],
        hours_warmup=hours_train[:base_config["memory_len"]],
        dows_warmup=dows_train[:base_config["memory_len"]],
        rcs_warmup=rcs_train[:base_config["memory_len"]],
        nb_warmup=nb_train[:base_config["memory_len"]],
    )

    # Warmup ContextBeta
    from benchmark_core import ContextBeta
    cb = ContextBeta(db=base_config["beta"], pct=95.0)
    wn = min(4096, len(X_train))
    Xw = X_train[:wn]
    hw = hours_train[:wn]
    dw = dows_train[:wn]
    rcw = rcs_train[:wn]
    nbw = nb_train[:wn]
    warmup_scores = np.empty(wn, dtype=np.float64)
    for i in range(wn):
        warmup_scores[i] = pipeline.scorer.score_point(Xw[i])
    for i in range(wn):
        ctx = compute_context_cell_id(int(hw[i]), int(dw[i]), float(rcw[i]))
        cb.rec(int(nbw[i]), ctx, warmup_scores[i])
    cb.fit()
    pipeline.cb = cb

    # Score with streaming memory update
    adj_scores, _ = pipeline.score_stream(
        X_val, hours_val, dows_val, rcs_val, nb_val,
        gt_mask=gt_mask_val,
        update_memory=True,
    )

    best_t, _ = find_best_threshold(adj_scores, gt_mask_val)
    metrics = evaluate_scores(adj_scores, gt_mask_val, threshold=best_t)
    metrics["setup"] = "D"
    metrics["description"] = "Denoise AE + Feature Engineering + ContextBeta"
    metrics["config"] = base_config
    return metrics, pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ablation Study - 4 setups: Normal AE, Denoise AE, Denoise AE+FE, Denoise AE+FE+CB"
    )
    parser.add_argument("--base", default=str(Path(__file__).parent.parent),
                       help="Base directory (final_benchmark_result_v2)")
    parser.add_argument("--grid-results", default="",
                       help="Path to grid_search final_results.json")
    parser.add_argument("--epochs", type=int, default=5000,
                       help="Training epochs (paper-recommended: 5000)")
    parser.add_argument("--max-train", type=int, default=200000,
                       help="Max training samples")
    parser.add_argument("--max-val", type=int, default=300000,
                       help="Max validation samples (test data, 0 = full dataset)")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs with different seeds")
    args = parser.parse_args()

    base = Path(args.base)
    data_dir = base / "data"
    train_path = str(data_dir / "train_clean.parquet")
    test_path = str(data_dir / "test_polluted.parquet")
    output_dir = str(base / "results" / "ablation")

    os.makedirs(output_dir, exist_ok=True)

    # Load best config
    if args.grid_results:
        grid_path = args.grid_results
    else:
        grid_path = str(base / "results" / "grid_search" / "final_results.json")

    if os.path.exists(grid_path):
        with open(grid_path) as f:
            grid_data = json.load(f)
        best_config = grid_data.get("best_config", {})
        LOGGER.info("Loaded best config from grid search: %s",
                   best_config.get("config_id", "N/A"))
    else:
        LOGGER.warning("No grid search results found, using default config")
        best_config = {
            "memory_len": 1024, "k": 5, "gamma": 0.0, "beta": 0.5,
            "noise_std": 0.001, "lr": 0.01, "out_dim": 38, "epochs": 500,
        }

    LOGGER.info("=" * 60)
    LOGGER.info("Ablation Study - 4 Setups")
    LOGGER.info("=" * 60)
    LOGGER.info("A: Normal Autoencoder (no noise, no memory)")
    LOGGER.info("B: Denoise Autoencoder (raw features, no FE)")
    LOGGER.info("C: Denoise AE + Feature Engineering (streaming memory)")
    LOGGER.info("D: Denoise AE + FE + ContextBeta (streaming memory)")
    LOGGER.info("Base config: mem=%d, k=%d, gamma=%.2f, beta=%.3f",
               best_config["memory_len"], best_config["k"],
               best_config["gamma"], best_config["beta"])
    LOGGER.info("Epochs: %d, Runs: %d", args.epochs, args.runs)
    LOGGER.info("Train: %d samples, Val: %d samples", args.max_train, args.max_val)
    LOGGER.info("Adam: betas=(%.3f, %.3f)",
               best_config.get("adam_betas", [0.9, 0.999])[0],
               best_config.get("adam_betas", [0.9, 0.999])[1])
    LOGGER.info("=" * 60)

    # Load data
    LOGGER.info("Loading data...")
    X_train_34, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
        train_path, max_rows=args.max_train
    )
    X_train_7, _, _, _, _ = extract_raw_features(
        train_path, max_rows=args.max_train
    )
    X_test_34, hours_test, dows_test, rcs_test, nb_test = extract_features_from_parquet(
        test_path, max_rows=args.max_val
    )
    X_test_7, _, _, _, _ = extract_raw_features(
        test_path, max_rows=args.max_val
    )
    gt_mask = np.load(str(data_dir / "test" / "ground_truth_mask.npy"))
    gt_mask = gt_mask[-len(X_test_34):]
    LOGGER.info("Train: %d rows (38D), %d rows (7D)", len(X_train_34), len(X_train_7))
    LOGGER.info("Test: %d rows (GT: %d anomalies)", len(X_test_34), gt_mask.sum())

    results_A, results_B, results_C, results_D = [], [], [], []

    for run in range(args.runs):
        seed = 42 + run * 111
        LOGGER.info("")
        LOGGER.info("=== Run %d/%d (seed=%d) ===", run + 1, args.runs, seed)
        cfg = {**best_config, "epochs": args.epochs, "seed": seed}

        # Setup A: Normal Autoencoder
        t0 = time.time()
        try:
            m_A, _ = setup_A(X_train_7, X_test_7, gt_mask, cfg, epochs=args.epochs)
            m_A["run"] = run
            m_A["seed"] = seed
            m_A["time_s"] = time.time() - t0
            results_A.append(m_A)
            LOGGER.info(
                "  [A] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f, sep=%.2fx [%.1fs]",
                m_A["auc_pr"], m_A["auc_roc"], m_A["f1"],
                m_A["separation_ratio"], m_A["time_s"]
            )
        except Exception as e:
            LOGGER.error("  [A] Error: %s", e)

        # Setup B: Denoise Autoencoder (raw)
        t0 = time.time()
        try:
            m_B, _ = setup_B(X_train_7, X_test_7, gt_mask, cfg, epochs=args.epochs)
            m_B["run"] = run
            m_B["seed"] = seed
            m_B["time_s"] = time.time() - t0
            results_B.append(m_B)
            LOGGER.info(
                "  [B] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f, sep=%.2fx [%.1fs]",
                m_B["auc_pr"], m_B["auc_roc"], m_B["f1"],
                m_B["separation_ratio"], m_B["time_s"]
            )
        except Exception as e:
            LOGGER.error("  [B] Error: %s", e)

        # Setup C: Denoise AE + FE (streaming)
        t0 = time.time()
        try:
            m_C, _ = setup_C(
                X_train_34, hours_train, dows_train, rcs_train, nb_train,
                X_test_34, hours_test, dows_test, rcs_test, nb_test, gt_mask,
                cfg, epochs=args.epochs,
            )
            m_C["run"] = run
            m_C["seed"] = seed
            m_C["time_s"] = time.time() - t0
            results_C.append(m_C)
            LOGGER.info(
                "  [C] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f, sep=%.2fx [%.1fs]",
                m_C["auc_pr"], m_C["auc_roc"], m_C["f1"],
                m_C["separation_ratio"], m_C["time_s"]
            )
        except Exception as e:
            LOGGER.error("  [C] Error: %s", e)

        # Setup D: Denoise AE + FE + CB
        t0 = time.time()
        try:
            m_D, _ = setup_D(
                X_train_34, hours_train, dows_train, rcs_train, nb_train,
                X_test_34, hours_test, dows_test, rcs_test, nb_test, gt_mask,
                cfg, epochs=args.epochs,
            )
            m_D["run"] = run
            m_D["seed"] = seed
            m_D["time_s"] = time.time() - t0
            results_D.append(m_D)
            LOGGER.info(
                "  [D] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f, sep=%.2fx [%.1fs]",
                m_D["auc_pr"], m_D["auc_roc"], m_D["f1"],
                m_D["separation_ratio"], m_D["time_s"]
            )
        except Exception as e:
            LOGGER.error("  [D] Error: %s", e)

    # Summary
    def summarize(results: List[Dict], name: str) -> Dict:
        if not results:
            return {}
        keys = ["auc_roc", "auc_pr", "f1", "precision", "recall", "separation_ratio"]
        summary = {"name": name, "n_runs": len(results)}
        for k in keys:
            vals = [r.get(k, 0) for r in results]
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))
            summary[f"{k}_min"] = float(np.min(vals))
            summary[f"{k}_max"] = float(np.max(vals))
        summary["time_mean"] = float(np.mean([r["time_s"] for r in results]))
        return summary

    summary_A = summarize(results_A, "Normal Autoencoder")
    summary_B = summarize(results_B, "Denoise AE (raw features)")
    summary_C = summarize(results_C, "Denoise AE + Feature Engineering")
    summary_D = summarize(results_D, "Denoise AE + FE + ContextBeta")

    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("ABLATION SUMMARY (%d runs)", args.runs)
    LOGGER.info("=" * 80)

    for s, label, desc in [
        (summary_A, "A", "Normal Autoencoder"),
        (summary_B, "B", "Denoise AE (raw)"),
        (summary_C, "C", "Denoise AE + FE"),
        (summary_D, "D", "Denoise AE + FE + CB"),
    ]:
        if not s:
            continue
        LOGGER.info("")
        LOGGER.info("[%s] %s", label, desc)
        LOGGER.info("  AUC-ROC:  %.4f ± %.4f  [%.4f - %.4f]",
                   s["auc_roc_mean"], s["auc_roc_std"],
                   s["auc_roc_min"], s["auc_roc_max"])
        LOGGER.info("  AUC-PR:   %.4f ± %.4f  [%.4f - %.4f]",
                   s["auc_pr_mean"], s["auc_pr_std"],
                   s["auc_pr_min"], s["auc_pr_max"])
        LOGGER.info("  F1:       %.4f ± %.4f  [%.4f - %.4f]",
                   s["f1_mean"], s["f1_std"],
                   s["f1_min"], s["f1_max"])
        LOGGER.info("  Precision: %.4f ± %.4f  [%.4f - %.4f]",
                   s["precision_mean"], s["precision_std"],
                   s["precision_min"], s["precision_max"])
        LOGGER.info("  Recall:   %.4f ± %.4f  [%.4f - %.4f]",
                   s["recall_mean"], s["recall_std"],
                   s["recall_min"], s["recall_max"])
        LOGGER.info("  Separation: %.4f ± %.4f  [%.4f - %.4f]",
                   s["separation_ratio_mean"], s["separation_ratio_std"],
                   s["separation_ratio_min"], s["separation_ratio_max"])

    # Delta analysis
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("Delta Analysis")
    LOGGER.info("=" * 60)
    pairs = [
        ("B vs A", summary_B, summary_A, "Effect of denoise noise"),
        ("C vs B", summary_C, summary_B, "Effect of feature engineering"),
        ("D vs C", summary_D, summary_C, "Effect of ContextBeta"),
        ("C vs A", summary_C, summary_A, "Cumulative: FE+streaming"),
        ("D vs A", summary_D, summary_A, "Cumulative: all components"),
    ]
    for label, s_new, s_old, desc in pairs:
        if s_new and s_old:
            LOGGER.info("")
            LOGGER.info("Delta %s (%s):", label, desc)
            for k in ["auc_roc", "auc_pr", "f1"]:
                delta = s_new[f"{k}_mean"] - s_old[f"{k}_mean"]
                LOGGER.info("  %s: %+.4f", k, delta)

    # Save
    output = {
        "setup_A": summary_A,
        "setup_B": summary_B,
        "setup_C": summary_C,
        "setup_D": summary_D,
        "all_runs": {
            "A": results_A, "B": results_B, "C": results_C, "D": results_D,
        },
        "base_config": best_config,
        "epochs": args.epochs,
        "n_runs": args.runs,
    }
    out_path = os.path.join(output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    LOGGER.info("")
    LOGGER.info("Results saved: %s", out_path)


if __name__ == "__main__":
    sys.exit(main())
