#!/usr/bin/env python3
"""
MemStream Architecture Ablation — Why NormalAE wins, and can we close the gap?

This script tests the core hypothesis:
  NormalAE (38->64->32->64->38, ReLU, 100K samples) >> MemStream (38->76->38, Tanh, 2K samples)
  The gap comes from: bottleneck > training size > activation > scoring mechanism.

Experiments:
  [1] Vary bottleneck dim (76, 38, 19, 10) with kNN scoring + memory (2K train)
  [2] Vary activation (ReLU vs Tanh) with best bottleneck + kNN + 2K train
  [3] Vary training size (2K vs 100K) with best bottleneck + kNN
  [4] Best bottleneck + 100K train + RECONSTRUCTION ERROR scoring (like NormalAE)
  [5] Best bottleneck + 100K train + kNN on larger memory (8K, 32K)
  [6] Paper-exact: 38->76->38 (D=2d) with Tanh + 100K train + kNN

All use valid_polluted for evaluation (consistent with grid search).
"""
from __future__ import annotations

import os, sys, json, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
LOGGER = logging.getLogger("memstream_ae_ablation")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.benchmark_core import (
    extract_features_from_parquet, evaluate_scores, find_best_threshold,
    Memory, compute_auc_roc, compute_auc_pr,
)


DEVICE = "cuda"  # fixed, no dynamic device detection needed here
import torch
import torch.nn as nn

DEVICE_T = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Generic AE with configurable architecture
# ---------------------------------------------------------------------------
class ConfigurableAE(nn.Module):
    """AE with configurable bottleneck dim and activation."""

    def __init__(self, d: int, bottleneck: int, activation: str = "relu"):
        super().__init__()
        self.d = d
        self.bottleneck = bottleneck
        self.encoder = nn.Linear(d, bottleneck)
        self.decoder = nn.Linear(bottleneck, d)
        self.activation = activation.lower()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.activation == "tanh":
            z = torch.tanh(z)
        elif self.activation == "relu":
            z = torch.relu(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class MemStreamAEWrapper(nn.Module):
    """Paper-exact MemStream AE: d->2d->d, tanh (no bottleneck)."""

    def __init__(self, d: int, out_dim: int):
        super().__init__()
        self.encoder = nn.Linear(d, out_dim)
        self.decoder = nn.Linear(out_dim, d)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


# ---------------------------------------------------------------------------
# Scoring modes
# ---------------------------------------------------------------------------
def score_by_reconstruction(
    model: nn.Module, X: np.ndarray,
    input_mean: np.ndarray, input_std: np.ndarray,
    device,
) -> np.ndarray:
    """Score by MSE reconstruction error (like NormalAE)."""
    model.eval()
    Xn = (X.astype(np.float32) - input_mean) / (input_std + 1e-8)
    Xn_t = torch.from_numpy(Xn).to(device)
    with torch.no_grad():
        recon = model(Xn_t)
        mse = torch.mean((recon - Xn_t) ** 2, dim=1).cpu().numpy()
    return mse


def score_by_knn(
    model: nn.Module, X: np.ndarray,
    input_mean: np.ndarray, input_std: np.ndarray,
    memory_M: np.ndarray,
    k: int, device,
) -> np.ndarray:
    """Score by kNN L1 distance in latent space (like MemStream)."""
    model.eval()
    Xn = (X.astype(np.float32) - input_mean) / (input_std + 1e-8)
    Xn_t = torch.from_numpy(Xn).to(device)

    with torch.no_grad():
        Z = model.encode(Xn_t).cpu().numpy()

    mem_t = torch.from_numpy(memory_M.astype(np.float32))
    n = len(Z)
    scores = np.zeros(n, dtype=np.float64)

    for s in range(0, n, 2000):
        e = min(s + 2000, n)
        chunk = torch.from_numpy(Z[s:e]).unsqueeze(1)
        diff = (chunk - mem_t.unsqueeze(0)).abs().sum(2).numpy()
        top_k = np.sort(diff, axis=1)[:, :k]
        scores[s:e] = top_k.sum(axis=1)

    return scores


# ---------------------------------------------------------------------------
# Run one experiment config
# ---------------------------------------------------------------------------
def run_experiment(
    X_train: np.ndarray,
    X_val: np.ndarray,
    gt_mask: np.ndarray,
    config: Dict,
) -> Dict:
    """Run a single ablation experiment."""
    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    bottleneck = config["bottleneck"]
    activation = config.get("activation", "relu")
    n_train_samples = config.get("n_train_samples", 2048)
    scoring = config.get("scoring", "knn")  # "knn" or "reconstruction"
    memory_len = config.get("memory_len", 2048)
    k = config.get("k", 10)
    epochs = config.get("epochs", 5000)
    lr = config.get("lr", 0.01)
    noise_std = config.get("noise_std", 0.001)
    batch_size = config.get("batch_size", 256)
    update_memory = config.get("update_memory", False)
    d = X_train.shape[1]

    t0 = time.time()

    # Sample training data
    n_use = min(n_train_samples, len(X_train))
    idx = np.random.permutation(len(X_train))[:n_use]
    X_tr = X_train[idx].astype(np.float32)

    # Normalize stats from training data
    input_mean = X_tr.mean(axis=0).astype(np.float32)
    input_std = np.clip(X_tr.std(axis=0), 1e-6, None).astype(np.float32)
    Xn_tr = (X_tr - input_mean) / (input_std + 1e-8)
    Xn_tr_t = torch.from_numpy(Xn_tr).to(DEVICE_T)

    # Build model
    if config.get("is_memstream_ae", False):
        model = MemStreamAEWrapper(d=d, out_dim=bottleneck).to(DEVICE_T)
    else:
        model = ConfigurableAE(d=d, bottleneck=bottleneck, activation=activation).to(DEVICE_T)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.MSELoss()

    # Training
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xn_tr_t), device=DEVICE_T)
        total_loss = 0.0
        for i in range(0, len(Xn_tr_t), batch_size):
            b = Xn_tr_t[perm[i:i + batch_size]]
            optimizer.zero_grad()
            noisy = b + torch.randn_like(b) * noise_std
            recon = model(noisy)
            loss = criterion(recon, b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if (ep + 1) % 1000 == 0:
            pass  # silent

    model.eval()

    # Initialize memory (for kNN scoring)
    memory = Memory(memory_len, bottleneck)
    with torch.no_grad():
        enc_tr = model.encode(Xn_tr_t).cpu().numpy()
        for i in range(min(n_use, memory_len)):
            memory.update(enc_tr[i], enc_tr[i])

    # Score validation
    if scoring == "reconstruction":
        scores = score_by_reconstruction(model, X_val, input_mean, input_std, DEVICE_T)
    else:  # knn
        scores = score_by_knn(
            model, X_val, input_mean, input_std,
            memory.active(), k, DEVICE_T,
        )

    # Streaming memory update
    if update_memory:
        Xn_val = (X_val.astype(np.float32) - input_mean) / (input_std + 1e-8)
        Xn_val_t = torch.from_numpy(Xn_val).to(DEVICE_T)
        with torch.no_grad():
            Z_val = model.encode(Xn_val_t).cpu().numpy()
        n_updates = 0
        beta_thres = np.percentile(scores, 95)
        for i in range(len(scores)):
            if scores[i] < beta_thres and memory.cnt > 0:
                memory.update(Z_val[i], Z_val[i])
                n_updates += 1
        n_memory_updates = n_updates
    else:
        n_memory_updates = 0

    elapsed = time.time() - t0

    # Normalize scores
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    best_t, best_f1 = find_best_threshold(scores, gt_mask)
    metrics = evaluate_scores(scores, gt_mask, threshold=best_t)

    LOGGER.info(
        "  [Exp %s] %s | bot=%3d | %-4s | n_train=%6d | %-13s | mem=%4d | AUC-PR=%.4f AUC-ROC=%.4f F1=%.4f | sep=%.2fx | [%.1fs]",
        config.get("exp_id", "?"),
        config.get("label", ""),
        bottleneck,
        activation.upper() if bottleneck != d * 2 else "TANH",
        n_use,
        f"kNN(k={k})" if scoring == "knn" else "MSERecon",
        memory_len,
        metrics["auc_pr"],
        metrics["auc_roc"],
        metrics["f1"],
        metrics["separation_ratio"],
        elapsed,
    )

    return {
        "exp_id": config.get("exp_id", "?"),
        "label": config.get("label", ""),
        "bottleneck": bottleneck,
        "activation": activation,
        "is_memstream_ae": config.get("is_memstream_ae", False),
        "n_train_samples": n_use,
        "scoring": scoring,
        "memory_len": memory_len,
        "k": k,
        "epochs": epochs,
        "update_memory": update_memory,
        "auc_pr": metrics["auc_pr"],
        "auc_roc": metrics["auc_roc"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "best_threshold": float(best_t),
        "score_normal_mean": float(metrics["score_normal_mean"]),
        "score_anomaly_mean": float(metrics["score_anomaly_mean"]),
        "separation_ratio": float(metrics["separation_ratio"]),
        "time_s": elapsed,
        "n_memory_updates": n_memory_updates,
    }


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
def build_experiments() -> List[Dict]:
    """Define all ablation experiments."""
    configs = []

    # Baseline: Current MemStream (for reference)
    configs.append({
        "exp_id": "A0", "label": "Current MemStream",
        "bottleneck": 76, "activation": "tanh", "is_memstream_ae": True,
        "n_train_samples": 2048, "scoring": "knn", "memory_len": 2048, "k": 10,
        "epochs": 5000, "seed": 42,
    })

    # [1] Bottleneck sweep with kNN (2K train)
    for bot in [38, 19, 10, 5]:
        configs.append({
            "exp_id": f"B{bot}", f"label": f"Bot{bot}_ReLU_kNN",
            "bottleneck": bot, "activation": "relu",
            "n_train_samples": 2048, "scoring": "knn", "memory_len": 2048, "k": 10,
            "epochs": 5000, "seed": 42,
        })

    # [2] Activation comparison (best bottleneck)
    for act in ["tanh", "relu"]:
        configs.append({
            "exp_id": f"C{act[:1]}", f"label": f"Bot19_{act.upper()}_kNN",
            "bottleneck": 19, "activation": act,
            "n_train_samples": 2048, "scoring": "knn", "memory_len": 2048, "k": 10,
            "epochs": 5000, "seed": 42,
        })

    # [3] Training size (with best bottleneck 19)
    for n_tr in [8192, 32768, 100000]:
        configs.append({
            "exp_id": f"D{n_tr}", f"label": f"Bot19_ReLU_n{n_tr}_kNN",
            "bottleneck": 19, "activation": "relu",
            "n_train_samples": n_tr, "scoring": "knn", "memory_len": 2048, "k": 10,
            "epochs": 5000 if n_tr <= 32768 else 100, "seed": 42,
            "batch_size": 1024 if n_tr > 10000 else 256,
        })

    # [4] Scoring: reconstruction error (like NormalAE)
    for bot in [19, 32]:
        configs.append({
            "exp_id": f"ER{bot}", f"label": f"Bot{bot}_ReLU_MSERecon",
            "bottleneck": bot, "activation": "relu",
            "n_train_samples": 100000, "scoring": "reconstruction",
            "epochs": 100, "seed": 42, "batch_size": 1024,
        })

    # [5] Larger memory + kNN (100K train)
    for mem_len in [8192, 32768]:
        for k in [5, 10, 20]:
            configs.append({
                "exp_id": f"F{mem_len}k{k}", f"label": f"Bot19_ReLU_n100K_mem{mem_len}_k{k}",
                "bottleneck": 19, "activation": "relu",
                "n_train_samples": 100000, "scoring": "knn", "memory_len": mem_len, "k": k,
                "epochs": 100, "seed": 42, "batch_size": 1024,
            })

    # [6] Paper D=2d but 100K train + kNN (fair comparison)
    configs.append({
        "exp_id": "G76", "label": "Bot76_ReLU_n100K_kNN",
        "bottleneck": 76, "activation": "relu",
        "n_train_samples": 100000, "scoring": "knn", "memory_len": 8192, "k": 10,
        "epochs": 100, "seed": 42, "batch_size": 1024,
    })

    # [7] 38->19->38 with streaming (update_memory=True)
    configs.append({
        "exp_id": "H", "label": "Bot19_ReLU_n100K_kNN_stream",
        "bottleneck": 19, "activation": "relu",
        "n_train_samples": 100000, "scoring": "knn", "memory_len": 8192, "k": 10,
        "epochs": 100, "seed": 42, "batch_size": 1024,
        "update_memory": True,
    })

    # Fix label keys
    label_map = {
        "A0": "Current (76D exp + tanh + 2K + kNN)",
        "B38": "Bot38 + ReLU + 2K + kNN",
        "B19": "Bot19 + ReLU + 2K + kNN",
        "B10": "Bot10 + ReLU + 2K + kNN",
        "B5": "Bot5 + ReLU + 2K + kNN",
        "Ct": "Bot19 + TANH + 2K + kNN",
        "Cr": "Bot19 + ReLU + 2K + kNN",
        "D8192": "Bot19 + ReLU + 8K + kNN",
        "D32768": "Bot19 + ReLU + 32K + kNN",
        "D100000": "Bot19 + ReLU + 100K + kNN",
        "ER19": "Bot19 + ReLU + 100K + MSERecon (like NormalAE)",
        "ER32": "Bot32 + ReLU + 100K + MSERecon (like NormalAE)",
        "F8192k5": "Bot19 + ReLU + 100K + mem8K + k5",
        "F8192k10": "Bot19 + ReLU + 100K + mem8K + k10",
        "F8192k20": "Bot19 + ReLU + 100K + mem8K + k20",
        "F32768k10": "Bot19 + ReLU + 100K + mem32K + k10",
        "G76": "Bot76 + ReLU + 100K + mem8K + k10",
        "H": "Bot19 + ReLU + 100K + mem8K + k10 + STREAM",
    }
    for cfg in configs:
        cfg["label"] = label_map.get(cfg["exp_id"], cfg.get("label", ""))

    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(Path(__file__).parent.parent))
    parser.add_argument("--max-train", type=int, default=500000)
    parser.add_argument("--max-val", type=int, default=500000)
    parser.add_argument("--exp", default="all",
                       help="Comma-separated exp IDs to run, or 'all'")
    args = parser.parse_args()

    base = Path(args.base)
    data_dir = base / "data"
    output_dir = base / "results" / "memstream_ae_ablation"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    train_path = str(data_dir / "train_clean.parquet")
    valid_path = str(data_dir / "valid_polluted.parquet")

    LOGGER.info("Loading data...")
    X_train, *_ = extract_features_from_parquet(train_path, max_rows=args.max_train)
    X_val, *_ = extract_features_from_parquet(valid_path, max_rows=args.max_val)
    gt_mask = np.load(str(data_dir / "valid" / "ground_truth_mask.npy"))
    gt_mask = gt_mask[-len(X_val):]
    LOGGER.info("Train: %d, Val: %d (GT: %d anomalies, %.2f%%)",
               len(X_train), len(X_val), gt_mask.sum(), gt_mask.mean() * 100)

    # Build experiments
    all_configs = build_experiments()

    exp_filter = args.exp.lower()
    if exp_filter != "all":
        allowed = set(e.strip() for e in exp_filter.split(","))
        all_configs = [c for c in all_configs if c["exp_id"] in allowed]

    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("MEMSTREAM AE ABLATION (%d experiments)", len(all_configs))
    LOGGER.info("=" * 80)
    LOGGER.info(f"{'ID':<4} {'Label':<50} {'Bot':>4} {'Act':>5} {'N_train':>8} {'Scoring':>13} "
               f"{'Mem':>5} {'AUC-PR':>8} {'AUC-ROC':>8} {'F1':>7}")
    LOGGER.info("-" * 80)

    results = []
    for cfg in all_configs:
        try:
            result = run_experiment(X_train, X_val, gt_mask, cfg)
            results.append(result)
        except Exception as e:
            LOGGER.error("  [Exp %s] FAILED: %s", cfg.get("exp_id", "?"), e)
            results.append({**cfg, "error": str(e), "auc_pr": 0, "auc_roc": 0, "f1": 0})

    # Sort by AUC-PR
    results.sort(key=lambda r: r.get("auc_pr", 0), reverse=True)

    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("FINAL RANKING (by AUC-PR)")
    LOGGER.info("=" * 80)
    LOGGER.info(f"{'Rank':<5} {'ID':<4} {'AUC-PR':>8} {'AUC-ROC':>8} {'F1':>8} {'Sep':>8} {'Label'}")
    LOGGER.info("-" * 80)
    for rank, r in enumerate(results, 1):
        label = r.get("label", r.get("exp_id", "?"))[:50]
        LOGGER.info(f"{rank:<5} {r.get('exp_id','?'):<4} {r.get('auc_pr',0):>8.4f} "
                   f"{r.get('auc_roc',0):>8.4f} {r.get('f1',0):>8.4f} "
                   f"{r.get('separation_ratio',0):>8.2f}  {label}")

    # Save results
    out_path = output_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    LOGGER.info("")
    LOGGER.info("Results saved: %s", out_path)

    # Print key insight
    best = results[0] if results else {}
    baseline = next((r for r in results if r.get("exp_id") == "A0"), None)
    LOGGER.info("")
    LOGGER.info("KEY INSIGHTS:")
    LOGGER.info("  Best AUC-PR: %.4f (%s)", best.get("auc_pr", 0), best.get("label", ""))
    if baseline:
        LOGGER.info("  Baseline AUC-PR: %.4f (Current MemStream)", baseline.get("auc_pr", 0))
        LOGGER.info("  Improvement: +%.4f", best.get("auc_pr", 0) - baseline.get("auc_pr", 0))
    LOGGER.info("  NormalAE reference: AUC-PR=0.9616")


if __name__ == "__main__":
    main()
