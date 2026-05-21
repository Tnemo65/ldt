#!/usr/bin/env python3
"""
================================================================================
STEP 2: ABLATION STUDY — Analyze Component Contributions (File 2/3)
================================================================================

So sanh 6 setup:
  A. No Feature Engineering (Raw 7D)
  B. Feature Engineering only (34D)
  C. Feature Engineering + ContextBeta (34D)
  D. PCA Feature Extraction (paper Table 6b)
  E. IB Feature Extraction (paper Table 6b)
  F. AE Feature Extraction (paper Table 6b, paper-default config)

Dung bo HP tot nhat tu grid search lam base config.

MUC TIEU:
  - Paper Table 6(b): Feature Extraction = Identity, PCA, IB, AE
  - Paper Table 6(a): Memory Update = None, LRU, RR, FIFO
  - Paper Table 6(c): Memory Length N
  - Paper Table 6(d): Output Dimension D
  - Paper Table 6(e): Update Threshold beta
  - Paper Table 6(f): KNN coefficient gamma

CO SO LY THUYET:
  Paper Section 4.3: Feature extraction la key cua MemStream
  Paper Section 4.4: ContextBeta giup adapt ve threshold theo context
  Paper Section 5.5: Ablation study cho tung component
"""
from __future__ import annotations

import os
import sys
import json
import time
import io
import pickle
import hashlib
import hmac
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

sys.path.insert(0, r"c:\proj\ldt\HP_benchmark_v4")
from benchmark_core import (
    MemStreamPipeline, extract_features_from_parquet, extract_raw_features,
    evaluate_scores, find_best_threshold,
    compute_context_cell_id, DEVICE, PCAScorer, IBScorer,
    ContextBeta, MemStreamAE, Memory, PaperScorer
)
from sklearn.decomposition import PCA as SkPCA


# ---------------------------------------------------------------------------
# Setup A: No Feature Engineering (Raw 7D)
# ---------------------------------------------------------------------------
class RawMemStreamPipeline:
    """MemStream voi raw 7D features (khong engineering)."""

    def __init__(
        self,
        d: int = 7,
        memory_len: int = 2048,
        k: int = 5,
        gamma: float = 0.0,
        beta: float = 0.5,
        noise_std: float = 0.001,
        lr: float = 0.01,
        epochs: int = 500,
        batch_size: int = 256,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.d = d
        self.out_dim = d * 2
        self.memory_len = memory_len
        self.k = k
        self.gamma = gamma
        self.beta = beta
        self.noise_std = noise_std
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

        import torch.nn as nn
        self.ae = MemStreamAE(d=d, out_dim=self.out_dim).to(DEVICE)
        self.opt = torch.optim.Adam(self.ae.parameters(), lr=lr,
                                   betas=(0.9, 0.999))  # paper Section 5
        self.crit = nn.MSELoss()
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.memory: Optional[Memory] = None
        self.scorer: Optional[PaperScorer] = None
        self._fitted = False

    def train(self, X_train: np.ndarray) -> "RawMemStreamPipeline":
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        n_train = min(self.memory_len, len(X_train))
        idx = np.random.permutation(len(X_train))[:n_train]
        X_init = X_train[idx].astype(np.float32)

        self.mean = X_init.mean(axis=0).astype(np.float32)
        self.std = np.clip(X_init.std(axis=0), 1e-6, None).astype(np.float32)
        Xn = (X_init - self.mean) / (self.std + 1e-8)
        Xn_t = torch.from_numpy(Xn).to(DEVICE)

        from torch.nn import MSELoss
        from benchmark_core import MemStreamAE, Memory
        self.ae = MemStreamAE(d=self.d, out_dim=self.out_dim).to(DEVICE)
        self.opt = torch.optim.Adam(
            self.ae.parameters(), lr=self.lr,
            betas=(0.9, 0.999),  # paper Section 5
        )
        self.crit = MSELoss()
        self.ae.train()
        for ep in range(self.epochs):
            perm = torch.randperm(len(Xn_t), device=DEVICE)
            for i in range(0, len(Xn_t), self.batch_size):
                b = Xn_t[perm[i: i + self.batch_size]]
                loss = self.crit(
                    self.ae(b + torch.randn_like(b) * self.noise_std), b
                )
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
                self.opt.step()
        self.ae.eval()

        self.memory = Memory(self.memory_len, self.out_dim)
        self._input_mean = self.mean
        self._input_std = self.std
        with torch.no_grad():
            enc = self.ae.encode(Xn_t).cpu().numpy()
            for i in range(n_train):
                self.memory.update(enc[i], enc[i])

        self._enc_mean = enc.mean(axis=0).astype(np.float32)
        self._enc_std = np.clip(enc.std(axis=0), 1e-6, None).astype(np.float32)
        self._build_scorer()
        self._fitted = True
        return self

    def _build_scorer(self) -> None:
        ae_state = {
            "input_dim": self.d,
            "out_dim": self.out_dim,
            "encoder.weight": self.ae.encoder.weight.detach().cpu(),
            "encoder.bias": self.ae.encoder.bias.detach().cpu(),
            "decoder.weight": self.ae.decoder.weight.detach().cpu(),
            "decoder.bias": self.ae.decoder.bias.detach().cpu(),
        }
        self.scorer = PaperScorer(
            ae_state=ae_state,
            mean=self._enc_mean,
            std=self._enc_std,
            mem_M=self.memory.active(),
            k=self.k,
            gamma=self.gamma,
            input_mean=self._input_mean,
            input_std=self._input_std,
        )

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.scorer.score_batch(X)


# ---------------------------------------------------------------------------
# Setup B: Feature Engineering only (34D, no ContextBeta)
# ---------------------------------------------------------------------------
def setup_B(
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
    """Setup B: Feature Engineering only (34D), no ContextBeta."""
    if verbose:
        LOGGER.info("  [Setup B] Feature Engineering only (34D, no ContextBeta)")

    pipeline = MemStreamPipeline(
        d=34, out_dim=base_config.get("out_dim", 68),
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

    adj_scores, _ = pipeline.score_stream(
        X_val, hours_val, dows_val, rcs_val, nb_val,
        gt_mask=gt_mask_val,
    )

    best_t, best_f1 = find_best_threshold(adj_scores, gt_mask_val)
    metrics = evaluate_scores(adj_scores, gt_mask_val, threshold=best_t)
    metrics["setup"] = "B"
    metrics["description"] = "Feature Engineering (34D), no ContextBeta"
    metrics["config"] = base_config

    return metrics, pipeline


# ---------------------------------------------------------------------------
# Setup C: Feature Engineering + ContextBeta (34D)
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
    """Setup C: Feature Engineering + ContextBeta (34D)."""
    if verbose:
        LOGGER.info("  [Setup C] Feature Engineering + ContextBeta (34D)")

    pipeline = MemStreamPipeline(
        d=34, out_dim=base_config.get("out_dim", 68),
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

    # Warmup ContextBeta with per-point scoring (consistent with score_stream)
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

    # Streaming score + memory update per point
    adj_scores, _ = pipeline.score_stream(
        X_val, hours_val, dows_val, rcs_val, nb_val,
        gt_mask=gt_mask_val,
    )

    best_t, best_f1 = find_best_threshold(adj_scores, gt_mask_val)
    metrics = evaluate_scores(adj_scores, gt_mask_val, threshold=best_t)
    metrics["setup"] = "C"
    metrics["description"] = "Feature Engineering (34D) + ContextBeta"
    metrics["config"] = base_config

    return metrics, pipeline


# ---------------------------------------------------------------------------
# Setup D: PCA Feature Extraction (paper Table 6b)
# ---------------------------------------------------------------------------
def setup_D(
    X_train: np.ndarray,
    X_val: np.ndarray,
    gt_mask_val: np.ndarray,
    base_config: Dict,
    verbose: bool = True,
) -> Dict:
    """Setup D: PCA feature extraction (paper: out_dim=8, Table 6b)."""
    if verbose:
        LOGGER.info("  [Setup D] PCA Feature Extraction (d=8)")

    n_components = 8  # paper line 520: D=8 for PCA/IB
    k = base_config.get("k", 5)
    gamma = base_config.get("gamma", 0.0)
    beta = base_config.get("beta", 0.5)
    seed = base_config.get("seed", 42)

    # Fit PCA on training data
    mean = X_train.mean(axis=0).astype(np.float32)
    std = np.clip(X_train.std(axis=0), 1e-6, None).astype(np.float32)
    n_train = min(base_config.get("memory_len", 1024), len(X_train))
    idx = np.random.RandomState(seed).permutation(len(X_train))[:n_train]
    Xn_train = (X_train[idx] - mean) / (std + 1e-8)

    from sklearn.decomposition import PCA as _SkPCA
    pca = _SkPCA(n_components=n_components)
    pca.fit(Xn_train)

    # Init memory = PCA(training samples)
    mem_raw = pca.transform(Xn_train).astype(np.float32)
    mem_len = base_config.get("memory_len", 1024)
    mem_M = np.zeros((mem_len, n_components), dtype=np.float32)
    mem_raw_full = np.zeros((mem_len, X_train.shape[1]), dtype=np.float32)
    cnt = min(len(mem_raw), mem_len)
    mem_M[:cnt] = mem_raw[:cnt]
    mem_raw_full[:cnt] = X_train[idx[:cnt]]
    mem_idx = cnt  # current write position

    scorer = PCAScorer(
        n_components=n_components,
        mean=mean, std=std,
        mem_M=mem_M,
        k=k, gamma=gamma,
    )
    scorer.fit(X_train[idx])

    # Streaming scoring + memory update per point
    n = len(X_val)
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        raw = scorer.score_point(X_val[i])
        scores[i] = raw
        if raw < beta:
            z = pca.transform(
                ((X_val[i] - mean) / (std + 1e-8)).reshape(1, -1)
            ).flatten().astype(np.float32)
            if mem_idx < mem_len:
                scorer.mem_M[mem_idx] = z
                mem_raw_full[mem_idx] = X_val[i]
            else:
                # FIFO: overwrite oldest
                scorer.mem_M[:-1] = scorer.mem_M[1:]
                scorer.mem_M[-1] = z
                mem_raw_full[:-1] = mem_raw_full[1:]
                mem_raw_full[-1] = X_val[i]
            mem_idx += 1

    best_t, _ = find_best_threshold(scores, gt_mask_val)
    metrics = evaluate_scores(scores, gt_mask_val, threshold=best_t)
    metrics["setup"] = "D"
    metrics["description"] = "PCA Feature Extraction (D=8)"
    metrics["params"] = {"n_components": n_components, "k": k, "gamma": gamma}
    return metrics


# ---------------------------------------------------------------------------
# Setup E: Information Bottleneck (paper Table 6b)
# ---------------------------------------------------------------------------
def setup_E(
    X_train: np.ndarray,
    X_val: np.ndarray,
    gt_mask_val: np.ndarray,
    base_config: Dict,
    verbose: bool = True,
) -> Dict:
    """Setup E: IB feature extraction (paper: out_dim=8, beta=0.5, Table 6b)."""
    if verbose:
        LOGGER.info("  [Setup E] Information Bottleneck (D=8, beta_ib=0.5)")

    out_dim = 8  # paper line 520: D=8 for PCA/IB
    beta_ib = 0.5  # paper line 518: beta=0.5
    k = base_config.get("k", 5)
    gamma = base_config.get("gamma", 0.0)
    seed = base_config.get("seed", 42)

    n_train = min(base_config.get("memory_len", 1024), len(X_train))
    idx = np.random.RandomState(seed).permutation(len(X_train))[:n_train]

    scorer = IBScorer(
        d=X_train.shape[1], out_dim=out_dim, hidden_dim=16,
        k=k, gamma=gamma, beta_ib=beta_ib,
        lr=0.001, epochs=500, seed=seed,
    )
    scorer.fit(X_train[idx])

    # Init memory
    mem_Z = scorer.transform(X_train[idx]).astype(np.float32)
    scorer.mem_M = mem_Z
    mem_len = mem_Z.shape[0]
    mem_idx = mem_len

    # Streaming scoring + memory update per point
    n = len(X_val)
    scores = np.empty(n, dtype=np.float64)
    beta_val = base_config.get("beta", 0.5)
    for i in range(n):
        raw = scorer.score_point(X_val[i])
        scores[i] = raw
        if raw < beta_val:
            z = scorer.transform(X_val[i].reshape(1, -1)).flatten().astype(np.float32)
            if mem_idx < mem_len:
                scorer.mem_M[mem_idx] = z
            else:
                # FIFO: overwrite oldest
                scorer.mem_M[:-1] = scorer.mem_M[1:]
                scorer.mem_M[-1] = z
            mem_idx += 1

    best_t, _ = find_best_threshold(scores, gt_mask_val)
    metrics = evaluate_scores(scores, gt_mask_val, threshold=best_t)
    metrics["setup"] = "E"
    metrics["description"] = "Information Bottleneck (D=8, beta=0.5)"
    metrics["params"] = {"out_dim": out_dim, "beta_ib": beta_ib, "k": k, "gamma": gamma}
    return metrics


# ---------------------------------------------------------------------------
# Setup F: AE Feature Extraction (paper Table 6b, full paper-default)
# ---------------------------------------------------------------------------
def setup_F(
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
    verbose: bool = True,
) -> Tuple[Dict, MemStreamPipeline]:
    """Setup F: AE feature extraction + ContextBeta (paper Table 6b, full paper-default)."""
    if verbose:
        LOGGER.info("  [Setup F] AE Feature Extraction + ContextBeta (paper-default)")

    pipeline = MemStreamPipeline(
        d=34, out_dim=base_config.get("out_dim", 68),
        memory_len=base_config["memory_len"],
        k=base_config["k"],
        gamma=base_config["gamma"],
        beta=base_config["beta"],
        noise_std=base_config["noise_std"],
        lr=base_config["lr"],
        epochs=base_config.get("epochs", 5000),
        batch_size=1024,
        seed=base_config.get("seed", 42),
        cb_warmup=min(4096, base_config["memory_len"] * 4),
        verbose=False,
        adam_betas=tuple(base_config.get("adam_betas", [0.9, 0.999])),
    )
    pipeline.train(
        X_train, hours_train, dows_train, rcs_train, nb_train,
        X_warmup=X_train[:base_config["memory_len"]],
        hours_warmup=hours_train[:base_config["memory_len"]],
        dows_warmup=dows_train[:base_config["memory_len"]],
        rcs_warmup=rcs_train[:base_config["memory_len"]],
        nb_warmup=nb_train[:base_config["memory_len"]],
    )

    # Warmup ContextBeta with per-point scoring (consistent with score_stream)
    cb = pipeline.cb
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

    # Streaming score + memory update per point
    adj_scores, _ = pipeline.score_stream(
        X_val, hours_val, dows_val, rcs_val, nb_val,
        gt_mask=gt_mask_val,
    )

    best_t, _ = find_best_threshold(adj_scores, gt_mask_val)
    metrics = evaluate_scores(adj_scores, gt_mask_val, threshold=best_t)
    metrics["setup"] = "F"
    metrics["description"] = "AE Feature Extraction + ContextBeta (paper-default)"
    metrics["config"] = base_config
    return metrics, pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ablation Study — MemStream v4")
    parser.add_argument("--base", default=r"c:\proj\ldt\HP_benchmark_v4",
                       help="Base directory (v4)")
    parser.add_argument("--grid-results", default="",
                       help="Path to grid_search final_results.json")
    parser.add_argument("--epochs", type=int, default=2000,
                       help="Training epochs")
    parser.add_argument("--max-train", type=int, default=200000,
                       help="Max training samples")
    parser.add_argument("--max-val", type=int, default=500000,
                       help="Max validation samples (0 = full dataset)")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs with different seeds")
    args = parser.parse_args()

    base = Path(args.base)
    v3 = base.parent / "HP_benchmark_v3"
    train_path = str(v3 / "train_clean.parquet")
    test_path = str(v3 / "test_polluted.parquet")
    output_dir = str(base / "results" / "ablation")

    os.makedirs(output_dir, exist_ok=True)

    # Load best config from grid search
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
            "noise_std": 0.001, "lr": 0.01, "out_dim": 68, "epochs": 500,
        }

    LOGGER.info("=" * 60)
    LOGGER.info("Ablation Study — 6 Setups")
    LOGGER.info("=" * 60)
    LOGGER.info("Setup A: Raw 7D (No Feature Engineering)")
    LOGGER.info("Setup B: 34D + AE (Feature Engineering only)")
    LOGGER.info("Setup C: 34D + AE + ContextBeta")
    LOGGER.info("Setup D: PCA Feature Extraction (D=8, paper Table 6b)")
    LOGGER.info("Setup E: IB Feature Extraction (D=8, beta=0.5, paper Table 6b)")
    LOGGER.info("Setup F: AE Feature Extraction + ContextBeta (paper-default)")
    LOGGER.info("Base config: mem=%d, k=%d, gamma=%.2f, beta=%.3f",
               best_config["memory_len"], best_config["k"],
               best_config["gamma"], best_config["beta"])
    LOGGER.info("Epochs: %d, Runs: %d (paper: 5 runs)", args.epochs, args.runs)
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
    gt_mask = np.load(str(v3 / "test" / "ground_truth_mask.npy"))
    gt_mask = gt_mask[-len(X_test_34):]
    LOGGER.info("Train: %d rows (34D), %d rows (7D)", len(X_train_34), len(X_train_7))
    LOGGER.info("Test: %d rows (GT: %d anomalies)", len(X_test_34), gt_mask.sum())

    # Multi-run results
    results_A = []
    results_B = []
    results_C = []
    results_D = []
    results_E = []
    results_F = []

    for run in range(args.runs):
        seed = 42 + run * 111
        LOGGER.info("")
        LOGGER.info("=== Run %d/%d (seed=%d) ===", run + 1, args.runs, seed)

        # Setup A: Raw 7D
        t0 = time.time()
        try:
            pipeline_A = RawMemStreamPipeline(
                d=7,
                memory_len=best_config["memory_len"],
                k=best_config["k"],
                gamma=best_config["gamma"],
                beta=best_config["beta"],
                noise_std=best_config["noise_std"],
                lr=best_config["lr"],
                epochs=args.epochs,
                batch_size=1024,
                seed=seed,
                verbose=False,
            )
            pipeline_A.train(X_train_7)
            scores_A = pipeline_A.score(X_test_7)
            best_t_A, _ = find_best_threshold(scores_A, gt_mask)
            m_A = evaluate_scores(scores_A, gt_mask, threshold=best_t_A)
            m_A["setup"] = "A"
            m_A["description"] = "No Feature Engineering (Raw 7D)"
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

        # Setup B: Feature Engineering only
        t0 = time.time()
        try:
            m_B, _ = setup_B(
                X_train_34, hours_train, dows_train, rcs_train, nb_train,
                X_test_34, hours_test, dows_test, rcs_test, nb_test, gt_mask,
                {**best_config, "epochs": args.epochs, "seed": seed},
                epochs=args.epochs, verbose=False
            )
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

        # Setup C: Feature Engineering + ContextBeta
        t0 = time.time()
        try:
            m_C, _ = setup_C(
                X_train_34, hours_train, dows_train, rcs_train, nb_train,
                X_test_34, hours_test, dows_test, rcs_test, nb_test, gt_mask,
                {**best_config, "epochs": args.epochs, "seed": seed},
                epochs=args.epochs, verbose=False
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

        # Setup D: PCA Feature Extraction
        t0 = time.time()
        try:
            m_D = setup_D(X_train_34, X_test_34, gt_mask, {**best_config, "seed": seed})
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

        # Setup E: IB Feature Extraction
        t0 = time.time()
        try:
            m_E = setup_E(X_train_34, X_test_34, gt_mask, {**best_config, "seed": seed})
            m_E["run"] = run
            m_E["seed"] = seed
            m_E["time_s"] = time.time() - t0
            results_E.append(m_E)
            LOGGER.info(
                "  [E] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f, sep=%.2fx [%.1fs]",
                m_E["auc_pr"], m_E["auc_roc"], m_E["f1"],
                m_E["separation_ratio"], m_E["time_s"]
            )
        except Exception as e:
            LOGGER.error("  [E] Error: %s", e)

        # Setup F: AE + ContextBeta (paper-default)
        t0 = time.time()
        try:
            m_F, _ = setup_F(
                X_train_34, hours_train, dows_train, rcs_train, nb_train,
                X_test_34, hours_test, dows_test, rcs_test, nb_test, gt_mask,
                {**best_config, "epochs": args.epochs, "seed": seed},
            )
            m_F["run"] = run
            m_F["seed"] = seed
            m_F["time_s"] = time.time() - t0
            results_F.append(m_F)
            LOGGER.info(
                "  [F] AUC-PR=%.4f, AUC-ROC=%.4f, F1=%.4f, sep=%.2fx [%.1fs]",
                m_F["auc_pr"], m_F["auc_roc"], m_F["f1"],
                m_F["separation_ratio"], m_F["time_s"]
            )
        except Exception as e:
            LOGGER.error("  [F] Error: %s", e)

    # Summary statistics
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

    summary_A = summarize(results_A, "A: Raw 7D")
    summary_B = summarize(results_B, "B: 34D + AE")
    summary_C = summarize(results_C, "C: 34D + AE + ContextBeta")
    summary_D = summarize(results_D, "D: PCA (D=8)")
    summary_E = summarize(results_E, "E: IB (D=8, beta=0.5)")
    summary_F = summarize(results_F, "F: AE + ContextBeta (paper-default)")

    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("ABLATION SUMMARY (%d runs, paper: 5 runs)", args.runs)
    LOGGER.info("=" * 80)

    for s, label in [
        (summary_A, "A"), (summary_B, "B"), (summary_C, "C"),
        (summary_D, "D"), (summary_E, "E"), (summary_F, "F"),
    ]:
        if not s:
            continue
        LOGGER.info("")
        LOGGER.info("[%s] %s", label, s["name"])
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

    # Delta analysis: Feature Extraction comparison (Table 6b paper)
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("Feature Extraction Comparison (paper Table 6b)")
    LOGGER.info("=" * 60)
    pairs = [
        ("D vs B", summary_D, summary_B, "PCA vs AE"),
        ("E vs B", summary_E, summary_B, "IB vs AE"),
        ("F vs C", summary_F, summary_C, "AE-paper-default vs AE"),
    ]
    for label, s_new, s_old, desc in pairs:
        if s_new and s_old:
            LOGGER.info("")
            LOGGER.info("Delta %s (%s):", label, desc)
            for k in ["auc_roc", "auc_pr", "f1"]:
                delta = s_new[f"{k}_mean"] - s_old[f"{k}_mean"]
                LOGGER.info("  %s: %+.4f", k, delta)

    # Save results
    output = {
        "setup_A": summary_A,
        "setup_B": summary_B,
        "setup_C": summary_C,
        "setup_D": summary_D,
        "setup_E": summary_E,
        "setup_F": summary_F,
        "all_runs": {
            "A": results_A, "B": results_B, "C": results_C,
            "D": results_D, "E": results_E, "F": results_F,
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
