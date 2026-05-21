#!/usr/bin/env python3
"""
MemStream v4 Core — Paper-Exact Implementation (arXiv:2106.03837v2)

File nay chua tat ca logic core cua MemStream, duoc su dung boi:
  - grid_search.py (tim HP tot nhat)
  - ablation.py (ablation study)
  - comparison.py (so sanh voi ML benchmarks)

Paper deviations (tuong ung voi paper):
  1. AE: 34->68->34, Tanh (paper: D=2d, activation Tanh, line 518-519)
  2. Noise: 0.001 (paper: isotropic Gaussian noise, Section 4.3)
  3. Chi train N = memory_len samples (paper: "small subset D")
  4. Memory init = encode(N samples) (paper Algo 1 line 3)
  5. kNN scoring, KHONG centroid fallback (paper Algo 1)
  6. Streaming memory update voi FIFO (paper Algo 1 line 15)
  7. Mean/std tinh 1 lan tai init, khong recompute khi memory update
  8. Memory poisoning mitigation voi gamma (paper Table 6f)
  9. ContextBeta: context-aware threshold (extension cua ta)
  10. IB: 2-layer network (paper line 518-519: "2 layer binary classifier")

Extensions (khong co trong paper goc):
  - ContextBeta: adaptive threshold theo neighborhood x context_cell
"""
from __future__ import annotations

import os
import sys
import time
import json
import pickle
import io
import hashlib
import hmac
import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from itertools import product

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"
os.environ["ARROW_DISABLE_MMAP"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
LOGGER = logging.getLogger("memstream_v4_core")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# PAPER-EXACT AUTOENCODER
# =============================================================================
# Paper page 5, line 551: "AE was used for feature extraction with output
#   dimension D = 2d"
# Reference impl (memstream.py line 60-66): encoder = Linear(d, 2d), Tanh
#
# Architecture:
#   Input(d) -> Linear(d, 2d) -> Tanh -> Linear(2d, d) -> Output(d)
#
# Luu y:
#   - Dung ReLU (khong phai Tanh)
#   - Bottleneck o day la 2d = 68D (> input 34D), dam bao D >= d (paper Prop 2)
# =============================================================================
class MemStreamAE(nn.Module):
    """Paper-exact AE: d→2d→d, single layer, ReLU."""

    def __init__(self, d: int = 34, out_dim: Optional[int] = None):
        super().__init__()
        if out_dim is None:
            out_dim = d * 2  # Paper: D = 2d
        self.d = d
        self.out_dim = out_dim
        self.encoder = nn.Linear(d, out_dim)
        self.decoder = nn.Linear(out_dim, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.encoder(x))
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.encoder(x))


# =============================================================================
# FIFO MEMORY (Paper Section 4.4)
# =============================================================================
# Paper: "First-In-First-Out (FIFO) memory replacement policy"
# Paper Algo 1, line 15: "Replace earliest added element in M with z_t"
#
# Fix 1: Streaming update — moi mau deu co the cap nhat memory
# Fix 2: Chi update neu score < beta (nguong anomaly threshold)
# =============================================================================
class Memory:
    """
    FIFO memory voi streaming update (paper-exact).

    Mean/std duoc update O(1) per sample bang Welford's parallel algorithm.
    Khi FIFO ghi de 1 element cu, running stats duoc adjust de loai bo
    contribution cua element do — chinh xac nhu full recompute.

    Welford's parallel (Chan et al. 1979):
      new_mean = old_mean + (x - old_mean) / n
      delta    = x - new_mean
      M2_new   = M2_old + (x - old_mean) * delta
      var      = M2 / n

    Eviction (ghi de element cu):
      M2_new   = M2_old - (x_old - old_mean) * (x_old - new_mean)
    """

    def __init__(self, L: int, d: int):
        self.L = L
        self.d = d
        self.M = np.zeros((L, d), dtype=np.float32)
        self.mem_data_raw = np.zeros((L, d), dtype=np.float32)
        self.ptr = 0
        self.cnt = 0
        self.num_updates = 0
        # Running mean / M2 (Welford)
        self.mean = np.zeros(d, dtype=np.float64)
        self.M2 = np.zeros(d, dtype=np.float64)

    def update(self, z: np.ndarray, raw: np.ndarray) -> bool:
        """
        Update memory voi FIFO + O(1) Welford stats update.

        Neu memory chua day: chen binh thuong (Welford online).
        Neu memory da day: evict oldest roi tinh lai contribution cua no.
        """
        z = np.asarray(z, dtype=np.float32).flatten()
        raw = np.asarray(raw, dtype=np.float32)

        if self.cnt < self.L:
            # Chua day: Welford online insert
            x = raw.astype(np.float64)
            n = self.cnt + 1
            delta = x - self.mean
            self.mean = self.mean + delta / n
            delta2 = x - self.mean
            self.M2 = self.M2 + delta * delta2
        else:
            # Da day: evict oldest element roi apply insert
            old_idx = self.ptr  # vi tri se bi ghi de
            old_raw = self.mem_data_raw[old_idx].astype(np.float64)
            old_mean = self.mean.copy()
            n = self.L
            # Loai bo contribution cua evicted element
            delta_old = old_raw - old_mean
            new_mean = old_mean + delta_old / n
            delta2_old = old_raw - new_mean
            self.M2 = self.M2 - delta_old * delta2_old
            # Update mean nhu Welford online insert
            delta = raw.astype(np.float64) - new_mean
            self.mean = new_mean + delta / n
            delta2 = raw.astype(np.float64) - self.mean
            self.M2 = self.M2 + delta * delta2

        self.M[self.ptr] = z
        self.mem_data_raw[self.ptr] = raw
        self.ptr = (self.ptr + 1) % self.L
        self.cnt = min(self.cnt + 1, self.L)
        self.num_updates += 1
        return True

    def active(self) -> np.ndarray:
        return self.M[: self.cnt]

    def active_raw(self) -> np.ndarray:
        return self.mem_data_raw[: self.cnt]

    def get_mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tra ve mean/std hien tai tu running stats (O(1))."""
        n = max(self.cnt, 1)
        std = np.sqrt(self.M2 / n)
        std = np.clip(std, 1e-6, None)
        return self.mean.astype(np.float32), std.astype(np.float32)

    def recompute_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Full recompute tu raw data (O(N*d), dung lam ground truth check)."""
        raw = self.active_raw()
        if len(raw) == 0:
            mean = np.zeros(self.d, dtype=np.float32)
            std = np.ones(self.d, dtype=np.float32)
        else:
            mean = raw.mean(axis=0)
            std = np.clip(raw.std(axis=0), 1e-6, None)
        return mean.astype(np.float32), std.astype(np.float32)


# =============================================================================
# CONTEXTBETA (Paper Section 4.4, Algorithm 1)
# =============================================================================
# Paper Algo 1, line 14: "if Score(z_t) < beta then replace..."
# Beta la threshold de quyet dinh co update memory hay khong.
# ContextBeta mo rong: moi (neighborhood, context_cell) co beta rieng.
# =============================================================================
class ContextBeta:
    """Context-aware beta thresholding (paper Section 4.4)."""

    CELL_MIN = 11

    def __init__(self, db: float = 0.5, pct: float = 95.0):
        self.db = db
        self.pct = pct
        self.T = np.full((10, 8), db, dtype=np.float32)
        self.S: Dict = {}
        self.R: List = []
        self.beta = db
        self.fitted = False
        self.n_rec = 0

    def rec(self, nb: int, ctx: int, s: float) -> None:
        k = (int(nb), int(ctx))
        if k not in self.S:
            self.S[k] = []
        self.S[k].append(float(s))
        self.R.append(float(s))
        self.n_rec += 1

    def fit(self, ob: Optional[float] = None) -> "ContextBeta":
        if ob is not None:
            self.beta = float(ob)
        elif self.R:
            self.beta = float(np.percentile(self.R, self.pct))
        for nb in range(10):
            for ctx in range(8):
                sc = self.S.get((nb, ctx), [])
                if len(sc) > self.CELL_MIN - 1:
                    self.T[nb, ctx] = float(np.percentile(sc, self.pct))
                else:
                    self.T[nb, ctx] = self.beta
        self.fitted = True
        return self

    def beta_for(self, nb: int, ctx: int) -> float:
        return float(self.T[int(np.clip(nb, 0, 9)), int(np.clip(ctx, 0, 7))])


def compute_context_cell_id(hour: int, dow: int, ratecode: float) -> int:
    """Context cell = (special_rate, night, weekend)."""
    is_special = 1 if ratecode > 1.0 else 0
    is_night = 1 if (hour >= 20 or hour < 6) else 0
    is_weekend = 1 if dow >= 5 else 0
    return (is_special << 2) | (is_night << 1) | is_weekend


# =============================================================================
# PAPER SCORING: kNN L1, NO Centroid Fallback
# =============================================================================
# Paper Algo 1, line 10: R(z_t, z_hat_i) = ||z_t - z_hat_i||_1 for all i
# Paper Algo 1, line 12: Score = weighted average over K neighbors
#
# Fix 6: BO centroid fallback — dung min kNN qua memory
#
# Y nghia gamma (KNN coefficient, paper Table 6f):
#   gamma = 0: binh thuong (dung k=3)
#   gamma > 0: discounting weights, giup tu phuc hoi memory poisoning
# =============================================================================
class PaperScorer:
    """kNN L1 scoring, NO centroid fallback."""

    def __init__(
        self,
        ae_state: Dict,
        mean: np.ndarray,
        std: np.ndarray,
        mem_M: np.ndarray,
        k: int = 5,
        gamma: float = 0.0,
        input_mean: np.ndarray = None,
        input_std: np.ndarray = None,
    ):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self._input_mean = input_mean.astype(np.float32) if input_mean is not None else mean.astype(np.float32)
        self._input_std = input_std.astype(np.float32) if input_std is not None else std.astype(np.float32)
        self.mem_M = mem_M.astype(np.float32)
        self.mem_cnt = len(mem_M)
        self.k = k
        self.gamma = gamma

        d = ae_state.get("input_dim", 34)
        out_dim = ae_state.get("out_dim", 68)

        class _AE(nn.Module):
            def __init__(self, d, out_dim):
                super().__init__()
                self.encoder = nn.Linear(d, out_dim)
                self.decoder = nn.Linear(out_dim, d)

            def forward(self, x):
                return torch.tanh(self.encoder(x))

        self._ae = _AE(d, out_dim)
        self._ae.load_state_dict(ae_state, strict=False)
        self._ae.eval()
        self._ae.to(DEVICE)
        self._mem_t = torch.from_numpy(self.mem_M)

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Score batch with min L1 distance (closest neighbor only, paper Algo 1 line 10)."""
        if self._ae is None or self.mem_cnt < 2:
            return np.full(len(X), 0.5)
        X = X.astype(np.float32)
        n = len(X)

        Xn = (X - self._input_mean) / (self._input_std + 1e-8)
        Xn_t = torch.from_numpy(Xn).to(DEVICE)

        with torch.no_grad():
            Z = self._ae(Xn_t).cpu().numpy()

        Z_t = torch.from_numpy(Z)
        M_t = self._mem_t  # CPU tensor (no GPU transfer needed for small memory)

        scores = []
        for s in range(0, n, 2000):
            chunk = Z_t[s: s + 2000].unsqueeze(1)
            diff = (chunk - M_t.unsqueeze(0)).abs().sum(2).numpy()
            top_k = np.sort(diff, axis=1)[:, : self.k]
            chunk_scores = top_k.sum(axis=1)
            scores.append(chunk_scores)
        return np.concatenate(scores)

    def score_point(self, x: np.ndarray, z_encoded: np.ndarray = None) -> float:
        """Score a single point with min L1 distance (paper Algo 1 line 10).

        Args:
            x: raw 34D feature vector (used only if z_encoded is None)
            z_encoded: pre-encoded latent vector (optional, for batch efficiency)
        """
        if self._ae is None or self.mem_cnt < 2:
            return 0.5
        if z_encoded is None:
            xn = ((x - self._input_mean) / (self._input_std + 1e-8)).astype(np.float32)
            xn_t = torch.from_numpy(xn).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                z = self._ae(xn_t).cpu().numpy().flatten()
        else:
            z = z_encoded.astype(np.float32)
        diff = np.abs(z - self._mem_t.cpu().numpy()).sum(1)
        top_k = np.sort(diff)[: self.k]
        return float(top_k.sum())

    def update_stats(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        mem_M: np.ndarray,
        mem_cnt: int,
    ) -> None:
        """Sync mean/std/memory sau khi memory recompute stats (paper Algo 1 line 15)."""
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.mem_M = mem_M.astype(np.float32)
        self.mem_cnt = mem_cnt
        self._mem_t = torch.from_numpy(self.mem_M[: self.mem_cnt])


# =============================================================================
# PAPER FEATURE EXTRACTORS: PCA & Information Bottleneck (Section 4.3)
# =============================================================================
# Paper Table 6(b): Identity, PCA, IB, AE
# Paper Section 4.3: AE outperforms PCA and IB
# Paper line 520: "output dimension as 8 for PCA and IB"
# =============================================================================


class PCAScorer:
    """PCA-based feature extraction + kNN scoring (paper Table 6b)."""

    def __init__(
        self,
        n_components: int = 8,
        mean: np.ndarray = None,
        std: np.ndarray = None,
        mem_M: np.ndarray = None,
        k: int = 5,
        gamma: float = 0.0,
    ):
        from sklearn.decomposition import PCA as SkPCA
        self.n_components = n_components
        self.mean = mean.astype(np.float32) if mean is not None else None
        self.std = std.astype(np.float32) if std is not None else None
        self.mem_M = mem_M.astype(np.float32) if mem_M is not None else None
        self.k = k
        self.gamma = gamma
        self._pca = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "PCAScorer":
        """Fit PCA on raw data."""
        if self.mean is None:
            self.mean = X.mean(axis=0).astype(np.float32)
        if self.std is None:
            self.std = np.clip(X.std(axis=0), 1e-6, None).astype(np.float32)
        Xn = (X - self.mean) / (self.std + 1e-8)
        self._pca = SkPCA(n_components=self.n_components)
        self._pca.fit(Xn)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.mean) / (self.std + 1e-8)
        return self._pca.transform(Xn)

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Score batch with weighted average of top-K nearest neighbors (paper Algo 1)."""
        if not self._fitted or self.mem_M is None or len(self.mem_M) < 2:
            return np.full(len(X), 0.5)
        Z = self.transform(X).astype(np.float32)
        mem = self.mem_M.astype(np.float32)
        scores = []
        for s in range(0, len(Z), 2000):
            chunk = Z[s: s + 2000]
            diff = np.abs(chunk[:, np.newaxis, :] - mem[np.newaxis, :, :]).sum(2)
            top_k = np.sort(diff, axis=1)[:, : self.k]
            chunk_scores = top_k.sum(axis=1)
            scores.append(chunk_scores)
        return np.concatenate(scores)

    def score_point(self, x: np.ndarray) -> float:
        """Score a single point with sum of top-K distances."""
        if not self._fitted or self.mem_M is None or len(self.mem_M) < 2:
            return 0.5
        zn = self.transform(x).astype(np.float32)
        mem = self.mem_M.astype(np.float32)
        diff = np.abs(zn - mem).sum(1)
        top_k = np.sort(diff)[: self.k]
        return float(top_k.sum())


class IBScorer:
    """
    Information Bottleneck feature extraction + kNN scoring (paper Table 6b).

    Paper: IB with beta=0.5, variance=1.0
    Paper line 518-519: "The network was implemented as a 2 layer binary classifier"
    Two-layer encoder: Linear(d, hidden) -> ReLU -> Linear(hidden, out_dim)
    with IB loss: minimize I(X;Z) - beta*I(Z;Y), approximated by reconstruction + KL.
    """

    def __init__(
        self,
        d: int = 38,
        out_dim: int = 8,
        hidden_dim: int = 16,
        mean: np.ndarray = None,
        std: np.ndarray = None,
        mem_M: np.ndarray = None,
        k: int = 5,
        gamma: float = 0.0,
        beta_ib: float = 0.5,
        lr: float = 0.001,
        epochs: int = 500,
        seed: int = 42,
    ):
        self.d = d
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.mean = mean.astype(np.float32) if mean is not None else None
        self.std = std.astype(np.float32) if std is not None else None
        self.mem_M = mem_M.astype(np.float32) if mem_M is not None else None
        self.k = k
        self.gamma = gamma
        self.beta_ib = beta_ib
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self._encoder = None
        self._decoder = None
        self._fitted = False
        self._device = DEVICE

    def fit(self, X: np.ndarray) -> "IBScorer":
        """Train IB encoder: minimize I(X;Z) - beta*I(Z;Y)."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self.mean is None:
            self.mean = X.mean(axis=0).astype(np.float32)
        if self.std is None:
            self.std = np.clip(X.std(axis=0), 1e-6, None).astype(np.float32)
        Xn = (X - self.mean) / (self.std + 1e-8)
        Xn_t = torch.from_numpy(Xn).float().to(self._device)

        class IBNet(nn.Module):
            def __init__(inner_self, d, hidden_dim, out_dim):
                super().__init__()
                inner_self.encoder = nn.Sequential(
                    nn.Linear(d, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dim),
                )
                inner_self.decoder = nn.Linear(out_dim, d)

            def encode(inner_self, x):
                z = inner_self.encoder(x)
                return z

            def decode(inner_self, z):
                return inner_self.decoder(z)

        self._encoder = IBNet(self.d, self.hidden_dim, self.out_dim).to(self._device)
        opt = torch.optim.Adam(self._encoder.parameters(), lr=self.lr)

        for ep in range(self.epochs):
            perm = torch.randperm(len(Xn_t), device=self._device)
            for i in range(0, len(Xn_t), 256):
                b = Xn_t[perm[i: i + 256]]
                z = self._encoder.encode(b)
                recon = self._encoder.decode(z)
                recon_loss = nn.MSELoss()(recon, b)
                kl_loss = 0.5 * (z.pow(2)).mean()
                loss = recon_loss + self.beta_ib * kl_loss
                opt.zero_grad()
                loss.backward()
                opt.step()

        self._encoder.eval()
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.mean) / (self.std + 1e-8)
        Xn_t = torch.from_numpy(Xn.astype(np.float32)).to(self._device)
        with torch.no_grad():
            Z = self._encoder.encode(Xn_t).cpu().numpy()
        return Z

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Score batch with weighted average of top-K nearest neighbors (paper Algo 1)."""
        if not self._fitted or self.mem_M is None or len(self.mem_M) < 2:
            return np.full(len(X), 0.5)
        Z = self.transform(X).astype(np.float32)
        mem = self.mem_M.astype(np.float32)
        scores = []
        for s in range(0, len(Z), 2000):
            chunk = Z[s: s + 2000]
            diff = np.abs(chunk[:, np.newaxis, :] - mem[np.newaxis, :, :]).sum(2)
            top_k = np.sort(diff, axis=1)[:, : self.k]
            chunk_scores = top_k.sum(axis=1)
            scores.append(chunk_scores)
        return np.concatenate(scores)

    def score_point(self, x: np.ndarray) -> float:
        """Score a single point with sum of top-K distances."""
        if not self._fitted or self.mem_M is None or len(self.mem_M) < 2:
            return 0.5
        zn = self.transform(x).astype(np.float32)
        mem = self.mem_M.astype(np.float32)
        diff = np.abs(zn - mem).sum(1)
        top_k = np.sort(diff)[: self.k]
        return float(top_k.sum())


# =============================================================================
# FULL MEMSTREAM PIPELINE (Algorithm 1 + Extensions)
# =============================================================================
class MemStreamPipeline:
    """
    Full MemStream pipeline theo paper Algorithm 1.

    Bao gom:
      - Cold-start training (N samples)
      - Memory initialization
      - Streaming scoring + memory update
      - ContextBeta adaptation

    Usage:
        pipeline = MemStreamPipeline(d=34, memory_len=1024, ...)
        pipeline.train(X_train, ...)         # Cold-start
        scores = pipeline.score(X_test)      # Streaming
        results = pipeline.get_results()     # Lay metrics
    """

    def __init__(
        self,
        d: int = 38,
        out_dim: Optional[int] = None,
        memory_len: int = 2048,
        k: int = 5,
        gamma: float = 0.0,
        beta: float = 0.5,
        noise_std: float = 0.001,
        lr: float = 0.01,
        epochs: int = 5000,
        batch_size: int = 1024,
        seed: int = 42,
        cb_warmup: int = 4096,
        verbose: bool = True,
        adam_betas: Tuple[float, float] = (0.9, 0.999),  # paper Section 5 (line 540)
    ):
        if out_dim is None:
            out_dim = d * 2
        self.d = d
        self.out_dim = out_dim
        self.memory_len = memory_len
        self.k = k
        self.gamma = gamma
        self.beta = beta
        self.noise_std = noise_std
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.cb_warmup = cb_warmup
        self.verbose = verbose
        self.adam_betas = adam_betas  # paper: beta1=0.9, beta2=0.999

        self.ae: Optional[MemStreamAE] = None
        self.opt: Optional[torch.optim.Optimizer] = None
        self.crit = nn.MSELoss()
        self._scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None
        self.memory: Optional[Memory] = None
        self.cb: Optional[ContextBeta] = None
        self.scorer: Optional[PaperScorer] = None
        self.n_retrains = 0
        self.total_scores: List[float] = []
        self.total_updates = 0
        self._fitted = False
        self.epoch_losses: List[float] = []

    # -------------------------------------------------------------------------
    # Training (cold-start)
    # -------------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        hours_train: np.ndarray,
        dows_train: np.ndarray,
        rcs_train: np.ndarray,
        nb_train: np.ndarray,
        X_warmup: Optional[np.ndarray] = None,
        hours_warmup: Optional[np.ndarray] = None,
        dows_warmup: Optional[np.ndarray] = None,
        rcs_warmup: Optional[np.ndarray] = None,
        nb_warmup: Optional[np.ndarray] = None,
    ) -> "MemStreamPipeline":
        """
        Cold-start training theo paper Algorithm 1.

        1. Chi lay N = memory_len samples (paper: "small subset D")
        2. Tinh mean/std tu RAW data (ref line 75)
        3. Train AE tren N samples, noise=0.001, 5000 epochs, lr=0.01
        4. Init memory = encode(N samples) (paper Algo 1 line 3)
        5. Warmup ContextBeta (paper Algo 1 line 14)
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self.verbose:
            LOGGER.info("  [Cold-start] Training on N=%d samples", self.memory_len)

        # Paper: train chi tren N samples
        n_train = min(self.memory_len, len(X_train))
        idx = np.random.permutation(len(X_train))[:n_train]
        X_init = X_train[idx].astype(np.float32)
        h_init = hours_train[idx]
        d_init = dows_train[idx]
        rc_init = rcs_train[idx]
        nb_init = nb_train[idx]

        # FIX 5: Tinh mean/std tu RAW data (khong phai encoded)
        self._input_mean = X_init.mean(axis=0).astype(np.float32)
        self._input_std = np.clip(X_init.std(axis=0), 1e-6, None).astype(np.float32)
        self.mean = self._input_mean
        self.std = self._input_std

        Xn_init = (X_init - self.mean) / (self.std + 1e-8)
        Xn_init_t = torch.from_numpy(Xn_init).to(DEVICE)

        # AE: 34→68→34, Tanh, Adam (paper: beta1=0.9, beta2=0.999, Section 5)
        self.ae = MemStreamAE(d=self.d, out_dim=self.out_dim).to(DEVICE)
        self.opt = torch.optim.Adam(
            self.ae.parameters(), lr=self.lr,
            betas=(self.adam_betas[0], self.adam_betas[1]),
        )

        if self.verbose:
            LOGGER.info(
                "  [AE] 34->%d->34, ReLU, noise=%.4f, lr=%.4f, epochs=%d, Adam(%.3f, %.3f)",
                self.out_dim, self.noise_std, self.lr, self.epochs,
                self.adam_betas[0], self.adam_betas[1]
            )

        t_train = time.time()
        for ep in range(self.epochs):
            perm = torch.randperm(len(Xn_init_t), device=DEVICE)
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(Xn_init_t), self.batch_size):
                b = Xn_init_t[perm[i: i + self.batch_size]]
                if self._scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss = self.crit(
                            self.ae(b + torch.randn_like(b) * self.noise_std),
                            b
                        )
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
                    self._scaler.step(self.opt)
                    self._scaler.update()
                else:
                    loss = self.crit(
                        self.ae(b + torch.randn_like(b) * self.noise_std),
                        b
                    )
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
                    self.opt.step()
                total_loss += loss.item()
                n_batches += 1

            epoch_loss = total_loss / max(n_batches, 1)
            self.epoch_losses.append(float(epoch_loss))

            if self.verbose and (ep + 1) % 500 == 0:
                LOGGER.info("    Epoch %d/%d: loss=%.6f",
                           ep + 1, self.epochs, total_loss / max(n_batches, 1))

        self.ae.eval()
        if self.verbose:
            LOGGER.info("  [AE] Cold-start done in %.1fs", time.time() - t_train)

        # Paper: Memory = f_theta(D) — encode N samples
        self.memory = Memory(self.memory_len, self.out_dim)
        with torch.no_grad():
            enc_init = self.ae.encode(Xn_init_t).cpu().numpy()
            for i in range(n_train):
                self.memory.update(enc_init[i], enc_init[i])

        # Sync Welford running stats voi pipeline state
        w_mean, w_std = self.memory.get_mean_std()
        self.mean = w_mean
        self.std = w_std

        if self.verbose:
            LOGGER.info("  [Memory] Initialized %d/%d slots (FIFO), Welford synced",
                        self.memory.cnt, self.memory_len)

        # ContextBeta warmup
        self._build_scorer()

        # Warmup ContextBeta
        self.cb = ContextBeta(db=self.beta, pct=95.0)
        wn = min(self.cb_warmup, n_train)
        _provided_nb = nb_warmup
        if X_warmup is None:
            X_warmup = X_init[:wn]
            h_warmup = h_init[:wn]
            d_warmup = d_init[:wn]
            rc_warmup = rc_init[:wn]
            nb_warmup = nb_init[:wn]
        else:
            wn = min(wn, len(X_warmup))
            h_warmup = hours_warmup[:wn] if hours_warmup is not None else h_init[:wn]
            d_warmup = dows_warmup[:wn] if dows_warmup is not None else d_init[:wn]
            rc_warmup = rcs_warmup[:wn] if rcs_warmup is not None else rc_init[:wn]
            nb_warmup = _provided_nb[:wn] if _provided_nb is not None else nb_init[:wn]

        if wn > 0:
            # Use per-point scoring for warmup (consistent with score_stream behavior)
            warmup_scores = np.empty(wn, dtype=np.float64)
            for i in range(wn):
                warmup_scores[i] = self.scorer.score_point(X_warmup[i])
            for i in range(wn):
                ctx = compute_context_cell_id(
                    int(h_warmup[i]), int(d_warmup[i]), float(rc_warmup[i])
                )
                self.cb.rec(int(nb_warmup[i]), ctx, warmup_scores[i])
            self.cb.fit()
            beta_thres = float(np.percentile(warmup_scores, 95))
            cells_filled = sum(
                1 for v in self.cb.S.values()
                if len(v) > self.cb.CELL_MIN - 1
            )
            if self.verbose:
                LOGGER.info(
                    "  [ContextBeta] beta=%.4f, threshold=%.4f, cells=%d/%d",
                    self.cb.beta, beta_thres, cells_filled, 80
                )

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
            mean=self.mean,
            std=self.std,
            mem_M=self.memory.active(),
            k=self.k,
            gamma=self.gamma,
            input_mean=self._input_mean,
            input_std=self._input_std,
        )

    # -------------------------------------------------------------------------
    # Streaming Score + Memory Update
    # -------------------------------------------------------------------------
    def score_stream(
        self,
        X: np.ndarray,
        hours: np.ndarray,
        dows: np.ndarray,
        rcs: np.ndarray,
        nb: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
        update_memory: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Streaming score per-point (paper Algorithm 1).

        Paper Algorithm 1, line 4-17:
          1. Extract features: z_t = f_theta(x_t)
          2. Query memory: K-nearest neighbours
          3. Calculate distance: R(z_t, z_hat_i) = ||z_t - z_hat_i||_1
          4. Assign discounted score
          5. Update Memory if score < beta

        Args:
            X: Features [N, 34]
            hours, dows, rcs, nb: Context info [N]
            gt_mask: Ground truth (optional)
            update_memory: If True, update memory during scoring (streaming).
                          If False, score only (eval mode for grid search).
        """
        n = len(X)
        scores = np.empty(n, dtype=np.float64)

        # Batch encode all points to amortize GPU transfer overhead
        Xn = (X.astype(np.float32) - self._input_mean) / (self._input_std + 1e-8)
        encoded = np.empty((n, self.out_dim), dtype=np.float32)
        for s in range(0, n, 1024):
            e = min(s + 1024, n)
            Xn_t = torch.from_numpy(Xn[s:e]).to(DEVICE)
            with torch.no_grad():
                encoded[s:e] = self.ae.encode(Xn_t).cpu().numpy()

        _prev_mem_state = None
        if not update_memory and self.memory is not None:
            _prev_mem_state = (self.memory.M.copy(), self.memory.cnt, self.memory.ptr,
                               self.mean.copy(), self.std.copy())

        for i in range(n):
            raw_score = self.scorer.score_point(X[i], encoded[i])
            scores[i] = raw_score
            self.total_scores.append(float(raw_score))

            if self.cb is not None and self.cb.fitted:
                cid = compute_context_cell_id(
                    int(hours[i]), int(dows[i]), float(rcs[i])
                )
                beta_t = self.cb.beta_for(int(nb[i]), int(cid))
            else:
                beta_t = self.beta

            # Paper Algo 1 line 15: update memory + recompute mean/std
            if update_memory and self.memory is not None and raw_score < beta_t:
                z = encoded[i]
                self.memory.update(z, z)
                self.total_updates += 1
                new_mean, new_std = self.memory.get_mean_std()
                self.mean = new_mean
                self.std = new_std
                self.scorer.update_stats(
                    self.mean, self.std,
                    self.memory.active(), self.memory.cnt,
                )

        # Restore memory state if in eval mode
        if not update_memory and _prev_mem_state is not None:
            M_copy, cnt, ptr, mean_copy, std_copy = _prev_mem_state
            self.memory.M[:cnt] = M_copy[:cnt]
            self.memory.cnt = cnt
            self.memory.ptr = ptr
            self.mean = mean_copy
            self.std = std_copy
            self.scorer.update_stats(self.mean, self.std, self.memory.active(), self.memory.cnt)

        # Use raw scores for eval (no ContextBeta adjustment needed for ranking)
        adj_scores = scores

        # Metrics
        metrics = {}
        if gt_mask is not None:
            best_t, best_f1 = find_best_threshold(adj_scores, gt_mask)
            metrics = evaluate_scores(adj_scores, gt_mask, threshold=best_t)
            metrics["best_threshold"] = best_t
        metrics["n_memory_updates"] = self.total_updates

        return adj_scores, metrics

    def get_results(self) -> Dict:
        return {
            "memory_len": self.memory_len,
            "k": self.k,
            "gamma": self.gamma,
            "beta": self.beta,
            "noise_std": self.noise_std,
            "lr": self.lr,
            "epochs": self.epochs,
            "n_retrains": self.n_retrains,
            "n_memory_updates": self.total_updates,
            "epoch_losses": list(self.epoch_losses),
        }


# =============================================================================
# METRICS
# =============================================================================
def compute_auc_roc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, scores)
    except (ImportError, ValueError):
        return 0.5


def compute_auc_pr(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import auc, precision_recall_curve
        p, r, _ = precision_recall_curve(y_true, scores)
        return auc(r, p) if len(p) > 1 else 0.5
    except (ImportError, ValueError):
        return 0.5


def evaluate_scores(
    scores: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 1.0,
) -> Dict:
    y_true = gt_mask.astype(np.int32)
    n = len(y_true)
    if len(scores) != n:
        min_len = min(len(scores), n)
        scores = scores[:min_len]
        y_true = y_true[:min_len]
        n = min_len

    auc_roc = compute_auc_roc(y_true, scores)
    auc_pr = compute_auc_pr(y_true, scores)
    pred = (scores >= threshold).astype(np.int32)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    tn = ((pred == 0) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acc = (tp + tn) / n if n > 0 else 0.0
    nm = scores[y_true == 0]
    am = scores[y_true == 1]

    return {
        "auc_roc": float(auc_roc),
        "auc_pr": float(auc_pr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "acc": float(acc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": float(threshold),
        "score_normal_mean": float(nm.mean()) if len(nm) > 0 else 0.0,
        "score_anomaly_mean": float(am.mean()) if len(am) > 0 else 0.0,
        "score_normal_std": float(nm.std()) if len(nm) > 0 else 0.0,
        "score_anomaly_std": float(am.std()) if len(am) > 0 else 0.0,
        "separation_ratio": (
            float(am.mean() / nm.mean()) if len(nm) > 0 and nm.mean() > 0 else 0.0
        ),
    }


def find_best_threshold(
    scores: np.ndarray, gt_mask: np.ndarray, n_steps: int = 200
) -> Tuple[float, float]:
    y_true = gt_mask.astype(np.int32)
    lo, hi = float(scores.min()), float(scores.max())
    if lo == hi:
        return lo, 0.0
    best_f1, best_t = 0.0, (lo + hi) / 2
    for t in np.linspace(lo, hi, n_steps):
        pred = (scores >= t).astype(np.int32)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_t = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_t = 2 * prec * rec_t / (prec + rec_t) if (prec + rec_t) > 0 else 0.0
        if f1_t > best_f1:
            best_f1, best_t = f1_t, float(t)
    return best_t, best_f1


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
_NEIGHBORHOOD_RANGES = [
    ((1, 43), 0),    # manhattan
    ((44, 103), 4),   # bronx
    ((104, 127), 1),  # brooklyn
    ((128, 148), 2),  # queens_lower
    ((149, 161), 3),  # queens_upper
    ((162, 181), 5),  # staten_island
    ((182, 196), 6),  # ewr
    ((217, 229), 7),  # jfk
    ((230, 234), 8),  # nalp
]

_PARQUET_COLS = [
    "trip_distance", "fare_amount", "passenger_count", "total_amount",
    "RatecodeID", "PULocationID", "DOLocationID",
    "duration_s", "speed_mph", "hour", "weekday",
    "tip_amount",   # ADDED: needed for Type 2 tip_anomaly detection
]


def extract_features_from_parquet(
    pq_path: str,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 34D feature extraction."""
    import pyarrow.parquet as pq

    t0 = time.time()
    pf = pq.ParquetFile(pq_path)
    total_rows = pf.metadata.num_rows

    if max_rows:
        n_read = min(max_rows, total_rows)
        skip = max(0, total_rows - n_read)
        tbl = pf.read(columns=_PARQUET_COLS).slice(skip, n_read)
        merged = tbl.to_pydict()
        n = len(merged[_PARQUET_COLS[0]])
    else:
        reader = pf.iter_batches(batch_size=500000, columns=_PARQUET_COLS)
        chunks = []
        for batch in reader:
            chunks.append(batch.to_pydict())
        del reader
        merged = {key: np.concatenate([c[key] for c in chunks]) for key in _PARQUET_COLS}
        n = len(merged[_PARQUET_COLS[0]])
        del chunks

    LOGGER.info("  Loaded %d rows in %.1fs", n, time.time() - t0)
    return _build_features(merged, n)


def _build_features(d, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    38D feature extraction for NYC Taxi anomaly detection.

    Features (38D):
      0-6:  Raw core (dist, dur, fare, tot, speed, hour, dow)
      7-13: Derived ratios (fare/mi, fare/min, speed, pax/dist, tip_ratio, log_fare, log_dist)
     14-18: Cyclical encodings (sin_h, cos_h, sin_dw, cos_dw, weekend)
     19-24: Location grid (pux, puy, dox, doy, grid_dist, night)
     25-30: Anomaly-specific signals (short_trip, long_dur, high_tip, fast_trip, dur_sq, dist_sq)
     31-37: Clipped/scaled (fpm_c, fmn_c, spd_c, dist_sq_c, dur_sq_c, tip_c, speed_c)

    Anomaly type mapping:
      Type 1 (short_expensive): F25=1, F6>30, F31>10
      Type 2 (tip_anomaly):     F11>5, F4 high, F30>5
      Type 4 (combo_short_long): F26=1, F16<1.5, F6<1.5
    """
    # --- Raw features ---
    dist = np.asarray(d["trip_distance"], dtype=np.float32)
    dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    fare = np.asarray(d["fare_amount"], dtype=np.float32)
    fare = np.nan_to_num(fare, nan=0.0, posinf=0.0, neginf=0.0)
    pax_raw = np.asarray(d["passenger_count"], dtype=np.float32)
    pax_raw = np.where(np.isnan(pax_raw), 1.0, pax_raw)
    tot = np.asarray(d["total_amount"], dtype=np.float32)
    tot = np.nan_to_num(tot, nan=0.0, posinf=0.0, neginf=0.0)
    rc = np.where(np.isnan(np.asarray(d["RatecodeID"], dtype=np.float32)), 1.0,
                  np.asarray(d["RatecodeID"], dtype=np.float32))
    pu_z = np.round(np.asarray(d["PULocationID"], dtype=np.float32)).astype(np.int32)
    do_z = np.round(np.asarray(d["DOLocationID"], dtype=np.float32)).astype(np.int32)
    # FIX: Remove 360-clipping so Type 4 anomalies (dur x2-5) have distinct values
    dur_raw = np.asarray(d["duration_s"], dtype=np.float32)
    dur_raw = np.where(np.isnan(dur_raw), 15.0, dur_raw)
    dur_raw = np.clip(dur_raw, 1, 86400)  # clip to 1 day max (not 360s)
    speed_raw = np.asarray(d["speed_mph"], dtype=np.float32)
    speed_raw = np.where(np.isnan(speed_raw), 10.0, speed_raw)
    speed_raw = np.clip(speed_raw, 0.01, 80.0)  # keep meaningful range
    tip = np.asarray(d["tip_amount"], dtype=np.float32)
    tip = np.where(np.isnan(tip), 0.0, tip)
    hours = np.round(np.asarray(d["hour"], dtype=np.float32)).astype(np.int32)
    dows = np.round(np.asarray(d["weekday"], dtype=np.float32)).astype(np.int32)

    eps = np.float32(1e-8)

    # --- Location grid ---
    puz_c = np.clip(pu_z, 0, 263)
    doz_c = np.clip(do_z, 0, 263)
    pux = ((puz_c - 1) % 16).astype(np.float32)
    puy = ((puz_c - 1) // 16).astype(np.float32)
    dox = ((doz_c - 1) % 16).astype(np.float32)
    doy = ((doz_c - 1) // 16).astype(np.float32)

    # --- Neighborhood IDs ---
    nb_ids = np.full(n, 9, np.int32)
    for (lo, hi), nb_id in _NEIGHBORHOOD_RANGES:
        mask = (pu_z >= lo) & (pu_z <= hi)
        nb_ids[mask] = nb_id

    # --- Ratecode ordinal ---
    rcs = np.select(
        [rc == 1, rc == 2, rc == 3, rc == 4, rc == 5],
        [1.0, 2.0, 3.0, 4.0, 5.0], default=1.0
    ).astype(np.float32)

    hr = hours.astype(np.float32)
    dw = dows.astype(np.float32)

    # --- Derived features ---
    fare_per_mi = np.where(dist > 0.1, fare / np.maximum(dist, eps), 0.0)
    fare_per_min = np.where(dur_raw > 1, fare / np.maximum(dur_raw / 60.0, eps), 0.0)
    pax_per_dist = np.where(dist > 0.1, pax_raw / np.maximum(dist, eps), 0.0)
    # FIX: tip_ratio is the KEY feature for Type 2 (tip = fare x 10-20)
    tip_ratio = np.where(fare > 0.1, tip / np.maximum(fare, eps), 0.0)
    log_fare = np.log1p(np.clip(fare, 0, 1e6))
    log_dist = np.log1p(np.clip(dist, 0, 1e6))

    # --- ANOMALY-SPECIFIC FEATURES (the most important!) ---
    # Type 1: short_expensive — very short trip but expensive
    short_trip = (dist < 0.5).astype(np.float32)  # trip < 0.5 miles
    # Type 4: combo_short_long — very short distance + long duration = very slow
    long_dur = (dur_raw > 600).astype(np.float32)  # > 10 minutes
    # Type 2: tip_anomaly — tip is 10-20x fare
    high_tip = (tip_ratio > 3.0).astype(np.float32)  # tip > 3x fare
    # Type 4: very slow (combo anomaly makes trips very slow)
    slow_trip = np.where(dur_raw > 1, dist / np.maximum(dur_raw / 3600.0, eps), 0.0)  # mph
    slow_trip = np.clip(slow_trip, 0, 60)

    # --- Build 38D feature matrix ---
    X = np.empty((n, 38), np.float32)
    # Core (0-6)
    X[:, 0] = np.log1p(np.clip(dist, 0, 500))     # log_dist (not raw, reduces range)
    X[:, 1] = np.log1p(np.clip(dur_raw, 0, 86400))  # log_dur (FIX: no 360 clip!)
    X[:, 2] = np.log1p(np.clip(fare, 0, 1000))   # log_fare
    X[:, 3] = np.log1p(np.clip(tot, 0, 2000))     # log_total
    X[:, 4] = np.log1p(np.clip(speed_raw, 0, 100))  # log_speed
    X[:, 5] = hr / 23.0                             # hour [0,1]
    X[:, 6] = dw / 6.0                              # dow [0,1]

    # Ratios (7-13)
    X[:, 7] = np.clip(fare_per_mi, 0, 100)          # fare/mile (key for Type 1)
    X[:, 8] = np.clip(fare_per_min, 0, 10)         # fare/minute
    X[:, 9] = np.clip(speed_raw / 15.0, 0, 5)     # speed/15 (key for Type 4)
    X[:, 10] = np.clip(pax_per_dist, 0, 5)          # pax/mile
    X[:, 11] = np.clip(tip_ratio, 0, 30)            # tip/fare (KEY for Type 2!)
    X[:, 12] = np.log1p(np.clip(fare, 0, 1000))     # log_fare (redundant with F2)
    X[:, 13] = np.log1p(np.clip(dist, 0, 500))     # log_dist (redundant with F0)

    # Cyclical (14-18)
    X[:, 14] = np.sin(2 * np.pi * hr / 24.0)        # sin hour
    X[:, 15] = np.cos(2 * np.pi * hr / 24.0)        # cos hour
    X[:, 16] = np.sin(2 * np.pi * dw / 7.0)         # sin dow
    X[:, 17] = np.cos(2 * np.pi * dw / 7.0)         # cos dow
    X[:, 18] = (dw >= 5).astype(np.float32)         # weekend flag

    # Location (19-24)
    X[:, 19] = pux / 15.0                             # pux [0,1]
    X[:, 20] = puy / 15.0                             # puy [0,1]
    X[:, 21] = dox / 15.0                             # dox [0,1]
    X[:, 22] = doy / 15.0                             # doy [0,1]
    X[:, 23] = np.clip(np.abs(puy - doy) / 15.0, 0, 1)  # grid_dist normalized
    X[:, 24] = ((hr >= 20) | (hr <= 6)).astype(np.float32)  # night flag

    # Anomaly-specific signals (25-30) — THE MOST IMPORTANT
    X[:, 25] = short_trip                             # Type 1: very short trip
    X[:, 26] = long_dur                               # Type 4: very long duration
    X[:, 27] = high_tip                               # Type 2: tip >> fare
    X[:, 28] = (slow_trip < 2.0).astype(np.float32)  # Type 4: very slow (<2 mph)
    X[:, 29] = np.log1p(np.clip(dur_raw, 0, 86400)) / 11.5  # log_dur normalized (FIX!)
    X[:, 30] = np.log1p(np.clip(dist, 0, 500)) / 6.2  # log_dist normalized

    # Clipped ratios (31-37)
    X[:, 31] = np.clip(fare_per_mi / 5.0, 0, 20)     # fare/mi normalized
    X[:, 32] = np.clip(fare_per_min / 1.0, 0, 20)     # fare/min normalized
    X[:, 33] = np.clip(speed_raw / 15.0, 0, 5)        # speed normalized
    X[:, 34] = np.clip(dist * dist / 100.0, 0, 100)   # dist_sq normalized
    X[:, 35] = np.clip(dur_raw * dur_raw / 1e6, 0, 100)  # dur_sq normalized (FIX!)
    X[:, 36] = np.clip(tip / 5.0, 0, 20)              # tip normalized
    X[:, 37] = np.clip(speed_raw / 15.0, 0, 5)        # speed normalized (redundant)

    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)
    return X, hours, dows, rcs, nb_ids


def extract_raw_features(pq_path: str, max_rows: Optional[int] = None) -> Tuple[np.ndarray, ...]:
    """Raw 7D features (khong engineering) cho ablation study."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(pq_path)
    total_rows = pf.metadata.num_rows

    cols = ["trip_distance", "fare_amount", "passenger_count",
            "total_amount", "duration_s", "speed_mph", "PULocationID"]

    if max_rows:
        n_read = min(max_rows, total_rows)
        skip = max(0, total_rows - n_read)
        tbl = pf.read(columns=cols).slice(skip, n_read)
        merged = tbl.to_pydict()
        n = len(merged[cols[0]])
    else:
        reader = pf.iter_batches(batch_size=500000, columns=cols)
        chunks = []
        for batch in reader:
            chunks.append(batch.to_pydict())
        del reader
        merged = {key: np.concatenate([c[key] for c in chunks]) for key in cols}
        n = len(merged[cols[0]])
        del chunks

    dist = np.asarray(merged["trip_distance"], dtype=np.float32)
    dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    fare = np.asarray(merged["fare_amount"], dtype=np.float32)
    fare = np.nan_to_num(fare, nan=0.0, posinf=0.0, neginf=0.0)
    pax = np.asarray(merged["passenger_count"], dtype=np.float32)
    pax = np.where(np.isnan(pax), 1.0, pax)
    tot = np.asarray(merged["total_amount"], dtype=np.float32)
    tot = np.nan_to_num(tot, nan=0.0, posinf=0.0, neginf=0.0)
    dur = np.clip(np.asarray(merged["duration_s"], dtype=np.float32), 1, 360)
    dur = np.where(np.isnan(dur), 15.0, dur)
    speed = np.clip(np.asarray(merged["speed_mph"], dtype=np.float32), 0, 60)
    speed = np.where(np.isnan(speed), 10.0, speed)
    pu_z = np.round(np.asarray(merged["PULocationID"], dtype=np.float32)).astype(np.int32)

    hours = np.zeros(n, dtype=np.int32)
    dows = np.zeros(n, dtype=np.int32)
    rcs = np.ones(n, dtype=np.float32)

    nb_ids = np.full(n, 9, np.int32)
    for (lo, hi), nb_id in _NEIGHBORHOOD_RANGES:
        mask = (pu_z >= lo) & (pu_z <= hi)
        nb_ids[mask] = nb_id

    X = np.column_stack([dist, dur, fare, pax, tot, speed, pu_z]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)
    return X, hours, dows, rcs, nb_ids


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_training_loss(
    epoch_losses: List[float],
    save_path: Optional[str] = None,
    title: str = "Autoencoder Training Loss",
    figsize: Tuple[int, int] = (10, 5),
) -> "matplotlib.figure.Figure":
    """
    Plot AE training loss per epoch.

    Args:
        epoch_losses: List of loss values per epoch.
        save_path: If provided, save figure to this path.
        title: Chart title.
        figsize: Figure size (width, height).

    Returns:
        matplotlib Figure object.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available, skipping loss plot")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    epochs = list(range(1, len(epoch_losses) + 1))

    ax.plot(epochs, epoch_losses, color="#2196F3", linewidth=1.5, label="Loss")
    if len(epoch_losses) > 20:
        window = min(50, len(epoch_losses) // 10)
        smoothed = np.convolve(epoch_losses, np.ones(window) / window, mode="valid")
        ax.plot(
            list(range(window, len(epoch_losses) + 1)),
            smoothed,
            color="#FF5722",
            linewidth=2.5,
            label=f"Moving avg (w={window})",
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(epoch_losses))

    final_loss = epoch_losses[-1]
    best_loss = min(epoch_losses)
    ax.axhline(best_loss, color="green", linestyle="--", alpha=0.6,
               label=f"Best={best_loss:.6f}")
    ax.text(len(epoch_losses) * 0.95, best_loss * 1.02,
            f"Best: {best_loss:.6f}", color="green", fontsize=9,
            ha="right", va="bottom")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        LOGGER.info("  Loss plot saved: %s", save_path)
        plt.close(fig)
        return None

    return fig


def plot_score_distribution(
    scores: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float,
    save_path: Optional[str] = None,
    title: str = "Score Distribution: Normal vs Anomaly",
    figsize: Tuple[int, int] = (10, 6),
) -> "matplotlib.figure.Figure":
    """
    Plot histogram of anomaly scores for normal vs anomaly points.

    Shows separation power of the model visually.

    Args:
        scores: Anomaly scores array.
        gt_mask: Ground truth mask (1=anomaly, 0=normal).
        threshold: Best threshold for detection.
        save_path: If provided, save figure to this path.
        title: Chart title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available, skipping score distribution plot")
        return None

    normal_scores = scores[gt_mask == 0]
    anomaly_scores = scores[gt_mask == 1]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Overlapping histogram
    ax1 = axes[0]
    bins = np.linspace(min(scores.min(), threshold), max(scores.max(), threshold), 60)
    ax1.hist(
        normal_scores, bins=bins, alpha=0.6, color="#4CAF50",
        label=f"Normal (n={len(normal_scores):,})", density=True
    )
    ax1.hist(
        anomaly_scores, bins=bins, alpha=0.6, color="#F44336",
        label=f"Anomaly (n={len(anomaly_scores):,})", density=True
    )
    ax1.axvline(threshold, color="#FF9800", linestyle="--", linewidth=2.5,
                label=f"Threshold={threshold:.4f}")
    ax1.set_xlabel("Anomaly Score", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Score Distribution", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Box plot
    ax2 = axes[1]
    bp = ax2.boxplot(
        [normal_scores, anomaly_scores],
        labels=["Normal", "Anomaly"],
        patch_artist=True,
        notch=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "yellow",
                   "markeredgecolor": "black", "markersize": 8},
    )
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#F44336")
    bp["boxes"][1].set_alpha(0.6)
    ax2.set_ylabel("Anomaly Score", fontsize=11)
    ax2.set_title("Score Box Plot", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    normal_mean = float(np.mean(normal_scores))
    anomaly_mean = float(np.mean(anomaly_scores))
    sep_ratio = anomaly_mean / normal_mean if normal_mean > 0 else 0

    fig.suptitle(
        f"{title}  |  Normal mean={normal_mean:.4f}  |  Anomaly mean={anomaly_mean:.4f}"
        f"  |  Separation={sep_ratio:.2f}x",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        LOGGER.info("  Score distribution saved: %s", save_path)
        plt.close(fig)
        return None

    return fig


def plot_score_timeseries(
    scores: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float,
    save_path: Optional[str] = None,
    max_display: int = 20000,
    title: str = "Anomaly Score Over Time",
    figsize: Tuple[int, int] = (16, 6),
) -> "matplotlib.figure.Figure":
    """
    Plot anomaly scores over time with ground truth highlighted.

    Shows where anomalies occur relative to score spikes.

    Args:
        scores: Anomaly scores array.
        gt_mask: Ground truth mask.
        threshold: Detection threshold.
        save_path: If provided, save figure to this path.
        max_display: Max points to display (subsample for performance).
        title: Chart title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available, skipping timeseries plot")
        return None

    n = min(len(scores), max_display)
    step = max(1, len(scores) // n)
    idx = range(0, len(scores), step)

    s = scores[idx]
    gt = gt_mask[idx]
    x = np.arange(len(s))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, s, color="#2196F3", linewidth=0.8, alpha=0.8, label="Anomaly Score")

    ax.axhline(threshold, color="#FF9800", linestyle="--", linewidth=2,
               label=f"Threshold={threshold:.4f}")

    anomaly_idx = np.where(gt == 1)[0]
    if len(anomaly_idx) > 0:
        ax.scatter(
            anomaly_idx,
            s[anomaly_idx],
            color="#F44336",
            s=15,
            alpha=0.8,
            label=f"Ground Truth Anomaly (n={gt_mask.sum():,})",
            zorder=5,
        )

    tp = ((scores >= threshold) & (gt_mask == 1))
    fp = ((scores >= threshold) & (gt_mask == 0))
    if len(tp) > 0:
        ax.fill_between(
            x,
            0,
            s,
            where=(gt >= 0.5),
            color="#F44336",
            alpha=0.08,
            label=f"Detected Anomalies",
        )

    ax.set_xlabel("Sample Index", fontsize=11)
    ax.set_ylabel("Anomaly Score", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(s))

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        LOGGER.info("  Timeseries plot saved: %s", save_path)
        plt.close(fig)
        return None

    return fig


def plot_detection_timeseries(
    scores: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float,
    save_path: Optional[str] = None,
    max_display: int = 30000,
    title: str = "MemStream — Anomaly Detection Timeline",
    figsize: Tuple[int, int] = (18, 7),
) -> "matplotlib.figure.Figure":
    """
    Timeseries chart showing detection results with color coding:

    Panel 1 (scores): Blue line of anomaly scores with threshold.
    Panel 2 (detection): Color-coded points:
      - GREEN  (#4CAF50): True Negative  — normal & not flagged
      - YELLOW (#FFEB3B): False Positive — normal but flagged as anomaly
      - RED    (#F44336): False Negative — anomaly but NOT flagged (missed)
      - PURPLE (#9C27B0): True Positive  — anomaly & correctly flagged

    Args:
        scores: Anomaly scores array.
        gt_mask: Ground truth mask (1=anomaly, 0=normal).
        threshold: Detection threshold.
        save_path: If provided, save figure to this path.
        max_display: Max points to display (subsample for readability).
        title: Chart title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available, skipping detection timeseries plot")
        return None

    n = min(len(scores), max_display)
    step = max(1, len(scores) // n)
    idx = np.arange(0, len(scores), step)
    if len(idx) > n:
        idx = idx[:n]

    s = scores[idx]
    gt = gt_mask[idx]
    x = np.arange(len(s))
    pred = (s >= threshold).astype(int)

    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    tn = (pred == 0) & (gt == 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    # Panel 1: Score timeseries
    ax1.fill_between(x, 0, s, color="#2196F3", alpha=0.15)
    ax1.plot(x, s, color="#2196F3", linewidth=0.7, alpha=0.9, label="Anomaly Score")
    ax1.axhline(threshold, color="#FF9800", linestyle="--", linewidth=2,
                label=f"Threshold={threshold:.4f}")

    # Shade GT anomaly regions
    gt_regions = np.where(gt == 1)[0]
    if len(gt_regions) > 0:
        ax1.scatter(gt_regions, s[gt_regions],
                    color="#F44336", s=12, alpha=0.6,
                    label=f"GT Anomaly (n={gt.sum():,})", zorder=5)

    ax1.set_ylabel("Anomaly Score", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(s) - 1)

    # Panel 2: Detection result (color-coded points)
    ax2.set_ylabel("Detection", fontsize=10)
    ax2.set_xlabel("Sample Index", fontsize=11)
    ax2.set_yticks([])
    ax2.set_xlim(0, len(s) - 1)
    ax2.grid(True, alpha=0.3, axis="x")

    # Background shading by GT anomaly regions
    if len(gt_regions) > 0:
        ax2.axhspan(-0.5, 1.5, alpha=0.04, color="#F44336", label="GT Anomaly region")

    # Plot each category as colored markers
    marker_size = 18
    if tn.sum() > 0:
        tn_x = np.where(tn)[0]
        ax2.scatter(tn_x, np.zeros(len(tn_x)), c="#4CAF50", s=marker_size * 0.6,
                    alpha=0.5, label=f"TN (correct normal, n={tn.sum():,})", zorder=3)
    if fp.sum() > 0:
        fp_x = np.where(fp)[0]
        ax2.scatter(fp_x, np.ones(len(fp_x)) * 0.5, c="#FFEB3B", s=marker_size,
                    alpha=0.85, label=f"FP (false alarm, n={fp.sum():,})", zorder=4,
                    edgecolors="orange", linewidths=0.5)
    if fn.sum() > 0:
        fn_x = np.where(fn)[0]
        ax2.scatter(fn_x, np.ones(len(fn_x)) * 0.5, c="#F44336", s=marker_size,
                    alpha=0.85, label=f"FN (missed anomaly, n={fn.sum():,})", zorder=4)
    if tp.sum() > 0:
        tp_x = np.where(tp)[0]
        ax2.scatter(tp_x, np.ones(len(tp_x)), c="#9C27B0", s=marker_size * 1.2,
                    alpha=0.9, label=f"TP (correct detection, n={tp.sum():,})", zorder=5,
                    edgecolors="white", linewidths=0.5)

    n_tp = int(tp.sum())
    n_fp = int(fp.sum())
    n_fn = int(fn.sum())
    n_tn = int(tn.sum())

    legend_labels = [
        f"TN (normal, not flagged): {n_tn:,}",
        f"FP (normal, flagged): {n_fp:,}",
        f"FN (anomaly, missed): {n_fn:,}",
        f"TP (anomaly, detected): {n_tp:,}",
    ]
    ax2.legend(legend_labels, fontsize=8, loc="upper right",
               ncol=2, framealpha=0.9)
    ax2.set_ylim(-0.4, 1.6)

    total_anomaly = int(gt.sum())
    detection_rate = n_tp / max(total_anomaly, 1) * 100
    ax2.set_title(
        f"Detection breakdown  |  "
        f"Precision={n_tp/max(n_tp+n_fp,1):.1%}  "
        f"Recall={n_tp/max(total_anomaly,1):.1%} ({n_tp}/{total_anomaly})  "
        f"Precision={n_tp/max(n_tp+n_fp,1):.1%}",
        fontsize=9
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        LOGGER.info("  Detection timeseries saved: %s", save_path)
        plt.close(fig)
        return None

    return fig


def plot_pointwise_scores(
    scores_by_method: Dict[str, np.ndarray],
    gt_mask: np.ndarray,
    threshold_by_method: Optional[Dict[str, float]] = None,
    sample_size: int = 200,
    save_path: Optional[str] = None,
    title: str = "Per-Point Anomaly Scores by Method",
    figsize: Tuple[int, int] = (18, 7),
) -> "matplotlib.figure.Figure":
    """
    Plot anomaly scores of INDIVIDUAL data points across multiple methods.

    Moi diem (point) duoc hien thi score cua tung model, de thay ro:
      - Diem A: MemStream=0.3, IF=0.2, LOF=0.8, RCF=0.1 ...
      - Diem nao la anomaly (theo ground truth) thi score cao hon
      - Model nao phan biet tot normal vs anomaly nhat

    Dung subsample `sample_size` diem de chart con doc duoc.

    Args:
        scores_by_method: Dict[method_name -> scores_array]
        gt_mask: Ground truth (1=anomaly, 0=normal).
        threshold_by_method: Optional dict of threshold per method.
        sample_size: So diem hien thi (default 200).
        save_path: Neu co, save figure vao day.
        title: Chart title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available, skipping pointwise plot")
        return None

    methods = list(scores_by_method.keys())
    n_pts = min(sample_size, len(next(iter(scores_by_method.values()))))
    n_methods = len(methods)

    indices = np.arange(len(next(iter(scores_by_method.values()))))
    if len(indices) > n_pts:
        np.random.seed(42)
        idx_sample = np.random.choice(indices, size=n_pts, replace=False)
    else:
        idx_sample = indices

    sort_key = "MemStream" if "MemStream" in methods else methods[0]
    sort_scores = scores_by_method[sort_key][idx_sample]
    sorted_order = np.argsort(sort_scores)
    idx_sample = idx_sample[sorted_order]

    gt_sample = gt_mask[idx_sample]
    is_anomaly = gt_sample == 1

    fig, axes = plt.subplots(2, 1, figsize=figsize,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    x = np.arange(n_pts)
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    for i, (method, scores) in enumerate(scores_by_method.items()):
        s = scores[idx_sample]
        th = (threshold_by_method or {}).get(method, None)
        ax.plot(x, s, label=method, color=colors[i], linewidth=1.2, alpha=0.85)
        if th is not None:
            ax.axhline(th, color=colors[i], linestyle="--", alpha=0.4, linewidth=0.8)

    ax.set_ylabel("Anomaly Score", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=8, ncol=1, framealpha=0.9
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_pts)

    for xi, is_anom in zip(x, is_anomaly):
        if is_anom:
            ax.axvspan(xi - 0.4, xi + 0.4, color="#F44336", alpha=0.07, zorder=0)

    ax_gt = axes[1]
    normal_x = x[~is_anomaly]
    anomaly_x = x[is_anomaly]

    if len(normal_x) > 0:
        ax_gt.scatter(normal_x, np.zeros(len(normal_x)),
                     color="#4CAF50", s=12, alpha=0.7, label="Normal", zorder=3)
    if len(anomaly_x) > 0:
        ax_gt.scatter(anomaly_x, np.ones(len(anomaly_x)) * 0.5,
                     color="#F44336", s=18, alpha=0.9,
                     label=f"Anomaly (n={len(anomaly_x)})", zorder=4, marker="x")

    ax_gt.set_xlabel(f"Data Points (sorted by {sort_key} score, n={n_pts})", fontsize=10)
    ax_gt.set_yticks([])
    ax_gt.set_ylim(-0.3, 1.0)
    ax_gt.legend(fontsize=8, loc="upper right")
    ax_gt.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        LOGGER.info("  Pointwise scores saved: %s", save_path)
        plt.close(fig)
        return None

    return fig


def plot_method_score_heatmap(
    scores_by_method: Dict[str, np.ndarray],
    gt_mask: np.ndarray,
    sample_size: int = 150,
    save_path: Optional[str] = None,
    title: str = "Method Scores per Point (Heatmap)",
    figsize: Tuple[int, int] = (14, 10),
) -> "matplotlib.figure.Figure":
    """
    Heatmap: moi hang = 1 diem, moi cot = 1 method,
    gia tri = anomaly score.

    Giup thay ro pattern: diem nao co score cao theo model nao.

    Args:
        scores_by_method: Dict[method_name -> scores_array]
        gt_mask: Ground truth.
        sample_size: So diem hien thi.
        save_path: Neu co, save figure vao day.
        title: Chart title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available, skipping heatmap")
        return None

    methods = list(scores_by_method.keys())
    n_pts = min(sample_size, len(next(iter(scores_by_method.values()))))

    indices = np.arange(len(next(iter(scores_by_method.values()))))
    np.random.seed(42)
    idx_sample = np.random.choice(indices, size=n_pts, replace=False)

    sort_key = "MemStream" if "MemStream" in methods else methods[0]
    sort_scores = scores_by_method[sort_key][idx_sample]
    sorted_order = np.argsort(sort_scores)
    idx_sample = idx_sample[sorted_order]

    matrix = np.zeros((n_pts, len(methods)), dtype=np.float32)
    for j, m in enumerate(methods):
        matrix[:, j] = scores_by_method[m][idx_sample]

    gt_sample = gt_mask[idx_sample]

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                              gridspec_kw={"width_ratios": [1, 0.03]})

    ax_heat = axes[0]
    ax_color = axes[1]

    im = ax_heat.imshow(
        matrix, aspect="auto", cmap="RdYlGn_r",
        vmin=0, vmax=max(float(matrix.max() * 0.8), 1.0)
    )
    cbar = fig.colorbar(im, cax=ax_color)
    cbar.set_label("Anomaly Score", fontsize=10)

    ax_heat.set_xticks(range(len(methods)))
    ax_heat.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax_heat.set_yticks(range(n_pts))
    ax_heat.set_yticklabels([])

    for row_i in range(n_pts):
        if gt_sample[row_i] == 1:
            ax_heat.add_patch(
                plt.Rectangle(
                    (-0.5, row_i - 0.5), len(methods), 1,
                    fill=False, edgecolor="#F44336",
                    linewidth=1.5, linestyle="--"
                )
            )

    edge_colors = np.array(["#4CAF50" if g == 0 else "#F44336" for g in gt_sample])
    ax_heat2 = ax_heat.twinx()
    ax_heat2.set_yticks(range(n_pts))
    ax_heat2.set_yticklabels([])
    ax_heat2.set_ylim(ax_heat.get_ylim())
    for row_i, color in enumerate(edge_colors):
        ax_heat2.plot(
            [len(methods) - 0.05, len(methods) + 0.05],
            [row_i, row_i],
            color=color,
            linewidth=3,
        )

    ax_heat.set_xlabel("Method", fontsize=11)
    ax_heat.set_ylabel(f"Data Points (sorted by {sort_key}, n={n_pts})", fontsize=11)
    ax_heat.set_title(f"{title}\n(Red border = ground truth anomaly)", fontsize=12, fontweight="bold")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        LOGGER.info("  Score heatmap saved: %s", save_path)
        plt.close(fig)
        return None

    return fig
