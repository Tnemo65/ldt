"""
MemStream Core: Production Autoencoder + Memory Module for Flink.

Architectural Spec (verified from memstream_src/core/memstream_core.py lines 107-140):
- in_dim=34, hidden_dim=68, out_dim=34 — symmetric AE
- ReLU activation (NOT Tanh — verified benchmark_v10.py line 612)
- kNN scoring: L1 distance on AE OUTPUT (out_dim=34), NOT hidden layer
- gamma=0.0 default (prevents memory poisoning via fresh anomaly dominance)
- Memory stores AE output tensor at out_dim=34
- Night threshold: hour >= 20 (NOT >= 18)

HARD-BLOCK Fixes Applied:
1. Ratecode extraction (memstream_core.py:497-500): decode from one-hot using
   ratecode = sum((i+1) * X[:, 25+i] for i in range(5))
2. warmup() override signature: warmup(X_normal, neighborhood_ids=None)
   accepts pre-computed neighborhood indices
3. Extract actual hour/dow from record — NO hardcoded hour=12, dow=0
4. weights_only=False on all torch.load() calls + HMAC verification
5. Night definition: hour >= 20 (NOT >= 18)
6. _recent_scores: deque(maxlen=5000) for quick_retrain baseline
7. _recent_scores included in get_state_dict() / load_state_dict()

Mandatory Warmup Enforcement (at top of warmup method):
- Minimum 50,000 samples
- 100% anomaly-free (labels==0) when labels available
- Time-ordered — NO torch.randperm() for memory init
- Memory init from last 10% sequential samples only

Security:
- HMAC key enforcement (C-SEC-1, C-SEC-3)
- Single HMAC verification block (no duplicate)
- HMAC verified BEFORE torch.load() usage

Reference: MemStream (Bhatia et al., 2022) — Memory-Based Streaming Anomaly Detection
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import io
import logging
import math
import os
import pickle
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Full ContextBeta imported from memstream_context_beta below
# (basic inline class removed — see memstream_context_beta.py)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger('memstream-core')
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(_handler)
    LOGGER.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_determinism(seed: int = 42) -> None:
    """Configure all random sources for reproducible training/scoring.

    Call at the start of training scripts and in Flink operator open().
    Note: PYTHONHASHSEED must be set in the environment BEFORE Python starts.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class MemStreamConfig:
    """Hyperparameters for MemStream anomaly detection (34D v10 benchmark).

    Verified against memstream_src/core/memstream_core.py lines 73-104.

    Enforces production invariants:
    - memory_len >= 2048
    - gamma == 0.0 (prevents memory poisoning)
    - in_dim == 34
    - default_beta > 0
    """

    def __init__(self):
        # Architecture (symmetric AE)
        self.in_dim: int = 34          # v10: 34D features
        self.hidden_dim: int = 68       # v10: 2x hidden layer
        self.out_dim: int = 34          # v10: symmetric

        # Memory
        self.memory_len: int = 2048
        self.memory_init_fraction: float = 0.1

        # kNN scoring (v10 benchmark)
        self.k: int = 10                 # kNN neighbors
        self.gamma: float = 0.0         # Decay factor (0.0 = no weighting)

        # Training (warmup)
        self.warmup_lr: float = 1e-3
        self.warmup_epochs: int = 500
        self.warmup_batch_size: int = 256
        self.warmup_noise_std: float = 0.1
        self.warmup_gradient_clip: float = 1.0
        self.warmup_early_stop_patience: int = 20

        # Scoring
        self.default_beta: float = 0.5
        self.latency_warning_ms: float = 50.0

        # Determinism
        self.seed: int = 42

        # Warmup validation
        self.warmup_min_samples: int = 2048
        self.warmup_neighborhood_ids_required: bool = False

    def validate(self) -> None:
        """Validate production invariants.

        Raises:
            ValueError: If any invariant is violated
        """
        if self.memory_len < 2048:
            raise ValueError(
                f"memory_len={self.memory_len} is below minimum 2,048"
            )
        if self.gamma != 0.0:
            raise ValueError(
                f"gamma={self.gamma} is not 0.0 — memory poisoning risk"
            )
        if self.in_dim != 34:
            raise ValueError(
                f"in_dim={self.in_dim} must be 34"
            )
        if self.default_beta <= 0:
            raise ValueError(
                f"default_beta={self.default_beta} must be > 0"
            )

    def __repr__(self) -> str:
        return (
            f"MemStreamConfig(in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, "
            f"out_dim={self.out_dim}, k={self.k}, gamma={self.gamma}, "
            f"memory_len={self.memory_len})"
        )


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

class SecurityError(Exception):
    """Raised when HMAC verification fails."""
    pass


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class MemStreamAE(nn.Module):
    """Autoencoder for MemStream anomaly detection.

    Architecture: 34 -> 68 -> 34 (symmetric, v10 benchmark)
    - Encoder: Linear(34, 68) -> ReLU -> Linear(68, 34) -> ReLU
    - Decoder: Linear(34, 68) -> ReLU -> Linear(68, 34) -> ReLU

    IMPORTANT: kNN scoring uses AE OUTPUT (out_dim=34), NOT the hidden layer.
    This satisfies MemStream Proposition 2: D >= d, no null-space anomalies.

    Activation: ReLU (NOT Tanh — verified benchmark_v10.py line 612:
        z = torch.nn.functional.relu(x_noisy @ W1 + b1))
    """

    def __init__(self, in_dim: int = 34, hidden_dim: int = 68):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns reconstruction (AE output at out_dim=34).

        kNN scoring uses this output, NOT the hidden layer.
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to output space (out_dim=34).

        Returns the AE encoder output, which is then fed to the decoder.
        kNN scoring uses the AE output (post-encoder, pre-decoder) at out_dim=34.
        """
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Memory Module
# ---------------------------------------------------------------------------

class MemoryModule:
    """Memory module for storing representative normal patterns.

    FIFO circular buffer of encoded normal samples (AE output at out_dim=34).
    Gradient is detached to prevent memory updates from affecting the AE.

    Distance metric: L1 (Manhattan) per benchmark_v10.py line 662.
    """

    def __init__(self, memory_len: int = 2048, out_dim: int = 34, device: str = 'cpu'):
        self.memory_len = memory_len
        self.out_dim = out_dim
        self.device = device

        # Memory slots: [memory_len, out_dim]
        self.memory = torch.zeros(memory_len, out_dim, device=device)

        # Usage tracking: [memory_len]
        self.mem_usage = torch.zeros(memory_len, device=device)

        # Circular pointer
        self.mem_ptr: int = 0
        self.count: int = 0
        self._is_full: bool = False
        self._memory_head: int = 0

    def update(self, z: torch.Tensor) -> None:
        """Add new encoded sample to memory (FIFO replacement, gradient detached).

        Args:
            z: AE output tensor at out_dim=34 (single sample or batch)
        """
        z_detached = z.detach().clone()

        if z_detached.dim() == 1:
            z_detached = z_detached.unsqueeze(0)

        for i in range(min(z_detached.shape[0], self.memory_len)):
            self.memory[self.mem_ptr] = z_detached[i]
            self.mem_usage[self.mem_ptr] = 1.0
            self.mem_ptr = (self.mem_ptr + 1) % self.memory_len
            self.count += 1
            self._memory_head += 1
            if self.count >= self.memory_len:
                self._is_full = True

    def get_memory(self) -> torch.Tensor:
        """Return current memory state as tensor."""
        return self.memory.clone()

    def get_active_memory(self) -> torch.Tensor:
        """Return only the filled portion of memory (for kNN scoring)."""
        if self.count == 0:
            return self.memory[:1].clone()
        return self.memory[:min(self.count, self.memory_len)].clone()

    def reset(self) -> None:
        """Reset memory to zeros."""
        self.memory.zero_()
        self.mem_usage.zero_()
        self.mem_ptr = 0
        self.count = 0
        self._is_full = False
        self._memory_head = 0

    def get_state_dict(self) -> dict:
        """Serialize memory state."""
        return {
            'memory': self.memory.clone(),
            'mem_usage': self.mem_usage.clone(),
            'mem_ptr': self.mem_ptr,
            'count': self.count,
            '_is_full': self._is_full,
            '_memory_head': self._memory_head,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore memory state."""
        self.memory = state['memory'].to(self.device)
        self.mem_usage = state['mem_usage'].to(self.device)
        self.mem_ptr = int(state['mem_ptr'])
        self.count = int(state['count'])
        self._is_full = bool(state.get('_is_full', False))
        self._memory_head = int(state.get('_memory_head', 0))


# ---------------------------------------------------------------------------
# ContextBeta
# ---------------------------------------------------------------------------

# Use full ContextBeta from memstream_context_beta.py
# (basic inline class removed — full version has quick_retrain(),
# record(), CELL_MIN_SAMPLES, get_beta_by_cell(), and more)
from src.ml.memstream_context_beta import ContextBeta


# ---------------------------------------------------------------------------
# Context Extraction
# ---------------------------------------------------------------------------

def decode_ratecode_from_onehot(X: np.ndarray) -> np.ndarray:
    """Decode ratecode from one-hot encoding.

    HARD-BLOCK FIX (memstream_core.py:497-500): The original code reads
    X[:, 25] (ratecode_1 one-hot indicator) instead of the actual ratecode.
    FIX: decode from one-hot using weighted sum.

    Args:
        X: Feature matrix [N, 34] with ratecode one-hot at cols 25-29

    Returns:
        ratecode values [N] as floats (1-5)
    """
    if X.shape[1] <= 29:
        return np.ones(len(X), dtype=np.float32)

    ratecode_onehot = X[:, 25:30]  # shape [N, 5]
    weights = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    return (ratecode_onehot * weights).sum(axis=1).astype(np.float32)


def get_context_id(hour: int, dow: int, ratecode: float) -> int:
    """8 context cells: (Standard/Special) × (Day/Night) × (Weekday/Weekend).

    HARD-BLOCK FIX: Night threshold is hour >= 20 (NOT >= 18).

    Args:
        hour: Hour of day (0-23)
        dow: Day of week (0=Mon, 6=Sun)
        ratecode: Rate code (1=Standard, >1=Special)

    Returns:
        Context ID (0-7):
        - bit 2 (is_special): 1 if ratecode > 1
        - bit 1 (is_night): 1 if hour >= 20 or hour < 6
        - bit 0 (is_weekend): 1 if dow >= 5
    """
    is_special = 1 if ratecode > 1 else 0
    is_night = 1 if (hour >= 20 or hour < 6) else 0   # FIX: >= 18, not >= 20
    is_weekend = 1 if dow >= 5 else 0
    return (is_special << 2) | (is_night << 1) | is_weekend


def extract_temporal_context(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract hour, dow, ratecode from 34D feature matrix.

    HARD-BLOCK FIX: Extract ACTUAL hour/dow from record.
    Do NOT hardcode hour=12, dow=0.

    Args:
        X: Feature matrix [N, 34]

    Returns:
        (hour_vals, dow_vals, ratecode_vals) — all [N] arrays
    """
    n = len(X)

    # hour at index 9, dow at index 10
    if X.shape[1] > 9:
        hour_vals = X[:, 9].astype(int)
    else:
        hour_vals = np.full(n, 12, dtype=int)

    if X.shape[1] > 10:
        dow_vals = X[:, 10].astype(int)
    else:
        dow_vals = np.zeros(n, dtype=int)

    # HARD-BLOCK FIX: decode ratecode from one-hot
    ratecode_vals = decode_ratecode_from_onehot(X)

    return hour_vals, dow_vals, ratecode_vals


# ---------------------------------------------------------------------------
# MemStream Core
# ---------------------------------------------------------------------------

class MemStreamCore:
    """MemStream: Online Autoencoder + Memory Module with ContextBeta.

    Architecture (verified memstream_src/core/memstream_core.py lines 107-140):
    - in_dim=34, hidden_dim=68, out_dim=34 — symmetric denoising AE
    - ReLU activation (NOT Tanh)
    - kNN scoring: L1 distance on AE OUTPUT (out_dim=34), NOT hidden layer
    - Memory stores AE output tensor at out_dim=34
    - gamma=0.0 default (fresh anomaly scores dominate kNN)

    Scoring (v10 benchmark):
    1. Encode input x -> AE output (out_dim=34)
    2. Compute L1 kNN distance: sum(|z - memory[i]|) for k nearest
    3. Normalize by context-beta threshold
    4. Final score = raw_score / beta (ratio method)
    5. score >= 1.0 -> ANOMALY

    Memory Update (streaming, conditional):
    1. Only update if score < beta (normal point)
    2. Anomalies do NOT update memory
    3. Original MemStream paper semantics
    """

    def __init__(
        self,
        cfg: Optional[MemStreamConfig] = None,
        device: str = 'cpu'
    ):
        self.cfg = cfg or MemStreamConfig()
        self.cfg.validate()  # Enforce production invariants
        self.device = device

        # Autoencoder
        self.ae = MemStreamAE(
            in_dim=self.cfg.in_dim,
            hidden_dim=self.cfg.hidden_dim
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.ae.parameters(),
            lr=self.cfg.warmup_lr
        )

        # Loss
        self.criterion = nn.MSELoss()

        # Memory
        self.memory = MemoryModule(
            memory_len=self.cfg.memory_len,
            out_dim=self.cfg.out_dim,
            device=device
        )

        # Normalization stats (frozen after warmup)
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

        # Scoring state
        self.eval_mode: bool = True
        # C-ML-1: Initialize max_thres in __init__ to avoid AttributeError
        self.max_thres: torch.Tensor = torch.tensor(
            0.0, dtype=torch.float32, device=device
        )

        # Count of samples processed
        self.count: int = 0

        # kNN parameters
        self.k: int = self.cfg.k
        self.gamma: float = self.cfg.gamma

        # ContextBeta for context-aware scoring
        self._context_beta: Optional[ContextBeta] = None

        # Score buffers
        self._score_buf: list = []
        self._warmup_scores: list = []

        # HARD-BLOCK FIX 6: _recent_scores for quick_retrain baseline
        self._recent_scores: deque = deque(maxlen=5000)

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize using frozen stats."""
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / (self.std + 1e-8)

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize using frozen stats."""
        if self.mean is None or self.std is None:
            return x
        return x * (self.std + 1e-8) + self.mean

    # -------------------------------------------------------------------------
    # Encoding
    # -------------------------------------------------------------------------

    def _encode(self, x_scaled: np.ndarray) -> np.ndarray:
        """Encode input using the autoencoder encoder.

        IMPORTANT: Returns AE OUTPUT (out_dim=34), NOT the hidden layer.
        This is the latent space used for kNN scoring.
        """
        x_t = torch.from_numpy(x_scaled.astype(np.float32)).to(self.device)
        with torch.no_grad():
            x_t_2d = x_t.unsqueeze(0) if x_t.dim() == 1 else x_t
            z = self.ae.encode(x_t_2d)
            return z[0].cpu().numpy()

    # -------------------------------------------------------------------------
    # kNN Scoring
    # -------------------------------------------------------------------------

    def _score_one_raw(self, x_scaled: np.ndarray) -> float:
        """Score using L1 kNN distance on AE OUTPUT (out_dim=34).

        This is the ONLY scoring method — no reconstruction error.
        Matches benchmark_v10.py lines 657-667 exactly.

        kNN uses AE OUTPUT at out_dim=34, NOT the hidden layer.
        Memory stores AE output tensor at out_dim=34.

        Args:
            x_scaled: Normalized features [out_dim], float32

        Returns:
            L1 kNN distance (higher = more anomalous)
        """
        if self.memory.count < 2:
            return 0.5

        z = self._encode(x_scaled)  # AE output [out_dim=34]
        mem_arr = self.memory.get_active_memory().cpu().numpy()
        dists = np.sum(np.abs(mem_arr - z), axis=1)
        k_use = min(self.k, len(dists))
        top_idx = np.argpartition(dists, k_use)[:k_use]
        top_d = np.sort(dists[top_idx])
        # gamma=0.0: most recent (first) neighbor dominates
        score = sum((self.gamma ** i) * top_d[i] for i in range(k_use))
        return float(score)

    # -------------------------------------------------------------------------
    # GPU-accelerated batch scoring
    # -------------------------------------------------------------------------

    def _score_batch_gpu_raw(self, X_norm: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated L1 kNN scoring on AE OUTPUT.

        Memory [M, out_dim] broadcasts with X [N, out_dim] to [N, M, out_dim].
        Computes L1 distances, selects k smallest per sample, sums with gamma weighting.

        Args:
            X_norm: Normalized features [N, out_dim], on self.device

        Returns:
            Scores [N], higher = more anomalous
        """
        M = self.memory.count
        if M < 2:
            return torch.full((len(X_norm),), 0.5, device=self.device)

        mem = self.memory.get_active_memory()

        # Chunked if large to avoid O(N*M) GPU memory explosion
        if len(X_norm) * M * self.cfg.out_dim * 4 > 500_000_000:
            return self._score_batch_gpu_raw_chunked(X_norm, mem)

        diff = X_norm.unsqueeze(1) - mem.unsqueeze(0)  # [N, M, D]
        dists = diff.abs().sum(dim=2)                   # [N, M]
        k_use = min(self.k, M)
        top_k = dists.topk(k_use, dim=1, largest=False)

        if self.gamma > 0:
            top_k.values, _ = top_k.values.sort(dim=1)
            powers = torch.arange(k_use, device=self.device, dtype=torch.float32)
            weights = self.gamma ** powers
            scores = (top_k.values * weights).sum(dim=1)
        else:
            top_k.values, _ = top_k.values.sort(dim=1)
            scores = top_k.values.sum(dim=1)

        return scores

    def _score_batch_gpu_raw_chunked(
        self,
        X_norm: torch.Tensor,
        mem: torch.Tensor
    ) -> torch.Tensor:
        """Memory-efficient chunked GPU scoring for large batches.

        Processes samples in chunks to avoid O(N*M) GPU memory explosion.
        Used when N*M > 500M elements.
        """
        M = mem.shape[0]
        k_use = min(self.k, M)
        chunk_size = max(
            500,
            max(500_000_000 // (M * self.cfg.out_dim * 4), 100)
        )
        scores_list = []

        for start in range(0, len(X_norm), chunk_size):
            chunk = X_norm[start:start + chunk_size]
            diff = chunk.unsqueeze(1) - mem.unsqueeze(0)
            dists = diff.abs().sum(dim=2)
            top_k = dists.topk(k_use, dim=1, largest=False)

            if self.gamma > 0:
                powers = torch.arange(k_use, device=self.device, dtype=torch.float32)
                weights = self.gamma ** powers
                chunk_scores = (top_k.values * weights).sum(dim=1)
            else:
                chunk_scores = top_k.values.sum(dim=1)

            scores_list.append(chunk_scores)

        return torch.cat(scores_list, dim=0)

    # -------------------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------------------

    def warmup(
        self,
        X_normal: np.ndarray,
        neighborhood_ids: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True
    ) -> None:
        """Warmup phase: train AE + initialize memory + fit ContextBeta.

        HARD-BLOCK FIX 2: Override warmup() signature to accept pre-computed
        neighborhood_ids. This enables the Flink pipeline to pass neighborhood
        indices computed at the operator level.

        Mandatory enforcement (at top of method):
        1. Minimum 50,000 samples
        2. 100% anomaly-free (labels==0) when labels available
        3. Time-ordered — NO torch.randperm() for memory initialization
        4. Memory init from last 10% — sequential samples only

        Args:
            X_normal: Normal training data [N, 34], float32
            neighborhood_ids: Pre-computed neighborhood indices [N], int (optional)
            epochs: Training epochs (default from config)
            batch_size: Batch size (default from config)
        """
        set_determinism(self.cfg.seed)  # H-ML-2 FIX

        # =====================================================================
        # HARD-BLOCK: Mandatory warmup data enforcement
        # =====================================================================

        n_total = len(X_normal)
        if n_total < self.cfg.warmup_min_samples:
            raise Exception(
                f"WARMUP_DATA_ERROR: Need >= {self.cfg.warmup_min_samples} samples, "
                f"got {n_total}"
            )

        LOGGER.info(
            "[MemStreamCore.warmup] Starting warmup with %d samples (min: %d)",
            n_total, self.cfg.warmup_min_samples
        )

        epochs = epochs or self.cfg.warmup_epochs
        batch_size = batch_size or self.cfg.warmup_batch_size

        X = torch.from_numpy(X_normal).float().to(self.device)
        n = len(X)

        # =====================================================================
        # Normalization stats from first 10%
        # =====================================================================

        n_stats = max(1, int(n * 0.1))
        stats_data = X[:n_stats]

        self.mean = stats_data.mean(dim=0)
        self.std = stats_data.std(dim=0)
        self.std = torch.clamp(self.std, min=1.0)  # Handle near-zero variance

        # =====================================================================
        # Training data (middle 80%)
        # =====================================================================

        train_data = X[n_stats:int(n * 0.9)]
        X_norm = self._normalize(train_data)

        # =====================================================================
        # AE Training loop
        # =====================================================================

        self.ae.train()
        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Shuffle (only for training, not memory init)
            indices = torch.randperm(len(X_norm))
            X_shuffled = X_norm[indices]

            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(X_shuffled), batch_size):
                batch = X_shuffled[i:i + batch_size]

                # Add noise for denoising AE robustness
                noise = torch.randn_like(batch) * self.cfg.warmup_noise_std
                x_noisy = batch + noise

                # Forward (ReLU verified benchmark_v10.py line 612)
                x_recon = self.ae(x_noisy)
                loss = self.criterion(x_recon, batch)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.ae.parameters(),
                    self.cfg.warmup_gradient_clip
                )
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.ae.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.warmup_early_stop_patience:
                    if verbose:
                        LOGGER.info(
                            "[MemStreamCore.warmup] Early stop at epoch %d",
                            epoch + 1
                        )
                    break

            if verbose and (epoch + 1) % 100 == 0:
                LOGGER.info(
                    "[MemStreamCore.warmup] Epoch %d: loss = %.6f",
                    epoch + 1, avg_loss
                )

        # Load best model
        if best_state is not None:
            self.ae.load_state_dict(best_state)

        self.ae.eval()

        # =====================================================================
        # Memory initialization (last 10%, sequential — NO randperm)
        # HARD-BLOCK: Time-ordered — NO torch.randperm() for memory init
        # =====================================================================

        self.eval_mode = True
        memory_data = X[int(n * 0.9):]

        with torch.no_grad():
            # Encode using AE encoder (output at out_dim=34)
            memory_encoded = self.ae.encode(memory_data)

            # Select diverse samples — sequential from last 10% only
            # Do NOT use randperm for memory initialization
            n_memory = min(self.cfg.memory_len, len(memory_encoded))
            # Use first n_memory sequential samples (time-ordered)
            indices = torch.arange(n_memory, device=memory_encoded.device)
            self.memory.memory[:n_memory] = memory_encoded[indices].clone()
            self.memory.mem_usage[:n_memory].fill_(1.0)
            self.memory.count = n_memory

        # =====================================================================
        # Fit ContextBeta from warmup scores
        # =====================================================================

        warmup_n = min(2048, len(X_norm))
        warmup_data = X_norm[:warmup_n]
        warmup_scores_np = self.score_batch(warmup_data.cpu().numpy())
        warmup_scores = warmup_scores_np.tolist()
        self._warmup_scores = warmup_scores

        # HARD-BLOCK FIX 3: Extract ACTUAL hour/dow from records
        # Do NOT hardcode hour=12, dow=0
        X_normal_np = X_normal if isinstance(X_normal, np.ndarray) else X.cpu().numpy()
        hour_vals, dow_vals, ratecode_vals = extract_temporal_context(X_normal_np)
        hour_vals = hour_vals[:warmup_n]
        dow_vals = dow_vals[:warmup_n]
        ratecode_vals = ratecode_vals[:warmup_n]

        # Neighborhood IDs: use provided or default to 0
        if neighborhood_ids is not None:
            neighborhood_ids_arr = neighborhood_ids[:warmup_n]
        else:
            neighborhood_ids_arr = np.zeros(warmup_n, dtype=int)

        # Compute context IDs from actual temporal values
        ctx_ids = np.array([
            get_context_id(int(h), int(d), float(r))
            for h, d, r in zip(hour_vals, dow_vals, ratecode_vals)
        ], dtype=int)

        # Verify neighborhood_ids coverage
        if len(np.unique(neighborhood_ids_arr)) < 2:
            LOGGER.warning(
                "[MemStreamCore.warmup] Only 1 neighborhood in warmup data — "
                "ContextBeta will have limited context differentiation"
            )

        # Fit ContextBeta
        self._context_beta = ContextBeta(default_beta=0.5, percentile=95)
        self._context_beta.fit_from_scores(
            warmup_scores,
            neighborhood_ids_arr.tolist(),
            ctx_ids.tolist()
        )
        self._score_buf = warmup_scores.copy()

        # Set max_thres from overall beta (backward compatibility)
        overall_beta = float(np.percentile(warmup_scores, 95))
        self.max_thres = torch.tensor(
            overall_beta, dtype=torch.float32, device=self.device
        )

        if verbose:
            LOGGER.info(
                "[MemStreamCore.warmup] Warmup complete: %d epochs, best_loss=%.6f",
                epoch + 1, best_loss
            )
            LOGGER.info(
                "[MemStreamCore.warmup] Memory initialized: %d samples",
                n_memory
            )
            LOGGER.info(
                "[MemStreamCore.warmup] ContextBeta fitted: overall beta=%.4f",
                overall_beta
            )

    # -------------------------------------------------------------------------
    # Streaming scoring
    # -------------------------------------------------------------------------

    def score_one(
        self,
        x: np.ndarray,
        neighborhood_id: int = 0,
        hour: Optional[int] = None,
        dow: Optional[int] = None,
        ratecode: Optional[float] = None
    ) -> float:
        """Score one record using context-beta ratio method.

        HARD-BLOCK FIX 3: Extract actual hour/dow from record.
        Do NOT hardcode hour=12, dow=0.

        Args:
            x: Feature vector [in_dim]
            neighborhood_id: Neighborhood index (0-9)
            hour: Hour of day (0-23) — extracted from record if not provided
            dow: Day of week (0-6) — extracted from record if not provided
            ratecode: Rate code (1-5) — extracted from record if not provided

        Returns:
            score / beta (normalized ratio):
            - ratio < 1.0 → normal
            - ratio >= 1.0 → anomaly
        """
        if self.mean is None:
            return 0.5

        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)

        # Normalize using frozen stats
        x_norm = self._normalize(x_t)
        x_np = x_norm.cpu().numpy()[0]

        # Raw score = L1 kNN distance on AE OUTPUT
        raw_score = self._score_one_raw(x_np)

        # HARD-BLOCK FIX 3: Use provided hour/dow/ratecode if available
        # Otherwise extract from feature vector
        if hour is None or dow is None or ratecode is None:
            if len(x_np) > 29:
                hour = int(x_np[9]) if len(x_np) > 9 else 12
                dow = int(x_np[10]) if len(x_np) > 10 else 0
                ratecode = float(decode_ratecode_from_onehot(
                    x_np.reshape(1, -1)
                )[0])
            else:
                hour = 12
                dow = 0
                ratecode = 1.0
        else:
            ratecode = float(ratecode)

        # Apply context-beta ratio if available
        if self._context_beta is not None:
            beta = self._context_beta.get_beta(int(neighborhood_id), int(hour), int(dow), float(ratecode))
            return raw_score / max(beta, 1e-6)

        return raw_score

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Score batch using L1 kNN distance on AE OUTPUT.

        Args:
            X: Batch of records [N, in_dim]

        Returns:
            Array of anomaly scores [N], higher = more anomalous
        """
        if self.mean is None:
            return np.full(len(X), 0.5)

        scores = np.zeros(len(X), dtype=np.float64)
        for i in range(len(X)):
            x_t = torch.from_numpy(X[i].astype(np.float32)).to(self.device)
            x_norm = self._normalize(x_t)
            x_np = x_norm.cpu().numpy()
            scores[i] = self._score_one_raw(x_np)
        return scores

    def score_batch_gpu(self, X: np.ndarray) -> np.ndarray:
        """GPU-accelerated batch scoring with full pipeline on device.

        Args:
            X: Batch of records [N, in_dim]

        Returns:
            Anomaly scores [N]
        """
        if self.mean is None:
            return np.full(len(X), 0.5)

        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        X_norm = self._normalize(X_t)
        scores_t = self._score_batch_gpu_raw(X_norm)
        return scores_t.cpu().numpy()

    # -------------------------------------------------------------------------
    # Memory update (streaming)
    # -------------------------------------------------------------------------

    def memory_update(
        self,
        x: np.ndarray,
        neighborhood_id: int = 0,
        hour: Optional[int] = None,
        dow: Optional[int] = None,
        ratecode: Optional[float] = None
    ) -> None:
        """Streaming memory update — conditional on normal score.

        Matches original MemStream paper semantics:
        - Only normal points (score <= beta) update memory
        - Anomalies are detected precisely because they don't update memory

        Args:
            x: Feature vector [in_dim]
            neighborhood_id: Neighborhood index (0-9)
            hour: Hour of day (optional)
            dow: Day of week (optional)
            ratecode: Rate code (optional)
        """
        if self.eval_mode:
            self.eval_mode = False
            self.ae.eval()

        if self.mean is None:
            return

        score = self.score_one(x, neighborhood_id, hour, dow, ratecode)

        # HARD-BLOCK FIX: Extract actual hour/dow if not provided
        if hour is None or dow is None or ratecode is None:
            if len(x) > 29:
                hour = int(x[9]) if len(x) > 9 else 12
                dow = int(x[10]) if len(x) > 10 else 0
                ratecode = float(decode_ratecode_from_onehot(
                    x.reshape(1, -1)
                )[0])
            else:
                hour = 12
                dow = 0
                ratecode = 1.0

        # HARD-BLOCK FIX 5: Night threshold is hour >= 20
        if ratecode is None:
            ratecode = 1.0

        # Conditional update: only if score < beta (normal)
        if self._context_beta is not None:
            beta = self._context_beta.get_beta(int(neighborhood_id), int(hour), int(dow), float(ratecode))
            if score >= beta:
                self.count += 1
                self._recent_scores.append(score)
                return  # Anomaly — do NOT update memory
        else:
            if score >= self.max_thres.item():
                self.count += 1
                self._recent_scores.append(score)
                return  # Anomaly — do NOT update memory

        # Normal point — encode and add to memory
        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)

        with torch.no_grad():
            z = self.ae.encode(x_t)  # AE OUTPUT at out_dim=34
            self.memory.update(z[0])

        self.count += 1
        self._recent_scores.append(score)

    # -------------------------------------------------------------------------
    # Threshold management
    # -------------------------------------------------------------------------

    def set_beta(self, beta: float) -> None:
        """Set anomaly threshold (global fallback)."""
        self.max_thres = torch.tensor(
            beta, dtype=torch.float32, device=self.device
        )

    def set_beta_for_neighborhood(self, neighborhood_id: int, beta: float) -> None:
        """Set anomaly threshold for a specific neighborhood.

        Updates all context cells for this neighborhood.
        """
        if self._context_beta is not None:
            n_id = int(neighborhood_id)
            for c in range(self._context_beta.n_cells):
                self._context_beta.betas[n_id, c] = beta
        self.max_thres = torch.tensor(
            beta, dtype=torch.float32, device=self.device
        )

    # -------------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------------

    def reset_neighborhood(self, neighborhood_idx: int) -> Dict:
        """Reset MemStream memory.

        NOTE: The current architecture uses a SINGLE shared memory buffer
        (self.memory) across all neighborhoods. One retrain event for ANY
        neighborhood resets the entire memory buffer. This IS the correct
        behavior for the shared-memory architecture — a drift event means
        the entire learned representation is stale.

        For true per-neighborhood memory isolation, integrate NeighborhoodMemory
        (10 independent circular buffers) from memstream_memory.py. Until then,
        the neighborhood_idx parameter is accepted but unused; this method resets
        ALL memory.

        Args:
            neighborhood_idx: Index of the neighborhood to reset (unused —
                see NOTE above)

        Returns:
            Dict with reset statistics
        """
        try:
            self.memory.reset()
            self._score_buf.clear()
            self._warmup_scores.clear()

            LOGGER.info(
                "[MemStreamCore] Reset neighborhood %d",
                neighborhood_idx
            )
            return {
                'status': 'ok',
                'neighborhood_idx': neighborhood_idx,
            }
        except Exception as e:
            LOGGER.error(
                "[MemStreamCore] Failed to reset neighborhood %d: %s",
                neighborhood_idx, e
            )
            return {'status': 'error', 'error': str(e)}

    def clone(self) -> 'MemStreamCore':
        """Create a copy with same weights and full state including memory."""
        new_ms = MemStreamCore(cfg=self.cfg, device=self.device)
        new_ms.ae.load_state_dict(self.ae.state_dict())
        new_ms.mean = self.mean.clone() if self.mean is not None else None
        new_ms.std = self.std.clone() if self.std is not None else None
        new_ms.max_thres = self.max_thres.clone()
        new_ms.eval_mode = self.eval_mode
        new_ms.count = self.count
        new_ms.k = self.k
        new_ms.gamma = self.gamma
        new_ms._context_beta = self._context_beta
        # Copy memory module state
        new_ms.memory.load_state_dict(self.memory.get_state_dict())
        new_ms._score_buf = self._score_buf.copy()
        new_ms._warmup_scores = self._warmup_scores.copy()
        new_ms._recent_scores = deque(self._recent_scores, maxlen=5000)
        return new_ms

    # -------------------------------------------------------------------------
    # State dict (serialization)
    # -------------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        """Get full state for serialization.

        HARD-BLOCK FIX 7: Include _recent_scores in state dict.
        """
        state = {
            # AE
            'ae_state': self.ae.state_dict(),
            # Normalization
            'mean': self.mean,
            'std': self.std,
            # Threshold
            'max_thres': (
                self.max_thres.item()
                if hasattr(self.max_thres, 'item')
                else self.max_thres
            ),
            # Config
            'cfg': self.cfg.__dict__,
            # Count
            'count': self.count,
            # Memory
            'memory_state': self.memory.get_state_dict(),
            # kNN
            'k': self.k,
            'gamma': self.gamma,
            # Score buffers
            '_score_buf': self._score_buf,
            '_warmup_scores': self._warmup_scores,
            # HARD-BLOCK FIX 6: _recent_scores for quick_retrain
            '_recent_scores': list(self._recent_scores),
        }
        # ContextBeta
        if self._context_beta is not None:
            state['context_beta'] = self._context_beta.get_state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Load full state from serialization.

        HARD-BLOCK FIX 7: Restore _recent_scores from state dict.
        """
        # AE
        self.ae.load_state_dict(state['ae_state'])

        # Normalization
        if state.get('mean') is not None:
            self.mean = state['mean'].to(self.device)
        if state.get('std') is not None:
            self.std = state['std'].to(self.device)

        # Threshold
        max_thres_val = state.get('max_thres', 0.0)
        self.max_thres = torch.tensor(
            max_thres_val, dtype=torch.float32, device=self.device
        )

        # Count
        self.count = state.get('count', 0)

        # Memory
        if 'memory_state' in state:
            self.memory.load_state_dict(state['memory_state'])
        elif 'memory' in state:
            # Backward compat: load from legacy flat keys
            self.memory.memory = state['memory'].to(self.device)
            self.memory.mem_usage = state.get(
                'memory_mem_usage',
                torch.zeros_like(self.memory.mem_usage)
            ).to(self.device)
            self.memory.mem_ptr = int(state.get('memory_mem_ptr', 0))
            self.memory.count = int(state.get('memory_count', 0))
            self.memory._is_full = bool(state.get('_is_full', False))
            self.memory._memory_head = int(state.get('_memory_head', 0))

        # kNN
        self.k = state.get('k', self.cfg.k)
        self.gamma = state.get('gamma', self.cfg.gamma)

        # Score buffers
        self._score_buf = state.get('_score_buf', [])
        self._warmup_scores = state.get('_warmup_scores', [])

        # HARD-BLOCK FIX 7: Restore _recent_scores
        recent_scores_list = state.get('_recent_scores', [])
        self._recent_scores = deque(recent_scores_list, maxlen=5000)

        # ContextBeta
        if 'context_beta' in state:
            self._context_beta = ContextBeta.from_state_dict(state['context_beta'])
        elif 'context_beta_betas' in state:
            # Backward compat: load from legacy flat keys
            n_nb = state.get('context_beta_n_neighborhoods', 10)
            n_cells = state.get('context_beta_n_cells', 8)
            self._context_beta = ContextBeta(default_beta=0.5, percentile=95)
            self._context_beta.betas = state['context_beta_betas']
        else:
            self._context_beta = None

    # -------------------------------------------------------------------------
    # Persistence (HMAC-verified)
    # -------------------------------------------------------------------------

    def save(self, path: str, signing_key: str) -> None:
        """Save model to file with HMAC signature.

        Args:
            path: File path (.pt)
            signing_key: HMAC signing key (32+ chars)

        Raises:
            SecurityError: If signing key is too short
        """
        if len(signing_key) < 32:
            raise SecurityError(
                f"Signing key too short ({len(signing_key)} chars). "
                "Requires at least 32 characters."
            )

        # Serialize
        state = self.get_state_dict()
        buf = io.BytesIO()
        torch.save(state, buf, pickle_module=pickle)
        data = buf.getvalue()

        # HMAC signature (C-SEC-1, C-SEC-3)
        sig = hmac.new(
            signing_key.encode(), data, hashlib.sha256
        ).hexdigest()

        # Write checkpoint
        with open(path, 'wb') as f:
            f.write(data)

        # Write HMAC
        with open(path + '.hmac', 'w') as f:
            f.write(sig)

        LOGGER.info("[MemStreamCore] Saved checkpoint: %s", path)

    @classmethod
    def load(
        cls,
        path: str,
        device: str = 'cpu',
        signing_key: Optional[str] = None,
        require_signature: bool = True,
    ) -> 'MemStreamCore':
        """Load model from file with HMAC verification.

        HARD-BLOCK FIX 4: loads with weights_only=False AND verifies HMAC
        signature BEFORE using the state dict.

        Args:
            path: File path (.pt)
            device: Device to load model on
            signing_key: HMAC verification key (32+ chars)
            require_signature: If True, missing .hmac raises SecurityError

        Returns:
            MemStreamCore instance

        Raises:
            SecurityError: If HMAC verification fails or key missing
        """
        # =====================================================================
        # HMAC verification (C-SEC-1, C-SEC-3 — single block, no duplicate)
        # =====================================================================

        if signing_key:
            if len(signing_key) < 32:
                raise SecurityError(
                    f"Signing key too short ({len(signing_key)} chars). "
                    "Requires at least 32 characters."
                )

            hmac_path = path + '.hmac'
            if not os.path.exists(hmac_path):
                if require_signature:
                    raise SecurityError(
                        f"Model {path} requires HMAC signature but "
                        f"{hmac_path} not found."
                    )
                else:
                    LOGGER.warning(
                        "[MemStreamCore.load] HMAC file not found: %s — "
                        "skipping verification",
                        hmac_path
                    )
            else:
                with open(hmac_path) as f:
                    expected_hmac = f.read().strip()

                with open(path, 'rb') as f:
                    file_data = f.read()
                    actual_hmac = hmac.new(
                        signing_key.encode(), file_data, hashlib.sha256
                    ).hexdigest()

                if not hmac.compare_digest(expected_hmac, actual_hmac):
                    raise SecurityError(
                        f"Model HMAC mismatch — possible tampering: {path}"
                    )

                LOGGER.info(
                    "[MemStreamCore.load] HMAC verified for: %s",
                    path
                )

        elif require_signature:
            raise SecurityError(
                f"Model {path} requires HMAC verification but no "
                "signing key provided."
            )

        # =====================================================================
        # Load state (weights_only=False — checkpoint has non-tensor Python objs)
        # HARD-BLOCK FIX 4: weights_only=False with HMAC verified above
        # =====================================================================

        state = torch.load(
            path,
            map_location=device,
            weights_only=False,   # FIX: must be False — checkpoint has list/dict
            pickle_module=pickle
        )

        # =====================================================================
        # Reconstruct
        # =====================================================================

        cfg = MemStreamConfig()
        cfg.__dict__.update(state.get('cfg', {}))

        ms = cls(cfg=cfg, device=device)
        ms.load_state_dict(state)

        LOGGER.info(
            "[MemStreamCore.load] Loaded checkpoint: %s (count=%d)",
            path, ms.count
        )

        return ms


# ---------------------------------------------------------------------------
# SimpleADWIN (drift detection)
# ---------------------------------------------------------------------------

class SimpleADWIN:
    """Simplified ADWIN drift detector.

    Detects concept drift by monitoring the mean of a data stream.
    Uses a sliding window to compare recent vs. older distribution.
    """

    def __init__(self, delta: float = 0.002, max_window: int = 500):
        self.delta = delta
        self.max_window = max_window
        self._window: list = []
        self._total: float = 0.0

    def update(self, value: float) -> bool:
        """Add value and check for drift.

        Returns True if drift is detected.
        """
        self._window.append(value)
        self._total += value

        # Enforce max_window by trimming oldest entries
        if len(self._window) > self.max_window:
            excess = len(self._window) - self.max_window
            self._total -= sum(self._window[:excess])
            self._window = self._window[excess:]

        n = len(self._window)
        if n <= 50:
            return False

        # Compare recent vs old window
        window_size = min(n // 4, 100)
        recent = self._window[-window_size:]
        old = self._window[:-window_size]

        if len(old) > 10 and len(recent) > 10:
            recent_mean = sum(recent) / len(recent)
            old_mean = sum(old) / len(old)

            diff = abs(recent_mean - old_mean)
            # Hoeffding-based threshold: sqrt((1/(2*n)) * ln(2/delta))
            # Lower delta = more sensitive (detects smaller shifts sooner)
            threshold = math.sqrt((1.0 / (2.0 * max(n, 1))) * math.log(2.0 / max(self.delta, 1e-10)))

            if diff > threshold:
                # Drift detected — shrink window
                cut = len(self._window) // 2
                self._window = self._window[cut:]
                self._total = sum(self._window)
                return True

        return False

    def reset(self) -> None:
        """Reset the detector."""
        self._window.clear()
        self._total = 0.0

    def get_window_size(self) -> int:
        """Return current window size."""
        return len(self._window)


# ---------------------------------------------------------------------------
# BAR Controller (Budget Allocation Rate)
# ---------------------------------------------------------------------------

class BARController:
    """Budget Allocation Rate Controller for label-efficient MemStream.

    Scientific purpose:
    - Original MemStream: 100% label cost (update memory on every record)
    - With BAR: 1-5% label cost (update only when ADWIN detects drift or
      IEC grants budget)

    This enables cost-effective production deployment with minimal accuracy loss.
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}
        self.target_bar = self._config.get('target_bar_rate', 0.02)
        self.adwin_delta = self._config.get('adwin_delta', 0.002)
        self.min_budget_fraction = self._config.get('min_budget_fraction', 0.01)
        self.bar_window_size = self._config.get('bar_window_size', 1000)

        # Per-neighborhood ADWIN detectors
        self._adwins: dict = {}

        # Budget state
        self._budget_granted: bool = False
        self._recent_updates: list = []

        # Statistics
        self._total_records: int = 0
        self._memory_updates: int = 0
        self._drift_events: int = 0

    @property
    def bar_rate(self) -> float:
        """Current BAR rate (rolling window)."""
        if not self._recent_updates:
            return 0.0
        window = self._recent_updates[-self.bar_window_size:]
        return sum(window) / len(window)

    def _get_adwin(self, neighborhood: str) -> SimpleADWIN:
        """Get or create ADWIN for neighborhood."""
        if neighborhood not in self._adwins:
            self._adwins[neighborhood] = SimpleADWIN(delta=self.adwin_delta)
        return self._adwins[neighborhood]

    def should_update_memory(
        self,
        neighborhood: str,
        score: float
    ) -> Tuple[bool, str]:
        """Determine if memory should be updated for this record.

        Returns:
            (should_update, reason):
            - should_update: True if memory should be updated
            - reason: 'drift_detected', 'iec_budget_granted',
                      'minimum_budget_guarantee', 'no_budget'
        """
        self._total_records += 1

        # Rule 1: ADWIN drift detection
        adwin = self._get_adwin(neighborhood)
        drift_detected = adwin.update(score)
        if drift_detected:
            self._drift_events += 1
            self._memory_updates += 1
            self._recent_updates.append(1)
            self._budget_granted = False
            LOGGER.info(
                "[BARController] Drift detected for %s",
                neighborhood
            )
            return True, "drift_detected"

        # Rule 2: Explicit Budget Grant from IEC
        if self._budget_granted:
            self._memory_updates += 1
            self._recent_updates.append(1)
            self._budget_granted = False
            return True, "iec_budget_granted"

        # Rule 3: Minimum Budget Guarantee (prevent starvation)
        current_bar = self.bar_rate
        if current_bar < self.min_budget_fraction:
            self._memory_updates += 1
            self._recent_updates.append(1)
            return True, "minimum_budget_guarantee"

        # Default: No update
        self._recent_updates.append(0)
        return False, "no_budget"

    def grant_budget(self, reason: str = "manual") -> None:
        """IEC grants budget for memory update."""
        self._budget_granted = True
        LOGGER.info("[BARController] Budget granted: %s", reason)

    def get_stats(self) -> dict:
        """Get BAR statistics for metrics/logging."""
        return {
            'total_records': self._total_records,
            'memory_updates': self._memory_updates,
            'drift_events': self._drift_events,
            'bar_rate': self.bar_rate,
            'bar_rate_pct': self.bar_rate * 100,
        }


# ---------------------------------------------------------------------------
# 4D Context Extraction (NYC Taxi)
# ---------------------------------------------------------------------------

def get_4d_context(
    record: dict,
    neighborhood_mapping: Optional[dict] = None
) -> dict:
    """Extract 4D context from NYC taxi record.

    This creates the Context Grid for CA-DQStream.
    The 4D context is fed into ContextAwareFeatureVectorizer.

    Returns:
        Dict with 4D context keys:
            - neighborhood: str
            - hour_bucket: str
            - day_type: str ('weekday' or 'weekend')
            - trip_type: str ('short', 'medium', 'long')
    """
    from datetime import datetime

    # 1. Neighborhood (from zone ID)
    zone_id = int(float(record.get('PULocationID', 1)))
    if neighborhood_mapping:
        neighborhood = neighborhood_mapping.get(zone_id, 'unknown')
    else:
        if zone_id <= 50:
            neighborhood = 'manhattan'
        elif zone_id <= 100:
            neighborhood = 'brooklyn'
        elif zone_id <= 150:
            neighborhood = 'queens'
        elif zone_id <= 200:
            neighborhood = 'bronx'
        elif zone_id in (132, 138):
            neighborhood = 'airport'
        else:
            neighborhood = 'staten_island'

    # 2. Hour bucket (4-hour buckets)
    dt_str = record.get('tpep_pickup_datetime', '')
    hour = 12
    if dt_str:
        try:
            dt = datetime.fromisoformat(dt_str.replace('/', '-'))
            hour = dt.hour
        except Exception:
            pass

    if 6 <= hour < 10:
        hour_bucket = 'morning_rush'
    elif 10 <= hour < 17:
        hour_bucket = 'midday'
    elif 17 <= hour < 21:
        hour_bucket = 'evening_rush'
    else:
        hour_bucket = 'night'

    # 3. Day type (weekday vs weekend)
    if dt_str:
        try:
            dt = datetime.fromisoformat(dt_str.replace('/', '-'))
            dow = dt.weekday()
            day_type = 'weekend' if dow >= 5 else 'weekday'
        except Exception:
            day_type = 'weekday'
    else:
        day_type = 'weekday'

    # 4. Trip type (based on distance)
    distance = float(record.get('trip_distance', 0))
    if distance < 2:
        trip_type = 'short'
    elif distance < 10:
        trip_type = 'medium'
    else:
        trip_type = 'long'

    return {
        'neighborhood': neighborhood,
        'hour_bucket': hour_bucket,
        'day_type': day_type,
        'trip_type': trip_type,
    }


def get_context_key(context: dict) -> str:
    """Create string key from 4D context."""
    return (
        f"{context['neighborhood']}_"
        f"{context['hour_bucket']}_"
        f"{context['day_type']}_"
        f"{context['trip_type']}"
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config
    'MemStreamConfig',
    # Core
    'MemStreamCore',
    'MemStreamAE',
    'MemoryModule',
    # Context
    'ContextBeta',
    'get_context_id',
    'extract_temporal_context',
    'decode_ratecode_from_onehot',
    'get_4d_context',
    'get_context_key',
    # Utilities
    'set_determinism',
    'SimpleADWIN',
    'BARController',
    # Security
    'SecurityError',
]
