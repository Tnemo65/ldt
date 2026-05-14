"""
MemStream Core: Online Autoencoder + Memory Module.

This module contains all MemStream logic with v10 benchmark integration:
- C-ML-1: max_thres initialized in __init__
- H-ML-2: Determinism flags in warmup()
- H-ML-3: CUDA seeds in set_determinism()
- C-SEC-1: HMAC key enforcement
- C-SEC-3: Single HMAC block (no duplicate)
- ContextBeta: 80 context-aware thresholds (10 neighborhoods x 8 cells)
- L1 kNN distance scoring (no reconstruction error)
- Conditional memory update (original paper semantics)
"""

import copy
import hashlib
import hmac
import io
import logging
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

LOGGER = logging.getLogger('memstream')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def set_determinism(seed: int = 42):
    """Configure all random sources for reproducible training/scoring.

    Call this at the start of training scripts and in the Flink operator
    open() method. Does not guarantee bit-exact reproducibility across
    PyTorch versions or hardware, but eliminates most sources of variance.

    Note: PYTHONHASHSEED must be set in the environment BEFORE Python starts.
    Set in docker-compose.yml: environment: PYTHONHASHSEED: "42"
    """
    # Python built-ins
    import random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # H-ML-3: CUDA seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # All GPUs in multi-GPU training

    # CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Deterministic algorithms (PyTorch 1.8+)
    torch.use_deterministic_algorithms(True, warn_only=True)

    os.environ['PYTHONHASHSEED'] = str(seed)


class MemStreamConfig:
    """Hyperparameters for MemStream (34D input from v10 benchmark)."""

    def __init__(self):
        # Architecture
        self.in_dim: int = 34  # v10: 34D features
        self.hidden_dim: int = 68  # v10: 2x hidden layer
        self.out_dim: int = 34  # v10: symmetric
        self.latent_dim: int = 60  # Alias for hidden_dim (benchmark compatibility)

        # Memory
        self.memory_len: int = 256  # v10: 256 (not 100)
        self.memory_init_fraction: float = 0.1

        # kNN scoring (v10 benchmark)
        self.k: int = 10  # kNN neighbors
        self.gamma: float = 0.0  # Decay factor for weighted kNN

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


class MemStreamAE(nn.Module):
    """Autoencoder for MemStream anomaly detection.

    Architecture: 34 -> 68 -> 34 (symmetric, v10 benchmark)
    - Encoder: Linear(34, 68) -> ReLU -> Linear(68, 34) -> ReLU
    - Decoder: Linear(34, 68) -> ReLU -> Linear(68, 34) -> ReLU
    
    NOTE: ReLU (not Tanh) per benchmark_v10.py line 612:
        z = torch.nn.functional.relu(x_noisy @ W1 + b1)
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
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


class MemoryModule:
    """Memory module for storing representative normal patterns.

    FIFO queue of encoded normal samples. Uses gradient detachment
    to prevent memory updates from affecting the autoencoder.
    
    Distance metric: L1 (Manhattan) per benchmark_v10.py line 662.
    """

    def __init__(self, memory_len: int = 256, out_dim: int = 34, device: str = 'cpu'):
        self.memory_len = memory_len
        self.out_dim = out_dim
        self.device = device

        # Memory slots: [memory_len, out_dim]
        self.memory = torch.zeros(memory_len, out_dim, device=device)

        # Usage tracking: [memory_len]
        self.mem_usage = torch.zeros(memory_len, device=device)

        # Circular pointer
        self.mem_ptr = 0
        self.count = 0

        # Additional state for state_dict
        self._is_full = False
        self._memory_head = 0

    def update(self, z: torch.Tensor):
        """Add new encoded sample to memory.

        Uses FIFO replacement (circular buffer).
        Gradient is detached to prevent backprop through memory.
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

    def reset(self):
        """Reset memory to zeros."""
        self.memory.zero_()
        self.mem_usage.zero_()
        self.mem_ptr = 0
        self.count = 0
        self._is_full = False
        self._memory_head = 0


class SecurityError(Exception):
    """Raised when HMAC verification fails."""
    pass


class ContextBeta:
    """80 context-beta thresholds: 10 neighborhoods x 8 cells.

    Each context cell has its own beta threshold fitted from warmup scores.
    During scoring, score/beta gives normalized ratio:
    - ratio < 1.0 → normal
    - ratio >= 1.0 → anomaly
    """

    def __init__(self, n_neighborhoods=10, n_cells=8, percentile=95):
        self.n_neighborhoods = n_neighborhoods
        self.n_cells = n_cells
        self.percentile = percentile
        self.betas = np.ones((n_neighborhoods, n_cells), dtype=np.float32) * 0.5

    def fit_from_scores(self, scores, neighborhood_ids, context_ids):
        """Fit beta thresholds from warmup scores (train set only)."""
        for n in range(self.n_neighborhoods):
            for c in range(self.n_cells):
                cell_scores = [s for s, nm, ctx in
                              zip(scores, neighborhood_ids, context_ids)
                              if nm == n and ctx == c]
                if len(cell_scores) >= 50:
                    self.betas[n, c] = float(np.percentile(cell_scores, self.percentile))

    def get_beta(self, neighborhood_id, context_id):
        n = min(int(neighborhood_id), self.n_neighborhoods - 1)
        c = min(int(context_id), self.n_cells - 1)
        return float(self.betas[n, c])


def get_context_id(hour: int, dow: int, ratecode: float) -> int:
    """8 context cells: (Standard/Special) x (Day/Night) x (Weekday/Weekend).

    Args:
        hour: Hour of day (0-23)
        dow: Day of week (0=Mon, 6=Sun)
        ratecode: Rate code (1=Standard, >1=Special)

    Returns:
        Context ID (0-7)
    """
    is_special = 1 if ratecode > 1 else 0
    is_night = 1 if (hour >= 18 or hour < 6) else 0
    is_weekend = 1 if dow >= 5 else 0
    return (is_special << 2) | (is_night << 1) | is_weekend


class MemStreamCore:
    """MemStream: Online Autoencoder + Memory Module with ContextBeta.

    Scoring (v10 benchmark):
    1. Encode input x -> z
    2. Compute L1 kNN distance: sum(|z - memory[i]|) for k nearest
    3. Normalize by context-beta threshold
    4. Final score = raw_score / beta (ratio method)
    5. score >= 1.0 -> ANOMALY

    Memory Update (streaming, conditional):
    1. Only update if score < beta (normal point)
    2. Anomalies do NOT update memory
    3. This is the original MemStream paper semantics
    """

    def __init__(
        self,
        cfg: MemStreamConfig = None,
        device: str = 'cpu'
    ):
        self.cfg = cfg or MemStreamConfig()
        self.device = device

        # Model
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
        self.eval_mode = True
        # C-ML-1 FIX: Initialize max_thres in __init__ to avoid AttributeError
        self.max_thres: torch.Tensor = torch.tensor(
            0.0, dtype=torch.float32, device=device
        )

        # Count of samples processed
        self.count = 0

        # v10 benchmark attributes
        self.latent_dim = self.cfg.hidden_dim  # For compatibility
        self.k = getattr(self.cfg, 'k', 10)  # kNN neighbors
        self.gamma = getattr(self.cfg, 'gamma', 0.0)  # Decay factor

        # ContextBeta for context-aware scoring
        self._context_beta: Optional[ContextBeta] = None

        # Score buffer for warmup fitting
        self._score_buf: list = []
        self._warmup_scores: list = []

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize using frozen stats."""
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / (self.std + 1e-8)

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize."""
        if self.mean is None or self.std is None:
            return x
        return x * (self.std + 1e-8) + self.mean

    def _encode(self, x_scaled: np.ndarray) -> np.ndarray:
        """Encode input using the autoencoder encoder."""
        x_t = torch.from_numpy(x_scaled.astype(np.float32)).to(self.device)
        with torch.no_grad():
            x_t_2d = x_t.unsqueeze(0) if x_t.dim() == 1 else x_t
            z = self.ae.encoder(x_t_2d)
            return z[0].cpu().numpy()

    def _score_one_raw(self, x_scaled: np.ndarray) -> float:
        """Score using L1 kNN distance (benchmark v10 logic).

        This is the ONLY scoring method - no reconstruction error.
        Matches benchmark_v10.py line 657-667 exactly.
        """
        if len(self.memory.memory) < 2:
            return 0.5

        z = self._encode(x_scaled)
        mem_arr = self.memory.get_memory().cpu().numpy()
        dists = np.sum(np.abs(mem_arr - z), axis=1)
        k_use = min(self.k, len(dists))
        top_idx = np.argpartition(dists, k_use)[:k_use]
        top_d = np.sort(dists[top_idx])
        score = sum((self.gamma ** i) * top_d[i] for i in range(k_use))
        return float(score)

    def warmup(
        self,
        X_normal: np.ndarray,
        epochs: int = None,
        batch_size: int = None,
        verbose: bool = True
    ):
        """Warmup phase: train AE + initialize memory + fit ContextBeta.

        Args:
            X_normal: Normal training data [N, 34], float32
            epochs: Training epochs (default from config)
            batch_size: Batch size (default from config)
        """
        set_determinism(self.cfg.seed)  # H-ML-2 FIX

        epochs = epochs or self.cfg.warmup_epochs
        batch_size = batch_size or self.cfg.warmup_batch_size

        X = torch.from_numpy(X_normal).float().to(self.device)
        n = len(X)

        # Compute normalization stats from first 10%
        n_stats = max(1, int(n * 0.1))
        stats_data = X[:n_stats]

        self.mean = stats_data.mean(dim=0)
        self.std = stats_data.std(dim=0)
        # Clamp to min=1.0 to handle near-zero-variance features
        self.std = torch.clamp(self.std, min=1.0)

        # Normalize training data (middle 80%)
        train_data = X[n_stats:int(n * 0.9)]
        X_norm = self._normalize(train_data)

        # Training loop
        self.ae.train()
        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Shuffle
            indices = torch.randperm(len(X_norm))
            X_shuffled = X_norm[indices]

            # Mini-batch training
            total_loss = 0.0
            for i in range(0, len(X_shuffled), batch_size):
                batch = X_shuffled[i:i+batch_size]

                # Add noise for robustness
                noise = torch.randn_like(batch) * self.cfg.warmup_noise_std
                x_noisy = batch + noise

                # Forward
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

            avg_loss = total_loss / (len(X_shuffled) / batch_size)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.ae.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.warmup_early_stop_patience:
                    if verbose:
                        print(f"  Early stop at epoch {epoch+1}")
                    break

            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}: loss = {avg_loss:.6f}")

        # Load best model
        if best_state is not None:
            self.ae.load_state_dict(best_state)

        self.ae.eval()

        # Initialize memory with last 10% (detached)
        self.eval_mode = True
        memory_data = X[int(n * 0.9):]
        with torch.no_grad():
            memory_encoded = self.ae.encoder(memory_data)
            # Select diverse samples using random sampling
            n_memory = min(self.cfg.memory_len, len(memory_encoded))
            indices = torch.randperm(len(memory_encoded))[:n_memory]
            self.memory.memory = memory_encoded[indices].clone()
            self.memory.mem_usage.fill_(1.0)
            self.memory.count = n_memory

        # =====================================================================
        # Fit ContextBeta from warmup scores (v10 benchmark logic)
        # =====================================================================
        warmup_n = min(5000, len(X_norm))
        warmup_scores = []
        for i in range(warmup_n):
            x = X_norm[i].cpu().numpy()
            warmup_scores.append(self._score_one_raw(x))
        self._warmup_scores = warmup_scores

        # Extract context info from raw data if available
        # Feature indices: hour at [:,9], dow at [:,10], ratecode at [:,25]
        X_normal_np = X_normal if isinstance(X_normal, np.ndarray) else X.cpu().numpy()
        if X_normal_np.shape[1] > 9:
            hour_vals = X_normal_np[:, 9].astype(int)[:warmup_n]
        else:
            hour_vals = np.zeros(warmup_n, dtype=int)

        if X_normal_np.shape[1] > 10:
            dow_vals = X_normal_np[:, 10].astype(int)[:warmup_n]
        else:
            dow_vals = np.zeros(warmup_n, dtype=int)

        if X_normal_np.shape[1] > 25:
            ratecode_vals = X_normal_np[:, 25].astype(int)[:warmup_n]
        else:
            ratecode_vals = np.ones(warmup_n, dtype=int)

        # Compute neighborhood IDs (0-9, default all 0 for batch scoring)
        neighborhood_ids = np.zeros(warmup_n, dtype=int)
        ctx_ids = np.array([get_context_id(int(h), int(d), float(r))
                           for h, d, r in zip(hour_vals, dow_vals, ratecode_vals)])

        # Fit ContextBeta
        self._context_beta = ContextBeta(n_neighborhoods=10, n_cells=8, percentile=95)
        self._context_beta.fit_from_scores(warmup_scores, neighborhood_ids, ctx_ids)
        self._score_buf = warmup_scores.copy()

        # Set max_thres from overall beta (for backward compatibility)
        overall_beta = float(np.percentile(warmup_scores, 95))
        self.max_thres = torch.tensor(overall_beta, dtype=torch.float32, device=self.device)

        if verbose:
            print(f"  Warmup complete: {epoch+1} epochs, best_loss = {best_loss:.6f}")
            print(f"  Memory initialized with {n_memory} samples")
            print(f"  ContextBeta fitted: overall beta = {overall_beta:.4f}")

    def memory_update(self, x: np.ndarray, neighborhood_id: int = 0, hour: int = 12,
                      dow: int = 0, ratecode: float = 1.0):
        """Streaming memory update. Only updates if score < beta (normal point).

        Matches original MemStream paper semantics:
        - Only normal points (score <= beta) update memory
        - Anomalies are detected precisely because they don't update memory
        """
        if self.eval_mode:
            self.eval_mode = False
            self.ae.eval()

        if self.mean is None:
            return

        score = self.score_one(x, neighborhood_id, hour, dow, ratecode)

        # Conditional update: only if score < beta (normal)
        if self._context_beta is not None:
            ctx_id = get_context_id(int(hour), int(dow), float(ratecode))
            beta = self._context_beta.get_beta(int(neighborhood_id), ctx_id)
            if score >= beta:
                self.count += 1
                return  # Anomaly - do NOT update memory
        else:
            if score >= self.max_thres.item():
                self.count += 1
                return  # Anomaly - do NOT update memory

        # Normal point - encode and add to memory
        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)

        with torch.no_grad():
            z = self.ae.encoder(x_t)
            self.memory.update(z[0])

        self.count += 1

    def score_one(self, x: np.ndarray, neighborhood_id: int = 0, hour: int = 12,
                  dow: int = 0, ratecode: float = 1.0) -> float:
        """Score one record. Uses context-beta ratio method.

        Returns: score / beta (normalized ratio)
        - ratio < 1.0 → normal
        - ratio >= 1.0 → anomaly

        Matches benchmark_v10.py line 669-680.
        """
        if self.mean is None:
            return 0.5

        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)

        # Normalize using frozen stats
        x_norm = self._normalize(x_t)
        x_np = x_norm.cpu().numpy()[0]

        # Raw score = L1 kNN distance
        raw_score = self._score_one_raw(x_np)

        # Apply context-beta ratio if available
        if self._context_beta is not None:
            ctx_id = get_context_id(int(hour), int(dow), float(ratecode))
            beta = self._context_beta.get_beta(int(neighborhood_id), ctx_id)
            return raw_score / max(beta, 1e-6)

        return raw_score

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Score batch using L1 kNN distance (v10 benchmark logic).

        Args:
            X: Batch of records, shape [N, in_dim]

        Returns:
            Array of anomaly scores [N], higher = more anomalous
        """
        if self.mean is None:
            return np.full(len(X), 0.5)

        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        X_norm = self._normalize(X_t)
        scores = []
        for i in range(len(X_norm)):
            x_np = X_norm[i].cpu().numpy()
            scores.append(self._score_one_raw(x_np))
        return np.array(scores, dtype=np.float64)

    def set_beta(self, beta: float):
        """Set anomaly threshold (global fallback)."""
        self.max_thres = torch.tensor(
            beta, dtype=torch.float32, device=self.device
        )

    def set_beta_for_neighborhood(self, neighborhood_id: int, beta: float):
        """
        Set anomaly threshold for a specific neighborhood.

        Updates the ContextBeta array so that scoring uses the new threshold
        for this neighborhood across all context cells.

        Args:
            neighborhood_id: Index of the neighborhood (0-9)
            beta: New threshold value
        """
        if self._context_beta is not None:
            n_id = int(neighborhood_id)
            for c in range(self._context_beta.n_cells):
                self._context_beta.betas[n_id, c] = beta
        self.max_thres = torch.tensor(beta, dtype=torch.float32, device=self.device)

    def clone(self) -> 'MemStreamCore':
        """Create a copy with same weights but fresh memory."""
        new_ms = MemStreamCore(cfg=self.cfg, device=self.device)
        new_ms.ae.load_state_dict(self.ae.state_dict())
        new_ms.mean = self.mean.clone() if self.mean is not None else None
        new_ms.std = self.std.clone() if self.std is not None else None
        new_ms.max_thres = self.max_thres.clone()
        new_ms.eval_mode = self.eval_mode
        new_ms.count = self.count
        new_ms.latent_dim = self.latent_dim
        new_ms.k = self.k
        new_ms.gamma = self.gamma
        new_ms._context_beta = self._context_beta
        new_ms._score_buf = self._score_buf.copy()
        new_ms._warmup_scores = self._warmup_scores.copy()
        return new_ms

    def reset_neighborhood(self, neighborhood_idx: int) -> Dict:
        """
        Reset MemStream memory for a specific neighborhood.

        Args:
            neighborhood_idx: Index of the neighborhood to reset

        Returns:
            Dict with reset statistics
        """
        try:
            if hasattr(self, 'memory'):
                self.memory.reset()

            if hasattr(self, '_score_buf'):
                self._score_buf.clear()
            if hasattr(self, '_warmup_scores'):
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

    def get_state_dict(self) -> dict:
        """Get full state for serialization."""
        state = {
            'ae_state': self.ae.state_dict(),
            'mean': self.mean,
            'std': self.std,
            'max_thres': self.max_thres.item() if hasattr(self.max_thres, 'item') else self.max_thres,
            'cfg': self.cfg.__dict__,
            'count': self.count,
            'memory': self.memory.get_memory(),
            'memory_mem_usage': self.memory.mem_usage,
            'memory_mem_ptr': self.memory.mem_ptr,
            'memory_count': self.memory.count,
            '_is_full': getattr(self.memory, '_is_full', False),
            '_memory_head': getattr(self.memory, '_memory_head', 0),
            'latent_dim': self.latent_dim,
            'k': self.k,
            'gamma': self.gamma,
            '_score_buf': self._score_buf,
            '_warmup_scores': self._warmup_scores,
        }
        if self._context_beta is not None:
            state['context_beta_betas'] = self._context_beta.betas
            state['context_beta_n_neighborhoods'] = self._context_beta.n_neighborhoods
            state['context_beta_n_cells'] = self._context_beta.n_cells
        return state

    def load_state_dict(self, state: dict):
        """Load full state from serialization."""
        self.ae.load_state_dict(state['ae_state'])
        if state.get('mean') is not None:
            self.mean = state['mean'].to(self.device)
        if state.get('std') is not None:
            self.std = state['std'].to(self.device)
        max_thres_val = state.get('max_thres', 0.0)
        self.max_thres = torch.tensor(max_thres_val, dtype=torch.float32, device=self.device)
        self.count = state.get('count', 0)

        # Restore memory state
        if 'memory' in state:
            self.memory.memory = state['memory'].to(self.device)
        if 'memory_mem_usage' in state:
            self.memory.mem_usage = state['memory_mem_usage'].to(self.device)
        if 'memory_mem_ptr' in state:
            self.memory.mem_ptr = state['memory_mem_ptr']
        if 'memory_count' in state:
            self.memory.count = state['memory_count']
        setattr(self.memory, '_is_full', state.get('_is_full', False))
        setattr(self.memory, '_memory_head', state.get('_memory_head', 0))

        # Restore v10 attributes
        self.latent_dim = state.get('latent_dim', self.cfg.hidden_dim)
        self.k = state.get('k', getattr(self.cfg, 'k', 10))
        self.gamma = state.get('gamma', getattr(self.cfg, 'gamma', 0.0))
        self._score_buf = state.get('_score_buf', [])
        self._warmup_scores = state.get('_warmup_scores', [])

        # Restore ContextBeta
        if 'context_beta_betas' in state:
            n_neighborhoods = state.get('context_beta_n_neighborhoods', 10)
            n_cells = state.get('context_beta_n_cells', 8)
            self._context_beta = ContextBeta(n_neighborhoods=n_neighborhoods, n_cells=n_cells)
            self._context_beta.betas = state['context_beta_betas']
        else:
            self._context_beta = None

    # =========================================================================
    # Persistence (HMAC-verified)
    # =========================================================================

    def save(self, path: str, signing_key: str):
        """Save model to file with HMAC signature.

        Args:
            path: File path (.pt)
            signing_key: HMAC signing key (32+ chars)
        """
        # Serialize using get_state_dict for complete state
        state = self.get_state_dict()
        buf = io.BytesIO()
        torch.save(state, buf, pickle_module=pickle)
        data = buf.getvalue()

        # HMAC signature
        sig = hmac.new(
            signing_key.encode(), data, hashlib.sha256
        ).hexdigest()

        # Write files
        with open(path, 'wb') as f:
            f.write(data)

        with open(path + '.hmac', 'w') as f:
            f.write(sig)

    @classmethod
    def load(
        cls,
        path: str,
        device: str = 'cpu',
        signing_key: Optional[str] = None,
        require_signature: bool = True,
    ) -> 'MemStreamCore':
        """Load model from file with HMAC verification.

        Args:
            path: File path (.pt)
            device: Device to load model on
            signing_key: HMAC verification key (32+ chars)
            require_signature: If True, missing .hmac raises SecurityError

        Returns:
            MemStreamCore instance

        Raises:
            SecurityError: If HMAC verification fails
        """
        # C-SEC-3 FIX: Single HMAC block, no duplicate
        if signing_key:
            hmac_path = path + '.hmac'
            if not os.path.exists(hmac_path):
                if require_signature:
                    raise SecurityError(
                        f"Model {path} requires HMAC signature but {hmac_path} not found."
                    )
                else:
                    LOGGER.warning(f"HMAC file not found: {hmac_path} - skipping verification")
            else:
                with open(hmac_path) as f:
                    expected_hmac = f.read().strip()

                with open(path, 'rb') as f:
                    actual_hmac = hmac.new(
                        signing_key.encode(), f.read(), hashlib.sha256
                    ).hexdigest()

                if not hmac.compare_digest(expected_hmac, actual_hmac):
                    raise SecurityError(
                        f"Model HMAC mismatch — possible tampering: {path}"
                    )
        elif require_signature:
            raise SecurityError(
                f"Model {path} requires HMAC verification but no signing key provided."
            )

        # Load state
        state = torch.load(
            path, map_location=device, weights_only=True, pickle_module=pickle
        )

        # Reconstruct
        cfg = MemStreamConfig()
        cfg.__dict__.update(state.get('cfg', {}))

        ms = cls(cfg=cfg, device=device)
        ms.load_state_dict(state)

        return ms


# =============================================================================
# BAR Controller (Budget Allocation Rate)
# =============================================================================

class SimpleADWIN:
    """Simplified ADWIN drift detector.

    Detects concept drift by monitoring the mean of a data stream.
    Uses an exponential histogram to maintain time-sorted windows.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._window: list = []
        self._total: float = 0.0
        self._variance: float = 0.0

    def update(self, value: float) -> bool:
        """Add value and check for drift.

        Returns True if drift is detected.
        """
        self._window.append(value)
        self._total += value

        # Compute mean
        n = len(self._window)
        mean = self._total / n if n > 0 else 0.0

        # Check for drift using naive approach
        # (simplified ADWIN - real implementation would use exponential histogram)
        if n > 50:
            # Compare recent vs old window
            window_size = min(n // 4, 100)
            recent = self._window[-window_size:]
            old = self._window[:-window_size]

            if len(old) > 10 and len(recent) > 10:
                recent_mean = sum(recent) / len(recent)
                old_mean = sum(old) / len(old)

                # Check if means differ significantly
                diff = abs(recent_mean - old_mean)
                threshold = 2.0 * (1.0 / n ** 0.5)  # Simplified confidence bound

                if diff > threshold:
                    # Drift detected - shrink window
                    cut = len(self._window) // 2
                    self._window = self._window[cut:]
                    self._total = sum(self._window)
                    return True

        return False

    def reset(self):
        """Reset the detector."""
        self._window.clear()
        self._total = 0.0
        self._variance = 0.0


class BARController:
    """
    Budget Allocation Rate Controller for label-efficient MemStream.

    Scientific purpose:
    - Original MemStream: 100% label cost (update memory on every record)
    - With BAR: 1-5% label cost (update only when ADWIN detects drift or IEC grants budget)

    This enables cost-effective production deployment with minimal accuracy loss.
    """

    def __init__(self, config=None):
        self._config = config or {}
        self.target_bar = self._config.get('target_bar_rate', 0.02)  # 2% default
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
        if len(self._recent_updates) == 0:
            return 0.0
        window = self._recent_updates[-self.bar_window_size:]
        return sum(window) / len(window) if window else 0.0

    def _get_adwin(self, neighborhood: str) -> SimpleADWIN:
        """Get or create ADWIN for neighborhood."""
        if neighborhood not in self._adwins:
            self._adwins[neighborhood] = SimpleADWIN(delta=self.adwin_delta)
        return self._adwins[neighborhood]

    def should_update_memory(
        self,
        neighborhood: str,
        score: float
    ) -> tuple[bool, str]:
        """
        Determine if memory should be updated for this record.

        Returns:
            (should_update, reason)
            - should_update: True if memory should be updated
            - reason: 'drift_detected', 'budget_granted', 'minimum_budget', 'no_budget'
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
            LOGGER.info(f"[BARController] Drift detected for {neighborhood}")
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

    def grant_budget(self, reason: str = "manual"):
        """IEC grants budget for memory update."""
        self._budget_granted = True
        LOGGER.info(f"[BARController] Budget granted: {reason}")

    def get_stats(self) -> dict:
        """Get BAR statistics for metrics/logging."""
        return {
            'total_records': self._total_records,
            'memory_updates': self._memory_updates,
            'drift_events': self._drift_events,
            'bar_rate': self.bar_rate,
            'bar_rate_pct': self.bar_rate * 100,
        }


# =============================================================================
# 4D Context Extraction
# =============================================================================

def get_4d_context(
    record: dict,
    neighborhood_mapping: dict = None
) -> dict:
    """
    Extract 4D context from NYC taxi record.

    This is the core function that creates the "Context Grid" for CA-DQStream.
    The 4D context is then fed into ContextAwareFeatureVectorizer.

    Returns:
        Dict with 4D context keys:
            - neighborhood: str (e.g., 'manhattan')
            - hour_bucket: str (e.g., 'evening_rush')
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
        elif zone_id in [132, 138]:
            neighborhood = 'airport'
        else:
            neighborhood = 'staten_island'

    # 2. Hour bucket (4-hour buckets)
    dt_str = record.get('tpep_pickup_datetime', '')
    hour = 12  # Default
    if dt_str:
        try:
            dt = datetime.fromisoformat(dt_str.replace('/', '-'))
            hour = dt.hour
        except:
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
        except:
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
    return f"{context['neighborhood']}_{context['hour_bucket']}_{context['day_type']}_{context['trip_type']}"
