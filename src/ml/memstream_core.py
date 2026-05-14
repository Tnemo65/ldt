"""
MemStream Core: Online Autoencoder + Memory Module for Flink.

This module provides the full MemStream implementation for the CA-DQStream pipeline:
- Denoising Autoencoder (30D -> 60D -> 30D)
- Memory Module (FIFO queue of encoded representations)
- ADWIN drift detection
- BAR Controller (Budget Allocation Rate)

Reference: Bhatia et al. (2022) - MemStream: Memory-Based Streaming Anomaly Detection
"""

import copy
import logging
import os
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

LOGGER = logging.getLogger('memstream-core')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def set_determinism(seed: int = 42):
    """Configure all random sources for reproducible training/scoring."""
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


# =============================================================================
# Configuration
# =============================================================================

class MemStreamConfig:
    """Hyperparameters for MemStream (30D input)."""

    def __init__(self):
        # Architecture
        self.in_dim: int = 30
        self.hidden_dim: int = 60
        self.out_dim: int = 30

        # Memory
        self.memory_len: int = 1000  # Increased for production
        self.memory_init_fraction: float = 0.1

        # Training (warmup)
        self.warmup_lr: float = 1e-3
        self.warmup_epochs: int = 500
        self.warmup_batch_size: int = 256
        self.warmup_noise_std: float = 0.1
        self.warmup_gradient_clip: float = 1.0
        self.warmup_early_stop_patience: int = 20

        # Scoring
        self.default_beta: float = 0.5
        self.k_neighbors: int = 10  # k for kNN in memory

        # Determinism
        self.seed: int = 42


# =============================================================================
# Autoencoder
# =============================================================================

class MemStreamAE(nn.Module):
    """Denoising Autoencoder for MemStream.

    Architecture: 30 -> 60 -> 30 (symmetric)
    - Encoder: Linear(30, 60) -> Tanh -> Linear(60, 30) -> Tanh
    - Decoder: Linear(30, 60) -> Tanh -> Linear(60, 30) -> Tanh
    """

    def __init__(self, in_dim: int = 30, hidden_dim: int = 60):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, in_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, in_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon


# =============================================================================
# Memory Module
# =============================================================================

class MemoryModule:
    """Memory module for storing representative normal patterns.

    FIFO queue of encoded normal samples. Uses gradient detachment
    to prevent memory updates from affecting the autoencoder.
    """

    def __init__(self, memory_len: int = 1000, out_dim: int = 25, device: str = 'cpu'):
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

    def update(self, z: torch.Tensor):
        """Add new encoded sample to memory (FIFO replacement)."""
        z_detached = z.detach().clone()

        if z_detached.dim() == 1:
            z_detached = z_detached.unsqueeze(0)

        for i in range(min(z_detached.shape[0], self.memory_len)):
            self.memory[self.mem_ptr] = z_detached[i]
            self.mem_usage[self.mem_ptr] = 1.0
            self.mem_ptr = (self.mem_ptr + 1) % self.memory_len
            self.count += 1

    def get_memory(self) -> torch.Tensor:
        """Return current memory state as tensor."""
        return self.memory.clone()

    def reset(self):
        """Reset memory to zeros."""
        self.memory.zero_()
        self.mem_usage.zero_()
        self.mem_ptr = 0
        self.count = 0


# =============================================================================
# ADWIN Drift Detection
# =============================================================================

class ADWIN:
    """ADWIN-U: Adaptive Windowing for Drift Detection.

    Detects concept drift by monitoring the mean of a data stream.
    Uses sliding window comparison to detect distributional changes.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._window: deque = deque()
        self._total: float = 0.0
        self._n: int = 0

    def update(self, value: float) -> bool:
        """Add value and check for drift.

        Returns:
            True if drift detected, False otherwise
        """
        # Add to window
        self._window.append(value)
        self._total += value
        self._n += 1

        # Check for drift
        drift_detected = False
        if self._n > 100:
            mean = self._total / self._n
            drift_detected = self._detect_drift(mean)

        # Limit window size
        if len(self._window) > 1000:
            removed = self._window.popleft()
            self._total -= removed
            self._n -= 1

        return drift_detected

    def _detect_drift(self, overall_mean: float) -> bool:
        """Detect drift using ADWIN's variance-based test."""
        n = len(self._window)
        if n < 50:
            return False

        for split in range(n // 4, 3 * n // 4):
            left = list(self._window)[:split]
            right = list(self._window)[split:]

            n1, n2 = len(left), len(right)
            if n1 < 20 or n2 < 20:
                continue

            mean1 = sum(left) / n1
            mean2 = sum(right) / n2

            # ADWIN's drift detection threshold
            m = 1.0 / (1.0 / n1 + 1.0 / n2)
            epsilon_cut = (2.0 / m) * (self.delta ** 0.5)

            if abs(mean1 - mean2) > epsilon_cut:
                return True

        return False

    def reset(self):
        """Reset ADWIN state."""
        self._window.clear()
        self._total = 0.0
        self._n = 0


# =============================================================================
# BAR Controller
# =============================================================================

class BARController:
    """Budget Allocation Rate Controller for label-efficient MemStream.

    Controls when MemStream is allowed to update its memory module.
    Only updates when ADWIN detects drift or explicitly grants budget.

    Scientific purpose:
    - Original MemStream: 100% label cost (update memory on every record)
    - With BAR: 1-5% label cost (update only when needed)
    """

    def __init__(
        self,
        memory_len: int = 1000,
        min_budget_fraction: float = 0.01,
        max_budget_fraction: float = 0.05,
        adwin_delta: float = 0.002,
    ):
        self.memory_len = memory_len
        self.min_budget_fraction = min_budget_fraction
        self.max_budget_fraction = max_budget_fraction

        # ADWIN per neighborhood
        self._adwins: dict = {}

        # Budget state
        self._budget_granted: bool = False
        self._recent_updates: deque = deque(maxlen=10000)
        self._recent_records: deque = deque(maxlen=10000)

        # Statistics
        self._total_records: int = 0
        self._memory_updates: int = 0
        self._drift_events: int = 0

        # ADWIN config
        self._adwin_delta = adwin_delta

    @property
    def bar_rate(self) -> float:
        """Current BAR score (Budget Allocation Rate)."""
        if not self._recent_records:
            return 0.0
        return sum(self._recent_updates) / len(self._recent_records)

    @property
    def bar_rate_pct(self) -> float:
        """Current BAR score as percentage."""
        return self.bar_rate * 100

    def _get_adwin(self, neighborhood: str) -> ADWIN:
        """Get or create ADWIN for neighborhood."""
        if neighborhood not in self._adwins:
            self._adwins[neighborhood] = ADWIN(delta=self._adwin_delta)
        return self._adwins[neighborhood]

    def should_update_memory(
        self,
        neighborhood: str,
        score: float
    ) -> tuple[bool, str]:
        """Determine if memory should be updated for this record.

        Returns:
            Tuple of (should_update: bool, reason: str)
        """
        self._total_records += 1
        self._recent_records.append(1)

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

        # Rule 2: Explicit Budget Grant
        if self._budget_granted:
            self._memory_updates += 1
            self._recent_updates.append(1)
            self._budget_granted = False
            return True, "iec_budget_granted"

        # Rule 3: Minimum Budget Guarantee
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
        """Get BAR statistics."""
        return {
            'total_records': self._total_records,
            'memory_updates': self._memory_updates,
            'drift_events': self._drift_events,
            'bar_rate': self.bar_rate,
            'bar_rate_pct': self.bar_rate_pct,
        }

    def reset(self):
        """Reset all counters and state."""
        self._total_records = 0
        self._memory_updates = 0
        self._drift_events = 0
        self._budget_granted = False
        self._recent_updates.clear()
        self._recent_records.clear()
        for adwin in self._adwins.values():
            adwin.reset()


# =============================================================================
# MemStream Core
# =============================================================================

class MemStreamCore:
    """MemStream: Online Autoencoder + Memory Module.

    Scoring:
    1. Encode input x -> z
    2. Compute reconstruction error: ||x - decoder(z)||
    3. Find k nearest neighbors in memory
    4. Final score = max(recon_error, memory_distance)
    5. Score > beta -> ANOMALY

    Memory Update (streaming):
    1. Encode input x -> z (detached)
    2. Add z to memory (FIFO replacement, controlled by BAR)
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
        self.max_thres: torch.Tensor = torch.tensor(
            0.0, dtype=torch.float32, device=device
        )

        # Count of samples processed
        self.count = 0

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize using frozen stats."""
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / (self.std + 1e-8)

    def warmup(
        self,
        X_normal: np.ndarray,
        epochs: int = None,
        batch_size: int = None,
        verbose: bool = True
    ):
        """Warmup phase: train AE + initialize memory.

        Args:
            X_normal: Normal training data [N, 25], float32
            epochs: Training epochs (default from config)
            batch_size: Batch size (default from config)
        """
        set_determinism(self.cfg.seed)

        epochs = epochs or self.cfg.warmup_epochs
        batch_size = batch_size or self.cfg.warmup_batch_size

        X = torch.from_numpy(X_normal).float().to(self.device)
        n = len(X)

        # Compute normalization stats from first 10%
        n_stats = max(1, int(n * 0.1))
        stats_data = X[:n_stats]

        self.mean = stats_data.mean(dim=0)
        self.std = stats_data.std(dim=0)
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

            avg_loss = total_loss / max(1, len(X_shuffled) / batch_size)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.ae.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.warmup_early_stop_patience:
                    if verbose:
                        LOGGER.info(f"[MemStream] Early stop at epoch {epoch+1}")
                    break

            if verbose and (epoch + 1) % 100 == 0:
                LOGGER.info(f"[MemStream] Epoch {epoch+1}: loss = {avg_loss:.6f}")

        # Load best model
        if best_state is not None:
            self.ae.load_state_dict(best_state)

        self.ae.eval()

        # Initialize memory with last 10%
        self.eval_mode = True
        memory_data = X[int(n * 0.9):]
        with torch.no_grad():
            memory_encoded = self.ae.encoder(memory_data)
            n_memory = min(self.cfg.memory_len, len(memory_encoded))
            indices = torch.randperm(len(memory_encoded))[:n_memory]
            self.memory.memory = memory_encoded[indices].clone()
            self.memory.mem_usage.fill_(1.0)
            self.memory.count = n_memory

        if verbose:
            LOGGER.info(f"[MemStream] Warmup complete: {epoch+1} epochs, best_loss = {best_loss:.6f}")
            LOGGER.info(f"[MemStream] Memory initialized with {n_memory} samples")

    def score_one(self, x: np.ndarray) -> float:
        """Score a single record.

        Returns anomaly score (higher = more anomalous).
        """
        x_t = torch.from_numpy(x).float().to(self.device)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)

        with torch.no_grad():
            x_norm = self._normalize(x_t)
            x_recon = self.ae(x_norm)

            # Reconstruction error
            recon_error = torch.mean(
                (x_norm - x_recon) ** 2, dim=1
            )

            # Memory distance (kNN)
            z = self.ae.encoder(x_norm)
            if z.dim() == 1:
                z = z.unsqueeze(0)

            memory = self.memory.get_memory()
            dist_to_memory = torch.cdist(z, memory, p=2)
            k = min(self.cfg.k_neighbors, memory.shape[0])
            min_dist = dist_to_memory[0, :k].mean()

            # Final score = max(recon_error, memory_distance)
            score = max(recon_error[0].item(), min_dist.item())

            return score

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Score a batch of records.

        Args:
            X: Batch of records, shape [N, in_dim]

        Returns:
            Array of anomaly scores [N], higher = more anomalous
        """
        X_t = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            x_norm = self._normalize(X_t)
            x_recon = self.ae(x_norm)

            # Reconstruction error per sample
            recon_error = torch.mean((x_norm - x_recon) ** 2, dim=1)

            # Memory distance
            z = self.ae.encoder(x_norm)
            memory = self.memory.get_memory()
            dist_to_memory = torch.cdist(z, memory, p=2)
            k = min(self.cfg.k_neighbors, memory.shape[0])
            min_dist = dist_to_memory[:, :k].mean(dim=1)

            # Final score = max(recon_error, memory_distance)
            scores = torch.maximum(recon_error, min_dist)

            return scores.cpu().numpy()

    def memory_update(self, x: np.ndarray):
        """Streaming memory update (call after scoring each record).

        Encodes input (detached) and adds to memory.
        """
        x_t = torch.from_numpy(x).float().to(self.device)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)

        with torch.no_grad():
            z = self.ae.encoder(x_t)
            self.memory.update(z[0])

        self.count += 1

    def set_beta(self, beta: float):
        """Set anomaly threshold."""
        self.max_thres = torch.tensor(
            beta, dtype=torch.float32, device=self.device
        )

    def clone(self) -> 'MemStreamCore':
        """Create a copy with same weights but fresh memory."""
        new_ms = MemStreamCore(cfg=self.cfg, device=self.device)
        new_ms.ae.load_state_dict(self.ae.state_dict())
        new_ms.mean = self.mean.clone() if self.mean is not None else None
        new_ms.std = self.std.clone() if self.std is not None else None
        new_ms.max_thres = self.max_thres.clone()
        new_ms.eval_mode = self.eval_mode
        return new_ms

    def get_state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            'memory': self.memory.memory.cpu(),
            'memory_count': self.memory.count,
            'memory_ptr': self.memory.mem_ptr,
            'count': self.count,
            'max_thres': self.max_thres.item() if hasattr(self.max_thres, 'item') else self.max_thres,
            'eval_mode': self.eval_mode,
        }

    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self.memory.memory = state['memory'].to(self.device)
        self.memory.count = state.get('memory_count', 0)
        self.memory.mem_ptr = state.get('memory_ptr', 0)
        self.count = state.get('count', 0)
        self.max_thres = torch.tensor(
            state.get('max_thres', 0.0),
            dtype=torch.float32,
            device=self.device
        )
        self.eval_mode = state.get('eval_mode', True)
