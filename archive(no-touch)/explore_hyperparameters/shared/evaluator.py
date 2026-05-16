"""
GPU evaluator for hyperparameter experiments.
Wraps a PyTorch autoencoder + memory + kNN scoring.
Supports both streaming (per-record) and batch scoring modes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


# ============================
# GPU Memory Module
# ============================

class GPUMemory:
    """GPU-side circular buffer for streaming memory."""

    def __init__(self, memory_len, out_dim, device):
        self.memory    = torch.zeros(memory_len, out_dim, device=device)
        self.mem_usage = torch.zeros(memory_len, device=device)
        self.ptr       = 0
        self.count     = 0
        self._is_full  = False
        self.device    = device

    def update(self, z):
        """Add encoded sample to memory."""
        self.memory[self.ptr] = z.detach()
        self.mem_usage[self.ptr] = 1.0
        self.ptr = (self.ptr + 1) % self.memory.shape[0]
        if not self._is_full:
            self.count += 1
            if self.count >= self.memory.shape[0]:
                self._is_full = True

    def get(self):
        return self.memory[:self.count]


# ============================
# GPU ContextBeta
# ============================

class GPUContextBeta:
    """GPU-friendly ContextBeta implementation."""

    def __init__(self, n_neighborhoods=10, n_cells=8, percentile=95):
        self.n_neighborhoods = n_neighborhoods
        self.n_cells        = n_cells
        self.percentile     = percentile
        self.betas = np.ones((n_neighborhoods, n_cells), dtype=np.float32) * 0.5

    def fit(self, scores, neighborhood_ids, context_ids):
        """Fit thresholds per (neighborhood, context) cell."""
        from .data_loader import get_context_id
        for n in range(self.n_neighborhoods):
            for c in range(self.n_cells):
                cell_scores = [
                    s for s, nm, ctx in zip(scores, neighborhood_ids, context_ids)
                    if nm == n and ctx == c
                ]
                if len(cell_scores) >= 50:
                    self.betas[n, c] = float(np.percentile(cell_scores, self.percentile))

    def get_beta(self, neighborhood_id, context_id):
        n = min(int(neighborhood_id), self.n_neighborhoods - 1)
        c = min(int(context_id), self.n_cells - 1)
        return float(self.betas[n, c])

    @property
    def non_default_count(self):
        return int((self.betas != 0.5).sum())

    def coverage_report(self, neighborhood_ids, context_ids):
        """Return coverage per context cell."""
        from collections import Counter
        cell_counts = Counter(zip(neighborhood_ids, context_ids))
        return {f'{n},{c}': cell_counts.get((n, c), 0)
                for n in range(self.n_neighborhoods)
                for c in range(self.n_cells)}


# ============================
# GPU Experiment Model
# ============================

class GPUExperimentModel:
    """End-to-end GPU model for anomaly detection experiments.

    Wraps:
      - StandardScaler
      - Denoising Autoencoder (PyTorch, GPU)
      - GPU Memory Buffer (circular, kNN)
      - ContextBeta thresholds
    """

    def __init__(self, memory_len=256, k=10, gamma=0.0, latent_dim=60,
                 default_beta=0.5, seed=42, device='cuda'):
        """
        Args:
            memory_len: Maximum memory buffer size
            k: Number of nearest neighbors for scoring
            gamma: Decay factor for weighted kNN (0 = uniform)
            latent_dim: Autoencoder hidden dimension
            default_beta: Fallback anomaly threshold
            seed: Random seed
            device: 'cuda' or 'cpu'
        """
        self.memory_len   = memory_len
        self.k            = k
        self.gamma        = gamma
        self.latent_dim   = latent_dim
        self.default_beta = default_beta
        self.seed         = seed
        self.device       = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.scaler       = None
        self._W1 = self._b1 = self._W2 = self._b2 = None
        self.memory        = None
        self.cb            = None
        self._trained      = False
        self._warmup_size  = 0
        self._warmup_mem   = None
        self._warmup_count = 0

    # ------------------------------------------------------------------
    # Training / Warmup
    # ------------------------------------------------------------------

    def fit(self, X_warmup, neighborhood_ids=None, hour_vals=None,
            dow_vals=None, ratecode_vals=None, epochs=20,
            batch_size=256, lr=1e-3, noise_std=0.1):
        """Train autoencoder and initialize memory from warmup data.

        Args:
            X_warmup: Training data (n_samples, n_features)
            neighborhood_ids: Optional array of neighborhood IDs
            hour_vals: Optional array of hour values
            dow_vals: Optional array of day-of-week values
            ratecode_vals: Optional array of ratecode values
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            noise_std: Denoising noise std
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Standardize
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_warmup.astype(np.float64)).astype(np.float32)
        self._warmup_size = len(Xs)

        # Build autoencoder
        d = Xs.shape[1]
        W1 = torch.nn.Parameter(
            torch.randn(d, self.latent_dim, dtype=torch.float32, device=self.device) * np.sqrt(2.0 / d))
        b1 = torch.nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32, device=self.device))
        W2 = torch.nn.Parameter(
            torch.randn(self.latent_dim, d, dtype=torch.float32, device=self.device) * np.sqrt(2.0 / self.latent_dim))
        b2 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32, device=self.device))
        optimizer = torch.optim.Adam([W1, b1, W2, b2], lr=lr)

        Xt = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
        n_batches = max(1, len(Xt) // batch_size)

        for epoch in range(epochs):
            idx = torch.randperm(len(Xt), device=self.device)
            total_loss = 0.0
            for i in range(n_batches):
                batch_idx = idx[i * batch_size:(i + 1) * batch_size]
                xb = Xt[batch_idx]
                x_noisy = xb + torch.randn_like(xb) * noise_std
                z    = F.relu(x_noisy @ W1 + b1)
                x_rec = z @ W2 + b2
                loss = F.mse_loss(x_rec, xb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([W1, b1, W2, b2], 1.0)
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)
            total_loss /= len(Xt)

        # Freeze weights
        self._W1 = W1.detach()
        self._b1 = b1.detach()
        self._W2 = W2.detach()
        self._b2 = b2.detach()

        # Encode warmup data
        with torch.no_grad():
            Z = F.relu(Xt @ self._W1 + self._b1).cpu().numpy()

        # Initialize warmup memory for scoring: holds all warmup encodings
        warmup_n = len(Z)
        warmup_mem = torch.zeros(warmup_n, self.latent_dim, device=self.device)
        for i in range(warmup_n):
            warmup_mem[i] = torch.from_numpy(Z[i:i+1].astype(np.float32)).to(self.device)

        # Create GPUMemory as circular streaming buffer (limited by memory_len)
        self.memory = GPUMemory(self.memory_len, self.latent_dim, self.device)
        # Seed streaming buffer with last memory_len warmup samples
        seed_n = min(self.memory_len, warmup_n)
        for i in range(seed_n):
            idx = warmup_n - seed_n + i
            self.memory.memory[i] = warmup_mem[idx]
            self.memory.mem_usage[i] = 1.0
            self.memory.count += 1
        if seed_n >= self.memory_len:
            self.memory._is_full = True

        # Store warmup memory for kNN scoring
        self._warmup_mem = warmup_mem
        self._warmup_count = warmup_n

        # Fit ContextBeta from warmup scores
        hr_v  = (X_warmup[:, 9].astype(int) if hour_vals is None else hour_vals.astype(int))[:len(Xs)]
        dw_v  = (X_warmup[:, 10].astype(int) if dow_vals is None else dow_vals.astype(int))[:len(Xs)]
        nb_v  = (np.zeros(len(Xs), dtype=int) if neighborhood_ids is None
                 else neighborhood_ids.astype(int)[:len(Xs)])
        rc_v  = (X_warmup[:, 25].astype(int) if ratecode_vals is None
                 else ratecode_vals.astype(int))[:len(Xs)]

        from .data_loader import get_context_id
        ctx_ids = np.array([get_context_id(int(h), int(d), int(r))
                           for h, d, r in zip(hr_v, dw_v, rc_v)])

        # Compute warmup scores (chunked to avoid OOM)
        wu_t = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
        with torch.no_grad():
            Z_wu = F.relu(wu_t @ self._W1 + self._b1)
        warmup_scores = self._score_batch_raw(Z_wu)

        self.cb = GPUContextBeta(percentile=95)
        self.cb.fit(warmup_scores, nb_v, ctx_ids)

        self._trained = True

        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, X):
        """Encode input to latent space on GPU."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        with torch.no_grad():
            return F.relu(X @ self._W1 + self._b1)

    def _score_batch_raw(self, Z):
        """Compute raw anomaly scores using chunked kNN (OOM-safe).

        Uses warmup memory if available, otherwise falls back to circular buffer.
        """
        # Use warmup memory if available (warmup phase)
        if self._warmup_mem is not None:
            M = self._warmup_count
            mem = self._warmup_mem[:M]
        else:
            M = self.memory.count
            if M < 2:
                return np.full(len(Z), 0.5)
            mem = self.memory.memory[:M]

        k_use = min(self.k, M)
        n = len(Z)
        scores = np.zeros(n, dtype=np.float64)

        if self.gamma > 0:
            weights = self.gamma ** torch.arange(k_use, device=self.device, dtype=torch.float32)
        else:
            weights = None

        # Process samples in chunks
        sample_chunk_size = 500
        mem_chunk_size   = 2000

        for s_start in range(0, n, sample_chunk_size):
            s_end = min(s_start + sample_chunk_size, n)
            Z_chunk = Z[s_start:s_end]

            best_vals = torch.full((s_end - s_start, k_use), 1e9,
                                  device=self.device, dtype=torch.float32)

            for m_start in range(0, M, mem_chunk_size):
                m_end = min(m_start + mem_chunk_size, M)
                mem_chunk = mem[m_start:m_end]

                diff   = Z_chunk.unsqueeze(1) - mem_chunk.unsqueeze(0)
                dists  = diff.abs().sum(dim=2)

                candidates = torch.cat([best_vals, dists], dim=1)
                best_vals, _ = candidates.topk(k_use, dim=1, largest=False)

            if weights is not None:
                chunk_scores = (best_vals * weights).sum(dim=1)
            else:
                chunk_scores = best_vals.sum(dim=1)
            scores[s_start:s_end] = chunk_scores.cpu().numpy()

        return scores

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_batch(self, X_test, neighborhood_ids=None, hour_vals=None,
                    dow_vals=None, ratecode_vals=None, update_memory=False,
                    warmup_end=500):
        """Score all test records in batch mode.

        Args:
            X_test: Test features (n, 34)
            neighborhood_ids: Array of neighborhood IDs
            hour_vals: Array of hour values
            dow_vals: Array of day-of-week values
            ratecode_vals: Array of ratecode values
            update_memory: If True, update memory during scoring (streaming)
            warmup_end: Index after which to start memory updates

        Returns:
            scores: np.ndarray of anomaly scores (n,)
        """
        if not self._trained:
            raise RuntimeError("Model must be fit() before scoring")

        Xs = self.scaler.transform(X_test.astype(np.float64)).astype(np.float32)
        n  = len(Xs)

        # Batch encode
        Xt = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
        with torch.no_grad():
            Z = self._encode(Xt)

        # Raw scores (batch kNN)
        raw_scores = self._score_batch_raw(Z)

        # Normalize by ContextBeta
        nb_ids  = (np.zeros(n, dtype=int) if neighborhood_ids is None
                   else neighborhood_ids.astype(int))
        hr_vals = (np.zeros(n, dtype=int) if hour_vals is None
                   else hour_vals.astype(int))
        dw_vals = (np.zeros(n, dtype=int) if dow_vals is None
                   else dow_vals.astype(int))
        rc_vals = (np.ones(n, dtype=float) if ratecode_vals is None
                   else ratecode_vals.astype(float))

        # Score: use raw kNN distances directly (higher = more anomalous)
        # ContextBeta context is used for threshold selection, not score normalization
        return raw_scores

    def score_streaming(self, X_test, neighborhood_ids=None, hour_vals=None,
                       dow_vals=None, ratecode_vals=None):
        """Score records one-by-one (streaming mode, records memory updates).

        Also tracks per-record latency.

        Args:
            X_test: Test features (n, 34)
            neighborhood_ids: Array of neighborhood IDs
            hour_vals: Array of hour values
            dow_vals: Array of day-of-week values
            ratecode_vals: Array of ratecode values

        Returns:
            scores: np.ndarray of anomaly scores (n,)
            latencies: np.ndarray of per-record latencies in ms (n,)
        """
        if not self._trained:
            raise RuntimeError("Model must be fit() before scoring")

        Xs = self.scaler.transform(X_test.astype(np.float64)).astype(np.float32)
        n  = len(Xs)

        nb_ids  = (np.zeros(n, dtype=int) if neighborhood_ids is None
                   else neighborhood_ids.astype(int))
        hr_vals = (np.zeros(n, dtype=int) if hour_vals is None
                   else hour_vals.astype(int))
        dw_vals = (np.zeros(n, dtype=int) if dow_vals is None
                   else dow_vals.astype(int))
        rc_vals = (np.ones(n, dtype=float) if ratecode_vals is None
                   else ratecode_vals.astype(float))

        from .data_loader import get_context_id
        scores    = np.zeros(n, dtype=np.float64)
        latencies = np.zeros(n, dtype=np.float64)

        warmup_end = 500
        # Use warmup memory for scoring (stable reference set)
        scoring_mem = self._warmup_mem[:self._warmup_count] if self._warmup_mem is not None else self.memory.get()
        scoring_count = self._warmup_count if self._warmup_mem is not None else self.memory.count

        for i in range(n):
            t_start = time.perf_counter()

            x_t = torch.from_numpy(Xs[i:i+1].astype(np.float32)).to(self.device)
            with torch.no_grad():
                z = self._encode(x_t)

            # Score against WARMUP memory (stable reference)
            if scoring_count >= 2:
                mem = scoring_mem[:scoring_count]
                diff  = z - mem.unsqueeze(0)
                dists = diff.abs().sum(dim=2)
                k_use = min(self.k, scoring_count)
                tk    = dists.topk(k_use, dim=1, largest=False, sorted=True)
                if self.gamma > 0:
                    w = self.gamma ** torch.arange(k_use, device=self.device, dtype=torch.float32)
                    scores[i] = (tk.values * w).sum(dim=1)[0].cpu().item()
                else:
                    scores[i] = tk.values.sum(dim=1)[0].cpu().item()
            else:
                scores[i] = 0.5

            t_end = time.perf_counter()
            latencies[i] = (t_end - t_start) * 1000.0

            # Memory update (for latency tracking; scoring uses warmup mem)
            if i >= warmup_end:
                z_det = z[0].detach()
                self.memory.update(z_det)

        return scores, latencies

    def clone(self):
        """Return a deep copy of this model with fresh memory."""
        import copy
        new_model = GPUExperimentModel(
            memory_len=self.memory_len,
            k=self.k,
            gamma=self.gamma,
            latent_dim=self.latent_dim,
            default_beta=self.default_beta,
            seed=self.seed,
            device=str(self.device),
        )
        new_model.scaler = copy.deepcopy(self.scaler)
        new_model._W1 = self._W1
        new_model._b1 = self._b1
        new_model._W2 = self._W2
        new_model._b2 = self._b2
        new_model.cb  = copy.deepcopy(self.cb)
        new_model.memory = GPUMemory(self.memory_len, self.latent_dim, self.device)
        new_model._trained = True
        new_model._warmup_size = self._warmup_size
        return new_model

    def reset_memory(self, seed_samples=None):
        """Reset memory buffer. If seed_samples provided, reinitialize from those."""
        self.memory = GPUMemory(self.memory_len, self.latent_dim, self.device)
        if seed_samples is not None:
            Z = self._encode(seed_samples.astype(np.float32))
            for z in Z:
                self.memory.update(z.detach())

    def get_memory_utilization(self):
        """Return memory buffer utilization fraction."""
        if self.memory is None:
            return 0.0
        return self.memory.count / self.memory_len
