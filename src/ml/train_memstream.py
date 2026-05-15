#!/usr/bin/env python3
"""
MemStream Training Pipeline (Phase 2B)
=====================================

Production training script for MemStream CA-DQStream pipeline.

Key Features:
- HARD-BLOCK warmup data enforcement (minimum 50K samples, 100% anomaly-free, Mon-Sun coverage)
- Leakage-free data splits: stats from first 10%, train on middle 80%, memory init from last 10%
- 500 epochs training with early stopping
- HMAC-SHA256 checkpoint signing
- ContextBeta threshold management (80 thresholds: 10 neighborhoods x 8 cells)
- Near-zero-variance feature detection and clamping
- Quick retrain capability for streaming adaptation

Usage:
  python src/ml/train_memstream.py --data data/clean/jan_2024_clean_baseline.parquet
  python src/ml/train_memstream.py --data data/clean/jan_2024_clean_baseline.parquet --epochs 500
  python src/ml/train_memstream.py --data data/clean/jan_2024_clean_baseline.parquet --quick-retrain
"""

import argparse
import copy
import hashlib
import hmac
import io
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =============================================================================
# PATH SETUP
# =============================================================================

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml.memstream_core import (
    MemStreamCore,
    MemStreamConfig,
    MemStreamAE,
    MemoryModule,
    ContextBeta,
    set_determinism,
)

# =============================================================================
# LOGGING
# =============================================================================

LOGGER = logging.getLogger('train-memstream')
if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


# =============================================================================
# FEATURE EXTRACTION (34D)
# =============================================================================

def _zone_to_grid(zone_id: int) -> Tuple[int, int]:
    """Map zone [1,265] to grid coordinates (16x17 grid = 272 cells)."""
    if zone_id <= 0:
        return 0, 0
    gx = (zone_id - 1) % 16
    gy = (zone_id - 1) // 16
    return gx, gy


def extract_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 34D feature vector from DataFrame.

    Returns:
        Tuple of (X, hour_vals, dow_vals, ratecode_vals)
    """
    n = len(df)
    N_FEATURES = 34

    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    # Parse datetime
    pickup = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    hour = pickup.dt.hour.fillna(12).astype(np.int32).values
    dow = pickup.dt.dayofweek.fillna(0).astype(np.int32).values

    # Extract base features
    dist = df['trip_distance'].fillna(0).values.astype(np.float32)
    dur_min = df['duration_s'].fillna(1).values.astype(np.float32) / 60.0
    fare = df['fare_amount'].fillna(0).values.astype(np.float32)
    pax = df['passenger_count'].fillna(1).values.astype(np.float32)
    total = df['total_amt'].fillna(0).values.astype(np.float32)
    speed = df['speed_mph'].fillna(0).values.astype(np.float32)
    ratecode = df['RatecodeID'].fillna(1).values.astype(np.float32)

    pu_loc = df['PULocationID'].fillna(0).values
    do_loc = df['DOLocationID'].fillna(0).values

    eps = np.float32(0.01)

    # Raw + trip (indices 0-5)
    X[:, 0] = dist
    X[:, 1] = dur_min
    X[:, 2] = fare
    X[:, 3] = pax
    X[:, 4] = total
    X[:, 5] = speed

    # Ratios (indices 6-8)
    X[:, 6] = fare / np.maximum(dist, eps)
    X[:, 7] = fare / np.maximum(dur_min, eps)
    X[:, 8] = fare / np.maximum(pax, eps)

    # Temporal raw (indices 9-11)
    X[:, 9] = hour.astype(np.float32)
    X[:, 10] = dow.astype(np.float32)
    X[:, 11] = (dow >= 5).astype(np.float32)

    # SPATIAL GRID: 4 features (indices 12-15)
    for i in range(n):
        pux, puy = _zone_to_grid(int(pu_loc[i]) if not pd.isna(pu_loc[i]) else 0)
        dox, doy = _zone_to_grid(int(do_loc[i]) if not pd.isna(do_loc[i]) else 0)
        X[i, 12] = float(pux)
        X[i, 13] = float(puy)
        X[i, 14] = float(dox)
        X[i, 15] = float(doy)

    # Normalized ratios (indices 16-19)
    X[:, 16] = X[:, 6] / np.float32(2.5)
    X[:, 17] = X[:, 7] / np.float32(0.67)
    X[:, 18] = speed / np.float32(12.0)
    X[:, 19] = pax / np.maximum(dist, eps)

    # Cyclic temporal (indices 20-24)
    X[:, 20] = np.sin(np.float32(2 * np.pi) * hour.astype(np.float32) / 24.0)
    X[:, 21] = np.cos(np.float32(2 * np.pi) * hour.astype(np.float32) / 24.0)
    X[:, 22] = np.sin(np.float32(2 * np.pi) * dow.astype(np.float32) / 7.0)
    X[:, 23] = np.cos(np.float32(2 * np.pi) * dow.astype(np.float32) / 7.0)

    # Distance squared (index 24)
    X[:, 24] = dist * dist

    # RatecodeID one-hot (indices 25-29)
    for i, rc in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        X[:, 25 + i] = (ratecode == rc).astype(np.float32)

    # is_night indicator (index 30)
    X[:, 30] = ((hour >= 20) | (hour <= 6)).astype(np.float32)

    # Log-transformed fare (index 31)
    X[:, 31] = np.log1p(fare)

    # Log-transformed distance (index 32)
    X[:, 32] = np.log1p(dist)

    # Inter-borough rough indicator (index 33)
    pu_gy = np.zeros(n, dtype=np.float32)
    do_gy = np.zeros(n, dtype=np.float32)
    for i in range(n):
        _, puy = _zone_to_grid(int(pu_loc[i]) if not pd.isna(pu_loc[i]) else 0)
        _, doy = _zone_to_grid(int(do_loc[i]) if not pd.isna(do_loc[i]) else 0)
        pu_gy[i] = float(puy)
        do_gy[i] = float(doy)
    X[:, 33] = np.abs(pu_gy - do_gy)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)

    # Ratecode extraction from one-hot (decode to scalar)
    ratecode_vals = np.array([
        sum((i + 1) * X[i, 25 + i] for i in range(5))
        for i in range(n)
    ], dtype=np.int32)

    return X, hour.astype(np.int32), dow.astype(np.int32), ratecode_vals


# =============================================================================
# NEIGHBORHOOD MAPPING
# =============================================================================

def location_to_neighborhood(loc_id) -> int:
    """Map PULocationID to neighborhood (10 clusters)."""
    if pd.isna(loc_id):
        return 9  # unknown
    z = int(loc_id)
    if 1 <= z <= 43:
        return 0   # manhattan
    elif 44 <= z <= 103:
        return 4   # bronx
    elif 104 <= z <= 127:
        return 1   # brooklyn
    elif 128 <= z <= 148:
        return 2   # queens_lower
    elif 149 <= z <= 161:
        return 3   # queens_upper (JFK area)
    elif 162 <= z <= 181:
        return 5   # staten_island
    elif 182 <= z <= 196:
        return 6   # ewr
    elif 217 <= z <= 229:
        return 7   # jfk
    elif 230 <= z <= 234:
        return 8   # nalp
    else:
        return 9   # unknown


# =============================================================================
# CONTEXT COMPUTATION
# =============================================================================

def compute_context_cell(ratecode_id: float, hour: int, day_of_week: int) -> int:
    """
    Compute context cell ID from record attributes.

    Context cell ID = (is_special << 2) | (is_night << 1) | is_weekend
    """
    is_special = 1 if ratecode_id > 1.0 else 0
    is_night = 1 if (hour >= 20 or hour < 6) else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    return (is_special << 2) | (is_night << 1) | is_weekend


# =============================================================================
# WARMUP DATA ENFORCEMENT
# =============================================================================

def enforce_warmup_data_requirements(
    X_normal: np.ndarray,
    labels: np.ndarray,
    hour_vals: np.ndarray,
    dow_vals: np.ndarray
) -> Dict:
    """
    HARD-BLOCK enforcement of warmup data requirements.

    Raises Exception on any violation.

    Requirements:
    1. Minimum 50,000 samples
    2. 100% anomaly-free (labels==0)
    3. Mon-Sun coverage (all 7 DOW values)
    4. Sequential memory initialization from last 10%

    Returns:
        Dict with validation results for logging
    """
    validation_results = {}

    # ========== WARMUP DATA ENFORCEMENT ==========
    # 1. Minimum 50,000 samples
    if len(X_normal) < 50000:
        raise Exception(
            f"WARMUP_DATA_ERROR: Need >=50000 samples, got {len(X_normal)}"
        )
    validation_results['n_samples'] = len(X_normal)

    # 2. 100% anomaly-free (labels==0)
    if (labels != 0).any():
        n_anomalous = (labels != 0).sum()
        raise Exception(
            f"WARMUP_DATA_ERROR: Found {n_anomalous} anomalous samples. "
            f"Warmup must be 100% anomaly-free."
        )
    validation_results['anomaly_free'] = True

    # 3. Mon-Sun coverage (all 7 DOW values)
    unique_dow = set(int(d) for d in np.unique(dow_vals))
    if len(unique_dow) < 7:
        missing = set(range(7)) - unique_dow
        raise Exception(
            f"WARMUP_DATA_ERROR: Mon-Sun coverage incomplete. "
            f"Missing DOW values: {missing}"
        )
    validation_results['dow_coverage'] = sorted(unique_dow)

    # ========== END WARMUP DATA ENFORCEMENT ==========

    # Additional validation
    validation_results['hour_range'] = (int(hour_vals.min()), int(hour_vals.max()))

    return validation_results


def check_normal_ratio(
    labels: np.ndarray
) -> Tuple[bool, float]:
    """
    Check normal sample ratio and emit WARNING if < 85%.

    Returns:
        Tuple of (is_adequate, normal_ratio)
    """
    n_total = len(labels)
    n_normal = (labels == 0).sum()
    normal_ratio = n_normal / n_total if n_total > 0 else 0.0

    is_adequate = normal_ratio >= 0.85

    return is_adequate, normal_ratio


# =============================================================================
# NEAR-ZERO-VARIANCE DETECTION
# =============================================================================

def detect_near_zero_variance(
    X: np.ndarray,
    threshold: float = 0.001
) -> Tuple[List[int], np.ndarray]:
    """
    Detect features with near-zero variance.

    Args:
        X: Feature matrix [N, D]
        threshold: Variance threshold for detection

    Returns:
        Tuple of (low_var_indices, std_array)
    """
    std_array = np.std(X, axis=0)
    low_var_mask = std_array < threshold
    low_var_indices = np.where(low_var_mask)[0].tolist()

    return low_var_indices, std_array


# =============================================================================
# LEAKAGE-FREE DATA SPLITS
# =============================================================================

def prepare_leakage_free_splits(
    X: np.ndarray,
    hour_vals: np.ndarray,
    dow_vals: np.ndarray,
    ratecode_vals: np.ndarray,
    neighborhood_vals: np.ndarray
) -> Dict:
    """
    Prepare leakage-free data splits for warmup training.

    CRITICAL: Time-ordered splits, NO shuffle for memory initialization.

    Data flow:
      [10%] -> Compute normalization stats ONLY
      [80%] -> Train autoencoder (MAY use shuffle via torch.randperm)
      [10%] -> Initialize memory (SEQUENTIAL ONLY, NO shuffle)

    Returns:
        Dict with split data and indices
    """
    n = len(X)

    stats_end = int(n * 0.1)
    memory_start = int(n * 0.9)

    splits = {
        'stats_data': X[:stats_end],
        'stats_hour': hour_vals[:stats_end],
        'stats_dow': dow_vals[:stats_end],
        'stats_ratecode': ratecode_vals[:stats_end],
        'stats_neighborhood': neighborhood_vals[:stats_end],
        'train_data': X[stats_end:memory_start],
        'train_hour': hour_vals[stats_end:memory_start],
        'train_dow': dow_vals[stats_end:memory_start],
        'train_ratecode': ratecode_vals[stats_end:memory_start],
        'train_neighborhood': neighborhood_vals[stats_end:memory_start],
        'memory_data': X[memory_start:],
        'memory_hour': hour_vals[memory_start:],
        'memory_dow': dow_vals[memory_start:],
        'memory_ratecode': ratecode_vals[memory_start:],
        'memory_neighborhood': neighborhood_vals[memory_start:],
        'stats_indices': list(range(0, stats_end)),
        'train_indices': list(range(stats_end, memory_start)),
        'memory_indices': list(range(memory_start, n)),
    }

    LOGGER.info(
        f"Data splits: stats={stats_end:,} ({stats_end/n*100:.1f}%), "
        f"train={memory_start - stats_end:,} ({(memory_start-stats_end)/n*100:.1f}%), "
        f"memory={n - memory_start:,} ({(n-memory_start)/n*100:.1f}%)"
    )

    return splits


# =============================================================================
# MEMSTREAM TRAINING
# =============================================================================

class MemStreamTrainer:
    """
    Production MemStream trainer with full checkpoint management.

    Supports:
    - Full 500-epoch warmup training
    - Quick retrain from checkpoint
    - HMAC-signed checkpoints
    - ContextBeta threshold management
    """

    CHECKPOINT_NAME = 'memstream_checkpoint_v1.pt'
    HMAC_SUFFIX = '.hmac'

    def __init__(
        self,
        cfg: MemStreamConfig = None,
        device: str = 'cpu',
        signing_key: str = None,
    ):
        self.cfg = cfg or MemStreamConfig()
        self.device = device
        self.signing_key = signing_key or os.environ.get(
            'MEMSTREAM_SIGNING_KEY', 'default-signing-key'
        )

        # Initialize MemStream
        self.ms = MemStreamCore(cfg=self.cfg, device=device)

        # State
        self._is_trained = False
        self._recent_scores: List[float] = []
        self._training_history: List[Dict] = []

    def train_full(
        self,
        X_normal: np.ndarray,
        labels: np.ndarray,
        hour_vals: np.ndarray,
        dow_vals: np.ndarray,
        ratecode_vals: np.ndarray,
        neighborhood_vals: np.ndarray,
        epochs: int = 500,
        batch_size: int = 256,
        verbose: bool = True
    ) -> Dict:
        """
        Full warmup training with leakage-free splits.

        Args:
            X_normal: Feature matrix [N, 34], float32
            labels: Label array [N], int (0=normal, 1+=anomaly)
            hour_vals: Hour values [N]
            dow_vals: Day of week values [N]
            ratecode_vals: Ratecode values [N]
            neighborhood_vals: Neighborhood IDs [N]
            epochs: Training epochs (default 500)
            batch_size: Batch size (default 256)
            verbose: Print progress

        Returns:
            Dict with training results
        """
        start_time = time.time()

        # ===== ENFORCE WARMUP DATA REQUIREMENTS =====
        validation = enforce_warmup_data_requirements(
            X_normal, labels, hour_vals, dow_vals
        )

        # ===== CHECK NORMAL RATIO =====
        is_adequate, normal_ratio = check_normal_ratio(labels)
        if not is_adequate:
            LOGGER.warning(
                f"WARNING: Normal ratio {normal_ratio:.1%} < 85%. "
                f"Consider filtering anomalous samples."
            )

        # ===== NEAR-ZERO-VARIANCE DETECTION =====
        low_var_indices, std_array = detect_near_zero_variance(X_normal)
        if low_var_indices:
            LOGGER.warning(
                f"WARNING: Near-zero-variance features detected at indices: {low_var_indices}"
            )
            LOGGER.warning(
                f"  Std values: {[f'{i}={std_array[i]:.6f}' for i in low_var_indices]}"
            )
            LOGGER.warning(
                f"  Using max(std, 0.001) for normalization clamping"
            )

        # ===== PREPARE LEAKAGE-FREE SPLITS =====
        splits = prepare_leakage_free_splits(
            X_normal, hour_vals, dow_vals,
            ratecode_vals, neighborhood_vals
        )

        # ===== COMPUTE NORMALIZATION STATS FROM FIRST 10% =====
        LOGGER.info("Computing normalization stats from first 10%...")
        stats_data = torch.from_numpy(splits['stats_data']).float().to(self.device)
        self.ms.mean = stats_data.mean(dim=0)
        self.ms.std = stats_data.std(dim=0)

        # Clamp near-zero variance features
        clamp_min = 0.001
        self.ms.std = torch.clamp(self.ms.std, min=clamp_min)

        LOGGER.info(
            f"Stats: mean shape={self.ms.mean.shape}, std range=[{self.ms.std.min():.4f}, {self.ms.std.max():.4f}]"
        )

        # ===== NORMALIZE MIDDLE 80% =====
        train_data = torch.from_numpy(splits['train_data']).float().to(self.device)
        train_norm = (train_data - self.ms.mean) / (self.ms.std + 1e-8)

        # ===== AUTOENCODER TRAINING =====
        LOGGER.info(f"Training autoencoder for {epochs} epochs...")
        self._train_autoencoder(train_norm, epochs, batch_size, verbose)

        # ===== MEMORY INITIALIZATION FROM LAST 10% (SEQUENTIAL ONLY) =====
        LOGGER.info("Initializing memory from last 10% (sequential, NO shuffle)...")
        self._init_memory_sequential(splits)

        # ===== FIT CONTEXTBETA THRESHOLDS =====
        LOGGER.info("Fitting ContextBeta thresholds...")
        self._fit_context_beta(splits)

        self._is_trained = True

        # ===== TRAINING SUMMARY =====
        elapsed = time.time() - start_time
        results = {
            'status': 'success',
            'n_samples': len(X_normal),
            'epochs_trained': epochs,
            'training_time_seconds': elapsed,
            'validation': validation,
            'normal_ratio': normal_ratio,
            'low_var_features': low_var_indices,
        }

        LOGGER.info(
            f"Training complete in {elapsed:.1f}s "
            f"({len(X_normal)/elapsed:.0f} samples/sec)"
        )

        return results

    def _train_autoencoder(
        self,
        X_norm: torch.Tensor,
        epochs: int,
        batch_size: int,
        verbose: bool
    ):
        """
        Train autoencoder with denoising and early stopping.

        NOTE: AE training MAY use torch.randperm() (order-independent).
        """
        self.ms.ae.train()
        self.ms.optimizer = torch.optim.Adam(
            self.ms.ae.parameters(),
            lr=self.cfg.warmup_lr
        )

        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            # Shuffle for training (order-independent)
            indices = torch.randperm(len(X_norm), device=self.device)
            X_shuffled = X_norm[indices]

            # Mini-batch training with denoising
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_shuffled), batch_size):
                batch = X_shuffled[i:i + batch_size]

                # Add noise for robustness
                noise = torch.randn_like(batch) * self.cfg.warmup_noise_std
                x_noisy = batch + noise

                # Forward
                x_recon = self.ms.ae(x_noisy)
                loss = self.ms.criterion(x_recon, batch)

                # Backward
                self.ms.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.ms.ae.parameters(),
                    self.cfg.warmup_gradient_clip
                )
                self.ms.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = copy.deepcopy(self.ms.ae.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.warmup_early_stop_patience:
                    if verbose:
                        LOGGER.info(f"  Early stop at epoch {epoch + 1}")
                    break

            if verbose and (epoch + 1) % 100 == 0:
                LOGGER.info(f"  Epoch {epoch + 1}: loss = {avg_loss:.6f}")

            # Record history
            self._training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
            })

        # Load best model
        if best_state is not None:
            self.ms.ae.load_state_dict(best_state)

        self.ms.ae.eval()
        LOGGER.info(f"  Best loss: {best_loss:.6f}")

    def _init_memory_sequential(self, splits: Dict):
        """
        Initialize memory from last 10% using SEQUENTIAL samples.

        CRITICAL: This must use sequential (time-ordered) samples.
        DO NOT use torch.randperm() here.
        """
        memory_data = torch.from_numpy(splits['memory_data']).float().to(self.device)
        n_memory = min(self.cfg.memory_len, len(memory_data))

        with torch.no_grad():
            # Encode using current AE
            memory_encoded = self.ms.ae.encoder(memory_data)

            # CRITICAL: Use SEQUENTIAL samples from end, NO shuffle
            # This preserves time-ordering for streaming evaluation
            sequential_indices = torch.arange(
                len(memory_encoded) - n_memory,
                len(memory_encoded),
                device=self.device
            )

            self.ms.memory.memory = memory_encoded[sequential_indices].clone()
            self.ms.memory.mem_usage.fill_(1.0)
            self.ms.memory.count = n_memory
            self.ms.memory.mem_ptr = 0

        LOGGER.info(f"  Memory initialized with {n_memory} sequential samples")

    def _fit_context_beta(self, splits: Dict):
        """
        Fit ContextBeta thresholds from warmup scores.

        Computes 80 thresholds: 10 neighborhoods x 8 context cells.
        """
        # Compute scores for warmup subset
        warmup_n = min(50000, len(splits['train_data']))
        warmup_data = torch.from_numpy(splits['train_data'][:warmup_n]).float()

        # Normalize
        warmup_norm = (warmup_data - self.ms.mean) / (self.ms.std + 1e-8)

        # Compute scores
        self.ms.eval_mode = True
        scores = []

        with torch.no_grad():
            for i in range(len(warmup_norm)):
                x = warmup_norm[i].unsqueeze(0).to(self.device)
                z = self.ms.ae.encoder(x)

                # kNN in memory
                memory = self.ms.memory.get_memory()
                if memory.shape[0] >= 2:
                    dists = torch.cdist(z, memory, p=1).squeeze()
                    k = min(self.cfg.k_neighbors, len(dists))
                    top_k = dists.topk(k, largest=False).values
                    score = top_k.mean().item()
                else:
                    score = 0.5

                scores.append(score)

        # Get context info
        neighborhood_ids = splits['train_neighborhood'][:warmup_n]
        context_ids = np.array([
            compute_context_cell(
                splits['train_ratecode'][i],
                int(splits['train_hour'][i]),
                int(splits['train_dow'][i])
            )
            for i in range(warmup_n)
        ])

        # Fit ContextBeta
        self.ms._context_beta = ContextBeta(n_neighborhoods=10, n_cells=8, percentile=95)
        self.ms._context_beta.fit_from_scores(
            scores,
            neighborhood_ids.astype(int),
            context_ids.astype(int)
        )

        # Compute overall beta
        overall_beta = float(np.percentile(scores, 95))
        self.ms.set_beta(overall_beta)
        self.ms._score_buf = scores.copy()
        self.ms._warmup_scores = scores.copy()

        LOGGER.info(f"  ContextBeta fitted: overall beta = {overall_beta:.4f}")

        # Store recent scores
        self._recent_scores = scores[-10000:] if len(scores) > 10000 else scores

    def quick_retrain(
        self,
        X_recent: np.ndarray,
        hour_vals: np.ndarray,
        dow_vals: np.ndarray,
        ratecode_vals: np.ndarray,
        neighborhood_vals: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = True
    ) -> Dict:
        """
        Quick retrain for streaming adaptation.

        Warms-start from current weights and fine-tunes on recent window.

        Args:
            X_recent: Recent window of normal samples [N, 34]
            hour_vals: Hour values [N]
            dow_vals: Day of week values [N]
            ratecode_vals: Ratecode values [N]
            neighborhood_vals: Neighborhood IDs [N]
            epochs: Fine-tuning epochs (default 100)
            batch_size: Batch size (default 256)
            verbose: Print progress

        Returns:
            Dict with retraining results
        """
        if not self._is_trained:
            raise RuntimeError("Must train full model before quick retrain")

        start_time = time.time()

        LOGGER.info(f"Quick retrain on {len(X_recent)} samples for {epochs} epochs...")

        # Normalize using existing stats
        X_t = torch.from_numpy(X_recent).float().to(self.device)
        X_norm = (X_t - self.ms.mean) / (self.ms.std + 1e-8)

        # Fine-tune with lower learning rate
        self.ms.ae.train()
        fine_tune_lr = self.cfg.warmup_lr * 0.1  # Lower LR for fine-tuning
        optimizer = torch.optim.Adam(self.ms.ae.parameters(), lr=fine_tune_lr)

        for epoch in range(epochs):
            # Shuffle
            indices = torch.randperm(len(X_norm), device=self.device)
            X_shuffled = X_norm[indices]

            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_shuffled), batch_size):
                batch = X_shuffled[i:i + batch_size]

                noise = torch.randn_like(batch) * self.cfg.warmup_noise_std * 0.5
                x_noisy = batch + noise

                x_recon = self.ms.ae(x_noisy)
                loss = self.ms.criterion(x_recon, batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.ms.ae.parameters(),
                    self.cfg.warmup_gradient_clip
                )
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)

            if verbose and (epoch + 1) % 20 == 0:
                LOGGER.info(f"  Epoch {epoch + 1}: loss = {avg_loss:.6f}")

        self.ms.ae.eval()

        # Update recent scores
        self._update_recent_scores(X_norm)

        elapsed = time.time() - start_time
        results = {
            'status': 'success',
            'n_samples': len(X_recent),
            'epochs': epochs,
            'retrain_time_seconds': elapsed,
        }

        LOGGER.info(f"Quick retrain complete in {elapsed:.1f}s")

        return results

    def _update_recent_scores(self, X_norm: torch.Tensor):
        """Update recent scores buffer after retraining."""
        scores = []
        with torch.no_grad():
            for i in range(len(X_norm)):
                x = X_norm[i].unsqueeze(0).to(self.device)
                z = self.ms.ae.encoder(x)

                memory = self.ms.memory.get_memory()
                if memory.shape[0] >= 2:
                    dists = torch.cdist(z, memory, p=1).squeeze()
                    k = min(self.cfg.k_neighbors, len(dists))
                    top_k = dists.topk(k, largest=False).values
                    score = top_k.mean().item()
                else:
                    score = 0.5

                scores.append(score)

        # Keep last 10000 scores
        self._recent_scores = scores[-10000:] if len(scores) > 10000 else scores

    def save_checkpoint(
        self,
        path: str,
        include_hmac: bool = True
    ) -> str:
        """
        Save unified checkpoint with HMAC signature.

        Checkpoint contains:
        - ae_state: Autoencoder state dict
        - memory: Memory module state
        - mean/std: Normalization statistics
        - context_beta: 80 threshold values
        - _recent_scores: Recent score buffer

        Args:
            path: Output directory
            include_hmac: Generate HMAC signature

        Returns:
            Path to checkpoint file
        """
        if not self._is_trained:
            raise RuntimeError("Must train before saving checkpoint")

        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / self.CHECKPOINT_NAME

        # Build checkpoint state
        state = {
            'ae_state': self.ms.ae.state_dict(),
            'memory': self.ms.memory.get_memory().cpu(),
            'memory_mem_usage': self.ms.memory.mem_usage.cpu(),
            'memory_mem_ptr': self.ms.memory.mem_ptr,
            'memory_count': self.ms.memory.count,
            'mean': self.ms.mean.cpu(),
            'std': self.ms.std.cpu(),
            'context_beta': self.ms._context_beta.betas if self.ms._context_beta else None,
            'context_beta_n_neighborhoods': self.ms._context_beta.n_neighborhoods if self.ms._context_beta else 10,
            'context_beta_n_cells': self.ms._context_beta.n_cells if self.ms._context_beta else 8,
            'max_thres': self.ms.max_thres.item(),
            '_recent_scores': self._recent_scores,
            'cfg': self.cfg.__dict__,
            'count': self.ms.count,
        }

        # Serialize
        buf = io.BytesIO()
        torch.save(state, buf, pickle_module=pickle)
        data = buf.getvalue()

        # Write checkpoint
        with open(checkpoint_path, 'wb') as f:
            f.write(data)

        LOGGER.info(f"Checkpoint saved: {checkpoint_path}")

        # Generate HMAC signature
        if include_hmac:
            sig = hmac.new(
                self.signing_key.encode(),
                data,
                hashlib.sha256
            ).hexdigest()

            hmac_path = str(checkpoint_path) + self.HMAC_SUFFIX
            with open(hmac_path, 'w') as f:
                f.write(sig)

            LOGGER.info(f"HMAC signature saved: {hmac_path}")

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        path: str,
        verify_hmac: bool = True
    ) -> Dict:
        """
        Load checkpoint with optional HMAC verification.

        Args:
            path: Path to checkpoint file
            verify_hmac: Verify HMAC signature (default True)

        Returns:
            Dict with checkpoint metadata
        """
        checkpoint_path = Path(path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Verify HMAC if requested
        if verify_hmac:
            hmac_path = str(checkpoint_path) + self.HMAC_SUFFIX
            if not Path(hmac_path).exists():
                raise Exception(
                    f"HMAC file not found: {hmac_path}. "
                    f"Set verify_hmac=False to skip verification."
                )

            with open(hmac_path) as f:
                expected_hmac = f.read().strip()

            with open(checkpoint_path, 'rb') as f:
                actual_hmac = hmac.new(
                    self.signing_key.encode(),
                    f.read(),
                    hashlib.sha256
                ).hexdigest()

            if not hmac.compare_digest(expected_hmac, actual_hmac):
                raise Exception(
                    f"HMAC mismatch — possible tampering: {path}"
                )

            LOGGER.info("HMAC verification passed")

        # Load state
        state = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
            pickle_module=pickle
        )

        # Restore model state
        self.ms.ae.load_state_dict(state['ae_state'])
        self.ms.memory.memory = state['memory'].to(self.device)
        self.ms.memory.mem_usage = state['memory_mem_usage'].to(self.device)
        self.ms.memory.mem_ptr = state['memory_mem_ptr']
        self.ms.memory.count = state['memory_count']
        self.ms.mean = state['mean'].to(self.device)
        self.ms.std = state['std'].to(self.device)
        self.ms.max_thres = torch.tensor(
            state['max_thres'],
            dtype=torch.float32,
            device=self.device
        )

        # Restore ContextBeta
        if state.get('context_beta') is not None:
            n_nb = state.get('context_beta_n_neighborhoods', 10)
            n_cells = state.get('context_beta_n_cells', 8)
            self.ms._context_beta = ContextBeta(
                n_neighborhoods=n_nb,
                n_cells=n_cells
            )
            self.ms._context_beta.betas = state['context_beta']

        # Restore recent scores
        self._recent_scores = state.get('_recent_scores', [])

        # Restore config
        if 'cfg' in state:
            self.cfg.__dict__.update(state['cfg'])

        self._is_trained = True

        LOGGER.info(f"Checkpoint loaded: {path}")

        return {
            'status': 'loaded',
            'count': state.get('count', 0),
            'memory_count': state['memory_count'],
            'recent_scores_n': len(self._recent_scores),
        }

    def get_recent_scores(self) -> List[float]:
        """Get recent anomaly scores for monitoring."""
        return self._recent_scores


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_memstream(
    data_path: str,
    output_dir: str = 'models/memstream',
    epochs: int = 500,
    batch_size: int = 256,
    memory_size: int = 100000,
    seed: int = 42,
    signing_key: str = None,
    quick_retrain_mode: bool = False,
    checkpoint_path: str = None,
) -> Dict:
    """
    Main MemStream training function.

    Args:
        data_path: Path to clean baseline parquet file
        output_dir: Output directory for model
        epochs: Training epochs (default 500)
        batch_size: Batch size (default 256)
        memory_size: Memory module size (default 100000)
        seed: Random seed (default 42)
        signing_key: HMAC signing key
        quick_retrain_mode: Enable quick retrain mode
        checkpoint_path: Load from checkpoint for retraining

    Returns:
        Dict with training results
    """
    print("=" * 60)
    print("MemStream Training Pipeline (Phase 2B)")
    print("=" * 60)

    # Set determinism
    set_determinism(seed)
    print(f"Reproducibility seed: {seed}")

    # Initialize trainer
    cfg = MemStreamConfig()
    cfg.in_dim = 34
    cfg.hidden_dim = 68
    cfg.out_dim = 34
    cfg.memory_len = memory_size
    cfg.warmup_epochs = epochs
    cfg.warmup_batch_size = batch_size
    cfg.seed = seed

    trainer = MemStreamTrainer(
        cfg=cfg,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        signing_key=signing_key,
    )

    # Load checkpoint if specified
    if checkpoint_path:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path, verify_hmac=True)

    # Load data
    print(f"\nLoading data: {data_path}")
    t0 = time.time()

    df = pd.read_parquet(data_path)
    print(f"  Loaded {len(df):,} records in {time.time() - t0:.1f}s")

    # Preprocess
    print("\nPreprocessing...")
    t0 = time.time()

    # Add derived columns
    pickup = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    dropoff = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df['duration_s'] = (dropoff - pickup).dt.total_seconds()

    # Speed
    df['speed_mph'] = df['trip_distance'] / (df['duration_s'] / 3600)

    # Total amount
    df['total_amt'] = df.get('total_amount', df['fare_amount'])

    # Sort by time (CRITICAL for temporal splits)
    df = df.sort_values('tpep_pickup_datetime').reset_index(drop=True)
    print(f"  Preprocessed in {time.time() - t0:.1f}s")

    # Extract features
    print("\nExtracting 34D features...")
    t0 = time.time()
    X, hour_vals, dow_vals, ratecode_vals = extract_features(df)
    print(f"  Extracted {X.shape[0]:,} x {X.shape[1]}D features in {time.time() - t0:.1f}s")

    # Get labels (if available)
    if 'label' in df.columns:
        labels = df['label'].values.astype(np.int32)
    else:
        # Assume all normal if no labels
        labels = np.zeros(len(df), dtype=np.int32)

    # Get neighborhood IDs
    neighborhood_vals = np.array([
        location_to_neighborhood(loc)
        for loc in df['PULocationID'].values
    ], dtype=np.int32)

    # Training
    if quick_retrain_mode:
        print("\nQuick retrain mode...")
        results = trainer.quick_retrain(
            X, hour_vals, dow_vals,
            ratecode_vals, neighborhood_vals,
            epochs=min(epochs, 100),  # Cap at 100 for quick retrain
            batch_size=batch_size,
            verbose=True
        )
    else:
        print(f"\nFull training for {epochs} epochs...")
        results = trainer.train_full(
            X, labels, hour_vals, dow_vals,
            ratecode_vals, neighborhood_vals,
            epochs=epochs,
            batch_size=batch_size,
            verbose=True
        )

    # Save checkpoint
    print(f"\nSaving checkpoint to: {output_dir}")
    checkpoint_file = trainer.save_checkpoint(output_dir, include_hmac=True)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Status: {results['status']}")
    print(f"  Total samples: {results.get('n_samples', 'N/A'):,}")
    print(f"  Epochs: {results.get('epochs_trained', epochs)}")
    print(f"  Training time: {results.get('training_time_seconds', 0):.1f}s")
    print(f"  Checkpoint: {checkpoint_file}")
    print(f"  Device: {trainer.device}")
    print(f"  Memory size: {memory_size:,}")
    print("=" * 60)

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='MemStream Training Pipeline (Phase 2B)'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data/clean/jan_2024_clean_baseline.parquet',
        help='Path to clean baseline parquet file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/memstream',
        help='Output directory for model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        help='Training epochs (default 500)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Training batch size (default 256)'
    )
    parser.add_argument(
        '--memory-size',
        type=int,
        default=100000,
        help='Memory module size (default 100000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default 42)'
    )
    parser.add_argument(
        '--signing-key',
        type=str,
        default=None,
        help='HMAC signing key (default: MEMSTREAM_SIGNING_KEY env var)'
    )
    parser.add_argument(
        '--quick-retrain',
        action='store_true',
        help='Enable quick retrain mode (50-100 epochs)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Load from checkpoint for retraining'
    )

    args = parser.parse_args()

    # Check data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {args.data}")
        return 1

    try:
        train_memstream(
            data_path=str(data_path),
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            memory_size=args.memory_size,
            seed=args.seed,
            signing_key=args.signing_key,
            quick_retrain_mode=args.quick_retrain,
            checkpoint_path=args.checkpoint,
        )
        return 0

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
