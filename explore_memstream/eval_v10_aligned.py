#!/usr/bin/env python3
"""
GPU-accelerated eval pipeline aligned with v10 benchmark.

Key alignment points with C:\proj\ldt\results\v10\benchmark_v10.py:
  1. Data cleaning (quantile filtering, valid ranges)
  2. StandardScaler on ALL training data (not 10% stats)
  3. Autoencoder: train on ALL normalized data, 20 epochs
  4. Memory init: from LAST samples of training data
  5. ContextBeta: 5K warmup samples (not 50K)
  6. Streaming evaluation with memory updates after warmup_end
  7. 7 Canary Rules for pre-filtering
  8. 3 fraud types: Type1 (60%), Type2 (30%), Type3 (10%)
  9. Threshold at percentile 95 of scores (not F1-maximizing)
  10. Neighborhood-level ADWIN

Usage:
    python eval_v10_aligned.py --data C:/proj/ldt/data/nyc_taxi_300k.parquet --output results/v10_eval
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from memstream_src.core.memstream_core import (
    MemStreamCore, MemStreamConfig, set_determinism, get_context_id
)
from memstream_src.core.feature_extractor import FeatureVectorizer

# ---------------------------------------------------------------------------
# Constants (aligned with v10)
# ---------------------------------------------------------------------------
N_FEATURES = 34
N_NEIGHBORHOODS = 10
JFK_FLAT_FARE = 70.0

NEIGHBORHOOD_NAMES = [
    'manhattan', 'brooklyn', 'queens_lower', 'queens_upper',
    'bronx', 'staten_island', 'ewr', 'jfk', 'nalp', 'unknown'
]

# ---------------------------------------------------------------------------
# Data cleaning (aligned with v10 benchmark_v10.py lines 100-126)
# ---------------------------------------------------------------------------

def location_to_neighborhood(loc_id):
    if pd.isna(loc_id):
        return 9
    z = int(loc_id)
    if 1 <= z <= 43:   return 0
    elif 44 <= z <= 103: return 4
    elif 104 <= z <= 127: return 1
    elif 128 <= z <= 148: return 2
    elif 149 <= z <= 161: return 3
    elif 162 <= z <= 181: return 5
    elif 182 <= z <= 196: return 6
    elif 217 <= z <= 229: return 7
    elif 230 <= z <= 234: return 8
    else: return 9


def clean(df):
    """Apply same cleaning as v10 benchmark."""
    df = df.copy()
    for col in ['PULocationID', 'DOLocationID', 'fare_amount',
                'trip_distance', 'passenger_count', 'RatecodeID']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=[
        'PULocationID', 'DOLocationID', 'fare_amount',
        'trip_distance', 'passenger_count', 'RatecodeID'])

    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    df = df[(df['PULocationID'] >= 1) & (df['PULocationID'] <= 265)]
    df = df[(df['DOLocationID'] >= 1) & (df['DOLocationID'] <= 265)]
    df = df[(df['RatecodeID'] >= 1) & (df['RatecodeID'] <= 99)]

    df['fare_amount']    = df['fare_amount'].abs()
    df['trip_distance']  = df['trip_distance'].abs()

    pickup  = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    dropoff = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df['duration_s'] = (dropoff - pickup).dt.total_seconds()
    df = df[(df['duration_s'] > 0) & (df['duration_s'] < 86400)]

    df['speed_mph'] = df['trip_distance'] / (df['duration_s'] / 3600)
    df = df[(df['speed_mph'] > 0) & (df['speed_mph'] < 100)]

    for col in ['fare_amount', 'trip_distance', 'duration_s']:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    df['dur_min']  = df['duration_s'] / 60.0
    df['total_amt'] = df.get('total_amount', df['fare_amount'])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 34D Feature extraction (aligned with v10)
# ---------------------------------------------------------------------------

def zone_to_grid(zone_id):
    z = int(zone_id) if not pd.isna(zone_id) else 0
    if z <= 0:
        return 0, 0
    gx = (z - 1) % 16
    gy = (z - 1) // 16
    return gx, gy


def features(df):
    """34D feature vector matching v10 benchmark_v10.py."""
    n = len(df)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    pickup   = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    hour     = pickup.dt.hour.fillna(12).astype(np.int32).values
    dow      = pickup.dt.dayofweek.fillna(0).astype(np.int32).values
    dist     = df['trip_distance'].fillna(0).values.astype(np.float32)
    dur      = df['dur_min'].fillna(1).values.astype(np.float32)
    fare     = df['fare_amount'].fillna(0).values.astype(np.float32)
    pax      = df['passenger_count'].fillna(1).values.astype(np.float32)
    total    = df['total_amt'].fillna(0).values.astype(np.float32)
    spd      = df['speed_mph'].fillna(0).values.astype(np.float32)
    ratecode = df['RatecodeID'].fillna(1).values.astype(np.float32)
    pu_loc   = df['PULocationID'].fillna(0).values
    do_loc   = df['DOLocationID'].fillna(0).values
    eps      = np.float32(0.01)

    X[:, 0]  = dist
    X[:, 1]  = dur
    X[:, 2]  = fare
    X[:, 3]  = pax
    X[:, 4]  = total
    X[:, 5]  = spd
    X[:, 6]  = fare / np.maximum(dist, eps)
    X[:, 7]  = fare / np.maximum(dur, eps)
    X[:, 8]  = fare / np.maximum(pax, eps)
    X[:, 9]  = hour.astype(np.float32)
    X[:, 10] = dow.astype(np.float32)
    X[:, 11] = (dow >= 5).astype(np.float32)

    # Grid spatial (12-15)
    pu_gx = np.zeros(n, dtype=np.float32)
    pu_gy = np.zeros(n, dtype=np.float32)
    do_gx = np.zeros(n, dtype=np.float32)
    do_gy = np.zeros(n, dtype=np.float32)
    for i in range(n):
        pux, puy = zone_to_grid(pu_loc[i])
        dox, doy = zone_to_grid(do_loc[i])
        pu_gx[i], pu_gy[i] = float(pux), float(puy)
        do_gx[i], do_gy[i] = float(dox), float(doy)
    X[:, 12] = pu_gx
    X[:, 13] = pu_gy
    X[:, 14] = do_gx
    X[:, 15] = do_gy

    X[:, 16] = X[:, 6] / np.float32(2.5)
    X[:, 17] = X[:, 7] / np.float32(0.67)
    X[:, 18] = spd / np.float32(12.0)
    X[:, 19] = pax / np.maximum(dist, eps)
    X[:, 20] = np.sin(np.float32(2 * np.pi) * hour.astype(np.float32) / 24.0)
    X[:, 21] = np.cos(np.float32(2 * np.pi) * hour.astype(np.float32) / 24.0)
    X[:, 22] = np.sin(np.float32(2 * np.pi) * dow.astype(np.float32) / 7.0)
    X[:, 23] = np.cos(np.float32(2 * np.pi) * dow.astype(np.float32) / 7.0)
    X[:, 24] = dist * dist

    for i, rc in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        X[:, 25 + i] = (ratecode == rc).astype(np.float32)

    X[:, 30] = ((hour >= 20) | (hour <= 6)).astype(np.float32)
    X[:, 31] = np.log1p(fare)
    X[:, 32] = np.log1p(dist)
    X[:, 33] = np.abs(pu_gy - do_gy)

    return np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)


# ---------------------------------------------------------------------------
# 7 Canary Rules (aligned with v10)
# ---------------------------------------------------------------------------

def check_canary(df_row):
    fare   = float(df_row.get('fare_amount', 0))
    dist   = float(df_row.get('trip_distance', 0))
    dur    = float(df_row.get('dur_min', 1))
    pax    = float(df_row.get('passenger_count', 1))
    spd    = float(df_row.get('speed_mph', 0))
    total  = float(df_row.get('total_amt', 0))
    tip    = float(df_row.get('tip_amount', 0))
    ptype  = float(df_row.get('payment_type', 0))

    if fare <= 0:      return False
    if fare > 500:     return False
    if dur > 0 and (fare / dur) > 5.0: return False
    if spd > 80:       return False
    if dist == 0 and fare > 0: return False
    if pax < 1 or pax > 6: return False
    if ptype == 1 and tip == 0: return False
    return True


def is_canary_clean(df):
    """Vectorized canary check."""
    n = len(df)
    fare  = df['fare_amount'].fillna(0).values
    dist  = df['trip_distance'].fillna(0).values
    dur   = df['dur_min'].fillna(1).values
    pax   = df['passenger_count'].fillna(1).values
    spd   = df['speed_mph'].fillna(0).values
    tip   = df.get('tip_amount', pd.Series(np.zeros(n))).fillna(0).values
    ptype = df.get('payment_type', pd.Series(np.ones(n))).fillna(1).values

    clean = np.ones(n, dtype=bool)
    clean &= (fare > 0) & (fare <= 500)
    clean &= (dist > 0) | (fare == 0)
    fare_per_min = np.where(dur > 0, fare / np.maximum(dur, 0.01), 0)
    clean &= (fare_per_min <= 5.0) | (dur == 0)
    clean &= (spd > 0) & (spd <= 80)
    clean &= (pax >= 1) & (pax <= 6)
    credit_card = (ptype == 1)
    clean &= ~(credit_card & (tip == 0))
    return clean


# ---------------------------------------------------------------------------
# Fraud injection (aligned with v10)
# ---------------------------------------------------------------------------

def inject_realistic_fraud(df, rng, fraud_type='mixed', anomaly_rate=0.05):
    """
    v10-aligned fraud injection with 3 types:
      Type 1 (60%): Short-trip meter fraud — fare $40-80, dist<1mi
      Type 2 (30%): Duration manipulation — duration x8-15x
      Type 3 (10%): Ratecode mismatch — JFK flat fare + wrong spatial
    Only injects into canary-clean records.
    """
    df = df.copy()
    y  = np.zeros(len(df), dtype=np.int8)
    n_anom = int(len(df) * anomaly_rate)

    canary_clean = is_canary_clean(df)
    ratecode = df['RatecodeID'].fillna(1).values.astype(np.float32)
    dist_arr = df['trip_distance'].fillna(0).values.astype(np.float32)

    is_standard = (ratecode == 1.0)
    pool_clean   = np.where(is_standard & canary_clean)[0]

    type1_pool = np.where(is_standard & canary_clean & (dist_arr < 1.0))[0]
    type2_pool = np.where(is_standard & canary_clean & (dist_arr >= 2.0) & (dist_arr <= 4.0))[0]
    type3_pool = np.where(is_standard & canary_clean)[0]

    pool1 = type1_pool[:int(n_anom * 0.60)]
    pool2 = type2_pool[:int(n_anom * 0.30)]
    pool3 = type3_pool[:int(n_anom * 0.10)]
    pool  = np.concatenate([pool1, pool2, pool3])

    if len(pool) < n_anom:
        extra = rng.choice(pool, size=n_anom - len(pool), replace=True)
        pool  = np.concatenate([pool, extra])
    pool = pool[:n_anom]
    y[pool] = 1

    for idx in pool:
        if idx in type1_pool[:int(n_anom * 0.60)]:
            new_fare = float(rng.uniform(40.0, 80.0))
            df.at[df.index[idx], 'fare_amount'] = new_fare
            df.at[df.index[idx], 'total_amt']  = new_fare
        elif idx in type2_pool[:int(n_anom * 0.30)]:
            dur_mult = rng.uniform(8.0, 15.0)
            old_dur  = float(df.at[df.index[idx], 'dur_min'])
            old_dist = float(df.at[df.index[idx], 'trip_distance'])
            new_dur  = old_dur * dur_mult
            df.at[df.index[idx], 'dur_min']   = new_dur
            df.at[df.index[idx], 'speed_mph']  = old_dist / max(new_dur / 60.0, 0.01)
        else:
            df.at[df.index[idx], 'fare_amount'] = JFK_FLAT_FARE
            df.at[df.index[idx], 'total_amt']   = JFK_FLAT_FARE
            df.at[df.index[idx], 'RatecodeID']   = 2.0

    return df, y


# ---------------------------------------------------------------------------
# Evaluation (streaming, aligned with v10)
# ---------------------------------------------------------------------------

class StreamingEvaluator:
    """
    Streaming evaluation matching v10 benchmark_v10.py evaluate_streaming().
    Key: memory updates after warmup_end (not batch-only).
    """

    def __init__(self, cfg: MemStreamConfig, device='cuda'):
        self.cfg = cfg
        self.device = device
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X_train: np.ndarray, neighborhood_ids=None,
            hour_vals=None, dow_vals=None):
        """Fit StandardScaler and autoencoder on ALL training data."""
        X_float = X_train.astype(np.float64)
        X_scaled = self.scaler.fit_transform(X_float).astype(np.float32)

        # Build MemStream config
        cfg = MemStreamConfig()
        cfg.in_dim = N_FEATURES
        cfg.hidden_dim = 60
        cfg.out_dim = N_FEATURES
        cfg.latent_dim = 60
        cfg.warmup_epochs = 20      # v10: 20 epochs
        cfg.warmup_batch_size = 256
        cfg.warmup_lr = 1e-3
        cfg.warmup_noise_std = 0.1
        cfg.seed = self.cfg.seed
        cfg.memory_len = self.cfg.memory_len
        cfg.k = self.cfg.k
        cfg.gamma = self.cfg.gamma
        cfg.default_beta = 0.5
        for k, v in self.cfg.__dict__.items():
            if not k.startswith('_'):
                setattr(cfg, k, v)

        set_determinism(cfg.seed)
        self.model = MemStreamCore(cfg=cfg, device=self.device)

        # Train autoencoder on ALL data (v10 style)
        X_t = torch.from_numpy(X_scaled.astype(np.float32)).to(self.device)
        d = X_t.shape[1]
        torch.manual_seed(cfg.seed)

        W1 = torch.nn.Parameter(
            torch.randn(d, cfg.latent_dim, dtype=torch.float32, device=self.device) * np.sqrt(2.0 / d))
        b1 = torch.nn.Parameter(torch.zeros(cfg.latent_dim, dtype=torch.float32, device=self.device))
        W2 = torch.nn.Parameter(
            torch.randn(cfg.latent_dim, d, dtype=torch.float32, device=self.device) * np.sqrt(2.0 / cfg.latent_dim))
        b2 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32, device=self.device))
        params = [W1, b1, W2, b2]
        optimizer = torch.optim.Adam(params, lr=cfg.warmup_lr)

        for _ in range(cfg.warmup_epochs):
            X_noisy = X_t + torch.randn_like(X_t) * cfg.warmup_noise_std
            z     = torch.nn.functional.relu(X_noisy @ W1 + b1)
            x_rec = z @ W2 + b2
            loss  = torch.nn.functional.mse_loss(x_rec, X_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Build encoder/decoder
        self._W1 = W1.detach()
        self._b1 = b1.detach()
        self._W2 = W2.detach()
        self._b2 = b2.detach()
        self._latent_dim = cfg.latent_dim

        # Initialize memory from LAST samples of training
        Z = self._encode_batch(X_scaled)
        n_init = min(cfg.memory_len, len(Z))
        init_Z = Z[-n_init:]
        # Memory stores latent vectors; create memory with correct out_dim=latent_dim
        self.model.memory = MemoryModuleLite(
            memory_len=cfg.memory_len, out_dim=cfg.latent_dim, device=self.device)
        self.model.memory.memory[:n_init] = torch.from_numpy(init_Z).to(self.device)
        self.model.memory.mem_usage[:n_init] = 1.0
        self.model.memory.count = n_init

        # Fit ContextBeta from 5K warmup samples (v10: min(5000, len))
        warmup_n = min(5000, len(X_scaled))
        warmup_X = X_scaled[:warmup_n]
        warmup_scores = self._score_batch_raw(warmup_X)

        if hour_vals is None:
            hour_vals = X_train[:, 9].astype(int)[:warmup_n]
        if dow_vals is None:
            dow_vals = X_train[:, 10].astype(int)[:warmup_n]
        if neighborhood_ids is None:
            neighborhood_ids = np.zeros(warmup_n, dtype=int)

        ratecode_vals = X_train[:, 25].astype(int)[:warmup_n]
        ctx_ids = np.array([
            get_context_id(int(h), int(d), int(r))
            for h, d, r in zip(hour_vals, dow_vals, ratecode_vals)
        ])

        self.model._context_beta = ContextBetaLite(
            n_neighborhoods=N_NEIGHBORHOODS, n_cells=8, percentile=95)
        self.model._context_beta.fit_from_scores(
            warmup_scores, neighborhood_ids[:warmup_n], ctx_ids)
        self.model.mean = torch.from_numpy(self.scaler.mean_).float().to(self.device)
        self.model.std  = torch.from_numpy(self.scaler.scale_).float().to(self.device)
        self.model.max_thres = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.model.eval_mode = True

    def _encode_batch(self, X_scaled: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X_scaled.astype(np.float32)).to(self.device)
        with torch.no_grad():
            z = torch.nn.functional.relu(X_t @ self._W1.to(self.device) + self._b1.to(self.device))
        return z.cpu().numpy()

    def _score_batch_raw(self, X_scaled: np.ndarray) -> np.ndarray:
        """Score using GPU batch kNN."""
        if self.model.memory.count < 2:
            return np.full(len(X_scaled), 0.5)

        X_t = torch.from_numpy(X_scaled.astype(np.float32)).to(self.device)
        Z   = torch.nn.functional.relu(X_t @ self._W1.to(self.device) + self._b1.to(self.device))

        M   = self.model.memory.count
        mem = self.model.memory.memory[:M]

        if len(X_t) * M * N_FEATURES * 4 > 500_000_000:
            return self._score_chunked(Z, mem)

        diff  = Z.unsqueeze(1) - mem.unsqueeze(0)
        dists = diff.abs().sum(dim=2)
        k_use = min(self.cfg.k, M)
        top_k = dists.topk(k_use, dim=1, largest=False, sorted=True)
        if self.cfg.gamma > 0:
            powers = torch.arange(k_use, device=self.device, dtype=torch.float32)
            weights = self.cfg.gamma ** powers
            scores  = (top_k.values * weights).sum(dim=1)
        else:
            scores  = top_k.values.sum(dim=1)
        return scores.cpu().numpy()

    def _score_chunked(self, Z, mem):
        M = mem.shape[0]
        k_use = min(self.cfg.k, M)
        scores_list = []
        for start in range(0, len(Z), 500):
            chunk = Z[start:start+500]
            diff  = chunk.unsqueeze(1) - mem.unsqueeze(0)
            dists = diff.abs().sum(dim=2)
            top_k = dists.topk(k_use, dim=1, largest=False, sorted=True)
            scores_list.append(top_k.values.sum(dim=1))
        return torch.cat(scores_list, dim=0).cpu().numpy()

    def decision_function(self, X_test: np.ndarray, neighborhood_ids=None,
                          hour_vals=None, dow_vals=None, ratecode_vals=None,
                          update_memory=True):
        """Streaming evaluation: score + optional memory update."""
        X_scaled = self.scaler.transform(X_test.astype(np.float64)).astype(np.float32)

        n = len(X_scaled)
        scores = np.zeros(n, dtype=np.float64)
        warmup_end = 500

        for i in range(n):
            nb = int(neighborhood_ids[i]) if neighborhood_ids is not None else 0
            hr = int(hour_vals[i]) if hour_vals is not None else 12
            dw = int(dow_vals[i]) if dow_vals is not None else 0
            rc = float(ratecode_vals[i]) if ratecode_vals is not None else 1.0

            # Score this sample
            x_t = torch.from_numpy(X_scaled[i:i+1].astype(np.float32)).to(self.device)
            with torch.no_grad():
                z = torch.nn.functional.relu(x_t @ self._W1.to(self.device) + self._b1.to(self.device))

            M   = self.model.memory.count
            mem = self.model.memory.memory[:M]
            if M >= 2:
                diff  = z - mem.unsqueeze(0)
                dists = diff.abs().sum(dim=2)
                k_use = min(self.cfg.k, M)
                top_k = dists.topk(k_use, dim=1, largest=False, sorted=True)
                if self.cfg.gamma > 0:
                    powers = torch.arange(k_use, device=self.device, dtype=torch.float32)
                    weights = self.cfg.gamma ** powers
                    raw = (top_k.values * weights).sum(dim=1)
                else:
                    raw = top_k.values.sum(dim=1)
                raw_score = float(raw[0].cpu())
            else:
                raw_score = 0.5

            # Context-beta normalization
            ctx_id = get_context_id(hr, dw, rc)
            beta   = self.model._context_beta.get_beta(nb, ctx_id)
            scores[i] = raw_score / max(beta, 1e-6)

            # Memory update after warmup (streaming semantics)
            if update_memory and i >= warmup_end:
                z_detached = z[0].detach()
                if not self.model.memory._is_full:
                    self.model.memory.memory[self.model.memory.mem_ptr] = z_detached
                    self.model.memory.mem_usage[self.model.memory.mem_ptr] = 1.0
                    self.model.memory.mem_ptr = (self.model.memory.mem_ptr + 1) % self.cfg.memory_len
                    self.model.memory.count += 1
                    if self.model.memory.count >= self.cfg.memory_len:
                        self.model.memory._is_full = True
                else:
                    self.model.memory.memory[self.model.memory.mem_ptr] = z_detached
                    self.model.memory.mem_ptr = (self.model.memory.mem_ptr + 1) % self.cfg.memory_len

        return scores

    def decision_function_batch(self, X_test: np.ndarray, neighborhood_ids=None,
                                 hour_vals=None, dow_vals=None, ratecode_vals=None):
        """Batch scoring (for fast comparison)."""
        X_scaled = self.scaler.transform(X_test.astype(np.float64)).astype(np.float32)
        scores = self._score_batch_raw(X_scaled)

        n = len(scores)
        nb_ids  = np.zeros(n, dtype=int) if neighborhood_ids is None else neighborhood_ids.astype(int)
        hr_vals = np.zeros(n, dtype=int) if hour_vals is None else hour_vals.astype(int)
        dw_vals = np.zeros(n, dtype=int) if dow_vals is None else dow_vals.astype(int)
        rc_vals = np.ones(n, dtype=float) if ratecode_vals is None else ratecode_vals.astype(float)

        for i in range(n):
            ctx_id = get_context_id(int(hr_vals[i]), int(dw_vals[i]), float(rc_vals[i]))
            beta   = self.model._context_beta.get_beta(int(nb_ids[i]), ctx_id)
            scores[i] /= max(beta, 1e-6)

        return scores


class MemoryModuleLite:
    def __init__(self, memory_len, out_dim, device):
        self.memory = torch.zeros(memory_len, out_dim, device=device)
        self.mem_usage = torch.zeros(memory_len, device=device)
        self.mem_ptr = 0
        self.count = 0
        self._is_full = False

    @property
    def n_records(self):
        return self.count


class ContextBetaLite:
    """80 context-beta thresholds (aligned with v10)."""
    def __init__(self, n_neighborhoods=10, n_cells=8, percentile=95):
        self.n_neighborhoods = n_neighborhoods
        self.n_cells = n_cells
        self.percentile = percentile
        self.betas = np.ones((n_neighborhoods, n_cells), dtype=np.float32) * 0.5

    def fit_from_scores(self, scores, neighborhood_ids, context_ids):
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


# ---------------------------------------------------------------------------
# Metrics (aligned with v10)
# ---------------------------------------------------------------------------

def compute_metrics(y_true, scores, threshold=None):
    if len(np.unique(y_true)) < 2:
        return {k: np.nan for k in ['AUC_PR', 'AUC_ROC', 'F1', 'Precision', 'Recall', 'FPR', 'TP', 'FP', 'TN', 'FN']}
    try:
        auc_roc = roc_auc_score(y_true, scores)
    except ValueError:
        auc_roc = np.nan
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, scores)
    auc_pr = auc(rec_curve, prec_curve) if len(rec_curve) > 1 else np.nan
    if threshold is None:
        threshold = np.percentile(scores, 95)
    y_pred = (scores > threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        'AUC_PR': auc_pr, 'AUC_ROC': auc_roc, 'F1': f1,
        'Precision': precision, 'Recall': recall, 'FPR': fpr,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'threshold_used': float(threshold),
    }


# ---------------------------------------------------------------------------
# Configs (aligned with v10)
# ---------------------------------------------------------------------------

EVAL_CONFIGS = [
    {'name': 'v10_mem256_k10', 'memory_len': 256, 'k': 10, 'gamma': 0.0},
    {'name': 'v10_mem512_k10', 'memory_len': 512, 'k': 10, 'gamma': 0.0},
    {'name': 'v10_mem1024_k10', 'memory_len': 1024, 'k': 10, 'gamma': 0.0},
    {'name': 'v10_mem256_k5', 'memory_len': 256, 'k': 5, 'gamma': 0.0},
    {'name': 'v10_mem512_k10_g05', 'memory_len': 512, 'k': 10, 'gamma': 0.5},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='v10-Aligned GPU Evaluation')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/v10_eval')
    parser.add_argument('--fraud-type', type=str, default='mixed')
    parser.add_argument('--anomaly-rate', type=float, default=0.05)
    parser.add_argument('--warmup-samples', type=int, default=10000)
    parser.add_argument('--test-samples', type=int, default=15000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval-mode', type=str, default='streaming',
                       choices=['streaming', 'batch'])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')

    # 1. Load and clean data
    print("=" * 62)
    print("  v10-Aligned GPU Evaluation")
    print("=" * 62)
    print(f"  Data: {args.data}")
    print(f"  Fraud: {args.fraud_type}, rate={args.anomaly_rate}")
    print(f"  Device: {args.device}")
    print(f"  Eval mode: {args.eval_mode}")
    print("=" * 62)

    print("\n[1] Loading raw data...")
    df_raw = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data)
    print(f"    Raw: {len(df_raw):,} records")

    print("\n[2] Cleaning data (v10 filter)...")
    df_clean = clean(df_raw)
    print(f"    After clean: {len(df_clean):,} records ({len(df_clean)/len(df_raw)*100:.1f}%)")

    # 2. Split train / test
    n_train = args.warmup_samples
    n_test  = args.test_samples
    n_total = min(n_train + n_test, len(df_clean))
    df_clean = df_clean.iloc[:n_total].reset_index(drop=True)

    df_train = df_clean.iloc[:n_train].reset_index(drop=True)
    df_test  = df_clean.iloc[n_train:n_train + n_test].reset_index(drop=True)
    print(f"    Train: {len(df_train):,} | Test: {len(df_test):,}")

    # 3. Inject fraud
    print(f"\n[3] Injecting fraud ({args.fraud_type}, rate={args.anomaly_rate})...")
    rng = np.random.RandomState(args.seed)
    df_test_inj, y_test = inject_realistic_fraud(
        df_test, rng, fraud_type=args.fraud_type, anomaly_rate=args.anomaly_rate)
    n_anom = int(y_test.sum())
    print(f"    Test: {len(df_test_inj):,}, anomalies: {n_anom:,} ({n_anom/len(df_test)*100:.2f}%)")

    # 4. Extract features
    print("\n[4] Extracting features...")
    X_train = features(df_train)
    X_test  = features(df_test_inj)
    print(f"    X_train: {X_train.shape}, X_test: {X_test.shape}")

    # 5. Extract context info
    pickup_test = pd.to_datetime(df_test_inj['tpep_pickup_datetime'], errors='coerce')
    hour_test   = pickup_test.dt.hour.fillna(12).astype(int).values
    dow_test    = pickup_test.dt.dayofweek.fillna(0).astype(int).values
    ratecode_test = df_test_inj['RatecodeID'].fillna(1).astype(float).values
    nb_test = np.array([
        location_to_neighborhood(loc) for loc in df_test_inj['PULocationID'].fillna(1).values
    ], dtype=int)

    # 6. Run configs
    print(f"\n[5] Running {len(EVAL_CONFIGS)} configs...")
    results = []
    for cfg_overrides in EVAL_CONFIGS:
        name = cfg_overrides['name']
        print(f"  [{name}]...", end='', flush=True)
        t0 = time.time()

        cfg = MemStreamConfig()
        cfg.memory_len = cfg_overrides['memory_len']
        cfg.k          = cfg_overrides['k']
        cfg.gamma      = cfg_overrides['gamma']
        cfg.seed       = args.seed

        try:
            ev = StreamingEvaluator(cfg=cfg, device=args.device)
            ev.fit(X_train, neighborhood_ids=None,
                   hour_vals=X_train[:, 9].astype(int),
                   dow_vals=X_train[:, 10].astype(int))

            if args.eval_mode == 'streaming':
                scores = ev.decision_function(
                    X_test, neighborhood_ids=nb_test,
                    hour_vals=hour_test, dow_vals=dow_test,
                    ratecode_vals=ratecode_test, update_memory=True)
            else:
                scores = ev.decision_function_batch(
                    X_test, neighborhood_ids=nb_test,
                    hour_vals=hour_test, dow_vals=dow_test,
                    ratecode_vals=ratecode_test)

            m = compute_metrics(y_test, scores)
            m['name'] = name
            m['config'] = cfg_overrides
            m['elapsed_s'] = time.time() - t0
            results.append(m)
            print(f" F1={m['F1']:.4f} AUC-PR={m['AUC_PR']:.4f} "
                  f"AUC-ROC={m['AUC_ROC']:.4f} ({m['elapsed_s']:.1f}s)")
        except Exception as e:
            print(f" ERROR: {e}")
            traceback.print_exc()
            results.append({'name': name, 'config': cfg_overrides, 'error': str(e)})

    # 7. Save results
    out_file = output_dir / f'v10_eval_{ts}.json'
    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': ts,
            'fraud_type': args.fraud_type,
            'anomaly_rate': args.anomaly_rate,
            'n_train': len(df_train),
            'n_test': len(df_test_inj),
            'n_anomalies': int(y_test.sum()),
            'eval_mode': args.eval_mode,
            'device': args.device,
            'results': results,
        }, f, indent=2)
    print(f"\n  Results saved to {out_file}")

    # 8. Print summary
    print("\n" + "=" * 80)
    print(f"  SUMMARY (sorted by F1)")
    print("=" * 80)
    valid = [r for r in results if 'error' not in r]
    valid.sort(key=lambda x: x['F1'], reverse=True)
    print(f"\n{'Name':<22} {'F1':>6} {'Prec':>6} {'Rec':>6} {'FPR':>7} {'AUC-PR':>8} {'AUC-ROC':>9} {'Time':>6}")
    print("-" * 78)
    for r in valid:
        print(f"{r['name']:<22} {r['F1']:6.4f} {r['Precision']:6.4f} {r['Recall']:6.4f} "
              f"{r['FPR']:7.4f} {r['AUC_PR']:8.4f} {r['AUC_ROC']:9.4f} {r.get('elapsed_s',0):5.1f}s")

    # 9. Score distribution plot
    print("\n[6] Generating score distribution plot...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('MemStream v10-Aligned: Detection Score Analysis', fontsize=13, fontweight='bold')

    for i, r in enumerate(valid[:3]):
        ax = axes[i]
        cfg = r['config']
        score_file = output_dir / f"scores_{cfg['name']}_{ts}.npy"
        if score_file.exists():
            s = np.load(str(score_file))
        elif 'scores' in r:
            s = np.array(r['scores'])
        else:
            ax.text(0.5, 0.5, 'No scores available', ha='center', va='center')
            ax.set_title(cfg['name'])
            continue

        norm_s = s[y_test == 0]
        anom_s = s[y_test == 1]
        bins = np.linspace(min(s.min(), norm_s.min()),
                          np.percentile(s, 99.5), 60)
        ax.hist(norm_s, bins=bins, alpha=0.6, label=f'Normal ({len(norm_s):,})',
                color='#4A90D9', density=True)
        ax.hist(anom_s, bins=bins, alpha=0.6, label=f'Anomaly ({len(anom_s):,})',
                color='#E24A33', density=True)
        ax.axvline(r.get('threshold_used', np.percentile(s, 95)),
                   color='black', linestyle='--', lw=2,
                   label=f'Thresh={r.get("threshold_used", 0):.2f}')
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title(f"{r['name']}\nF1={r['F1']:.4f} AUC-PR={r['AUC_PR']:.4f}")
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / f'score_dist_{ts}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved score distribution plot")

    print(f"\nAll outputs: {output_dir}")


if __name__ == '__main__':
    main()
