#!/usr/bin/env python3
"""
Comprehensive Grid Search for MemStream on NYC Taxi Anomaly Detection.
Finds the best configuration by testing a wide range of strategies.

Key axes explored:
  1. Model: memory_len, k, gamma, latent_dim, beta_percentile
  2. Data: single-month (seq) vs multi-month (temporal split), warmup size
  3. Eval: streaming vs batch
  4. Fraud: mixed, type1, type2, type3
  5. Threshold: percentile-based vs F1-maximizing
"""

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from itertools import product
from collections import defaultdict

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================
# Constants & Data Setup
# ============================
N_FEATURES    = 34
N_NEIGHBORHOODS = 10
JFK_FLAT_FARE = 70.0
DEVICE = 'cuda'

OUT_DIR = Path('C:/proj/ldt/explore_memstream/results/grid_search')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# Helper functions (aligned with v10)
# ============================

def location_to_neighborhood(loc_id):
    if pd.isna(loc_id): return 9
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

def get_context_id(hour, dow, ratecode):
    is_special = 1 if ratecode > 1 else 0
    is_night   = 1 if (hour >= 18 or hour < 6) else 0
    is_weekend = 1 if dow >= 5 else 0
    return (is_special << 2) | (is_night << 1) | is_weekend

def zone_to_grid(zone_id):
    z = int(zone_id) if not pd.isna(zone_id) else 0
    if z <= 0: return 0, 0
    gx = (z - 1) % 16
    gy = (z - 1) // 16
    return gx, gy

def clean(df):
    df = df.copy()
    for col in ['PULocationID', 'DOLocationID', 'fare_amount',
                'trip_distance', 'passenger_count', 'RatecodeID']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['PULocationID', 'DOLocationID', 'fare_amount',
                            'trip_distance', 'passenger_count', 'RatecodeID'])
    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    df = df[(df['PULocationID'] >= 1) & (df['PULocationID'] <= 265)]
    df = df[(df['DOLocationID'] >= 1) & (df['DOLocationID'] <= 265)]
    df = df[(df['RatecodeID'] >= 1) & (df['RatecodeID'] <= 99)]
    df['fare_amount']   = df['fare_amount'].abs()
    df['trip_distance'] = df['trip_distance'].abs()
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

def features(df):
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

    pu_gx = np.zeros(n, dtype=np.float32); pu_gy = np.zeros(n, dtype=np.float32)
    do_gx = np.zeros(n, dtype=np.float32); do_gy = np.zeros(n, dtype=np.float32)
    for i in range(n):
        pux, puy = zone_to_grid(pu_loc[i])
        dox, doy = zone_to_grid(do_loc[i])
        pu_gx[i], pu_gy[i] = float(pux), float(puy)
        do_gx[i], do_gy[i] = float(dox), float(doy)
    X[:, 12] = pu_gx; X[:, 13] = pu_gy; X[:, 14] = do_gx; X[:, 15] = do_gy

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

def is_canary_clean(df):
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
    clean &= ~((ptype == 1) & (tip == 0))
    return clean

def inject_realistic_fraud(df, rng, fraud_type='mixed', anomaly_rate=0.05):
    df = df.copy()
    y  = np.zeros(len(df), dtype=np.int8)
    n_anom = int(len(df) * anomaly_rate)

    canary_clean = is_canary_clean(df)
    ratecode = df['RatecodeID'].fillna(1).values.astype(np.float32)
    dist_arr = df['trip_distance'].fillna(0).values.astype(np.float32)

    is_standard = (ratecode == 1.0)
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
            df.at[df.index[idx], 'total_amt']   = new_fare
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

# ============================
# Memory Module (GPU, minimal)
# ============================

class MemoryModule:
    def __init__(self, memory_len, out_dim, device):
        self.memory   = torch.zeros(memory_len, out_dim, device=device)
        self.mem_usage = torch.zeros(memory_len, device=device)
        self.mem_ptr  = 0
        self.count    = 0
        self._is_full = False
    @property
    def n_records(self): return self.count

class ContextBeta:
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

# ============================
# GPU-accelerated MemStream Evaluator
# ============================

class GPUEvaluator:
    """GPU-accelerated MemStream with full config flexibility."""

    def __init__(self, memory_len, k, gamma, latent_dim, beta_percentile,
                 warmup_samples, seed=42, device='cuda'):
        self.memory_len    = memory_len
        self.k             = k
        self.gamma         = gamma
        self.latent_dim    = latent_dim
        self.beta_percentile = beta_percentile
        self.warmup_samples = warmup_samples
        self.seed          = seed
        self.device        = device
        self.scaler        = None
        self._W1 = self._b1 = self._W2 = self._b2 = None

    def fit(self, X_train, neighborhood_ids=None, hour_vals=None, dow_vals=None):
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_train.astype(np.float64)).astype(np.float32)

        torch.manual_seed(self.seed)
        d = Xs.shape[1]
        W1 = torch.nn.Parameter(
            torch.randn(d, self.latent_dim, dtype=torch.float32, device=self.device) * np.sqrt(2.0 / d))
        b1 = torch.nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32, device=self.device))
        W2 = torch.nn.Parameter(
            torch.randn(self.latent_dim, d, dtype=torch.float32, device=self.device) * np.sqrt(2.0 / self.latent_dim))
        b2 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float32, device=self.device))
        params = [W1, b1, W2, b2]
        optimizer = torch.optim.Adam(params, lr=1e-3)

        Xt = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
        for _ in range(20):
            noise   = Xt + torch.randn_like(Xt) * 0.1
            z       = torch.nn.functional.relu(noise @ W1 + b1)
            x_rec   = z @ W2 + b2
            loss    = torch.nn.functional.mse_loss(x_rec, Xt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        self._W1 = W1.detach()
        self._b1 = b1.detach()
        self._W2 = W2.detach()
        self._b2 = b2.detach()

        # Encode training data
        with torch.no_grad():
            Z = torch.nn.functional.relu(Xt @ self._W1 + self._b1).cpu().numpy()

        # Init memory from LAST samples
        n_init = min(self.memory_len, len(Z))
        self.memory = MemoryModule(self.memory_len, self.latent_dim, self.device)
        init_Z = torch.from_numpy(Z[-n_init:].astype(np.float32)).to(self.device)
        self.memory.memory[:n_init] = init_Z
        self.memory.mem_usage[:n_init] = 1.0
        self.memory.count = n_init
        if n_init >= self.memory_len:
            self.memory._is_full = True

        # Fit ContextBeta from warmup
        warmup_n = min(self.warmup_samples, len(Xs))
        warmup_X = torch.from_numpy(Xs[:warmup_n].astype(np.float32)).to(self.device)
        with torch.no_grad():
            Z_w = torch.nn.functional.relu(warmup_X @ self._W1 + self._b1)
        warmup_scores = self._score_batch_raw(Z_w)

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

        self._context_beta = ContextBeta(percentile=self.beta_percentile)
        self._context_beta.fit_from_scores(
            warmup_scores, neighborhood_ids[:warmup_n], ctx_ids)

        self.mean = torch.from_numpy(self.scaler.mean_).float().to(self.device)
        self.std  = torch.from_numpy(self.scaler.scale_).float().to(self.device)

    def _encode(self, X_t):
        return torch.nn.functional.relu(X_t @ self._W1.to(self.device) + self._b1.to(self.device))

    def _score_batch_raw(self, Z):
        M = self.memory.count
        if M < 2:
            return np.full(len(Z), 0.5)
        mem = self.memory.memory[:M]
        diff  = Z.unsqueeze(1) - mem.unsqueeze(0)
        dists = diff.abs().sum(dim=2)
        k_use = min(self.k, M)
        top_k = dists.topk(k_use, dim=1, largest=False, sorted=True)
        if self.gamma > 0:
            powers  = torch.arange(k_use, device=self.device, dtype=torch.float32)
            weights = self.gamma ** powers
            scores  = (top_k.values * weights).sum(dim=1)
        else:
            scores = top_k.values.sum(dim=1)
        return scores.cpu().numpy()

    def score_streaming(self, X_test, neighborhood_ids=None, hour_vals=None,
                       dow_vals=None, ratecode_vals=None, update_memory=True):
        Xs = self.scaler.transform(X_test.astype(np.float64)).astype(np.float32)
        n  = len(Xs)
        scores = np.zeros(n, dtype=np.float64)
        warmup_end = 500

        nb_ids  = np.zeros(n, dtype=int) if neighborhood_ids is None else neighborhood_ids.astype(int)
        hr_vals = np.zeros(n, dtype=int) if hour_vals is None else hour_vals.astype(int)
        dw_vals = np.zeros(n, dtype=int) if dow_vals is None else dow_vals.astype(int)
        rc_vals = np.ones(n, dtype=float) if ratecode_vals is None else ratecode_vals.astype(float)

        for i in range(n):
            x_t = torch.from_numpy(Xs[i:i+1].astype(np.float32)).to(self.device)
            with torch.no_grad():
                z = self._encode(x_t)

            M = self.memory.count
            mem = self.memory.memory[:M]
            if M >= 2:
                diff  = z - mem.unsqueeze(0)
                dists = diff.abs().sum(dim=2)
                k_use = min(self.k, M)
                top_k = dists.topk(k_use, dim=1, largest=False, sorted=True)
                if self.gamma > 0:
                    powers  = torch.arange(k_use, device=self.device, dtype=torch.float32)
                    weights = self.gamma ** powers
                    raw     = (top_k.values * weights).sum(dim=1)
                else:
                    raw = top_k.values.sum(dim=1)
                raw_score = float(raw[0].cpu())
            else:
                raw_score = 0.5

            ctx_id = get_context_id(int(hr_vals[i]), int(dw_vals[i]), float(rc_vals[i]))
            beta   = self._context_beta.get_beta(int(nb_ids[i]), ctx_id)
            scores[i] = raw_score / max(beta, 1e-6)

            if update_memory and i >= warmup_end:
                z_det = z[0].detach()
                if not self.memory._is_full:
                    self.memory.memory[self.memory.mem_ptr] = z_det
                    self.memory.mem_ptr = (self.memory.mem_ptr + 1) % self.memory_len
                    self.memory.count += 1
                    if self.memory.count >= self.memory_len:
                        self.memory._is_full = True
                else:
                    self.memory.memory[self.memory.mem_ptr] = z_det
                    self.memory.mem_ptr = (self.memory.mem_ptr + 1) % self.memory_len

        return scores

    def score_batch(self, X_test, neighborhood_ids=None, hour_vals=None,
                    dow_vals=None, ratecode_vals=None):
        Xs = self.scaler.transform(X_test.astype(np.float64)).astype(np.float32)
        X_t = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
        with torch.no_grad():
            Z = self._encode(X_t)
        scores = self._score_batch_raw(Z)

        n  = len(scores)
        nb_ids  = np.zeros(n, dtype=int) if neighborhood_ids is None else neighborhood_ids.astype(int)
        hr_vals = np.zeros(n, dtype=int) if hour_vals is None else hour_vals.astype(int)
        dw_vals = np.zeros(n, dtype=int) if dow_vals is None else dow_vals.astype(int)
        rc_vals = np.ones(n, dtype=float) if ratecode_vals is None else ratecode_vals.astype(float)

        for i in range(n):
            ctx_id = get_context_id(int(hr_vals[i]), int(dw_vals[i]), float(rc_vals[i]))
            beta   = self._context_beta.get_beta(int(nb_ids[i]), ctx_id)
            scores[i] /= max(beta, 1e-6)
        return scores

# ============================
# Metrics
# ============================

def compute_metrics(y_true, scores, threshold=None, return_best=False):
    if len(np.unique(y_true)) < 2:
        return {k: np.nan for k in ['AUC_PR', 'AUC_ROC', 'F1', 'Precision', 'Recall', 'FPR', 'TP', 'FP', 'TN', 'FN']}
    try:
        auc_roc = roc_auc_score(y_true, scores)
    except:
        auc_roc = np.nan
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, scores)
    auc_pr = auc(rec_curve, prec_curve) if len(rec_curve) > 1 else np.nan

    if return_best:
        best_f1, best_thresh = 0, 0
        for t in np.linspace(scores.min(), scores.max(), 500):
            yp = (scores > t).astype(int)
            tp = ((yp == 1) & (y_true == 1)).sum()
            fp = ((yp == 1) & (y_true == 0)).sum()
            fn = ((yp == 0) & (y_true == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        yp = (scores > best_thresh).astype(int)
        tp = int(((yp == 1) & (y_true == 1)).sum())
        fp = int(((yp == 1) & (y_true == 0)).sum())
        tn = int(((yp == 0) & (y_true == 0)).sum())
        fn = int(((yp == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0
        return {'AUC_PR': auc_pr, 'AUC_ROC': auc_roc, 'F1': best_f1,
                'Precision': precision, 'Recall': recall, 'FPR': fpr,
                'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
                'threshold_used': float(best_thresh), 'threshold_type': 'best_F1'}
    else:
        if threshold is None:
            threshold = np.percentile(scores, 95)
        yp = (scores > threshold).astype(int)
        tp = int(((yp == 1) & (y_true == 1)).sum())
        fp = int(((yp == 1) & (y_true == 0)).sum())
        tn = int(((yp == 0) & (y_true == 0)).sum())
        fn = int(((yp == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return {'AUC_PR': auc_pr, 'AUC_ROC': auc_roc, 'F1': f1,
                'Precision': precision, 'Recall': recall, 'FPR': fpr,
                'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
                'threshold_used': float(threshold), 'threshold_type': 'percentile_95'}

# ============================
# Data Loading
# ============================

def load_data(data_mode, seed=42):
    """
    data_mode options:
      'single_seq': train+test from month 1, sequential split
      'multi_month': train on months 1-5, test on month 6 (v10 approach)
    """
    if data_mode == 'single_seq':
        df_raw = pd.read_parquet('C:/proj/ldt/data/nyc_taxi_300k.parquet')
        df_clean = clean(df_raw)
        n_train, n_test = 10000, 15000
        df_clean = df_clean.iloc[:n_train + n_test].reset_index(drop=True)
        df_train = df_clean.iloc[:n_train].reset_index(drop=True)
        df_test  = df_clean.iloc[n_train:n_train + n_test].reset_index(drop=True)
        fraud_type = 'mixed'
        anomaly_rate = 0.05
        rng = np.random.RandomState(seed)
        df_test_inj, y_test = inject_realistic_fraud(df_test, rng, fraud_type=fraud_type, anomaly_rate=anomaly_rate)
        return df_train, df_test_inj, y_test

    elif data_mode == 'multi_month':
        dfs = []
        for m in range(1, 6):
            df = pd.read_parquet(f'C:/proj/ldt/data/raw/yellow_tripdata_2024-{m:02d}.parquet')
            dfs.append(clean(df))
        df_train_raw = pd.concat(dfs, ignore_index=True)
        df_train_raw = df_train_raw.sample(n=min(10000, len(df_train_raw)), random_state=seed)

        df_test_raw = clean(pd.read_parquet('C:/proj/ldt/data/raw/yellow_tripdata_2024-06.parquet'))
        df_test_raw = df_test_raw.iloc[:15000].reset_index(drop=True)
        df_train_raw = df_train_raw.reset_index(drop=True)

        fraud_type = 'mixed'
        anomaly_rate = 0.05
        rng = np.random.RandomState(seed)
        df_test_inj, y_test = inject_realistic_fraud(df_test_raw, rng, fraud_type=fraud_type, anomaly_rate=anomaly_rate)
        return df_train_raw, df_test_inj, y_test

    elif data_mode == 'single_jan':
        df_raw = pd.read_parquet('C:/proj/ldt/data/raw/yellow_tripdata_2024-01.parquet')
        df_clean = clean(df_raw)
        n_train, n_test = 10000, 15000
        df_clean = df_clean.iloc[:n_train + n_test].reset_index(drop=True)
        df_train = df_clean.iloc[:n_train].reset_index(drop=True)
        df_test  = df_clean.iloc[n_train:n_train + n_test].reset_index(drop=True)
        rng = np.random.RandomState(seed)
        df_test_inj, y_test = inject_realistic_fraud(df_test, rng, fraud_type='mixed', anomaly_rate=0.05)
        return df_train, df_test_inj, y_test

    elif data_mode == 'single_monthly_big':
        # Use first 2 months for bigger dataset
        dfs = []
        for m in range(1, 3):
            df = pd.read_parquet(f'C:/proj/ldt/data/raw/yellow_tripdata_2024-{m:02d}.parquet')
            dfs.append(clean(df))
        df_big = pd.concat(dfs, ignore_index=True)
        n_total = min(50000, len(df_big))
        df_big = df_big.iloc[:n_total].reset_index(drop=True)
        n_train = 15000
        n_test  = 15000
        df_train = df_big.iloc[:n_train].reset_index(drop=True)
        df_test  = df_big.iloc[n_train:n_train + n_test].reset_index(drop=True)
        rng = np.random.RandomState(seed)
        df_test_inj, y_test = inject_realistic_fraud(df_test, rng, fraud_type='mixed', anomaly_rate=0.05)
        return df_train, df_test_inj, y_test

    else:
        raise ValueError(f"Unknown data_mode: {data_mode}")

# ============================
# Configuration Grid
# ============================

GRID_CONFIGS = []

# --- Strategy A: Core hyperparameter sweep (streaming, single_seq) ---
for mem_len in [128, 256, 512, 1024, 2048]:
    for k in [5, 10, 15, 20]:
        for gamma in [0.0, 0.5, 0.9]:
            for latent_dim in [34, 60]:
                GRID_CONFIGS.append({
                    'strategy': 'core_sweep',
                    'memory_len': mem_len,
                    'k': k,
                    'gamma': gamma,
                    'latent_dim': latent_dim,
                    'beta_percentile': 95,
                    'warmup_samples': 30000,
                    'data_mode': 'single_seq',
                    'eval_mode': 'streaming',
                })

# --- Strategy B: Data mode comparison (mem=256, k=10, gamma=0, streaming) ---
for data_mode in ['single_seq', 'multi_month', 'single_jan', 'single_monthly_big']:
    GRID_CONFIGS.append({
        'strategy': 'data_mode_cmp',
        'memory_len': 256,
        'k': 10,
        'gamma': 0.0,
        'latent_dim': 60,
        'beta_percentile': 95,
        'warmup_samples': 30000,
        'data_mode': data_mode,
        'eval_mode': 'streaming',
    })

# --- Strategy C: Batch vs Streaming comparison ---
for eval_mode in ['streaming', 'batch']:
    GRID_CONFIGS.append({
        'strategy': 'eval_mode_cmp',
        'memory_len': 256,
        'k': 10,
        'gamma': 0.0,
        'latent_dim': 60,
        'beta_percentile': 95,
        'warmup_samples': 30000,
        'data_mode': 'single_seq',
        'eval_mode': eval_mode,
    })

# --- Strategy D: Beta percentile sweep ---
for bp in [85, 90, 95, 97, 99]:
    GRID_CONFIGS.append({
        'strategy': 'beta_pct_sweep',
        'memory_len': 256,
        'k': 10,
        'gamma': 0.0,
        'latent_dim': 60,
        'beta_percentile': bp,
        'warmup_samples': 30000,
        'data_mode': 'single_seq',
        'eval_mode': 'streaming',
    })

# --- Strategy E: Warmup size sweep ---
for wu in [5000, 10000, 20000, 30000]:
    GRID_CONFIGS.append({
        'strategy': 'warmup_sweep',
        'memory_len': 256,
        'k': 10,
        'gamma': 0.0,
        'latent_dim': 60,
        'beta_percentile': 95,
        'warmup_samples': wu,
        'data_mode': 'single_seq',
        'eval_mode': 'streaming',
    })

# --- Strategy F: Fraud type analysis ---
for fraud_type in ['type1_only', 'type2_only', 'type3_only', 'mixed']:
    GRID_CONFIGS.append({
        'strategy': 'fraud_type_cmp',
        'memory_len': 256,
        'k': 10,
        'gamma': 0.0,
        'latent_dim': 60,
        'beta_percentile': 95,
        'warmup_samples': 30000,
        'data_mode': 'single_seq',
        'eval_mode': 'streaming',
        'fraud_type': fraud_type,
    })

# --- Strategy G: Multi-month temporal eval (v10-style, best config) ---
for mem_len in [256, 512, 1024]:
    for k in [5, 10]:
        GRID_CONFIGS.append({
            'strategy': 'v10_style',
            'memory_len': mem_len,
            'k': k,
            'gamma': 0.0,
            'latent_dim': 60,
            'beta_percentile': 95,
            'warmup_samples': 30000,
            'data_mode': 'multi_month',
            'eval_mode': 'streaming',
        })

print(f"Total configs: {len(GRID_CONFIGS)}")

# ============================
# Main Run
# ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-configs', type=int, default=0, help='0=all')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--quick', action='store_true', help='Skip multi-month and big data modes')
    args = parser.parse_args()

    configs = GRID_CONFIGS
    if args.max_configs > 0:
        configs = configs[:args.max_configs]
    if args.quick:
        configs = [c for c in configs if c['data_mode'] not in ('multi_month', 'single_monthly_big')]

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_file = OUT_DIR / f'grid_search_{ts}.json'
    summary_file = OUT_DIR / f'grid_summary_{ts}.txt'

    # Pre-load all data modes
    print("\nPre-loading data...")
    data_modes = set(c['data_mode'] for c in configs)
    data_cache = {}
    for dm in data_modes:
        print(f"  Loading {dm}...")
        df_train, df_test_inj, y_test = load_data(dm, args.seed)
        X_train = features(df_train)
        X_test  = features(df_test_inj)

        pickup_test = pd.to_datetime(df_test_inj['tpep_pickup_datetime'], errors='coerce')
        hour_test   = pickup_test.dt.hour.fillna(12).astype(int).values
        dow_test    = pickup_test.dt.dayofweek.fillna(0).astype(int).values
        ratecode_test = df_test_inj['RatecodeID'].fillna(1).astype(float).values
        nb_test = np.array([
            location_to_neighborhood(loc) for loc in df_test_inj['PULocationID'].fillna(1).values
        ], dtype=int)

        data_cache[dm] = {
            'X_train': X_train, 'X_test': X_test, 'y_test': y_test,
            'nb_test': nb_test, 'hour_test': hour_test, 'dow_test': dow_test,
            'ratecode_test': ratecode_test,
            'df_test_inj': df_test_inj,
        }
        print(f"    Train: {len(df_train):,}  Test: {len(df_test_inj):,}  Anomalies: {int(y_test.sum()):,}")

    total = len(configs)
    results = []
    grouped  = defaultdict(list)  # strategy -> [(config_key, metrics)]

    print(f"\nRunning {total} configurations...")
    t_start = time.time()

    for i, cfg in enumerate(configs):
        t0 = time.time()
        dm  = cfg['data_mode']
        ft  = cfg.get('fraud_type', 'mixed')

        # Get cached data
        dc = data_cache[dm]
        X_train = dc['X_train']
        X_test  = dc['X_test']
        y_test  = dc['y_test']
        nb_test = dc['nb_test']
        hour_test = dc['hour_test']
        dow_test  = dc['dow_test']
        rc_test   = dc['ratecode_test']
        df_test_inj = dc['df_test_inj']

        # Re-inject fraud for different fraud types
        if ft != 'mixed':
            rng = np.random.RandomState(args.seed)
            df_test_fresh = dc['df_test_inj'].copy()
            _, y_test = inject_realistic_fraud(df_test_fresh, rng, fraud_type=ft, anomaly_rate=0.05)

        # Create evaluator
        try:
            ev = GPUEvaluator(
                memory_len=cfg['memory_len'],
                k=cfg['k'],
                gamma=cfg['gamma'],
                latent_dim=cfg['latent_dim'],
                beta_percentile=cfg['beta_percentile'],
                warmup_samples=cfg['warmup_samples'],
                seed=args.seed,
                device=args.device,
            )
            ev.fit(X_train,
                   hour_vals=X_train[:, 9].astype(int),
                   dow_vals=X_train[:, 10].astype(int))

            if cfg['eval_mode'] == 'streaming':
                scores = ev.score_streaming(
                    X_test, neighborhood_ids=nb_test,
                    hour_vals=hour_test, dow_vals=dow_test,
                    ratecode_vals=rc_test, update_memory=True)
            else:
                scores = ev.score_batch(
                    X_test, neighborhood_ids=nb_test,
                    hour_vals=hour_test, dow_vals=dow_test,
                    ratecode_vals=rc_test)

            # Compute metrics with best threshold
            m = compute_metrics(y_test, scores, return_best=True)
            m['config'] = cfg
            m['config_name'] = (f"mem{cfg['memory_len']}_k{cfg['k']}_g{cfg['gamma']}_"
                               f"ld{cfg['latent_dim']}_bp{cfg['beta_percentile']}_"
                               f"wu{cfg['warmup_samples']}_"
                               f"{cfg['data_mode']}_{cfg['eval_mode']}")
            m['elapsed_s'] = time.time() - t0
            m['n_test']    = len(y_test)
            m['n_anom']    = int(y_test.sum())
            m['anom_rate'] = float(y_test.mean())
            results.append(m)
            grouped[cfg['strategy']].append((m['config_name'], m))

        except Exception as e:
            m = {'config': cfg, 'error': str(e),
                 'elapsed_s': time.time() - t0}
            results.append(m)
            grouped[cfg['strategy']].append((cfg.get('strategy','?'), m))
            print(f"  ERROR: {e}")

        elapsed_total = time.time() - t_start
        eta = elapsed_total / (i + 1) * (total - i - 1)
        status = "OK" if 'error' not in results[-1] else "ERR"
        last_m = results[-1]
        f1_str = f"F1={last_m.get('F1',0):.4f}" if 'error' not in last_m else "ERROR"
        aucpr_str = f"AUC-PR={last_m.get('AUC_PR',0):.4f}" if 'error' not in last_m else ""
        print(f"  [{i+1}/{total}] {results[-1].get('config_name','?')[:70]} "
              f"{status} {f1_str} {aucpr_str} "
              f"({results[-1]['elapsed_s']:.1f}s | ETA: {eta:.0f}s)")

    # ============================
    # Save Results
    # ============================
    total_time = time.time() - t_start

    output = {
        'timestamp': ts,
        'total_configs': total,
        'successful': sum(1 for r in results if 'error' not in r),
        'failed': sum(1 for r in results if 'error' in r),
        'total_time_s': total_time,
        'results': results,
    }

    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    # ============================
    # Print Summary
    # ============================
    valid = [r for r in results if 'error' not in r]

    print("\n" + "=" * 90)
    print(f"  GRID SEARCH RESULTS ({len(valid)}/{total} successful, {total_time:.0f}s total)")
    print("=" * 90)

    # Overall top 20
    valid.sort(key=lambda x: x['F1'], reverse=True)
    print(f"\n### TOP 20 OVERALL (by F1) ###")
    print(f"{'Rank':<5} {'Config':<65} {'F1':>6} {'AUC-PR':>8} {'AUC-ROC':>9} {'Prec':>6} {'Rec':>6}")
    print("-" * 105)
    for rank, r in enumerate(valid[:20], 1):
        cfg = r['config']
        name = (f"m{cfg['memory_len']}_k{cfg['k']}_g{cfg['gamma']}_"
                f"ld{cfg['latent_dim']}_{cfg['data_mode']}_{cfg['eval_mode']}")
        print(f"{rank:<5} {name:<65} {r['F1']:6.4f} {r['AUC_PR']:8.4f} {r['AUC_ROC']:9.4f} "
              f"{r['Precision']:6.4f} {r['Recall']:6.4f}")

    # Top 5 per strategy
    print(f"\n### TOP 3 PER STRATEGY ###")
    for strategy, items in sorted(grouped.items()):
        valid_items = [(name, m) for name, m in items if 'error' not in m]
        if not valid_items: continue
        valid_items.sort(key=lambda x: x[1]['F1'], reverse=True)
        print(f"\n  [{strategy}]")
        print(f"  {'Config':<55} {'F1':>6} {'AUC-PR':>8} {'AUC-ROC':>9}")
        print(f"  {'-'*80}")
        for name, m in valid_items[:3]:
            print(f"  {name[:55]:<55} {m['F1']:6.4f} {m['AUC_PR']:8.4f} {m['AUC_ROC']:9.4f}")

    # Best for each metric
    print(f"\n### BEST PER METRIC ###")
    for metric in ['F1', 'AUC_PR', 'AUC_ROC', 'Precision', 'Recall']:
        best = max(valid, key=lambda x: x[metric])
        cfg = best['config']
        name = f"m{cfg['memory_len']}_k{cfg['k']}_g{cfg['gamma']}_{cfg['data_mode']}_{cfg['eval_mode']}"
        print(f"  Best {metric}: {best[metric]:.4f} [{name}]")

    # Write text summary
    with open(summary_file, 'w') as f:
        f.write(f"Grid Search Results\n")
        f.write(f"===================\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Total configs: {total}\n")
        f.write(f"Successful: {len(valid)}\n")
        f.write(f"Failed: {total - len(valid)}\n")
        f.write(f"Total time: {total_time:.0f}s\n\n")
        for r in valid[:30]:
            cfg = r['config']
            name = (f"m{cfg['memory_len']}_k{cfg['k']}_g{cfg['gamma']}_"
                    f"ld{cfg['latent_dim']}_bp{cfg['beta_percentile']}_"
                    f"wu{cfg['warmup_samples']}_{cfg['data_mode']}_{cfg['eval_mode']}")
            f.write(f"F1={r['F1']:.4f} AUC_PR={r['AUC_PR']:.4f} AUC_ROC={r['AUC_ROC']:.4f} "
                     f"P={r['Precision']:.4f} R={r['Recall']:.4f} | {name}\n")

    print(f"\nResults: {out_file}")
    print(f"Summary: {summary_file}")
    print(f"\nAll outputs: {OUT_DIR}")


if __name__ == '__main__':
    main()
