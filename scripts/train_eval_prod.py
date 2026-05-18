#!/usr/bin/env python3
"""
MemStream Production Training + Evaluation Pipeline
GPU-accelerated: CUDA for AE training + batch scoring
Vectorized feature extraction: no Python loops

Usage:
  docker run --gpus '"device=0"' --rm -v c:/proj/ldt/data:/data -v c:/proj/ldt/scripts:/out \
    --network host ldt-train-eval-gpu \
    --train /data/prod/train_oct2024.parquet \
    --test  /data/prod/prod.parquet \
    --gt    /data/prod/prod_gt_mask.npy \
    --out   /out \
    --signing-key changeme_memstream_signing_key_replace_with_openssl_rand_hex32 \
    --minio-endpoint minio:9000 \
    --memory-len 8192 --epochs 200
"""

import argparse, copy, hashlib, hmac, io, json, logging, os, pickle, sys, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch, torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)
LOGGER = logging.getLogger('train-eval')


# =============================================================================
# Device: CUDA if available, else CPU
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOGGER.info("Device: %s", DEVICE)
if torch.cuda.is_available():
    LOGGER.info("  GPU: %s", torch.cuda.get_device_name(0))
    LOGGER.info("  VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)


# =============================================================================
# Autoencoder
# =============================================================================

class MemStreamAE(nn.Module):
    def __init__(self, in_dim=34, hidden_dim=68):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, in_dim), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, in_dim), nn.ReLU())
    def forward(self, x): return self.decoder(self.encoder(x))
    def encode(self, x): return self.encoder(x)


# =============================================================================
# Memory Module
# =============================================================================

class MemoryModule:
    def __init__(self, memory_len=8192, out_dim=34):
        self.memory_len = memory_len
        self.out_dim = out_dim
        self.memory = torch.zeros(memory_len, out_dim, device=DEVICE)
        self.mem_usage = torch.zeros(memory_len, device=DEVICE)
        self.mem_ptr = self.count = 0
        self._is_full = self._memory_head = False

    def update(self, z):
        z = z.detach().clone().unsqueeze(0) if z.dim() == 1 else z.detach().clone()
        for i in range(min(z.shape[0], self.memory_len)):
            self.memory[self.mem_ptr] = z[i]
            self.mem_usage[self.mem_ptr] = 1.0
            self.mem_ptr = (self.mem_ptr + 1) % self.memory_len
            self.count += 1
            self._memory_head += 1
            if self.count >= self.memory_len: self._is_full = True

    def get_active(self): return self.memory[:min(self.count, self.memory_len)].clone()

    def get_state_dict(self):
        return {
            'memory': self.memory.cpu().clone(),
            'mem_usage': self.mem_usage.cpu().clone(),
            'mem_ptr': self.mem_ptr, 'count': self.count,
            '_is_full': self._is_full, '_memory_head': self._memory_head}
    def load_state_dict(self, s):
        self.memory = s['memory'].to(DEVICE)
        self.mem_usage = s['mem_usage'].to(DEVICE)
        self.mem_ptr = int(s['mem_ptr'])
        self.count = int(s['count'])
        self._is_full = bool(s.get('_is_full', False))
        self._memory_head = int(s.get('_memory_head', 0))


# =============================================================================
# ContextBeta
# =============================================================================

class ContextBeta:
    CELL_MIN = 11
    def __init__(self, default_beta=0.5, percentile=95.0):
        self.default_beta = default_beta
        self.percentile = percentile
        self._thresholds = np.full((10, 8), default_beta, dtype=np.float32)
        self._scores_per_cell: Dict[Tuple, List] = {}
        self._recent_scores: List[float] = []
        self._overall_beta = default_beta
        self._fitted = False

    def record(self, nb, ctx, score):
        k = (int(nb), int(ctx))
        if k not in self._scores_per_cell: self._scores_per_cell[k] = []
        self._scores_per_cell[k].append(float(score))
        self._recent_scores.append(float(score))

    def fit(self, overall_beta=None):
        if overall_beta is not None: self._overall_beta = float(overall_beta)
        elif self._recent_scores:
            self._overall_beta = float(np.percentile(self._recent_scores, self.percentile))
        for nb in range(10):
            for ctx in range(8):
                k = (nb, ctx)
                scores = self._scores_per_cell.get(k, [])
                self._thresholds[nb, ctx] = float(np.percentile(scores, self.percentile)) \
                    if len(scores) > self.CELL_MIN - 1 else self._overall_beta
        self._fitted = True
        return self

    def get_beta(self, nb, ctx): return float(self._thresholds[int(nb), int(ctx)])

    @staticmethod
    def compute_cell_id(hour, dow, ratecode):
        sp = 1 if ratecode > 1.0 else 0
        sn = 1 if (hour >= 20 or hour < 6) else 0
        sw = 1 if dow >= 5 else 0
        return (sp << 2) | (sn << 1) | sw

    def get_state_dict(self):
        return {'thresholds_array': self._thresholds.copy(),
                'default_beta': self.default_beta, 'percentile': self.percentile,
                'overall_beta': self._overall_beta, 'fitted': self._fitted,
                'recent_scores': list(self._recent_scores)}


# =============================================================================
# MemStream Model (GPU-accelerated)
# =============================================================================

class MemStreamModel:
    def __init__(self, memory_len=8192, hidden_dim=68):
        self.memory_len = memory_len
        self.hidden_dim = hidden_dim
        self.ae = MemStreamAE(34, hidden_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.memory = MemoryModule(memory_len, 34)
        self.mean = self.std = None
        self.max_thres = 0.5
        self.k, self.gamma = 10, 0.0
        self._context_beta: Optional[ContextBeta] = None
        self._warmup_scores: List = []
        self.count = 0

    def _normalize(self, x):
        return x if self.mean is None else (x - self.mean) / (self.std + 1e-8)

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """GPU batch scoring: compute reconstruction distance to k-nearest memory neighbors."""
        if self.mean is None or self.memory.count < 2:
            return np.full(len(X), 0.5)
        X_t = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
        X_norm = self._normalize(X_t)
        M = self.memory.count
        mem = self.memory.get_active()  # [M, 34]
        n = len(X_norm)
        # Chunk to bound O(n*M) GPU memory: fit at most 50M elements
        chunk_size = max(100, min(10000, 50_000_000 // (M * 34)))
        scores = []
        for start in range(0, n, chunk_size):
            chunk = X_norm[start:start+chunk_size]
            diff = chunk.unsqueeze(1) - mem.unsqueeze(0)  # [C, M, 34]
            dists = diff.abs().sum(dim=2)  # [C, M]
            k_u = min(self.k, M)
            vals, _ = dists.topk(k_u, dim=1, largest=False)
            scores.append(vals.sum(dim=1).cpu().numpy())
        return np.concatenate(scores)

    def score_batch_ctx(self, X: np.ndarray, nb_ids: np.ndarray,
                        hours: np.ndarray, dows: np.ndarray,
                        ratecodes: np.ndarray) -> np.ndarray:
        """Batch scoring with ContextBeta normalization."""
        raw = self.score_batch(X)
        if self._context_beta is None: return raw
        betas = np.array([
            self._context_beta.get_beta(
                int(nb_ids[i]),
                ContextBeta.compute_cell_id(int(hours[i]), int(dows[i]), float(ratecodes[i])))
            for i in range(len(X))], dtype=np.float32)
        return raw / np.maximum(betas, 1e-6)

    def warmup(self, X_normal, neighborhood_ids=None, epochs=200, batch_size=1024):
        """GPU-accelerated warmup: stats + AE training + memory init + ContextBeta."""
        torch.manual_seed(42); np.random.seed(42)
        n = len(X_normal)

        # Normalization from first 10%
        t0 = time.time()
        stats_data = torch.from_numpy(X_normal[:max(1, int(n*0.1))]).float().to(DEVICE)
        self.mean = stats_data.mean(dim=0)
        self.std = stats_data.std(dim=0)
        self.std = torch.clamp(self.std, min=1.0)
        LOGGER.info("  Stats computed in %.1fs", time.time()-t0)

        # AE training on middle 80%
        train_data = torch.from_numpy(X_normal[int(n*0.1):int(n*0.9)]).float().to(DEVICE)
        X_norm = self._normalize(train_data)
        self.ae.train()
        best_loss, patience, best_state = float('inf'), 0, None
        t0 = time.time()
        for epoch in range(epochs):
            idx = torch.randperm(len(X_norm), device=DEVICE)
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(X_norm), batch_size):
                batch = X_norm[idx[i:i+batch_size]]
                noise = torch.randn_like(batch) * 0.1
                loss = self.criterion(self.ae(batch + noise), batch)
                self.optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item(); n_batches += 1
            avg = total_loss / max(n_batches, 1)
            if avg < best_loss: best_loss = avg; best_state = copy.deepcopy(self.ae.state_dict()); patience = 0
            else: patience += 1
            if patience >= 20: LOGGER.info("  Early stop @ epoch %d", epoch+1); break
            if (epoch+1) % 50 == 0: LOGGER.info("  Epoch %d/%d: loss=%.6f", epoch+1, epochs, avg)
        if best_state is not None: self.ae.load_state_dict(best_state)
        self.ae.eval()
        LOGGER.info("  AE trained: %d epochs, loss=%.6f, time=%.1fs", epoch+1, best_loss, time.time()-t0)

        # Memory init from last 10% (sequential, on GPU)
        mem_data = torch.from_numpy(X_normal[int(n*0.9):]).float().to(DEVICE)
        n_mem = min(self.memory_len, len(mem_data))
        with torch.no_grad():
            enc = self.ae.encode(mem_data)
            idx_start = len(enc) - n_mem
            self.memory.memory[:n_mem] = enc[idx_start:].clone()
            self.memory.mem_usage[:n_mem].fill_(1.0)
            self.memory.count = n_mem
        LOGGER.info("  Memory initialized: %d / %d slots", n_mem, self.memory_len)

        # ContextBeta fitting (batch scoring on GPU)
        warmup_n = min(4096, len(X_norm))
        w_data = X_norm[:warmup_n].cpu().numpy()
        w_nb = (neighborhood_ids[:warmup_n] if neighborhood_ids is not None
                else np.zeros(warmup_n, dtype=int))
        raw_scores = self.score_batch(w_data)
        cb = ContextBeta(0.5, 95)
        # Vectorized context ID + batch record
        h_w = w_data[:, 9].astype(int)
        d_w = w_data[:, 10].astype(int)
        rc_w = np.full(warmup_n, 1.0)
        for i_rc, rc_val in enumerate([1,2,3,4,5]):
            mask = w_data[:, 25+i_rc] > 0.5
            rc_w[mask] = rc_val
        ctx_ids = ContextBeta.compute_cell_id(h_w, d_w, rc_w)
        for i in range(warmup_n):
            cb.record(int(w_nb[i]), int(ctx_ids[i]), raw_scores[i])
        cb.fit()
        self._context_beta = cb
        self.max_thres = float(np.percentile(raw_scores, 95))
        self._warmup_scores = raw_scores.tolist()
        LOGGER.info("  ContextBeta: beta=%.4f, warmup_samples=%d", self.max_thres, warmup_n)

    def get_state_dict(self):
        s = {'ae_state': self.ae.state_dict(),
             'mean': self.mean.cpu() if self.mean is not None else None,
             'std': self.std.cpu() if self.std is not None else None,
             'max_thres': self.max_thres,
             'memory_state': self.memory.get_state_dict(),
             'k': self.k, 'gamma': self.gamma,
             '_warmup_scores': self._warmup_scores,
             'count': self.count,
             'cfg': {'memory_len': self.memory_len, 'hidden_dim': self.hidden_dim,
                     'in_dim': 34, 'out_dim': 34, 'k': 10, 'gamma': 0.0,
                     'warmup_epochs': 200, 'warmup_batch_size': 1024,
                     'warmup_noise_std': 0.1, 'default_beta': 0.5,
                     'warmup_min_samples': 8192, 'seed': 42}}
        if self._context_beta: s['context_beta'] = self._context_beta.get_state_dict()
        return s


# =============================================================================
# FAST 34D Feature Extraction — Fully vectorized, NO Python loops
# =============================================================================

def extract_features_fast(tbl) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fully vectorized 34D feature extraction from pyarrow Table.
    Returns: (X, hours, dows, ratecodes, nb_ids)
    """
    import pyarrow.compute as pc
    n = tbl.num_rows

    def get_np(name, default=0.0):
        col = tbl.column(name)
        mask = pc.is_null(col)
        arr = pc.if_else(mask, default, col).to_numpy().astype(np.float32)
        return np.nan_to_num(arr, nan=default)

    def get_int(name, default=0):
        col = tbl.column(name)
        mask = pc.is_null(col)
        arr = pc.if_else(mask, default, col).to_numpy()
        return np.round(arr).astype(np.int32)

    # Raw columns (vectorized)
    dist = get_np('trip_distance', 0.0)
    fare = get_np('fare_amount', 0.0)
    pax = get_np('passenger_count', 1.0)
    total_amt = get_np('total_amount', 0.0)
    ratecode = get_np('RatecodeID', 1.0)
    pu_arr = get_int('PULocationID', 0)
    do_arr = get_int('DOLocationID', 0)

    # Datetime: extract hour/dow fully vectorized
    pickup_col = tbl.column('tpep_pickup_datetime')
    dropoff_col = tbl.column('tpep_dropoff_datetime')

    # Try to cast to timestamp (fast path)
    try:
        pickup_ts = pc.cast(pickup_col, 'timestamp[us]').to_numpy()
        dropoff_ts = pc.cast(dropoff_col, 'timestamp[us]').to_numpy()
        pickup_ms = pickup_ts.astype('datetime64[ms]').astype('int64')
        dropoff_ms = dropoff_ts.astype('datetime64[ms]').astype('int64')
        dur_min = np.clip((dropoff_ms - pickup_ms) / 60000.0, 1, 360).astype(np.float32)
        hours = ((pickup_ms // 3600000) % 24).astype(np.int32)
        dows = ((pickup_ms // 86400000) % 7).astype(np.int32)
    except Exception:
        # Fallback: parse from string
        pickup_str = pc.cast(pickup_col, 'str').to_pylist()
        dropoff_str = pc.cast(dropoff_col, 'str').to_pylist()
        hours = np.zeros(n, dtype=np.int32)
        dows = np.zeros(n, dtype=np.int32)
        dur_min = np.full(n, 15.0, dtype=np.float32)
        for i, pv in enumerate(pickup_str):
            try:
                dt = __import__('datetime').datetime.fromisoformat(str(pv).replace('Z','').replace('/','-'))
                hours[i] = dt.hour
                dows[i] = dt.weekday()
            except: pass
        for i, (pv, dv) in enumerate(zip(pickup_str, dropoff_str)):
            try:
                d1 = __import__('datetime').datetime.fromisoformat(str(pv).replace('Z','').replace('/','-'))
                d2 = __import__('datetime').datetime.fromisoformat(str(dv).replace('Z','').replace('/','-'))
                dur_min[i] = max(1, min(360, (d2-d1).total_seconds()/60.0))
            except: pass

    # Speed
    eps = np.float32(0.01)
    speed = np.clip(dist / (dur_min / 60.0 + eps) * (dur_min > 0), 0, 60).astype(np.float32)

    # Build 34D feature array
    X = np.zeros((n, 34), dtype=np.float32)
    hr = hours.astype(np.float32)
    dw = dows.astype(np.float32)

    # Zone → grid (fully vectorized)
    pu_zone = np.clip(pu_arr, 0, 263)
    do_zone = np.clip(do_arr, 0, 263)
    pux = ((pu_zone - 1) % 16).astype(np.float32)
    puy = ((pu_zone - 1) // 16).astype(np.float32)
    dox = ((do_zone - 1) % 16).astype(np.float32)
    doy = ((do_zone - 1) // 16).astype(np.float32)

    # Neighborhood (vectorized)
    nb_ids = np.select(
        [pu_arr <= 0, (pu_arr >= 1) & (pu_arr <= 43), (pu_arr >= 44) & (pu_arr <= 103),
         (pu_arr >= 104) & (pu_arr <= 127), (pu_arr >= 128) & (pu_arr <= 148),
         (pu_arr >= 149) & (pu_arr <= 161), (pu_arr >= 162) & (pu_arr <= 263)],
        [9, 0, 4, 1, 2, 3, 5], default=9).astype(np.int32)

    # Ratecode for context
    ratecodes = np.select(
        [ratecode == 1, ratecode == 2, ratecode == 3, ratecode == 4, ratecode == 5],
        [1, 2, 3, 4, 5], default=1).astype(np.float32)

    # Features 0-8: raw + derived
    X[:, 0] = dist; X[:, 1] = dur_min; X[:, 2] = fare
    X[:, 3] = pax; X[:, 4] = total_amt; X[:, 5] = speed
    X[:, 6] = np.where(dist > 0.1, fare / np.maximum(dist, eps), 0.0)
    X[:, 7] = np.where(dur_min > 1, fare / np.maximum(dur_min, eps), 0.0)
    X[:, 8] = np.where(pax > 0, fare / np.maximum(pax, eps), 0.0)

    # Features 9-11: temporal
    X[:, 9] = hr; X[:, 10] = dw; X[:, 11] = (dw >= 5).astype(np.float32)

    # Features 12-15: spatial grid
    X[:, 12] = pux; X[:, 13] = puy; X[:, 14] = dox; X[:, 15] = doy

    # Features 16-19: normalized ratios
    X[:, 16] = np.clip(X[:, 6] / 2.5, 0, 20)
    X[:, 17] = np.clip(X[:, 7] / 0.67, 0, 20)
    X[:, 18] = np.clip(speed / 12.0, 0, 20)
    X[:, 19] = np.where(dist > 0.1, pax / np.maximum(dist, eps), 0.0)

    # Features 20-24: cyclic + squared
    X[:, 20] = np.sin(2*np.pi * hr / 24.0)
    X[:, 21] = np.cos(2*np.pi * hr / 24.0)
    X[:, 22] = np.sin(2*np.pi * dw / 7.0)
    X[:, 23] = np.cos(2*np.pi * dw / 7.0)
    X[:, 24] = dist * dist

    # Features 25-29: ratecode one-hot
    X[:, 25] = (ratecode == 1).astype(np.float32)
    X[:, 26] = (ratecode == 2).astype(np.float32)
    X[:, 27] = (ratecode == 3).astype(np.float32)
    X[:, 28] = (ratecode == 4).astype(np.float32)
    X[:, 29] = (ratecode == 5).astype(np.float32)

    # Feature 30: night
    X[:, 30] = ((hr >= 20) | (hr <= 6)).astype(np.float32)

    # Features 31-32: log
    X[:, 31] = np.log1p(np.clip(fare, 0, 1e6))
    X[:, 32] = np.log1p(np.clip(dist, 0, 1e6))

    # Feature 33: inter-zone
    X[:, 33] = np.abs(puy - doy)

    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)
    return X, hours, dows, ratecodes, nb_ids


# =============================================================================
# MinIO Upload
# =============================================================================

def upload_to_minio(data, bucket, key, endpoint, access_key, secret_key):
    import boto3
    from botocore.config import Config
    client = boto3.client('s3', endpoint_url=f'http://{endpoint}',
                          aws_access_key_id=access_key, aws_secret_access_key=secret_key,
                          config=Config(signature_version='s3v4', retries={'max_attempts': 3},
                                      s3={'addressing_style': 'path'}))
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType='application/octet-stream')
    LOGGER.info("Uploaded: %s/%s (%.0fKB)", bucket, key, len(data)/1024)


# =============================================================================
# Training
# =============================================================================

def train_model(train_path, memory_len, epochs, batch_size, signing_key,
                minio_endpoint, minio_access, minio_secret, output_dir):
    LOGGER.info("=" * 60)
    LOGGER.info("GPU Training: memory_len=%d, epochs=%d, batch=%d", memory_len, epochs, batch_size)
    LOGGER.info("=" * 60)

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(train_path)
    n_total = pf.metadata.num_rows
    n_rg = pf.metadata.num_row_groups
    max_needed = memory_len * 10  # 10/80/10 split
    rows_to_read = min(n_total, max_needed)

    LOGGER.info("  Total: %d rows | Need last %d (%.1fx memory_len)",
                n_total, rows_to_read, rows_to_read / memory_len)

    # Read only needed row groups + slice
    t0 = time.time()
    if n_rg > 0 and rows_to_read < n_total:
        rows_per_rg = n_total // n_rg
        skip_rgs = max(0, (n_total - rows_to_read) // rows_per_rg)
        tbl = pf.read_row_groups(list(range(skip_rgs, n_rg)))
        LOGGER.info("  Row groups: %d total, skipped %d, reading %d",
                    n_rg, skip_rgs, n_rg - skip_rgs)
    else:
        tbl = pf.read()
    LOGGER.info("  Loaded %d rows in %.1fs", tbl.num_rows, time.time()-t0)

    # Extract features (vectorized)
    LOGGER.info("Extracting 34D features (vectorized)...")
    t0 = time.time()
    X_all, hours_all, dows_all, ratecodes_all, nb_ids_all = extract_features_fast(tbl)
    total = X_all.shape[0]
    LOGGER.info("  %d x 34D features in %.1fs", total, time.time()-t0)

    # Slice to last rows_to_read
    if total > rows_to_read:
        X_all = X_all[-rows_to_read:]
        hours_all = hours_all[-rows_to_read:]
        dows_all = dows_all[-rows_to_read:]
        ratecodes_all = ratecodes_all[-rows_to_read:]
        nb_ids_all = nb_ids_all[-rows_to_read:]
    n_actual = X_all.shape[0]
    LOGGER.info("  Training on last %d rows (10%% stats / 80%% AE / 10%% memory)", n_actual)

    # Warmup
    model = MemStreamModel(memory_len=memory_len, hidden_dim=68)
    t0 = time.time()
    model.warmup(X_all, neighborhood_ids=nb_ids_all, epochs=epochs, batch_size=batch_size)
    LOGGER.info("Training done in %.1fs", time.time()-t0)

    # Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, 'memstream_memory.pt')
    state = model.get_state_dict()
    buf = io.BytesIO()
    torch.save(state, buf, pickle_module=pickle)
    data = buf.getvalue()
    with open(ckpt_path, 'wb') as f: f.write(data)
    sig = hmac.new(signing_key.encode(), data, hashlib.sha256).hexdigest()
    with open(ckpt_path + '.hmac', 'w') as f: f.write(sig)
    LOGGER.info("Checkpoint: %s (%.1fMB)", ckpt_path, len(data)/1024/1024)

    if minio_endpoint:
        try:
            upload_to_minio(data, 'cadqstream-drift', 'checkpoints/memstream/memstream_memory.pt',
                          minio_endpoint, minio_access, minio_secret)
            upload_to_minio(sig.encode(), 'cadqstream-drift',
                          'checkpoints/memstream/memstream_memory.pt.hmac',
                          minio_endpoint, minio_access, minio_secret)
        except Exception as e:
            LOGGER.warning("MinIO upload failed: %s", e)

    return model, {'memory_len': memory_len, 'train_samples': n_actual,
                    'epochs': epochs, 'max_thres': float(model.max_thres)}


# =============================================================================
# Evaluation (GPU batch scoring)
# =============================================================================

def evaluate_model(model, test_path, gt_path):
    LOGGER.info("=" * 60)
    LOGGER.info("GPU Evaluation: prod.parquet + prod_gt_mask.npy")
    LOGGER.info("=" * 60)

    gt = np.load(gt_path)
    LOGGER.info("Ground truth: %d samples, %d anomalies (%.2f%%)",
                len(gt), int(gt.sum()), gt.sum()/len(gt)*100)

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(test_path)
    n_test = pf.metadata.num_rows
    use_n = min(n_test, len(gt))
    LOGGER.info("Test parquet: %d rows | Aligning to %d", n_test, use_n)

    # Score by row groups
    t0 = time.time()
    all_scores, all_nb, all_h, all_d, all_rc = [], [], [], [], []
    offset = 0

    for rg_idx in range(pf.metadata.num_row_groups):
        if offset >= use_n: break
        tbl = pf.read_row_groups([rg_idx])
        X_b, h_b, d_b, rc_b, nb_b = extract_features_fast(tbl)
        n_b = X_b.shape[0]

        if offset + n_b > use_n:
            excess = offset + n_b - use_n
            X_b, h_b, d_b, rc_b, nb_b = X_b[:-excess], h_b[:-excess], d_b[:-excess], rc_b[:-excess], nb_b[:-excess]
            n_b = len(X_b)
        if n_b == 0: break

        scores = model.score_batch_ctx(X_b, nb_b, h_b, d_b, rc_b)
        all_scores.extend(scores.tolist())
        offset += n_b
        LOGGER.info("  Scored %d / %d (%.0f%%)", offset, use_n, offset/use_n*100)

    scores_arr = np.array(all_scores[:use_n], dtype=np.float64)
    gt_al = gt[:use_n]
    elapsed = time.time() - t0
    LOGGER.info("Scored %d samples in %.1fs (%.0f rec/s)", len(scores_arr), elapsed, len(scores_arr)/max(elapsed,1))

    # Metrics
    pred = (scores_arr > 1.0).astype(int)
    tp = int(((pred==1)&(gt_al==1)).sum())
    fp = int(((pred==1)&(gt_al==0)).sum())
    tn = int(((pred==0)&(gt_al==0)).sum())
    fn = int(((pred==0)&(gt_al==1)).sum())
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    rec = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    fpr = fp/(fp+tn) if (fp+tn) > 0 else 0.0
    acc = (tp+tn)/len(gt_al)

    try:
        from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
        auc_roc = roc_auc_score(gt_al, scores_arr)
        p, r, _ = precision_recall_curve(gt_al, scores_arr)
        auc_pr = auc(r, p)
    except:
        auc_roc = auc_pr = 0.5

    nm = scores_arr[gt_al==0]; am = scores_arr[gt_al==1]
    LOGGER.info("=" * 60)
    LOGGER.info("RESULTS")
    LOGGER.info("=" * 60)
    LOGGER.info("AUC-ROC: %.4f | AUC-PR: %.4f", auc_roc, auc_pr)
    LOGGER.info("Precision: %.4f | Recall: %.4f | F1: %.4f", prec, rec, f1)
    LOGGER.info("TP: %d | FP: %d | TN: %d | FN: %d", tp, fp, tn, fn)
    LOGGER.info("FPR: %.4f | Accuracy: %.4f", fpr, acc)
    LOGGER.info("Score dist - Normal: mean=%.4f, Anomaly: mean=%.4f", nm.mean(), am.mean())

    return {'n_samples': use_n, 'n_anomalies': int(gt_al.sum()),
            'tp':tp,'fp':fp,'tn':tn,'fn':fn,
            'precision':prec,'recall':rec,'f1':f1,'fpr':fpr,'accuracy':acc,
            'auc_roc':auc_roc,'auc_pr':auc_pr,
            'score_normal_mean': float(nm.mean()), 'score_anomaly_mean': float(am.mean()),
            'eval_time': elapsed}


# =============================================================================
# Entry Point
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out', default='/out')
    ap.add_argument('--signing-key', required=True)
    ap.add_argument('--minio-endpoint', default='')
    ap.add_argument('--minio-access', default='minioadmin')
    ap.add_argument('--minio-secret', default='minioadmin123')
    ap.add_argument('--memory-len', type=int, default=8192)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--eval-only', default='')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.eval_only:
        state = torch.load(args.eval_only, map_location='cpu', weights_only=False)
        model = MemStreamModel(memory_len=args.memory_len)
        model.ae.load_state_dict(state['ae_state'])
        model.mean = state['mean'].to(DEVICE) if state.get('mean') is not None else None
        model.std = state['std'].to(DEVICE) if state.get('std') is not None else None
        model.max_thres = float(state.get('max_thres', 0.5))
        model.memory.load_state_dict(state['memory_state'])
        if 'context_beta' in state:
            cb = ContextBeta(0.5, 95)
            cb._thresholds = state['context_beta']['thresholds_array']
            cb._overall_beta = float(state['context_beta']['overall_beta'])
            cb._fitted = True
            model._context_beta = cb
        LOGGER.info("Loaded checkpoint: %s", args.eval_only)
    else:
        model, meta = train_model(
            args.train, args.memory_len, args.epochs, args.batch_size,
            args.signing_key, args.minio_endpoint, args.minio_access,
            args.minio_secret, args.out)
        with open(os.path.join(args.out, 'train_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    results = evaluate_model(model, args.test, args.gt)
    with open(os.path.join(args.out, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Done!")

if __name__ == '__main__':
    sys.exit(main())
