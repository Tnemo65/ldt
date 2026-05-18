#!/usr/bin/env python3
"""
MemStream Training + Evaluation - Ultra-optimized pipeline.
Vectorized feature extraction (no Python loops), fast AE training.
"""
import argparse, copy, hashlib, hmac, io, json, logging, os, pickle, sys, time
from typing import Dict, List, Tuple
import numpy as np
import torch, torch.nn as nn
import pyarrow.parquet as pq
import pyarrow.compute as pc

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)
LOGGER = logging.getLogger('memstream')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOGGER.info("Device: %s", DEVICE)


# =============================================================================
# Autoencoder
# =============================================================================
class AE(nn.Module):
    def __init__(self, d=34, h=68):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,h), nn.ReLU(), nn.Linear(h,d), nn.ReLU())
    def forward(self, x): return self.net(x)
    def encode(self, x): return self.net[:2](x)


# =============================================================================
# Memory
# =============================================================================
class Memory:
    def __init__(self, L, d):
        self.L, self.d = L, d
        self.M = torch.zeros(L, d, device=DEVICE)
        self.U = torch.zeros(L, device=DEVICE)
        self.ptr = self.cnt = 0
        self.full, self.head = False, 0

    def update(self, z):
        z = z.detach().clone().unsqueeze(0) if z.dim() == 1 else z.detach().clone()
        for i in range(min(z.shape[0], self.L)):
            self.M[self.ptr] = z[i]; self.U[self.ptr] = 1.0
            self.ptr = (self.ptr + 1) % self.L; self.cnt += 1; self.head += 1
            if self.cnt >= self.L: self.full = True

    def active(self): return self.M[:min(self.cnt, self.L)].clone()

    def sd(self):
        return {'M': self.M.cpu().clone(), 'U': self.U.cpu().clone(),
                'ptr': self.ptr, 'cnt': self.cnt, 'full': self.full, 'head': self.head}

    def ld(self, s):
        self.M = s['M'].to(DEVICE); self.U = s['U'].to(DEVICE)
        self.ptr = int(s['ptr']); self.cnt = int(s['cnt'])
        self.full = bool(s.get('full', False)); self.head = int(s.get('head', 0))


# =============================================================================
# ContextBeta
# =============================================================================
class CB:
    CELL_MIN = 11

    def __init__(self, db=0.5, pct=95.0):
        self.db = db; self.pct = pct
        self.T = np.full((10, 8), db, dtype=np.float32)
        self.S: Dict[tuple, List] = {}
        self.R: List[float] = []; self.beta = db; self.fitted = False

    def rec(self, nb, ctx, s):
        k = (int(nb), int(ctx))
        if k not in self.S: self.S[k] = []
        self.S[k].append(float(s)); self.R.append(float(s))

    def fit(self, ob=None):
        if ob is not None: self.beta = float(ob)
        elif self.R: self.beta = float(np.percentile(self.R, self.pct))
        for nb in range(10):
            for ctx in range(8):
                sc = self.S.get((nb, ctx), [])
                self.T[nb, ctx] = float(np.percentile(sc, self.pct)) if len(sc) > self.CELL_MIN - 1 else self.beta
        self.fitted = True; return self

    def beta_for(self, nb, ctx): return float(self.T[int(nb), int(ctx)])

    @staticmethod
    def cid(hour, dow, rc):
        sp = 1 if rc > 1 else 0
        sn = 1 if hour >= 20 or hour < 6 else 0
        sw = 1 if dow >= 5 else 0
        return (sp << 2) | (sn << 1) | sw

    def sd(self):
        return {'T': self.T.copy(), 'db': self.db, 'pct': self.pct,
                'beta': self.beta, 'fitted': self.fitted, 'R': list(self.R)}


# =============================================================================
# FAST Vectorized Feature Extraction (NO Python loops)
# =============================================================================

def _col_f32(tbl, name, default=0.0):
    c = tbl.column(name)
    mask = pc.is_null(c)
    arr = pc.if_else(mask, default, c).to_numpy().astype(np.float32)
    return np.nan_to_num(arr, nan=default)

def _col_i32(tbl, name, default=0):
    c = tbl.column(name)
    mask = pc.is_null(c)
    arr = pc.if_else(mask, default, c).to_numpy()
    return np.round(arr).astype(np.int32)

def extract_features_from_table(tbl) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized: pyarrow Table -> (X_34D, hours, dows, ratecodes, nb_ids). Zero Python loops."""
    n = tbl.num_rows
    if n == 0:
        return np.zeros((0, 34), np.float32), np.array([]), np.array([]), np.array([]), np.array([])

    # Raw columns
    dist = _col_f32(tbl, 'trip_distance', 0.0)
    fare = _col_f32(tbl, 'fare_amount', 0.0)
    pax   = _col_f32(tbl, 'passenger_count', 1.0)
    tot   = _col_f32(tbl, 'total_amount', 0.0)
    rc    = _col_f32(tbl, 'RatecodeID', 1.0)
    pu_z  = _col_i32(tbl, 'PULocationID', 0)
    do_z  = _col_i32(tbl, 'DOLocationID', 0)

    # Datetime (vectorized)
    try:
        pts = pc.cast(tbl.column('tpep_pickup_datetime'), 'timestamp[ms]').to_numpy()
        dts = pc.cast(tbl.column('tpep_dropoff_datetime'), 'timestamp[ms]').to_numpy()
        hours = ((pts // 3600000) % 24).astype(np.int32)
        dows  = ((pts // 86400000) % 7).astype(np.int32)
        dur    = np.clip((dts - pts) / 60000.0, 1, 360).astype(np.float32)
    except Exception:
        hours = np.zeros(n, np.int32); dows = np.zeros(n, np.int32)
        dur   = np.full(n, 15.0, np.float32)

    # Speed
    eps = np.float32(1e-8)
    speed = np.clip(dist / (dur / 60.0 + eps) * (dur > 0), 0, 60).astype(np.float32)

    # Zone->grid (fully vectorized)
    puz_c = np.clip(pu_z, 0, 263); doz_c = np.clip(do_z, 0, 263)
    pux = ((puz_c - 1) % 16).astype(np.float32)
    puy = ((puz_c - 1) // 16).astype(np.float32)
    dox = ((doz_c - 1) % 16).astype(np.float32)
    doy = ((doz_c - 1) // 16).astype(np.float32)

    # Neighborhood IDs (vectorized)
    nb_ids = np.select(
        [pu_z <= 0, (pu_z >= 1) & (pu_z <= 43), (pu_z >= 44) & (pu_z <= 103),
         (pu_z >= 104) & (pu_z <= 127), (pu_z >= 128) & (pu_z <= 148),
         (pu_z >= 149) & (pu_z <= 161), (pu_z >= 162) & (pu_z <= 263)],
        [9, 0, 4, 1, 2, 3, 5], default=9).astype(np.int32)

    # Ratecodes (for context)
    rcs = np.select([rc == 1, rc == 2, rc == 3, rc == 4, rc == 5],
                    [1, 2, 3, 4, 5], default=1).astype(np.float32)

    # Build 34D (all numpy, no loops)
    X = np.empty((n, 34), np.float32)
    hr = hours.astype(np.float32); dw = dows.astype(np.float32)

    X[:, 0] = dist; X[:, 1] = dur; X[:, 2] = fare; X[:, 3] = pax
    X[:, 4] = tot; X[:, 5] = speed
    X[:, 6] = np.where(dist > 0.1, fare / np.maximum(dist, eps), 0.0)
    X[:, 7] = np.where(dur > 1, fare / np.maximum(dur, eps), 0.0)
    X[:, 8] = np.where(pax > 0, fare / np.maximum(pax, eps), 0.0)
    X[:, 9] = hr; X[:, 10] = dw; X[:, 11] = (dw >= 5).astype(np.float32)
    X[:, 12] = pux; X[:, 13] = puy; X[:, 14] = dox; X[:, 15] = doy
    X[:, 16] = np.clip(X[:, 6] / 2.5, 0, 20)
    X[:, 17] = np.clip(X[:, 7] / 0.67, 0, 20)
    X[:, 18] = np.clip(speed / 12.0, 0, 20)
    X[:, 19] = np.where(dist > 0.1, pax / np.maximum(dist, eps), 0.0)
    X[:, 20] = np.sin(2 * np.pi * hr / 24.0); X[:, 21] = np.cos(2 * np.pi * hr / 24.0)
    X[:, 22] = np.sin(2 * np.pi * dw / 7.0); X[:, 23] = np.cos(2 * np.pi * dw / 7.0)
    X[:, 24] = dist * dist
    X[:, 25] = (rc == 1).astype(np.float32); X[:, 26] = (rc == 2).astype(np.float32)
    X[:, 27] = (rc == 3).astype(np.float32); X[:, 28] = (rc == 4).astype(np.float32)
    X[:, 29] = (rc == 5).astype(np.float32)
    X[:, 30] = ((hr >= 20) | (hr <= 6)).astype(np.float32)
    X[:, 31] = np.log1p(np.clip(fare, 0, 1e6))
    X[:, 32] = np.log1p(np.clip(dist, 0, 1e6))
    X[:, 33] = np.abs(puy - doy)
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)
    return X, hours, dows, rcs, nb_ids


def extract_features(pq_path, max_rows=None):
    """Load parquet and extract features. Reads only last max_rows if specified."""
    pf = pq.ParquetFile(pq_path)
    n_total = pf.metadata.num_rows
    n_rg = pf.metadata.num_row_groups

    if max_rows and max_rows < n_total and n_rg > 0:
        rpg = n_total // n_rg
        skip_rgs = max(0, (n_total - max_rows) // rpg)
        tbl = pf.read_row_groups(list(range(skip_rgs, n_rg)))
    else:
        tbl = pf.read()

    if max_rows and tbl.num_rows > max_rows:
        tbl = tbl.slice(tbl.num_rows - max_rows, max_rows)

    return extract_features_from_table(tbl)


# =============================================================================
# Batch Scoring
# =============================================================================
class Scorer:
    def __init__(self, ae, mean, std, mem, cb, k=10):
        self.ae = ae; self.mean = mean; self.std = std
        self.mem = mem; self.cb = cb; self.k = k

    def score(self, X):
        if self.mean is None or self.mem.cnt < 2:
            return np.full(len(X), 0.5)
        M = self.mem.active()
        Xn = (torch.from_numpy(X).to(DEVICE) - self.mean) / (self.std + 1e-8)
        n = len(Xn); cnt = len(M)
        cs = max(50, min(5000, 100_000_000 // (cnt * 34)))
        out = []
        for s in range(0, n, cs):
            chunk = Xn[s:s+cs].unsqueeze(1)
            diff = chunk - M.unsqueeze(0)
            d = diff.abs().sum(2)
            v, _ = d.topk(min(self.k, cnt), dim=1, largest=False)
            out.append(v.sum(1).cpu().numpy())
        return np.concatenate(out)

    def score_ctx(self, X, nb, h, d, rc):
        raw = self.score(X)
        if self.cb is None:
            return raw
        cids = np.array([CB.cid(int(h[i]), int(d[i]), float(rc[i])) for i in range(len(X))], dtype=np.int32)
        betas = np.array([self.cb.beta_for(int(nb[i]), int(cids[i])) for i in range(len(X))], dtype=np.float32)
        return raw / np.maximum(betas, 1e-6)


# =============================================================================
# MinIO Upload
# =============================================================================
def upload(data, bucket, key, ep, ak, sk):
    import boto3
    from botocore.config import Config
    boto3.client('s3', endpoint_url=f'http://{ep}', aws_access_key_id=ak,
        aws_secret_access_key=sk,
        config=Config(signature_version='s3v4', retries={'max_attempts': 3},
                      s3={'addressing_style': 'path'})
    ).put_object(Bucket=bucket, Key=key, Body=data, ContentType='application/octet-stream')
    LOGGER.info("Uploaded %s/%s", bucket, key)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True); ap.add_argument('--test', required=True)
    ap.add_argument('--gt', required=True); ap.add_argument('--out', default='/out')
    ap.add_argument('--signing-key', required=True)
    ap.add_argument('--minio-ep', default=''); ap.add_argument('--minio-ak', default='minioadmin')
    ap.add_argument('--minio-sk', default='minioadmin123')
    ap.add_argument('--mlen', type=int, default=8192); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=256); ap.add_argument('--eval-only', default='')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ===== TRAINING =====
    if not args.eval_only:
        LOGGER.info("=" * 60)
        LOGGER.info("PHASE 1: Training (mem=%d, epochs=%d, batch=%d)", args.mlen, args.epochs, args.bs)
        LOGGER.info("=" * 60)
        t0 = time.time()

        rows_needed = args.mlen * 10
        LOGGER.info("Loading last %d rows from training data...", rows_needed)
        X, hours, dows, rcs, nb_ids = extract_features(args.train, max_rows=rows_needed)
        n = len(X)
        LOGGER.info("  Loaded %d rows (features extracted in %.1fs)", n, time.time() - t0)

        # Normalization (first 10%)
        X_stats = torch.from_numpy(X[:max(1, int(n * 0.1))]).float().to(DEVICE)
        mean = X_stats.mean(0); std = X_stats.std(0); std = torch.clamp(std, min=1.0)
        LOGGER.info("  Stats: mean, std computed")

        # AE training (middle 80%)
        X_tr = torch.from_numpy(X[int(n*0.1):int(n*0.9)]).float().to(DEVICE)
        Xn = (X_tr - mean) / (std + 1e-8)
        ae = AE(34, 68).to(DEVICE)
        opt = torch.optim.Adam(ae.parameters(), lr=5e-3)
        crit = nn.MSELoss()

        LOGGER.info("Training AE (%d samples x %d epochs)...", len(Xn), args.epochs)
        t1 = time.time()
        for ep in range(args.epochs):
            idx = torch.randperm(len(Xn), device=DEVICE)
            tot = 0.0; nb = 0
            for i in range(0, len(Xn), args.bs):
                b = Xn[idx[i:i+args.bs]]
                loss = crit(ae(b + torch.randn_like(b) * 0.1), b)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0); opt.step()
                tot += loss.item(); nb += 1
            if (ep + 1) % 10 == 0:
                LOGGER.info("  Epoch %d/%d: loss=%.6f", ep+1, args.epochs, tot / max(nb, 1))
        ae.eval()
        LOGGER.info("  AE done in %.1fs (%.0f samp/s)", time.time()-t1, len(Xn)*args.epochs / max(time.time()-t1, 1))

        # Memory init (last 10%)
        mem = Memory(args.mlen, 34)
        X_mem = torch.from_numpy(X[int(n*0.9):]).float().to(DEVICE)
        n_mem = min(args.mlen, len(X_mem))
        with torch.no_grad():
            enc = ae.encode((X_mem - mean) / (std + 1e-8))
            mem.M[:n_mem] = enc[-n_mem:].clone()
            mem.U[:n_mem].fill_(1.0); mem.cnt = n_mem
        LOGGER.info("  Memory: %d/%d slots", n_mem, args.mlen)

        # ContextBeta (warmup on 4K sample)
        cb = CB(0.5, 95)
        wn = min(4096, len(Xn))
        scorer_w = Scorer(ae, mean, std, mem, None, 10)
        raw = scorer_w.score(X[:wn])
        cids = np.array([CB.cid(int(hours[i]), int(dows[i]), float(rcs[i])) for i in range(wn)], dtype=np.int32)
        for i in range(wn): cb.rec(int(nb_ids[i]), int(cids[i]), raw[i])
        cb.fit()
        beta_thres = float(np.percentile(raw, 95))
        LOGGER.info("  ContextBeta: beta=%.4f", beta_thres)
        LOGGER.info("  Total training time: %.1fs", time.time() - t0)

        # Save checkpoint
        state = {
            'ae': ae.state_dict(), 'mean': mean.cpu(), 'std': std.cpu(),
            'mem': mem.sd(), 'cb': cb.sd(),
            'k': 10, 'gamma': 0.0, 'mlen': args.mlen,
            'cfg': {'mlen': args.mlen, 'hidden': 68, 'k': 10, 'gamma': 0,
                    'warmup_epochs': args.epochs}
        }
        buf = io.BytesIO(); torch.save(state, buf, pickle_module=pickle)
        data = buf.getvalue()
        ckpt = os.path.join(args.out, 'memstream_memory.pt')
        with open(ckpt, 'wb') as f: f.write(data)
        sig = hmac.new(args.signing_key.encode(), data, hashlib.sha256).hexdigest()
        with open(ckpt + '.hmac', 'w') as f: f.write(sig)
        LOGGER.info("Checkpoint: %s (%.1fMB)", ckpt, len(data)/1024/1024)
        if args.minio_ep:
            try:
                upload(data, 'cadqstream-drift', 'checkpoints/memstream/memstream_memory.pt',
                       args.minio_ep, args.minio_ak, args.minio_sk)
                upload(sig.encode(), 'cadqstream-drift', 'checkpoints/memstream/memstream_memory.pt.hmac',
                       args.minio_ep, args.minio_ak, args.minio_sk)
            except Exception as e:
                LOGGER.warning("MinIO upload failed: %s", e)
    else:
        # Load checkpoint
        state = torch.load(args.eval_only, map_location='cpu', weights_only=False)
        ae = AE(34, 68); ae.load_state_dict(state['ae']); ae.to(DEVICE); ae.eval()
        mean = state['mean'].to(DEVICE); std = state['std'].to(DEVICE)
        mem = Memory(state['mlen'], 34); mem.ld(state['mem'])
        cb = CB(0.5, 95); cb.T = state['cb']['T']; cb.beta = float(state['cb']['beta']); cb.fitted = True
        LOGGER.info("Loaded checkpoint: %s", args.eval_only)

    # ===== EVALUATION =====
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 2: Evaluation (prod.parquet + gt_mask)")
    LOGGER.info("=" * 60)

    gt = np.load(args.gt)
    LOGGER.info("Ground truth: %d samples, %d anomalies (%.2f%%)",
                len(gt), int(gt.sum()), gt.sum() / len(gt) * 100)

    scorer = Scorer(ae, mean, std, mem, cb, 10)
    use_n = min(pq.ParquetFile(args.test).metadata.num_rows, len(gt))
    LOGGER.info("Scoring %d samples...", use_n)

    t0 = time.time(); scores = []; offset = 0
    pf = pq.ParquetFile(args.test)

    for rgi in range(pf.metadata.num_row_groups):
        if offset >= use_n: break
        tbl = pf.read_row_groups([rgi])
        n_t = tbl.num_rows
        if offset + n_t > use_n:
            n_t = use_n - offset
            tbl = tbl.slice(0, n_t)
        if n_t == 0: break
        X_b, h_b, d_b, rc_b, nb_b = extract_features_from_table(tbl)
        sc = scorer.score_ctx(X_b, nb_b, h_b, d_b, rc_b)
        scores.extend(sc.tolist()); offset += len(sc)
        LOGGER.info("  %d / %d (%.0f%%)", offset, use_n, offset / use_n * 100)

    scores = np.array(scores[:use_n], np.float64)
    gt_a = gt[:use_n]
    elapsed = time.time() - t0
    LOGGER.info("Scored in %.1fs (%.0f rec/s)", elapsed, use_n / max(elapsed, 1))

    # Metrics
    pred = (scores > 1.0).astype(int)
    tp = int(((pred == 1) & (gt_a == 1)).sum())
    fp = int(((pred == 1) & (gt_a == 0)).sum())
    tn = int(((pred == 0) & (gt_a == 0)).sum())
    fn = int(((pred == 0) & (gt_a == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0
    acc  = (tp + tn) / len(gt_a)
    try:
        from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
        auc_roc = roc_auc_score(gt_a, scores)
        p, r, _ = precision_recall_curve(gt_a, scores)
        auc_pr = auc(r, p)
    except:
        auc_roc = auc_pr = 0.5
    nm = scores[gt_a == 0]; am = scores[gt_a == 1]

    LOGGER.info("=" * 60)
    LOGGER.info("RESULTS")
    LOGGER.info("=" * 60)
    LOGGER.info("AUC-ROC: %.4f | AUC-PR: %.4f", auc_roc, auc_pr)
    LOGGER.info("Precision: %.4f | Recall: %.4f | F1: %.4f", prec, rec, f1)
    LOGGER.info("TP: %d | FP: %d | TN: %d | FN: %d", tp, fp, tn, fn)
    LOGGER.info("FPR: %.4f | Accuracy: %.4f", fpr, acc)
    LOGGER.info("Score dist - Normal: mean=%.4f, Anomaly: mean=%.4f", nm.mean(), am.mean())

    res = {'n': use_n, 'anom': int(gt_a.sum()),
           'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
           'prec': prec, 'rec': rec, 'f1': f1, 'fpr': fpr, 'acc': acc,
           'auc_roc': auc_roc, 'auc_pr': auc_pr,
           'nm_mean': float(nm.mean()), 'am_mean': float(am.mean()), 'time': elapsed}
    with open(os.path.join(args.out, 'eval_results.json'), 'w') as f:
        json.dump(res, f, indent=2)
    LOGGER.info("Done!")

if __name__ == '__main__':
    sys.exit(main())
