#!/usr/bin/env python3
"""Best Config Experiment with progress tracking."""
import sys, json, time, warnings, os
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
OUT  = ROOT / 'results'
OUT.mkdir(exist_ok=True)

# Progress file
PROGRESS_FILE = OUT / 'best_config_progress.json'
sys.path.insert(0, str(ROOT))
from shared import load_data, GPUExperimentModel, compute_all_metrics


def log(msg):
    """Write to both console and progress file."""
    print(msg)
    try:
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE) as f:
                lines = f.readlines()
        else:
            lines = []
        lines.append(msg + '\n')
        with open(PROGRESS_FILE, 'w') as f:
            f.writelines(lines[-1000:])  # Keep last 1000 lines
    except Exception:
        pass


class GPUContextBeta:
    def __init__(self, n_neighborhoods=10, n_cells=8, percentile=95):
        self.n_neighborhoods = n_neighborhoods
        self.n_cells = n_cells
        self.percentile = percentile
        self.betas = np.ones((n_neighborhoods, n_cells), dtype=np.float32) * 0.5
        self.non_default_count = 0

    def fit(self, scores, nb_ids, ctx_ids):
        self.non_default_count = 0
        for n in range(self.n_neighborhoods):
            for c in range(self.n_cells):
                cell_scores = [s for s, nm, ctx in zip(scores, nb_ids, ctx_ids)
                              if nm == n and ctx == c]
                if len(cell_scores) >= 50:
                    self.betas[n, c] = float(np.percentile(cell_scores, self.percentile))
                    self.non_default_count += 1

    def get_beta(self, nb_id, ctx_id):
        n = min(max(int(nb_id), 0), self.n_neighborhoods - 1)
        c = min(max(int(ctx_id), 0), self.n_cells - 1)
        return float(self.betas[n, c])


def score_with_cb(model, X_test, nb_ids, hr_vals, dw_vals, rc_vals, percentile=95):
    """Batch score with ContextBeta normalization."""
    from shared.data_loader import get_context_id
    Xs = model.scaler.transform(X_test.astype(np.float64)).astype(np.float32)
    n = len(Xs)
    Xt = torch.from_numpy(Xs.astype(np.float32)).to(model.device)
    with torch.no_grad():
        Z = F.relu(Xt @ model._W1 + model._b1)
    raw = model._score_batch_raw(Z)
    cb = GPUContextBeta(percentile=percentile)
    ctx_ids = np.array([get_context_id(int(hr), int(dw), float(rc))
                        for hr, dw, rc in zip(hr_vals, dw_vals, rc_vals)])
    cb.fit(raw, nb_ids, ctx_ids)
    scores = np.array([raw[i] / max(cb.get_beta(nb_ids[i], ctx_ids[i]), 1e-6)
                       for i in range(n)], dtype=np.float64)
    return scores


def run():
    ts = time.strftime('%Y%m%d_%H%M%S')
    # Clear progress file
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    log("=" * 70)
    log("  BEST CONFIG EXPERIMENT")
    log("  Grid search best: mem=128, k=20, gamma=0.9, latent=34, warmup=30K")
    log("  Hyperparam best: k=50, epochs=100")
    log("=" * 70)

    log("\nLoading data (30K warmup, 15K test)...")
    data = load_data(n_warmup=30000, n_test=15000)
    log(f"  Warmup: {data['n_warmup']:,}  Test: {data['n_test']:,}  Anomalies: {data['y_test'].sum():,}")

    configs = [
        {'name': 'grid_best_ld34', 'memory_len': 128, 'k': 20, 'gamma': 0.9, 'latent_dim': 34, 'warmup_epochs': 20, 'use_cb': True},
        {'name': 'grid_best_ld60', 'memory_len': 128, 'k': 20, 'gamma': 0.9, 'latent_dim': 60, 'warmup_epochs': 20, 'use_cb': True},
        {'name': 'hyp_best_k50_ep100_noCB', 'memory_len': 256, 'k': 50, 'gamma': 0.0, 'latent_dim': 60, 'warmup_epochs': 100, 'use_cb': False},
        {'name': 'hyp_best_k50_ep100_CB', 'memory_len': 256, 'k': 50, 'gamma': 0.0, 'latent_dim': 60, 'warmup_epochs': 100, 'use_cb': True},
        {'name': 'hybrid_k50_g09_ld60', 'memory_len': 256, 'k': 50, 'gamma': 0.9, 'latent_dim': 60, 'warmup_epochs': 100, 'use_cb': True},
        {'name': 'large_k100_g09', 'memory_len': 256, 'k': 100, 'gamma': 0.9, 'latent_dim': 60, 'warmup_epochs': 100, 'use_cb': True},
        {'name': 'grid_best_ep50', 'memory_len': 128, 'k': 20, 'gamma': 0.9, 'latent_dim': 34, 'warmup_epochs': 50, 'use_cb': True},
        {'name': 'grid_best_ep100', 'memory_len': 128, 'k': 20, 'gamma': 0.9, 'latent_dim': 34, 'warmup_epochs': 100, 'use_cb': True},
    ]

    results = []
    for idx, cfg in enumerate(configs):
        t0 = time.time()
        log(f"\n[{idx+1}/{len(configs)}] {cfg['name']}")
        log(f"    mem={cfg['memory_len']} k={cfg['k']} gamma={cfg['gamma']} ld={cfg['latent_dim']} ep={cfg['warmup_epochs']}")

        model = GPUExperimentModel(
            memory_len=cfg['memory_len'], k=cfg['k'], gamma=cfg['gamma'],
            latent_dim=cfg['latent_dim'], default_beta=0.5, seed=42, device='cuda')

        model.fit(data['X_warmup'],
                  neighborhood_ids=data['nb_warmup'],
                  hour_vals=data['hr_warmup'],
                  dow_vals=data['dw_warmup'],
                  ratecode_vals=data['rc_warmup'],
                  epochs=cfg['warmup_epochs'], batch_size=256)

        scores_stream, lat = model.score_streaming(
            data['X_test'], neighborhood_ids=data['nb_test'],
            hour_vals=data['hr_test'], dow_vals=data['dw_test'],
            ratecode_vals=data['rc_test'])

        if cfg['use_cb']:
            scores = score_with_cb(model, data['X_test'],
                                  data['nb_test'], data['hr_test'],
                                  data['dw_test'], data['rc_test'], percentile=95)
            lat_stream = lat
        else:
            scores = scores_stream
            lat_stream = lat

        m = compute_all_metrics(data['y_test'], scores, latency_per_record_ms=lat_stream)
        m['elapsed_s'] = time.time() - t0
        m['cb_used'] = cfg['use_cb']

        anom_mask = data['y_test'] == 1
        norm_mask = data['y_test'] == 0
        m['anom_mean'] = float(scores[anom_mask].mean())
        m['norm_mean'] = float(scores[norm_mask].mean())
        m['sep_ratio'] = m['anom_mean'] / max(m['norm_mean'], 0.01)

        log(f"    AUC-ROC={m['AUC-ROC']:.4f} AUC-PR={m['AUC-PR']:.4f} F1={m['F1']:.4f} Prec={m['Precision']:.4f} Rec={m['Recall']:.4f}")
        log(f"    TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']} Sep={m['sep_ratio']:.2f}x CB={cfg['use_cb']}")
        log(f"    Latency: mean={m['latency_mean_ms']:.2f}ms p99={m['latency_p99_ms']:.2f}ms {m['elapsed_s']:.1f}s")

        results.append({'config': cfg, 'metrics': m})

    results.sort(key=lambda r: r['metrics']['F1'], reverse=True)

    log(f"\n{'='*70}")
    log(f"  RESULTS (sorted by F1)")
    log(f"  {'Name':<30} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>6} {'Prec':>6} {'Rec':>6} {'TP':>5} {'Sep':>6} {'CB':>3}")
    log(f"  {'-'*70}")
    for r in results:
        m = r['metrics']
        cfg = r['config']
        name = cfg['name'][:30]
        log(f"  {name:<30} {m['AUC-ROC']:8.4f} {m['AUC-PR']:8.4f} {m['F1']:6.4f} {m['Precision']:6.4f} {m['Recall']:6.4f} {m['TP']:5} {m['sep_ratio']:6.2f}x {str(cfg['use_cb']):>3}")

    best = results[0]
    log(f"\n  BEST: {best['config']['name']}")
    log(f"    AUC-ROC={best['metrics']['AUC-ROC']:.4f} F1={best['metrics']['F1']:.4f} Prec={best['metrics']['Precision']:.4f} Rec={best['metrics']['Recall']:.4f}")
    log(f"    Sep={best['metrics']['sep_ratio']:.2f}x (anom={best['metrics']['anom_mean']:.1f} norm={best['metrics']['norm_mean']:.1f})")

    output = {
        'experiment': 'best_config_combined',
        'timestamp': ts,
        'data': {'warmup': data['n_warmup'], 'test': data['n_test'], 'anomalies': int(data['y_test'].sum())},
        'results': results,
        'best_config': best,
    }
    out_path = OUT / f'best_config_combined_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\n  Saved: {out_path}")
    log("  DONE")

    # Remove progress file marker
    if PROGRESS_FILE.exists():
        pass  # Keep for debugging


if __name__ == '__main__':
    run()
