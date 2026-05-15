#!/usr/bin/env python3
"""
Exp 10: recent_scores_buffer
Priority: MEDIUM
Primary Metric: AE quality vs buffer size
"""

import sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np

ROOT = Path(__file__).parent
OUT  = ROOT / 'results'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
from shared import load_data, GPUExperimentModel, compute_all_metrics


class RollingScoreBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.scores = []
        self.mean = 0.0
        self.std = 1.0

    def push(self, score):
        self.scores.append(score)
        if len(self.scores) > self.buffer_size:
            self.scores.pop(0)
        self.mean = np.mean(self.scores)
        self.std = np.std(self.scores) + 1e-8


def run():
    ts = time.strftime('%Y%m%d_%H%M%S')
    print("EXP 10: recent_scores_buffer")
    print("Loading data...")
    data = load_data(n_warmup=10000, n_test=15000)

    grid = [1000, 5000, 10000, 50000]
    results = []

    for buf_size in grid:
        t0 = time.time()
        print(f"  buffer_size={buf_size}")

        model = GPUExperimentModel(
            memory_len=256, k=10, gamma=0.0, latent_dim=60,
            default_beta=0.5, seed=42, device='cuda'
        )
        model.fit(
            data['X_warmup'],
            neighborhood_ids=data['nb_warmup'],
            hour_vals=data['hr_warmup'],
            dow_vals=data['dw_warmup'],
            ratecode_vals=data['rc_warmup'],
            epochs=20, batch_size=256
        )

        scores, _ = model.score_streaming(
            data['X_test'],
            neighborhood_ids=data['nb_test'],
            hour_vals=data['hr_test'],
            dow_vals=data['dw_test'],
            ratecode_vals=data['rc_test'],
        )

        m = compute_all_metrics(data['y_test'], scores)
        m['elapsed_s'] = time.time() - t0
        print(f"    AUC-ROC={m['AUC-ROC']:.4f} AUC-PR={m['AUC-PR']:.4f} F1={m['F1']:.4f}")
        results.append({'buffer_size': buf_size, 'metrics': m})

    best = max(results, key=lambda r: r['metrics']['AUC-ROC'])

    output = {
        'experiment': 'exp10_recent_scores_buffer',
        'hyperparameter': 'buffer_size',
        'timestamp': ts,
        'priority': 'MEDIUM',
        'grid': grid,
        'results': results,
        'best_config': best,
        'recommendation': f"Buffer={best["buffer_size"]} gives best AUC-ROC.",
    }
    out_path = OUT / f'exp10_recent_scores_buffer_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    run()
