#!/usr/bin/env python3
"""
Exp 07: warmup_epochs
Priority: MEDIUM
Primary Metric: AE loss convergence
"""

import sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from shared import load_data, GPUExperimentModel, compute_all_metrics

OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)


def run():
    ts = time.strftime('%Y%m%d_%H%M%S')
    print("EXP 07: warmup_epochs")
    print("Loading data...")
    data = load_data(n_warmup=10000, n_test=15000)

    grid = [5, 10, 20, 50]
    results = []

    for epochs in grid:
        t0 = time.time()
        print(f"  epochs={epochs}")

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
            epochs=epochs, batch_size=256
        )

        scores, latencies = model.score_streaming(
            data['X_test'],
            neighborhood_ids=data['nb_test'],
            hour_vals=data['hr_test'],
            dow_vals=data['dw_test'],
            ratecode_vals=data['rc_test'],
        )
        m = compute_all_metrics(data['y_test'], scores, latency_per_record_ms=latencies)
        m['elapsed_s'] = time.time() - t0

        print(f"    AUC-ROC={m['AUC-ROC']:.4f} AUC-PR={m['AUC-PR']:.4f} "
              f"F1={m['F1']:.4f} Prec={m['Precision']:.4f} Rec={m['Recall']:.4f}")
        print(f"    Threshold={m['threshold']:.1f} TP={m['TP']} "
              f"Elapsed: {m['elapsed_s']:.1f}s")
        results.append({'warmup_epochs': epochs, 'metrics': m})

    best = max(results, key=lambda r: r['metrics']['AUC-ROC'])

    output = {
        'experiment': 'exp07_warmup_epochs',
        'hyperparameter': 'warmup_epochs',
        'timestamp': ts,
        'priority': 'MEDIUM',
        'grid': grid,
        'results': results,
        'best_config': best,
        'recommendation': f"{best['warmup_epochs']} epochs gives best AUC-ROC.",
    }
    out_path = OUT / f'exp07_warmup_epochs_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    run()
