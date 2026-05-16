#!/usr/bin/env python3
"""
Exp 02: k (Number of Neighbors)
Priority: HIGH
Primary Metric: AUC-ROC, P@K, R@K

Rationale: k controls how many nearest neighbors contribute to the anomaly score.
Small k (5): tight boundary, high precision, low recall.
Large k (50): smooth boundary, lower precision, higher recall.
Optimal k balances separation quality and noise robustness.
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
    print("=" * 60)
    print("  EXP 02: k (Number of Neighbors)")
    print("  Priority: HIGH  |  Metric: AUC-ROC, P@K, R@K")
    print("=" * 60)

    print("\nLoading data...")
    data = load_data(n_warmup=10000, n_test=15000)
    print(f"  Warmup: {data['n_warmup']:,}  Test: {data['n_test']:,}  "
          f"Anomalies: {data['n_anomalies']:,}")

    # Grid: k values
    grid = [5, 10, 20, 30, 50]
    results = []

    for k in grid:
        t0 = time.time()
        print(f"\n  [k={k}]")

        model = GPUExperimentModel(
            memory_len=256, k=k, gamma=0.0, latent_dim=60,
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

        scores, latencies = model.score_streaming(
            data['X_test'],
            neighborhood_ids=data['nb_test'],
            hour_vals=data['hr_test'],
            dow_vals=data['dw_test'],
            ratecode_vals=data['rc_test'],
        )

        m = compute_all_metrics(data['y_test'], scores, latency_per_record_ms=latencies)
        elapsed = time.time() - t0
        m['elapsed_s'] = elapsed

        print(f"    AUC-ROC={m['AUC-ROC']:.4f}  AUC-PR={m['AUC-PR']:.4f}  "
              f"F1={m['F1']:.4f}  Prec={m['Precision']:.4f}  Rec={m['Recall']:.4f}")
        for kk in [10, 50, 100, 500]:
            print(f"    P@{kk}={m.get(f'P@{kk}',0):.4f}  R@{kk}={m.get(f'R@{kk}',0):.4f}")

        results.append({'k': k, 'metrics': m})

    best = max(results, key=lambda r: r['metrics']['AUC-ROC'])

    print(f"\n{'='*60}")
    print(f"  RESULTS:")
    print(f"  {'k':>5}  {'AUC-ROC':>8}  {'AUC-PR':>8}  {'F1':>8}  "
          f"{'P@50':>7}  {'R@50':>7}  {'P@100':>7}  {'R@100':>7}")
    print(f"  {'-'*60}")
    for r in results:
        m = r['metrics']
        print(f"  {r['k']:>5}  {m['AUC-ROC']:8.4f}  {m['AUC-PR']:8.4f}  "
              f"{m['F1']:8.4f}  "
              f"{m.get('P@50',0):7.4f}  {m.get('R@50',0):7.4f}  "
              f"{m.get('P@100',0):7.4f}  {m.get('R@100',0):7.4f}")
    print(f"\n  BEST: k={best['k']}  AUC-ROC={best['metrics']['AUC-ROC']:.4f}")

    rec = (f"k={best['k']} is optimal for this dataset. "
           f"Smaller k gives tighter boundaries but noisier scores; "
           f"larger k smooths but may dilute the anomaly signal.")

    output = {
        'experiment': 'exp02_k_neighbors',
        'hyperparameter': 'k',
        'timestamp': ts,
        'priority': 'HIGH',
        'primary_metric': 'AUC-ROC',
        'grid': grid,
        'results': results,
        'best_config': best,
        'recommendation': rec,
    }
    out_path = OUT / f'exp02_k_neighbors_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    run()
