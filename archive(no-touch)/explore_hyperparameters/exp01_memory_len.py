#!/usr/bin/env python3
"""
Exp 01: Memory Length
Priority: HIGH
Primary Metric: AUC-ROC, Latency P99

Rationale: memory_len controls how many encoded normal samples are stored.
Larger memory = more normal patterns captured = better discrimination.
BUT: larger memory = higher latency (kNN search over larger buffer).
Tests range from 10K (production scale) down to 100K.
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
    print("  EXP 01: Memory Length")
    print("  Priority: HIGH  |  Metric: AUC-ROC, Latency P99")
    print("=" * 60)

    print("\nLoading data...")
    data = load_data(n_warmup=10000, n_test=15000)
    print(f"  Warmup: {data['n_warmup']:,}  Test: {data['n_test']:,}  "
          f"Anomalies: {data['n_anomalies']:,}")

    # Grid: values from 10K to 100K
    grid = [10000, 25000, 50000, 100000]
    results = []

    for mem_len in grid:
        t0 = time.time()
        print(f"\n  [memory_len={mem_len:,}]")

        model = GPUExperimentModel(
            memory_len=mem_len, k=10, gamma=0.0, latent_dim=60,
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

        # Streaming score (with latency tracking)
        scores, latencies = model.score_streaming(
            data['X_test'],
            neighborhood_ids=data['nb_test'],
            hour_vals=data['hr_test'],
            dow_vals=data['dw_test'],
            ratecode_vals=data['rc_test'],
        )

        m = compute_all_metrics(data['y_test'], scores, latency_per_record_ms=latencies)

        # ContextBeta coverage
        if model.cb is not None:
            m['context_cells_covered'] = model.cb.non_default_count
            m['context_cells_total']  = 80

        elapsed = time.time() - t0
        m['elapsed_s'] = elapsed

        print(f"    AUC-ROC={m['AUC-ROC']:.4f}  AUC-PR={m['AUC-PR']:.4f}  "
              f"F1={m['F1']:.4f}  Prec={m['Precision']:.4f}  Rec={m['Recall']:.4f}")
        print(f"    Latency: mean={m['latency_mean_ms']:.2f}ms  "
              f"p50={m['latency_p50_ms']:.2f}ms  p90={m['latency_p90_ms']:.2f}ms  "
              f"p99={m['latency_p99_ms']:.2f}ms")
        print(f"    Threshold={m['threshold']:.1f}  TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")
        print(f"    ContextBeta cells: {m.get('context_cells_covered',0)}/80")

        results.append({'memory_len': mem_len, 'metrics': m})

    # Find best
    best = max(results, key=lambda r: r['metrics']['AUC-ROC'])

    print(f"\n{'='*60}")
    print(f"  RESULTS:")
    print(f"  {'memory_len':>12}  {'AUC-ROC':>8}  {'AUC-PR':>8}  {'F1':>8}  "
          f"{'Precision':>9}  {'Recall':>8}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'LatP99ms':>10}")
    print(f"  {'-'*60}")
    for r in results:
        m = r['metrics']
        print(f"  {r['memory_len']:>12,}  {m['AUC-ROC']:8.4f}  {m['AUC-PR']:8.4f}  "
              f"{m['F1']:8.4f}  {m['Precision']:9.4f}  {m['Recall']:8.4f}  "
              f"{m['TP']:5}  {m['FP']:5}  {m['FN']:5}  {m.get('latency_p99_ms',0):10.2f}")
    print(f"\n  BEST: memory_len={best['memory_len']:,}  "
          f"AUC-ROC={best['metrics']['AUC-ROC']:.4f}  F1={best['metrics']['F1']:.4f}")

    # Recommendation
    rec = "memory_len should be sized to capture at least one full daily cycle of normal traffic."
    if best['memory_len'] == 10000:
        rec = "10K is sufficient for this dataset. Larger memory shows diminishing returns."
    elif best['memory_len'] == 100000:
        rec = "100K recommended - model benefits from more historical normal patterns."

    # Save
    output = {
        'experiment': 'exp01_memory_len',
        'hyperparameter': 'memory_len',
        'timestamp': ts,
        'priority': 'HIGH',
        'primary_metric': 'AUC-ROC',
        'grid': grid,
        'results': results,
        'best_config': best,
        'recommendation': rec,
    }
    out_path = OUT / f'exp01_memory_len_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    run()
