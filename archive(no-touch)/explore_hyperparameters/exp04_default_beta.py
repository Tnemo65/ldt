#!/usr/bin/env python3
"""
Exp 04: default_beta (Anomaly Threshold)
Priority: MEDIUM
Primary Metric: FPR on clean validation set

Rationale: default_beta sets the fallback anomaly threshold for context cells
with insufficient warmup data. Lower beta = more sensitive (higher recall, higher FPR).
Higher beta = more conservative (lower recall, lower FPR).
"""

import sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from shared import load_data, GPUExperimentModel, compute_all_metrics, compute_fpr_at_threshold

OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)


def run():
    ts = time.strftime('%Y%m%d_%H%M%S')
    print("=" * 60)
    print("  EXP 04: default_beta (Anomaly Threshold)")
    print("  Priority: MEDIUM  |  Metric: FPR on clean val set")
    print("=" * 60)

    print("\nLoading data...")
    data = load_data(n_warmup=10000, n_test=15000)
    print(f"  Warmup: {data['n_warmup']:,}  Test: {data['n_test']:,}  "
          f"Anomalies: {data['n_anomalies']:,}")

    grid = [0.3, 0.5, 0.7]
    results = []

    for beta in grid:
        t0 = time.time()
        print(f"\n  [default_beta={beta}]")

        model = GPUExperimentModel(
            memory_len=256, k=10, gamma=0.0, latent_dim=60,
            default_beta=beta, seed=42, device='cuda'
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

        # Compute FPR at specific thresholds
        fpr_results = {}
        for target_fpr in [0.01, 0.05, 0.10]:
            thresh, actual_fpr, fpr_m = compute_fpr_at_threshold(
                data['y_test'], scores, target_fpr=target_fpr)
            fpr_results[f'FPR@{int(target_fpr*100)}'] = {
                'threshold': thresh,
                'actual_fpr': actual_fpr,
                'precision': fpr_m['Precision'],
                'recall': fpr_m['Recall'],
                'f1': fpr_m['F1'],
            }

        m['fpr_grid'] = fpr_results

        # Beta distribution
        if model.cb is not None:
            betas = model.cb.betas
            m['beta_nondefault_count'] = model.cb.non_default_count
            m['beta_used_default_count'] = 80 - model.cb.non_default_count

        elapsed = time.time() - t0
        m['elapsed_s'] = elapsed

        print(f"    AUC-ROC={m['AUC-ROC']:.4f}  F1={m['F1']:.4f}")
        for k, v in fpr_results.items():
            print(f"    {k}: thresh={v['threshold']:.2f}  actual_fpr={v['actual_fpr']:.4f}  "
                  f"P={v['precision']:.4f}  R={v['recall']:.4f}")

        results.append({'default_beta': beta, 'metrics': m})

    best = max(results, key=lambda r: r['metrics']['AUC-ROC'])

    print(f"\n{'='*60}")
    print(f"  RESULTS:")
    print(f"  {'beta':>6}  {'AUC-ROC':>8}  {'F1':>8}  {'DefaultCnt':>10}  {'NonDefault':>10}")
    print(f"  {'-'*60}")
    for r in results:
        m = r['metrics']
        print(f"  {r['default_beta']:>6.1f}  {m['AUC-ROC']:8.4f}  {m['F1']:8.4f}  "
              f"{m.get('beta_used_default_count', 0):>10}  {m.get('beta_nondefault_count', 0):>10}")

    output = {
        'experiment': 'exp04_default_beta',
        'hyperparameter': 'default_beta',
        'timestamp': ts,
        'priority': 'MEDIUM',
        'primary_metric': 'FPR',
        'grid': grid,
        'results': results,
        'best_config': best,
        'recommendation': f"default_beta={best["default_beta"]} is optimal for AUC-ROC.",
    }
    out_path = OUT / f'exp04_default_beta_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    run()
