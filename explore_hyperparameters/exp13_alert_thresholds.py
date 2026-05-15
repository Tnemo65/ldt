#!/usr/bin/env python3
"""
Exp 13: Alert Thresholds (Multiple Thresholds)
Priority: LOW
Primary Metric: False positive rate

Rationale: Different alert levels can use different thresholds for different severity levels.
This experiment tests a grid of alert thresholds to find optimal operating points.
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


def compute_fpr_grid(y_true, scores, thresholds):
    """Compute FPR at multiple thresholds."""
    results = {}
    for name, thresh in thresholds.items():
        yp = (scores > thresh).astype(int)
        tp = int(((yp == 1) & (y_true == 1)).sum())
        fp = int(((yp == 1) & (y_true == 0)).sum())
        tn = int(((yp == 0) & (y_true == 0)).sum())
        fn = int(((yp == 0) & (y_true == 1)).sum())
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        results[name] = {
            'threshold': float(thresh), 'FPR': float(fpr),
            'Precision': float(prec), 'Recall': float(rec), 'F1': float(f1),
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        }
    return results


def run():
    ts = time.strftime('%Y%m%d_%H%M%S')
    print("=" * 60)
    print("  EXP 13: Alert Thresholds")
    print("  Priority: LOW  |  Metric: False positive rate")
    print("=" * 60)

    print("\nLoading data...")
    data = load_data(n_warmup=10000, n_test=15000)

    print("\nTraining model...")
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

    scores, latencies = model.score_streaming(
        data['X_test'],
        neighborhood_ids=data['nb_test'],
        hour_vals=data['hr_test'],
        dow_vals=data['dw_test'],
        ratecode_vals=data['rc_test'],
    )

    m_base = compute_all_metrics(data['y_test'], scores, latency_per_record_ms=latencies)

    # Alert levels
    p50  = float(np.percentile(scores, 50))
    p75  = float(np.percentile(scores, 75))
    p90  = float(np.percentile(scores, 90))
    p95  = float(np.percentile(scores, 95))
    p99  = float(np.percentile(scores, 99))

    thresholds = {
        'LOW': p75,       # Sensitive - catch more
        'MEDIUM': p90,
        'HIGH': p95,
        'CRITICAL': p99,
    }

    print(f"\n  Thresholds:")
    for name, thresh in thresholds.items():
        print(f"    {name}: {thresh:.2f}")

    fpr_results = compute_fpr_grid(data['y_test'], scores, thresholds)

    print(f"\n  Alert Level Results:")
    print(f"  {'Level':>10}  {'Thresh':>10}  {'FPR':>8}  "
          f"{'Prec':>8}  {'Rec':>8}  {'F1':>8}  {'TP':>5}  {'FP':>5}")
    print(f"  {'-'*70}")
    for name, r in fpr_results.items():
        print(f"  {name:>10}  {r['threshold']:>10.2f}  "
              f"{r['FPR']:>8.4f}  {r['Precision']:>8.4f}  "
              f"{r['Recall']:>8.4f}  {r['F1']:>8.4f}  "
              f"{r['TP']:>5}  {r['FP']:>5}")

    # Optimal threshold for each alert level
    optimal_thresholds = {}
    for target_fpr in [0.01, 0.05, 0.10, 0.20]:
        norm_scores = np.array(scores)[data['y_test'] == 0]
        t = float(np.percentile(norm_scores, (1 - target_fpr) * 100))
        optimal_thresholds[f'FPR@{int(target_fpr*100)}'] = t

    print(f"\n  Optimal thresholds for target FPR:")
    for name, t in optimal_thresholds.items():
        yp = (np.array(scores) > t).astype(int)
        tp = int(((yp == 1) & (data['y_test'] == 1)).sum())
        fp = int(((yp == 1) & (data['y_test'] == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + (data['y_test'] == 1).sum() - tp) if (data['y_test'] == 1).sum() > 0 else 0
        print(f"    {name}: thresh={t:.2f}  Prec={prec:.4f}  Rec={rec:.4f}")

    output = {
        'experiment': 'exp13_alert_thresholds',
        'hyperparameter': 'alert_thresholds',
        'timestamp': ts,
        'priority': 'LOW',
        'primary_metric': 'FPR',
        'alert_levels': fpr_results,
        'optimal_thresholds': optimal_thresholds,
        'base_metrics': {k: v for k, v in m_base.items() if isinstance(v, (int, float))},
        'recommendation': "Use LOW for sensitive monitoring, HIGH for production alerts.",
    }
    out_path = OUT / f'exp13_alert_thresholds_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    run()
