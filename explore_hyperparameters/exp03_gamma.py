#!/usr/bin/env python3
"""
Exp 03: gamma (kNN Decay)
Priority: HIGH
Primary Metric: Score contamination, AUC-ROC

Rationale: gamma controls the decay of neighbor importance in kNN scoring.
gamma=0: Uniform weights (all k neighbors contribute equally).
gamma>0: Recent neighbors weighted more heavily.
High gamma = faster adaptation to new patterns = less "contamination" from old data.
But too high gamma may overfit to recent noise.
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


def score_contamination(scores_normal, scores_anomaly):
    """Compute score contamination metrics.

    Contamination: % of normal samples with score > median anomaly score.
    High contamination = normal samples get anomalously high scores = model confusion.
    """
    median_anom = np.median(scores_anomaly)
    pct_above = (scores_normal > median_anom).mean() * 100
    pct_below = (scores_normal < median_anom).mean() * 100
    return {
        'contamination_pct': float(pct_above),
        'clean_pct': float(pct_below),
        'separation_mean_gap': float(scores_anomaly.mean() - scores_normal.mean()),
        'separation_median_gap': float(median_anom - np.median(scores_normal)),
    }


def run():
    ts = time.strftime('%Y%m%d_%H%M%S')
    print("=" * 60)
    print("  EXP 03: gamma (kNN Decay)")
    print("  Priority: HIGH  |  Metric: AUC-ROC, Score contamination")
    print("=" * 60)

    print("\nLoading data...")
    data = load_data(n_warmup=10000, n_test=15000)
    print(f"  Warmup: {data['n_warmup']:,}  Test: {data['n_test']:,}  "
          f"Anomalies: {data['n_anomalies']:,}")

    # Grid
    grid = [0.0, 0.3, 0.5, 0.7, 0.9]
    results = []

    for gamma in grid:
        t0 = time.time()
        label = "uniform" if gamma == 0.0 else f"decay-{gamma}"
        print(f"\n  [gamma={gamma}] ({label})")

        model = GPUExperimentModel(
            memory_len=256, k=10, gamma=gamma, latent_dim=60,
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

        m = compute_all_metrics(data['y_test'], scores)

        # Score contamination analysis
        norm_scores = np.array(scores)[data['y_test'] == 0]
        anom_scores = np.array(scores)[data['y_test'] == 1]
        cont = score_contamination(norm_scores, anom_scores)
        m.update(cont)

        elapsed = time.time() - t0
        m['elapsed_s'] = elapsed

        print(f"    AUC-ROC={m['AUC-ROC']:.4f}  AUC-PR={m['AUC-PR']:.4f}  "
              f"F1={m['F1']:.4f}")
        print(f"    Contamination: {cont['contamination_pct']:.2f}%  "
              f"Separation gap: {cont['separation_mean_gap']:.1f}")

        results.append({'gamma': gamma, 'metrics': m})

    best = max(results, key=lambda r: r['metrics']['AUC-ROC'])
    best_contamination = min(results, key=lambda r: r['metrics']['contamination_pct'])

    print(f"\n{'='*60}")
    print(f"  RESULTS:")
    print(f"  {'gamma':>8}  {'AUC-ROC':>8}  {'AUC-PR':>8}  {'F1':>8}  "
          f"{'Contamin%':>10}  {'Sep Gap':>10}")
    print(f"  {'-'*60}")
    for r in results:
        m = r['metrics']
        print(f"  {r['gamma']:>8.2f}  {m['AUC-ROC']:8.4f}  {m['AUC-PR']:8.4f}  "
              f"{m['F1']:8.4f}  {m['contamination_pct']:10.2f}  "
              f"{m['separation_mean_gap']:10.1f}")
    print(f"\n  BEST AUC-ROC: gamma={best['gamma']} ({best['metrics']['AUC-ROC']:.4f})")
    print(f"  BEST contamination: gamma={best_contamination['gamma']} "
          f"({best_contamination['metrics']['contamination_pct']:.2f}%)")

    rec = (f"gamma={best['gamma']} gives best AUC-ROC. "
           f"Note: lower contamination ({best_contamination['metrics']['contamination_pct']:.1f}%) "
           f"at gamma={best_contamination['gamma']}.")

    output = {
        'experiment': 'exp03_gamma',
        'hyperparameter': 'gamma',
        'timestamp': ts,
        'priority': 'HIGH',
        'primary_metric': 'AUC-ROC',
        'grid': grid,
        'results': results,
        'best_config': best,
        'best_contamination_config': best_contamination,
        'recommendation': rec,
    }
    out_path = OUT / f'exp03_gamma_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    run()
