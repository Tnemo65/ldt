#!/usr/bin/env python3
"""
Exp 09: cell_minimum (Minimum samples per context cell)
Priority: LOW
Primary Metric: Beta estimate variance

Rationale: cell_minimum sets the minimum number of warmup samples required
before ContextBeta fits a threshold for a context cell.
Low cell_minimum = more cells fitted but with unreliable thresholds.
High cell_minimum = fewer cells fitted but more reliable estimates.
"""

import sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from shared import load_data, GPUExperimentModel, compute_all_metrics
from shared.data_loader import get_context_id

OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)


def fit_context_beta_custom(scores, nb_ids, ctx_ids, cell_min=50):
    """Fit ContextBeta with custom minimum samples per cell."""
    betas = np.ones((10, 8), dtype=np.float32) * 0.5
    fitted_cells = []
    for n in range(10):
        for c in range(8):
            cell_scores = [
                s for s, nm, ctx in zip(scores, nb_ids, ctx_ids)
                if nm == n and ctx == c
            ]
            if len(cell_scores) >= cell_min:
                betas[n, c] = float(np.percentile(cell_scores, 95))
                fitted_cells.append((n, c, len(cell_scores)))
    non_default = (betas != 0.5).sum()
    fitted_betas = betas[betas != 0.5]
    variance = float(fitted_betas.std()) if len(fitted_betas) > 1 else 0.0
    return betas, non_default, variance, fitted_cells


def run():
    ts = time.strftime('%Y%m%d_%H%M%S')
    print("=" * 60)
    print("  EXP 09: cell_minimum")
    print("  Priority: LOW  |  Metric: Beta estimate variance")
    print("=" * 60)

    print("\nLoading data...")
    data = load_data(n_warmup=10000, n_test=15000)
    print(f"  Warmup: {data['n_warmup']:,}  Test: {data['n_test']:,}")

    grid = [5, 10, 15, 25, 50]
    results = []

    for cell_min in grid:
        t0 = time.time()
        print(f"\n  [cell_minimum={cell_min}]")

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

        m = compute_all_metrics(data['y_test'], scores, latency_per_record_ms=latencies)

        # Fit with custom cell minimum
        ctx_ids_warmup = np.array([
            get_context_id(int(h), int(d), int(r))
            for h, d, r in zip(data['hr_warmup'], data['dw_warmup'], data['rc_warmup'])
        ])

        # Get warmup scores
        wu_t = model.scaler.transform(data['X_warmup'].astype(np.float64)).astype(np.float32)
        import torch
        Xt = torch.from_numpy(wu_t).to('cuda')
        with torch.no_grad():
            Z = torch.nn.functional.relu(Xt @ model._W1 + model._b1)
        raw_scores = model._score_batch_raw(Z)
        betas, non_default, variance, fitted_cells = fit_context_beta_custom(
            raw_scores, data['nb_warmup'], ctx_ids_warmup, cell_min=cell_min)

        m['non_default_betas'] = non_default
        m['beta_variance'] = variance
        m['fitted_cells'] = len(fitted_cells)
        m['elapsed_s'] = time.time() - t0

        print(f"    AUC-ROC={m['AUC-ROC']:.4f}  F1={m['F1']:.4f}")
        print(f"    Fitted cells: {non_default}/80  "
              f"Variance: {variance:.4f}")

        results.append({'cell_minimum': cell_min, 'metrics': m})

    best = max(results, key=lambda r: r['metrics']['AUC-ROC'])

    print(f"\n{'='*60}")
    print(f"  RESULTS:")
    print(f"  {'cell_min':>10}  {'AUC-ROC':>9}  {'F1':>8}  "
          f"{'fitted':>7}  {'variance':>10}")
    print(f"  {'-'*50}")
    for r in results:
        m = r['metrics']
        print(f"  {r['cell_minimum']:>10}  {m['AUC-ROC']:>9.4f}  {m['F1']:>8.4f}  "
              f"{m['non_default_betas']:>7}  {m['beta_variance']:>10.4f}")

    output = {
        'experiment': 'exp09_cell_minimum',
        'hyperparameter': 'cell_minimum',
        'timestamp': ts,
        'priority': 'LOW',
        'primary_metric': 'Beta_variance',
        'grid': grid,
        'results': results,
        'best_config': best,
        'recommendation': f"cell_minimum={best["cell_minimum"]} balances reliability vs coverage.",
    }
    out_path = OUT / f'exp09_cell_minimum_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    run()
