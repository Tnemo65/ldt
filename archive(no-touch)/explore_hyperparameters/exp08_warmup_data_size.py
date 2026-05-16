#!/usr/bin/env python3
"""
Exp 08: warmup_data_size
Priority: HIGH
Primary Metric: % context cells >= min threshold

Rationale: warmup_data_size determines how many records are used to fit the model.
More data = better ContextBeta estimates = more context cells covered.
NYC taxi has 8 context cells. Need sufficient samples per cell for reliable thresholds.
Critical: ContextBeta only fits cells with >= 50 samples by default.
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


def analyze_context_coverage(X, nb_ids, hr_vals, dw_vals, rc_vals, min_samples=50):
    """Analyze how many context cells are well-covered."""
    ctx_ids = np.array([get_context_id(int(h), int(d), int(r))
                       for h, d, r in zip(hr_vals, dw_vals, rc_vals)])
    from collections import Counter
    counts = Counter(zip(nb_ids, ctx_ids))
    total_cells = 80  # 10 neighborhoods * 8 cells
    covered = sum(1 for v in counts.values() if v >= min_samples)
    return {
        'total_cells': total_cells,
        'cells_covered': covered,
        'coverage_pct': covered / total_cells * 100,
        'per_cell_counts': dict(counts),
        'min_count': min(counts.values()) if counts else 0,
        'max_count': max(counts.values()) if counts else 0,
        'mean_count': np.mean(list(counts.values())) if counts else 0,
    }


def run():
    ts = time.strftime('%Y%m%d_%H%M%S')
    print("=" * 60)
    print("  EXP 08: warmup_data_size")
    print("  Priority: HIGH  |  Metric: % context cells >= min threshold")
    print("=" * 60)

    # Load large dataset for this experiment
    print("\nLoading full NYC taxi dataset...")
    import pandas as pd
    df_raw = pd.read_parquet('C:/proj/ldt/data/nyc_taxi_300k.parquet')
    from shared.data_loader import clean
    df_clean = clean(df_raw)

    grid = [50000, 100000, 200000, 500000]
    results = []

    for wu_size in grid:
        t0 = time.time()
        print(f"\n  [warmup_data_size={wu_size:,}]")

        if len(df_clean) < wu_size:
            print(f"    WARNING: dataset has {len(df_clean):,} records, "
                  f"using all available")
            wu_size_actual = len(df_clean)
        else:
            wu_size_actual = wu_size

        df_warmup = df_clean.iloc[:wu_size_actual].reset_index(drop=True)
        from shared.data_loader import extract_features

        X_warmup = extract_features(df_warmup)

        nb_ids = np.array([
            0 for _ in range(len(df_warmup))
        ], dtype=int)
        hr_vals = X_warmup[:, 9].astype(int)
        dw_vals = X_warmup[:, 10].astype(int)
        rc_vals = X_warmup[:, 25].astype(int)

        # Context coverage analysis
        cov = analyze_context_coverage(X_warmup, nb_ids, hr_vals, dw_vals, rc_vals)
        print(f"    Dataset size: {wu_size_actual:,}")
        print(f"    Context cells covered: {cov['cells_covered']}/{cov['total_cells']} "
              f"({cov['coverage_pct']:.1f}%)")
        print(f"    Count range: {cov['min_count']} - {cov['max_count']}  "
              f"(mean: {cov['mean_count']:.0f})")

        m = {
            'warmup_size_used': wu_size_actual,
            'warmup_data_size_requested': wu_size,
            'context_coverage': {
                'total_cells': cov['total_cells'],
                'cells_covered': cov['cells_covered'],
                'coverage_pct': cov['coverage_pct'],
                'min_count': cov['min_count'],
                'max_count': cov['max_count'],
                'mean_count': cov['mean_count'],
            },
            'elapsed_s': time.time() - t0,
        }
        results.append({'warmup_data_size': wu_size, 'metrics': m})

        print(f"    Elapsed: {m['elapsed_s']:.1f}s")

    best = max(results, key=lambda r: r['metrics']['context_coverage']['coverage_pct'])

    print(f"\n{'='*60}")
    print(f"  RESULTS:")
    print(f"  {'wu_size':>10}  {'used':>10}  {'cells':>6}  {'coverage%':>10}  "
          f"{'min_ct':>7}  {'max_ct':>7}  {'mean_ct':>8}")
    print(f"  {'-'*60}")
    for r in results:
        m = r['metrics']
        cov = m['context_coverage']
        print(f"  {r['warmup_data_size']:>10,}  {m['warmup_size_used']:>10,}  "
              f"{cov['cells_covered']:>6}  {cov['coverage_pct']:>10.1f}  "
              f"{cov['min_count']:>7}  {cov['max_count']:>7}  {cov['mean_count']:>8.0f}")

    output = {
        'experiment': 'exp08_warmup_data_size',
        'hyperparameter': 'warmup_data_size',
        'timestamp': ts,
        'priority': 'HIGH',
        'primary_metric': 'Context_coverage_pct',
        'grid': grid,
        'results': results,
        'best_config': best,
        'recommendation': f"{best['warmup_data_size']:,} samples gives "
                          f"{best['metrics']['context_coverage']['coverage_pct']:.1f}% context coverage.",
    }
    out_path = OUT / f'exp08_warmup_data_size_{ts}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    run()
