#!/usr/bin/env python3
"""
Exp 13: Warmup Data Size Check (HIGH Priority).
Tests whether warmup_data_size = 50K is sufficient to fill ContextBeta cells.

ContextBeta: 10 neighborhoods × 8 context cells = 80 cells total.
Each cell needs minimum ~11 samples for beta estimation.
Total needed: 80 × 11 = 880 samples minimum.

But distribution is non-uniform:
- Manhattan ~60% of traffic
- JFK ~2% of traffic
- Staten Island ~1% of traffic

Question: Does 50K samples give enough coverage across all 80 cells?
If rare neighborhoods (Staten Island, JFK) get < 11 samples per cell,
ContextBeta falls back to default_beta → threshold noise.

Grid: [50K, 100K, 200K, 500K]
Metric: % ContextBeta cells meeting min threshold (>=11 samples)
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout,
)
LOG = logging.getLogger('exp13-warmup')


# Neighborhood definitions
MANHATTAN = set(range(1, 44))
BRONX = set(range(44, 104))
BROOKLYN = set(range(104, 128))
QUEENS_LOWER = set(range(128, 149))
QUEENS_UPPER = set(range(149, 162))
STATEN_ISLAND = set(range(162, 182))
EWR = set(range(182, 197))
JFK = set(range(217, 230))
NALP = set(range(230, 235))
UNKNOWN = set(range(235, 266)) | set(range(197, 217))

NEIGHBORHOODS = ['manhattan', 'brooklyn', 'queens_lower', 'queens_upper',
                 'bronx', 'staten_island', 'ewr', 'jfk', 'nalp', 'unknown']

CONTEXT_CELLS = [
    'std_day_weekday',    # 0
    'std_night_weekday',  # 1
    'std_day_weekend',    # 2
    'std_night_weekend',  # 3
    'sp_day_weekday',     # 4
    'sp_night_weekday',   # 5
    'sp_day_weekend',     # 6
    'sp_night_weekend',   # 7
]


def loc_to_nb(loc_id):
    if loc_id in MANHATTAN:
        return 0
    elif loc_id in BRONX:
        return 4
    elif loc_id in BROOKLYN:
        return 1
    elif loc_id in QUEENS_LOWER:
        return 2
    elif loc_id in QUEENS_UPPER:
        return 3
    elif loc_id in STATEN_ISLAND:
        return 5
    elif loc_id in EWR:
        return 6
    elif loc_id in JFK:
        return 7
    elif loc_id in NALP:
        return 8
    else:
        return 9


def get_context_bucket(row):
    """Map a trip to a ContextBeta cell (matches memstream_context_beta.py).

    Context cell = (is_special << 2) | (is_night << 1) | is_weekend
    is_special = ratecode > 1
    is_night = hour >= 20  (UNIFIED as per plan)
    is_weekend = dow >= 5

    8 Context Cells:
        0: std_day_weekday  (is_special=0, is_night=0, is_weekend=0)
        1: std_night_weekday (is_special=0, is_night=1, is_weekend=0)
        2: std_day_weekend   (is_special=0, is_night=0, is_weekend=1)
        3: std_night_weekend (is_special=0, is_night=1, is_weekend=1)
        4: sp_day_weekday    (is_special=1, is_night=0, is_weekend=0)
        5: sp_night_weekday  (is_special=1, is_night=1, is_weekend=0)
        6: sp_day_weekend    (is_special=1, is_night=0, is_weekend=1)
        7: sp_night_weekend  (is_special=1, is_night=1, is_weekend=1)
    """
    pickup = pd.to_datetime(row['tpep_pickup_datetime'], errors='coerce')
    if pickup is pd.NaT or pickup is None:
        hour = 12
        dow = 0
    else:
        hour = int(pickup.hour)
        dow = int(pickup.dayofweek) if hasattr(pickup, 'dayofweek') else 0

    ratecode = int(row.get('RatecodeID', 1))
    is_special = 1 if ratecode > 1 else 0
    is_night = 1 if hour >= 20 else 0  # UNIFIED: >= 20
    is_weekend = 1 if dow >= 5 else 0

    cell_id = (is_special << 2) | (is_night << 1) | is_weekend
    return cell_id


def count_context_coverage(df: pd.DataFrame, sample_size: int) -> dict:
    """Count samples per neighborhood × context cell. 10 neighborhoods × 8 cells = 80 cells."""
    # Sample if needed
    if len(df) > sample_size:
        df = df.iloc[:sample_size]  # First N rows (chronological)

    cell_counts = {}
    for _, row in df.iterrows():
        nb = loc_to_nb(int(row['PULocationID']))
        ctx = get_context_bucket(row)  # Returns 0-7
        key = (nb, ctx)
        cell_counts[key] = cell_counts.get(key, 0) + 1

    return cell_counts


def analyze_coverage(cell_counts: dict, min_threshold: int = 11) -> dict:
    """Analyze ContextBeta cell coverage (10 neighborhoods × 8 context cells = 80 total cells)."""
    total_cells = 10 * 8  # 80 cells

    non_zero_cells = len(cell_counts)
    cells_meeting_threshold = sum(1 for v in cell_counts.values() if v >= min_threshold)
    cells_below_threshold = sum(1 for v in cell_counts.values() if 0 < v < min_threshold)
    empty_cells = total_cells - non_zero_cells

    min_samples_in_cell = min(cell_counts.values()) if cell_counts else 0
    max_samples_in_cell = max(cell_counts.values()) if cell_counts else 0
    avg_samples = np.mean(list(cell_counts.values())) if cell_counts else 0

    # Per-neighborhood coverage
    nb_coverage = {}
    for nb_idx, nb_name in enumerate(NEIGHBORHOODS):
        nb_cells = {k: v for k, v in cell_counts.items() if k[0] == nb_idx}
        nb_total = sum(nb_cells.values())
        nb_meeting = sum(1 for v in nb_cells.values() if v >= min_threshold)
        nb_min = min(nb_cells.values()) if nb_cells else 0
        nb_max = max(nb_cells.values()) if nb_cells else 0
        nb_coverage[nb_name] = {
            'total_samples': int(nb_total),
            'cells_with_data': len(nb_cells),
            'cells_meeting_threshold': nb_meeting,
            'min_samples': int(nb_min),
            'max_samples': int(nb_max),
        }

    # Per-context-cell coverage
    ctx_coverage = {}
    for ctx_id in range(8):
        ctx_cells = {k: v for k, v in cell_counts.items() if k[1] == ctx_id}
        ctx_total = sum(ctx_cells.values())
        ctx_meeting = sum(1 for v in ctx_cells.values() if v >= min_threshold)
        ctx_coverage[CONTEXT_CELLS[ctx_id]] = {
            'total_samples': int(ctx_total),
            'neighborhoods_with_data': len(ctx_cells),
            'cells_meeting_threshold': ctx_meeting,
        }

    return {
        'total_cells_in_data': non_zero_cells,
        'cells_meeting_threshold': cells_meeting_threshold,
        'cells_below_threshold': cells_below_threshold,
        'empty_cells': empty_cells,
        'total_cells': total_cells,
        'min_samples_in_cell': int(min_samples_in_cell),
        'max_samples_in_cell': int(max_samples_in_cell),
        'avg_samples_per_cell': float(avg_samples),
        'coverage_pct': 100.0 * cells_meeting_threshold / total_cells,
        'per_neighborhood': nb_coverage,
        'per_context_cell': ctx_coverage,
    }


def main():
    parser = argparse.ArgumentParser(description='Exp 13: Warmup Data Size Check')
    parser.add_argument('--clean', type=str,
                        default='C:/proj/ldt/data/clean/clean_dataset.parquet',
                        help='Clean dataset path')
    parser.add_argument('--polluted', type=str,
                        default='C:/proj/ldt/data/polluted/polluted_dataset.parquet',
                        help='Polluted dataset path')
    parser.add_argument('--output', type=str,
                        default='C:/proj/ldt/results/exp13_warmup_size.json',
                        help='Output results path')
    args = parser.parse_args()

    LOG.info("=" * 60)
    LOG.info("  EXP 13: Warmup Data Size Check")
    LOG.info("  Priority: HIGH  |  Metric: ContextBeta cell coverage")
    LOG.info("=" * 60)

    # Grid
    grid = [50000, 100000, 200000, 500000]

    # Load clean dataset (sorted by time, use first 1M for speed)
    LOG.info(f"\nLoading clean dataset...")
    t0 = time.time()
    clean = pd.read_parquet(args.clean)

    # Sort by datetime
    pickup = pd.to_datetime(clean['tpep_pickup_datetime'], errors='coerce')
    sort_idx = pickup.argsort()
    clean = clean.iloc[sort_idx].reset_index(drop=True)

    LOG.info(f"Loaded {len(clean):,} rows in {time.time()-t0:.1f}s")
    LOG.info(f"Date range: {clean['tpep_pickup_datetime'].min()} to "
             f"{clean['tpep_pickup_datetime'].max()}")

    results = []

    for warmup_size in grid:
        LOG.info(f"\n  [warmup_size={warmup_size:,}]")
        t1 = time.time()

        # Count coverage
        cell_counts = count_context_coverage(clean, warmup_size)
        analysis = analyze_coverage(cell_counts, min_threshold=11)

        elapsed = time.time() - t1
        analysis['warmup_size'] = warmup_size
        analysis['elapsed_s'] = elapsed

        LOG.info(f"    Cells with data:    {analysis['total_cells_in_data']}")
        LOG.info(f"    Cells >= 11 samples: {analysis['cells_meeting_threshold']} "
                 f"({analysis['coverage_pct']:.1f}%)")
        LOG.info(f"    Cells < 11 samples: {analysis['cells_below_threshold']}")
        LOG.info(f"    Avg samples/cell:   {analysis['avg_samples_per_cell']:.1f}")
        LOG.info(f"    Min samples in cell: {analysis['min_samples_in_cell']}")
        LOG.info(f"    Max samples in cell: {analysis['max_samples_in_cell']}")

        # Per-neighborhood summary
        for nb_name, nb_stats in analysis['per_neighborhood'].items():
            LOG.info(f"      {nb_name:15s}: {nb_stats['total_samples']:>8,} samples, "
                     f"{nb_stats['cells_with_data']:>3} cells, "
                     f"min={nb_stats['min_samples']:>5}, "
                     f"max={nb_stats['max_samples']:>6}")

        results.append(analysis)

    # Find best
    best = max(results, key=lambda r: r['cells_meeting_threshold'])

    # Recommendation
    rec = ""
    if best['warmup_size'] >= 500000:
        rec = "500K recommended - largest grid gives best ContextBeta coverage."
    elif best['warmup_size'] >= 200000:
        rec = "200K recommended - good coverage across neighborhoods."
    elif best['warmup_size'] >= 100000:
        rec = "100K is sufficient for ContextBeta cell coverage."
    elif best['warmup_size'] >= 50000:
        rec = "50K MINIMUM - rare neighborhoods may still have insufficient samples."

    # Check if 50K meets minimum requirement
    for r in results:
        if r['warmup_size'] == 50000:
            if r['cells_meeting_threshold'] < 50:  # At least 50% of cells meeting threshold
                rec = "50K INSUFFICIENT - at least 50% of cells below threshold. Need >= 100K."
                LOG.warning(f"\n  WARNING: 50K is INSUFFICIENT for ContextBeta coverage!")
                LOG.warning(f"  Only {r['cells_meeting_threshold']} cells have >= 11 samples")

    # Save results
    output = {
        'experiment': 'exp13_warmup_data_size',
        'hyperparameter': 'warmup_data_size',
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        'priority': 'HIGH',
        'primary_metric': 'cells_meeting_threshold',
        'grid': grid,
        'results': results,
        'best_config': best,
        'recommendation': rec,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    LOG.info(f"\n{'='*60}")
    LOG.info(f"  RESULTS:")
    LOG.info(f"  {'warmup_size':>12}  {'cells_data':>10}  {'cells_>=11':>10}  "
             f"{'coverage%':>10}  {'avg/cell':>10}")
    LOG.info(f"  {'-'*60}")
    for r in results:
        LOG.info(f"  {r['warmup_size']:>12,}  {r['total_cells_in_data']:>10,}  "
                 f"{r['cells_meeting_threshold']:>10,}  "
                 f"{r['coverage_pct']:>9.1f}%  "
                 f"{r['avg_samples_per_cell']:>10.1f}")
    LOG.info(f"\n  BEST: warmup_size={best['warmup_size']:,}  "
             f"cells_>=11={best['cells_meeting_threshold']}")
    LOG.info(f"\n  RECOMMENDATION: {rec}")
    LOG.info(f"\n  Saved: {output_path}")
    LOG.info(f"  DONE!")


if __name__ == '__main__':
    main()
