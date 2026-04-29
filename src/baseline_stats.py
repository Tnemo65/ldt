"""
Step 1: Compute baseline statistics per context group
====================================================
Trainable baseline: computes mean, std, p5, p95, counts per context group.

Fixes applied:
  - NaN counts tracked explicitly per column
  - Uses shared utils (pipeline_utils.py)
  - VECTORIZED context assignment (no row-by-row apply)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time

import pipeline_utils as pu

DATA_DIR = Path('data/raw')
CLEAN_DIR = Path('data')
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

BASELINE_COLS = ['fare_amount', 'trip_distance', 'total_amount', 'passenger_count']

# ── Load Jan 2024 ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1a: Loading Jan 2024")
print("=" * 60)

df = pd.read_parquet(DATA_DIR / 'yellow_tripdata_2024-01.parquet')
print(f"Loaded: {len(df):,} rows, {df.shape[1]} cols")
print(f"Columns: {list(df.columns)}")

# ── VECTORIZED context assignment (NO apply()) ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1b: VECTORIZED context assignment")
print("=" * 60)

t0 = time.time()

# Vectorized: no row-by-row apply()
hours = df['tpep_pickup_datetime'].dt.hour
ratecodes = df['RatecodeID']

time_bin = pd.cut(
    hours,
    bins=[-1, 5, 11, 17, 24],
    labels=['night', 'morning', 'afternoon', 'evening']
).astype(str)

ratecode_bin = np.where(ratecodes == 1, 'standard', 'special')
df['context'] = time_bin + '_' + ratecode_bin

print(f"Context assigned in {time.time() - t0:.2f}s (vectorized)")
print(f"\nContext distribution:")
print(df['context'].value_counts().sort_index())

# ── Compute baseline stats (with NaN tracking) ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1c: Computing baseline statistics")
print("=" * 60)

t0 = time.time()
baseline = pu.compute_baseline_stats(df, 'context', BASELINE_COLS)
print(f"Computed in {time.time() - t0:.2f}s")

# ── Print summary (with NaN warnings) ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1d: Baseline summary (fare_amount)")
print("=" * 60)

pu.print_baseline_summary(baseline, 'fare_amount')

# Check for high NULL columns
print("\n\nNULL analysis per context per column:")
print("-" * 70)
any_warnings = False
for ctx in sorted(baseline.keys()):
    s = baseline[ctx]
    for col in BASELINE_COLS:
        col_stats = s.get(col, {})
        n_null = col_stats.get('n_null', 0)
        n_total = s['n_total']
        null_pct = n_null / n_total * 100 if n_total > 0 else 0
        if null_pct > 1:
            print(f"  {ctx:<22} / {col:<20}: {n_null:,} null ({null_pct:.1f}%)")
            any_warnings = True

if not any_warnings:
    print("  All columns have <1% NULL — OK")

# ── Validate context separation ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1e: Context separation validation")
print("=" * 60)

fare_means = {ctx: baseline[ctx].get('fare_amount', {}).get('mean', 0)
              for ctx in baseline}

print("\nFare by time_bin (standard vs special):")
for hb in ['night', 'morning', 'afternoon', 'evening']:
    std = fare_means.get(f'{hb}_standard', 0) or 0
    spc = fare_means.get(f'{hb}_special', 0) or 0
    ratio = spc / std if std > 0 else 0
    print(f"  {hb:<12}: standard=${std:6.2f}, special=${spc:6.2f} ({ratio:.1f}x)")

# Group size check
min_n = min(s['n_total'] for s in baseline.values())
max_n = max(s['n_total'] for s in baseline.values())
print(f"\nGroup sizes: {min_n:,} - {max_n:,}")
if min_n < 50_000:
    print("  NOTE: Some groups are small (night_special). OK for thesis purposes.")
else:
    print("  OK: All groups have sufficient data.")

# ── Save ─────────────────────────────────────────────────────────────────────
output_path = OUTPUT_DIR / 'baseline_stats.json'
with open(output_path, 'w') as f:
    json.dump(baseline, f, indent=2)

print(f"\nSaved: {output_path}")
print(f"Groups: {len(baseline)}")
print("\nDONE: Step 1 complete")
