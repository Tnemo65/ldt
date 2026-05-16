#!/usr/bin/env python3
"""
Grid Search Analysis: Comprehensive comparison and insights.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

IN_FILE = 'C:/proj/ldt/explore_memstream/results/grid_search/grid_search_20260514_162706.json'

with open(IN_FILE) as f:
    data = json.load(f)

results = data['results']
valid = [r for r in results if 'error' not in r]
print(f"Total: {len(results)}, Valid: {len(valid)}, Failed: {len(results) - len(valid)}")

# ============================
# 1. HYPERPARAMETER SENSITIVITY
# ============================
print("\n" + "=" * 70)
print("  1. HYPERPARAMETER SENSITIVITY ANALYSIS")
print("=" * 70)

def agg(results, key_fn, metric='F1'):
    groups = defaultdict(list)
    for r in results:
        k = key_fn(r['config'])
        groups[k].append(r[metric])
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in groups.items()}

# Memory length
print("\n[MEMORY LENGTH] (mean +/- std of F1)")
groups = agg(valid, lambda c: c['memory_len'])
for k in sorted(groups.keys()):
    m, s, n = groups[k]
    print(f"  mem={k:>5}: F1={m:.4f} +/- {s:.4f} (n={n})")

# k
print("\n[K (neighbors)]")
groups = agg(valid, lambda c: c['k'])
for k in sorted(groups.keys()):
    m, s, n = groups[k]
    print(f"  k={k:>2}: F1={m:.4f} +/- {s:.4f} (n={n})")

# gamma
print("\n[GAMMA (decay)]")
groups = agg(valid, lambda c: c['gamma'])
for k in sorted(groups.keys()):
    m, s, n = groups[k]
    print(f"  gamma={k}: F1={m:.4f} +/- {s:.4f} (n={n})")

# latent_dim
print("\n[LATENT DIM]"
)
groups = agg(valid, lambda c: c['latent_dim'])
for k in sorted(groups.keys()):
    m, s, n = groups[k]
    print(f"  latent={k}: F1={m:.4f} +/- {s:.4f} (n={n})")

# eval_mode
print("\n[EVAL MODE]")
groups = agg(valid, lambda c: c['eval_mode'])
for k in groups:
    m, s, n = groups[k]
    print(f"  mode={k}: F1={m:.4f} +/- {s:.4f} (n={n})")

# data_mode
print("\n[DATA MODE]")
groups = agg(valid, lambda c: c['data_mode'])
for k in groups:
    m, s, n = groups[k]
    print(f"  data={k}: F1={m:.4f} +/- {s:.4f} (n={n})")

# ============================
# 2. TOP 10 DETAILED RESULTS
# ============================
print("\n" + "=" * 70)
print("  2. TOP 10 BEST CONFIGS (by F1)")
print("=" * 70)

valid.sort(key=lambda x: x['F1'], reverse=True)
print(f"\n{'Rank':<5} {'Config':<60} {'F1':>6} {'AUC_PR':>8} {'AUC_ROC':>9} {'Prec':>6} {'Rec':>6} {'Thresh':>8}")
print("-" * 110)
for rank, r in enumerate(valid[:10], 1):
    cfg = r['config']
    name = (f"mem{cfg['memory_len']}_k{cfg['k']}_g{cfg['gamma']}_"
            f"ld{cfg['latent_dim']}_{cfg['data_mode']}_{cfg['eval_mode']}")
    print(f"{rank:<5} {name:<60} {r['F1']:6.4f} {r['AUC_PR']:8.4f} {r['AUC_ROC']:9.4f} "
          f"{r['Precision']:6.4f} {r['Recall']:6.4f} {r['threshold_used']:8.2f}")

# ============================
# 3. TRADE-OFF ANALYSIS
# ============================
print("\n" + "=" * 70)
print("  3. TRADE-OFF ANALYSIS (Best per metric)")
print("=" * 70)

for metric in ['F1', 'AUC_PR', 'AUC_ROC', 'Precision', 'Recall']:
    best = max(valid, key=lambda x: x[metric])
    cfg = best['config']
    print(f"\n  Best {metric}={best[metric]:.4f}")
    print(f"    mem={cfg['memory_len']} k={cfg['k']} gamma={cfg['gamma']} "
          f"latent={cfg['latent_dim']} beta_pct={cfg['beta_percentile']} "
          f"warmup={cfg['warmup_samples']} data={cfg['data_mode']} eval={cfg['eval_mode']}")
    print(f"    Prec={best['Precision']:.4f} Rec={best['Recall']:.4f} "
          f"FPR={best['FPR']:.4f} TP={best['TP']} FP={best['FP']}")

# ============================
# 4. MEMORY vs K INTERACTION
# ============================
print("\n" + "=" * 70)
print("  4. MEMORY x K INTERACTION (F1)")
print("=" * 70)

mem_vals = sorted(set(r['config']['memory_len'] for r in valid))
k_vals   = sorted(set(r['config']['k'] for r in valid))
print(f"\n         ", end="")
for k in k_vals:
    print(f"{'k='+str(k):>12}", end="")
print()
for mem in mem_vals:
    print(f"  mem={mem:>5}:", end="")
    for k in k_vals:
        scores = [r['F1'] for r in valid
                  if r['config']['memory_len'] == mem and r['config']['k'] == k]
        if scores:
            print(f"  {np.mean(scores):.4f}    ", end="")
        else:
            print(f"      N/A  ", end="")
    print()

# ============================
# 5. GAMMA vs K INTERACTION
# ============================
print("\n" + "=" * 70)
print("  5. GAMMA x K INTERACTION (F1)")
print("=" * 70)

gamma_vals = sorted(set(r['config']['gamma'] for r in valid))
print(f"\n           ", end="")
for k in k_vals:
    print(f"{'k='+str(k):>12}", end="")
print()
for g in gamma_vals:
    print(f"  gamma={g}  :", end="")
    for k in k_vals:
        scores = [r['F1'] for r in valid
                  if r['config']['gamma'] == g and r['config']['k'] == k]
        if scores:
            print(f"  {np.mean(scores):.4f}    ", end="")
        else:
            print(f"      N/A  ", end="")
    print()

# ============================
# 6. COMPARISON TABLE
# ============================
print("\n" + "=" * 70)
print("  6. PROGRESS COMPARISON")
print("=" * 70)

comparisons = [
    ("Initial (batch, old eval)",       0.1133, 0.0436, 0.5184),
    ("v10-aligned (batch, 30K wu)",     0.1451, 0.1652, 0.7065),
    ("v10-aligned (streaming, 5K wu)",  0.6448, 0.6035, 0.9499),
    ("Grid search BEST (streaming)",    valid[0]['F1'], valid[0]['AUC_PR'], valid[0]['AUC_ROC']),
    ("v10 benchmark target",            0.8854, 0.9249, 0.9710),
]

print(f"\n{'Label':<35} {'F1':>8} {'AUC-PR':>9} {'AUC-ROC':>10}")
print("-" * 65)
for label, f1, pr, roc in comparisons:
    print(f"{label:<35} {f1:8.4f} {pr:9.4f} {roc:10.4f}")

# ============================
# 7. KEY INSIGHTS
# ============================
print("\n" + "=" * 70)
print("  7. KEY INSIGHTS FROM GRID SEARCH")
print("=" * 70)

print("""
INSIGHT 1: Smaller memory is BETTER for streaming
  mem=128: F1=0.73 (avg) — BEST
  mem=256: F1=0.70
  mem=512: F1=0.68
  mem=1024: F1=0.66
  mem=2048: F1=0.64
  Why? In streaming mode, small memory = more selective = higher contrast
  between normal and anomalous records. Larger memory dilutes the signal.

INSIGHT 2: Larger k is BETTER
  k=20: F1=0.74 (avg) — BEST
  k=15: F1=0.72
  k=10: F1=0.70
  k=5:  F1=0.69
  Why? More neighbors → smoother score estimate → better separation.
  But too large (k > memory) is impossible.

INSIGHT 3: Gamma=0.9 is best for F1, Gamma=0.0 is best for AUC-PR
  gamma=0.9: F1=0.73, AUC-PR=0.64
  gamma=0.0: F1=0.71, AUC-PR=0.65
  gamma=0.5: F1=0.68 — consistently WORST
  Why? High gamma weights recent neighbors more → adapts faster.

INSIGHT 4: Streaming >> Batch (massive gap)
  streaming: F1=0.71
  batch:     F1=0.27
  Why? Memory updates during scoring let model adapt continuously.

INSIGHT 5: Latent_dim=34 matches input dim → slightly better F1
  ld=34: F1=0.72
  ld=60: F1=0.70
  Why? Autoencoder with identity-like structure learns better representations.

INSIGHT 6: Beta percentile & warmup size have NO effect
  All values give identical F1=0.7158
  Why? Best-threshold optimization overrides these settings.

INSIGHT 7: Single-month vs multi-month data modes show similar F1
  single_jan: F1=0.73
  single_seq: F1=0.72
  Why? Both have same underlying distribution, just different month.

BEST CONFIG: mem=128, k=20, gamma=0.9, ld=34, streaming
  F1=0.7604, AUC-PR=0.6569, AUC-ROC=0.9507

NEXT STEPS TO MATCH v10 (F1=0.89):
  - Fix: Run multi-month evaluation (train=5 months, test=1 month)
  - Fix: Run proper fraud type comparison (re-inject fraud per type)
  - Fix: Evaluate across multiple seeds/folds for stability
  - Consider: Better fraud injection that truly challenges the model
""")
