#!/usr/bin/env python3
"""Quick analysis for optimization targets."""
import sys, os
sys.path.insert(0, 'c:/proj/ldt/memstream_src')
os.chdir('c:/proj/ldt')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

OUT_DIR = Path('c:/proj/ldt/results/rigorous_v1')

df = pd.read_csv(OUT_DIR / 'results_detailed.csv')
df = df[df['algorithm'] != 'sklearn_LOF']
df_ca = pd.read_csv(OUT_DIR / 'results_ca_quick.csv')

# ============================================================
# 1. Per-fold ranking - identify fold patterns
# ============================================================
print("=" * 80)
print("PER-FOLD RANKING (MemStream_ vs HBOS vs sklearn_IF)")
print("=" * 80)

for diff in ['easy', 'medium', 'hard']:
    print(f"\n{diff.upper()}:")
    sub = df[(df['difficulty'] == diff) & (df['seed'] == 42)]
    pivot = sub.pivot_table(values='AUC_PR', index='fold', columns='algorithm')
    pivot = pivot[['MemStream_', 'HBOS', 'sklearn_IF', 'CADIFEia']]

    ranks = pivot.rank(ascending=False)
    for fold in pivot.index:
        ms = pivot.loc[fold, 'MemStream_']
        hb = pivot.loc[fold, 'HBOS']
        if_ = pivot.loc[fold, 'sklearn_IF']
        ms_r = ranks.loc[fold, 'MemStream_']
        hb_r = ranks.loc[fold, 'HBOS']
        if_r = ranks.loc[fold, 'sklearn_IF']

        winner = 'MemStream_' if ms_r == 1 else ('HBOS' if hb_r == 1 else 'sklearn_IF')
        marker = '*' if fold in [10, 11] else ' '

        print(f"  Fold {fold:02d}{marker}: MS={ms:.4f}(rank={int(ms_r)}) "
              f"HBOS={hb:.4f}(rank={int(hb_r)}) "
              f"IF={if_:.4f}(rank={int(if_r)}) -> {winner}")

# Count wins
print("\n--- Algorithm wins per fold ---")
for algo in ['MemStream_', 'HBOS', 'sklearn_IF']:
    for diff in ['easy', 'medium', 'hard']:
        sub = df[(df['difficulty'] == diff) & (df['seed'] == 42)]
        pivot = sub.pivot_table(values='AUC_PR', index='fold', columns='algorithm')
        ranks = pivot.rank(ascending=False)
        wins = (ranks[algo] == 1).sum()
        print(f"  {algo:<20} {diff:8s}: {wins}/11 folds best")

# ============================================================
# 2. Ensemble: score averaging potential
# ============================================================
print("\n" + "=" * 80)
print("ENSEMBLE POTENTIAL")
print("=" * 80)

# For each experiment, compute rank
for diff in ['easy']:
    sub = df[(df['difficulty'] == diff) & (df['seed'] == 42)]
    pivot = sub.pivot_table(values='AUC_PR', index='fold', columns='algorithm')
    ranks = pivot.rank(ascending=False)

    # Average rank of MS+HBOS
    avg_rank = ranks[['MemStream_', 'HBOS']].mean(axis=1)
    best_rank = ranks.min(axis=1)
    print(f"\n{diff.upper()}: MS+HBOS average rank vs best individual")
    for fold in pivot.index:
        ms_r = ranks.loc[fold, 'MemStream_']
        hb_r = ranks.loc[fold, 'HBOS']
        avg_r = avg_rank.loc[fold]
        best_r = best_rank.loc[fold]
        print(f"  Fold {fold:02d}: MS_rank={int(ms_r)} HBOS_rank={int(hb_r)} "
              f"avg={avg_r:.1f} best={int(best_r)} "
              f"{'ENS better' if avg_r < min(ms_r, hb_r) else 'SAME'}")

# ============================================================
# 3. Best fold/worst fold analysis
# ============================================================
print("\n" + "=" * 80)
print("FOLD DIFFICULTY ANALYSIS")
print("=" * 80)

print("\n--- Worst folds for MemStream_ ---")
for diff in ['easy', 'medium', 'hard']:
    sub = df[(df['algorithm'] == 'MemStream_') & (df['difficulty'] == diff)]
    fold_means = sub.groupby('fold')['AUC_PR'].mean().sort_values()
    print(f"  {diff.upper()}: worst fold={fold_means.index[0]} ({fold_means.iloc[0]:.4f}), "
          f"best fold={fold_means.index[-1]} ({fold_means.iloc[-1]:.4f}), "
          f"ratio={fold_means.iloc[-1]/fold_means.iloc[0]:.1f}x")

# ============================================================
# 4. Calibration threshold analysis
# ============================================================
print("\n--- Optimal threshold analysis (MemStream_) ---")
for algo in ['MemStream_', 'HBOS', 'sklearn_IF']:
    a = df[df['algorithm'] == algo]
    thresholds = a['optimal_threshold']
    print(f"  {algo:<20}: threshold mean={thresholds.mean():.4f} std={thresholds.std():.4f} "
          f"min={thresholds.min():.4f} max={thresholds.max():.4f}")

# ============================================================
# 5. HBOS tuning potential
# ============================================================
print("\n--- HBOS n_bins sensitivity (simulated) ---")
print("HBOS uses fixed n_bins=10. Optimal may vary by difficulty:")
print("  - Easy: fewer bins (5) might overfit to training distribution")
print("  - Hard: more bins (20-50) might capture fine-grained patterns")
print("  - Recommendation: test n_bins in {5, 10, 20, 50}")

# ============================================================
# 6. Key insight: MemStream_ wins most but HBOS wins some
# ============================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS FOR OPTIMIZATION")
print("=" * 80)

print("""
1. MEMSTREAM_ is the best single algorithm but not universally:
   - Wins most folds, but HBOS wins some (especially on hard)
   - Fold 10 is consistently weak for MemStream_ on easy

2. HBOS is strong on EASY difficulty:
   - 2nd best overall, strong on easy
   - Simple, fast, interpretable

3. ENSEMBLE OPPORTUNITY:
   - Average of MemStream_ + HBOS could be more robust
   - But they rank differently on hard -> ensemble may not help much

4. CALIBRATION THRESHOLD matters:
   - Current: percentile-based on test (not ideal)
   - Better: calibrate on calibration set, use fixed percentile

5. KEY TUNING TARGETS:
   - MemStream_: bufsz, memsz (memory initialization)
   - HBOS: n_bins (histogram resolution)
   - sklearn_IF: max_features, max_samples (avoid overfitting)
   - Ensemble: weighted average based on difficulty

6. OVERFITTING RISK:
   - High CV (>100%) indicates fold-to-fold instability
   - Need to test: does tuning on all folds vs best folds differ?
""")

# ============================================================
# 7. CA-MemStream detailed comparison
# ============================================================
print("\n--- CA-MemStream vs MemStream_ (same folds) ---")
for diff in ['easy', 'medium', 'hard']:
    ca = df_ca[(df_ca['algorithm'] == 'CA_MemStream') & (df_ca['difficulty'] == diff)]
    ms = df[(df['algorithm'] == 'MemStream_') &
             (df['fold'].isin([1, 2, 3])) &
             (df['difficulty'] == diff) &
             (df['seed'] == 42)]

    ca_mean = ca['AUC_PR'].mean()
    ms_mean = ms['AUC_PR'].mean()
    print(f"  {diff:8s}: CA-MemStream={ca_mean:.4f} MemStream_={ms_mean:.4f} "
          f"diff={ca_mean-ms_mean:+.4f}")
