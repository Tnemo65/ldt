#!/usr/bin/env python3
"""Statistical analysis of benchmark results."""
import sys, os
sys.path.insert(0, 'c:/proj/ldt/memstream_src')
os.chdir('c:/proj/ldt')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from itertools import combinations

OUT_DIR = Path('c:/proj/ldt/results/rigorous_v1')
DIFFICULTIES = ['easy', 'medium', 'hard']
METRIC = 'AUC_PR'

print("Loading results...")
df = pd.read_csv(OUT_DIR / 'results_detailed.csv')
print(f"Loaded {len(df)} results, {df['algorithm'].nunique()} algorithms")

# ============================================================
# 1. SUMMARY STATISTICS
# ============================================================
print("\n=== SUMMARY (AUC-PR) ===")
summary = df.groupby('algorithm').agg(
    mean=('AUC_PR', 'mean'),
    std=('AUC_PR', 'std'),
    median=('AUC_PR', 'median'),
    min_=('AUC_PR', 'min'),
    max_=('AUC_PR', 'max'),
    n=('AUC_PR', 'count'),
).round(4)
summary['cv'] = (summary['std'] / (summary['mean'] + 1e-10) * 100).round(1)
summary = summary.sort_values('mean', ascending=False)
print(summary.to_string())
summary.to_csv(OUT_DIR / 'summary_auc_pr.csv')

# ============================================================
# 2. BY DIFFICULTY
# ============================================================
print("\n=== BY DIFFICULTY ===")
by_diff = df.groupby(['algorithm', 'difficulty'])['AUC_PR'].agg(['mean', 'std']).round(4)
print(by_diff.to_string())
by_diff.to_csv(OUT_DIR / 'summary_by_difficulty.csv')

# ============================================================
# 3. WILCOXON SIGNED-RANK TESTS
# ============================================================
print("\n=== WILCOXON SIGNED-RANK TESTS (pairwise, vs sklearn_IF) ===")

algos = df['algorithm'].unique()
control = 'sklearn_IF'

# Pairwise tests for each difficulty
wilcoxon_results = []

for diff in DIFFICULTIES:
    df_diff = df[df['difficulty'] == diff]

    for algo in algos:
        if algo == control:
            continue

        ctrl_vals = df_diff[df_diff['algorithm'] == control]['AUC_PR'].values
        algo_vals = df_diff[df_diff['algorithm'] == algo]['AUC_PR'].values

        if len(ctrl_vals) < 3 or len(algo_vals) < 3:
            continue

        try:
            # Wilcoxon signed-rank (two-sided)
            d = algo_vals - ctrl_vals
            valid = ~np.isnan(d)
            if valid.sum() < 3 or np.all(d[valid] == 0):
                p_val = 1.0
                stat = np.nan
            else:
                wr = stats.wilcoxon(d[valid], alternative='two-sided')
                stat, p_val = float(wr.statistic), float(wr.pvalue)

            # Effect size: Cliff's delta
            def cliffs_delta(x, y):
                n1, n2 = len(x), len(y)
                more = sum(1 for a in x for b in y if a > b)
                less = sum(1 for a in x for b in y if a < b)
                return (more - less) / (n1 * n2)

            cd = cliffs_delta(algo_vals, ctrl_vals)

            # Significance
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''

            wilcoxon_results.append({
                'difficulty': diff,
                'algorithm': algo,
                'control': control,
                'ctrl_mean': ctrl_vals.mean(),
                'algo_mean': algo_vals.mean(),
                'improvement': algo_vals.mean() - ctrl_vals.mean(),
                'wilcoxon_stat': stat,
                'p_value': p_val,
                'cliffs_delta': cd,
                'sig': sig,
            })
        except Exception as e:
            wilcoxon_results.append({
                'difficulty': diff, 'algorithm': algo, 'control': control,
                'ctrl_mean': ctrl_vals.mean(), 'algo_mean': algo_vals.mean(),
                'improvement': algo_vals.mean() - ctrl_vals.mean(),
                'wilcoxon_stat': np.nan, 'p_value': np.nan, 'cliffs_delta': np.nan, 'sig': 'ERR'
            })

df_wilcoxon = pd.DataFrame(wilcoxon_results)
print(df_wilcoxon.to_string(index=False))
df_wilcoxon.to_csv(OUT_DIR / 'wilcoxon_tests.csv', index=False)

# ============================================================
# 4. HOLM-BONFERRONI CORRECTION
# ============================================================
print("\n=== HOLM-BONFERRONI CORRECTION (overall) ===")

# Pool all comparisons across difficulties for Holm
holm_data = []
for diff in DIFFICULTIES:
    sub = df_wilcoxon[df_wilcoxon['difficulty'] == diff].copy()
    for _, row in sub.iterrows():
        holm_data.append({
            'difficulty': diff,
            'algorithm': row['algorithm'],
            'p_value': row['p_value']
        })

df_holm = pd.DataFrame(holm_data)
if not df_holm.empty:
    pvals = df_holm['p_value'].values
    # Sort by p-value
    idx = np.argsort(pvals)
    sorted_p = pvals[idx]
    m = len(sorted_p)
    Holm_critical = {}
    for rank, (original_idx, p) in enumerate(zip(idx, sorted_p), 1):
        adj = min(p * (m - rank + 1), 1.0)
        Holm_critical[original_idx] = adj

    df_holm['holm_adj_p'] = [Holm_critical.get(i, np.nan) for i in range(len(df_holm))]
    df_holm['holm_sig'] = df_holm['holm_adj_p'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '')

    print(df_holm.to_string(index=False))
    df_holm.to_csv(OUT_DIR / 'holm_bonferroni.csv', index=False)

# ============================================================
# 5. RANKING TABLE
# ============================================================
print("\n=== RANKING TABLE ===")
ranking_data = []
for diff in DIFFICULTIES:
    sub = df[df['difficulty'] == diff]
    means = sub.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False)
    for rank, (algo, val) in enumerate(means.items(), 1):
        std = sub[sub['algorithm'] == algo]['AUC_PR'].std()
        ranking_data.append({
            'Difficulty': diff,
            'Rank': rank,
            'Algorithm': algo,
            'AUC-PR': f'{val:.4f} ± {std:.4f}',
            'AUC-PR_mean': val,
            'AUC-PR_std': std,
        })

df_rank = pd.DataFrame(ranking_data)
print(df_rank.to_string(index=False))
df_rank.to_csv(OUT_DIR / 'ranking_table.csv', index=False)

# ============================================================
# 6. CRITICAL DIFFERENCE DIAGRAMS
# ============================================================
print("\n=== CRITICAL DIFFERENCE DIAGRAMS ===")

def nemenyi_cd(k, n_alpha):
    """Nemenyi critical difference at alpha=0.05."""
    if k <= 2:
        return 0.0
    q_alpha = {
        3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268
    }
    q = q_alpha.get(k, 3.0)
    return q * np.sqrt((k * (k + 1)) / (6 * n_alpha))

def plot_cd_diagram(df, diff, out_path):
    """Plot Critical Difference diagram."""
    sub = df[df['difficulty'] == diff]
    if sub.empty:
        return

    algos = sub['algorithm'].unique()
    k = len(algos)

    # Average ranks
    avg_ranks = {}
    for algo in algos:
        vals = sub[sub['algorithm'] == algo]['AUC_PR'].values
        ranks = []
        for _, row in sub.iterrows():
            ctrl_vals = sub[sub['algorithm'] == algo]['AUC_PR'].values
            # Actually compute ranks per fold-seed
            pass

    # Compute per-fold ranks
    folds = sub[['fold', 'seed']].drop_duplicates().apply(tuple, axis=1).tolist()
    rank_data = []
    for fold, seed in folds:
        fold_data = sub[(sub['fold'] == fold) & (sub['seed'] == seed)]
        ranks = fold_data['AUC_PR'].rank(ascending=False).values
        for algo, r in zip(fold_data['algorithm'].values, ranks):
            rank_data.append({'fold': fold, 'seed': seed, 'algorithm': algo, 'rank': r})

    df_ranks = pd.DataFrame(rank_data)
    avg_ranks = df_ranks.groupby('algorithm')['rank'].mean().sort_values()

    n_alpha = len(folds)
    cd = nemenyi_cd(k, n_alpha)

    # Plot
    fig, ax = plt.subplots(figsize=(max(10, k * 1.5), 4))
    ax.set_xlim(0, k + 1)
    ax.set_ylim(-1.5, 2.5)

    # Draw CD bar
    y_bar = 1.0
    ax.plot([0.5, k + 0.5], [y_bar, y_bar], 'k-', linewidth=0.5)
    x_left = 0.5
    x_right = k + 0.5
    ax.plot([x_left, x_left], [y_bar - 0.1, y_bar], 'k-', linewidth=0.5)
    ax.plot([x_right, x_right], [y_bar - 0.1, y_bar], 'k-', linewidth=0.5)
    ax.text(k / 2 + 0.5, y_bar + 0.3, f'CD = {cd:.3f}', ha='center', fontsize=9)

    # Plot algorithms
    for i, (algo, avg_rank) in enumerate(avg_ranks.items()):
        x = i + 1
        ax.plot(x, avg_rank, 'ko', markersize=8)
        ax.plot([x, x], [y_bar, avg_rank], 'k-', linewidth=0.5)
        ax.text(x, avg_rank - 0.3, algo, ha='center', va='top', fontsize=8, rotation=45)

    # Significance bars for groups
    # Simple: connect top performers
    best_rank = avg_ranks.values[0]
    if best_rank + cd >= avg_ranks.values[-1]:
        ax.plot([1, k], [-1, -1], 'k-', linewidth=2)
        ax.text(k / 2 + 0.5, -1.2, 'No significant difference', ha='center', fontsize=8)

    ax.set_title(f'Critical Difference Diagram - {diff.capitalize()} (Nemenyi, α=0.05)', fontsize=11)
    ax.set_xlabel('Algorithm (ordered by average rank)', fontsize=9)
    ax.set_ylabel('Average Rank (lower = better)')
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved CD diagram: {out_path}")

for diff in DIFFICULTIES:
    plot_cd_diagram(df, diff, OUT_DIR / f'cd_diagram_{diff}.png')

# ============================================================
# 7. SUMMARY COMPARISON PLOTS
# ============================================================
print("\n=== GENERATING PLOTS ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7a. Overall AUC-PR bar chart
ax = axes[0, 0]
sorted_algos = summary.index.tolist()
colors = ['#c0392b' if a in ['MemStream_', 'CA_MemStream', 'CA_MemStream_BAR'] else
          '#2980b9' for a in sorted_algos]
bars = ax.barh(range(len(sorted_algos)), summary['mean'], xerr=summary['std'],
               color=colors, alpha=0.8, capsize=3)
ax.set_yticks(range(len(sorted_algos)))
ax.set_yticklabels(sorted_algos, fontsize=9)
ax.set_xlabel('AUC-PR')
ax.set_title('Overall AUC-PR Comparison (baseline algorithms)')
ax.set_xlim(0, 1)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

# 7b. By difficulty
ax = axes[0, 1]
pivot = df.groupby(['algorithm', 'difficulty'])['AUC_PR'].mean().unstack()
if not pivot.empty:
    diff_order = ['easy', 'medium', 'hard']
    pivot = pivot[diff_order]
    pivot.plot(kind='bar', ax=ax, color=['#27ae60', '#f39c12', '#c0392b'], alpha=0.8, width=0.8)
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR by Difficulty')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.legend(title='Difficulty', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

# 7c. Coefficient of Variation
ax = axes[1, 0]
cv = (summary['std'] / (summary['mean'] + 1e-10) * 100).sort_values()
cv_colors = ['#c0392b' if a in ['MemStream_', 'CA_MemStream', 'CA_MemStream_BAR'] else
             '#2980b9' for a in cv.index]
ax.barh(range(len(cv)), cv.values, color=cv_colors, alpha=0.8)
ax.set_yticks(range(len(cv)))
ax.set_yticklabels(cv.index, fontsize=8)
ax.set_xlabel('Coefficient of Variation (%)')
ax.set_title('Result Stability (lower = more stable)')
ax.grid(axis='x', alpha=0.3)

# 7d. Ranking heatmap
ax = axes[1, 1]
pivot_rank = df.groupby(['algorithm', 'difficulty'])['AUC_PR'].mean().unstack()
if not pivot_rank.empty:
    pivot_rank = pivot_rank[['easy', 'medium', 'hard']]
    rank_matrix = pivot_rank.rank(ascending=False)
    im = ax.imshow(rank_matrix.values, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(rank_matrix.columns)))
    ax.set_xticklabels(rank_matrix.columns, fontsize=9)
    ax.set_yticks(range(len(rank_matrix.index)))
    ax.set_yticklabels(rank_matrix.index, fontsize=8)
    ax.set_title('Rank Heatmap (1=best, 8=worst)')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Add rank numbers
    for i in range(len(rank_matrix.index)):
        for j in range(len(rank_matrix.columns)):
            val = rank_matrix.values[i, j]
            color = 'white' if val <= 3 else 'black'
            ax.text(j, i, f'{int(val)}', ha='center', va='center', color=color, fontsize=8)

plt.suptitle('Rigorous Benchmark Results: 8 Algorithms x 11 Folds x 5 Seeds = 440 experiments per algorithm',
             fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / 'fig_summary_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT_DIR / 'fig_summary_comparison.png'}")

# ============================================================
# 8. FINAL SUMMARY REPORT
# ============================================================
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)
print(f"\nBenchmark: 8 algorithms x 11 folds x 5 seeds = 440 experiments each")
print(f"Metric: AUC-PR (Area Under Precision-Recall Curve)")
print(f"Control: sklearn_IF")
print()
print("Overall AUC-PR ranking:")
for i, (algo, row) in enumerate(summary.iterrows(), 1):
    marker = '>>>' if row['cv'] < 100 else '!  '
    print(f"  {i}. {marker} {algo:<22} AUC-PR = {row['mean']:.4f} ± {row['std']:.4f}  (CV={row['cv']:.0f}%)")

print()
print("Key findings:")
memstream_row = summary.loc['MemStream_']
if_memstream_row = summary.loc['sklearn_IF']
best_overall = summary.index[0]
print(f"  1. Best overall: {best_overall} (AUC-PR = {summary.loc[best_overall, 'mean']:.4f})")
print(f"  2. MemStream_ vs sklearn_IF: +{memstream_row['mean'] - if_memstream_row['mean']:.4f} AUC-PR "
      f"({memstream_row['mean']/if_memstream_row['mean']:.1f}x better)")

print("\nDone!")
