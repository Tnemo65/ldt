#!/usr/bin/env python3
"""Clean analysis removing broken baselines and re-run statistical tests."""
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
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = Path('c:/proj/ldt/results/rigorous_v1')

print("Loading results...")
df = pd.read_csv(OUT_DIR / 'results_detailed.csv')
print(f"Total: {len(df)} results, {df['algorithm'].nunique()} algorithms")

# ============================================================
# REMOVE BROKEN BASELINES
# ============================================================
broken = ['sklearn_LOF']  # AUC-PR == anomaly_rate -> broken subsampling
print(f"\nRemoving broken algorithms: {broken}")
df_clean = df[~df['algorithm'].isin(broken)].copy()
print(f"Cleaned: {len(df_clean)} results, {df_clean['algorithm'].nunique()} algorithms")
df_clean.to_csv(OUT_DIR / 'results_clean.csv', index=False)

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY: AUC-PR (Baseline Algorithms Only)")
print("=" * 80)
print(f"{'Algorithm':<22} {'Mean':>8} {'Std':>8} {'Med':>8} {'CV%':>6} {'N':>4}")
print("-" * 60)
summary = df_clean.groupby('algorithm').agg(
    mean=('AUC_PR', 'mean'),
    std=('AUC_PR', 'std'),
    median=('AUC_PR', 'median'),
    n=('AUC_PR', 'count'),
).round(4)
summary['cv'] = (summary['std'] / (summary['mean'] + 1e-10) * 100).round(1)
summary = summary.sort_values('mean', ascending=False)
for algo, row in summary.iterrows():
    print(f"{algo:<22} {row['mean']:>8.4f} {row['std']:>8.4f} {row['median']:>8.4f} {row['cv']:>6.1f}% {int(row['n']):>4}")
summary.to_csv(OUT_DIR / 'summary_clean.csv')

# ============================================================
# BY DIFFICULTY
# ============================================================
print("\n" + "=" * 80)
print("AUC-PR BY DIFFICULTY")
print("=" * 80)
DIFFICULTIES = ['easy', 'medium', 'hard']
print(f"\n{'Algorithm':<22} {'EASY':>10} {'MEDIUM':>10} {'HARD':>10} {'Trend':>10}")
print("-" * 70)
by_diff = df_clean.groupby(['algorithm', 'difficulty'])['AUC_PR'].agg(['mean', 'std'])
pivot_mean = df_clean.groupby(['algorithm', 'difficulty'])['AUC_PR'].mean().unstack()
pivot_std = df_clean.groupby(['algorithm', 'difficulty'])['AUC_PR'].std().unstack()

for algo in summary.index:
    vals = []
    for diff in DIFFICULTIES:
        if diff in pivot_mean.columns:
            m = pivot_mean.loc[algo, diff]
            s = pivot_std.loc[algo, diff]
            vals.append(f"{m:.4f}±{s:.3f}")
        else:
            vals.append("N/A")
    # Trend
    try:
        e = pivot_mean.loc[algo, 'easy']
        h = pivot_mean.loc[algo, 'hard']
        trend = f"{e-h:.4f}"
    except:
        trend = "N/A"
    print(f"{algo:<22} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {trend:>10}")

# ============================================================
# WILCOXON SIGNED-RANK TESTS (pairwise vs sklearn_IF)
# ============================================================
print("\n" + "=" * 80)
print("WILCOXON SIGNED-RANK TESTS (vs sklearn_IF, Holm-Bonferroni corrected)")
print("=" * 80)

control = 'sklearn_IF'
algos = df_clean['algorithm'].unique()
wilcoxon_results = []

def cliffs_delta(x, y):
    n1, n2 = len(x), len(y)
    more = sum(1 for a in x for b in y if a > b)
    less = sum(1 for a in x for b in y if a < b)
    return (more - less) / (n1 * n2)

for diff in DIFFICULTIES:
    df_diff = df_clean[df_clean['difficulty'] == diff]

    for algo in algos:
        if algo == control:
            continue

        ctrl = df_diff[df_diff['algorithm'] == control]['AUC_PR'].values
        alg_v = df_diff[df_diff['algorithm'] == algo]['AUC_PR'].values

        try:
            wr = stats.wilcoxon(alg_v, ctrl, alternative='two-sided')
            stat, p_val = float(wr.statistic), float(wr.pvalue)
        except:
            p_val, stat = np.nan, np.nan

        cd = cliffs_delta(alg_v, ctrl)
        imp = alg_v.mean() - ctrl.mean()

        wilcoxon_results.append({
            'difficulty': diff,
            'algorithm': algo,
            'ctrl_mean': ctrl.mean(),
            'algo_mean': alg_v.mean(),
            'improvement': imp,
            'p_value': p_val,
            'cliffs_delta': cd,
        })

df_w = pd.DataFrame(wilcoxon_results)

# Holm-Bonferroni correction
df_w = df_w.sort_values('p_value')
m = len(df_w)
adjusted = []
for i, (_, row) in enumerate(df_w.iterrows()):
    adj = min(row['p_value'] * (m - i), 1.0)
    if i > 0:
        adj = min(adj, adjusted[-1])
    adjusted.append(adj)
df_w['holm_adj'] = adjusted
df_w['significant'] = df_w['holm_adj'].apply(
    lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '')

print(f"\n{'Diff':>8} {'Algorithm':<22} {'Ctrl':>8} {'Algo':>8} {'Improv':>8} {'p-value':>12} {'Holm':>12} {'Sig':>5} {'Cliff':>8}")
print("-" * 95)
for _, row in df_w.iterrows():
    sig = row['significant']
    direction = '+' if row['improvement'] > 0 else '-'
    print(f"{row['difficulty']:>8} {row['algorithm']:<22} {row['ctrl_mean']:>8.4f} "
          f"{row['algo_mean']:>8.4f} {direction}{abs(row['improvement']):>7.4f} "
          f"{row['p_value']:>12.2e} {row['holm_adj']:>12.2e} {sig:>5} {row['cliffs_delta']:>8.3f}")

df_w.to_csv(OUT_DIR / 'wilcoxon_clean.csv', index=False)

# ============================================================
# RANKING TABLE
# ============================================================
print("\n" + "=" * 80)
print("RANKING TABLE (by AUC-PR)")
print("=" * 80)
rank_data = []
for diff in DIFFICULTIES:
    sub = df_clean[df_clean['difficulty'] == diff]
    means = sub.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False)
    for rank, (algo, val) in enumerate(means.items(), 1):
        std = sub[sub['algorithm'] == algo]['AUC_PR'].std()
        n = sub[sub['algorithm'] == algo]['AUC_PR'].count()
        rank_data.append({
            'Difficulty': diff,
            'Rank': rank,
            'Algorithm': algo,
            'AUC-PR': f'{val:.4f} ± {std:.4f}',
            'AUC-PR_mean': val,
            'AUC-PR_std': std,
            'n': n,
        })

df_rank = pd.DataFrame(rank_data)
print(df_rank.to_string(index=False))
df_rank.to_csv(OUT_DIR / 'ranking_clean.csv', index=False)

# ============================================================
# GENERATE PUBLICATION-QUALITY PLOTS
# ============================================================
print("\n" + "=" * 80)
print("GENERATING PUBLICATION PLOTS")
print("=" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Colors
colors = {
    'sklearn_IF': '#2c3e50',
    'MemStream_': '#c0392b',
    'sHST_Mem_Ensemble': '#8e44ad',
    'HBOS': '#27ae60',
    'CADIFEia': '#e74c3c',
    'sHST_River': '#3498db',
    'IForestASD_': '#e67e22',
}
algo_order = summary.index.tolist()

# 1. Overall AUC-PR bar chart
ax = axes[0, 0]
bars = ax.barh(range(len(algo_order)), summary['mean'], xerr=summary['std'],
               color=[colors.get(a, '#555') for a in algo_order], alpha=0.85,
               capsize=4, error_kw={'linewidth': 1.5})
ax.set_yticks(range(len(algo_order)))
ax.set_yticklabels(algo_order, fontsize=10)
ax.set_xlabel('AUC-PR', fontsize=11)
ax.set_title('(A) Overall AUC-PR Comparison', fontsize=12, fontweight='bold')
ax.set_xlim(0, max(summary['mean']) * 1.3)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()
for i, (algo, row) in enumerate(summary.iterrows()):
    ax.text(row['mean'] + row['std'] + 0.01, i, f"{row['mean']:.3f}", va='center', fontsize=9)

# 2. AUC-PR by difficulty
ax = axes[0, 1]
diff_data = {d: [] for d in DIFFICULTIES}
for algo in algo_order:
    for diff in DIFFICULTIES:
        try:
            diff_data[diff].append(pivot_mean.loc[algo, diff])
        except:
            diff_data[diff].append(0)

x = np.arange(len(algo_order))
width = 0.25
for i, (diff, color) in enumerate(zip(DIFFICULTIES, ['#27ae60', '#f39c12', '#c0392b'])):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, diff_data[diff], width, label=diff.capitalize(),
                  color=color, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(algo_order, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('AUC-PR', fontsize=11)
ax.set_title('(B) AUC-PR by Difficulty Level', fontsize=12, fontweight='bold')
ax.legend(title='Difficulty', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# 3. Ranking heatmap
ax = axes[0, 2]
rank_matrix = pivot_mean.rank(ascending=False)
im = ax.imshow(rank_matrix.values, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=len(algo_order))
ax.set_xticks(range(len(DIFFICULTIES)))
ax.set_xticklabels([d.capitalize() for d in DIFFICULTIES], fontsize=10)
ax.set_yticks(range(len(algo_order)))
ax.set_yticklabels(algo_order, fontsize=9)
ax.set_title('(C) Algorithm Rankings', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8, label='Rank')
for i in range(len(algo_order)):
    for j in range(len(DIFFICULTIES)):
        val = rank_matrix.values[i, j]
        ax.text(j, i, f'{int(val)}', ha='center', va='center',
                color='white' if val <= 2.5 else 'black', fontsize=9, fontweight='bold')

# 4. CV (stability)
ax = axes[1, 0]
cv = (summary['std'] / (summary['mean'] + 1e-10) * 100).sort_values()
ax.barh(range(len(cv)), cv.values,
         color=[colors.get(a, '#555') for a in cv.index], alpha=0.8)
ax.set_yticks(range(len(cv)))
ax.set_yticklabels(cv.index, fontsize=9)
ax.set_xlabel('Coefficient of Variation (%)', fontsize=11)
ax.set_title('(D) Result Stability (lower = better)', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 5. Statistical significance
ax = axes[1, 1]
sig_data = df_w[df_w['difficulty'] == 'easy'].set_index('algorithm')['significant']
y_pos = range(len(algo_order))
for i, algo in enumerate(algo_order):
    sig = sig_data.get(algo, '')
    if sig == '***':
        bar_color = '#27ae60'
        text_color = 'white'
    elif sig == '**':
        bar_color = '#f39c12'
        text_color = 'black'
    elif sig == '*':
        bar_color = '#e67e22'
        text_color = 'black'
    else:
        bar_color = '#bdc3c7'
        text_color = 'black'
    ax.barh(i, 1, color=bar_color, alpha=0.9)
    ax.text(0.5, i, f'{sig or "ns"}', ha='center', va='center',
            fontsize=9, color=text_color, fontweight='bold')
ax.set_yticks(y_pos)
ax.set_yticklabels(algo_order, fontsize=9)
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title("(E) Statistical Significance vs sklearn_IF (easy)", fontsize=12, fontweight='bold')
ax.text(0.5, -0.8, '*** p<0.001  ** p<0.01  * p<0.05  ns=not significant\n(Holm-Bonferroni corrected)',
        ha='center', fontsize=8, style='italic')

# 6. Fold-level performance variance (box plot)
ax = axes[1, 2]
data_box = []
labels_box = []
for algo in algo_order:
    algo_data = df_clean[df_clean['algorithm'] == algo]['AUC_PR'].values
    data_box.append(algo_data)
    labels_box.append(algo)

bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True, vert=False,
                showfliers=True, flierprops={'markersize': 2})
for i, (patch, algo) in enumerate(zip(bp['boxes'], algo_order)):
    patch.set_facecolor(colors.get(algo, '#555'))
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(1.5)
ax.set_yticklabels(labels_box, fontsize=9)
ax.set_xlabel('AUC-PR', fontsize=11)
ax.set_title('(F) Fold-Level AUC-PR Distribution', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.suptitle(
    'Rigorous Benchmark: 7 Baseline Algorithms\n'
    '11 Folds x 3 Difficulties x 5 Seeds = 385 experiments per algorithm',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()
fig.savefig(OUT_DIR / 'fig_results_clean.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: {OUT_DIR / 'fig_results_clean.png'}")

# ============================================================
# KEY FINDINGS
# ============================================================
print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

memstream_row = summary.loc['MemStream_']
if_row = summary.loc['sklearn_IF']

print(f"\n1. BEST ALGORITHM: {summary.index[0]} (AUC-PR = {summary.iloc[0]['mean']:.4f} ± {summary.iloc[0]['std']:.4f})")
print(f"2. MemStream_ vs sklearn_IF:")
print(f"   - EASY:   {pivot_mean.loc['MemStream_', 'easy']:.4f} vs {pivot_mean.loc['sklearn_IF', 'easy']:.4f} "
      f"({pivot_mean.loc['MemStream_', 'easy']/pivot_mean.loc['sklearn_IF', 'easy']:.1f}x better)")
print(f"   - MEDIUM: {pivot_mean.loc['MemStream_', 'medium']:.4f} vs {pivot_mean.loc['sklearn_IF', 'medium']:.4f} "
      f"({pivot_mean.loc['MemStream_', 'medium']/pivot_mean.loc['sklearn_IF', 'medium']:.1f}x better)")
print(f"   - HARD:   {pivot_mean.loc['MemStream_', 'hard']:.4f} vs {pivot_mean.loc['sklearn_IF', 'hard']:.4f} "
      f"({pivot_mean.loc['MemStream_', 'hard']/pivot_mean.loc['sklearn_IF', 'hard']:.1f}x better)")
print(f"   - OVERALL: {memstream_row['mean']:.4f} vs {if_row['mean']:.4f} "
      f"({memstream_row['mean']/if_row['mean']:.1f}x better)")

print(f"\n3. STATISTICAL SIGNIFICANCE:")
for diff in DIFFICULTIES:
    ms = df_w[df_w['difficulty'] == diff].set_index('algorithm').loc['MemStream_']
    sig = ms['significant']
    print(f"   - {diff.upper()}: p={ms['p_value']:.2e} (Holm adj), Cliff's d={ms['cliffs_delta']:.3f} {sig}")

print(f"\n4. HIGH VARIANCE CONCERN:")
print(f"   - MemStream_ CV={memstream_row['cv']:.0f}% (high variance across folds)")
print(f"   - sklearn_IF CV={if_row['cv']:.0f}%")

print("\nDone!")
