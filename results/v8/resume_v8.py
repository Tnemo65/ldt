"""
Resume script for benchmark_v8 — skips job execution, loads checkpoint
and runs stats + plots + report.
"""
import sys, os, gc, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from sklearn.metrics import (
    auc, precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix
)

warnings.filterwarnings('ignore')

OUT_DIR = Path(r'C:\proj\ldt\results\v8')

# Config (must match benchmark_v8.py)
SEEDS = [42, 123, 456]
DIFFICULTIES = ['easy', 'medium', 'hard']
LABEL_BUDGETS = [0, 500]
ALGO_NAMES_BATCH = ['Random', 'sklearn_IF', 'sklearn_OCSVM', 'DenoisingAE',
                    'CA-DIF-EIA', 'AE+IF', 'IF-baseline']
ALGO_NAMES_STREAM = ['Random', 'sHST-River', 'MemStream', 'CA-DIF-EIA-Stream']
ANOMALY_RATE = 0.05
ANOMALY_N = 500
TEST_N = 10000
METRICS = ['AUC_PR', 'AUC_ROC', 'F1', 'Precision', 'Recall', 'FPR']

COLORS = {
    'Random': '#bdc3c7', 'sklearn_IF': '#95a5a6', 'sklearn_OCSVM': '#27ae60',
    'DenoisingAE': '#9b59b6', 'CA-DIF-EIA': '#e74c3c', 'AE+IF': '#c0392b',
    'IF-baseline': '#7f8c8d', 'sHST-River': '#3498db', 'MemStream': '#2980b9',
    'CA-DIF-EIA-Stream': '#c0392b',
}


def compute_bar_score(auc_pr, labels_used, label_budget):
    if labels_used == 0 or label_budget == 0:
        return 0.0
    efficiency = auc_pr / labels_used
    utilization = min(1.0, labels_used / label_budget)
    return 100.0 * efficiency * utilization


def statistical_analysis(df, group_name, algos):
    pivot = df.pivot_table(index=['fold', 'difficulty'], columns='algorithm', values='AUC_PR')
    pivot = pivot[algos]
    pivot = pivot.dropna()

    groups = [pivot[a].dropna().values for a in algos]
    if any(len(g) < 2 for g in groups):
        return {'group': group_name, 'significant': False, 'reason': 'insufficient data'}

    try:
        friedman_stat, friedman_p = friedmanchisquare(*groups)
    except Exception:
        return {'group': group_name, 'significant': False, 'reason': 'Friedman test failed'}

    if friedman_p >= 0.05:
        return {
            'group': group_name, 'friedman_stat': float(friedman_stat),
            'friedman_p': float(friedman_p), 'significant': False,
            'conclusion': f'No significant differences (Friedman p={friedman_p:.4f} >= 0.05).',
        }

    def rank_row(row):
        return rankdata(row.values, method='average')
    ranks = pivot.apply(rank_row, axis=1)
    avg_ranks = pd.Series(ranks.mean(axis=0)).sort_values()

    target = 'CA-DIF-EIA'
    baselines = [a for a in algos if a != target and a != 'Random']
    pairwise = []

    for baseline in baselines:
        try:
            t_vals = pivot[target].dropna().values
            b_vals = pivot[baseline].dropna().values
            min_len = min(len(t_vals), len(b_vals))
            if min_len < 2:
                continue
            stat, p_raw = wilcoxon(t_vals[:min_len], b_vals[:min_len],
                                    alternative='greater')
            pairwise.append({
                'target': target, 'baseline': baseline,
                'stat': float(stat), 'p_raw': float(p_raw),
            })
        except Exception:
            pass

    if not pairwise:
        return {'group': group_name, 'friedman_stat': float(friedman_stat),
                'friedman_p': float(friedman_p), 'significant': True,
                'avg_ranks': avg_ranks.to_dict(), 'pairwise_comparisons': []}

    m = len(pairwise)
    sorted_pairs = sorted(pairwise, key=lambda x: x['p_raw'])
    for rank_i, pair in enumerate(sorted_pairs, 1):
        holm_alpha = 0.05 / (m - rank_i + 1)
        pair['holm_alpha'] = holm_alpha
        pair['p_corrected'] = min(pair['p_raw'] * (m - rank_i + 1), 1.0)
        pair['significant'] = pair['p_corrected'] < 0.05

    rng_ci = np.random.RandomState(42)
    ci = {}
    for a in algos:
        vals = pivot[a].dropna().values
        boots = []
        for _ in range(1000):
            idx = rng_ci.choice(len(vals), len(vals), replace=True)
            boots.append(np.mean(vals[idx]))
        ci[a] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    return {
        'group': group_name, 'friedman_stat': float(friedman_stat),
        'friedman_p': float(friedman_p), 'significant': True,
        'avg_ranks': avg_ranks.to_dict(), 'pairwise_comparisons': sorted_pairs,
        'confidence_intervals': ci,
    }


def report_stat_results(stat_result, fp_out=None):
    lines = []
    lines.append(f"\n=== {stat_result['group']} ===")
    lines.append(f"Friedman: stat={stat_result['friedman_stat']:.3f}, p={stat_result['friedman_p']:.4f}")
    if not stat_result['significant']:
        lines.append(f"  NOT SIGNIFICANT -- {stat_result.get('conclusion', '')}")
    else:
        lines.append("  SIGNIFICANT -- proceeding to Holm-Bonferroni post-hoc.")
        lines.append(f"\n  Average ranks (lower = better):")
        for algo, rank in sorted(stat_result['avg_ranks'].items(), key=lambda x: x[1]):
            lines.append(f"    {str(algo):25s}: {rank:.2f}")
        lines.append(f"\n  Pairwise comparisons (Wilcoxon, Holm-corrected):")
        lines.append(f"  {'Comparison':35s} {'p_raw':>8s} {'p_holm':>8s} {'alpha':>8s} {'sig':>4s}")
        for pair in stat_result['pairwise_comparisons']:
            comp = f"{pair['target']} vs {pair['baseline']}"
            sig_str = 'YES' if pair['significant'] else 'no'
            lines.append(f"  {comp:35s} {pair['p_raw']:8.4f} {pair['p_corrected']:8.4f} "
                        f"{pair['holm_alpha']:8.4f} {sig_str:>4s}")
        lines.append(f"\n  Bootstrap 95% Confidence Intervals:")
        if 'confidence_intervals' in stat_result:
            for algo, (lo, hi) in sorted(stat_result['confidence_intervals'].items()):
                lines.append(f"    {str(algo):25s}: [{lo:.4f}, {hi:.4f}]")
        else:
            lines.append("    (not computed)")
    text = '\n'.join(lines)
    if fp_out:
        fp_out.write(text + '\n')
    print(text)


def plot_overview(df, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    batch_df = df[df['algorithm'].isin(ALGO_NAMES_BATCH)]
    algos = batch_df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index

    ax = axes[0, 0]
    data = [batch_df[batch_df['algorithm'] == a]['AUC_PR'].dropna().values for a in algos]
    bp = ax.boxplot(data, patch_artist=True)
    for patch, algo in zip(bp['boxes'], algos):
        patch.set_facecolor(COLORS.get(algo, '#333'))
        patch.set_alpha(0.7)
    ax.set_xticklabels(algos, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR Distribution (Batch)')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 1]
    stream_df = df[df['algorithm'].isin(ALGO_NAMES_STREAM)]
    algos_s = stream_df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index
    data_s = [stream_df[stream_df['algorithm'] == a]['AUC_PR'].dropna().values for a in algos_s]
    bp_s = ax.boxplot(data_s, patch_artist=True)
    for patch, algo in zip(bp_s['boxes'], algos_s):
        patch.set_facecolor(COLORS.get(algo, '#333'))
        patch.set_alpha(0.7)
    ax.set_xticklabels(algos_s, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR Distribution (Streaming)')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[0, 2]
    means = df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False)
    bars = ax.bar(means.index, means.values, color=[COLORS.get(a, '#333') for a in means.index])
    ax.set_xticklabels(means.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean AUC-PR')
    ax.set_title('Mean AUC-PR by Algorithm')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 0]
    pivot = df.pivot_table(index='difficulty', columns='algorithm', values='AUC_PR', aggfunc='mean')
    pivot.plot(kind='bar', ax=ax, color=[COLORS.get(c, '#333') for c in pivot.columns])
    ax.set_title('AUC-PR by Difficulty')
    ax.set_ylabel('AUC-PR')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 1]
    pivot2 = df.pivot_table(index='fold', columns='algorithm', values='AUC_PR', aggfunc='mean')
    pivot2.plot(ax=ax, marker='o', color=[COLORS.get(c, '#333') for c in pivot2.columns])
    ax.set_title('AUC-PR by Fold (Learning Curve)')
    ax.set_xlabel('Fold')
    ax.set_ylabel('AUC-PR')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    times = df.groupby('algorithm')['train_ms'].mean().sort_values(ascending=False)
    ax.barh(times.index, times.values / 1000, color=[COLORS.get(a, '#333') for a in times.index])
    ax.set_xlabel('Training Time (s)')
    ax.set_title('Mean Training Time')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'fig_overview_v8.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_overview_v8.png')


def plot_difficulty(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, diff in enumerate(['easy', 'medium', 'hard']):
        sub = df[df['difficulty'] == diff]
        means = sub.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False)
        colors = [COLORS.get(a, '#333') for a in means.index]
        axes[i].barh(means.index, means.values, color=colors)
        axes[i].set_xlabel('AUC-PR')
        axes[i].set_title(f'{diff.capitalize()} Difficulty')
        axes[i].grid(axis='x', alpha=0.3)
        for j, v in enumerate(means.values):
            axes[i].text(v + 0.005, j, f'{v:.3f}', va='center', fontsize=7)
        axes[i].set_xlim(0, max(means.values) * 1.15)
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_difficulty_v8.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_difficulty_v8.png')


def plot_ablation(df, out_dir):
    ablation_algos = ['IF-baseline', 'AE+IF', 'CA-DIF-EIA']
    sub = df[df['algorithm'].isin(ablation_algos)]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pivot = sub.pivot_table(index='algorithm', columns='difficulty', values='AUC_PR', aggfunc='mean')
    pivot = pivot.loc[ablation_algos]
    pivot.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[0].set_title('Ablation Study: AUC-PR by Difficulty')
    axes[0].set_ylabel('AUC-PR')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend(title='Difficulty')
    axes[0].grid(axis='y', alpha=0.3)

    pivot2 = sub.pivot_table(index='algorithm', columns='difficulty', values='F1', aggfunc='mean')
    pivot2 = pivot2.loc[ablation_algos]
    pivot2.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[1].set_title('Ablation Study: F1 by Difficulty')
    axes[1].set_ylabel('F1')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(title='Difficulty')
    axes[1].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_ablation_v8.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_ablation_v8.png')


def plot_bar_score(df, out_dir):
    stream_df = df[df['algorithm'].isin(ALGO_NAMES_STREAM)].copy()
    if stream_df.empty:
        return
    bar_df = stream_df[stream_df['label_budget'] > 0]
    if bar_df.empty:
        return
    bar_pivot = bar_df.pivot_table(
        index='algorithm', columns='label_budget', values='AUC_PR', aggfunc='mean'
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_pivot.plot(kind='bar', ax=ax, color=['#3498db', '#2980b9'])
    ax.set_title('AUC-PR by Streaming Algorithm and Label Budget')
    ax.set_ylabel('AUC-PR')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(title='Label Budget', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_bar_score_v8.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_bar_score_v8.png')


def plot_pareto_frontier(df, out_dir):
    stream_df = df[df['algorithm'].isin(ALGO_NAMES_STREAM)].copy()
    if stream_df.empty:
        return
    stream_df['BAR_score'] = stream_df.apply(
        lambda r: compute_bar_score(r['AUC_PR'], max(r['labels_consumed'], 1),
                                   max(r.get('label_budget', 0), 1)), axis=1
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for lb in sorted(stream_df['label_budget'].unique()):
        sub = stream_df[stream_df['label_budget'] == lb]
        for algo in sub['algorithm'].unique():
            a_df = sub[sub['algorithm'] == algo]
            axes[0].scatter(
                a_df['labels_consumed'], a_df['AUC_PR'],
                color=COLORS.get(algo, '#333'), label=algo, s=50, alpha=0.7
            )
    axes[0].set_xlabel('Labels Consumed')
    axes[0].set_ylabel('AUC-PR')
    axes[0].set_title('Pareto Frontier')
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    pivot = stream_df.pivot_table(
        index='algorithm', columns='label_budget', values='AUC_PR', aggfunc='mean'
    )
    for algo in pivot.index:
        axes[1].plot(pivot.columns, pivot.loc[algo],
                     marker='o', label=algo, color=COLORS.get(algo, '#333'))
    axes[1].set_xlabel('Label Budget')
    axes[1].set_ylabel('AUC-PR')
    axes[1].set_title('AUC-PR vs Label Budget')
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'fig_pareto_frontier_v8.png', dpi=150)
    plt.close(fig)
    print(f'  Saved fig_pareto_frontier_v8.png')


def write_report(df, stat_results, out_dir):
    lines = []
    lines.append('# Benchmark v8 — MemStream Scientific Correction + Concept Drift\n')
    lines.append(f'**Generated:** {datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}')
    lines.append(f'**Source:** checkpoint_v8.csv ({len(df)} rows)')
    lines.append(f'**Folds:** Leave-one-month-out (5 folds)')
    lines.append(f'**Seeds:** {SEEDS}')
    lines.append(f'**Difficulties:** {DIFFICULTIES}')
    lines.append(f'**Anomaly Rate:** {ANOMALY_RATE:.0%} ({ANOMALY_N} / {TEST_N})\n')
    lines.append('## Summary: Mean AUC-PR by Algorithm\n')
    lines.append('| Algorithm | Mean AUC-PR | Std | N |')
    lines.append('|-----------|-------------|-----|---|')
    summary = df.groupby('algorithm')['AUC_PR'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    for algo, row in summary.iterrows():
        lines.append(f'| {algo} | {row["mean"]:.4f} | {row["std"]:.4f} | {int(row["count"])} |')

    lines.append('\n## AUC-PR by Difficulty\n')
    lines.append('| Algorithm | EASY | MEDIUM | HARD |')
    lines.append('|-----------|------|--------|------|')
    pivot = df.pivot_table(index='algorithm', columns='difficulty', values='AUC_PR', aggfunc='mean')
    for algo in pivot.index:
        row_str = f'| {algo} |'
        for diff in ['easy', 'medium', 'hard']:
            v = pivot.loc[algo, diff] if diff in pivot.columns else '-'
            row_str += f' {v:.4f} |' if isinstance(v, float) else f' {v} |'
        lines.append(row_str)

    lines.append('\n## AUC-PR by Fold\n')
    pivot2 = df.pivot_table(index='algorithm', columns='fold', values='AUC_PR', aggfunc='mean')
    lines.append('| Algorithm | ' + ' | '.join([str(int(f)) for f in pivot2.columns]) + ' |')
    lines.append('|-----------|' + '---|' * len(pivot2.columns))
    for algo in pivot2.index:
        vals = ' | '.join([f'{v:.4f}' if isinstance(v, float) else str(v) for v in pivot2.loc[algo]])
        lines.append(f'| {algo} | {vals} |')

    lines.append('\n## Streaming: AUC-PR by Label Budget\n')
    sdf = df[df['algorithm'].isin(ALGO_NAMES_STREAM)]
    if not sdf.empty:
        sp = sdf.pivot_table(index='algorithm', columns='label_budget', values='AUC_PR', aggfunc='mean')
        lines.append('| Algorithm | Budget=0 | Budget=500 |')
        lines.append('|-----------|----------|------------|')
        for algo in sp.index:
            b0 = sp.loc[algo, 0] if 0 in sp.columns else '-'
            b500 = sp.loc[algo, 500] if 500 in sp.columns else '-'
            lines.append(f'| {algo} | {b0:.4f} | {b500:.4f} |')

    lines.append('\n## Statistical Analysis')
    for name, res in stat_results.items():
        lines.append(f'\n### {res["group"]}')
        if not res.get('significant', False):
            lines.append(f'- Friedman: stat={res.get("friedman_stat", 0):.3f}, p={res.get("friedman_p", 1):.4f}')
            lines.append(f'- **Conclusion:** {res.get("conclusion", "NOT SIGNIFICANT")}')
        else:
            lines.append(f'- Friedman: stat={res.get("friedman_stat", 0):.3f}, p={res.get("friedman_p", 0):.4f} (**SIGNIFICANT**)')
            lines.append(f'- Average ranks:')
            for algo, rank in sorted(res.get('avg_ranks', {}).items(), key=lambda x: x[1]):
                lines.append(f'  - {algo}: {rank:.2f}')
            for pair in res.get('pairwise_comparisons', []):
                sig = '**YES**' if pair.get('significant') else 'no'
                lines.append(f'  - {pair["target"]} vs {pair["baseline"]}: p_raw={pair["p_raw"]:.4f}, p_holm={pair["p_corrected"]:.4f}, sig={sig}')

    with open(out_dir / 'benchmark_v8_results.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  Saved benchmark_v8_results.md')


def main():
    print('=' * 70)
    print('BENCHMARK v8 — Resume: Stats + Plots + Report')
    print('=' * 70)

    # Load checkpoint
    cp_path = OUT_DIR / 'checkpoint_v8.csv'
    if not cp_path.exists():
        print(f'ERROR: {cp_path} not found!')
        return
    bench_df = pd.read_csv(cp_path)
    print(f'\nLoaded {len(bench_df)} rows from checkpoint')

    # BAR scores
    print('\n[1/4] Computing BAR Scores...')
    bench_df['BAR_score'] = bench_df.apply(
        lambda r: compute_bar_score(r['AUC_PR'], max(r['labels_consumed'], 1),
                                   max(r.get('label_budget', 0), 1)), axis=1
    )
    bar_df = bench_df[bench_df['algorithm'].isin(ALGO_NAMES_STREAM)].copy()
    if not bar_df.empty:
        bar_summary = bar_df.pivot_table(
            index='algorithm', columns='label_budget',
            values='BAR_score', aggfunc='mean'
        )
        print('\n  BAR Score Summary:')
        print(bar_summary.to_string())
    bench_df.to_csv(OUT_DIR / 'benchmark_results_v8.csv', index=False)

    # Statistical analysis
    print('\n[2/4] Statistical analysis (Friedman + Holm-Bonferroni)...')
    stat_results = {}

    batch_algos = [a for a in ALGO_NAMES_BATCH if a in bench_df['algorithm'].values]
    if len(batch_algos) >= 2:
        stat_results['batch'] = statistical_analysis(bench_df[bench_df['algorithm'].isin(batch_algos)], 'Batch', batch_algos)

    stream_500 = bench_df[(bench_df['algorithm'].isin(ALGO_NAMES_STREAM)) & (bench_df['label_budget'] == 500)]
    stream_500 = stream_500.dropna(subset=['AUC_PR'])
    stream_algos_500 = [a for a in ALGO_NAMES_STREAM if a in stream_500['algorithm'].values]
    if len(stream_algos_500) >= 2:
        try:
            stat_results['streaming_500'] = statistical_analysis(stream_500, 'Streaming_500', stream_algos_500)
        except Exception as e:
            print(f'  streaming_500 skipped: {e}')

    all_stream = bench_df[bench_df['algorithm'].isin(ALGO_NAMES_STREAM)]
    all_stream = all_stream.dropna(subset=['AUC_PR'])
    stream_algos_all = [a for a in ALGO_NAMES_STREAM if a in all_stream['algorithm'].values]
    if len(stream_algos_all) >= 2:
        try:
            stat_results['all_stream'] = statistical_analysis(all_stream, 'All_Streaming', stream_algos_all)
        except Exception as e:
            print(f'  all_stream skipped: {e}')

    with open(OUT_DIR / 'statistical_results.txt', 'w') as fp:
        for name, res in stat_results.items():
            report_stat_results(res, fp)

    # Plots
    print('\n[3/4] Generating plots...')
    plot_overview(bench_df, OUT_DIR)
    plot_difficulty(bench_df, OUT_DIR)
    plot_ablation(bench_df, OUT_DIR)
    plot_bar_score(bench_df, OUT_DIR)
    plot_pareto_frontier(bench_df, OUT_DIR)

    # Report
    print('\n[4/4] Writing report...')
    write_report(bench_df, stat_results, OUT_DIR)

    print('\n  DONE!')


if __name__ == '__main__':
    main()
