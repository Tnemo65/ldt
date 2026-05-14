"""
PlottingAgent — generates all benchmark figures.
Creates:
  Table A: AUC-PR boxplot, CD diagrams (3 difficulties), p-value heatmap, per-fold plot
  Table B: AUC-PR boxplot, CD diagram (all difficulties)
  Cross-group: BAR Score curves, Ablation delta chart, Radar plot, Per-fold line plots

Uses matplotlib (single-threaded, no multiprocessing needed).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


OUT_DIR = Path(__file__).parent.parent / 'results' / 'v3'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIFFICULTIES = ['easy', 'medium', 'hard']

COLORS = {
    'sklearn_IF':       '#95a5a6',
    'sklearn_OCSVM':    '#bdc3c7',
    'sklearn_LOF':      '#7f8c8d',
    'sHST-River':       '#3498db',
    'MemStream':        '#2980b9',
    'IForestASD':       '#e67e22',
    'LSTM-AE':          '#9b59b6',
    'CA-DIF-EIA':       '#e74c3c',
    'METER-SCD':        '#27ae60',
    'CA-DIF-EIA-Stream': '#c0392b',
}


# ─── Helpers ────────────────────────────────────────────────────────────────

def _safe_save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path.name}')


def _boxplot_data(df: pd.DataFrame, group_col: str, metric: str = 'AUC_PR'):
    """Return list of arrays for boxplot."""
    algos = df.groupby(group_col)[metric].mean().sort_values(ascending=False).index
    return [(df[df[group_col] == a][metric].dropna().values) for a in algos], list(algos)


# ─── Table A Figures ───────────────────────────────────────────────────────

def plot_table_a_overview(batch_df: pd.DataFrame, gpu_df: pd.DataFrame | None, out: Path):
    """Combined Table A figure: AUC-PR boxplot + per-difficulty + F1 + folds."""
    if gpu_df is not None and not gpu_df.empty:
        df = pd.concat([batch_df, gpu_df], ignore_index=True)
    else:
        df = batch_df

    algos = df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index
    n = len(algos)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. AUC-PR boxplot
    ax = axes[0, 0]
    bp_data, bp_labels = _boxplot_data(df, 'algorithm', 'AUC_PR')
    bp = ax.boxplot(bp_data, patch_artist=True, widths=0.6)
    for patch, algo in zip(bp['boxes'], bp_labels):
        patch.set_facecolor(COLORS.get(algo, '#555'))
        patch.set_alpha(0.75)
    ax.set_xticklabels(bp_labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR Distribution (Table A)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    # 2. AUC-PR by difficulty
    ax = axes[0, 1]
    x = np.arange(len(algos))
    w = 0.25
    for di, diff in enumerate(DIFFICULTIES):
        means = [df[(df['algorithm'] == a) & (df['difficulty'] == diff)]['AUC_PR'].mean()
                 for a in algos]
        ax.bar(x + di * w, means, w, label=diff.capitalize(),
               color=['#27ae60', '#f39c12', '#c0392b'][di], alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(algos, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR by Difficulty', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 3. F1 score bar
    ax = axes[0, 2]
    f1 = df.groupby('algorithm')['F1'].mean().sort_values(ascending=True)
    bars = ax.barh(range(len(f1)), f1.values,
                   color=[COLORS.get(a, '#555') for a in f1.index])
    ax.set_yticks(range(len(f1)))
    ax.set_yticklabels(f1.index, fontsize=8)
    ax.set_xlabel('F1 Score')
    ax.set_title('Mean F1 Score (Table A)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 4. Per-fold AUC-PR
    ax = axes[1, 0]
    top3 = df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).head(3).index
    for algo in top3:
        mdata = df[df['algorithm'] == algo].groupby('fold')['AUC_PR'].mean()
        ax.plot(mdata.index, mdata.values, 'o-', label=algo,
                color=COLORS.get(algo, '#333'), linewidth=2, markersize=4)
    ax.set_xlabel('Fold (Month)')
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR Over Folds (Top 3)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 5. Recall vs Precision scatter
    ax = axes[1, 1]
    for algo in algos:
        mean_prc = df[df['algorithm'] == algo]['Precision'].mean()
        mean_rec = df[df['algorithm'] == algo]['Recall'].mean()
        ax.scatter(mean_prc, mean_rec, s=150, c=COLORS.get(algo, '#555'),
                   label=algo, edgecolors='white', linewidth=1)
    ax.set_xlabel('Mean Precision')
    ax.set_ylabel('Mean Recall')
    ax.set_title('Precision vs Recall (Table A)', fontweight='bold')
    ax.legend(fontsize=6, loc='best')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 6. Score time
    ax = axes[1, 2]
    times = df.groupby('algorithm')['score_ms'].mean().sort_values(ascending=True)
    ax.barh(range(len(times)), times.values,
            color=[COLORS.get(a, '#555') for a in times.index])
    ax.set_yticks(range(len(times)))
    ax.set_yticklabels(times.index, fontsize=8)
    ax.set_xlabel('Score time (ms)')
    ax.set_title('Mean Scoring Time', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Table A — Batch Algorithm Benchmark Overview', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    _safe_save(fig, out / 'figA1_overview.png')


def plot_cd_diagram(cd_df: pd.DataFrame, title: str, out: Path, diff_label: str = ''):
    """Critical Difference diagram."""
    fig, ax = plt.subplots(figsize=(12, 4))
    s  = cd_df.sort_values('avg_rank')
    cd = float(s['cd'].iloc[0]) if 'cd' in s.columns and len(s) > 0 else 2.0
    n  = len(s)

    ax.plot([s['avg_rank'].min() - 0.5, s['avg_rank'].max() + 0.5],
            [0.5, 0.5], '#888', linewidth=1, zorder=1)

    for i in range(n):
        for j in range(i+1, n):
            r1, r2 = s.iloc[i]['avg_rank'], s.iloc[j]['avg_rank']
            if abs(r1 - r2) < cd:
                ax.plot([r1, r2], [0.5, 0.5], color='#2c3e50',
                        linewidth=5, zorder=2, solid_capstyle='round')

    for _, row in s.iterrows():
        algo = row['algorithm']
        c = COLORS.get(algo, '#555')
        ax.scatter(row['avg_rank'], 0.8, c=c, s=250, zorder=5,
                   edgecolors='white', linewidth=2)
        ax.annotate(algo, (row['avg_rank'], 0.85),
                    ha='center', va='bottom', fontsize=9, rotation=15,
                    fontweight='bold')

    ax.set_xlim(s['avg_rank'].min() - 0.8, s['avg_rank'].max() + 0.8)
    ax.set_ylim(0, 1.1)
    ax.axis('off')
    ax.set_title(f'Critical Difference — {title}  (CD = {cd:.2f})',
                 fontsize=12, fontweight='bold')
    _safe_save(fig, out / f'fig_cd_{diff_label}.png')


def plot_table_b_overview(df: pd.DataFrame, out: Path):
    """Table B streaming overview."""
    if df.empty:
        return

    algos = df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    bp_data, bp_labels = _boxplot_data(df, 'algorithm', 'AUC_PR')
    bp = ax.boxplot(bp_data, patch_artist=True, widths=0.6)
    for patch, algo in zip(bp['boxes'], bp_labels):
        patch.set_facecolor(COLORS.get(algo, '#555'))
        patch.set_alpha(0.75)
    ax.set_xticklabels(bp_labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR Distribution (Table B — Streaming)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    for diff in DIFFICULTIES:
        means = [df[(df['algorithm'] == a) & (df['difficulty'] == diff)]['AUC_PR'].mean()
                 for a in algos]
        x = np.arange(len(algos))
        offset = DIFFICULTIES.index(diff) * 0.25
        ax.bar(x + offset, means, 0.22, label=diff.capitalize(),
               color=['#27ae60', '#f39c12', '#c0392b'][DIFFICULTIES.index(diff)], alpha=0.8)
    ax.set_xticks(x + 0.25)
    ax.set_xticklabels(algos, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR by Difficulty (Streaming)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1, 0]
    for algo in algos:
        mdata = df[df['algorithm'] == algo].groupby('fold')['AUC_PR'].mean()
        ax.plot(mdata.index, mdata.values, 'o-', label=algo,
                color=COLORS.get(algo, '#333'), linewidth=2, markersize=4)
    ax.set_xlabel('Fold (Month)')
    ax.set_ylabel('AUC-PR')
    ax.set_title('AUC-PR Over Folds (Streaming)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    lc = df.groupby('algorithm')['labels_consumed'].mean().sort_values(ascending=True)
    ax.barh(range(len(lc)), lc.values,
            color=[COLORS.get(a, '#555') for a in lc.index])
    ax.set_yticks(range(len(lc)))
    ax.set_yticklabels(lc.index, fontsize=8)
    ax.set_xlabel('Mean Labels Consumed')
    ax.set_title('Mean Labels Consumed (Streaming)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Table B — Streaming Algorithm Benchmark Overview', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    _safe_save(fig, out / 'figB1_overview.png')


def plot_bar_score(df: pd.DataFrame, out: Path):
    """BAR Score curves: AUC-PR vs labeled data budget."""
    if df.empty or 'budget_pct' not in df.columns:
        return

    budgets = sorted(df['budget_pct'].unique())
    algos   = df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).index

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for algo in algos:
        d = df[df['algorithm'] == algo].groupby('budget_pct')['AUC_PR'].mean().reindex(budgets)
        ax.plot(budgets, d.values, 'o-', label=algo,
                color=COLORS.get(algo, '#333'), linewidth=2, markersize=6)
    ax.set_xlabel('Labeled Data Budget (%)')
    ax.set_ylabel('AUC-PR')
    ax.set_title('BAR Score: AUC-PR vs. Label Budget\n(Higher at lower budget = more label-efficient)',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    ax = axes[1]
    at5 = df[df['budget_pct'] == 0.05].groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=True)
    ax.barh(range(len(at5)), at5.values,
            color=[COLORS.get(a, '#555') for a in at5.index])
    ax.set_yticks(range(len(at5)))
    ax.set_yticklabels(at5.index, fontsize=9)
    ax.set_xlabel('AUC-PR at 5% Budget')
    ax.set_title('Label Efficiency at 5% Budget', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    _safe_save(fig, out / 'figX1_bar_score.png')


def plot_ablation(df: pd.DataFrame, out: Path):
    """Ablation delta AUC-PR bar chart."""
    if df.empty or 'delta' not in df.columns:
        return

    algos = df.groupby('algorithm')['delta'].mean().sort_values(ascending=False).index
    diffs = DIFFICULTIES

    x = np.arange(len(algos))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    for di, diff in enumerate(diffs):
        means = [df[(df['algorithm'] == a) & (df['difficulty'] == diff)]['delta'].mean()
                 for a in algos]
        bars = ax.bar(x + di * w, means, w, label=diff.capitalize(),
                      color=['#27ae60', '#f39c12', '#c0392b'][di], alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x + w)
    ax.set_xticklabels(algos, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('ΔAUC-PR (Treatment − Control)')
    ax.set_title('Ablation: Context-aware Grid Effect\n(25D Treatment vs 15D Control)',
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    _safe_save(fig, out / 'figX2_ablation.png')


def plot_radar(df: pd.DataFrame, out: Path, group: str = 'Table A'):
    """Radar chart for top-3 algorithms across AUC-PR, F1, Recall, Precision."""
    if df.empty or group == 'Table B':
        return

    top3 = df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False).head(3).index
    metrics_r = ['AUC_PR', 'F1', 'Recall', 'Precision']

    angles = np.linspace(0, 2 * np.pi, len(metrics_r), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    for algo in top3:
        vals = [df[df['algorithm'] == algo][m].mean() for m in metrics_r]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2,
                color=COLORS.get(algo, '#333'), label=algo)
        ax.fill(angles, vals, alpha=0.1, color=COLORS.get(algo, '#333'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_r, fontsize=10)
    ax.set_title(f'Top 3 Algorithms — Radar Chart\n({group})', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=9)
    _safe_save(fig, out / f'fig_radar_{group.lower().replace(" ", "_")}.png')


# ─── Main Plotting Agent ─────────────────────────────────────────────────────

class PlottingAgent:
    """
    Generates all benchmark figures.
    Reads results from results/v3/, saves figures to results/v3/.
    """

    def __init__(self, base_dir: Path | None = None, out_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.out_dir  = Path(out_dir)  if out_dir  else OUT_DIR

    def run(self):
        print('\n[PlottingAgent] Generating figures...')

        # Load data
        batch_path   = self.out_dir / 'benchmark_results_batch.csv'
        stream_path  = self.out_dir / 'benchmark_results_streaming.csv'
        bar_path     = self.out_dir / 'bar_score_results.csv'
        ablate_path  = self.out_dir / 'ablation_results.csv'
        cd_batch_path= self.out_dir / 'cd_ranks_batch.csv'
        cd_stream_path= self.out_dir / 'cd_ranks_streaming.csv'

        batch_df   = pd.read_csv(batch_path)   if batch_path.exists()   else pd.DataFrame()
        stream_df  = pd.read_csv(stream_path)  if stream_path.exists()  else pd.DataFrame()
        bar_df     = pd.read_csv(bar_path)     if bar_path.exists()      else pd.DataFrame()
        ablate_df  = pd.read_csv(ablate_path)  if ablate_path.exists()   else pd.DataFrame()
        cd_batch   = pd.read_csv(cd_batch_path) if cd_batch_path.exists() else pd.DataFrame()
        cd_stream  = pd.read_csv(cd_stream_path) if cd_stream_path.exists() else pd.DataFrame()

        gpu_path = self.out_dir / 'gpu_lstm_results.csv'
        gpu_df = pd.read_csv(gpu_path) if gpu_path.exists() else None

        # Table A figures
        if not batch_df.empty:
            plot_table_a_overview(batch_df, gpu_df, self.out_dir)
            plot_radar(batch_df, self.out_dir, 'Table_A')

            for diff in DIFFICULTIES:
                cd_sub = cd_batch[cd_batch['difficulty'] == diff]
                if not cd_sub.empty:
                    plot_cd_diagram(cd_sub, f'Table A / {diff.capitalize()}',
                                   self.out_dir, f'cd_batch_{diff}')

        # Table B figures
        if not stream_df.empty:
            plot_table_b_overview(stream_df, self.out_dir)
            for diff in DIFFICULTIES:
                cd_sub = cd_stream[cd_stream['difficulty'] == diff]
                if not cd_sub.empty:
                    plot_cd_diagram(cd_sub, f'Table B / {diff.capitalize()}',
                                   self.out_dir, f'cd_stream_{diff}')

        # Cross-group figures
        if not bar_df.empty:
            plot_bar_score(bar_df, self.out_dir)
        if not ablate_df.empty:
            plot_ablation(ablate_df, self.out_dir)

        print(f'\n[PlottingAgent] All figures saved to {self.out_dir}')


if __name__ == '__main__':
    PlottingAgent().run()
