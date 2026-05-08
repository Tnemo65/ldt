#!/usr/bin/env python3
"""
Plot Benchmark Results
Generate visualization plots from 5-variant benchmark

Plots:
  1. Precision-Recall comparison (bar chart)
  2. F1 Score comparison (bar chart)
  3. FPR comparison (bar chart with 4% target line)
  4. Latency comparison (bar chart)
  5. Throughput comparison (bar chart)

Usage:
  python scripts/plot_benchmark_results.py \
    --input results/benchmark_5_variants.csv \
    --output-dir docs/figures/
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def load_and_aggregate(input_path: Path) -> pd.DataFrame:
    """Load and aggregate benchmark results."""
    df = pd.read_csv(input_path)

    # Aggregate by variant (mean ± std)
    metrics = ['precision', 'recall', 'f1', 'fpr', 'latency_ms', 'throughput_eps']

    agg = df.groupby('variant')[metrics].agg(['mean', 'std']).reset_index()

    # Flatten column names
    agg.columns = ['variant'] + [f'{m}_{stat}' for m in metrics for stat in ['mean', 'std']]

    return agg


def plot_precision_recall(agg_df: pd.DataFrame, output_dir: Path):
    """Plot precision and recall comparison."""

    fig, ax = plt.subplots(figsize=(10, 6))

    variants = agg_df['variant'].tolist()
    x = np.arange(len(variants))
    width = 0.35

    # Precision bars
    precision_means = agg_df['precision_mean'].values * 100
    precision_stds = agg_df['precision_std'].values * 100
    ax.bar(x - width/2, precision_means, width, yerr=precision_stds,
           label='Precision', alpha=0.8, capsize=5)

    # Recall bars
    recall_means = agg_df['recall_mean'].values * 100
    recall_stds = agg_df['recall_std'].values * 100
    ax.bar(x + width/2, recall_means, width, yerr=recall_stds,
           label='Recall', alpha=0.8, capsize=5)

    ax.set_xlabel('Variant')
    ax.set_ylabel('Score (%)')
    ax.set_title('Precision and Recall Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', '\n') for v in variants], rotation=0, ha='center')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'benchmark_precision_recall.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {output_path}")
    plt.close()


def plot_f1_score(agg_df: pd.DataFrame, output_dir: Path):
    """Plot F1 score comparison."""

    fig, ax = plt.subplots(figsize=(10, 6))

    variants = agg_df['variant'].tolist()
    x = np.arange(len(variants))

    f1_means = agg_df['f1_mean'].values * 100
    f1_stds = agg_df['f1_std'].values * 100

    colors = ['#1f77b4' if 'proposed' not in v else '#d62728' for v in variants]

    ax.bar(x, f1_means, yerr=f1_stds, color=colors, alpha=0.8, capsize=5)

    ax.set_xlabel('Variant')
    ax.set_ylabel('F1 Score (%)')
    ax.set_title('F1 Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', '\n') for v in variants], rotation=0, ha='center')
    ax.grid(axis='y', alpha=0.3)

    # Highlight best
    best_idx = np.argmax(f1_means)
    ax.axhline(f1_means[best_idx], color='red', linestyle='--', alpha=0.3,
               label=f'Best: {f1_means[best_idx]:.1f}%')
    ax.legend()

    plt.tight_layout()
    output_path = output_dir / 'benchmark_f1_score.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {output_path}")
    plt.close()


def plot_fpr(agg_df: pd.DataFrame, output_dir: Path):
    """Plot FPR comparison with 4% target line."""

    fig, ax = plt.subplots(figsize=(10, 6))

    variants = agg_df['variant'].tolist()
    x = np.arange(len(variants))

    fpr_means = agg_df['fpr_mean'].values * 100
    fpr_stds = agg_df['fpr_std'].values * 100

    colors = ['#2ca02c' if fpr < 4 else '#d62728' for fpr in fpr_means]

    ax.bar(x, fpr_means, yerr=fpr_stds, color=colors, alpha=0.8, capsize=5)

    # 4% target line
    ax.axhline(4.0, color='blue', linestyle='--', linewidth=2,
               label='Target: 4% FPR')

    ax.set_xlabel('Variant')
    ax.set_ylabel('False Positive Rate (%)')
    ax.set_title('False Positive Rate Comparison (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', '\n') for v in variants], rotation=0, ha='center')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'benchmark_fpr.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {output_path}")
    plt.close()


def plot_latency(agg_df: pd.DataFrame, output_dir: Path):
    """Plot latency comparison."""

    fig, ax = plt.subplots(figsize=(10, 6))

    variants = agg_df['variant'].tolist()
    x = np.arange(len(variants))

    latency_means = agg_df['latency_ms_mean'].values
    latency_stds = agg_df['latency_ms_std'].values

    colors = ['#ff7f0e'] * len(variants)

    ax.bar(x, latency_means, yerr=latency_stds, color=colors, alpha=0.8, capsize=5)

    ax.set_xlabel('Variant')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Processing Latency Comparison (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', '\n') for v in variants], rotation=0, ha='center')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'benchmark_latency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {output_path}")
    plt.close()


def plot_throughput(agg_df: pd.DataFrame, output_dir: Path):
    """Plot throughput comparison."""

    fig, ax = plt.subplots(figsize=(10, 6))

    variants = agg_df['variant'].tolist()
    x = np.arange(len(variants))

    throughput_means = agg_df['throughput_eps_mean'].values
    throughput_stds = agg_df['throughput_eps_std'].values

    colors = ['#9467bd'] * len(variants)

    ax.bar(x, throughput_means, yerr=throughput_stds, color=colors, alpha=0.8, capsize=5)

    ax.set_xlabel('Variant')
    ax.set_ylabel('Throughput (events/sec)')
    ax.set_title('Processing Throughput Comparison (Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace('_', '\n') for v in variants], rotation=0, ha='center')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'benchmark_throughput.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {output_path}")
    plt.close()


def plot_radar_chart(agg_df: pd.DataFrame, output_dir: Path):
    """Plot radar chart comparing all metrics (normalized)."""

    from math import pi

    # Normalize metrics to 0-1 scale
    metrics = ['precision_mean', 'recall_mean', 'f1_mean']

    # For FPR and latency, lower is better, so invert
    agg_df['fpr_normalized'] = 1 - agg_df['fpr_mean']
    agg_df['latency_normalized'] = 1 - (agg_df['latency_ms_mean'] / agg_df['latency_ms_mean'].max())

    metrics_normalized = metrics + ['fpr_normalized', 'latency_normalized']
    metric_labels = ['Precision', 'Recall', 'F1', 'FPR\n(inverted)', 'Latency\n(inverted)']

    # Only plot top 3 variants
    top_variants = agg_df.nlargest(3, 'f1_mean')['variant'].tolist()

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    angles = [n / len(metric_labels) * 2 * pi for n in range(len(metric_labels))]
    angles += angles[:1]  # Complete the circle

    for variant in top_variants:
        values = top_variants_df = agg_df[agg_df['variant'] == variant][metrics_normalized].values[0].tolist()
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=variant.replace('_', ' '))
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar Chart (Top 3 Variants)', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.tight_layout()
    output_path = output_dir / 'benchmark_radar.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot benchmark results'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file (from benchmark_5_variants.py)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for plots'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1

    print("="*80)
    print("GENERATING BENCHMARK PLOTS")
    print("="*80)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and aggregate data
    print(f"\n1. Loading data: {input_path}")
    agg_df = load_and_aggregate(input_path)
    print(f"   Variants: {len(agg_df)}")

    # Generate plots
    print(f"\n2. Generating plots...")

    plot_precision_recall(agg_df, output_dir)
    plot_f1_score(agg_df, output_dir)
    plot_fpr(agg_df, output_dir)
    plot_latency(agg_df, output_dir)
    plot_throughput(agg_df, output_dir)

    try:
        plot_radar_chart(agg_df, output_dir)
    except Exception as e:
        print(f"  ⚠️  Radar chart failed: {e}")

    print("\n" + "="*80)
    print("✅ ALL PLOTS GENERATED")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print("\nPlots created:")
    print("  - benchmark_precision_recall.png")
    print("  - benchmark_f1_score.png")
    print("  - benchmark_fpr.png")
    print("  - benchmark_latency.png")
    print("  - benchmark_throughput.png")
    print("  - benchmark_radar.png")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
