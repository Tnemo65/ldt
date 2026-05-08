#!/usr/bin/env python3
"""
Generate Benchmark Results Table
Task 2.6: Create LaTeX table from 5-variant benchmark results

Input: results/benchmark_5_variants.csv (from experiments/benchmark_5_variants.py)
Output: LaTeX table for thesis

Usage:
  python scripts/generate_benchmark_table.py \
    --input results/benchmark_5_variants.csv \
    --output docs/benchmark_table.tex
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(input_path: Path) -> pd.DataFrame:
    """Load benchmark results CSV."""
    df = pd.read_csv(input_path)

    # Verify required columns
    required_cols = [
        'variant', 'seed',
        'precision', 'recall', 'f1', 'fpr',
        'latency_ms', 'throughput_eps'
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by variant (mean ± std across seeds)."""

    metrics = ['precision', 'recall', 'f1', 'fpr', 'latency_ms', 'throughput_eps']

    agg_data = []

    for variant in df['variant'].unique():
        variant_df = df[df['variant'] == variant]

        row = {'variant': variant}

        for metric in metrics:
            mean = variant_df[metric].mean()
            std = variant_df[metric].std()
            row[f'{metric}_mean'] = mean
            row[f'{metric}_std'] = std

        agg_data.append(row)

    return pd.DataFrame(agg_data)


def format_metric(mean, std, is_percentage=False, decimals=1):
    """Format metric as mean ± std."""
    if is_percentage:
        return f"{mean*100:.{decimals}f}±{std*100:.{decimals}f}"
    else:
        return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def generate_latex_table(agg_df: pd.DataFrame, output_path: Path):
    """Generate LaTeX table."""

    # Variant display names
    variant_names = {
        'baseline_static': 'Baseline (Static)',
        'baseline_ratio': 'Baseline (Ratio Features)',
        'proposed_context_aware': '\\textbf{Proposed (Context-Aware)}',
        'arf_online': 'ARF (Online Learning)',
        'loda_lightweight': 'LODA (Lightweight)'
    }

    # Sort variants (proposed should be third)
    variant_order = [
        'baseline_static',
        'baseline_ratio',
        'proposed_context_aware',
        'arf_online',
        'loda_lightweight'
    ]

    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{5-Variant Benchmark Results (Mean ± Std, 5 seeds)}")
    latex.append("\\label{tab:benchmark_results}")
    latex.append("\\begin{tabular}{lcccccc}")
    latex.append("\\toprule")
    latex.append("Variant & Precision (\\%) & Recall (\\%) & F1 (\\%) & FPR (\\%) & Latency (ms) & Throughput (eps) \\\\")
    latex.append("\\midrule")

    for variant in variant_order:
        if variant not in agg_df['variant'].values:
            continue

        row_data = agg_df[agg_df['variant'] == variant].iloc[0]

        # Format metrics
        precision = format_metric(row_data['precision_mean'], row_data['precision_std'],
                                  is_percentage=True, decimals=1)
        recall = format_metric(row_data['recall_mean'], row_data['recall_std'],
                              is_percentage=True, decimals=1)
        f1 = format_metric(row_data['f1_mean'], row_data['f1_std'],
                          is_percentage=True, decimals=1)
        fpr = format_metric(row_data['fpr_mean'], row_data['fpr_std'],
                           is_percentage=True, decimals=1)
        latency = format_metric(row_data['latency_ms_mean'], row_data['latency_ms_std'],
                               is_percentage=False, decimals=0)
        throughput = format_metric(row_data['throughput_eps_mean'], row_data['throughput_eps_std'],
                                  is_percentage=False, decimals=0)

        # Build row
        variant_display = variant_names.get(variant, variant)

        latex.append(f"{variant_display} & {precision} & {recall} & {f1} & {fpr} & {latency} & {throughput} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))

    print(f"\n✅ LaTeX table saved: {output_path}")

    # Also print to console
    print("\n" + "="*80)
    print("LATEX TABLE (copy-paste ready)")
    print("="*80)
    print('\n'.join(latex))
    print("="*80)


def print_summary(agg_df: pd.DataFrame):
    """Print summary statistics."""

    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    variant_names = {
        'baseline_static': 'Baseline (Static)',
        'baseline_ratio': 'Baseline (Ratio)',
        'proposed_context_aware': 'Proposed (Ours)',
        'arf_online': 'ARF Online',
        'loda_lightweight': 'LODA'
    }

    # Find best variant for each metric
    best = {}
    for metric in ['f1', 'precision', 'recall']:
        best_variant = agg_df.loc[agg_df[f'{metric}_mean'].idxmax(), 'variant']
        best_value = agg_df.loc[agg_df[f'{metric}_mean'].idxmax(), f'{metric}_mean']
        best[metric] = (best_variant, best_value)

    # Find lowest FPR
    best_fpr_variant = agg_df.loc[agg_df['fpr_mean'].idxmin(), 'variant']
    best_fpr_value = agg_df.loc[agg_df['fpr_mean'].idxmin(), 'fpr_mean']
    best['fpr'] = (best_fpr_variant, best_fpr_value)

    # Find fastest latency
    best_latency_variant = agg_df.loc[agg_df['latency_ms_mean'].idxmin(), 'variant']
    best_latency_value = agg_df.loc[agg_df['latency_ms_mean'].idxmin(), 'latency_ms_mean']
    best['latency'] = (best_latency_variant, best_latency_value)

    print("\n🏆 Best Performance:")
    for metric_name, (variant, value) in best.items():
        display_name = variant_names.get(variant, variant)
        if metric_name == 'fpr':
            print(f"  {metric_name.upper():.<15} {display_name:.<35} {value*100:.2f}% (lower is better)")
        elif metric_name == 'latency':
            print(f"  {metric_name.upper():.<15} {display_name:.<35} {value:.1f} ms (lower is better)")
        else:
            print(f"  {metric_name.upper():.<15} {display_name:.<35} {value*100:.2f}%")

    # Check if proposed is best overall
    proposed_df = agg_df[agg_df['variant'] == 'proposed_context_aware']
    if not proposed_df.empty:
        proposed_f1 = proposed_df.iloc[0]['f1_mean']
        proposed_fpr = proposed_df.iloc[0]['fpr_mean']

        print(f"\n📊 Proposed Approach:")
        print(f"  F1 Score: {proposed_f1*100:.2f}%")
        print(f"  FPR: {proposed_fpr*100:.2f}%")

        if best['f1'][0] == 'proposed_context_aware':
            print(f"  ✅ BEST F1 SCORE")
        if best['fpr'][0] == 'proposed_context_aware':
            print(f"  ✅ LOWEST FPR")

        if proposed_fpr < 0.04:
            print(f"  ✅ FPR < 4% target achieved!")
        else:
            print(f"  ⚠️  FPR {proposed_fpr*100:.2f}% exceeds 4% target")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate benchmark results table'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file (from benchmark_5_variants.py)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output LaTeX file'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1

    print("="*80)
    print("GENERATING BENCHMARK TABLE")
    print("="*80)

    # Load results
    print(f"\n1. Loading results: {input_path}")
    df = load_results(input_path)
    print(f"   Records: {len(df):,}")
    print(f"   Variants: {df['variant'].nunique()}")
    print(f"   Seeds: {df['seed'].nunique()}")

    # Aggregate
    print(f"\n2. Aggregating results (mean ± std across seeds)...")
    agg_df = aggregate_results(df)

    # Generate LaTeX table
    print(f"\n3. Generating LaTeX table...")
    generate_latex_table(agg_df, output_path)

    # Print summary
    print_summary(agg_df)

    return 0


if __name__ == '__main__':
    exit(main())
