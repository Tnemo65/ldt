#!/usr/bin/env python3
"""
Statistical Significance Testing for Benchmark Results.
Task 2.21-2.25: Paired t-test, Wilcoxon, CI, Effect size

Tests whether Proposed Context-Aware significantly outperforms baselines.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def paired_ttest(results_df, metric='f1', variant_a='proposed_context_aware', variant_b='baseline_static'):
    """Perform paired t-test between two variants.

    Args:
        results_df: Benchmark results DataFrame
        metric: Metric to compare (f1, recall, fpr)
        variant_a: First variant name
        variant_b: Second variant name

    Returns:
        t-statistic, p-value, mean difference
    """
    scores_a = results_df[results_df['variant'] == variant_a][metric].values
    scores_b = results_df[results_df['variant'] == variant_b][metric].values

    if len(scores_a) == 0 or len(scores_b) == 0:
        print(f"⚠ Missing data for {variant_a} or {variant_b}")
        return None, None, None

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    mean_diff = np.mean(scores_a - scores_b)

    return t_stat, p_value, mean_diff


def wilcoxon_test(results_df, metric='f1', variant_a='proposed_context_aware', variant_b='baseline_static'):
    """Perform Wilcoxon signed-rank test (non-parametric).

    Args:
        results_df: Benchmark results DataFrame
        metric: Metric to compare
        variant_a: First variant name
        variant_b: Second variant name

    Returns:
        statistic, p-value
    """
    scores_a = results_df[results_df['variant'] == variant_a][metric].values
    scores_b = results_df[results_df['variant'] == variant_b][metric].values

    if len(scores_a) == 0 or len(scores_b) == 0:
        return None, None

    # Wilcoxon signed-rank test
    stat, p_value = stats.wilcoxon(scores_a, scores_b)

    return stat, p_value


def confidence_interval(results_df, metric='f1', variant='proposed_context_aware', confidence=0.95):
    """Calculate confidence interval for a variant's metric.

    Args:
        results_df: Benchmark results DataFrame
        metric: Metric to analyze
        variant: Variant name
        confidence: Confidence level (default 0.95)

    Returns:
        mean, ci_lower, ci_upper
    """
    scores = results_df[results_df['variant'] == variant][metric].values

    if len(scores) == 0:
        return None, None, None

    mean = np.mean(scores)
    sem = stats.sem(scores)
    ci = stats.t.interval(confidence, len(scores) - 1, loc=mean, scale=sem)

    return mean, ci[0], ci[1]


def cohens_d(results_df, metric='f1', variant_a='proposed_context_aware', variant_b='baseline_static'):
    """Calculate Cohen's d effect size.

    Args:
        results_df: Benchmark results DataFrame
        metric: Metric to compare
        variant_a: First variant name
        variant_b: Second variant name

    Returns:
        Cohen's d value
    """
    scores_a = results_df[results_df['variant'] == variant_a][metric].values
    scores_b = results_df[results_df['variant'] == variant_b][metric].values

    if len(scores_a) == 0 or len(scores_b) == 0:
        return None

    # Cohen's d
    mean_diff = np.mean(scores_a) - np.mean(scores_b)
    pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)

    if pooled_std == 0:
        return None

    d = mean_diff / pooled_std

    return d


def run_statistical_tests(
    results_path: str,
    output_path: str = None
):
    """Run all statistical tests on benchmark results.

    Args:
        results_path: Path to benchmark results CSV
        output_path: Path to save test results (optional)

    Returns:
        Test results DataFrame
    """
    print("="*60)
    print("Statistical Significance Testing")
    print("="*60)

    # Load results
    results_df = pd.read_csv(results_path)
    print(f"\nLoaded results: {len(results_df)} runs")

    variants = results_df['variant'].unique()
    print(f"Variants: {list(variants)}")

    # Reference variant (proposed)
    reference = 'proposed_context_aware'

    if reference not in variants:
        print(f"⚠ Reference variant '{reference}' not found, using first variant")
        reference = variants[0]

    # Compare reference vs all others
    metrics = ['f1', 'recall', 'fpr']

    test_results = []

    print(f"\n{'='*60}")
    print(f"Comparing '{reference}' vs Other Variants")
    print(f"{'='*60}")

    for variant in variants:
        if variant == reference:
            continue

        print(f"\n{reference} vs {variant}:")

        for metric in metrics:
            # Paired t-test
            t_stat, p_value, mean_diff = paired_ttest(
                results_df, metric=metric,
                variant_a=reference, variant_b=variant
            )

            # Wilcoxon test
            w_stat, w_pvalue = wilcoxon_test(
                results_df, metric=metric,
                variant_a=reference, variant_b=variant
            )

            # Effect size
            d = cohens_d(
                results_df, metric=metric,
                variant_a=reference, variant_b=variant
            )

            # Interpret results
            if p_value is not None:
                significant = "✅ SIGNIFICANT" if p_value < 0.05 else "❌ NOT SIGNIFICANT"
                effect_size = "large" if abs(d) > 0.8 else ("medium" if abs(d) > 0.5 else "small")

                print(f"\n  {metric.upper()}:")
                print(f"    Paired t-test: t={t_stat:.3f}, p={p_value:.4f} {significant}")
                print(f"    Wilcoxon: stat={w_stat:.1f}, p={w_pvalue:.4f}")
                print(f"    Mean difference: {mean_diff:+.4f}")
                print(f"    Cohen's d: {d:.3f} ({effect_size} effect)")

                test_results.append({
                    'reference': reference,
                    'comparison': variant,
                    'metric': metric,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'wilcoxon_stat': w_stat,
                    'wilcoxon_p': w_pvalue,
                    'mean_difference': mean_diff,
                    'cohens_d': d,
                    'significant': p_value < 0.05,
                    'effect_size': effect_size
                })

    # Confidence intervals for all variants
    print(f"\n{'='*60}")
    print("95% Confidence Intervals")
    print(f"{'='*60}")

    for variant in variants:
        print(f"\n{variant}:")

        for metric in metrics:
            mean, ci_lower, ci_upper = confidence_interval(
                results_df, metric=metric, variant=variant
            )

            if mean is not None:
                print(f"  {metric}: {mean:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Save results
    if output_path:
        test_results_df = pd.DataFrame(test_results)
        test_results_df.to_csv(output_path, index=False)
        print(f"\n✅ Test results saved: {output_path}")

    print(f"\n{'='*60}")
    print("Statistical Testing Complete")
    print(f"{'='*60}")

    return pd.DataFrame(test_results)


def main():
    parser = argparse.ArgumentParser(description='Statistical significance testing')
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to benchmark results CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='experiments/statistical_test_results.csv',
        help='Path to save test results'
    )

    args = parser.parse_args()

    # Validate input
    if not Path(args.results).exists():
        print(f"❌ Results file not found: {args.results}")
        return 1

    # Run tests
    test_results = run_statistical_tests(
        results_path=args.results,
        output_path=args.output
    )

    return 0


if __name__ == '__main__':
    exit(main())
