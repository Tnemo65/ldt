#!/usr/bin/env python3
"""
Statistical Evaluation Framework: MemStream vs IsolationForest Comparison.

Provides rigorous statistical comparison of anomaly detection models using:
  - Block bootstrap confidence intervals (preserves temporal autocorrelation)
  - McNemar's test for paired binary decisions
  - AUC-ROC and AUC-PR metrics with confidence intervals
  - Effective sample size accounting for autocorrelation

Reference: Efron & Tibshirani (1993) for bootstrap methodology,
           McNemar (1947) for paired nominal test.

Usage:
  python src/ml/eval_comparison.py --ytrue labels.npy --memstream scores_ms.npy --iforest scores_if.npy
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
)


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class AUCResults:
    """AUC metrics with block-bootstrap confidence intervals."""
    auc_roc: float
    auc_pr: float
    auc_roc_ci: tuple[float, float]
    auc_pr_ci: tuple[float, float]
    effective_sample_size: int
    block_size: int
    n_bootstrap: int


@dataclass
class ThresholdMetrics:
    """Classification metrics at a specific threshold with bootstrap CI."""
    threshold: float
    precision: float
    recall: float
    f1: float
    fpr: float
    precision_ci: tuple[float, float]
    recall_ci: tuple[float, float]
    f1_ci: tuple[float, float]
    fpr_ci: tuple[float, float]


@dataclass
class McNemarResult:
    """McNemar's test result for paired binary decisions."""
    statistic: float
    p_value: float
    odds_ratio: float
    model1_errors: int  # MemStream errors (false positives on anomalies)
    model2_errors: int  # IForest errors
    significant: bool
    better_model: str  # "memstream", "iforest", "none"


@dataclass
class ComparisonResult:
    """Full comparison result between two models."""
    auc: AUCResults
    threshold_metrics: ThresholdMetrics
    mcnemar: McNemarResult
    n_samples: int
    n_anomalies: int
    anomaly_rate: float


# =============================================================================
# Autocorrelation and Effective Sample Size
# =============================================================================

def compute_autocorrelation(
    scores: np.ndarray,
    max_lag: int = 100
) -> np.ndarray:
    """
    Compute autocorrelation function for a time series.

    Args:
        scores: Time series of anomaly scores
        max_lag: Maximum lag to compute (default: 100)

    Returns:
        Array of autocorrelation values for lags 0 to max_lag
    """
    n = len(scores)
    max_lag = min(max_lag, n - 1)
    scores = scores - np.mean(scores)
    variance = np.var(scores)

    if variance < 1e-10:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0

    for lag in range(1, max_lag + 1):
        if n - lag < 1:
            acf[lag] = 0.0
        else:
            acf[lag] = np.sum(scores[:n-lag] * scores[lag:]) / ((n - lag) * variance)

    return acf


def compute_effective_sample_size(
    scores: np.ndarray,
    block_size: int = 100
) -> int:
    """
    Compute effective sample size accounting for autocorrelation.

    Uses the autocorrelation integral method: ESS = n * (1 - rho_hat)
    where rho_hat is the estimated sum of autocorrelations.

    Args:
        scores: Time series of scores
        block_size: Block size used for analysis (for reporting)

    Returns:
        Effective sample size (integer, <= n)
    """
    n = len(scores)
    if n < 10:
        return n

    # Compute autocorrelation up to n/2
    max_lag = min(n // 2, 500)
    acf = compute_autocorrelation(scores, max_lag=max_lag)

    # Sum of positive autocorrelations
    # Using lag 1 to 2*block_size as relevant autocorrelation range
    relevant_lags = min(2 * block_size, max_lag)
    positive_sum = np.sum(acf[1:relevant_lags + 1] * (acf[1:relevant_lags + 1] > 0))

    # ESS = n / (1 + 2*sum of positive autocorrelations)
    # Clip to avoid division issues
    ess = n / (1.0 + 2.0 * positive_sum)

    return max(1, int(np.round(ess)))


# =============================================================================
# Block Bootstrap
# =============================================================================

def block_bootstrap_indices(
    n: int,
    block_size: int,
    n_bootstrap: int,
    rng: Optional[np.random.Generator] = None
) -> list[np.ndarray]:
    """
    Generate block bootstrap resampling indices.

    Uses circular block bootstrap which handles the series boundary
    by wrapping around, providing more efficient use of data.

    Args:
        n: Length of original time series
        block_size: Block size to preserve temporal structure
        n_bootstrap: Number of bootstrap replicates
        rng: Random number generator (for reproducibility)

    Returns:
        List of bootstrap sample indices
    """
    if rng is None:
        rng = np.random.default_rng()

    n_blocks = (n + block_size - 1) // block_size
    indices_list = []

    for _ in range(n_bootstrap):
        # Sample block starting positions
        block_starts = rng.integers(0, n, size=n_blocks)

        # Build bootstrap sample by concatenating blocks
        bootstrap_indices = np.zeros(n, dtype=np.intp)
        idx = 0
        for start in block_starts:
            for j in range(block_size):
                bootstrap_indices[idx] = (start + j) % n
                idx += 1
                if idx >= n:
                    break

        indices_list.append(bootstrap_indices)

    return indices_list


def block_bootstrap_ci(
    metric_func,
    y_true: np.ndarray,
    scores: np.ndarray,
    block_size: int = 100,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    threshold: Optional[float] = None,
    rng: Optional[np.random.Generator] = None
) -> tuple[float, tuple[float, float]]:
    """
    Compute block bootstrap confidence interval for a metric.

    Args:
        metric_func: Function to compute metric (e.g., roc_auc_score)
        y_true: Binary labels
        scores: Anomaly scores
        block_size: Block size for temporal preservation
        n_bootstrap: Number of bootstrap replicates
        alpha: Significance level (default: 0.05 for 95% CI)
        threshold: Optional threshold for binary decisions
        rng: Random number generator

    Returns:
        Tuple of (point_estimate, (ci_lower, ci_upper))
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(y_true)
    if n < block_size:
        # Not enough data for block bootstrap, return simple estimate
        point_est = metric_func(y_true, scores)
        return point_est, (point_est, point_est)

    # Compute point estimate
    try:
        point_est = metric_func(y_true, scores)
    except ValueError:
        # Handle edge cases (e.g., only one class in y_true)
        return 0.5, (0.5, 0.5)

    # Generate bootstrap indices
    bootstrap_indices = block_bootstrap_indices(n, block_size, n_bootstrap, rng)

    # Compute bootstrap replicates
    replicates = []
    for indices in bootstrap_indices:
        y_boot = y_true[indices]
        scores_boot = scores[indices]

        try:
            # Check for valid bootstrap sample
            if len(np.unique(y_boot)) < 2:
                continue
            replicate = metric_func(y_boot, scores_boot)
            replicates.append(replicate)
        except (ValueError, RuntimeWarning):
            continue

    if len(replicates) < 10:
        return point_est, (point_est, point_est)

    replicates = np.array(replicates)

    # Percentile method for CI
    ci_lower = np.percentile(replicates, (alpha / 2) * 100)
    ci_upper = np.percentile(replicates, (1 - alpha / 2) * 100)

    return point_est, (ci_lower, ci_upper)


# =============================================================================
# Metric Functions
# =============================================================================

def compute_auc_roc(
    y_true: np.ndarray,
    scores: np.ndarray
) -> float:
    """Compute AUC-ROC score with error handling."""
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, scores)
    except ValueError:
        return 0.5


def compute_auc_pr(
    y_true: np.ndarray,
    scores: np.ndarray
) -> float:
    """Compute AUC-PR (Area Under Precision-Recall Curve) with error handling."""
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5

        precision, recall, _ = precision_recall_curve(y_true, scores)

        # Handle empty precision/recall arrays
        if len(precision) < 2 or len(recall) < 2:
            return 0.5

        # AUC-PR is computed with recall in descending order
        return auc(recall, precision)
    except (ValueError, RuntimeWarning):
        return 0.5


def compute_precision_recall_f1_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> tuple[float, float, float, float]:
    """
    Compute precision, recall, F1, and FPR at a given threshold.

    Args:
        y_true: Binary labels (1 = anomaly, 0 = normal)
        scores: Anomaly scores (higher = more anomalous)
        threshold: Decision threshold

    Returns:
        Tuple of (precision, recall, f1, fpr)
    """
    predictions = (scores >= threshold).astype(int)

    tp = np.sum((predictions == 1) & (y_true == 1))
    fp = np.sum((predictions == 1) & (y_true == 0))
    tn = np.sum((predictions == 0) & (y_true == 0))
    fn = np.sum((predictions == 0) & (y_true == 1))

    total_positives = tp + fp
    total_actual_positives = tp + fn
    total_actual_negatives = tn + fp

    precision = tp / total_positives if total_positives > 0 else 0.0
    recall = tp / total_actual_positives if total_actual_positives > 0 else 0.0
    fpr = fp / total_actual_negatives if total_actual_negatives > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, fpr


def find_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_fpr: float = 0.01
) -> float:
    """
    Find threshold that achieves approximately target FPR.

    Args:
        y_true: Binary labels
        scores: Anomaly scores
        target_fpr: Target false positive rate (default: 0.01 = 1%)

    Returns:
        Threshold achieving target FPR
    """
    # Use percentile of normal scores
    normal_scores = scores[y_true == 0]
    if len(normal_scores) == 0:
        return np.median(scores)

    threshold = np.percentile(normal_scores, (1 - target_fpr) * 100)
    return threshold


# =============================================================================
# McNemar's Test
# =============================================================================

def mcnemar_test(
    y_true: np.ndarray,
    scores1: np.ndarray,
    scores2: np.ndarray,
    threshold: float,
    continuity_correction: bool = True
) -> McNemarResult:
    """
    Perform McNemar's test for paired binary decisions.

    McNemar's test is appropriate for comparing paired proportions
    (same subjects, different conditions/methods).

    The test constructs a 2x2 contingency table:
                    | IForest Positive | IForest Negative |
    MemStream Pos   |        b         |        a        |
    MemStream Neg   |        c         |        d        |

    Under null hypothesis: P(b) = P(c)
    Test statistic: chi2 = (|b - c| - 1)^2 / (b + c)  [with continuity correction]

    Args:
        y_true: Binary labels (1 = anomaly)
        scores1: MemStream scores
        scores2: IsolationForest scores
        threshold: Decision threshold
        continuity_correction: Use continuity correction (default: True)

    Returns:
        McNemarResult with test statistics and interpretation
    """
    # Generate binary predictions
    pred1 = (scores1 >= threshold).astype(int)  # MemStream
    pred2 = (scores2 >= threshold).astype(int)  # IsolationForest

    # Build contingency table
    # a: both predict anomaly (TP for both)
    # b: memstream=1, iforest=0 (MemStream detects, IForest misses)
    # c: memstream=0, iforest=1 (MemStream misses, IForest detects)
    # d: both predict normal (TN for both)

    a = np.sum((pred1 == 1) & (pred2 == 1) & (y_true == 1))  # Both correct on anomaly
    b = np.sum((pred1 == 1) & (pred2 == 0))  # MemStream positive, IForest negative
    c = np.sum((pred1 == 0) & (pred2 == 1))  # MemStream negative, IForest positive
    d = np.sum((pred1 == 0) & (pred2 == 0) & (y_true == 0))  # Both correct on normal

    # Focus on disagreement: b and c
    # b = cases where MemStream detects anomaly but IForest doesn't
    # c = cases where IForest detects anomaly but MemStream doesn't
    b = np.sum((pred1 == 1) & (pred2 == 0) & (y_true == 1))  # MemStream correct, IForest error
    c = np.sum((pred1 == 0) & (pred2 == 1) & (y_true == 1))  # MemStream error, IForest correct

    n_disagree = b + c

    if n_disagree == 0:
        # No disagreement - models make same decisions
        return McNemarResult(
            statistic=0.0,
            p_value=1.0,
            odds_ratio=float('inf') if c == 0 else 0.0,
            model1_errors=b,
            model2_errors=c,
            significant=False,
            better_model="none"
        )

    # McNemar's chi-squared statistic
    if continuity_correction and n_disagree > 0:
        # With continuity correction (more conservative)
        chi2_stat = ((abs(b - c) - 1) ** 2) / n_disagree
    else:
        chi2_stat = ((b - c) ** 2) / n_disagree

    # p-value from chi-squared distribution with 1 df
    p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=1)

    # Odds ratio: b/c (with +0.5 correction to avoid division by zero)
    odds_ratio = (b + 0.5) / (c + 0.5)

    # Determine which model is better
    significant = p_value < 0.05
    if significant:
        if b < c:
            better_model = "memstream"
        elif c < b:
            better_model = "iforest"
        else:
            better_model = "none"
    else:
        better_model = "none"

    return McNemarResult(
        statistic=chi2_stat,
        p_value=p_value,
        odds_ratio=odds_ratio,
        model1_errors=b,
        model2_errors=c,
        significant=significant,
        better_model=better_model
    )


# =============================================================================
# Main Comparison Function
# =============================================================================

def compare_models(
    y_true: np.ndarray,
    memstream_scores: np.ndarray,
    if_scores: np.ndarray,
    block_size: int = 100,
    n_bootstrap: int = 1000,
    alpha: float = 0.05
) -> ComparisonResult:
    """
    Compare MemStream vs IsolationForest anomaly detection.

    Computes:
    - AUC-ROC and AUC-PR with block-bootstrap confidence intervals
    - Threshold-based metrics (precision, recall, F1, FPR) with bootstrap CI
    - McNemar's test for paired binary decisions
    - Effective sample size accounting for autocorrelation

    Args:
        y_true: Binary labels (1 = anomaly, 0 = normal)
        memstream_scores: MemStream anomaly scores
        if_scores: IsolationForest anomaly scores
        block_size: Block size for bootstrap (preserves temporal structure)
        n_bootstrap: Number of bootstrap replicates
        alpha: Significance level for CI (default: 0.05 for 95% CI)

    Returns:
        ComparisonResult with all metrics and statistics
    """
    # Input validation
    y_true = np.asarray(y_true, dtype=np.int32)
    memstream_scores = np.asarray(memstream_scores, dtype=np.float64)
    if_scores = np.asarray(if_scores, dtype=np.float64)

    n = len(y_true)
    if n == 0:
        raise ValueError("Empty arrays provided")
    if len(memstream_scores) != n or len(if_scores) != n:
        raise ValueError("Score arrays must have same length as y_true")

    # Handle edge cases
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        raise ValueError("y_true must contain both classes (0 and 1)")

    n_anomalies = int(np.sum(y_true == 1))
    anomaly_rate = n_anomalies / n

    # Determine threshold using MemStream scores
    # (Both models should use same threshold for fair comparison)
    threshold = find_optimal_threshold(y_true, memstream_scores, target_fpr=0.01)

    # Compute effective sample size
    ess = compute_effective_sample_size(memstream_scores, block_size)

    rng = np.random.default_rng(42)

    # =========================================================================
    # AUC Metrics with Block Bootstrap CI
    # =========================================================================

    def auc_roc_wrapper(y, scores):
        return compute_auc_roc(y, scores)

    def auc_pr_wrapper(y, scores):
        return compute_auc_pr(y, scores)

    ms_roc_est, ms_roc_ci = block_bootstrap_ci(
        auc_roc_wrapper, y_true, memstream_scores,
        block_size, n_bootstrap, alpha, rng=rng
    )

    ms_pr_est, ms_pr_ci = block_bootstrap_ci(
        auc_pr_wrapper, y_true, memstream_scores,
        block_size, n_bootstrap, alpha, rng=rng
    )

    if_roc_est, if_roc_ci = block_bootstrap_ci(
        auc_roc_wrapper, y_true, if_scores,
        block_size, n_bootstrap, alpha, rng=rng
    )

    if_pr_est, if_pr_ci = block_bootstrap_ci(
        auc_pr_wrapper, y_true, if_scores,
        block_size, n_bootstrap, alpha, rng=rng
    )

    auc_results = AUCResults(
        auc_roc=ms_roc_est,
        auc_pr=ms_pr_est,
        auc_roc_ci=ms_roc_ci,
        auc_pr_ci=ms_pr_ci,
        effective_sample_size=ess,
        block_size=block_size,
        n_bootstrap=n_bootstrap
    )

    # =========================================================================
    # Threshold-Based Metrics
    # =========================================================================

    # Point estimates
    ms_prec, ms_rec, ms_f1, ms_fpr = compute_precision_recall_f1_fpr(
        y_true, memstream_scores, threshold
    )

    if_prec, if_rec, if_f1, if_fpr = compute_precision_recall_f1_fpr(
        y_true, if_scores, threshold
    )

    # Bootstrap CI for threshold metrics
    def precision_func(y, scores):
        p, _, _, _ = compute_precision_recall_f1_fpr(y, scores, threshold)
        return p

    def recall_func(y, scores):
        _, r, _, _ = compute_precision_recall_f1_fpr(y, scores, threshold)
        return r

    def f1_func(y, scores):
        _, _, f, _ = compute_precision_recall_f1_fpr(y, scores, threshold)
        return f

    def fpr_func(y, scores):
        _, _, _, f = compute_precision_recall_f1_fpr(y, scores, threshold)
        return f

    ms_prec_est, ms_prec_ci = block_bootstrap_ci(
        precision_func, y_true, memstream_scores,
        block_size, n_bootstrap, alpha, threshold, rng
    )

    ms_rec_est, ms_rec_ci = block_bootstrap_ci(
        recall_func, y_true, memstream_scores,
        block_size, n_bootstrap, alpha, threshold, rng
    )

    ms_f1_est, ms_f1_ci = block_bootstrap_ci(
        f1_func, y_true, memstream_scores,
        block_size, n_bootstrap, alpha, threshold, rng
    )

    ms_fpr_est, ms_fpr_ci = block_bootstrap_ci(
        fpr_func, y_true, memstream_scores,
        block_size, n_bootstrap, alpha, threshold, rng
    )

    threshold_metrics = ThresholdMetrics(
        threshold=threshold,
        precision=ms_prec,
        recall=ms_rec,
        f1=ms_f1,
        fpr=ms_fpr,
        precision_ci=ms_prec_ci,
        recall_ci=ms_rec_ci,
        f1_ci=ms_f1_ci,
        fpr_ci=ms_fpr_ci
    )

    # =========================================================================
    # McNemar's Test
    # =========================================================================

    mcnemar_result = mcnemar_test(
        y_true, memstream_scores, if_scores, threshold
    )

    return ComparisonResult(
        auc=auc_results,
        threshold_metrics=threshold_metrics,
        mcnemar=mcnemar_result,
        n_samples=n,
        n_anomalies=n_anomalies,
        anomaly_rate=anomaly_rate
    )


# =============================================================================
# Formatting and Reporting
# =============================================================================

def format_comparison_report(result: ComparisonResult) -> str:
    """
    Format comparison results into a human-readable report.

    Args:
        result: ComparisonResult from compare_models()

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("MemStream vs IsolationForest: Statistical Comparison Report")
    lines.append("=" * 70)

    # Dataset info
    lines.append("\n[DATASET]")
    lines.append(f"  Total samples:        {result.n_samples:,}")
    lines.append(f"  Anomalies:            {result.n_anomalies:,} ({result.anomaly_rate*100:.2f}%)")
    lines.append(f"  Normal:               {result.n_samples - result.n_anomalies:,}")
    lines.append(f"  ESS (block={result.auc.block_size}):  {result.auc.effective_sample_size:,}")

    # AUC Metrics
    lines.append("\n[AUC METRICS]")
    lines.append(f"  AUC-ROC:              {result.auc.auc_roc:.4f}")
    lines.append(f"    95% CI (block):     [{result.auc.auc_roc_ci[0]:.4f}, {result.auc.auc_roc_ci[1]:.4f}]")
    lines.append(f"  AUC-PR:               {result.auc.auc_pr:.4f}")
    lines.append(f"    95% CI (block):     [{result.auc.auc_pr_ci[0]:.4f}, {result.auc.auc_pr_ci[1]:.4f}]")

    # Threshold metrics
    lines.append("\n[THRESHOLD METRICS] (threshold = {:.4f})".format(
        result.threshold_metrics.threshold
    ))
    lines.append(f"  Precision:            {result.threshold_metrics.precision:.4f}")
    lines.append(f"    95% CI:             [{result.threshold_metrics.precision_ci[0]:.4f}, "
                 f"{result.threshold_metrics.precision_ci[1]:.4f}]")
    lines.append(f"  Recall:               {result.threshold_metrics.recall:.4f}")
    lines.append(f"    95% CI:             [{result.threshold_metrics.recall_ci[0]:.4f}, "
                 f"{result.threshold_metrics.recall_ci[1]:.4f}]")
    lines.append(f"  F1 Score:             {result.threshold_metrics.f1:.4f}")
    lines.append(f"    95% CI:             [{result.threshold_metrics.f1_ci[0]:.4f}, "
                 f"{result.threshold_metrics.f1_ci[1]:.4f}]")
    lines.append(f"  FPR:                  {result.threshold_metrics.fpr:.4f}")
    lines.append(f"    95% CI:             [{result.threshold_metrics.fpr_ci[0]:.4f}, "
                 f"{result.threshold_metrics.fpr_ci[1]:.4f}]")

    # McNemar's Test
    lines.append("\n[MCNEMAR'S TEST] (Paired Binary Decisions)")
    lines.append(f"  Chi-squared:          {result.mcnemar.statistic:.4f}")
    lines.append(f"  p-value:              {result.mcnemar.p_value:.6f}")
    lines.append(f"  Odds ratio:           {result.mcnemar.odds_ratio:.4f}")
    lines.append(f"  Significant (p<0.05): {result.mcnemar.significant}")
    if result.mcnemar.significant:
        lines.append(f"  Better model:         {result.mcnemar.better_model.upper()}")
        lines.append(f"    - MemStream errors:  {result.mcnemar.model1_errors}")
        lines.append(f"    - IForest errors:    {result.mcnemar.model2_errors}")
    else:
        lines.append(f"  Conclusion:           No significant difference detected")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare MemStream vs IsolationForest anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/ml/eval_comparison.py --ytrue labels.npy --memstream ms_scores.npy --iforest if_scores.npy
  python src/ml/eval_comparison.py -y labels.npy -m scores_ms.npy -i scores_if.npy -b 50 -n 500
        """
    )

    parser.add_argument(
        '--ytrue', '-y',
        required=True,
        help='Path to numpy array with binary labels (1=anomaly, 0=normal)'
    )
    parser.add_argument(
        '--memstream', '-m',
        required=True,
        help='Path to numpy array with MemStream anomaly scores'
    )
    parser.add_argument(
        '--iforest', '-i',
        required=True,
        help='Path to numpy array with IsolationForest anomaly scores'
    )
    parser.add_argument(
        '--block-size', '-b',
        type=int,
        default=100,
        help='Block size for bootstrap (default: 100)'
    )
    parser.add_argument(
        '--n-bootstrap', '-n',
        type=int,
        default=1000,
        help='Number of bootstrap replicates (default: 1000)'
    )
    parser.add_argument(
        '--alpha', '-a',
        type=float,
        default=0.05,
        help='Significance level for CI (default: 0.05)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Optional: save results to JSON file'
    )

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    try:
        y_true = np.load(args.ytrue)
        memstream_scores = np.load(args.memstream)
        if_scores = np.load(args.iforest)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    print(f"Data loaded: {len(y_true):,} samples, {np.sum(y_true):,} anomalies")

    # Run comparison
    print(f"\nRunning comparison (block_size={args.block_size}, n_bootstrap={args.n_bootstrap})...")
    try:
        result = compare_models(
            y_true,
            memstream_scores,
            if_scores,
            block_size=args.block_size,
            n_bootstrap=args.n_bootstrap,
            alpha=args.alpha
        )
    except ValueError as e:
        print(f"Error in comparison: {e}")
        return 1

    # Print report
    report = format_comparison_report(result)
    print(report)

    # Save to file if requested
    if args.output:
        import json
        output_data = {
            'n_samples': result.n_samples,
            'n_anomalies': result.n_anomalies,
            'anomaly_rate': result.anomaly_rate,
            'effective_sample_size': result.auc.effective_sample_size,
            'block_size': result.auc.block_size,
            'auc_roc': result.auc.auc_roc,
            'auc_roc_ci': list(result.auc.auc_roc_ci),
            'auc_pr': result.auc.auc_pr,
            'auc_pr_ci': list(result.auc.auc_pr_ci),
            'threshold': result.threshold_metrics.threshold,
            'precision': result.threshold_metrics.precision,
            'precision_ci': list(result.threshold_metrics.precision_ci),
            'recall': result.threshold_metrics.recall,
            'recall_ci': list(result.threshold_metrics.recall_ci),
            'f1': result.threshold_metrics.f1,
            'f1_ci': list(result.threshold_metrics.f1_ci),
            'fpr': result.threshold_metrics.fpr,
            'fpr_ci': list(result.threshold_metrics.fpr_ci),
            'mcnemar_statistic': result.mcnemar.statistic,
            'mcnemar_pvalue': result.mcnemar.p_value,
            'mcnemar_odds_ratio': result.mcnemar.odds_ratio,
            'mcnemar_significant': result.mcnemar.significant,
            'mcnemar_better_model': result.mcnemar.better_model,
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
