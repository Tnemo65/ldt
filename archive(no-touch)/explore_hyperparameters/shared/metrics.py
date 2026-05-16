"""
Metrics computation for anomaly detection experiments.
Provides all metrics needed across all hyperparameter experiments.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, precision_score, recall_score, f1_score,
    fbeta_score
)


def compute_metrics(y_true, scores, threshold=None):
    """Compute standard anomaly detection metrics.

    Args:
        y_true: Binary labels (1=anomaly, 0=normal)
        scores: Anomaly scores (higher = more anomalous)
        threshold: Fixed threshold. If None, finds best-F1 threshold.

    Returns:
        dict with all metrics
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        return _nan_metrics()

    # AUC-ROC
    try:
        auc_roc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc_roc = np.nan

    # AUC-PR
    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, scores)
        auc_pr = float(auc(rec_curve, prec_curve))
    except Exception:
        auc_pr = np.nan

    # Find best threshold (F1 optimization)
    best_f1, best_thresh = 0.0, 0.0
    best_tp = best_fp = best_tn = best_fn = 0
    best_prec, best_rec = 0.0, 0.0

    for t in np.linspace(scores.min(), scores.max(), 2000):
        yp = (scores > t).astype(int)
        tp = int(((yp == 1) & (y_true == 1)).sum())
        fp = int(((yp == 1) & (y_true == 0)).sum())
        tn = int(((yp == 0) & (y_true == 0)).sum())
        fn = int(((yp == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_tp = tp; best_fp = fp; best_tn = tn; best_fn = fn
            best_prec = prec; best_rec = rec

    # Use fixed threshold if provided (overrides best-F1)
    if threshold is not None:
        yp = (scores > threshold).astype(int)
        best_tp = int(((yp == 1) & (y_true == 1)).sum())
        best_fp = int(((yp == 1) & (y_true == 0)).sum())
        best_tn = int(((yp == 0) & (y_true == 0)).sum())
        best_fn = int(((yp == 0) & (y_true == 1)).sum())
        best_prec = best_tp / (best_tp + best_fp) if (best_tp + best_fp) > 0 else 0.0
        best_rec  = best_tp / (best_tp + best_fn) if (best_tp + best_fn) > 0 else 0.0
        best_f1   = 2 * best_prec * best_rec / (best_prec + best_rec) if (best_prec + best_rec) > 0 else 0.0
        best_thresh = float(threshold)

    fpr = best_fp / (best_fp + best_tn) if (best_fp + best_tn) > 0 else 0.0

    return {
        'AUC-ROC': float(auc_roc),
        'AUC-PR': float(auc_pr),
        'F1': float(best_f1),
        'Precision': float(best_prec),
        'Recall': float(best_rec),
        'FPR': float(fpr),
        'threshold': float(best_thresh),
        'TP': best_tp, 'FP': best_fp, 'TN': best_tn, 'FN': best_fn,
        'n_anomalies': int(y_true.sum()),
        'n_normal': int((y_true == 0).sum()),
    }


def compute_p_at_k(y_true, scores, k_values=None):
    """Compute Precision@K and Recall@K.

    Args:
        y_true: Binary labels
        scores: Anomaly scores
        k_values: List of K values (default: [10, 50, 100, 250, 500])

    Returns:
        dict mapping K -> {'P@K': float, 'R@K': float}
    """
    if k_values is None:
        k_values = [10, 50, 100, 250, 500, 1000]

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    total_anom = max(1, int(y_true.sum()))

    ranking = np.argsort(-scores)
    results = {}
    for k in k_values:
        top_k = ranking[:k]
        tp = int((y_true[top_k] == 1).sum())
        p_at_k = tp / k if k > 0 else 0.0
        r_at_k = tp / total_anom if total_anom > 0 else 0.0
        results[f'P@{k}'] = float(p_at_k)
        results[f'R@{k}'] = float(r_at_k)
    return results


def compute_fpr_at_threshold(y_true, scores, target_fpr=0.05):
    """Find threshold that achieves approximately target FPR.

    Args:
        y_true: Binary labels
        scores: Anomaly scores
        target_fpr: Target false positive rate (default 0.05)

    Returns:
        (threshold, fpr_achieved, metrics)
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    # Sort normal samples by score descending
    norm_scores = scores[y_true == 0]
    if len(norm_scores) == 0:
        return 0.0, 0.0, compute_metrics(y_true, scores)

    # Find threshold at target FPR
    threshold = float(np.percentile(norm_scores, (1 - target_fpr) * 100))
    yp = (scores > threshold).astype(int)
    tp = int(((yp == 1) & (y_true == 1)).sum())
    fp = int(((yp == 1) & (y_true == 0)).sum())
    tn = int(((yp == 0) & (y_true == 0)).sum())
    fn = int(((yp == 0) & (y_true == 1)).sum())
    actual_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return threshold, actual_fpr, {
        'Precision': float(prec),
        'Recall': float(rec),
        'F1': float(2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
    }


def compute_all_metrics(y_true, scores, latency_per_record_ms=None,
                        beta_estimates=None, context_coverage=None):
    """Compute comprehensive metric set for experiment reporting.

    Args:
        y_true: Binary labels
        scores: Anomaly scores
        latency_per_record_ms: Per-record scoring latency in ms (optional)
        beta_estimates: Array of ContextBeta thresholds (optional)
        context_coverage: dict of context cell coverage counts (optional)

    Returns:
        dict with all metrics
    """
    m = compute_metrics(y_true, scores)

    # P@K and R@K
    pak = compute_p_at_k(y_true, scores)
    m.update(pak)

    # Latency metrics
    if latency_per_record_ms is not None:
        lat_arr = np.asarray(latency_per_record_ms)
        m['latency_mean_ms']  = float(lat_arr.mean())
        m['latency_p50_ms']   = float(np.percentile(lat_arr, 50))
        m['latency_p90_ms']   = float(np.percentile(lat_arr, 90))
        m['latency_p95_ms']   = float(np.percentile(lat_arr, 95))
        m['latency_p99_ms']   = float(np.percentile(lat_arr, 99))
        m['latency_max_ms']   = float(lat_arr.max())

    # ContextBeta quality
    if beta_estimates is not None:
        beta_arr = np.asarray(beta_estimates).flatten()
        beta_arr = beta_arr[beta_arr != 0.5]  # Exclude defaults
        if len(beta_arr) > 1:
            m['beta_mean'] = float(beta_arr.mean())
            m['beta_std']  = float(beta_arr.std())
            m['beta_min']  = float(beta_arr.min())
            m['beta_max']  = float(beta_arr.max())

    # Context coverage
    if context_coverage is not None:
        m['context_cells_covered'] = sum(1 for v in context_coverage.values() if v >= 50)
        m['context_cells_total']   = len(context_coverage)
        m['context_coverage_pct'] = m['context_cells_covered'] / m['context_cells_total'] * 100

    return m


def _nan_metrics():
    """Return a metrics dict with all NaN values."""
    return {
        'AUC-ROC': np.nan, 'AUC-PR': np.nan, 'F1': np.nan,
        'Precision': np.nan, 'Recall': np.nan, 'FPR': np.nan,
        'threshold': np.nan, 'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
        'n_anomalies': 0, 'n_normal': 0,
    }
