#!/usr/bin/env python3
"""
Shared evaluation metrics utilities for MemStream.

Tái sử dụng cho tất cả evaluation scripts - không copy-paste nữa.

Usage:
    from eval_utils import find_optimal_threshold, compute_all_metrics
    
    threshold = find_optimal_threshold(scores, labels)
    metrics = compute_all_metrics(scores, labels, threshold)
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvalMetrics:
    """Container cho tất cả metrics - return một object thay vì dict."""
    
    # Core metrics
    f1: float
    precision: float
    recall: float
    fpr: float
    
    # AUC metrics
    auc_roc: float
    auc_pr: float
    
    # Counts
    tp: int
    fp: int
    tn: int
    fn: int
    n_normal: int
    n_anomaly: int
    
    # Threshold
    threshold: float
    
    # Optional: per-class scores
    normal_mean: float = 0.0
    anomaly_mean: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert sang dict để serialize."""
        return {
            'f1': self.f1,
            'precision': self.precision,
            'recall': self.recall,
            'fpr': self.fpr,
            'auc_roc': self.auc_roc,
            'auc_pr': self.auc_pr,
            'tp': self.tp,
            'fp': self.fp,
            'tn': self.tn,
            'fn': self.fn,
            'threshold': self.threshold,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"F1={self.f1:.4f} | P={self.precision:.4f} | R={self.recall:.4f} | "
            f"AUC-ROC={self.auc_roc:.4f} | AUC-PR={self.auc_pr:.4f}"
        )


@dataclass  
class ThresholdSweep:
    """Kết quả của threshold sweep operation."""
    thresholds: np.ndarray
    precisions: np.ndarray
    recalls: np.ndarray
    f1s: np.ndarray
    
    def find_best(self) -> Tuple[float, float]:
        """Return (threshold, best_f1)."""
        idx = np.argmax(self.f1s)
        return self.thresholds[idx], self.f1s[idx]


# =============================================================================
# Core Functions
# =============================================================================

def find_optimal_threshold(scores: np.ndarray, 
                          labels: np.ndarray,
                          method: str = 'f1') -> Tuple[float, float]:
    """
    Tìm optimal threshold bằng grid search.
    
    Args:
        scores: Anomaly scores
        labels: True labels (0=normal, 1=anomaly)
        method: 'f1' (default) hoặc 'youden' (Youden's J statistic)
        
    Returns:
        (threshold, best_score) tuple
        
    Usage:
        threshold, f1 = find_optimal_threshold(scores, labels)
    """
    from sklearn.metrics import roc_curve
    
    # Grid search on percentile of scores
    best_t, best_score = 0.0, -1.0
    
    if method == 'f1':
        for t in np.percentile(scores, np.arange(80, 100, 0.1)):
            preds = (scores >= t).astype(int)
            tp = int(np.sum((preds == 1) & (labels == 1)))
            fp = int(np.sum((preds == 1) & (labels == 0)))
            fn = int(np.sum((preds == 0) & (labels == 1)))
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            if f1 > best_score:
                best_score = f1
                best_t = t
                
    elif method == 'youden':
        fpr_arr, tpr_arr, thresholds = roc_curve(labels, scores)
        j_scores = tpr_arr - fpr_arr
        idx = np.argmax(j_scores)
        best_t = thresholds[idx]
        best_score = j_scores[idx]
    
    return best_t, best_score


def sweep_thresholds(scores: np.ndarray,
                    labels: np.ndarray,
                    n_points: int = 200) -> ThresholdSweep:
    """
    Sweep through thresholds và return arrays của metrics.
    
    Args:
        scores: Anomaly scores
        labels: True labels
        n_points: Số lượng threshold points
        
    Returns:
        ThresholdSweep object
        
    Usage:
        sweep = sweep_thresholds(scores, labels)
        best_t, best_f1 = sweep.find_best()
    """
    thresh_range = np.percentile(scores, np.linspace(80, 99.5, n_points))
    precisions, recalls, f1s = [], [], []
    
    for t in thresh_range:
        preds = (scores >= t).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    return ThresholdSweep(
        thresholds=thresh_range,
        precisions=np.array(precisions),
        recalls=np.array(recalls),
        f1s=np.array(f1s)
    )


def compute_all_metrics(scores: np.ndarray,
                       labels: np.ndarray,
                       threshold: Optional[float] = None) -> EvalMetrics:
    """
    Tính TẤT CẢ metrics từ scores và labels.
    
    Args:
        scores: Anomaly scores
        labels: True labels (0=normal, 1=anomaly)
        threshold: Optional pre-computed threshold, otherwise auto-find
        
    Returns:
        EvalMetrics object
        
    Usage:
        metrics = compute_all_metrics(scores, labels)
        print(f"F1: {metrics.f1}")
        
        # Hoặc với threshold cố định:
        metrics = compute_all_metrics(scores, labels, threshold=1.5)
    """
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
    
    # Find optimal threshold if not provided
    if threshold is None:
        threshold, _ = find_optimal_threshold(scores, labels)
    
    # Predictions
    preds = (scores >= threshold).astype(int)
    
    # Confusion matrix
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    
    n_normal = tn + fp
    n_anomaly = tp + fn
    
    # Core metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # AUC metrics
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    auc_roc = auc(fpr_arr, tpr_arr)
    
    prec_curve, rec_curve, _ = precision_recall_curve(labels, scores)
    auc_pr = auc(rec_curve, prec_curve)
    
    # Per-class score stats
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    return EvalMetrics(
        f1=f1,
        precision=precision,
        recall=recall,
        fpr=fpr,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        n_normal=n_normal,
        n_anomaly=n_anomaly,
        threshold=threshold,
        normal_mean=normal_scores.mean(),
        anomaly_mean=anomaly_scores.mean()
    )


def compute_batch_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Compute aggregate metrics từ nhiều evaluation runs.
    
    Args:
        results: List of dicts với keys: f1, precision, recall, auc_roc, auc_pr
        
    Returns:
        Dict với mean, std của mỗi metric
        
    Usage:
        all_metrics = [r['metrics'] for r in runs]
        agg = compute_batch_metrics(all_metrics)
        print(f"F1: {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}")
    """
    import statistics
    
    metric_names = ['f1', 'precision', 'recall', 'auc_roc', 'auc_pr', 'fpr']
    agg = {}
    
    for name in metric_names:
        values = [r[name] for r in results if name in r]
        if values:
            agg[f'{name}_mean'] = statistics.mean(values)
            agg[f'{name}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
        else:
            agg[f'{name}_mean'] = 0.0
            agg[f'{name}_std'] = 0.0
    
    return agg


# =============================================================================
# Utility Functions
# =============================================================================

def load_scores_labels(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load scores và labels từ npz file.
    
    Args:
        npz_path: Path to .npz file
        
    Returns:
        (scores, labels) tuple
    """
    data = np.load(npz_path)
    return data['scores'], data['labels']


def save_scores_labels(scores: np.ndarray,
                      labels: np.ndarray,
                      output_path: str,
                      **extra_arrays) -> None:
    """
    Save scores và labels (và optional extra arrays) to npz.
    
    Args:
        scores: Anomaly scores
        labels: True labels
        output_path: Output path (.npz)
        **extra_arrays: Additional arrays to save (e.g., X_test=features)
    """
    arrays = {'scores': scores, 'labels': labels}
    arrays.update(extra_arrays)
    np.savez(output_path, **arrays)
    print(f"  Saved to {output_path}")
