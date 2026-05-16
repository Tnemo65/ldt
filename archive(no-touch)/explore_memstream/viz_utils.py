#!/usr/bin/env python3
"""
Shared visualization utilities for MemStream evaluation.

Đây là module TÁI SỬ DỤNG - import từ đây, không copy-paste code.

Usage:
    from viz_utils import VizStyle, save_fig, COLORS
    
    style = VizStyle.publication()
    fig, ax = plt.subplots()
    save_fig(fig, 'output.png')
"""

from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
import numpy as np


# =============================================================================
# Constants - Color Palette
# =============================================================================

class ColorPalette:
    """Màu sắc chuẩn cho visualization - tái sử dụng across all plots."""
    
    # Primary colors
    NORMAL = '#4A90D9'
    ANOMALY = '#E24A33'
    TP = '#27AE60'
    FP = '#E74C3C'
    TN = '#85C1E9'
    FN = '#F39C12'
    
    # Metrics colors
    METRIC_ROC = '#3498DB'
    METRIC_PR = '#E74C3C'
    METRIC_F1 = '#27AE60'
    
    # Additional
    RUSH_HOUR = '#E74C3C'
    WEEKEND = '#9B59B6'
    
    # Borough colors
    BOROUGH_COLORS = {
        'manhattan': '#2C3E50',
        'brooklyn': '#3498DB', 
        'queens': '#27AE60',
        'bronx': '#E74C3C',
        'jfk': '#F39C12',
        'ewr': '#9B59B6',
        'other': '#95A5A6',
    }
    
    # Fraud type colors
    FRAUD_TYPE1 = '#E74C3C'  # Short-trip fraud
    FRAUD_TYPE3 = '#F39C12'  # Ratecode mismatch
    
    @classmethod
    def get_all(cls) -> dict:
        """Return dict of all colors for iteration."""
        return {
            'normal': cls.NORMAL,
            'anomaly': cls.ANOMALY,
            'tp': cls.TP,
            'fp': cls.FP,
            'tn': cls.TN,
            'fn': cls.FN,
        }


# Alias for backwards compatibility
COLORS = ColorPalette.get_all()


# =============================================================================
# Style Configuration
# =============================================================================

class VizStyle(Enum):
    """Presets cho style - gọi .apply() để áp dụng."""
    
    PAPER = "paper"           # For academic papers (font sizes smaller)
    PRESENTATION = "pres"     # For presentations (larger fonts)
    DEFAULT = "default"       # Default style
    
    def apply(self):
        """Áp dụng style cho matplotlib."""
        sns.set_style('whitegrid')
        
        if self.value == "paper":
            plt.rcParams.update({
                'font.family': 'DejaVu Sans',
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.titlesize': 10,
                'axes.labelsize': 8,
                'xtick.labelsize': 7,
                'ytick.labelsize': 7,
                'legend.fontsize': 7,
                'figure.dpi': 150,
                'font.size': 8,
            })
        elif self.value == "pres":
            plt.rcParams.update({
                'font.family': 'DejaVu Sans',
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.titlesize': 14,
                'axes.labelsize': 11,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.dpi': 100,
                'font.size': 11,
            })
        else:
            plt.rcParams.update({
                'font.family': 'DejaVu Sans',
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.titlesize': 11,
                'axes.labelsize': 9,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 8,
                'figure.dpi': 150,
            })


# =============================================================================
# Helper Functions
# =============================================================================

def save_fig(fig: plt.Figure, 
             out_path: str | Path,
             dpi: int = 150,
             bbox_inches: bool = True,
             facecolor: str = 'white',
             timestamp: bool = True) -> Path:
    """
    Lưu figure với defaults nhất quán.
    
    Args:
        fig: Matplotlib figure object
        out_path: Output path (có thể là string hoặc Path)
        dpi: Resolution (default 150)
        bbox_inches: Whether to use tight bbox (default True)
        facecolor: Background color (default white)
        timestamp: Whether to add timestamp to filename
        
    Returns:
        Path đã được lưu
        
    Usage:
        save_fig(fig, 'results/plot.png')
        save_fig(fig, Path('results/plot.png'), dpi=300)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if timestamp:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        stem = out_path.stem
        suffix = out_path.suffix
        out_path = out_path.parent / f"{stem}_{ts}{suffix}"
    
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight' if bbox_inches else None,
                facecolor=facecolor)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def format_axis_formatters(ax: plt.Axes,
                           x_fmt: str = 'plain',
                           y_fmt: str = 'plain',
                           eng_format: bool = False) -> None:
    """
    Áp dụng formatter cho axis.
    
    Args:
        ax: Matplotlib axes
        x_fmt: Format cho x-axis ('plain', 'sci', 'percent')
        y_fmt: Format cho y-axis
        eng_format: Sử dụng Engineering notation (1K, 1M, etc.)
    """
    if eng_format:
        ax.xaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
    else:
        if x_fmt == 'sci':
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
        if y_fmt == 'sci':
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))


def create_grid(n_panels: int, 
                n_cols: int = 3,
                figsize: Tuple[float, float] = (16, 10),
                title: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Tạo grid of subplots nhất quán.
    
    Args:
        n_panels: Số lượng panels cần tạo
        n_cols: Số cột (default 3)
        figsize: Figure size (width, height)
        title: Title cho figure
        
    Returns:
        (fig, axes) tuple
        
    Usage:
        fig, axes = create_grid(6, n_cols=3, title='My Analysis')
    """
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array
    if n_panels == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.flatten()
    
    # Hide unused axes
    for i in range(n_panels, len(axes)):
        axes[i].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    return fig, axes


def create_metrics_text(config: dict,
                       metrics: dict,
                       n_normal: int,
                       n_anomaly: int,
                       extra_lines: Optional[List[str]] = None) -> str:
    """
    Tạo text block cho hiển thị metrics.
    
    Args:
        config: Dict chứa config parameters
        metrics: Dict chứa metrics (f1, precision, recall, etc.)
        n_normal: Số lượng normal samples
        n_anomaly: Số lượng anomaly samples
        extra_lines: Extra lines để thêm vào text
        
    Returns:
        Formatted text string
        
    Usage:
        text = create_metrics_text(
            config={'memory_len': 50000, 'k': 10},
            metrics={'f1': 0.85, 'precision': 0.9},
            n_normal=97000,
            n_anomaly=3000
        )
    """
    lines = []
    
    # Config section
    if config:
        lines.append(f"Configuration:")
        for k, v in config.items():
            lines.append(f"  {k}: {v}")
        lines.append(f"{'─'*28}")
    
    # Data section
    total = n_normal + n_anomaly
    lines.append(f"Dataset")
    lines.append(f"  Normal:    {n_normal:,}")
    lines.append(f"  Anomaly:   {n_anomaly:,} ({n_anomaly/total*100:.1f}%)")
    lines.append(f"{'─'*28}")
    
    # Metrics section
    if metrics:
        lines.append(f"Results")
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:12s}: {v:.4f}")
            else:
                lines.append(f"  {k:12s}: {v}")
    
    # Extra lines
    if extra_lines:
        lines.append(f"{'─'*28}")
        lines.extend(extra_lines)
    
    return '\n'.join(lines)


def plot_confusion_matrix(ax: plt.Axes,
                          tn: int, fp: int, fn: int, tp: int,
                          threshold: float,
                          cmap: str = 'Blues') -> None:
    """
    Vẽ confusion matrix heatmap.
    
    Args:
        ax: Matplotlib axes
        tn, fp, fn, tp: Confusion matrix values
        threshold: Threshold used for predictions
        cmap: Colormap
    """
    cm = np.array([[tn, fp], [fn, tp]])
    
    sns.heatmap(cm, ax=ax, annot=True, fmt='d', cmap=cmap,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'shrink': 0.7})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (thresh={threshold:.2f})')


def plot_roc_pr_curves(ax_roc: plt.Axes, 
                       ax_pr: plt.Axes,
                       scores: np.ndarray, 
                       labels: np.ndarray) -> Tuple[float, float]:
    """
    Vẽ ROC và PR curves cùng lúc.
    
    Args:
        ax_roc: Axes cho ROC curve
        ax_pr: Axes cho PR curve
        scores: Anomaly scores
        labels: True labels (0=normal, 1=anomaly)
        
    Returns:
        (auc_roc, auc_pr) tuple
    """
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_roc = auc(fpr, tpr)
    
    ax_roc.plot(fpr, tpr, color=COLORS.get('normal', '#4A90D9'), lw=2,
                label=f'AUC={auc_roc:.4f}')
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1, label='Random')
    ax_roc.fill_between(fpr, 0, tpr, alpha=0.1, color=COLORS.get('normal', '#4A90D9'))
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc='lower right')
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    
    # PR Curve
    prec, rec, _ = precision_recall_curve(labels, scores)
    auc_pr = auc(rec, prec)
    baseline = labels.mean()
    
    ax_pr.plot(rec, prec, color=COLORS.get('anomaly', '#E24A33'), lw=2,
               label=f'AUC-PR={auc_pr:.4f}')
    ax_pr.axhline(baseline, color='gray', linestyle='--', lw=1,
                  label=f'Baseline={baseline:.3f}')
    ax_pr.fill_between(rec, 0, prec, alpha=0.1, color=COLORS.get('anomaly', '#E24A33'))
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend(loc='upper right')
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    
    return auc_roc, auc_pr


def plot_score_distribution(ax: plt.Axes,
                            scores: np.ndarray,
                            labels: np.ndarray,
                            threshold: Optional[float] = None,
                            bins: Optional[np.ndarray] = None) -> None:
    """
    Vẽ score distribution cho normal và anomaly.
    
    Args:
        ax: Matplotlib axes
        scores: Anomaly scores
        labels: True labels
        threshold: Optional threshold line
        bins: Optional bins array
    """
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    if bins is None:
        bins = np.linspace(scores.min(), np.percentile(scores, 99.5), 80)
    
    ax.hist(normal_scores, bins=bins, alpha=0.65, 
            label=f'Normal ({len(normal_scores):,})',
            color=COLORS.get('normal', '#4A90D9'), density=True)
    ax.hist(anomaly_scores, bins=bins, alpha=0.65,
            label=f'Anomaly ({len(anomaly_scores):,})',
            color=COLORS.get('anomaly', '#E24A33'), density=True)
    
    if threshold is not None:
        ax.axvline(threshold, color='#222', linestyle='--', lw=2,
                   label=f'Threshold = {threshold:.2f}')
    
    ax.set_xlabel('Anomaly Score (L1 kNN distance)')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.legend(loc='upper right')
