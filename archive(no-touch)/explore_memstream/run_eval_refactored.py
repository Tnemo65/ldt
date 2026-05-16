#!/usr/bin/env python3
"""
Refactored MemStream GPU Evaluation + Visualization.

ĐÂY LÀ VERSION SỬ DỤNG MODULES TÁI SỬ DỤNG.
Mỗi lần cần thêm feature mới, chỉ cần import từ utils modules.

Usage:
    python run_eval_refactored.py --data path/to/data.parquet

So với version cũ:
    - eval_utils.py: Tất cả metrics logic ở đây
    - fraud_utils.py: Tất cả fraud injection logic ở đây  
    - viz_utils.py: Tất cả visualization helpers ở đây
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# IMPORT TỪ SHARED MODULES - Không copy-paste code nữa!
# =============================================================================
sys.path.insert(0, str(Path(__file__).parent))

from viz_utils import (
    VizStyle, ColorPalette, COLORS,
    save_fig, create_grid, create_metrics_text,
    plot_confusion_matrix, plot_roc_pr_curves, plot_score_distribution,
    format_axis_formatters
)

from eval_utils import (
    EvalMetrics, compute_all_metrics, find_optimal_threshold,
    sweep_thresholds, load_scores_labels, save_scores_labels
)

from fraud_utils import (
    FraudType, FraudConfig, inject_anomalies, get_anomaly_summary
)

# Import MemStream components
from memstream_src.core.memstream_core import (
    MemStreamCore, MemStreamConfig, set_determinism
)
from memstream_src.core.feature_extractor import FeatureVectorizer


# =============================================================================
# PHASE 1: Run Evaluation
# =============================================================================

def run_memstream_eval(data_path: str,
                       fraud_config: FraudConfig,
                       warmup_frac: float = 0.5,
                       mem_len: int = 50000,
                       warmup_epochs: int = 100,
                       seed: int = 42) -> tuple:
    """
    Chạy MemStream evaluation với config chuẩn.
    
    Returns:
        (scores, labels, X_test, df_test, config_dict, metrics)
    """
    print("[1] Loading data...")
    df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
    print(f"    {len(df):,} records")

    # Feature extraction
    print("[2] Extracting features...")
    t0 = time.time()
    vectorizer = FeatureVectorizer()
    X = vectorizer.transform_df(df)
    print(f"    {X.shape} in {time.time()-t0:.1f}s")

    # Split warmup/test
    n_warmup = int(len(X) * warmup_frac)
    X_warmup = X[:n_warmup]
    X_rest = X[n_warmup:]
    df_rest = df.iloc[n_warmup:].reset_index(drop=True)
    print(f"    Warmup: {n_warmup:,} | Test: {len(X_rest):,}")

    # Inject anomalies - SỬ DỤNG UTILITY
    print("[3] Injecting anomalies...")
    df_test, labels = inject_anomalies(
        df_rest,
        anomaly_rate=fraud_config.anomaly_rate,
        fraud_type=fraud_config.fraud_type,
        seed=seed
    )
    vectorizer2 = FeatureVectorizer()
    X_test = vectorizer2.transform_df(df_test)
    print(f"    Test: {len(X_test):,} records, {labels.sum():,} anomalies ({labels.mean()*100:.2f}%)")

    # MemStream config
    cfg = MemStreamConfig()
    cfg.in_dim = 34
    cfg.hidden_dim = 68
    cfg.warmup_epochs = warmup_epochs
    cfg.warmup_batch_size = 256
    cfg.warmup_noise_std = 0.1
    cfg.seed = seed
    cfg.memory_len = mem_len
    cfg.gamma = 0.0
    cfg.k = 10
    cfg.default_beta = 0.5

    print(f"[4] Running MemStream (device=cuda, mem={mem_len:,})...")
    t0 = time.time()
    set_determinism(seed)
    model = MemStreamCore(cfg=cfg, device='cuda')
    model.warmup(X_warmup, epochs=cfg.warmup_epochs, verbose=False)
    print(f"    Warmup: {time.time()-t0:.1f}s")

    t0 = time.time()
    scores = model.score_batch_gpu(X_test)
    print(f"    Scoring: {time.time()-t0:.1f}s")

    # Compute metrics - SỬ DỤNG UTILITY
    metrics = compute_all_metrics(scores, labels)
    
    config_dict = {
        'memory_len': mem_len,
        'k': cfg.k,
        'gamma': cfg.gamma,
        'warmup_epochs': warmup_epochs,
    }

    return scores, labels, X_test, df_test, config_dict, metrics


# =============================================================================
# PHASE 2: Visualization Functions
# =============================================================================

def fig_detection_results(scores, labels, config_dict, metrics, out_path):
    """6-panel detection results - SỬ DỤNG SHARED COMPONENTS."""
    n_anom = int(labels.sum())
    n_norm = len(labels) - n_anom

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('MemStream NYC Taxi — GPU Evaluation: Detection Results', 
                 fontsize=14, fontweight='bold', y=0.98)

    # ── Panel 1: Score Distribution (DÙNG UTILITY)
    plot_score_distribution(
        axes[0, 0], scores, labels,
        threshold=metrics.threshold
    )

    # ── Panel 2 & 3: ROC & PR Curves (DÙNG UTILITY)
    plot_roc_pr_curves(axes[0, 1], axes[0, 2], scores, labels)

    # ── Panel 4: Confusion Matrix (DÙNG UTILITY)
    plot_confusion_matrix(
        axes[1, 0],
        tn=metrics.tn, fp=metrics.fp, fn=metrics.fn, tp=metrics.tp,
        threshold=metrics.threshold
    )

    # ── Panel 5: Threshold Tradeoff
    sweep = sweep_thresholds(scores, labels)
    axes[1, 1].plot(sweep.thresholds, sweep.precisions, 
                    color=COLORS['normal'], lw=2, label='Precision')
    axes[1, 1].plot(sweep.thresholds, sweep.recalls,
                    color=COLORS['anomaly'], lw=2, label='Recall')
    axes[1, 1].plot(sweep.thresholds, sweep.f1s,
                    color=COLORS['tp'], lw=2.5, label='F1')
    axes[1, 1].axvline(metrics.threshold, color='#222', linestyle='--', lw=1.5)
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision/Recall vs Threshold')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(sweep.thresholds.min(), sweep.thresholds.max())

    # ── Panel 6: Summary Metrics (DÙNG UTILITY)
    axes[1, 2].axis('off')
    extra_lines = [
        f"TP={metrics.tp:,}  FP={metrics.fp:,}",
        f"TN={metrics.tn:,}  FN={metrics.fn:,}"
    ]
    text = create_metrics_text(
        config=config_dict,
        metrics={
            'F1 Score': metrics.f1,
            'Precision': metrics.precision,
            'Recall': metrics.recall,
            'FPR': metrics.fpr,
            'AUC-ROC': metrics.auc_roc,
            'AUC-PR': metrics.auc_pr,
            'Threshold': f"{metrics.threshold:.2f}",
        },
        n_normal=n_norm,
        n_anomaly=n_anom,
        extra_lines=extra_lines
    )
    axes[1, 2].text(0.1, 0.95, text, transform=axes[1, 2].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#CCC'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, out_path)


def fig_clean_data(df, out_path):
    """6-panel clean data analysis."""
    df = df.copy()
    df['dt'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['hour'] = df['dt'].dt.hour.fillna(12).astype(int)
    df['dow'] = df['dt'].dt.dayofweek.fillna(0).astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('NYC Taxi — Clean Data Analysis', fontsize=14,
                 fontweight='bold', y=0.98)

    # ── Panel 1: Trips by Hour
    ax = axes[0, 0]
    hour_counts = df['hour'].value_counts().sort_index()
    ax.bar(hour_counts.index, hour_counts.values, color=COLORS['normal'],
           alpha=0.8, edgecolor='white')
    for r_start, r_end in [(0, 6), (6, 10), (17, 21), (21, 24)]:
        ax.axvspan(r_start - 0.5, r_end - 0.5, alpha=0.06, color='red')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Trip Count')
    ax.set_title('Trips by Hour-of-Day')
    ax.set_xticks(range(0, 24, 2))
    format_axis_formatters(ax, eng_format=True)

    # ── Panel 2: Fare Distribution
    ax = axes[0, 1]
    fare = df['fare_amount'].clip(0, 100)
    ax.hist(fare, bins=60, color='#E67E22', alpha=0.8, edgecolor='white')
    ax.axvline(fare.median(), color='#C0392B', linestyle='--', lw=2,
               label=f'Median ${fare.median():.2f}')
    ax.axvline(fare.mean(), color='#1A5276', linestyle='--', lw=2,
               label=f'Mean ${fare.mean():.2f}')
    ax.set_xlabel('Fare Amount ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Fare Distribution (clipped $0–100)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)

    # ── Panel 3: Day-of-Week
    ax = axes[0, 2]
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_counts = df['dow'].value_counts().sort_index()
    colors_dow = ['#27AE60'] * 5 + ['#E74C3C', '#C0392B']
    ax.bar(range(7), [dow_counts.get(i, 0) for i in range(7)],
           color=colors_dow, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Trip Count')
    ax.set_title('Trips by Day-of-Week')
    format_axis_formatters(ax, eng_format=True)

    # ── Panel 4: Fare vs Distance
    ax = axes[1, 0]
    dist = df['trip_distance'].clip(0, 30)
    fare_clip = df['fare_amount'].clip(0, 80)
    ax.hexbin(dist, fare_clip, gridsize=40, cmap='YlOrRd', alpha=0.8,
              mincnt=1, extent=[0, 30, 0, 80])
    ax.set_xlabel('Trip Distance (mi)')
    ax.set_ylabel('Fare Amount ($)')
    ax.set_title('Fare vs Distance (hexbin density)')
    plt.colorbar(ax.collections[0], ax=ax, label='Count', shrink=0.7)

    # ── Panel 5: Speed Distribution
    ax = axes[1, 1]
    dur = df['tpep_dropoff_datetime']
    if hasattr(dur, 'dt'):
        dur_s = (pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce') -
                 df['dt']).dt.total_seconds().fillna(900)
    else:
        dur_s = df['trip_distance'] / 15.0 * 3600
    dur_s = dur_s.clip(60, 14400)
    dist_arr = df['trip_distance'].fillna(1).clip(0.01, 100)
    speed = (dist_arr / (dur_s / 3600)).clip(0, 60).fillna(0)
    ax.hist(speed[speed > 0], bins=50, color='#8E44AD', alpha=0.75,
            edgecolor='white', density=True)
    ax.axvline(speed[speed > 0].median(), color='red', linestyle='--',
               lw=2, label=f'Median {speed[speed>0].median():.1f} mph')
    ax.set_xlabel('Speed (mph)')
    ax.set_ylabel('Density')
    ax.set_title('Speed Distribution (mph)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 60)

    # ── Panel 6: Trip Distance
    ax = axes[1, 2]
    dist_all = df['trip_distance'].clip(0, 30)
    ax.hist(dist_all, bins=50, color='#16A085', alpha=0.8, edgecolor='white',
            density=True)
    ax.axvline(dist_all.median(), color='red', linestyle='--', lw=2,
               label=f'Median {dist_all.median():.1f} mi')
    ax.set_xlabel('Trip Distance (mi)')
    ax.set_ylabel('Density')
    ax.set_title('Trip Distance Distribution (clipped 0–30 mi)')
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, out_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='MemStream GPU Eval (Refactored)')
    parser.add_argument('--data', type=str,
                       default='C:/proj/ldt/data/nyc_taxi_300k.parquet')
    parser.add_argument('--output', type=str,
                       default='C:/proj/ldt/explore_memstream/results/viz')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--anomaly-rate', type=float, default=0.03)
    parser.add_argument('--fraud-type', type=str, default='mixed',
                       choices=['short_trip', 'long_trip', 'ratecode_mismatch', 
                               'night', 'mixed'])
    args = parser.parse_args()

    # Apply style
    VizStyle.DEFAULT.apply()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Build fraud config - DÙNG UTILITY
    fraud_config = FraudConfig(
        anomaly_rate=args.anomaly_rate,
        fraud_type=FraudType(args.fraud_type),
        seed=args.seed
    )

    # Run evaluation
    scores, labels, X_test, df_test, config_dict, metrics = run_memstream_eval(
        args.data,
        fraud_config=fraud_config,
        seed=args.seed
    )

    # Save scores for future use
    scores_path = output_dir / f'scores_labels_{ts}.npz'
    save_scores_labels(scores, labels, str(scores_path))

    # Generate figures
    print("\n[5] Generating figures...")
    
    # Figure 1: Detection Results
    fig_detection_results(
        scores, labels, config_dict, metrics,
        output_dir / f'detection_results_{ts}.png'
    )

    # Figure 2: Clean Data
    print("  Loading full data for clean data viz...")
    df_full = pd.read_parquet(args.data) if args.data.endswith('.parquet') \
              else pd.read_csv(args.data)
    df_clean = df_full.head(int(len(df_full) * 0.5))
    fig_clean_data(df_clean, output_dir / f'clean_data_viz_{ts}.png')

    print(f"\n{'='*60}")
    print(f"ALL DONE! Results saved to {output_dir}")
    print(f"{'='*60}")
    print(f"\nMetrics Summary:")
    print(f"  {metrics.summary()}")


if __name__ == '__main__':
    main()
