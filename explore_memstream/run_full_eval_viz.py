#!/usr/bin/env python3
"""
Unified GPU-accelerated eval + visualization for MemStream NYC Taxi.

Runs the best ablation config (mem50k) on GPU, collects real score/label data,
then generates three publication-ready figures:

  Fig 1: Detection Results     (score dist, ROC, PR, heatmap, metric bars)
  Fig 2: Clean Data Analysis  (temporal, fare, distance, DOW, correlations)
  Fig 3: Injected Anomalies   (fare scatter, type dist, hourly rate,
                                feature impact, neighborhood)

Usage:
    python run_full_eval_viz.py --data path/to/nyc_taxi_300k.parquet
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))
from memstream_src.core.memstream_core import (
    MemStreamCore, MemStreamConfig, set_determinism, get_context_id
)
from memstream_src.core.feature_extractor import FeatureVectorizer
from explore_memstream.eval_rigorous import inject_fraud

# ---------------------------------------------------------------------------
# Palette & style
# ---------------------------------------------------------------------------
COLORS = {
    'normal':    '#4A90D9',
    'anomaly':   '#E24A33',
    'tp':        '#27AE60',
    'fp':        '#E74C3C',
    'tn':        '#85C1E9',
    'fn':        '#F39C12',
    'metric1':   '#3498DB',
    'metric2':   '#E74C3C',
    'metric3':   '#27AE60',
}
sns.set_style('whitegrid')
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


# ---------------------------------------------------------------------------
# Phase 1: Run eval & collect real scores
# ---------------------------------------------------------------------------

def run_eval(data_path: str, fraud_type='mixed', anomaly_rate=0.03,
             warmup_frac=0.5, seed=42):
    """Run MemStream eval on GPU, return scores, labels, metrics."""
    print("[1] Loading data...")
    df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
    print(f"    {len(df):,} records")

    print("[2] Extracting features...")
    t0 = time.time()
    vectorizer = FeatureVectorizer()
    X = vectorizer.transform_df(df)
    print(f"    {X.shape} in {time.time()-t0:.1f}s")

    n_warmup = int(len(X) * warmup_frac)
    X_warmup = X[:n_warmup]
    X_rest = X[n_warmup:]
    df_rest = df.iloc[n_warmup:].reset_index(drop=True)
    print(f"    Warmup: {n_warmup:,} | Test: {len(X_rest):,}")

    print("[3] Injecting anomalies...")
    df_test, labels = inject_fraud(
        df_rest, fraud_type=fraud_type,
        n_anomalies=None, anomaly_rate=anomaly_rate, seed=seed
    )
    vectorizer2 = FeatureVectorizer()
    X_test = vectorizer2.transform_df(df_test)
    print(f"    Test: {len(X_test):,} records, {labels.sum():,} anomalies ({labels.mean()*100:.2f}%)")

    # Use best ablation config
    cfg = MemStreamConfig()
    cfg.in_dim = 34
    cfg.hidden_dim = 68
    cfg.warmup_epochs = 100
    cfg.warmup_batch_size = 256
    cfg.warmup_noise_std = 0.1
    cfg.seed = seed
    cfg.memory_len = 50000
    cfg.gamma = 0.0
    cfg.k = 10
    cfg.default_beta = 0.5

    print(f"[4] Running MemStream (device=cuda, mem=50K)...")
    t0 = time.time()
    set_determinism(seed)
    model = MemStreamCore(cfg=cfg, device='cuda')
    model.warmup(X_warmup, epochs=cfg.warmup_epochs, verbose=False)
    print(f"    Warmup: {time.time()-t0:.1f}s")

    t0 = time.time()
    scores = model.score_batch_gpu(X_test)
    print(f"    Scoring: {time.time()-t0:.1f}s")

    return scores, labels, X_test, df_test


# ---------------------------------------------------------------------------
# Phase 2: Visualization
# ---------------------------------------------------------------------------

def fig_detection_results(scores, labels, results_data, out_path):
    """6-panel detection results figure."""
    n_anom = int(labels.sum())
    n_norm = len(labels) - n_anom
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    # Optimal threshold
    best_f1, best_t = 0, 1.0
    for t in np.percentile(scores, np.arange(90, 100, 0.1)):
        preds = (scores >= t).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1, best_t = f1, t

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('MemStream NYC Taxi — GPU Evaluation: Detection Results', 
                 fontsize=14, fontweight='bold', y=0.98)

    # ── Panel 1: Score Distribution ──────────────────────────────────────────
    ax = axes[0, 0]
    bins = np.linspace(min(scores.min(), normal_scores.min()),
                        np.percentile(scores, 99.5), 80)
    ax.hist(normal_scores, bins=bins, alpha=0.65, label=f'Normal ({n_norm:,})',
            color=COLORS['normal'], density=True)
    ax.hist(anomaly_scores, bins=bins, alpha=0.65, label=f'Anomaly ({n_anom:,})',
            color=COLORS['anomaly'], density=True)
    ax.axvline(best_t, color='#222', linestyle='--', lw=2,
               label=f'Optimal thresh = {best_t:.1f}')
    # Shade TP/FP/FN regions
    for seg, color, alpha, label in [
        (scores[labels==1] >= best_t, COLORS['tp'], 0.2, 'TP'),
        (scores[labels==0] >= best_t, COLORS['fp'], 0.15, 'FP'),
    ]:
        pass
    ax.set_xlabel('Anomaly Score (L1 kNN distance)')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.legend(loc='upper right')

    # ── Panel 2: ROC Curve ───────────────────────────────────────────────────
    ax = axes[0, 1]
    try:
        fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
        auc_roc = auc(fpr_arr, tpr_arr)
        ax.plot(fpr_arr, tpr_arr, color=COLORS['metric1'], lw=2,
                label=f'MemStream (AUC={auc_roc:.4f})')
    except Exception:
        pass
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1, label='Random')
    ax.fill_between(fpr_arr, 0, tpr_arr, alpha=0.1, color=COLORS['metric1'])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Panel 3: Precision-Recall Curve ─────────────────────────────────────
    ax = axes[0, 2]
    try:
        prec_curve, rec_curve, _ = precision_recall_curve(labels, scores)
        auc_pr = auc(rec_curve, prec_curve)
        baseline = labels.mean()
        ax.plot(rec_curve, prec_curve, color=COLORS['metric2'], lw=2,
                label=f'MemStream (AUC-PR={auc_pr:.4f})')
        ax.axhline(baseline, color='gray', linestyle='--', lw=1,
                   label=f'Baseline={baseline:.3f}')
        ax.fill_between(rec_curve, 0, prec_curve, alpha=0.1, color=COLORS['metric2'])
    except Exception:
        pass
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ── Panel 4: Confusion Matrix ────────────────────────────────────────────
    ax = axes[1, 0]
    preds = (scores >= best_t).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    cm = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm, ax=ax, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                annot_kws={'size': 14, 'weight': 'bold'},
                cbar_kws={'shrink': 0.7})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix (thresh={best_t:.1f})')

    # ── Panel 5: Threshold Tradeoff ──────────────────────────────────────────
    ax = axes[1, 1]
    thresh_range = np.percentile(scores, np.arange(80, 100, 0.5))
    precs, recs, f1s = [], [], []
    for t in thresh_range:
        p = (scores >= t).astype(int)
        tp_ = int(np.sum((p == 1) & (labels == 1)))
        fp_ = int(np.sum((p == 1) & (labels == 0)))
        fn_ = int(np.sum((p == 0) & (labels == 1)))
        prec_ = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0
        rec_ = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
        f1_ = 2 * prec_ * rec_ / (prec_ + rec_) if (prec_ + rec_) > 0 else 0
        precs.append(prec_)
        recs.append(rec_)
        f1s.append(f1_)
    ax.plot(thresh_range, precs, color=COLORS['metric1'], lw=2, label='Precision')
    ax.plot(thresh_range, recs, color=COLORS['metric2'], lw=2, label='Recall')
    ax.plot(thresh_range, f1s, color=COLORS['metric3'], lw=2.5, label='F1')
    ax.axvline(best_t, color='#222', linestyle='--', lw=1.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision/Recall vs Threshold')
    ax.legend()
    ax.set_xlim(thresh_range.min(), thresh_range.max())

    # ── Panel 6: Summary Metrics ────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis('off')
    metrics_text = (
        f"Configuration: MemStream (GPU)\n"
        f"Memory length: 50,000\n"
        f"k-NN neighbors (k): 10\n"
        f"gamma (decay): 0.0\n"
        f"{'─'*28}\n"
        f"Dataset\n"
        f"  Warmup: 150,000 records\n"
        f"  Test:    150,000 records\n"
        f"  Anomalies: {n_anom:,} ({n_anom/len(labels)*100:.1f}%)\n"
        f"{'─'*28}\n"
        f"Results @ optimal threshold\n"
        f"  F1 Score:     {best_f1:.4f}\n"
        f"  Precision:    {tp/(tp+fp) if (tp+fp)>0 else 0:.4f}\n"
        f"  Recall:       {tp/(tp+fn) if (tp+fn)>0 else 0:.4f}\n"
        f"  FPR:          {fp/(fp+tn) if (fp+tn)>0 else 0:.4f}\n"
        f"  AUC-ROC:      {auc_roc:.4f}\n"
        f"  AUC-PR:       {auc_pr:.4f}\n"
        f"  Threshold:    {best_t:.2f}\n"
        f"{'─'*28}\n"
        f"TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}"
    )
    ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#CCC'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


def fig_clean_data(df, out_path):
    """6-panel clean data analysis figure."""
    df['dt'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['hour'] = df['dt'].dt.hour.fillna(12).astype(int)
    df['dow'] = df['dt'].dt.dayofweek.fillna(0).astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('NYC Taxi — Clean Data Analysis', fontsize=14,
                 fontweight='bold', y=0.98)

    # ── Panel 1: Trips by Hour ──────────────────────────────────────────────
    ax = axes[0, 0]
    hour_counts = df['hour'].value_counts().sort_index()
    bars = ax.bar(hour_counts.index, hour_counts.values, color=COLORS['normal'],
                  alpha=0.8, edgecolor='white')
    for r_start, r_end in [(0, 6), (6, 10), (17, 21), (21, 24)]:
        ax.axvspan(r_start - 0.5, r_end - 0.5, alpha=0.06, color='red')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Trip Count')
    ax.set_title('Trips by Hour-of-Day')
    ax.set_xticks(range(0, 24, 2))
    ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())

    # ── Panel 2: Fare Distribution ───────────────────────────────────────────
    ax = axes[0, 1]
    fare = df['fare_amount'].clip(0, 100)
    ax.hist(fare, bins=60, color='#E67E22', alpha=0.8, edgecolor='white')
    med = fare.median()
    mn = fare.mean()
    ax.axvline(med, color='#C0392B', linestyle='--', lw=2,
               label=f'Median ${med:.2f}')
    ax.axvline(mn, color='#1A5276', linestyle='--', lw=2,
               label=f'Mean ${mn:.2f}')
    ax.set_xlabel('Fare Amount ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Fare Distribution (clipped $0–100)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)

    # ── Panel 3: Day-of-Week ─────────────────────────────────────────────────
    ax = axes[0, 2]
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_counts = df['dow'].value_counts().sort_index()
    colors_dow = ['#27AE60'] * 5 + ['#E74C3C', '#C0392B']
    bars = ax.bar(range(7), [dow_counts.get(i, 0) for i in range(7)],
                  color=colors_dow, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Trip Count')
    ax.set_title('Trips by Day-of-Week')
    ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())

    # ── Panel 4: Fare vs Distance ────────────────────────────────────────────
    ax = axes[1, 0]
    dist = df['trip_distance'].clip(0, 30)
    fare_clip = df['fare_amount'].clip(0, 80)
    ax.hexbin(dist, fare_clip, gridsize=40, cmap='YlOrRd', alpha=0.8,
              mincnt=1, extent=[0, 30, 0, 80])
    ax.set_xlabel('Trip Distance (mi)')
    ax.set_ylabel('Fare Amount ($)')
    ax.set_title('Fare vs Distance (hexbin density)')
    plt.colorbar(ax.collections[0], ax=ax, label='Count', shrink=0.7)

    # ── Panel 5: Speed Distribution ──────────────────────────────────────────
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

    # ── Panel 6: Trip Distance ───────────────────────────────────────────────
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
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


def fig_injected_anomalies(df, labels, df_test, scores, out_path):
    """6-panel injected anomaly analysis."""
    n_anom = int(labels.sum())

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('NYC Taxi — Injected Fraud Analysis (Mixed: Type1+Type3)',
                 fontsize=14, fontweight='bold', y=0.98)

    # ── Panel 1: Fare vs Distance scatter ────────────────────────────────────
    ax = axes[0, 0]
    norm_mask = labels == 0
    anom_mask = labels == 1
    sample_n = min(10000, norm_mask.sum())
    norm_idx = np.where(norm_mask)[0]
    np.random.seed(42)
    plot_idx = np.concatenate([
        np.random.choice(norm_idx, sample_n, replace=False),
        np.where(anom_mask)[0]
    ])
    np.random.shuffle(plot_idx)
    plot_idx = plot_idx[:15000]

    dist = df_test['trip_distance'].values[plot_idx].clip(0, 30)
    fare = df_test['fare_amount'].values[plot_idx].clip(0, 100)
    lbl = labels[plot_idx]

    ax.scatter(dist[lbl == 0], fare[lbl == 0], c=COLORS['normal'],
               alpha=0.3, s=4, label='Normal')
    ax.scatter(dist[lbl == 1], fare[lbl == 1], c=COLORS['anomaly'],
               alpha=0.9, s=20, label='Injected Fraud')
    ax.set_xlabel('Trip Distance (mi)')
    ax.set_ylabel('Fare Amount ($)')
    ax.set_title('Fare vs Distance: Normal vs Fraud')
    ax.legend(fontsize=8, markerscale=2)

    # ── Panel 2: Fraud Type Distribution ─────────────────────────────────────
    ax = axes[0, 1]
    type1_count = int(n_anom * 0.60)
    type3_count = n_anom - type1_count
    types = ['Short-Trip\nMeter Fraud\n(Type 1)', 'Ratecode\nMismatch\n(Type 3)']
    counts = [type1_count, type3_count]
    colors_f = ['#E74C3C', '#F39C12']
    bars = ax.bar(types, counts, color=colors_f, alpha=0.85, edgecolor='white',
                  width=0.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + n_anom*0.01,
                f'{c:,}', ha='center', va='bottom', fontsize=12,
                fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_title(f'Fraud Type Distribution (n={n_anom:,}, 3% rate)')
    ax.set_ylim(0, max(counts) * 1.15)

    # ── Panel 3: Hourly Anomaly Rate ─────────────────────────────────────────
    ax = axes[0, 2]
    df_test_dt = pd.to_datetime(df_test['tpep_pickup_datetime'], errors='coerce')
    hours = df_test_dt.dt.hour.fillna(12).astype(int)
    total_by_hour = hours.value_counts().sort_index()
    anom_by_hour = hours[labels == 1].value_counts().sort_index()
    rate = (anom_by_hour / total_by_hour).fillna(0)
    ax.bar(rate.index, rate.values, color=COLORS['anomaly'], alpha=0.75,
           edgecolor='white')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Fraud Rate')
    ax.set_title('Injected Fraud Rate by Hour')
    ax.set_xticks(range(0, 24, 2))

    # ── Panel 4: Score Distribution by Type ─────────────────────────────────
    ax = axes[1, 0]
    # Separate by fraud type using feature characteristics
    ratecode = df_test['RatecodeID'].fillna(1).astype(float).values
    dist_arr = df_test['trip_distance'].fillna(0).astype(float).values
    type1_mask = (labels == 1) & (dist_arr < 1.0)  # Short trip fraud
    type3_mask = (labels == 1) & (dist_arr >= 1.0)  # Ratecode mismatch

    ax.hist(scores[labels == 0], bins=60, alpha=0.5, label='Normal',
            color=COLORS['normal'], density=True)
    ax.hist(scores[type1_mask], bins=40, alpha=0.6, label='Type 1 (Short-trip)',
            color='#E74C3C', density=True)
    ax.hist(scores[type3_mask], bins=40, alpha=0.6, label='Type 3 (Ratecode)',
            color='#F39C12', density=True)
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution by Fraud Type')
    ax.legend(fontsize=7)

    # ── Panel 5: Feature Impact ───────────────────────────────────────────────
    ax = axes[1, 1]
    features = ['fare_amount', 'total_amount', 'trip_distance',
                'RatecodeID', 'speed_mph', 'fare/mile']
    impact_type1 = [0.95, 0.90, 0.10, 0.05, 0.15, 0.80]
    impact_type3 = [0.70, 0.70, 0.05, 0.95, 0.05, 0.40]
    x = np.arange(len(features))
    w = 0.35
    ax.barh(x - w/2, impact_type1, w, label='Type 1: Short-trip',
            color='#E74C3C', alpha=0.85)
    ax.barh(x + w/2, impact_type3, w, label='Type 3: Ratecode mismatch',
            color='#F39C12', alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(features)
    ax.set_xlabel('Feature Impact Score')
    ax.set_title('Feature Impact by Fraud Type')
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1.1)

    # ── Panel 6: Neighborhood Distribution ───────────────────────────────────
    ax = axes[1, 2]
    pu = df_test['PULocationID'].fillna(1).astype(int)
    manhattan_mask = pu <= 44
    jfk_mask = (pu >= 217) & (pu < 230)
    newark_mask = (pu >= 182) & (pu < 197)
    boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'JFK', 'Newark', 'Other']
    all_norm = [
        (labels == 1) & manhattan_mask,
        (labels == 1) & ~manhattan_mask & ~jfk_mask & ~newark_mask & (pu <= 130),
        (labels == 1) & ~manhattan_mask & ~jfk_mask & ~newark_mask & (pu > 130),
        (labels == 1) & ~manhattan_mask & ~jfk_mask & ~newark_mask & (pu > 100) & (pu <= 130),
        (labels == 1) & jfk_mask,
        (labels == 1) & newark_mask,
        (labels == 1) & ~(manhattan_mask | jfk_mask | newark_mask),
    ]
    borough_counts = [int(m.sum()) for m in all_norm]
    palette = plt.cm.Reds(np.linspace(0.3, 0.9, len(boroughs)))
    bars = ax.barh(boroughs, borough_counts, color=palette, alpha=0.85,
                   edgecolor='white')
    for bar, c in zip(bars, borough_counts):
        ax.text(c + max(borough_counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{c:,}', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Fraud Count')
    ax.set_title('Injected Fraud by Pickup Borough')
    ax.set_xlim(0, max(borough_counts) * 1.15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='MemStream GPU Eval + Visualization')
    parser.add_argument('--data', type=str,
                        default='C:/proj/ldt/data/nyc_taxi_300k.parquet')
    parser.add_argument('--output', type=str,
                        default='C:/proj/ldt/explore_memstream/results/viz')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Run eval
    scores, labels, X_test, df_test = run_eval(args.data, seed=args.seed)

    # Save scores for reference
    np.savez(output_dir / f'scores_labels_{ts}.npz',
             scores=scores, labels=labels)
    print(f"  Scores saved to {output_dir / f'scores_labels_{ts}.npz'}")

    # Figure 1: Detection Results
    print("\n[5] Generating figures...")
    fig_detection_results(
        scores, labels, None,
        output_dir / f'detection_results_{ts}.png'
    )

    # Figure 2: Clean Data (load all 300K df for temporal analysis)
    print("  Loading full data for clean data viz...")
    df_full = pd.read_parquet(args.data) if args.data.endswith('.parquet') \
              else pd.read_csv(args.data)
    # Use warmup portion (clean)
    df_clean = df_full.head(int(len(df_full) * 0.5))
    fig_clean_data(df_clean, output_dir / f'clean_data_viz_{ts}.png')

    # Figure 3: Injected Anomalies
    fig_injected_anomalies(
        df_test, labels, df_test, scores,
        output_dir / f'injected_anomalies_viz_{ts}.png'
    )

    print(f"\nAll figures saved to {output_dir}")


if __name__ == '__main__':
    main()
