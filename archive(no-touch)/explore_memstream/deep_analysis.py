#!/usr/bin/env python3
"""
Deep analysis: Why is F1 only 0.64?
- Score distribution: normal vs anomaly
- Fraud pool analysis
- Feature-level separation
- Threshold sensitivity
- Comparison with v10 setup
"""

import sys, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

# Replicate the exact eval setup
from eval_v10_aligned import (
    clean, features, is_canary_clean, inject_realistic_fraud,
    location_to_neighborhood, get_context_id, N_FEATURES,
    StreamingEvaluator, JFK_FLAT_FARE
)
from memstream_src.core.memstream_core import MemStreamConfig

# =====================
# 1. Load & prep data
# =====================
print("=" * 70)
print("  DEEP ANALYSIS: Why is F1 only 0.64?")
print("=" * 70)

df_raw = pd.read_parquet('C:/proj/ldt/data/nyc_taxi_300k.parquet')
df_clean = clean(df_raw)

n_train, n_test = 10000, 15000
df_train = df_clean.iloc[:n_train].reset_index(drop=True)
df_test  = df_clean.iloc[n_train:n_train + n_test].reset_index(drop=True)

rng = np.random.RandomState(42)
df_test_inj, y_test = inject_realistic_fraud(
    df_test, rng, fraud_type='mixed', anomaly_rate=0.05)

X_train = features(df_train)
X_test  = features(df_test_inj)

print(f"\n[DATA]")
print(f"  Train: {len(df_train):,}  Test: {len(df_test_inj):,}")
print(f"  Anomalies: {int(y_test.sum()):,} / {len(y_test):,} ({y_test.mean()*100:.2f}%)")
print(f"  Features: {X_train.shape[1]}D")

# =====================
# 2. Analyze fraud pools
# =====================
print(f"\n[FRAUD POOL ANALYSIS]")

canary_clean = is_canary_clean(df_test)
is_standard  = (df_test['RatecodeID'].fillna(1).values == 1.0)
dist_arr     = df_test['trip_distance'].fillna(0).values
dur_arr      = df_test['dur_min'].fillna(1).values
fare_arr     = df_test['fare_amount'].fillna(0).values

type1_pool = np.where(is_standard & canary_clean & (dist_arr < 1.0))[0]
type2_pool = np.where(is_standard & canary_clean & (dist_arr >= 2.0) & (dist_arr <= 4.0))[0]
type3_pool = np.where(is_standard & canary_clean)[0]

n_anom = int(len(df_test) * 0.05)
pool1 = type1_pool[:int(n_anom * 0.60)]
pool2 = type2_pool[:int(n_anom * 0.30)]
pool3 = type3_pool[:int(n_anom * 0.10)]
pool  = np.concatenate([pool1, pool2, pool3])

print(f"  Type1 pool (dist<1mi, clean): {len(type1_pool):,} available, using {len(pool1):,}")
print(f"  Type2 pool (dist 2-4mi, clean): {len(type2_pool):,} available, using {len(pool2):,}")
print(f"  Type3 pool (any clean): {len(type3_pool):,} available, using {len(pool3):,}")
print(f"  Total fraud injected: {len(pool):,} (target: {n_anom:,})")

# Canary filter impact
non_clean = (~canary_clean).sum()
print(f"\n  Canary-clean records: {canary_clean.sum():,} / {len(df_test):,} ({canary_clean.mean()*100:.1f}%)")
print(f"  Filtered by canary: {non_clean:,} ({non_clean/len(df_test)*100:.1f}%)")

# Show actual injected fare distributions
print(f"\n[INJECTED FARE ANALYSIS (Type 1)]")
if len(pool1) > 0:
    inj_fare = fare_arr[pool1]
    print(f"  Injected fare: {inj_fare.min():.1f} - {inj_fare.max():.1f} (mean: {inj_fare.mean():.1f})")
    print(f"  Original dist: {dist_arr[pool1].min():.2f} - {dist_arr[pool1].max():.2f}")
    print(f"  Fare/mile: {(inj_fare / np.maximum(dist_arr[pool1], 0.01)).mean():.1f}")
    print(f"  Normal fare/mile (median): {(fare_arr[~y_test.astype(bool)] / np.maximum(dist_arr[~y_test.astype(bool)], 0.01)).mean():.1f}")

print(f"\n[INJECTED DURATION ANALYSIS (Type 2)]")
if len(pool2) > 0:
    inj_dur = dur_arr[pool2]
    orig_dur = df_test['dur_min'].values[pool2]  # might be modified
    print(f"  Injected duration (dur_min): {inj_dur.min():.1f} - {inj_dur.max():.1f} (mean: {inj_dur.mean():.1f})")
    print(f"  Normal duration (median): {np.median(dur_arr[~y_test.astype(bool)]):.1f}")

# =====================
# 3. Train model & get scores
# =====================
print(f"\n[MODEL TRAINING]")
cfg = MemStreamConfig()
cfg.memory_len = 256
cfg.k = 10
cfg.gamma = 0.0
cfg.seed = 42

ev = StreamingEvaluator(cfg=cfg, device='cuda')
ev.fit(X_train,
       hour_vals=X_train[:, 9].astype(int),
       dow_vals=X_train[:, 10].astype(int))

nb_test = np.array([location_to_neighborhood(loc) for loc in df_test_inj['PULocationID'].fillna(1).values], dtype=int)
pickup_test = pd.to_datetime(df_test_inj['tpep_pickup_datetime'], errors='coerce')
hour_test   = pickup_test.dt.hour.fillna(12).astype(int).values
dow_test    = pickup_test.dt.dayofweek.fillna(0).astype(int).values
ratecode_test = df_test_inj['RatecodeID'].fillna(1).astype(float).values

print(f"  ContextBeta thresholds:")
betas = ev.model._context_beta.betas
non_default = (betas != 0.5).sum()
print(f"    Non-default betas: {non_default} / {betas.size}")
print(f"    Beta range: [{betas.min():.2f}, {betas.max():.2f}], mean: {betas[betas!=0.5].mean():.2f}")

# =====================
# 4. Score distribution analysis
# =====================
print(f"\n[SCORE DISTRIBUTION]")

# Get streaming scores (with memory updates)
scores_stream = ev.decision_function(
    X_test, neighborhood_ids=nb_test,
    hour_vals=hour_test, dow_vals=dow_test,
    ratecode_vals=ratecode_test, update_memory=True)

norm_scores = scores_stream[y_test == 0]
anom_scores = scores_stream[y_test == 1]

print(f"  STREAMING scores:")
print(f"    Normal:  mean={norm_scores.mean():.2f}  median={np.median(norm_scores):.2f}  std={norm_scores.std():.2f}")
print(f"    Anomaly: mean={anom_scores.mean():.2f}  median={np.median(anom_scores):.2f}  std={anom_scores.std():.2f}")
print(f"    Separation (mean diff): {anom_scores.mean() - norm_scores.mean():.2f}")
print(f"    Score range: [{scores_stream.min():.2f}, {scores_stream.max():.2f}]")

# Separation analysis
for pct in [50, 75, 90, 95, 99]:
    thresh = np.percentile(scores_stream, pct)
    tp = ((scores_stream > thresh) & (y_test == 1)).sum()
    fp = ((scores_stream > thresh) & (y_test == 0)).sum()
    fn = ((scores_stream <= thresh) & (y_test == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    print(f"    Thresh @ p{pct}={thresh:.2f}: Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} TP={tp} FP={fp}")

# Anomalies below threshold
thresh_95 = np.percentile(scores_stream, 95)
missed_anom = y_test[(y_test == 1) & (scores_stream <= thresh_95)]
print(f"\n  Missed anomalies: {len(missed_anom)} / {int(y_test.sum())}")
print(f"  Detected anomalies: {int(y_test.sum()) - len(missed_anom)} / {int(y_test.sum())}")

# =====================
# 5. Per-fraud-type score analysis
# =====================
print(f"\n[SCORES BY FRAUD TYPE]")

# Determine which fraud type each anomaly is
n_anom = int(len(df_test) * 0.05)
t1_max = min(len(type1_pool), int(n_anom * 0.60))
t2_max = min(len(type2_pool), int(n_anom * 0.30))

for label_name, idx_mask in [
    ('Type1 (short-trip)', np.isin(np.where(y_test == 1)[0], pool1)),
    ('Type2 (duration)',   np.isin(np.where(y_test == 1)[0], pool2)),
    ('Type3 (ratecode)',  np.isin(np.where(y_test == 1)[0], pool3)),
]:
    if idx_mask.sum() > 0:
        s = scores_stream[y_test == 1][idx_mask]
        n = scores_stream[y_test == 0]
        print(f"  {label_name}: n={idx_mask.sum():3d}, "
              f"mean={s.mean():.2f} median={np.median(s):.2f}, "
              f"vs normal(mean={n.mean():.2f})")

# =====================
# 6. Feature-level analysis
# =====================
print(f"\n[FEATURE SEPARATION (top features)]")
X_test_scaled = ev.scaler.transform(X_test.astype(np.float64)).astype(np.float32)
normal_X = X_test_scaled[y_test == 0]
anom_X   = X_test_scaled[y_test == 1]

# Compute per-feature t-test / effect size
feature_sep = []
for i in range(X_train.shape[1]):
    n_mean, n_std = normal_X[:, i].mean(), normal_X[:, i].std() + 1e-8
    a_mean = anom_X[:, i].mean()
    effect = abs(a_mean - n_mean) / n_std
    feature_sep.append((i, effect, n_mean, a_mean, n_std))

feature_sep.sort(key=lambda x: x[1], reverse=True)
print(f"  Top 10 separating features (effect size = |anom_mean - norm_mean| / norm_std):")
for i, eff, nm, am, ns in feature_sep[:10]:
    print(f"    F{i:02d}: effect={eff:.3f}  normal_mean={nm:8.3f}  anom_mean={am:8.3f}  std={ns:.3f}")

# =====================
# 7. Compare with v10 score distributions
# =====================
print(f"\n[COMPARISON WITH v10 BENCHMARK]")
print(f"  v10 mixed (MemStream): F1=0.8854 AUC-PR=0.9249 AUC-ROC=0.9710")
print(f"  Ours streaming:        F1=0.6448 AUC-PR=0.6035 AUC-ROC=0.9499")
print(f"  Gap:                   F1={0.8854-0.6448:.4f}  AUC-PR={0.9249-0.6035:.4f}  AUC-ROC={0.9710-0.9499:.4f}")
print()
print(f"  Key v10 differences:")
print(f"    - v10 uses MULTIPLE MONTHS for train (Jan-May) and tests on ONE MONTH (Jun)")
print(f"    - v10 uses 10-fold evaluation (5 folds x 2 seeds)")
print(f"    - v10's fraud pool is pre-filtered by canary_clean mask BEFORE injection")
print(f"    - v10's train set is 10K from multiple months (temporal diversity)")
print(f"    - v10's ContextBeta has 80 thresholds (10 neighborhoods x 8 context cells)")

# =====================
# 8. Temporal analysis
# =====================
print(f"\n[TEMPORAL SCORE PATTERN]")
# Check if scores change over time (streaming adaptation)
window = 1500
for t_start in range(0, 15001, window):
    t_end = min(t_start + window, 15000)
    mask_n = (y_test == 0) & (np.arange(len(y_test)) >= t_start) & (np.arange(len(y_test)) < t_end)
    mask_a = (y_test == 1) & (np.arange(len(y_test)) >= t_start) & (np.arange(len(y_test)) < t_end)
    if mask_n.sum() > 0:
        print(f"  t={t_start:5d}-{t_end:5d}: norm_mean={scores_stream[mask_n].mean():.2f}  "
              f"anom_mean={scores_stream[mask_a].mean() if mask_a.sum()>0 else 0:.2f}  "
              f"(n={mask_n.sum()}, a={mask_a.sum()})")

# =====================
# 9. Threshold optimization
# =====================
print(f"\n[THRESHOLD OPTIMIZATION]")
best_f1 = 0
best_thresh = 0
for thresh in np.linspace(scores_stream.min(), scores_stream.max(), 200):
    y_pred = (scores_stream > thresh).astype(int)
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print(f"  Best F1={best_f1:.4f} at threshold={best_thresh:.2f}")
print(f"  Default (p95) F1=0.6448 at threshold=1133.08")
print(f"  Potential improvement: {best_f1 - 0.6448:.4f}")

# =====================
# 10. Memory usage analysis
# =====================
print(f"\n[MEMORY USAGE]")
print(f"  Memory size: {cfg.memory_len}")
print(f"  Records stored: {ev.model.memory.count}")
print(f"  Memory full: {ev.model.memory._is_full}")

# Analyze memory content variance
M = ev.model.memory.count
mem = ev.model.memory.memory[:M].cpu().numpy()
print(f"  Memory latent mean: {mem.mean(axis=0)[:5]}")
print(f"  Memory latent std:   {mem.std(axis=0)[:5]}")
