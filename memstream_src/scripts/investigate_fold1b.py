#!/usr/bin/env python3
"""Deep investigation of Fold 1 failure root cause."""
import numpy as np
from pathlib import Path

cache = Path('c:/proj/ldt/results/v5_clean/fold_cache')

# ===== FIND THE ZERO-VARIANCE FEATURE =====
print("=== FINDING ZERO-VARIANCE FEATURE ===")
fd1 = cache / 'fold_01'
X_train1 = np.load(fd1 / 'X_train.npy')
stds = X_train1.std(axis=0)
zero_var_feat = np.where(stds < 1e-6)[0]
print(f"Features with near-zero std: {zero_var_feat}")
for f in zero_var_feat:
    col = X_train1[:, f]
    print(f"  Feature {f}: mean={col.mean():.6f}, std={stds[f]:.2e}, "
          f"unique_values={len(np.unique(col))}, range=[{col.min():.4f}, {col.max():.4f}]")

print()

# ===== CHECK ALL FOLDS FOR ZERO-VARIANCE FEATURES =====
print("=== ZERO-VARIANCE FEATURES ACROSS ALL FOLDS ===")
for fold in range(1, 12):
    fd = cache / f'fold_{fold:02d}'
    X_train = np.load(fd / 'X_train.npy')
    stds = X_train.std(axis=0)
    zero_feats = np.where(stds < 1e-6)[0]
    print(f"Fold {fold:02d}: zero-var features = {list(zero_feats)}, "
          f"min_std = {stds.min():.2e}, max_std = {stds.max():.4f}")

print()

# ===== TRAIN-TEST SHIFT ANALYSIS =====
print("=== TRAIN-TEST DISTRIBUTION SHIFT (ALL FOLDS) ===")
for fold in range(1, 12):
    fd = cache / f'fold_{fold:02d}'
    X_train = np.load(fd / 'X_train.npy')
    X_test = np.load(fd / 'easy' / 'X_test.npy')

    train_mean = X_train.mean(axis=0)
    test_mean = X_test.mean(axis=0)
    shift = np.abs(train_mean - test_mean).mean()
    max_shift = np.abs(train_mean - test_mean).max()
    print(f"Fold {fold:02d}: mean_shift={shift:.4f}, max_shift={max_shift:.4f}")

print()

# ===== FOLD 1 SPECIFIC: WHICH FEATURES CAUSE THE SHIFT =====
print("=== FOLD 1: WHICH FEATURES CAUSE THE SHIFT ===")
fd1 = cache / 'fold_01'
X_train1 = np.load(fd1 / 'X_train.npy')
X_test1 = np.load(fd1 / 'easy' / 'X_test.npy')

for diff in ['easy', 'medium', 'hard']:
    X_test_diff = np.load(fd1 / diff / 'X_test.npy')
    y = np.load(fd1 / diff / 'y_labels.npy')

    # Overall shift
    train_mean = X_train1.mean(axis=0)
    test_mean = X_test_diff.mean(axis=0)
    shift_per_feat = np.abs(train_mean - test_mean)

    print(f"Fold 1 - {diff}: anomaly_rate={y.mean()*100:.2f}%, "
          f"mean_shift={shift_per_feat.mean():.4f}, max_shift={shift_per_feat.max():.4f}")

    # Top 5 shifted features
    top5 = np.argsort(shift_per_feat)[-5:][::-1]
    for f in top5:
        print(f"  Feature {f}: shift={shift_per_feat[f]:.4f} "
              f"(train_mean={train_mean[f]:.4f}, test_mean={test_mean[f]:.4f})")

    # Anomaly distribution: are anomalies in shifted regions?
    X_anom = X_test_diff[y == 1]
    X_norm = X_test_diff[y == 0]
    print(f"  Anomalies: {len(X_anom)}, Normal: {len(X_norm)}")
    if len(X_anom) > 0:
        print(f"  Anomaly mean: {X_anom.mean(axis=0)[:3]}")
        print(f"  Normal mean:  {X_norm.mean(axis=0)[:3]}")
    print()
