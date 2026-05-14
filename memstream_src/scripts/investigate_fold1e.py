#!/usr/bin/env python3
"""Scan ALL features across ALL folds for data quality issues."""
import numpy as np
from pathlib import Path

cache = Path('c:/proj/ldt/results/v5_clean/fold_cache')

# Config
MIN_STD_TRAIN = 1e-4  # Feature with train std below this is problematic
MIN_SCALE = 1e-4      # Scaler scale below this is problematic

print(f"{'Fold':>5} {'Feat':>5} {'T_std':>12} {'S_scale':>12} {'T_unique':>8} "
      f"{'Te_mean':>10} {'Shift':>10} {'Issue':>30}")

issues = []
for fold in range(1, 12):
    fd = cache / f'fold_{fold:02d}'
    X_train = np.load(fd / 'X_train.npy')
    scaler_mean = np.load(fd / 'scaler_mean.npy')
    scaler_scale = np.load(fd / 'scaler_scale.npy')

    for diff in ['easy']:
        X_test = np.load(fd / diff / 'X_test.npy')
        y = np.load(fd / diff / 'y_labels.npy')

        for i in range(25):
            train_std = X_train[:, i].std()
            train_unique = len(np.unique(X_train[:, i]))
            test_mean = X_test[:, i].mean()
            shift = abs(X_train[:, i].mean() - test_mean)

            issue = ""
            if train_std < MIN_STD_TRAIN:
                issue = f"ZERO_TRAIN_STD({train_std:.2e})"
            elif scaler_scale[i] < MIN_SCALE:
                issue = f"SMALL_SCALE({scaler_scale[i]:.2e})"

            if issue or train_std < 1e-3:
                issues.append({
                    'fold': fold, 'feat': i,
                    'train_std': train_std,
                    'scaler_scale': scaler_scale[i],
                    'train_unique': train_unique,
                    'test_mean': test_mean,
                    'shift': shift,
                    'issue': issue,
                    'anomaly_rate': y.mean()
                })
                print(f"  {fold:5d} {i:5d} {train_std:12.6f} {scaler_scale[i]:12.6f} "
                      f"{train_unique:8d} {test_mean:10.4f} {shift:10.4f} {issue:>30}")

print(f"\nTotal features with issues: {len(issues)}")
print()

# Focus on Fold 1 fix
print("=== FOLD 1 FIX ANALYSIS ===")
fd1 = cache / 'fold_01'
X_train1 = np.load(fd1 / 'X_train.npy')
scaler_mean = np.load(fd1 / 'scaler_mean.npy')
scaler_scale = np.load(fd1 / 'scaler_scale.npy')

# Option A: Set small scale to 1.0
# Option B: Remove feature
# Option C: Scale-aware normalization

print("Feature 14: current approach vs fix options")
f14_scale = scaler_scale[14]
print(f"  Current scale: {f14_scale:.6f}")
print(f"  Option A (min_std threshold=1e-3): if scale < 1e-3 -> set to 1.0")
print(f"  Option B (use median scale): median={np.median(scaler_scale):.4f}")

# What would Option A do?
# If scale < 1e-3 -> set scale=1.0
# After normalization: (42.14 - scaler_mean[14]) / 1.0 = 42.14 - 1.0 = 41.14
# After normalization: 42.14 / 1.0 = 41.14 - still huge
# But at least it's not 1778x dominant

# Option C: Feature-wise min_scale
# For each feature: max(scale, 1e-3)
min_scale = 1e-3
scaler_scale_fixed = np.maximum(scaler_scale, min_scale)

# Rescale Fold 1 test with fixed scale
X_test_e = np.load(fd1 / 'easy' / 'X_test.npy')
y_e = np.load(fd1 / 'easy' / 'y_labels.npy')

X_orig = (X_test_e - scaler_mean) / scaler_scale
X_fixed = (X_test_e - scaler_mean) / scaler_scale_fixed

print(f"\nFold 1 easy - Feature 14 after normalization:")
print(f"  Original (scale={f14_scale:.4f}): mean={X_orig[:,14].mean():.2f}, "
      f"std={X_orig[:,14].std():.2f}, max={X_orig[:,14].max():.2f}")
print(f"  Fixed (min_scale={min_scale}): mean={X_fixed[:,14].mean():.2f}, "
      f"std={X_fixed[:,14].std():.2f}, max={X_fixed[:,14].max():.2f}")

# Check impact on other features
print(f"\nImpact of Option C on ALL features:")
for i in range(25):
    if i == 14:
        continue
    orig_mean = X_orig[:, i].mean()
    fixed_mean = X_fixed[:, i].mean()
    diff = abs(orig_mean - fixed_mean)
    if diff > 1e-6:
        print(f"  Feature {i}: changed by {diff:.6f}")

print()

# Check: what is the range of Feature 14 in test (anomaly vs normal)?
print("Feature 14 in test - anomaly vs normal:")
X_anom = X_test_e[y_e == 1]
X_norm = X_test_e[y_e == 0]
print(f"  Normal Feature 14: mean={X_anom[:, 14].mean():.4f}, std={X_anom[:, 14].std():.4f}")
print(f"  Anomaly Feature 14: mean={X_norm[:, 14].mean():.4f}, std={X_norm[:, 14].std():.4f}")

# After normalization
print(f"  Normal Feature 14 (orig norm): mean={X_orig[y_e==1, 14].mean():.2f}")
print(f"  Normal Feature 14 (fixed norm): mean={X_fixed[y_e==1, 14].mean():.2f}")

# But the REAL fix: check if Feature 14 can distinguish anomalies
from sklearn.metrics import roc_auc_score
print(f"\nFeature 14 AUC-ROC (Fold 1 easy): {roc_auc_score(y_e, X_orig[:, 14]):.4f}")
print(f"Feature 14 AUC-ROC (Fold 1 easy, fixed scale): {roc_auc_score(y_e, X_fixed[:, 14]):.4f}")
