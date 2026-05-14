#!/usr/bin/env python3
"""Verify the data quality fix for Fold 1 Feature 14."""
import numpy as np
from pathlib import Path

cache = Path('c:/proj/ldt/results/v5_clean/fold_cache')

fd1 = cache / 'fold_01'
X_train = np.load(fd1 / 'X_train.npy').astype(np.float32)
X_test = np.load(fd1 / 'easy' / 'X_test.npy').astype(np.float32)
y = np.load(fd1 / 'easy' / 'y_labels.npy')
scaler_mean = np.load(fd1 / 'scaler_mean.npy')
scaler_scale = np.load(fd1 / 'scaler_scale.npy')

train_std = X_train.std(axis=0)
bad_features = np.where(train_std < 1e-4)[0]

print(f"Bad features detected: {list(bad_features)}")
print()

# BEFORE FIX
print("=== BEFORE FIX (Fold 1 easy, Feature 14) ===")
print(f"  Train Feature 14: mean={X_train[:, 14].mean():.6f}, std={X_train[:, 14].std():.6e}")
print(f"  Test Feature 14:  mean={X_test[:, 14].mean():.4f}, std={X_test[:, 14].std():.4f}")
print(f"  Scaler scale[14]={scaler_scale[14]:.6f}")

# APPLY FIX
X_train_fixed = X_train.copy()
X_test_fixed = X_test.copy()
ss_fixed = scaler_scale.copy()

safe_scale = 1.0
for f in bad_features:
    old_scale = ss_fixed[f]
    raw_train = X_train_fixed[:, f] * old_scale + scaler_mean[f]
    raw_test = X_test_fixed[:, f] * old_scale + scaler_mean[f]

    X_train_fixed[:, f] = (raw_train - scaler_mean[f]) / safe_scale
    X_test_fixed[:, f] = (raw_test - scaler_mean[f]) / safe_scale
    ss_fixed[f] = safe_scale

    print(f"\n  Fixed Feature {f}:")
    print(f"    Old scale={old_scale:.4f} -> New scale={safe_scale:.4f}")
    print(f"    Train: {X_train_fixed[:, f].mean():.6f} (unchanged)")
    print(f"    Test:  BEFORE={X_test[:, f].mean():.4f} -> AFTER={X_test_fixed[:, f].mean():.4f}")

# AFTER FIX
print()
print("=== AFTER FIX (Fold 1 easy, Feature 14) ===")
print(f"  Train Feature 14: mean={X_train_fixed[:, 14].mean():.6f}, std={X_train_fixed[:, 14].std():.6e}")
print(f"  Test Feature 14:  mean={X_test_fixed[:, 14].mean():.4f}, std={X_test_fixed[:, 14].std():.4f}")

print()
print("=== OTHER FEATURES UNCHANGED? ===")
for i in [0, 1, 2, 3, 4]:
    t_orig = X_train[:, i].mean()
    t_fix = X_train_fixed[:, i].mean()
    te_orig = X_test[:, i].mean()
    te_fix = X_test_fixed[:, i].mean()
    diff = abs(t_fix - t_orig) + abs(te_fix - te_orig)
    print(f"  Feature {i}: train_diff={abs(t_fix-t_orig):.8f}, test_diff={abs(te_fix-te_orig):.8f}")

print()
print("=== TEST RANGE COMPARISON ===")
print("All 25 features, test set range (AFTER FIX):")
for i in range(25):
    mn = X_test_fixed[:, i].min()
    mx = X_test_fixed[:, i].max()
    flag = " <<< BAD" if i in bad_features else ""
    print(f"  Feature {i:2d}: [{mn:>10.4f}, {mx:>10.4f}] (width={mx-mn:>8.4f}){flag}")
