#!/usr/bin/env python3
"""Trace Feature 14 back to source - check preprocessing pipeline."""
import numpy as np
from pathlib import Path

cache = Path('c:/proj/ldt/results/v5_clean/fold_cache')

print("=== FEATURE 14: ACROSS ALL FOLDS, ALL DIFFICULTIES ===")
for fold in range(1, 12):
    fd = cache / f'fold_{fold:02d}'
    X_train = np.load(fd / 'X_train.npy')

    for diff in ['easy', 'medium', 'hard']:
        X_test = np.load(fd / diff / 'X_test.npy')
        f14_train = X_train[:, 14]
        f14_test = X_test[:, 14]

        shift = abs(f14_train.mean() - f14_test.mean())
        flag = " <<< ZERO_STD" if f14_train.std() < 1e-5 else ""
        print(f"  Fold {fold:02d}/{diff}: train_unique={len(np.unique(f14_train)):<4} "
              f"test_unique={len(np.unique(f14_test)):<4} "
              f"train_mean={f14_train.mean():>10.4f} test_mean={f14_test.mean():>10.4f} "
              f"shift={shift:>8.4f}{flag}")

print()

print("=== CHECK SCALER ORIGIN: Is scaler fitted on full fold data? ===")
# If scaler was fitted on all 80k training samples, the mean of Feature 14 would be ~0
# If scaler was fitted on full 280k (train+test), the mean would include test distribution

# Let's check: what would the scaler produce if fitted ONLY on training?
for fold in [1, 2]:
    fd = cache / f'fold_{fold:02d}'
    X_train = np.load(fd / 'X_train.npy')
    X_test_e = np.load(fd / 'easy' / 'X_test.npy')

    # What scaler_fit on training only would produce
    sm_train = X_train.mean(axis=0)
    ss_train = X_train.std(axis=0)
    ss_train = np.where(ss_train < 1e-8, 1.0, ss_train)

    # Rescale test with training-only scaler
    X_test_scaled = (X_test_e - sm_train) / ss_train

    # What the actual scaler produced
    sm_actual = np.load(fd / 'scaler_mean.npy')
    ss_actual = np.load(fd / 'scaler_scale.npy')

    print(f"Fold {fold:02d}:")
    print(f"  Train-only scaler - Feature 14: mean={sm_train[14]:.6f}, scale={ss_train[14]:.6e}")
    print(f"  Actual scaler     - Feature 14: mean={sm_actual[14]:.6f}, scale={ss_actual[14]:.6e}")
    print(f"  Test rescaled (train-only): Feature 14 mean={X_test_scaled[:,14].mean():.4f}, "
          f"std={X_test_scaled[:,14].std():.4f}")
    print(f"  Test rescaled (actual):     Feature 14 mean="
          f"{(X_test_e - sm_actual).mean(axis=0)[14] / ss_actual[14]:.4f}, "
          f"std={(X_test_e - sm_actual).std(axis=0)[14] / ss_actual[14]:.4f}")
    print()

print()

print("=== HYPOTHESIS: Is Feature 14 an index/ID feature? ===")
fd1 = cache / 'fold_01'
X_train1 = np.load(fd1 / 'X_train.npy')
X_test1 = np.load(fd1 / 'easy' / 'X_test.npy')

print(f"Fold 1 Train Feature 14: ALL unique values = {np.unique(X_train1[:, 14])}")
print(f"Fold 1 Test Feature 14:  unique values = {np.unique(X_test1[:, 14])}")

# Check if Feature 14 is monotonically increasing (would suggest index)
print(f"\nFold 1 Train Feature 14: is_sorted={np.all(np.diff(X_train1[:, 14]) >= 0)}")
print(f"Fold 1 Test Feature 14:  is_sorted={np.all(np.diff(X_test1[:, 14]) >= 0)}")

# Check ALL features for constant training values in Fold 1
print(f"\nFold 1 features with near-zero variance in TRAIN:")
for i in range(25):
    if X_train1[:, i].std() < 1e-5:
        print(f"  Feature {i}: train_const={X_train1[0, i]:.6f}, "
              f"test_mean={X_test1[:, i].mean():.4f}")

print()

print("=== CONCLUSION: Root Cause Summary ===")
print("Feature 14 in Fold 1:")
print("  - Training: ALL identical values (-0.0022)")
print("  - Test: mostly 42.14 (with a few anomalies)")
print("  - After z-score: train=0.0, test=~421,400 (extreme outlier)")
print("")
print("Impact:")
print("  - MemStream AE trained on 'flat' Feature 14, cannot learn meaningful reconstruction")
print("  - On test, Feature 14 dominates the reconstruction error (100K+ vs ~1.0 for others)")
print("  - Score distribution: anomalies vs normal are indistinguishable because Feature 14 overwhelms everything")
print("")
print("Fix Options:")
print("  1. Remove near-zero-variance features BEFORE training (recommended)")
print("  2. Replace z-score with RobustScaler for feature 14")
print("  3. Add min_std threshold (e.g., std < 1e-4 -> set scale to 1.0)")
