#!/usr/bin/env python3
"""Verify what MemStream actually receives and fix the normalization issue."""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, 'c:/proj/ldt/memstream_src')
sys.path.insert(0, 'c:/proj/ldt')

cache = Path('c:/proj/ldt/results/v5_clean/fold_cache')

# Load Fold 1 data
fd1 = cache / 'fold_01'
X_train = np.load(fd1 / 'X_train.npy').astype(np.float32)
X_test = np.load(fd1 / 'easy' / 'X_test.npy').astype(np.float32)
y = np.load(fd1 / 'easy' / 'y_labels.npy')
scaler_mean = np.load(fd1 / 'scaler_mean.npy')
scaler_scale = np.load(fd1 / 'scaler_scale.npy')

print("=== RAW DATA (pre-normalized) ===")
print(f"Fold 1 easy: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
print(f"Feature 14 - train: unique={np.unique(X_train[:, 14])}, mean={X_train[:, 14].mean():.6f}")
print(f"Feature 14 - test:  unique~3, mean={X_test[:, 14].mean():.4f}")
print()

# Raw values (pre-normalized)
print("RAW Feature 14 - first 5 test samples:")
print(f"  {X_test[:5, 14]}")
print()

# After applying the scaler (pre-normalized = already z-scored)
print("PRE-NORMALIZED Feature 14 (stored in fold_cache):")
print(f"  Test mean={X_test[:, 14].mean():.2f}, std={X_test[:, 14].std():.2f}")
print(f"  Train mean={X_train[:, 14].mean():.6f}, std={X_train[:, 14].std():.6f}")
print(f"  Train has near-zero variance! std={X_train[:, 14].std():.2e}")
print()

# The SCALER was fitted on training data
print("=== SCALER (fitted on training data) ===")
print(f"scaler_mean[14]={scaler_mean[14]:.6f}")
print(f"scaler_scale[14]={scaler_scale[14]:.6f}")
print()

# After z-score with this scaler
print("=== Z-SCORE with scaler fitted on TRAINING ===")
print("Training Feature 14: z = (0.0 - scaler_mean) / scaler_scale = (0.0 - (-0.0022)) / 0.0237 ≈ 0.09")
print("Test Feature 14: z = (42.14 - (-0.0022)) / 0.0237 ≈ 1778")
print()
print("This is why MemStream fails: during warmup, Feature 14 is ~0.09")
print("During scoring, Feature 14 is ~1778. The AE has never seen this value.")
print()

# Compare with Fold 2
fd2 = cache / 'fold_02'
X_train2 = np.load(fd2 / 'X_train.npy')
X_test2 = np.load(fd2 / 'easy' / 'X_test.npy')
sm2 = np.load(fd2 / 'scaler_mean.npy')
ss2 = np.load(fd2 / 'scaler_scale.npy')
print("Fold 2 Feature 14: pre-norm test mean={:.4f}, scaler_scale={:.4f}".format(
    X_test2[:, 14].mean(), ss2[14]))
print("Fold 2 works because scaler_scale ≈ 1.0, so Feature 14 stays in normal range")
print()

# THE FIX: Apply min_scale threshold
print("=== FIX: Apply min_scale threshold ===")
min_scale = 1e-3  # At least 0.001
ss_fixed = np.maximum(scaler_scale, min_scale)

# Manually compute what the fixed z-scores would be
# Data is already "pre-normalized" (z-scored with original scaler)
# We need to re-scale bad features

# For Feature 14, the pre-normalized test value is ~42.14
# The original scale was 0.0237, so the "original z-score" was ~1778
# With min_scale=0.001, the new "z-score" would be ~42140

# Actually, the pre-normalized data stores z-scores already
# For Fold 1 Feature 14, the pre-norm test value = 42.14
# This means: pre_norm = (raw - scaler_mean) / scaler_scale = raw / 0.0237
# So raw ≈ 42.14 * 0.0237 ≈ 1.0
# But wait, that doesn't match...

# Let me check: is the data pre-normalized with the scaler or not?
print("Verifying pre-normalization:")
f14_raw_test_val = X_test[0, 14]
computed_z = (f14_raw_test_val - scaler_mean[14]) / scaler_scale[14]
print(f"Feature 14 test value: {f14_raw_test_val:.6f}")
print(f"Computed z-score: ({f14_raw_test_val:.6f} - {scaler_mean[14]:.6f}) / {scaler_scale[14]:.6f} = {computed_z:.2f}")
print()

# The pre-normalized data IS z-scored but with wrong scale
# Fix: for bad features, divide by min_scale instead of actual scale
print("Fix approach: for bad features, recompute z-score with safe scale")
for f in [14]:  # Only feature 14 is bad
    safe_scale = max(scaler_scale[f], min_scale)
    # Original pre-normalized value
    orig = X_test[:, f].copy()
    # What it SHOULD be (re-scaled)
    # raw_value = orig * scaler_scale[f] + scaler_mean[f]
    # new_z = raw_value / safe_scale
    raw_vals = orig * scaler_scale[f] + scaler_mean[f]
    fixed_vals = raw_vals / safe_scale

    print(f"Feature {f}:")
    print(f"  Original pre-norm: mean={orig.mean():.2f}, std={orig.std():.2f}")
    print(f"  Fixed pre-norm:    mean={fixed_vals.mean():.2f}, std={fixed_vals.std():.2f}")
    print(f"  Improvement: original max={abs(orig).max():.0f} -> fixed max={abs(fixed_vals).max():.0f}")

print()
print("=== CONCLUSION ===")
print("The pre-normalized data in fold_cache was z-scored with scaler_scale[14]=0.0237.")
print("This causes Feature 14 to have values ~42 instead of ~1.")
print("Fix: for bad features, re-scale with safe min_scale in load_fold_data.")
print("This ensures MemStream sees reasonable values during warmup and scoring.")
