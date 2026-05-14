#!/usr/bin/env python3
"""Investigate Fold 1 failure."""
import numpy as np
from pathlib import Path

cache = Path('c:/proj/ldt/results/v5_clean/fold_cache')

for fold in [1, 2]:
    fd = cache / f'fold_{fold:02d}'
    X_train = np.load(fd / 'X_train.npy')

    diff_dir = fd / 'easy'
    X_test = np.load(diff_dir / 'X_test.npy')
    y = np.load(diff_dir / 'y_labels.npy')

    sm = np.load(fd / 'scaler_mean.npy')
    ss = np.load(fd / 'scaler_scale.npy')

    print(f"=== FOLD {fold:02d} ===")
    print(f"  X_train: shape={X_train.shape}, dtype={X_train.dtype}, min={X_train.min():.4f}, max={X_train.max():.4f}")
    print(f"  X_test:  shape={X_test.shape}, dtype={X_test.dtype}, min={X_test.min():.4f}, max={X_test.max():.4f}")
    print(f"  y:       unique={np.unique(y, return_counts=True)}, anomaly_rate={y.mean()*100:.2f}%")
    print(f"  scaler_mean: shape={sm.shape}, min={sm.min():.4f}, max={sm.max():.4f}")
    print(f"  scaler_scale: shape={ss.shape}, min={ss.min():.4f}, max={ss.max():.4f}")

    print(f"  X_train NaN: {np.isnan(X_train).sum()}, Inf: {np.isinf(X_train).sum()}")
    print(f"  X_test NaN: {np.isnan(X_test).sum()}, Inf: {np.isinf(X_test).sum()}")

    print(f"  X_train feature means (first 5): {X_train.mean(axis=0)[:5]}")
    print(f"  X_test feature means (first 5): {X_test.mean(axis=0)[:5]}")

    # Check train/test distribution shift
    train_mean = X_train.mean(axis=0)
    test_mean = X_test.mean(axis=0)
    dist_shift = np.abs(train_mean - test_mean).mean()
    print(f"  Train-Test mean shift: {dist_shift:.4f}")

    # Check warmup data (first 75%)
    n_warmup = int(len(X_train) * 0.75)
    X_warmup = X_train[:n_warmup]
    print(f"  X_warmup: shape={X_warmup.shape}, min={X_warmup.min():.4f}, max={X_warmup.max():.4f}")
    print(f"  X_warmup NaN: {np.isnan(X_warmup).sum()}")
    print()


print("=== DEEP DIVE FOLD 1 ===")
fd = cache / 'fold_01'
X_train = np.load(fd / 'X_train.npy')
n_warmup = int(len(X_train) * 0.75)
X_warmup = X_train[:n_warmup]

# Per-feature analysis
print(f"Per-feature stats for Fold 1 warmup (first 10 features):")
for i in range(10):
    col = X_warmup[:, i]
    print(f"  feat[{i}]: mean={col.mean():.4f}, std={col.std():.4f}, "
          f"min={col.min():.4f}, max={col.max():.4f}, "
          f"NaN={np.isnan(col).sum()}, zero_std={col.std() < 1e-8}")

# Check if any feature has near-zero variance
print(f"\nFeature std across all 25 dims:")
stds = X_warmup.std(axis=0)
for i, s in enumerate(stds):
    if s < 1e-6:
        print(f"  WARNING: feature {i} has near-zero std: {s:.2e}")
print(f"  Min std: {stds.min():.4f}, Max std: {stds.max():.4f}")
print(f"  Features with std < 0.01: {(stds < 0.01).sum()}")
print(f"  Features with std < 0.001: {(stds < 0.001).sum()}")

# Check Fold 2 for comparison
fd2 = cache / 'fold_02'
X_train2 = np.load(fd2 / 'X_train.npy')
n_warmup2 = int(len(X_train2) * 0.75)
X_warmup2 = X_train2[:n_warmup2]
stds2 = X_warmup2.std(axis=0)
print(f"\nFold 2 feature stds: min={stds2.min():.4f}, max={stds2.max():.4f}")
print(f"  Features with std < 0.01: {(stds2 < 0.01).sum()}")
