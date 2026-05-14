#!/usr/bin/env python3
"""Investigate Feature 14 shift root cause and source."""
import numpy as np
from pathlib import Path

cache = Path('c:/proj/ldt/results/v5_clean/fold_cache')

print("=== FEATURE 14: FULL ANALYSIS ACROSS ALL FOLDS ===")
for fold in range(1, 12):
    fd = cache / f'fold_{fold:02d}'
    X_train = np.load(fd / 'X_train.npy')
    X_test = np.load(fd / 'easy' / 'X_test.npy')

    feat14_train = X_train[:, 14]
    feat14_test = X_test[:, 14]

    print(f"Fold {fold:02d}: "
          f"train[mean={feat14_train.mean():.4f}, std={feat14_train.std():.4f}, "
          f"range=[{feat14_train.min():.4f}, {feat14_train.max():.4f}], "
          f"unique={len(np.unique(feat14_train))}] | "
          f"test[mean={feat14_test.mean():.4f}, std={feat14_test.std():.4f}, "
          f"range=[{feat14_test.min():.4f}, {feat14_test.max():.4f}], "
          f"unique={len(np.unique(feat14_test))}]")

print()

print("=== FEATURE 10: FULL ANALYSIS ACROSS ALL FOLDS ===")
for fold in range(1, 12):
    fd = cache / f'fold_{fold:02d}'
    X_train = np.load(fd / 'X_train.npy')
    X_test = np.load(fd / 'easy' / 'X_test.npy')

    feat10_train = X_train[:, 10]
    feat10_test = X_test[:, 10]

    print(f"Fold {fold:02d}: "
          f"train[mean={feat10_train.mean():.4f}, std={feat10_train.std():.4f}, "
          f"range=[{feat10_train.min():.4f}, {feat10_train.max():.4f}], "
          f"unique={len(np.unique(feat10_train))}] | "
          f"test[mean={feat10_test.mean():.4f}, std={feat10_test.std():.4f}, "
          f"range=[{feat10_test.min():.4f}, {feat10_test.max():.4f}], "
          f"unique={len(np.unique(feat10_test))}]")

print()

print("=== FOLD 1: ALL 25 FEATURES - TRAIN VS TEST ===")
fd1 = cache / 'fold_01'
X_train = np.load(fd1 / 'X_train.npy')
X_test_easy = np.load(fd1 / 'easy' / 'X_test.npy')

print(f"{'Feat':>5} {'TRAIN_mean':>12} {'TRAIN_std':>10} {'TEST_mean':>12} {'TEST_std':>10} {'SHIFT':>10} {'TRAIN_unique':>12}")
for i in range(25):
    t_mean = X_train[:, i].mean()
    t_std = X_train[:, i].std()
    te_mean = X_test_easy[:, i].mean()
    te_std = X_test_easy[:, i].std()
    shift = abs(t_mean - te_mean)
    tunique = len(np.unique(X_train[:, i]))
    flag = " <<<" if shift > 1.0 or t_std < 0.01 else ""
    print(f"{i:5d} {t_mean:12.4f} {t_std:10.4f} {te_mean:12.4f} {te_std:10.4f} {shift:10.4f} {tunique:12d}{flag}")

print()

print("=== CHECK IF FEATURE 14 TEST VALUES ARE ALL SAME (constant) ===")
for diff in ['easy', 'medium', 'hard']:
    X_test = np.load(fd1 / diff / 'X_test.npy')
    f14 = X_test[:, 14]
    print(f"Fold 1 {diff}: Feature 14 unique={len(np.unique(f14))}, "
          f"mean={f14.mean():.4f}, std={f14.std():.6f}, "
          f"min={f14.min():.4f}, max={f14.max():.4f}")

print()

print("=== CHECK ORIGINAL RAW DATA (if available) ===")
raw_path = Path('c:/proj/ldt/results/v5_clean')
for f in raw_path.glob('*'):
    if f.is_file() and 'fold' not in f.name.lower():
        print(f"  File: {f.name}")

print()

print("=== CHECK METADATA FOR FOLD 1 ===")
with open(cache / 'metadata.json') as f:
    import json
    meta = json.load(f)

for entry in meta:
    if entry['fold'] == 1:
        print(f"  {json.dumps(entry, indent=2)}")
        break
