#!/usr/bin/env python3
"""
CA-MemStream Scientific Benchmark (Fast - Batch Scoring)
Compares: sklearn IF vs MemStream 25D (AE + Memory)
Uses AUC-based metrics (ROC and PR) for fair comparison.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'memstream_src'))

import numpy as np
import pandas as pd
import json
import time
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.ensemble import IsolationForest

from core.memstream_core import MemStreamCore, MemStreamConfig, set_determinism
from core.context_aware import BARController, ADWIN

np.random.seed(42)
set_determinism(42)

print("=" * 70)
print("CA-MemStream Scientific Benchmark (Batch Scoring)")
print("=" * 70)

CACHE = Path('c:/proj/ldt/results/v5_clean/fold_cache')
with open(CACHE / 'metadata.json') as f:
    metadata = json.load(f)

DIFFICULTIES = ['easy', 'medium', 'hard']
all_results = []

for diff in DIFFICULTIES:
    print(f"\n{'='*70}")
    print(f"DIFFICULTY: {diff.upper()}")
    print(f"{'='*70}")

    fold_data = [(e['fold'], e) for e in metadata if e['difficulty'] == diff]
    fold_results_if = []
    fold_results_ms = []

    for fn, entry in fold_data:
        X_train = np.load(CACHE / f'fold_{fn:02d}' / 'X_train.npy').astype(np.float32)
        X_test = np.load(CACHE / f'fold_{fn:02d}' / diff / 'X_test.npy').astype(np.float32)
        y_test = np.load(CACHE / f'fold_{fn:02d}' / diff / 'y_labels.npy')

        n_train = len(X_train)
        X_warmup = X_train[:int(n_train * 0.75)]

        # IsolationForest
        if_clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
        if_clf.fit(X_warmup)
        scores_if = -if_clf.score_samples(X_test)

        # MemStream
        cfg = MemStreamConfig()
        cfg.warmup_epochs = 10
        cfg.warmup_batch_size = 256
        ms = MemStreamCore(cfg=cfg, device='cpu')
        ms.warmup(X_warmup, epochs=10, batch_size=256, verbose=False)
        ms.set_beta(0.5)
        ms.eval_mode = True

        t0 = time.perf_counter()
        scores_ms = ms.score_batch(X_test)
        t_score = (time.perf_counter() - t0) * 1000

        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            continue

        # AUC metrics
        fpr_if, tpr_if, _ = roc_curve(y_test, scores_if)
        auc_roc_if = auc(fpr_if, tpr_if)
        pr_if, rc_if, _ = precision_recall_curve(y_test, scores_if)
        auc_pr_if = auc(rc_if, pr_if)

        fpr_ms, tpr_ms, _ = roc_curve(y_test, scores_ms)
        auc_roc_ms = auc(fpr_ms, tpr_ms)
        pr_ms, rc_ms, _ = precision_recall_curve(y_test, scores_ms)
        auc_pr_ms = auc(rc_ms, pr_ms)

        fold_results_if.append({'fold': fn, 'auc_roc': auc_roc_if, 'auc_pr': auc_pr_if})
        fold_results_ms.append({'fold': fn, 'auc_roc': auc_roc_ms, 'auc_pr': auc_pr_ms})

        print(f"  Fold {fn:02d}: IF AUC-PR={auc_pr_if:.4f}  MS AUC-PR={auc_pr_ms:.4f}  ({t_score:.0f}ms)")

    if not fold_results_if:
        continue

    if_m = ({k: np.mean([r[k] for r in fold_results_if]) for k in ['auc_roc', 'auc_pr']},
             {k: np.std([r[k] for r in fold_results_if]) for k in ['auc_roc', 'auc_pr']})
    ms_m = ({k: np.mean([r[k] for r in fold_results_ms]) for k in ['auc_roc', 'auc_pr']},
             {k: np.std([r[k] for r in fold_results_ms]) for k in ['auc_roc', 'auc_pr']})

    print(f"  Mean: IF AUC-ROC={if_m[0]['auc_roc']:.4f} AUC-PR={if_m[0]['auc_pr']:.4f}")
    print(f"        MS AUC-ROC={ms_m[0]['auc_roc']:.4f} AUC-PR={ms_m[0]['auc_pr']:.4f}")
    print(f"        Delta AUC-PR: {ms_m[0]['auc_pr']-if_m[0]['auc_pr']:+.4f}")

    for fr in fold_results_if:
        fr['difficulty'] = diff
        fr['algorithm'] = 'sklearn_IF'
    for fr in fold_results_ms:
        fr['difficulty'] = diff
        fr['algorithm'] = 'MemStream_25D'
    all_results.extend(fold_results_if)
    all_results.extend(fold_results_ms)

# ─── BAR Controller ──────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("BAR CONTROLLER (ADWIN Drift Detection)")
print(f"{'='*70}")

X_train = np.load(CACHE / 'fold_01' / 'X_train.npy').astype(np.float32)
X_test = np.load(CACHE / 'fold_01' / 'easy' / 'X_test.npy').astype(np.float32)
y_test = np.load(CACHE / 'fold_01' / 'easy' / 'y_labels.npy')

X_warmup = X_train[:int(len(X_train) * 0.75)]

cfg = MemStreamConfig()
cfg.warmup_epochs = 10
ms_bar = MemStreamCore(cfg=cfg, device='cpu')
ms_bar.warmup(X_warmup, epochs=10, batch_size=256, verbose=False)
ms_bar.set_beta(0.5)
ms_bar.eval_mode = True

N_BAR = 5000
scores_bar = ms_bar.score_batch(X_test[:N_BAR])
bar = BARController(enable_adwin=True, adwin_delta=0.002)

bar_updates = 0
bar_reasons = {}
for i in range(N_BAR):
    score = scores_bar[i]
    should_update, reason = bar.should_update_memory({}, float(score), 'test')
    if should_update:
        bar_updates += 1
        bar_reasons[reason] = bar_reasons.get(reason, 0) + 1

bar_rate = bar.bar_rate
print(f"  Records: {N_BAR:,}")
print(f"  Memory updates: {bar_updates} ({bar_updates/N_BAR*100:.2f}%)")
print(f"  BAR rate: {bar_rate:.4f}")
target_ok = 0.005 <= bar_rate <= 0.10
print(f"  Status: {'OK (within range)' if target_ok else 'OUTSIDE_RANGE (expected ~1% for ADWIN+BAR)'}")
print(f"  Reasons: {bar_reasons}")

# ─── ADWIN Drift Detection ────────────────────────────────────────────────
print(f"\n{'='*70}")
print("ADWIN DRIFT DETECTION TEST")
print(f"{'='*70}")

adwin = ADWIN(delta=0.002)
N_DRIFT = 5000
drift_points = []

for i in range(N_DRIFT):
    score = 0.5 + np.random.normal(0, 0.05) if i < 2500 else 1.0 + np.random.normal(0, 0.1)
    if adwin.update(score) and len(drift_points) < 3:
        drift_points.append(i)

print(f"  Records: {N_DRIFT:,}")
print(f"  Drift injected at: 2500")
print(f"  First detections: {drift_points}")
print(f"  ADWIN: {'WORKING' if drift_points else 'NOT_DETECTING'}")

# ─── Summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")

if all_results:
    df = pd.DataFrame(all_results)
    summary = df.groupby(['algorithm', 'difficulty']).agg(
        AUC_ROC=('auc_roc', 'mean'), AUC_ROC_std=('auc_roc', 'std'),
        AUC_PR=('auc_pr', 'mean'), AUC_PR_std=('auc_pr', 'std'),
    ).round(4)
    print("\n" + summary.to_string())

    overall = df.groupby('algorithm').agg(
        AUC_ROC=('auc_roc', 'mean'), AUC_ROC_std=('auc_roc', 'std'),
        AUC_PR=('auc_pr', 'mean'), AUC_PR_std=('auc_pr', 'std'),
    ).round(4)
    print(f"\nOverall:")
    print(overall.to_string())

    out_dir = Path('c:/proj/ldt/results/v5_clean')
    df.to_csv(out_dir / 'ca_memstream_scientific_benchmark.csv', index=False)
    print(f"\nSaved: {out_dir / 'ca_memstream_scientific_benchmark.csv'}")

print(f"\n{'='*70}")
print("KEY FINDINGS")
print(f"{'='*70}")
if all_results:
    overall = pd.DataFrame(all_results).groupby('algorithm').agg(AUC_ROC=('auc_roc', 'mean'), AUC_PR=('auc_pr', 'mean'))
    if_r = overall.loc['sklearn_IF']
    ms_r = overall.loc['MemStream_25D']
    print(f"1. sklearn IF:    AUC-ROC={if_r['AUC_ROC']:.4f}  AUC-PR={if_r['AUC_PR']:.4f}")
    print(f"2. MemStream 25D: AUC-ROC={ms_r['AUC_ROC']:.4f}  AUC-PR={ms_r['AUC_PR']:.4f}")
    d_roc = ms_r['AUC_ROC'] - if_r['AUC_ROC']
    d_pr = ms_r['AUC_PR'] - if_r['AUC_PR']
    print(f"   Delta AUC-ROC: {d_roc:+.4f} ({'BETTER' if d_roc > 0 else 'WORSE'})")
    print(f"   Delta AUC-PR:  {d_pr:+.4f} ({'BETTER' if d_pr > 0 else 'WORSE'})")
print(f"3. BAR Controller: rate={bar_rate*100:.2f}% (target 1-5%)")
print(f"4. ADWIN: detected at {drift_points}")
print(f"{'='*70}")
