#!/usr/bin/env python3
"""
Concept-Drift-Only Learning Validation for NYC Taxi MemStream.

Scientific Claim: No auto-learning (no memory update) or drift-triggered learning
achieves comparable accuracy to full auto-learning with 95%+ fewer label costs.

Three experiments:
  A. No Auto-Learning:  Memory frozen after warmup (0% updates)
  B. Drift-Triggered:   Memory update only when ADWIN detects drift (1-5% updates)
  C. Full Auto-Learning: Update on every normal record (100% updates, baseline)

Success: B achieves >=95% of C's AUC-PR with <=5% memory update rate.

Usage:
    python eval_drift_learning.py --data /path/to/nyc_taxi.csv --output results/drift/
"""

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from memstream_src.core.memstream_core import (
    MemStreamCore, MemStreamConfig, set_determinism,
    BARController, SimpleADWIN
)
from memstream_src.core.feature_extractor import FeatureVectorizer
from memstream_src.scripts.inject_anomalies_multi import inject_anomalies


class DriftExperiment:
    """Framework for comparing drift learning strategies."""

    def __init__(self, cfg: MemStreamConfig, X_warmup: np.ndarray):
        self.cfg = cfg
        self.X_warmup = X_warmup
        self.model: MemStreamCore = None
        self.bar_ctrl: BARController = None
        self.adwin: SimpleADWIN = None

    def setup(self):
        """Initialize model from warmup."""
        set_determinism(self.cfg.seed)
        self.model = MemStreamCore(cfg=self.cfg, device='cpu')
        self.model.warmup(self.X_warmup, epochs=self.cfg.warmup_epochs, verbose=False)
        self.bar_ctrl = BARController(config={
            'target_bar_rate': 0.02,
            'adwin_delta': 0.002,
            'min_budget_fraction': 0.01,
            'bar_window_size': 10000,
        })
        self.adwin = SimpleADWIN(delta=0.002)

    def run_no_autolearning(self, X_test: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Experiment A: Memory frozen after warmup.
        Only scoring - no memory updates at all.
        """
        self.setup()
        scores = self.model.score_batch(X_test)
        preds = (scores >= 1.0).astype(int)
        return self._compute_metrics(scores, preds, labels, update_rate=0.0)

    def run_drift_triggered(self, X_test: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Experiment B: Update memory only when ADWIN detects drift.
        Simulates production streaming with concept drift detection.
        """
        self.setup()
        n = len(X_test)
        scores = np.zeros(n)
        preds = np.zeros(n)
        updates = 0
        drift_events = 0

        for i in range(n):
            x = X_test[i]
            score = self.model.score_one(x)
            scores[i] = score
            preds[i] = 1 if score >= 1.0 else 0

            if preds[i] == 0:  # Normal record
                should_update, reason = self.bar_ctrl.should_update_memory(
                    neighborhood=str(i % 10),  # Simulate neighborhood per record
                    score=score
                )
                if should_update:
                    self.model.memory_update(x, neighborhood_id=i % 10,
                                            hour=12, dow=0, ratecode=1.0)
                    updates += 1
                    if reason == 'drift_detected':
                        drift_events += 1

        update_rate = updates / max(n, 1)
        return self._compute_metrics(scores, preds, labels, update_rate, drift_events)

    def run_full_autolearning(self, X_test: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Experiment C: Update memory on every normal record.
        Upper-bound baseline.
        """
        self.setup()
        n = len(X_test)
        scores = np.zeros(n)
        preds = np.zeros(n)

        for i in range(n):
            x = X_test[i]
            score = self.model.score_one(x)
            scores[i] = score
            preds[i] = 1 if score >= 1.0 else 0

            if preds[i] == 0:  # Normal - update
                self.model.memory_update(x, neighborhood_id=i % 10,
                                        hour=12, dow=0, ratecode=1.0)

        return self._compute_metrics(scores, preds, labels, update_rate=1.0)

    def _compute_metrics(self, scores: np.ndarray, preds: np.ndarray,
                         labels: np.ndarray, update_rate: float,
                         drift_events: int = 0) -> Dict:
        """Compute evaluation metrics."""
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        try:
            auc_roc = roc_auc_score(labels, scores)
        except ValueError:
            auc_roc = 0.0

        prec_curve, rec_curve, _ = precision_recall_curve(labels, scores)
        auc_pr = auc(rec_curve, prec_curve)

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'fpr': float(fpr),
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'update_rate': float(update_rate),
            'drift_events': drift_events,
        }


def main():
    parser = argparse.ArgumentParser(description='Drift Learning Experiment')
    parser.add_argument('--data', type=str, required=True, help='NYC taxi CSV path')
    parser.add_argument('--output', type=str, default='results/drift',
                        help='Output directory')
    parser.add_argument('--n-anomalies', type=int, default=2000)
    parser.add_argument('--warmup-frac', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Concept-Drift-Only Learning Validation")
    print("=" * 60)

    # Load and prepare data
    print("\n[1] Loading data...")
    df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data)
    print(f"  Records: {len(df):,}")

    # Extract features (use transform_batch for efficiency)
    print("\n[2] Extracting features...")
    vectorizer = FeatureVectorizer()
    X = vectorizer.transform_batch(df.to_dict('records'))

    # Split
    n_warmup = int(len(X) * args.warmup_frac)
    X_warmup = X[:n_warmup]
    df_rest = df.iloc[n_warmup:].reset_index(drop=True)

    # Inject anomalies
    print(f"\n[3] Injecting {args.n_anomalies} anomalies...")
    df_test, labels = inject_anomalies(df_rest, n_anomalies=args.n_anomalies, seed=args.seed)
    vectorizer2 = FeatureVectorizer()
    X_test = vectorizer2.transform_batch(df_test.to_dict('records'))
    print(f"  Test: {len(X_test):,} records, {int(labels.sum())} anomalies")

    # Build config
    cfg = MemStreamConfig()
    cfg.memory_len = 1024
    cfg.k = 10
    cfg.gamma = 0.5
    cfg.warmup_epochs = 100

    experiment = DriftExperiment(cfg, X_warmup)

    # Run experiments
    print("\n[4] Running experiments...")

    print("  Experiment A (No Auto-Learning)...")
    result_a = experiment.run_no_autolearning(X_test, labels)
    print(f"    F1={result_a['f1']:.4f} AUC-PR={result_a['auc_pr']:.4f} Updates=0%")

    print("  Experiment B (Drift-Triggered)...")
    result_b = experiment.run_drift_triggered(X_test, labels)
    print(f"    F1={result_b['f1']:.4f} AUC-PR={result_b['auc_pr']:.4f} "
          f"Updates={result_b['update_rate']*100:.1f}% Drift={result_b['drift_events']}")

    print("  Experiment C (Full Auto-Learning)...")
    result_c = experiment.run_full_autolearning(X_test, labels)
    print(f"    F1={result_c['f1']:.4f} AUC-PR={result_c['auc_pr']:.4f} Updates=100%")

    # Compute relative performance
    rel_b_vs_c = result_b['auc_pr'] / result_c['auc_pr'] * 100 if result_c['auc_pr'] > 0 else 0
    rel_a_vs_c = result_a['auc_pr'] / result_c['auc_pr'] * 100 if result_c['auc_pr'] > 0 else 0
    update_savings = (1 - result_b['update_rate']) * 100

    # Summary
    print("\n" + "=" * 60)
    print("DRIFT LEARNING COMPARISON")
    print("=" * 60)
    print(f"{'Experiment':<25} {'F1':>6} {'AUC-PR':>8} {'AUC-ROC':>9} {'Updates':>8}")
    print("-" * 60)
    print(f"{'A: No Auto-Learning':<25} {result_a['f1']:6.4f} {result_a['auc_pr']:8.4f} "
          f"{result_a['auc_roc']:9.4f} {0:>7.1f}%")
    print(f"{'B: Drift-Triggered':<25} {result_b['f1']:6.4f} {result_b['auc_pr']:8.4f} "
          f"{result_b['auc_roc']:9.4f} {result_b['update_rate']*100:>7.1f}%")
    print(f"{'C: Full Auto-Learning':<25} {result_c['f1']:6.4f} {result_c['auc_pr']:8.4f} "
          f"{result_c['auc_roc']:9.4f} {100:>7.1f}%")
    print("-" * 60)
    print(f"  B vs C AUC-PR: {rel_b_vs_c:.1f}% (target >=95%)")
    print(f"  Update savings: {update_savings:.1f}%")
    print(f"  Drift events detected: {result_b['drift_events']}")

    success = rel_b_vs_c >= 95 and result_b['update_rate'] <= 0.05
    print(f"\n  {'[OK]' if success else '[NEEDS TUNING]'}: "
          f"Drift-triggered {'meets' if success else 'does not meet'} targets")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'timestamp': datetime.now().isoformat(),
        'n_warmup': n_warmup,
        'n_test': len(X_test),
        'n_anomalies': int(labels.sum()),
        'experiment_a': result_a,
        'experiment_b': result_b,
        'experiment_c': result_c,
        'relative_b_vs_c': float(rel_b_vs_c),
        'relative_a_vs_c': float(rel_a_vs_c),
        'update_savings_pct': float(update_savings),
        'success': success,
    }
    with open(output_dir / f'drift_results_{ts}.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
