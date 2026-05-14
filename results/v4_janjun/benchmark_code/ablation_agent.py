"""
AblationAgent — Context-aware Grid ablation study.
Tests whether the full 25D feature vector (treatment) improves over
the raw 15D feature vector (control) for each Table A algorithm.

Evaluates: 11 folds × 3 difficulties × 5 algorithms × 10 seeds = 1,650 runs.
Compares Config A (15D) vs Config B (25D) per fold/difficulty/seed.

Uses 3 CPU workers for parallelization.
"""
from __future__ import annotations

import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, f1_score


OUT_DIR   = Path(__file__).parent.parent / 'results' / 'v3'
CACHE_DIR = OUT_DIR / 'features' / 'fold_data'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS        = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]
DIFFICULTIES = ['easy', 'medium', 'hard']

from benchmark.table_a_agent import (
    SklearnIF, SklearnOCSVM, SklearnLOF, CADIFEia, METERSCD, _evaluate
)


class AblationEval:
    """Same evaluation as _evaluate but with separate control/treatment data."""
    @staticmethod
    def run(args) -> dict:
        (fold, diff, algo_name, algo_cls, seed_val,
         X_train_ctrl, X_test_ctrl, y_test_ctrl,
         X_train_treat, X_test_treat, y_test_treat) = args

        row = {
            'fold': fold, 'difficulty': diff,
            'algorithm': algo_name, 'seed': seed_val,
        }
        try:
            # Control (15D raw features)
            algo_ctrl = algo_cls(seed=seed_val)
            algo_ctrl.fit(X_train_ctrl)
            scores_ctrl = algo_ctrl.decision_function(X_test_ctrl).astype(np.float64)

            # Treatment (25D with context)
            algo_treat = algo_cls(seed=seed_val)
            algo_treat.fit(X_train_treat)
            scores_treat = algo_treat.decision_function(X_test_treat).astype(np.float64)

            def _auc_pr(scores, y):
                if len(scores) < 5 or y.sum() == 0:
                    return 0.0
                pr_curve, rc_curve, _ = precision_recall_curve(y, scores)
                return auc(rc_curve, pr_curve) if len(rc_curve) > 1 else 0.0

            auc_ctrl  = _auc_pr(scores_ctrl,  y_test_ctrl)
            auc_treat = _auc_pr(scores_treat, y_test_treat)
            delta     = auc_treat - auc_ctrl

            row.update({
                'AUC_PR_control': auc_ctrl,
                'AUC_PR_treatment': auc_treat,
                'delta': delta,
                'type': 'ablation',
            })
        except Exception as e:
            row.update({
                'AUC_PR_control': float('nan'),
                'AUC_PR_treatment': float('nan'),
                'delta': float('nan'),
                'type': 'ablation',
                'error': str(e),
            })
        return row


ABL_ALGOS = [SklearnIF, SklearnOCSVM, SklearnLOF, CADIFEia, METERSCD]


def _iter_ablation_bundles():
    for path in sorted(CACHE_DIR.glob('fold*.npz')):
        data = np.load(path)
        yield {
            'fold': int(data['fold']),
            'difficulty': str(data['difficulty']),
            'X_train': data['X_train'],
            'X_train_ctrl': data['X_train_ctrl'],
            'X_test_inj': data['X_test_inj'],
            'X_test_inj_ctrl': data['X_test_inj_ctrl'],
            'y_test': data['y_test'],
        }


class AblationAgent:
    """
    Runs ablation study: Config A (15D) vs Config B (25D).
    Results saved to results/v3/ablation_results.csv
    """

    def __init__(self, base_dir: Path | None = None, out_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.out_dir  = Path(out_dir)  if out_dir  else OUT_DIR
        self.out_file = self.out_dir / 'ablation_results.csv'

    def run(self, n_workers: int | None = None) -> pd.DataFrame:
        if n_workers is None:
            n_workers = max(1, cpu_count() - 12)

        print(f'\n[AblationAgent] Starting on {n_workers} CPU workers')
        print(f'  Comparing: 15D control vs 25D treatment')
        print(f'  Algorithms: {[a.name for a in ABL_ALGOS]}')

        jobs = []
        for bundle in _iter_ablation_bundles():
            for algo_cls in ABL_ALGOS:
                for seed_val in SEEDS:
                    jobs.append((
                        bundle['fold'], bundle['difficulty'],
                        algo_cls.name, algo_cls, seed_val,
                        bundle['X_train_ctrl'],
                        bundle['X_test_inj_ctrl'],
                        bundle['y_test'],
                        bundle['X_train'],
                        bundle['X_test_inj'],
                        bundle['y_test'],
                    ))

        print(f'  Jobs: {len(jobs)} (11 folds x 3 difficulties x 5 algos x 10 seeds)')

        t0 = time.perf_counter()
        results = []
        with Pool(n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(AblationEval.run, jobs, chunksize=50)):
                results.append(res)
                if (i + 1) % 300 == 0:
                    elapsed = time.perf_counter() - t0
                    rate    = (i + 1) / elapsed
                    remain  = (len(jobs) - i - 1) / rate / 60
                    print(f'  {i+1}/{len(jobs)} ({rate:.1f}/s, ~{remain:.1f}m remaining)')

        df = pd.DataFrame(results)
        df.to_csv(self.out_file, index=False)

        elapsed = time.perf_counter() - t0
        print(f'\n[AblationAgent] Done. {len(results)} rows saved in {elapsed:.1f}s ({elapsed/60:.1f} min)')
        self._print_summary(df)
        return df

    def _print_summary(self, df: pd.DataFrame):
        if 'delta' not in df.columns:
            return
        print('\n  Mean Delta AUC-PR by algorithm (Treatment - Control):')
        top = df.groupby('algorithm')['delta'].mean().sort_values(ascending=False)
        for a, v in top.items():
            print(f'    {a:20s}: {v:+.4f}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-workers', type=int, default=1)
    args = parser.parse_args()
    AblationAgent().run(n_workers=args.n_workers)
