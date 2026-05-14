"""
BarScoreAgent — BAR Score (Balanced Accuracy by Budget of Labeled Data).
Measures how well each streaming algorithm maintains AUC-PR as labeled data decreases.
Tests 5 budget levels: 1%, 5%, 10%, 25%, 50% of test-month labels available.
For each budget, measures AUC-PR across all folds and difficulties.

Evaluates: 5 budgets × 4 algorithms × 11 folds × 3 difficulties × 10 seeds
         = 660 runs (but each run is fast — just a small data subset).

Uses 2 CPU workers for parallelization.
"""
from __future__ import annotations

import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.ensemble import IsolationForest

from benchmark.table_b_agent import sHST_River, MemStream_, IForestASD_, CADIFEiaStream


OUT_DIR   = Path(__file__).parent.parent / 'results' / 'v3'
CACHE_DIR = OUT_DIR / 'features' / 'fold_data'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS        = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]
DIFFICULTIES = ['easy', 'medium', 'hard']
BAR_BUDGETS  = [0.01, 0.05, 0.10, 0.25, 0.50]  # fraction of test labels


def _bar_run(args) -> dict:
    """
    Evaluate one BAR score run: (fold, diff, budget, algorithm, seed).
    """
    fold, diff, budget, algo_name, algo_cls, seed_val, X_train, X_test_inj, y_test = args

    row = {
        'fold': fold, 'difficulty': diff,
        'budget_pct': budget,
        'algorithm': algo_name, 'seed': seed_val,
    }

    try:
        n_total  = len(X_train)
        n_budget = max(100, int(n_total * budget))
        rng = np.random.RandomState(seed_val)
        idx = rng.choice(n_total, n_budget, replace=False)
        X_sub = X_train[idx]

        algo = algo_cls(seed=seed_val)
        algo.fit(X_sub)
        scores = algo.decision_function(X_test_inj).astype(np.float64)

        if y_test.sum() == 0 or len(scores) < 5:
            return {**row, 'AUC_PR': 0.0}

        pr_curve, rc_curve, _ = precision_recall_curve(y_test, scores)
        auc_pr = auc(rc_curve, pr_curve) if len(rc_curve) > 1 else 0.0

        row['AUC_PR'] = auc_pr
    except Exception as e:
        row['AUC_PR'] = float('nan')
        row['error'] = str(e)

    return row


def _iter_bar_bundles():
    for path in sorted(CACHE_DIR.glob('fold*.npz')):
        data = np.load(path)
        yield {
            'fold': int(data['fold']),
            'difficulty': str(data['difficulty']),
            'X_train': data['X_train'],
            'X_test_inj': data['X_test_inj'],
            'y_test': data['y_test'],
        }


BAR_ALGOS = [sHST_River, MemStream_, IForestASD_, CADIFEiaStream]


class BarScoreAgent:
    """
    Computes BAR Score curves for streaming algorithms.
    Results saved to results/v3/bar_score_results.csv
    """

    def __init__(self, base_dir: Path | None = None, out_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.out_dir  = Path(out_dir)  if out_dir  else OUT_DIR
        self.out_file = self.out_dir / 'bar_score_results.csv'

    def run(self, n_workers: int | None = None) -> pd.DataFrame:
        if n_workers is None:
            n_workers = max(1, cpu_count() - 14)

        print(f'\n[BarScoreAgent] Starting on {n_workers} CPU workers')
        print(f'  Budgets: {BAR_BUDGETS}')
        print(f'  Algorithms: {[a.name for a in BAR_ALGOS]}')

        jobs = []
        for bundle in _iter_bar_bundles():
            for budget in BAR_BUDGETS:
                for algo_cls in BAR_ALGOS:
                    for seed_val in SEEDS:
                        jobs.append((
                            bundle['fold'], bundle['difficulty'], budget,
                            algo_cls.name, algo_cls, seed_val,
                            bundle['X_train'], bundle['X_test_inj'],
                            bundle['y_test'],
                        ))

        print(f'  Jobs: {len(jobs)} (5 budgets x 4 algos x 11 folds x 3 diffs x 10 seeds)')

        t0 = time.perf_counter()
        results = []
        with Pool(n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_bar_run, jobs, chunksize=100)):
                results.append(res)
                if (i + 1) % 500 == 0:
                    elapsed = time.perf_counter() - t0
                    rate    = (i + 1) / elapsed
                    remain  = (len(jobs) - i - 1) / rate / 60
                    print(f'  {i+1}/{len(jobs)} ({rate:.1f}/s, ~{remain:.1f}m remaining)')

        df = pd.DataFrame(results)
        df.to_csv(self.out_file, index=False)

        elapsed = time.perf_counter() - t0
        print(f'\n[BarScoreAgent] Done. {len(results)} rows saved in {elapsed:.1f}s ({elapsed/60:.1f} min)')
        self._print_summary(df)
        return df

    def _print_summary(self, df: pd.DataFrame):
        if 'AUC_PR' not in df.columns:
            return
        print('\n  Mean AUC-PR at each budget level:')
        pivot = df.pivot_table(values='AUC_PR', index='algorithm',
                               columns='budget_pct', aggfunc='mean')
        print(pivot.to_string())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-workers', type=int, default=1)
    args = parser.parse_args()
    BarScoreAgent().run(n_workers=args.n_workers)
