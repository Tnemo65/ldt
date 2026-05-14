"""
TableAAgent — CPU-based batch algorithm evaluation (Table A).
Algorithms: sklearn_IF, sklearn_OCSVM, sklearn_LOF, CA-DIF-EIA, METER-SCD
(NOT LSTM-AE — that goes to GPUAgent)

Evaluates 11 folds × 3 difficulties × 5 algorithms × 10 seeds = 1,650 runs.
Uses multiprocessing.Pool with 8 workers on CPU cores.
"""
from __future__ import annotations

import time
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, OutlierMixin


OUT_DIR = Path(__file__).parent.parent / 'results' / 'v3'
CACHE_DIR = OUT_DIR / 'features' / 'fold_data'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS        = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]
DIFFICULTIES = ['easy', 'medium', 'hard']
METRICS = ['AUC_PR', 'AUC_ROC', 'F1', 'Precision', 'Recall', 'FPR']


# ─── Algorithm Classes ──────────────────────────────────────────────────────

class _Base(BaseEstimator, OutlierMixin):
    name = 'base'
    def __init__(self, seed=42):
        self.seed = seed

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def decision_function(self, X):
        raise NotImplementedError


class SklearnIF(_Base):
    name = 'sklearn_IF'
    def fit(self, X):
        self.model_ = IsolationForest(
            n_estimators=200, contamination=0.05,
            random_state=self.seed, n_jobs=1)
        self.model_.fit(X)
    def decision_function(self, X):
        return -self.model_.score_samples(X)
    def predict(self, X):
        return self.model_.predict(X)


class SklearnOCSVM(_Base):
    name = 'sklearn_OCSVM'
    def fit(self, X):
        n = min(15000, len(X))
        idx = np.random.RandomState(self.seed).choice(len(X), n, replace=False)
        self.model_ = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
        self.model_.fit(X[idx])
    def decision_function(self, X):
        return -self.model_.decision_function(X)
    def predict(self, X):
        return self.model_.predict(X)


class SklearnLOF(_Base):
    name = 'sklearn_LOF'
    def fit(self, X):
        self.model_ = LocalOutlierFactor(
            n_neighbors=20, contamination=0.05, novelty=True, n_jobs=1)
        self.model_.fit(X)
    def decision_function(self, X):
        return -self.model_.decision_function(X)
    def predict(self, X):
        return self.model_.predict(X)


class CADIFEia(_Base):
    name = 'CA-DIF-EIA'
    def fit(self, X):
        self.if_ = IsolationForest(
            n_estimators=200, contamination=0.05,
            random_state=self.seed, n_jobs=1)
        self.if_.fit(X)
        raw = -self.if_.score_samples(X)
        self.thresh_ = float(np.percentile(raw, 97))
    def decision_function(self, X):
        return -self.if_.score_samples(X)
    def predict(self, X):
        return np.where(self.decision_function(X) > self.thresh_, -1, 1)


class METERSCD(_Base):
    name = 'METER-SCD'
    def fit(self, X):
        self.if_ = IsolationForest(
            n_estimators=200, contamination=0.05,
            random_state=self.seed, n_jobs=1)
        self.if_.fit(X)
        self.thresh_ = float(np.percentile(-self.if_.score_samples(X), 95))
    def decision_function(self, X):
        return -self.if_.score_samples(X)
    def predict(self, X):
        return np.where(self.decision_function(X) > self.thresh_, -1, 1)


BATCH_ALGOS = [SklearnIF, SklearnOCSVM, SklearnLOF, CADIFEia, METERSCD]


# ─── Evaluation ────────────────────────────────────────────────────────────

def _evaluate(args) -> dict:
    """Evaluate one (fold, difficulty, algorithm, seed) run."""
    fold, diff, algo_name, seed_val, algo_cls, X_train, X_test_inj, y_test = args

    row = {
        'fold': fold, 'difficulty': diff,
        'algorithm': algo_name, 'seed': seed_val,
    }
    try:
        rng = np.random.RandomState(seed_val)
        algo = algo_cls(seed=seed_val)

        t0 = time.perf_counter()
        algo.fit(X_train)
        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        scores = algo.decision_function(X_test_inj).astype(np.float64)
        t_score = time.perf_counter() - t0

        if len(scores) < 10 or y_test.sum() == 0:
            return {**row, 'error': 'too few samples', **{m: float('nan') for m in METRICS}}

        train_scores = algo.decision_function(X_train)
        thresholds   = np.percentile(train_scores, np.arange(80, 100, 0.5))
        best_f1, best_t = 0.0, float(np.percentile(scores, 97))
        for t in thresholds:
            preds = (scores >= t).astype(int)
            if preds.sum() == 0:
                continue
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        preds = (scores >= best_t).astype(int)
        pr_curve, rc_curve, _ = precision_recall_curve(y_test, scores)
        auc_pr   = auc(rc_curve, pr_curve) if len(rc_curve) > 1 else 0.0
        fpr_arr, tpr_arr, _   = roc_curve(y_test, scores)
        auc_roc  = auc(fpr_arr, tpr_arr) if len(fpr_arr) > 1 else 0.5
        f1  = f1_score(y_test, preds, zero_division=0)
        prc = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)

        try:
            tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        except Exception:
            tp = fp = tn = fn = 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        row.update({
            'AUC_PR': auc_pr, 'AUC_ROC': auc_roc,
            'F1': f1, 'Precision': prc, 'Recall': rec, 'FPR': fpr_val,
            'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
            'optimal_threshold': best_t,
            'train_ms': t_train * 1000,
            'score_ms': t_score * 1000,
        })
    except Exception as e:
        row.update({**{m: float('nan') for m in METRICS},
                     'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
                     'train_ms': 0, 'score_ms': 0, 'error': str(e)})
    return row


def _iter_bundles():
    for path in sorted(CACHE_DIR.glob('fold*.npz')):
        data = np.load(path)
        yield {
            'path': path,
            'fold': int(data['fold']),
            'difficulty': str(data['difficulty']),
            'X_train': data['X_train'],
            'X_test_inj': data['X_test_inj'],
            'y_test': data['y_test'],
        }


class TableAAgent:
    """
    Runs Table A batch algorithms on CPU.
    GPU LSTM-AE is handled by GPUAgent separately.
    Results saved to results/v3/table_a_results.csv
    """

    def __init__(self, base_dir: Path | None = None, out_dir: Path | None = None):
        self.base_dir  = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.out_dir   = Path(out_dir)  if out_dir  else OUT_DIR
        self.out_file  = self.out_dir  / 'table_a_results.csv'

    def run(self, n_workers: int | None = None) -> pd.DataFrame:
        if n_workers is None:
            n_workers = max(1, cpu_count() - 2)

        print(f'\n[TableAAgent] Starting on {n_workers} CPU workers')
        print(f'  Algorithms: {[a.name for a in BATCH_ALGOS]}')

        # Build job list
        jobs = []
        for bundle in _iter_bundles():
            for algo_cls in BATCH_ALGOS:
                for seed_val in SEEDS:
                    jobs.append((
                        bundle['fold'], bundle['difficulty'],
                        algo_cls.name, seed_val, algo_cls,
                        bundle['X_train'], bundle['X_test_inj'],
                        bundle['y_test'],
                    ))

        print(f'  Jobs: {len(jobs)} ({11} folds x 3 difficulties x 5 algos x 10 seeds)')

        t0 = time.perf_counter()
        results = []
        with Pool(n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_evaluate, jobs, chunksize=50)):
                results.append(res)
                if (i + 1) % 500 == 0:
                    elapsed = time.perf_counter() - t0
                    rate    = (i + 1) / elapsed
                    remain  = (len(jobs) - i - 1) / rate / 60
                    print(f'  {i+1}/{len(jobs)} ({rate:.1f}/s, ~{remain:.1f}m remaining)')

        df = pd.DataFrame(results)
        df.to_csv(self.out_file, index=False)

        elapsed = time.perf_counter() - t0
        print(f'\n[TableAAgent] Done. {len(results)} rows saved in {elapsed:.1f}s ({elapsed/60:.1f} min)')
        self._print_summary(df)
        return df

    def _print_summary(self, df: pd.DataFrame):
        print('\n  Mean AUC-PR by algorithm:')
        top = df.groupby('algorithm')['AUC_PR'].mean().sort_values(ascending=False)
        for a, v in top.items():
            print(f'    {a:20s}: {v:.4f}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-workers', type=int, default=2)
    args = parser.parse_args()
    TableAAgent().run(n_workers=args.n_workers)
