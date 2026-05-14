"""
StatsAgent — Statistical analysis for benchmark results.
Performs:
  1. Friedman test (overall significance)
  2. Wilcoxon Signed-Rank post-hoc pairwise tests (Holm + BH correction)
  3. Cohen's d effect sizes
  4. Critical Difference (CD) diagram data

Runs in the main process (single-threaded, minimal compute).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


OUT_DIR = Path(__file__).parent.parent / 'results' / 'v3'
METRICS = ['AUC_PR', 'AUC_ROC', 'F1', 'Precision', 'Recall', 'FPR']
DIFFICULTIES = ['easy', 'medium', 'hard']


# ─── Statistical Helpers ───────────────────────────────────────────────────

def holm_stepdown(pvals, alpha=0.05):
    """Holm-Bonferroni step-down correction."""
    pvals = np.array(pvals, dtype=float)
    valid = ~np.isnan(pvals)
    if valid.sum() == 0:
        return pvals
    idx   = np.argsort(pvals[valid])
    sp    = pvals[valid][idx]
    m     = len(sp)
    adj   = np.ones(m)
    for i in range(m):
        adj[i] = min(sp[i] * (m - i), 1.0)
        if i > 0:
            adj[i] = min(adj[i], adj[i-1])
    out = np.ones(len(pvals))
    out[valid] = adj
    return out


def cohens_d_paired(a, b):
    diff = a - b
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-10))


def wilcoxon_matrix(sm: np.ndarray, names: list[str], diff: str) -> list[dict]:
    """
    Pairwise Wilcoxon Signed-Rank tests for a score matrix (folds × algorithms).
    Returns list of dicts with W, p_raw, p_holm, p_bh, cohens_d, effect.
    """
    n, k = sm.shape
    rows = []
    p_raw_list = []

    for i in range(k):
        for j in range(i+1, k):
            d = sm[:, i] - sm[:, j]
            if np.all(d == 0) or np.isnan(d).all():
                W, p_raw = float('nan'), 1.0
            else:
                try:
                    wr  = stats.wilcoxon(d, alternative='two-sided', nan_policy='omit')
                    W, p_raw = float(wr.statistic), float(wr.pvalue)
                except Exception:
                    W, p_raw = float('nan'), 1.0

            d_eff = cohens_d_paired(sm[:, i], sm[:, j])
            ad    = abs(d_eff)
            eff   = ('negligible' if ad < 0.2 else
                     'small'      if ad < 0.5 else
                     'medium'     if ad < 0.8 else 'large')

            p_raw_list.append(p_raw)
            rows.append({
                'difficulty': diff,
                'alg_i': names[i], 'alg_j': names[j],
                'W_stat': W, 'p_raw': p_raw,
                'cohens_d': d_eff, 'effect': eff,
            })

    if rows:
        adj_h = holm_stepdown(p_raw_list)
        for r, h in zip(rows, adj_h):
            r['p_holm'] = h
            r['sig_holm'] = bool(h < 0.05)

        # Benjamini-Hochberg
        valid_p = [float('nan') if np.isnan(p) else p for p in p_raw_list]
        mBH     = len(valid_p)
        bh_adj  = np.array([
            min(p * mBH / (i + 1), 1.0)
            for i, p in enumerate(sorted(valid_p, key=lambda x: x if not np.isnan(x) else 1))
        ])
        for r, b in zip(rows, bh_adj):
            r['p_bh'] = b
            r['sig_bh'] = bool(b < 0.05)

    return rows


def friedman_test(sm: np.ndarray, k: int) -> tuple[float, float]:
    """Friedman test over folds × algorithms."""
    n = sm.shape[0]
    ranks = np.zeros_like(sm)
    for i in range(n):
        ranks[i] = np.argsort(np.argsort(-sm[i])) + 1
    rj  = np.mean(ranks, axis=0)
    chi2 = (12 * n / (k * (k + 1))) * np.sum((rj - (k + 1) / 2) ** 2)
    p    = 1 - stats.chi2.cdf(chi2, k - 1)
    return float(chi2), float(p)


def cd_ranks(sm: np.ndarray, names: list[str], k: int) -> tuple[pd.DataFrame, float]:
    """
    Critical Difference ranks (Demšar 2006).
    Returns DataFrame and CD value.
    """
    n = sm.shape[0]
    ranks = np.zeros_like(sm)
    for i in range(n):
        ranks[i] = np.argsort(np.argsort(-sm[i])) + 1
    avg_r = np.mean(ranks, axis=0)
    std_r = np.std(ranks, axis=0)

    # CD thresholds
    CD_TABLE = {
        4: {0.05: 0.850, 6: 1.030},
        5: {0.05: 0.900, 6: 1.150},
        6: {0.05: 1.030, 6: 1.270},
    }
    q_alpha = CD_TABLE.get(k, {}).get(0.05, 2.728)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    order = np.argsort(avg_r)
    rows  = []
    for pos, ai in enumerate(order):
        rows.append({
            'algorithm': names[ai],
            'avg_rank': float(avg_r[ai]),
            'std_rank': float(std_r[ai]),
            'rank_pos': pos + 1,
            'cd': float(cd),
        })
    return pd.DataFrame(rows), float(cd)


class StatsAgent:
    """
    Runs statistical analysis on benchmark results.
    Generates: statistical_tests_batch.csv, statistical_tests_streaming.csv,
               friedman_batch.csv, friedman_streaming.csv,
               cd_ranks_batch.csv, cd_ranks_streaming.csv,
               ablation_stats.csv (ablation p-values + Wilcoxon).
    """

    def __init__(self, base_dir: Path | None = None, out_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.out_dir  = Path(out_dir)  if out_dir  else OUT_DIR

    def run(self, table: str = 'batch') -> dict:
        if table == 'batch':
            return self._run_batch()
        elif table == 'streaming':
            return self._run_streaming()
        else:
            raise ValueError(f'Unknown table: {table}')

    def _run_batch(self) -> dict:
        print('\n[StatsAgent] Running Table A (batch) statistical analysis...')

        path = self.out_dir / 'benchmark_results_batch.csv'
        if not path.exists():
            print(f'  WARNING: {path} not found. Skipping.')
            return {}

        df = pd.read_csv(path)
        algos = sorted(df['algorithm'].unique())
        k     = len(algos)
        if k < 2:
            print('  Not enough algorithms. Skipping.')
            return {}

        print(f'  Algorithms: {algos}')

        all_friedman, all_wilcoxon, all_cd = [], [], []

        for diff in DIFFICULTIES:
            df_d = df[df['difficulty'] == diff]
            # Build score matrix: folds × algorithms
            folds = sorted(df_d['fold'].unique())
            sm    = np.zeros((len(folds), k))
            for mi, mfold in enumerate(folds):
                for ai, algo in enumerate(algos):
                    sub = df_d[(df_d['fold'] == mfold) & (df_d['algorithm'] == algo)]
                    sm[mi, ai] = sub['AUC_PR'].mean() if len(sub) > 0 else 0.0

            # Friedman
            chi2, p_f = friedman_test(sm, k)
            all_friedman.append({
                'difficulty': diff,
                'chi2': chi2, 'df': k - 1,
                'p_friedman': p_f,
                'significant': bool(p_f < 0.05),
            })

            # Wilcoxon
            pairs = wilcoxon_matrix(sm, algos, diff)
            all_wilcoxon.extend(pairs)

            # CD ranks
            cd_df, cd = cd_ranks(sm, algos, k)
            cd_df['difficulty'] = diff
            all_cd.append(cd_df)

            print(f'  {diff:7s}: Friedman p={p_f:.4f}, CD={cd:.3f}')

        # Save
        pd.DataFrame(all_friedman).to_csv(
            self.out_dir / 'friedman_batch.csv', index=False)
        pd.DataFrame(all_wilcoxon).to_csv(
            self.out_dir / 'statistical_tests_batch.csv', index=False)
        pd.concat(all_cd).to_csv(
            self.out_dir / 'cd_ranks_batch.csv', index=False)

        # Ablation stats
        self._ablation_stats()

        print('  Tables A statistical analysis complete.')
        return {'friedman': len(all_friedman), 'wilcoxon': len(all_wilcoxon)}

    def _run_streaming(self) -> dict:
        print('\n[StatsAgent] Running Table B (streaming) statistical analysis...')

        path = self.out_dir / 'benchmark_results_streaming.csv'
        if not path.exists():
            print(f'  WARNING: {path} not found. Skipping.')
            return {}

        df = pd.read_csv(path)
        algos = sorted(df['algorithm'].unique())
        k     = len(algos)
        if k < 2:
            print('  Not enough algorithms. Skipping.')
            return {}

        print(f'  Algorithms: {algos}')

        all_friedman, all_wilcoxon, all_cd = [], [], []

        for diff in DIFFICULTIES:
            df_d = df[df['difficulty'] == diff]
            folds = sorted(df_d['fold'].unique())
            sm    = np.zeros((len(folds), k))
            for mi, mfold in enumerate(folds):
                for ai, algo in enumerate(algos):
                    sub = df_d[(df_d['fold'] == mfold) & (df_d['algorithm'] == algo)]
                    sm[mi, ai] = sub['AUC_PR'].mean() if len(sub) > 0 else 0.0

            chi2, p_f = friedman_test(sm, k)
            all_friedman.append({
                'difficulty': diff,
                'chi2': chi2, 'df': k - 1,
                'p_friedman': p_f,
                'significant': bool(p_f < 0.05),
            })

            pairs = wilcoxon_matrix(sm, algos, diff)
            all_wilcoxon.extend(pairs)

            cd_df, cd = cd_ranks(sm, algos, k)
            cd_df['difficulty'] = diff
            all_cd.append(cd_df)

            print(f'  {diff:7s}: Friedman p={p_f:.4f}, CD={cd:.3f}')

        pd.DataFrame(all_friedman).to_csv(
            self.out_dir / 'friedman_streaming.csv', index=False)
        pd.DataFrame(all_wilcoxon).to_csv(
            self.out_dir / 'statistical_tests_streaming.csv', index=False)
        pd.concat(all_cd).to_csv(
            self.out_dir / 'cd_ranks_streaming.csv', index=False)

        print('  Table B statistical analysis complete.')
        return {'friedman': len(all_friedman), 'wilcoxon': len(all_wilcoxon)}

    def _ablation_stats(self):
        """Paired Wilcoxon for ablation study."""
        path = self.out_dir / 'ablation_results.csv'
        if not path.exists():
            print('  [Ablation] No data file found.')
            return

        df = pd.read_csv(path)
        rows = []
        for algo in df['algorithm'].unique():
            for diff in DIFFICULTIES:
                sub = df[(df['algorithm'] == algo) & (df['difficulty'] == diff)]
                ctrl  = sub['AUC_PR_control'].dropna()
                treat = sub['AUC_PR_treatment'].dropna()

                if len(ctrl) < 3 or len(treat) < 3:
                    continue
                min_len = min(len(ctrl), len(treat))
                ctrl, treat = ctrl.values[:min_len], treat.values[:min_len]

                delta = treat.mean() - ctrl.mean()
                try:
                    wr = stats.wilcoxon(treat - ctrl, alternative='greater')
                    wp, wstat = float(wr.pvalue), float(wr.statistic)
                except Exception:
                    wp, wstat = float('nan'), float('nan')

                try:
                    t_stat, p_tt = stats.ttest_rel(treat, ctrl)
                except Exception:
                    t_stat, p_tt = float('nan'), float('nan')

                sig = wp < 0.05 if not np.isnan(wp) else False
                rows.append({
                    'algorithm': algo, 'difficulty': diff,
                    'mean_ctrl': ctrl.mean(), 'mean_treat': treat.mean(),
                    'delta': delta,
                    'W_stat': wstat, 'p_wilcoxon': wp,
                    't_stat': t_stat, 'p_ttest': p_tt,
                    'sig_wilcoxon': sig,
                    'n': min_len,
                })

        if rows:
            pd.DataFrame(rows).to_csv(
                self.out_dir / 'ablation_stats.csv', index=False)
            print('  Ablation stats saved.')
            for r in rows:
                sig = '***' if r['p_wilcoxon'] < 0.001 else '**' if r['p_wilcoxon'] < 0.01 else '*' if r['p_wilcoxon'] < 0.05 else ''
                print(f"    {r['algorithm']:20s} {r['difficulty']:8s}: "
                      f"Δ={r['delta']:+.4f} p={r['p_wilcoxon']:.4f} {sig}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', choices=['batch', 'streaming'], required=True)
    args = parser.parse_args()
    StatsAgent().run(table=args.table)
