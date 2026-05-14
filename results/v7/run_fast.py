"""Fast benchmark run for v7 — 2 folds, 3 seeds, 2 budgets."""
import sys, os
sys.path.insert(0, r'C:\proj\ldt\results\v7')
import benchmark_v7 as bm

# Override for speed: 2 folds (Feb+Mar), 3 seeds, 2 label budgets
bm.MONTHS = [1, 2, 3]
bm.SEEDS = [42, 123, 456]
bm.LABEL_BUDGETS = [0, 500]
bm.ANOMALY_RATE = 0.05
bm.ANOMALY_N = int(bm.TEST_N * bm.ANOMALY_RATE)
bm.ANOMALY_PARAMS = {
    'easy':   {'type': 'extreme_fare', 'fare_range': (150, 500), 'n': bm.ANOMALY_N},
    'medium': {'type': 'extreme_fare', 'fare_range': (80, 150),  'n': bm.ANOMALY_N},
    'hard':   {'type': 'partition', 'n': bm.ANOMALY_N,
               'components': [
                   ('extreme_fare', (60, 80), 1),
                   ('zero_dist', None, 1),
                   ('slow_crawl', None, 1),
               ]},
}

# Clean slate
for f in ['checkpoint_v7.csv', 'benchmark_results_v7.csv', 'bar_score_results_v7.csv',
          'statistical_results.txt', 'benchmark_v7_results.md',
          'fig_overview_v7.png', 'fig_difficulty_v7.png', 'fig_ablation_v7.png',
          'fig_bar_score_v7.png', 'fig_pareto_frontier_v7.png']:
    path = bm.OUT_DIR / f
    if path.exists():
        os.remove(path)
        print('Removed old', f)

batch_jobs = 7 * 3 * 2 * 3
stream_jobs = 4 * 3 * 2 * 3 * 2
total = batch_jobs + stream_jobs
print('\nFast v7 config:')
print('  Months: Jan-Mar (folds: test=Feb, test=Mar)')
print('  Seeds:', bm.SEEDS)
print('  Label budgets:', bm.LABEL_BUDGETS)
print('  Anomaly rate: 5% (500 anomalies)')
print('  Batch jobs:', batch_jobs, '(7 algos x 3 seeds x 2 folds x 3 difficulties)')
print('  Stream jobs:', stream_jobs, '(4 algos x 3 seeds x 2 folds x 3 diffs x 2 budgets)')
print('  Total:', total)

bm.run()
