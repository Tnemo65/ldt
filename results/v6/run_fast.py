"""Fast benchmark run for v6 — 3 months, 3 seeds, 2 budgets."""
import sys, os
sys.path.insert(0, r'C:\proj\ldt\results\v6')
import benchmark_v6 as bm

# Override for speed: 3 months = 2 folds, 3 seeds, 2 label budgets
bm.MONTHS = [1, 2, 3]
bm.SEEDS = [42, 123, 456]
bm.LABEL_BUDGETS = [0, 500]  # unsupervised + budget=500 only

# Clean slate
for f in ['checkpoint_v6.csv', 'benchmark_results_v6.csv', 'bar_score_results_v6.csv',
          'statistical_results.txt', 'benchmark_v6_results.md']:
    path = bm.OUT_DIR / f
    if path.exists():
        os.remove(path)
        print('Removed old', f)

batch_jobs = len(bm.ALGO_NAMES_BATCH) * len(bm.SEEDS) * 2 * len(bm.DIFFICULTIES)
stream_jobs = len(bm.ALGO_NAMES_STREAM) * len(bm.SEEDS) * 2 * len(bm.DIFFICULTIES) * len(bm.LABEL_BUDGETS)
total = batch_jobs + stream_jobs
print('\nFast config:')
print('  Months: Jan-Mar (folds: test=Feb, test=Mar)')
print('  Seeds:', bm.SEEDS)
print('  Label budgets:', bm.LABEL_BUDGETS)
print('  Batch jobs:', batch_jobs, '(5 algos x 3 seeds x 2 folds x 3 difficulties)')
print('  Stream jobs:', stream_jobs, '(3 algos x 3 seeds x 2 folds x 3 diffs x 2 budgets)')
print('  Total:', total)

bm.run()
