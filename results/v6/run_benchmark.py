"""Full benchmark run for v6 — Scientific Rigour Overhaul."""
import sys, os
sys.path.insert(0, r'C:\proj\ldt\results\v6')
import benchmark_v6 as bm

for f in ['checkpoint_v6.csv', 'benchmark_results_v6.csv', 'bar_score_results_v6.csv',
          'statistical_results.txt', 'benchmark_v6_results.md']:
    path = bm.OUT_DIR / f
    if path.exists():
        os.remove(path)
        print('Removed old', f)

batch_jobs = len(bm.ALGO_NAMES_BATCH) * len(bm.SEEDS) * 5 * len(bm.DIFFICULTIES)
stream_jobs = len(bm.ALGO_NAMES_STREAM) * len(bm.SEEDS) * 5 * len(bm.DIFFICULTIES) * len(bm.LABEL_BUDGETS)
total = batch_jobs + stream_jobs
print('\nJob count:')
print('  Batch:', batch_jobs, '(5 algos x 5 seeds x 5 folds x 3 difficulties)')
print('  Stream:', stream_jobs, '(3 algos x 5 seeds x 5 folds x 3 difficulties x 4 budgets)')
print('  Total:', total)

bm.run()
