import os

# Delete stage2 and stage3 results (they're incomplete)
for stage in ['stage2', 'stage3']:
    stage_dir = rf'C:\proj\ldt\HP_benchmark_v4\results\grid_search\{stage}'
    if os.path.exists(stage_dir):
        for f in os.listdir(stage_dir):
            os.remove(os.path.join(stage_dir, f))
        print(f'Cleared: {stage_dir}')

# Delete stale final results
for f in ['final_results.json', 'best_config.json']:
    fp = rf'C:\proj\ldt\HP_benchmark_v4\results\grid_search\{f}'
    if os.path.exists(fp):
        os.remove(fp)
        print(f'Deleted: {fp}')

print('Done. Ready to re-run grid_search.py --stage 2,3')
