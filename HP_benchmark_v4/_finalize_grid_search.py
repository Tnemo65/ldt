import os, json

base = r'C:\proj\ldt\HP_benchmark_v4\results\grid_search'

for stage in ['stage1', 'stage2']:
    stage_dir = os.path.join(base, stage)
    if not os.path.exists(stage_dir):
        print(f'{stage}: dir missing')
        continue

    files = [f for f in os.listdir(stage_dir) if f.endswith('.json') and f != 'summary.json']
    results = []
    for f in files:
        with open(os.path.join(stage_dir, f)) as fh:
            d = json.load(fh)
        results.append(d)

    results.sort(key=lambda x: x.get('auc_pr', 0), reverse=True)
    err_count = sum(1 for r in results if 'error' in r)
    print(f'{stage}: {len(results)} configs, {err_count} errors')

    # Write summary
    summary_path = os.path.join(stage_dir, 'summary.json')
    summary = {
        'stage': int(stage.replace('stage', '')),
        'results': results,
        'best': results[0] if results else {}
    }
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'  -> {summary_path}')

    if results:
        best = results[0]
        print(f'  best: {best.get("config_id","?")} AUC-PR={best.get("auc_pr",0):.4f}')

# Overall best across stages
all_results = []
for stage in ['stage1', 'stage2', 'stage3']:
    sdir = os.path.join(base, stage)
    sp = os.path.join(sdir, 'summary.json')
    if os.path.exists(sp):
        with open(sp) as fh:
            s = json.load(fh)
        all_results.extend(s.get('results', []))

all_results.sort(key=lambda x: x.get('auc_pr', 0), reverse=True)
best = all_results[0] if all_results else {}

# Save best_config.json
best_path = os.path.join(base, 'best_config.json')
with open(best_path, 'w') as fh:
    json.dump(best, fh, indent=2)
print(f'\nbest_config: {best.get("config_id","?")} AUC-PR={best.get("auc_pr",0):.4f} -> {best_path}')

# Save final_results.json
final = {
    'all_results': all_results[:100],  # top 100
    'best_config': best,
    'stages_summary': {
        'stage1_best': 'see results',
        'stage2_best': 'see results',
        'stage3_best': 'see results',
    }
}
fp = os.path.join(base, 'final_results.json')
with open(fp, 'w') as fh:
    json.dump(final, fh, indent=2)
print(f'final_results -> {fp}')
