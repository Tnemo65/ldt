import os, json

base = r'C:\proj\ldt\HP_benchmark_v4\results\grid_search'

for stage in ['stage1', 'stage2']:
    stage_dir = os.path.join(base, stage)
    if not os.path.exists(stage_dir):
        continue

    files = [f for f in os.listdir(stage_dir) if f.endswith('.json') and f != 'summary.json']
    results = []
    for fn in files:
        with open(os.path.join(stage_dir, fn)) as fh:
            d = json.load(fh)
        results.append(d)

    real_results = [r for r in results if 'error' not in r and r.get('auc_pr', 0) > 0.25]
    corrupted_results = [r for r in results if 'error' not in r and r.get('auc_pr', 0) <= 0.25]

    print(stage + ': ' + str(len(real_results)) + ' real (>0.25 AUC-PR), ' + str(len(corrupted_results)) + ' corrupted')

    for r in corrupted_results[:5]:
        cid = r.get('config_id', '?')
        apr = r.get('auc_pr', 0)
        print('  CORRUPTED: ' + cid + ' AUC-PR=' + str(round(apr, 4)))

    for r in corrupted_results:
        fpath = os.path.join(stage_dir, r['config_id'] + '.json')
        if os.path.exists(fpath):
            os.remove(fpath)

    real_results.sort(key=lambda x: x.get('auc_pr', 0), reverse=True)
    stg_num = int(stage.replace('stage', ''))
    summary = {
        'stage': stg_num,
        'results': real_results,
        'best': real_results[0] if real_results else {}
    }
    with open(os.path.join(stage_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    if real_results:
        cid = real_results[0].get('config_id', '?')
        apr = real_results[0].get('auc_pr', 0)
        print('  Best: ' + cid + ' AUC-PR=' + str(round(apr, 4)))

all_results = []
for stage in ['stage1', 'stage2']:
    sp = os.path.join(base, stage, 'summary.json')
    if os.path.exists(sp):
        with open(sp) as fh:
            s = json.load(fh)
        all_results.extend(s.get('results', []))

all_results.sort(key=lambda x: x.get('auc_pr', 0), reverse=True)
best = all_results[0] if all_results else {}

with open(os.path.join(base, 'best_config.json'), 'w') as fh:
    json.dump(best, fh, indent=2)

with open(os.path.join(base, 'final_results.json'), 'w') as fh:
    json.dump({'all_results': all_results[:100], 'best_config': best}, fh, indent=2)

cid = best.get('config_id', '?')
apr = best.get('auc_pr', 0)
print('\nOverall best: ' + cid + ' AUC-PR=' + str(round(apr, 4)))
print('final_results.json: ' + str(len(all_results)) + ' total configs')

s3_summary = os.path.join(base, 'stage3', 'summary.json')
if os.path.exists(s3_summary):
    with open(s3_summary) as fh:
        s3 = json.load(fh)
    if len(s3.get('results', [])) == 0:
        os.remove(s3_summary)
        print('Deleted empty stage3/summary.json')
