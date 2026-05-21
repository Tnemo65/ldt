import os, json

# Check all stages
for stage in ['stage1', 'stage2', 'stage3']:
    stage_dir = rf'C:\proj\ldt\HP_benchmark_v4\results\grid_search\{stage}'
    if not os.path.exists(stage_dir):
        print(f'{stage}: directory does not exist')
        continue
    files = [f for f in os.listdir(stage_dir) if f.endswith('.json') and f != 'summary.json']
    ok = err = 0
    for f in files:
        with open(os.path.join(stage_dir, f)) as fh:
            d = json.load(fh)
        if 'error' in d:
            err += 1
        else:
            ok += 1
    print(f'{stage}: {len(files)} files, {ok} OK, {err} errors')

    # Summary check
    sum_path = os.path.join(stage_dir, 'summary.json')
    if os.path.exists(sum_path):
        with open(sum_path) as fh:
            s = json.load(fh)
        s_ok = sum(1 for r in s['results'] if 'error' not in r)
        s_err = sum(1 for r in s['results'] if 'error' in r)
        print(f'  summary.json: {len(s["results"])} entries, {s_ok} OK, {s_err} errors')
        best = s.get('best', {})
        print(f'  best: {best.get("config_id","?")} - error: {"error" in best}')
        if 'error' not in best:
            print(f'  best AUC-PR: {best.get("auc_pr", 0):.4f}, AUC-ROC: {best.get("auc_roc", 0):.4f}')
    else:
        print(f'  summary.json: does not exist')
    print()

# Check final results
for fname in ['best_config.json', 'final_results.json']:
    fpath = rf'C:\proj\ldt\HP_benchmark_v4\results\grid_search\{fname}'
    if os.path.exists(fpath):
        with open(fpath) as fh:
            d = json.load(fh)
        has_err = 'error' in d
        pr = d.get('auc_pr', 0)
        cid = d.get('config_id', '?')
        print(f'{fname}: error={has_err}, auc_pr={pr:.4f}, config_id={cid}')
    else:
        print(f'{fname}: does not exist')
