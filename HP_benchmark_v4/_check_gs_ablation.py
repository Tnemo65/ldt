import json, os

# Grid search best config
gs_path = r'C:\proj\ldt\HP_benchmark_v4\results\grid_search\best_config.json'
with open(gs_path) as f:
    cfg = json.load(f)

print('=== GRID SEARCH best_config ===')
print('epochs:', cfg.get('epochs'))
print('train_time_s:', round(cfg.get('train_time_s', 0), 1))
print('auc_pr:', round(cfg.get('auc_pr', 0), 4))
print('auc_roc:', round(cfg.get('auc_roc', 0), 4))
print('anomaly_rate: validation 8.0%')

# Stage 1 top 3
s1_path = r'C:\proj\ldt\HP_benchmark_v4\results\grid_search\stage1\summary.json'
with open(s1_path) as f:
    s1 = json.load(f)
print('\nStage 1 top 3:')
for i, r in enumerate(s1['results'][:3]):
    print('  epochs=' + str(r.get('epochs')) + ' time=' + str(round(r.get('train_time_s', 0), 1)) + 's auc_pr=' + str(round(r.get('auc_pr', 0), 4)))

# Check what dataset ablation uses
v3_path = os.path.join(r'c:\proj\ldt', 'HP_benchmark_v3')
test_gt = os.path.join(v3_path, 'test', 'ground_truth_mask.npy')
valid_gt = os.path.join(v3_path, 'valid', 'ground_truth_mask.npy')

import numpy as np
if os.path.exists(valid_gt):
    vg = np.load(valid_gt)
    print('\nValidation GT: ' + str(len(vg)) + ' rows, ' + str(int(vg.sum())) + ' anomalies, ' + str(round(vg.mean()*100, 2)) + '%')

if os.path.exists(test_gt):
    tg = np.load(test_gt)
    print('Test GT: ' + str(len(tg)) + ' rows, ' + str(int(tg.sum())) + ' anomalies, ' + str(round(tg.mean()*100, 2)) + '%')

# Check ablation output
ablation_path = r'C:\proj\ldt\HP_benchmark_v4\results\ablation\ablation_results.json'
if os.path.exists(ablation_path):
    with open(ablation_path) as f:
        ab = json.load(f)
    print('\n=== ABLATION results ===')
    for setup in ab.get('results', []):
        apr = setup.get('auc_pr', 0)
        aro = setup.get('auc_roc', 0)
        desc = setup.get('description', '')
        print('  ' + desc + ' => AUC-PR=' + str(round(apr, 4)) + ' AUC-ROC=' + str(round(aro, 4)))
else:
    print('\nAblation: no results file yet')

# Check if ablation directory exists
print('\nAblation dir contents:')
ab_dir = r'C:\proj\ldt\HP_benchmark_v4\results\ablation'
if os.path.exists(ab_dir):
    for fn in sorted(os.listdir(ab_dir)):
        print('  ' + fn)
