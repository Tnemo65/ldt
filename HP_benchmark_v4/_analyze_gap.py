import json, os, numpy as np

# 1. Grid search best config
gs_path = r'C:\proj\ldt\HP_benchmark_v4\results\grid_search\best_config.json'
with open(gs_path) as f:
    cfg = json.load(f)
print('=== GRID SEARCH best ===')
print('epochs:', cfg.get('epochs'))
print('auc_pr:', round(cfg.get('auc_pr'), 4))
print('auc_roc:', round(cfg.get('auc_roc'), 4))

# 2. Ablation results (if saved)
ab_dir = r'C:\proj\ldt\HP_benchmark_v4\results\ablation'
ab_file = os.path.join(ab_dir, 'ablation_results.json')
if os.path.exists(ab_file):
    with open(ab_file) as f:
        ab = json.load(f)
    print('\n=== ABLATION results (saved) ===')
    for r in ab.get('results', []):
        print('  ' + str(r.get('setup')) + ' AUC-PR=' + str(round(r.get('auc_pr', 0), 4)) + ' AUC-ROC=' + str(round(r.get('auc_roc', 0), 4)))

# 3. Key question: WHY different?
print('\n=== SO SANH ===')
print('Grid search: valid_polluted.parquet, 500K train, epochs=' + str(cfg.get('epochs')))
print('Ablation:    test_polluted.parquet,  200K train, epochs=2000')
print('')
print('Su khac biet:')
print('1. Dataset: validation vs test (8% anomaly nhung phan bo cuc bo khac)')
print('2. Train size: 500K vs 200K (AE co nhieu data hon -> tot hon)')
print('3. Phuong phap scoring: score_stream (streaming + online update) vs score_batch (batch, fixed memory)')
print('')
print('Yeu to quan trong nhat: Phuong phap scoring khac nhau!')
print('- score_stream: memory cap nhat online trong qua trinh scoring, Welford stats thay doi')
print('- score_batch: memory co dinh, chi dung de scoring')
print('')
print('Ket luan: Chi nao co the chay cung phuong phap scoring?')
print('Grid search: score_stream (streaming)')
print('Ablation: score_batch (batch, fixed memory)')
print('=> Day la ly do chinh gay ra chenh lenh AUC-PR')
