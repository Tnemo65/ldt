import json
import numpy as np

with open('C:/proj/ldt/explore_memstream/results/v10_eval/v10_eval_20260514_155728.json') as f:
    d = json.load(f)

print("=== EVALUATOR STATE ===")
print(f"n_train: {d['n_train']}, n_test: {d['n_test']}, n_anomalies: {d['n_anomalies']}")
print(f"Eval mode: {d['eval_mode']}, device: {d['device']}, fraud_type: {d['fraud_type']}")
print()

for r in d['results']:
    print(f"{'='*60}")
    print(f"Config: {r['name']}")
    print(f"  Metrics:")
    print(f"    F1={r['F1']:.4f}  AUC-PR={r['AUC_PR']:.4f}  AUC-ROC={r['AUC_ROC']:.4f}")
    print(f"    Precision={r['Precision']:.4f}  Recall={r['Recall']:.4f}  FPR={r['FPR']:.4f}")
    print(f"    TP={r['TP']}  FP={r['FP']}  TN={r['TN']}  FN={r['FN']}")
    print(f"    Threshold used: {r['threshold_used']:.4f}")
    print(f"    Elapsed: {r['elapsed_s']:.1f}s")
    print()

# Also load the batch results
print("\n=== BATCH vs STREAMING COMPARISON ===")
for mode_file, mode_label in [
    ('C:/proj/ldt/explore_memstream/results/v10_eval/v10_eval_20260514_155534.json', 'BATCH'),
    ('C:/proj/ldt/explore_memstream/results/v10_eval/v10_eval_20260514_155728.json', 'STREAMING'),
]:
    try:
        with open(mode_file) as f:
            d = json.load(f)
        print(f"\n{mode_label}:")
        for r in d['results']:
            f1 = r.get('F1', 'N/A')
            aucpr = r.get('AUC_PR', 'N/A')
            print(f"  {r['name']:<25} F1={f1}  AUC-PR={aucpr}")
    except Exception as e:
        print(f"  Error: {e}")
