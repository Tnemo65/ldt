import sys, numpy as np, json
sys.path.insert(0, r'C:\proj\ldt\HP_benchmark_v5')

print('='*80)
print('ANALYSIS 4: ANOMALY DETECTION PERFORMANCE ANALYSIS')
print('='*80)

from benchmark_core import extract_features_from_parquet, MemStreamPipeline, evaluate_scores, find_best_threshold
import pyarrow.parquet as pq
import pandas as pd

# Load validation data
X_val, hours_val, dows_val, rcs_val, nb_val = extract_features_from_parquet(
    r'C:\proj\ldt\HP_benchmark_v5\data\valid_polluted.parquet',
    max_rows=200000
)
gt_mask = np.load(r'C:\proj\ldt\HP_benchmark_v5\data\valid\ground_truth_mask.npy')
gt_mask = gt_mask[-len(X_val):]

print(f'\n=== VALIDATION DATA ===')
print(f'X_val shape: {X_val.shape}')
print(f'GT mask: {gt_mask.sum()} anomalies / {len(gt_mask)} = {gt_mask.mean()*100:.2f}%')

# Per-type ground truth
per_type = json.load(open(r'C:\proj\ldt\HP_benchmark_v5\data\valid\ground_truth_per_type.json'))
print(f'\n=== ANOMALY TYPES ===')
for t, info in sorted(per_type.items(), key=lambda x: int(x[0]) if x[0].lstrip('-').isdigit() else 999):
    indices = np.array(info['indices'])
    in_range = indices < len(X_val)
    print(f'Type {t}: {in_range.sum()} in val, name={info.get("name","")}, key_feature={info.get("key_feature","")}, key_signal={info.get("key_signal","")}')

# Load training data
X_train, hours_train, dows_train, rcs_train, nb_train = extract_features_from_parquet(
    r'C:\proj\ldt\HP_benchmark_v5\data\train_clean.parquet',
    max_rows=100000
)

# Train pipeline
print(f'\n=== TRAINING PIPELINE ===')
pipe = MemStreamPipeline(
    d=34, out_dim=68, memory_len=512, k=3,
    gamma=0.0, beta=0.5, noise_std=0.001,
    lr=0.01, epochs=200, batch_size=1024,
    verbose=False
)
pipe.train(X_train, hours_train, dows_train, rcs_train, nb_train)

# Score validation
scores, metrics = pipe.score_stream(
    X_val, hours_val, dows_val, rcs_val, nb_val,
    gt_mask=gt_mask, update_memory=False
)

print(f'\n=== SCORING RESULTS ===')
print(f'AUC-ROC: {metrics["auc_roc"]:.4f}')
print(f'AUC-PR: {metrics["auc_pr"]:.4f}')
print(f'F1: {metrics["f1"]:.4f}')
print(f'Precision: {metrics["precision"]:.4f}')
print(f'Recall: {metrics["recall"]:.4f}')
print(f'Best threshold: {metrics["best_threshold"]:.4f}')
print(f'Score normal mean: {metrics["score_normal_mean"]:.4f}')
print(f'Score anomaly mean: {metrics["score_anomaly_mean"]:.4f}')
print(f'Separation ratio: {metrics["separation_ratio"]:.2f}x')

# Score distribution analysis
print(f'\n=== SCORE DISTRIBUTION ===')
print(f'Scores min: {scores.min():.4f}, max: {scores.max():.4f}')
print(f'Scores mean: {scores.mean():.4f}, std: {scores.std():.4f}')
nm = scores[gt_mask == 0]
am = scores[gt_mask == 1]
print(f'Normal scores: mean={nm.mean():.4f}, std={nm.std():.4f}, p25={np.percentile(nm,25):.4f}, p75={np.percentile(nm,75):.4f}, p95={np.percentile(nm,95):.4f}, p99={np.percentile(nm,99):.4f}')
print(f'Anomaly scores: mean={am.mean():.4f}, std={am.std():.4f}, p25={np.percentile(am,25):.4f}, p75={np.percentile(am,75):.4f}, p95={np.percentile(am,95):.4f}, p99={np.percentile(am,99):.4f}')

# Per-anomaly-type analysis
print(f'\n=== PER-TYPE SCORE ANALYSIS ===')
for t, info in sorted(per_type.items(), key=lambda x: int(x[0]) if x[0].lstrip('-').isdigit() else 999):
    indices = np.array(info['indices'])
    in_range = indices < len(X_val)
    if in_range.sum() == 0:
        continue
    local_indices = indices[in_range]
    type_scores = scores[local_indices]
    type_gt = gt_mask[local_indices]
    name = info.get('name', '')
    print(f'Type {t} ({name}): n={len(local_indices)}, score_mean={type_scores.mean():.4f}, score_std={type_scores.std():.4f}, gt_anomaly={type_gt.mean()*100:.1f}%')

# Check overlap between normal and anomaly score distributions
print(f'\n=== SCORE OVERLAP ANALYSIS ===')
# Find threshold that gives best F1
best_t, best_f1 = find_best_threshold(scores, gt_mask)
print(f'Best threshold: {best_t:.4f} (F1={best_f1:.4f})')

# Score at different percentiles for normal vs anomaly
nm_sorted = np.sort(nm)
am_sorted = np.sort(am)
for pct in [50, 75, 90, 95, 99]:
    nm_pct = np.percentile(nm, pct)
    am_pct = np.percentile(am, pct)
    overlap = nm_pct < am_pct
    print(f'P{pct}: Normal={nm_pct:.4f}, Anomaly={am_pct:.4f}, overlap={overlap}')

# ROC-style analysis
print(f'\n=== RECALL AT DIFFERENT THRESHOLDS ===')
for t in [10, 20, 30, 40, 50, 60, 70, 80, 100]:
    pred = (scores >= t).astype(int)
    tp = ((pred == 1) & (gt_mask == 1)).sum()
    fn = ((pred == 0) & (gt_mask == 1)).sum()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fp = ((pred == 1) & (gt_mask == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f'Threshold {t:3d}: recall={recall:.4f}, precision={precision:.4f}, tp={tp}, fp={fp}, fn={fn}')
