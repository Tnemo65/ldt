#!/usr/bin/env python3
import json
from pathlib import Path

print("=" * 100)
print("  HYPERPARAMETER EXPERIMENT SUMMARY (STREAMING EVALUATION)")
print("  Model: MemStream GPU kNN | Dataset: NYC Taxi | Warmup: 10K | Test: 15K | Anomalies: 711")
print("=" * 100)

results_data = []
for f in sorted(Path('results').glob('exp*.json')):
    try:
        d = json.load(open(f))
        name = d.get('experiment', f.stem)
        results = d.get('results', [])
        best = d.get('best_config', {})
        if not best and results:
            best = results[0]
        if not best:
            best = {}
        m = best.get('metrics', best)

        hp_key = None
        hp_val = None
        for k in ['memory_len', 'k', 'gamma', 'default_beta', 'delta', 'max_window',
                   'warmup_epochs', 'warmup_data_size', 'cell_minimum', 'buffer_size',
                   'n_samples', 'n_epochs']:
            if k in best:
                hp_key = k
                hp_val = best[k]
                break

        results_data.append({
            'name': name,
            'hp_key': hp_key,
            'hp_val': hp_val,
            'auc_roc': m.get('AUC-ROC'),
            'auc_pr': m.get('AUC-PR'),
            'f1': m.get('F1'),
            'prec': m.get('Precision'),
            'rec': m.get('Recall'),
            'tp': m.get('TP'),
            'lat_p99': m.get('latency_p99_ms'),
            'threshold': m.get('threshold'),
        })
    except Exception as e:
        print(f"  ERROR {f.name}: {e}")

def fmt(v, decimals=4):
    if isinstance(v, (int, float)):
        return f"{v:.{decimals}f}"
    return "N/A"

print(f"\n{'Exp':<35} {'Best HP':<20} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>6} {'Prec':>6} {'Rec':>6} {'TP':>5} {'Thresh':>8} {'P99ms':>6}")
print("-" * 120)
for s in sorted(results_data, key=lambda x: x['auc_roc'] or 0, reverse=True):
    hp = f"{s['hp_key']}={s['hp_val']}" if s['hp_key'] else ""
    t = fmt(s['threshold'], 0) if s['threshold'] else "N/A"
    lat = fmt(s['lat_p99'], 1) if s['lat_p99'] else "N/A"
    print(f"{s['name']:<35} {hp:<20} {fmt(s['auc_roc']):>8} {fmt(s['auc_pr']):>8} "
          f"{fmt(s['f1']):>6} {fmt(s['prec']):>6} {fmt(s['rec']):>6} "
          f"{s['tp'] or 'N/A':>5} {t:>8} {lat:>6}")

print("\n" + "=" * 100)
print("  TOP RECOMMENDATIONS")
print("=" * 100)
print("""
  1. PRIMARY: Use k=50 neighbors (AUC-ROC=0.7329)
     - Best hyperparameter across all streaming experiments
     - AUC-PR=0.1863, F1=0.2022, Recall=57.4%

  2. EPOCHS: Increase warmup_epochs to 50-100 (AUC-ROC=0.7670 at 100 epochs)
     - Most impactful parameter for detection quality
     - Longer training = better feature learning

  3. WEIGHTING: Use uniform kNN (gamma=0.0)
     - Decay weighting degrades performance on this dataset
     - Separation gap: 1077 with uniform vs <263 with decay

  4. MEMORY: streaming memory_len has no impact on scoring
     - Warmup memory (10K) fully dominates kNN distance calculation
     - Use memory_len=256 for streaming buffer only (low latency overhead)

  5. CONTEXT: ContextBeta provides minimal value
     - Only 2-13/80 context cells populated in NYC taxi data
     - Limited feature diversity reduces context utility

  6. DRIFT: ADWIN detects concept drift reliably (Recall=100%)
     - But precision is very low (0.2%) due to score variability
     - Pair with context-aware thresholding for better precision

  7. THRESHOLD: Use FPR-based thresholds for operational control
     - FPR@1: Precision=27%, Recall=7% (conservative)
     - FPR@5: Precision=9%, Recall=10% (moderate)
     - FPR@20: Precision=12%, Recall=55% (aggressive)

  8. LATENCY: Streaming scores at 0.8ms mean, 2.7ms P99
     - Suitable for real-time fraud detection pipelines
     - Kafka throughput: ~1000 records/sec in batch mode
""")
print("=" * 100)
