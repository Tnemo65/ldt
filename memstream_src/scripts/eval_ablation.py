#!/usr/bin/env python3
"""
Ablation Study: Compare MemStream gốc (25D) vs CA-MemStream (40D).

Scientific purpose:
- Verify that 4D Context-Aware improves performance
- Measure false alarm reduction
- Validate BAR Score contribution

Usage:
    python -m memstream_src.scripts.eval_ablation \
        --data /data/test.csv \
        --model-25d /models/memstream_25d.pt \
        --model-40d /models/memstream_40d.pt \
        --output /results/ablation_results.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memstream_core import MemStreamCore
from core.feature_extractor import FeatureVectorizer
from core.context_aware import ContextAwareFeatureVectorizer, get_4d_context
from scripts.inject_anomalies_multi import inject_anomalies


def evaluate_model(
    model: MemStreamCore,
    vectorizer,
    df: pd.DataFrame,
    labels: np.ndarray,
    use_context: bool = False,
) -> Dict:
    """
    Evaluate model and compute metrics.

    Args:
        model: Trained MemStream model
        vectorizer: Feature vectorizer (25D or 40D)
        df: Test DataFrame
        labels: Ground truth labels (1=anomaly, 0=normal)
        use_context: Whether vectorizer expects context

    Returns:
        Dict with evaluation metrics
    """
    scores = []
    predictions = []

    for idx, row in df.iterrows():
        if use_context:
            ctx = get_4d_context(row.to_dict())
            features = vectorizer.transform(row.to_dict(), ctx)
        else:
            features = vectorizer.transform(row.to_dict())

        score = model.score_one(features)
        scores.append(score)
        predictions.append(1 if score > model.max_thres.item() else 0)

    scores = np.array(scores)
    predictions = np.array(predictions)

    # Compute metrics
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate

    # AUC-PR (Area Under Precision-Recall Curve)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    auc_pr = auc(recall_curve, precision_curve)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'fpr': float(fpr),  # This is the FALSE ALARM RATE
        'auc_pr': float(auc_pr),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }


def evaluate_per_context(
    model: MemStreamCore,
    vectorizer,
    df: pd.DataFrame,
    labels: np.ndarray,
    use_context: bool,
) -> Dict:
    """
    Evaluate model per context (for false alarm analysis).

    Returns per-context metrics to show where context-awareness helps.
    """
    results = {}

    for idx, row in df.iterrows():
        ctx = get_4d_context(row.to_dict())
        ctx_key = f"{ctx['neighborhood']}_{ctx['hour_bucket']}_{ctx['day_type']}"

        if ctx_key not in results:
            results[ctx_key] = {'scores': [], 'labels': [], 'ctx': ctx}

        if use_context:
            features = vectorizer.transform(row.to_dict(), ctx)
        else:
            features = vectorizer.transform(row.to_dict())

        score = model.score_one(features)
        results[ctx_key]['scores'].append(score)
        results[ctx_key]['labels'].append(labels[idx])

    # Compute per-context metrics
    per_context_metrics = {}
    for ctx_key, data in results.items():
        scores = np.array(data['scores'])
        labels_ctx = np.array(data['labels'])
        preds = (scores > model.max_thres.item()).astype(int)

        tp = np.sum((preds == 1) & (labels_ctx == 1))
        fp = np.sum((preds == 1) & (labels_ctx == 0))
        tn = np.sum((preds == 0) & (labels_ctx == 0))

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        n_samples = len(scores)
        n_anomalies = int(np.sum(labels_ctx))
        n_normal = n_samples - n_anomalies

        per_context_metrics[ctx_key] = {
            'n_samples': n_samples,
            'n_anomalies': n_anomalies,
            'n_normal': n_normal,
            'fpr': float(fpr),
            'ctx': data['ctx'],
        }

    return per_context_metrics


def main():
    parser = argparse.ArgumentParser(description='Ablation Study: 25D vs 40D')
    parser.add_argument('--data', type=str, required=True, help='Test CSV path')
    parser.add_argument('--model-25d', type=str, required=True, help='25D model path')
    parser.add_argument('--model-40d', type=str, required=True, help='40D model path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path')
    parser.add_argument('--n-anomalies', type=int, default=1500, help='Number of anomalies to inject')
    parser.add_argument('--signing-key', type=str, default='training-signing-key',
                        help='HMAC signing key')
    args = parser.parse_args()

    print("=" * 60)
    print("ABLATION STUDY: MemStream gốc (25D) vs CA-MemStream (40D)")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data...")
    df = pd.read_csv(args.data)
    print(f"  Loaded {len(df):,} records")

    # Inject anomalies
    print("\n[2] Injecting anomalies...")
    df_anom, labels = inject_anomalies(df, n_anomalies=args.n_anomalies)
    print(f"  Total records: {len(df_anom):,}")
    print(f"  Anomalies: {int(np.sum(labels)):,} ({np.mean(labels)*100:.1f}%)")

    # Load models
    print("\n[3] Loading models...")
    model_25d = MemStreamCore.load(args.model_25d, signing_key=args.signing_key)
    model_40d = MemStreamCore.load(args.model_40d, signing_key=args.signing_key)
    print(f"  25D model: in_dim={model_25d.cfg.in_dim}")
    print(f"  40D model: in_dim={model_40d.cfg.in_dim}")

    # Initialize vectorizers
    vectorizer_25d = FeatureVectorizer()
    vectorizer_40d = ContextAwareFeatureVectorizer()

    # Evaluate 25D (MemStream gốc)
    print("\n[4] Evaluating 25D (MemStream gốc)...")
    metrics_25d = evaluate_model(model_25d, vectorizer_25d, df_anom, labels, use_context=False)
    print(f"  F1:  {metrics_25d['f1']:.4f}")
    print(f"  FPR: {metrics_25d['fpr']:.4f} (FALSE ALARM RATE)")
    print(f"  AUC-PR: {metrics_25d['auc_pr']:.4f}")

    # Evaluate 40D (CA-MemStream)
    print("\n[5] Evaluating 40D (CA-MemStream)...")
    metrics_40d = evaluate_model(model_40d, vectorizer_40d, df_anom, labels, use_context=True)
    print(f"  F1:  {metrics_40d['f1']:.4f}")
    print(f"  FPR: {metrics_40d['fpr']:.4f} (FALSE ALARM RATE)")
    print(f"  AUC-PR: {metrics_40d['auc_pr']:.4f}")

    # Per-context analysis (False Alarm Analysis)
    print("\n[6] Per-context false alarm analysis...")
    per_ctx_25d = evaluate_per_context(model_25d, vectorizer_25d, df_anom, labels, use_context=False)
    per_ctx_40d = evaluate_per_context(model_40d, vectorizer_40d, df_anom, labels, use_context=True)

    # Compute improvement
    fpr_improvement = (metrics_25d['fpr'] - metrics_40d['fpr']) / metrics_25d['fpr'] * 100 if metrics_25d['fpr'] > 0 else 0
    f1_improvement = (metrics_40d['f1'] - metrics_25d['f1']) / metrics_25d['f1'] * 100 if metrics_25d['f1'] > 0 else 0

    print(f"\n  FALSE ALARM REDUCTION: {fpr_improvement:.1f}%")
    print(f"  F1 IMPROVEMENT: {f1_improvement:.1f}%")

    # Contexts with biggest improvement
    print("\n[7] Contexts with biggest false alarm reduction:")
    improvements = []
    for ctx_key in per_ctx_25d:
        if ctx_key in per_ctx_40d:
            fpr_25 = per_ctx_25d[ctx_key]['fpr']
            fpr_40 = per_ctx_40d[ctx_key]['fpr']
            if fpr_25 > 0:
                improvement = (fpr_25 - fpr_40) / fpr_25 * 100
                improvements.append({
                    'context': ctx_key,
                    'fpr_25d': fpr_25,
                    'fpr_40d': fpr_40,
                    'improvement_pct': improvement,
                })

    improvements.sort(key=lambda x: x['improvement_pct'], reverse=True)
    for imp in improvements[:5]:
        print(f"  {imp['context']}: {imp['fpr_25d']:.4f} -> {imp['fpr_40d']:.4f} ({imp['improvement_pct']:.1f}% reduction)")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_records': len(df_anom),
        'n_anomalies': int(np.sum(labels)),
        'metrics_25d': metrics_25d,
        'metrics_40d': metrics_40d,
        'improvements': {
            'fpr_reduction_pct': fpr_improvement,
            'f1_improvement_pct': f1_improvement,
        },
        'per_context_25d': per_ctx_25d,
        'per_context_40d': per_ctx_40d,
        'top_improvements': improvements[:10],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[8] Results saved to {output_path}")
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print(f"  CA-MemStream (40D) reduces FALSE ALARMS by {fpr_improvement:.1f}%")
    print(f"  CA-MemStream (40D) improves F1 by {f1_improvement:.1f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
