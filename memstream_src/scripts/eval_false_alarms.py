#!/usr/bin/env python3
"""
False Alarm Analysis: Compare MemStream gốc vs CA-MemStream.

Scientific purpose:
- Measure false positive rate (FPR) per context
- Identify where context-awareness helps most
- Analyze rush hour patterns

Usage:
    python -m memstream_src.scripts.eval_false_alarms \
        --data /data/test.csv \
        --model-25d /models/memstream_25d.pt \
        --model-40d /models/memstream_40d.pt \
        --output /results/false_alarm_results.json
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memstream_core import MemStreamCore
from core.feature_extractor import FeatureVectorizer
from core.context_aware import ContextAwareFeatureVectorizer, get_4d_context
from scripts.inject_anomalies_multi import inject_anomalies


def compute_context_metrics(
    df: pd.DataFrame,
    labels: np.ndarray,
    model: MemStreamCore,
    vectorizer,
    use_context: bool,
) -> Dict:
    """
    Compute per-context false alarm metrics.

    Returns metrics broken down by:
    - neighborhood
    - hour_bucket
    - day_type
    - Combined 4D context
    """
    results = {
        'by_neighborhood': defaultdict(lambda: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}),
        'by_hour_bucket': defaultdict(lambda: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}),
        'by_day_type': defaultdict(lambda: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}),
        'by_4d_context': defaultdict(lambda: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'samples': 0}),
    }

    threshold = model.max_thres.item()

    for idx, row in df.iterrows():
        label = labels[idx]
        record = row.to_dict()

        if use_context:
            ctx = get_4d_context(record)
            features = vectorizer.transform(record, ctx)
        else:
            ctx = get_4d_context(record)
            features = vectorizer.transform(record)

        score = model.score_one(features)
        pred = 1 if score > threshold else 0

        # Aggregate by context
        nbr = ctx.get('neighborhood', 'unknown')
        hour = ctx.get('hour_bucket', 'unknown')
        day = ctx.get('day_type', 'unknown')
        ctx_key = f'{nbr}_{hour}_{day}'

        if pred == 1 and label == 1:
            results['by_neighborhood'][nbr]['tp'] += 1
            results['by_hour_bucket'][hour]['tp'] += 1
            results['by_day_type'][day]['tp'] += 1
            results['by_4d_context'][ctx_key]['tp'] += 1
        elif pred == 1 and label == 0:
            results['by_neighborhood'][nbr]['fp'] += 1
            results['by_hour_bucket'][hour]['fp'] += 1
            results['by_day_type'][day]['fp'] += 1
            results['by_4d_context'][ctx_key]['fp'] += 1
        elif pred == 0 and label == 0:
            results['by_neighborhood'][nbr]['tn'] += 1
            results['by_hour_bucket'][hour]['tn'] += 1
            results['by_day_type'][day]['tn'] += 1
            results['by_4d_context'][ctx_key]['tn'] += 1
        else:  # pred == 0 and label == 1
            results['by_neighborhood'][nbr]['fn'] += 1
            results['by_hour_bucket'][hour]['fn'] += 1
            results['by_day_type'][day]['fn'] += 1
            results['by_4d_context'][ctx_key]['fn'] += 1

        results['by_4d_context'][ctx_key]['samples'] += 1

    return results


def compute_fpr(cm: Dict) -> float:
    """Compute False Positive Rate from confusion matrix."""
    fp = cm['fp']
    tn = cm['tn']
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='False Alarm Analysis')
    parser.add_argument('--data', type=str, required=True, help='Test CSV path')
    parser.add_argument('--model-25d', type=str, required=True, help='25D model path')
    parser.add_argument('--model-40d', type=str, required=True, help='40D model path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path')
    parser.add_argument('--n-anomalies', type=int, default=1500, help='Number of anomalies')
    parser.add_argument('--signing-key', type=str, default='training-signing-key',
                        help='HMAC signing key')
    args = parser.parse_args()

    print("=" * 60)
    print("FALSE ALARM ANALYSIS: MemStream gốc vs CA-MemStream")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data...")
    df = pd.read_csv(args.data)

    # Inject anomalies (focus on FALSE ALARMS = predicting anomaly when normal)
    print("\n[2] Injecting anomalies...")
    df_anom, labels = inject_anomalies(df, n_anomalies=args.n_anomalies)

    # Load models
    print("\n[3] Loading models...")
    model_25d = MemStreamCore.load(args.model_25d, signing_key=args.signing_key)
    model_40d = MemStreamCore.load(args.model_40d, signing_key=args.signing_key)

    vectorizer_25d = FeatureVectorizer()
    vectorizer_40d = ContextAwareFeatureVectorizer()

    # Compute metrics for both models
    print("\n[4] Computing per-context metrics for 25D...")
    metrics_25d = compute_context_metrics(
        df_anom, labels, model_25d, vectorizer_25d, use_context=False
    )

    print("[5] Computing per-context metrics for 40D...")
    metrics_40d = compute_context_metrics(
        df_anom, labels, model_40d, vectorizer_40d, use_context=True
    )

    # Analyze by hour bucket (where rush hour false alarms occur)
    print("\n[6] FALSE ALARM ANALYSIS BY HOUR BUCKET:")
    print("-" * 60)
    print(f"{'Hour Bucket':<20} {'25D FPR':<12} {'40D FPR':<12} {'Reduction':<12}")
    print("-" * 60)

    hour_improvements = []
    for hour in ['morning_rush', 'midday', 'evening_rush', 'night']:
        cm_25 = metrics_25d['by_hour_bucket'].get(hour, {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})
        cm_40 = metrics_40d['by_hour_bucket'].get(hour, {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})

        fpr_25 = compute_fpr(cm_25)
        fpr_40 = compute_fpr(cm_40)
        reduction = (fpr_25 - fpr_40) / fpr_25 * 100 if fpr_25 > 0 else 0

        hour_improvements.append({
            'hour_bucket': hour,
            'fpr_25d': fpr_25,
            'fpr_40d': fpr_40,
            'reduction_pct': reduction,
        })

        print(f"{hour:<20} {fpr_25:.4f}       {fpr_40:.4f}       {reduction:+.1f}%")

    # Analyze by neighborhood
    print("\n[7] FALSE ALARM ANALYSIS BY NEIGHBORHOOD:")
    print("-" * 60)
    print(f"{'Neighborhood':<20} {'25D FPR':<12} {'40D FPR':<12} {'Reduction':<12}")
    print("-" * 60)

    nbr_improvements = []
    for nbr in sorted(metrics_25d['by_neighborhood'].keys()):
        cm_25 = metrics_25d['by_neighborhood'][nbr]
        cm_40 = metrics_40d['by_neighborhood'].get(nbr, {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})

        fpr_25 = compute_fpr(cm_25)
        fpr_40 = compute_fpr(cm_40)
        reduction = (fpr_25 - fpr_40) / fpr_25 * 100 if fpr_25 > 0 else 0

        nbr_improvements.append({
            'neighborhood': nbr,
            'fpr_25d': fpr_25,
            'fpr_40d': fpr_40,
            'reduction_pct': reduction,
        })

        print(f"{nbr:<20} {fpr_25:.4f}       {fpr_40:.4f}       {reduction:+.1f}%")

    # Overall comparison
    total_cm_25 = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    total_cm_40 = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

    for cm in metrics_25d['by_hour_bucket'].values():
        for k, v in cm.items():
            total_cm_25[k] += v
    for cm in metrics_40d['by_hour_bucket'].values():
        for k, v in cm.items():
            total_cm_40[k] += v

    overall_fpr_25 = compute_fpr(total_cm_25)
    overall_fpr_40 = compute_fpr(total_cm_40)
    overall_reduction = (overall_fpr_25 - overall_fpr_40) / overall_fpr_25 * 100 if overall_fpr_25 > 0 else 0

    print("-" * 60)
    print(f"{'OVERALL':<20} {overall_fpr_25:.4f}       {overall_fpr_40:.4f}       {overall_reduction:+.1f}%")

    # Evening rush hour analysis
    evening_rush_idx = next((i for i, h in enumerate(hour_improvements) if h['hour_bucket'] == 'evening_rush'), 2)

    print("\n" + "=" * 60)
    print("SCIENTIFIC CONCLUSION:")
    print(f"  Overall False Alarm Reduction: {overall_reduction:.1f}%")
    print(f"  Evening Rush Hour FPR (25D): {hour_improvements[evening_rush_idx]['fpr_25d']:.4f}")
    print(f"  Evening Rush Hour FPR (40D): {hour_improvements[evening_rush_idx]['fpr_40d']:.4f}")
    print(f"  Evening Rush Hour Improvement: {hour_improvements[evening_rush_idx]['reduction_pct']:.1f}%")
    print("=" * 60)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'overall': {
            'fpr_25d': overall_fpr_25,
            'fpr_40d': overall_fpr_40,
            'reduction_pct': overall_reduction,
        },
        'by_hour_bucket': hour_improvements,
        'by_neighborhood': nbr_improvements,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[8] Results saved to {output_path}")


if __name__ == '__main__':
    main()
