#!/usr/bin/env python3
"""
BAR Score Analysis: Measure Budget Allocation Rate in production.

Scientific purpose:
- Verify BAR Score meets 1-5% target
- Analyze drift detection patterns
- Measure label cost savings

Usage:
    python -m memstream_src.scripts.eval_bar_score \
        --logs /data/scoring_logs.csv \
        --output /results/bar_score_results.json
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


def analyze_bar_score(logs_df: pd.DataFrame) -> Dict:
    """
    Analyze BAR Score from scoring logs.

    Expected columns in logs:
    - timestamp: Event timestamp
    - neighborhood: Context neighborhood
    - score: Anomaly score
    - memory_update: Whether memory was updated (1=yes, 0=no)
    - update_reason: 'drift_detected', 'budget_granted', 'minimum_budget', 'no_budget'
    - drift_detected: Whether ADWIN detected drift (1=yes, 0=no)
    """
    total_records = len(logs_df)

    # BAR Score calculation
    memory_updates = logs_df['memory_update'].sum()
    bar_score = memory_updates / total_records if total_records > 0 else 0

    # Breakdown by reason
    reason_counts = logs_df['update_reason'].value_counts().to_dict()
    reason_pcts = {k: v / total_records * 100 for k, v in reason_counts.items()}

    # Drift detection analysis
    drift_events = logs_df['drift_detected'].sum() if 'drift_detected' in logs_df.columns else 0
    drift_rate = drift_events / total_records if total_records > 0 else 0

    # Per-neighborhood analysis
    neighborhood_stats = defaultdict(lambda: {
        'total': 0, 'updates': 0, 'drifts': 0, 'bar': 0
    })

    if 'neighborhood' in logs_df.columns:
        for _, row in logs_df.iterrows():
            nbr = row.get('neighborhood', 'unknown')
            neighborhood_stats[nbr]['total'] += 1
            neighborhood_stats[nbr]['updates'] += row['memory_update']

    for nbr, stats in neighborhood_stats.items():
        if stats['total'] > 0:
            stats['bar'] = stats['updates'] / stats['total']

    # Time series analysis (BAR over time)
    if 'timestamp' in logs_df.columns:
        logs_df['hour'] = pd.to_datetime(logs_df['timestamp']).dt.floor('H')
        hourly_bar = logs_df.groupby('hour')['memory_update'].mean()

        # Peak hours analysis
        peak_hours = hourly_bar.nlargest(5)
        low_hours = hourly_bar.nsmallest(5)
    else:
        hourly_bar = pd.Series()
        peak_hours = pd.Series()
        low_hours = pd.Series()

    return {
        'summary': {
            'total_records': int(total_records),
            'memory_updates': int(memory_updates),
            'drift_events': int(drift_events),
            'bar_score': float(bar_score),
            'bar_score_pct': float(bar_score * 100),
            'target_met': 0.01 <= bar_score <= 0.05,
            'target_range': '1-5%',
        },
        'update_reasons': {
            'counts': reason_counts,
            'percentages': reason_pcts,
        },
        'drift_analysis': {
            'drift_rate': float(drift_rate),
            'drift_rate_pct': float(drift_rate * 100),
        },
        'per_neighborhood': dict(neighborhood_stats),
        'time_analysis': {
            'hourly_bar': hourly_bar.to_dict() if not hourly_bar.empty else {},
            'peak_hours': peak_hours.to_dict() if not peak_hours.empty else {},
            'low_hours': low_hours.to_dict() if not low_hours.empty else {},
        },
    }


def estimate_cost_savings(bar_score: float, total_records: int) -> Dict:
    """
    Estimate cost savings from BAR Score vs 100% update.

    Original MemStream: 100% label cost (update on every record)
    CA-MemStream with BAR: bar_score% label cost
    """
    original_cost = total_records  # 100% of records
    actual_cost = int(total_records * bar_score)
    savings = original_cost - actual_cost
    savings_pct = savings / original_cost * 100 if original_cost > 0 else 0

    # Estimated cost (assuming $0.01 per label)
    cost_per_label = 0.01
    original_dollar_cost = original_cost * cost_per_label
    actual_dollar_cost = actual_cost * cost_per_label
    dollar_savings = original_dollar_cost - actual_dollar_cost

    return {
        'original_labels': original_cost,
        'actual_labels': actual_cost,
        'labels_saved': savings,
        'savings_pct': savings_pct,
        'original_cost_dollar': original_dollar_cost,
        'actual_cost_dollar': actual_dollar_cost,
        'dollar_savings': dollar_savings,
        'cost_per_label': cost_per_label,
    }


def main():
    parser = argparse.ArgumentParser(description='BAR Score Analysis')
    parser.add_argument('--logs', type=str, required=True, help='Scoring logs CSV path')
    parser.add_argument('--output', type=str, required=True, help='Output JSON path')
    args = parser.parse_args()

    print("=" * 60)
    print("BAR SCORE ANALYSIS")
    print("=" * 60)

    # Load logs
    print(f"\n[1] Loading scoring logs from {args.logs}...")
    logs_df = pd.read_csv(args.logs)
    print(f"  Loaded {len(logs_df):,} records")

    # Analyze BAR Score
    print("\n[2] Analyzing BAR Score...")
    results = analyze_bar_score(logs_df)

    bar_score = results['summary']['bar_score']
    bar_score_pct = results['summary']['bar_score_pct']
    target_met = results['summary']['target_met']

    print(f"\n  BAR SCORE: {bar_score_pct:.2f}%")
    print(f"  TARGET: 1-5%")
    print(f"  TARGET MET: {'YES' if target_met else 'NO'}")
    print(f"\n  Total records: {results['summary']['total_records']:,}")
    print(f"  Memory updates: {results['summary']['memory_updates']:,}")
    print(f"  Drift events: {results['summary']['drift_events']:,}")

    # Update reasons breakdown
    print("\n[3] Update reason breakdown:")
    for reason, pct in results['update_reasons']['percentages'].items():
        print(f"  {reason}: {pct:.2f}%")

    # Cost savings
    print("\n[4] Cost savings estimation:")
    cost_savings = estimate_cost_savings(bar_score, results['summary']['total_records'])
    print(f"  Original labels: {cost_savings['original_labels']:,}")
    print(f"  Actual labels: {cost_savings['actual_labels']:,}")
    print(f"  Labels saved: {cost_savings['labels_saved']:,} ({cost_savings['savings_pct']:.1f}%)")
    print(f"  Dollar savings: ${cost_savings['dollar_savings']:,.2f}")

    # Per-neighborhood analysis
    print("\n[5] Per-neighborhood BAR Score:")
    for nbr, stats in results['per_neighborhood'].items():
        bar = stats['bar'] * 100
        in_range = 'OK' if 1 <= bar <= 5 else 'WARN'
        print(f"  {nbr}: {bar:.2f}% [{in_range}]")

    # Peak hours
    if results['time_analysis']['peak_hours']:
        print("\n[6] Peak BAR hours (highest memory update activity):")
        for hour, bar in list(results['time_analysis']['peak_hours'].items())[:3]:
            print(f"  {hour}: {bar*100:.2f}%")

    # Save results
    results['cost_savings'] = cost_savings

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[7] Results saved to {output_path}")
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    if target_met:
        print(f"  BAR Score {bar_score_pct:.2f}% meets 1-5% target")
        print(f"  Label cost reduced by {cost_savings['savings_pct']:.1f}%")
    else:
        print(f"  BAR Score {bar_score_pct:.2f}% outside 1-5% target")
        if bar_score < 0.01:
            print(f"  BAR too low - may cause underfitting")
        else:
            print(f"  BAR too high - increase ADWIN sensitivity")
    print("=" * 60)


if __name__ == '__main__':
    main()
