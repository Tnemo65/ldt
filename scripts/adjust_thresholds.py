#!/usr/bin/env python3
"""
Adjust detection thresholds to reduce FPR while maintaining recall.
Analysis shows FPR=26.3% is too high - need to increase thresholds.

Strategy:
1. Analyze current threshold distribution
2. Increase thresholds by percentile shift (95th -> 97th or 98th)
3. Validate new thresholds on synthetic data
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

def analyze_thresholds(threshold_path: str):
    """Analyze current threshold distribution."""
    with open(threshold_path) as f:
        data = json.load(f)

    thresholds = list(data['thresholds'].values())

    print("="*60)
    print("CURRENT THRESHOLD ANALYSIS")
    print("="*60)
    print(f"\nGlobal threshold: {data['global_threshold']:.4f}")
    print(f"Percentile used: {data['percentile']}th")
    print(f"\nContext-specific thresholds ({len(thresholds)}):")
    print(f"  Min:    {np.min(thresholds):.4f}")
    print(f"  Q1:     {np.percentile(thresholds, 25):.4f}")
    print(f"  Median: {np.median(thresholds):.4f}")
    print(f"  Q3:     {np.percentile(thresholds, 75):.4f}")
    print(f"  Max:    {np.max(thresholds):.4f}")
    print(f"  Mean:   {np.mean(thresholds):.4f}")
    print(f"  Std:    {np.std(thresholds):.4f}")

    return data

def adjust_thresholds(data: dict, new_percentile: int = 98):
    """Increase thresholds to reduce FPR.

    Current: 95th percentile -> FPR=26.3%
    Strategy: Use 98th percentile -> expect FPR ~5-10%

    Args:
        data: Current threshold data
        new_percentile: New percentile to use (default: 98)

    Returns:
        Adjusted threshold data
    """
    print(f"\n{'='*60}")
    print(f"ADJUSTING THRESHOLDS: {data['percentile']}th -> {new_percentile}th")
    print("="*60)

    # Calculate scaling factor
    # If thresholds are at 95th percentile, to get 98th we need to shift up
    # Approximate: scale by (100-new)/(100-old) in the tail
    old_p = data['percentile']
    scale_factor = (100 - old_p) / (100 - new_percentile)

    print(f"\nApproach: Shift thresholds higher")
    print(f"  Old percentile: {old_p}th")
    print(f"  New percentile: {new_percentile}th")
    print(f"  Approximate scale: {scale_factor:.3f}")

    # Adjust thresholds
    old_global = data['global_threshold']
    new_global = old_global + (1.0 - old_global) * (new_percentile - old_p) / (100 - old_p)

    adjusted_data = {
        'version': '2.0',
        'percentile': new_percentile,
        'old_percentile': old_p,
        'global_threshold': new_global,
        'old_global_threshold': old_global,
        'thresholds': {}
    }

    for context_key, old_threshold in data['thresholds'].items():
        # Shift each threshold proportionally
        new_threshold = old_threshold + (1.0 - old_threshold) * (new_percentile - old_p) / (100 - old_p)
        adjusted_data['thresholds'][context_key] = new_threshold

    new_thresholds = list(adjusted_data['thresholds'].values())

    print(f"\nAdjusted thresholds:")
    print(f"  Global: {old_global:.4f} -> {new_global:.4f} (+{new_global-old_global:.4f})")
    print(f"  Min:    {np.min(new_thresholds):.4f}")
    print(f"  Median: {np.median(new_thresholds):.4f}")
    print(f"  Max:    {np.max(new_thresholds):.4f}")
    print(f"  Mean:   {np.mean(new_thresholds):.4f} (+{np.mean(new_thresholds)-np.mean(list(data['thresholds'].values())):.4f})")

    return adjusted_data

def main():
    threshold_path = 'models/context_thresholds.json'

    print("\n" + "="*60)
    print("THRESHOLD ADJUSTMENT ANALYSIS")
    print("="*60)
    print("\nProblem: FPR=26.3% (target: <5%)")
    print("Root cause: Thresholds too low (95th percentile)")
    print("Solution: Increase to 98th percentile")

    # 1. Analyze current
    current_data = analyze_thresholds(threshold_path)

    # 2. Adjust thresholds
    adjusted_data = adjust_thresholds(current_data, new_percentile=98)

    # 3. Save adjusted thresholds
    output_path = 'models/context_thresholds_v2.json'
    with open(output_path, 'w') as f:
        json.dump(adjusted_data, f, indent=2)

    print(f"\n{'='*60}")
    print("✅ ADJUSTED THRESHOLDS SAVED")
    print("="*60)
    print(f"Output: {output_path}")
    print(f"\nTo test:")
    print(f"  python scripts/validate_model_synthetic.py \\")
    print(f"    --thresholds {output_path}")
    print(f"\nExpected impact:")
    print(f"  FPR: 26.3% -> ~5-10% (target: <5%)")
    print(f"  Recall: 47.2% -> ~45-50% (may decrease slightly)")
    print(f"\nNext: Combine with v2 model (n_trees=200) for best results")

    return 0

if __name__ == '__main__':
    exit(main())
