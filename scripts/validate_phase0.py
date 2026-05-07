"""
Validate Phase 0 success criteria (Go/No-Go gate).
Spec: Lines 2912-2916
"""

import pandas as pd
from pathlib import Path
import json
import pickle
import sys

def validate_phase0():
    """Check all Phase 0 deliverables."""

    print("="*60)
    print("PHASE 0 VALIDATION")
    print("="*60)

    passed = []
    failed = []

    # Check 1: Clean baseline exists
    print("\n1. Clean baseline:")
    path = Path('data/clean/jan_2024_clean_baseline.parquet')
    if path.exists():
        df = pd.read_parquet(path)
        null_rate = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100

        violations = (
            (df['fare_amount'] <= 0) |
            (df['trip_distance'] <= 0) |
            (df['passenger_count'] > 6)
        )
        viol_rate = violations.sum() / len(df) * 100

        print(f"   Records: {len(df):,}")
        print(f"   Null rate: {null_rate:.3f}%")
        print(f"   Violation rate: {viol_rate:.3f}%")

        if null_rate < 0.5 and viol_rate < 0.5:
            print("   ✅ PASS")
            passed.append("Clean baseline")
        else:
            print("   ❌ FAIL: Metrics exceed thresholds")
            failed.append("Clean baseline metrics")
    else:
        print("   ❌ FAIL: File not found")
        failed.append("Clean baseline file")

    # Check 2: Synthetic anomalies
    print("\n2. Synthetic anomalies:")
    anom_path = Path('data/clean/jan_2024_with_50k_anomalies.parquet')
    labels_path = Path('data/clean/anomaly_labels.csv')

    if anom_path.exists() and labels_path.exists():
        labels = pd.read_csv(labels_path)
        anom_count = (labels['is_anomaly'] == 1).sum()
        scenarios = labels[labels['is_anomaly'] == 1]['scenario'].nunique()

        print(f"   Anomalies: {anom_count:,}")
        print(f"   Scenarios: {scenarios}")

        if anom_count == 50_000 and scenarios == 5:
            print("   ✅ PASS")
            passed.append("Synthetic anomalies")
        else:
            print("   ❌ FAIL: Count/scenarios incorrect")
            failed.append("Synthetic anomalies")
    else:
        print("   ❌ FAIL: Files not found")
        failed.append("Synthetic anomaly files")

    # Check 3: Neighborhood mapping
    print("\n3. Neighborhood mapping:")
    map_path = Path('src/config/neighborhood_mapping.json')
    if map_path.exists():
        with open(map_path) as f:
            mapping = json.load(f)

        zones = len(mapping['mapping'])
        neighbors = len(set(mapping['mapping'].values()))

        print(f"   Zones mapped: {zones}")
        print(f"   Neighborhoods: {neighbors}")

        if zones == 265 and 5 <= neighbors <= 8:
            print("   ✅ PASS")
            passed.append("Neighborhood mapping")
        else:
            print("   ❌ FAIL: Count mismatch")
            failed.append("Neighborhood mapping")
    else:
        print("   ❌ FAIL: File not found")
        failed.append("Neighborhood mapping file")

    # Check 4: StandardScaler
    print("\n4. StandardScaler:")
    scaler_path = Path('models/scaler.pkl')
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        if hasattr(scaler, 'mean_') and len(scaler.mean_) == 15:
            print(f"   Feature dim: {len(scaler.mean_)}D")
            print("   ✅ PASS")
            passed.append("StandardScaler")
        else:
            print("   ❌ FAIL: Not fitted or wrong dimension")
            failed.append("StandardScaler")
    else:
        print("   ❌ FAIL: File not found")
        failed.append("StandardScaler file")

    # Check 5: Threshold matrix
    print("\n5. Threshold matrix:")
    thresh_path = Path('src/config/threshold_matrix.json')
    if thresh_path.exists():
        with open(thresh_path) as f:
            thresholds = json.load(f)

        count = len(thresholds['thresholds'])
        print(f"   Contexts: {count}")

        if count > 50:  # At least 50 contexts
            print("   ✅ PASS")
            passed.append("Threshold matrix")
        else:
            print("   ❌ FAIL: Too few contexts")
            failed.append("Threshold matrix")
    else:
        print("   ❌ FAIL: File not found")
        failed.append("Threshold matrix file")

    # Summary
    print("\n" + "="*60)
    print(f"PASSED: {len(passed)}/5")
    print(f"FAILED: {len(failed)}/5")
    print("="*60)

    if failed:
        print("\n❌ PHASE 0 INCOMPLETE")
        print("Failed checks:")
        for item in failed:
            print(f"  - {item}")
        sys.exit(1)
    else:
        print("\n✅ PHASE 0 COMPLETE - Ready for Phase 1")
        sys.exit(0)

if __name__ == "__main__":
    validate_phase0()
