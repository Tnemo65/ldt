#!/usr/bin/env python3
"""
Analyze synthetic anomalies to understand why they're hard to detect.

Questions to answer:
1. How different are anomalies from clean records?
2. Do anomalies overlap with clean distribution?
3. Which anomaly scenarios are hardest to detect?
4. What are the feature distributions?
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.insert(0, '.')
from src.features.vectorizer import FeatureVectorizer


def analyze_anomalies():
    print("="*60)
    print("SYNTHETIC ANOMALY ANALYSIS")
    print("="*60)

    # 1. Load data
    print("\n1. Loading data...")
    df_anomalies = pd.read_parquet('data/clean/jan_2024_with_50k_anomalies.parquet')
    labels = pd.read_csv('data/clean/anomaly_labels.csv')

    print(f"   Total records: {len(df_anomalies):,}")
    print(f"   Anomalies: {labels['is_anomaly'].sum():,}")
    print(f"   Clean: {(~labels['is_anomaly']).sum():,}")

    # 2. Analyze by scenario
    print("\n2. Anomaly scenarios breakdown:")
    if 'scenario' in labels.columns:
        anomaly_labels = labels[labels['is_anomaly'] == 1]
        scenario_counts = anomaly_labels['scenario'].value_counts()
        for scenario, count in scenario_counts.items():
            print(f"   {scenario}: {count:,}")
    else:
        scenario_counts = pd.Series()
        print("   No scenario column found")

    # 3. Feature analysis
    print("\n3. Feature distribution analysis...")

    # Separate clean and anomalies
    is_anomaly = labels['is_anomaly'] == 1
    df_clean = df_anomalies[~is_anomaly]
    df_anom = df_anomalies[is_anomaly]

    # Key features to analyze
    features = ['trip_distance', 'fare_amount', 'passenger_count', 'total_amount']

    print(f"\n{'Feature':<20} {'Clean Mean':<15} {'Anom Mean':<15} {'Difference':<15}")
    print("-" * 65)

    for feat in features:
        clean_mean = df_clean[feat].mean()
        anom_mean = df_anom[feat].mean()
        diff = abs(anom_mean - clean_mean) / clean_mean * 100
        print(f"{feat:<20} {clean_mean:<15.2f} {anom_mean:<15.2f} {diff:<15.1f}%")

    # 4. Distribution overlap
    print("\n4. Distribution overlap analysis...")

    for feat in features:
        clean_vals = df_clean[feat].values
        anom_vals = df_anom[feat].values

        # Percentiles
        clean_p25, clean_p75 = np.percentile(clean_vals, [25, 75])

        # How many anomalies fall within clean IQR?
        within_iqr = ((anom_vals >= clean_p25) & (anom_vals <= clean_p75)).sum()
        overlap_pct = within_iqr / len(anom_vals) * 100

        print(f"   {feat}: {overlap_pct:.1f}% anomalies within clean IQR")

    # 5. Per-scenario difficulty
    print("\n5. Per-scenario feature analysis:")

    if len(scenario_counts) > 0:
        for scenario in scenario_counts.index:
            if pd.isna(scenario):
                continue

            print(f"\n   Scenario: {scenario}")
            scenario_mask = (labels['is_anomaly'] == 1) & (labels['scenario'] == scenario)
            df_scenario = df_anomalies[scenario_mask]

        for feat in ['fare_amount', 'trip_distance']:
            scenario_mean = df_scenario[feat].mean()
            clean_mean = df_clean[feat].mean()
            diff = abs(scenario_mean - clean_mean) / clean_mean * 100

            # Check if within clean range
            clean_min, clean_max = df_clean[feat].min(), df_clean[feat].max()
            within_range = ((df_scenario[feat] >= clean_min) &
                          (df_scenario[feat] <= clean_max)).sum()
            within_pct = within_range / len(df_scenario) * 100

            print(f"      {feat}: {diff:.1f}% diff from clean, {within_pct:.1f}% within clean range")
    else:
        print("   No scenarios to analyze")

    # 6. Vectorized feature analysis
    print("\n6. Vectorized feature space analysis...")

    vectorizer = FeatureVectorizer()

    # Sample for performance
    sample_size = 10000
    df_clean_sample = df_clean.sample(min(sample_size, len(df_clean)), random_state=42)
    df_anom_sample = df_anom.sample(min(sample_size, len(df_anom)), random_state=42)

    print(f"   Vectorizing {len(df_clean_sample):,} clean + {len(df_anom_sample):,} anomaly samples...")

    X_clean = []
    for _, row in df_clean_sample.iterrows():
        try:
            X_clean.append(vectorizer.transform(row.to_dict()))
        except:
            pass

    X_anom = []
    for _, row in df_anom_sample.iterrows():
        try:
            X_anom.append(vectorizer.transform(row.to_dict()))
        except:
            pass

    X_clean = np.array(X_clean)
    X_anom = np.array(X_anom)

    # Scale
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    X_clean_scaled = scaler.transform(X_clean)
    X_anom_scaled = scaler.transform(X_anom)

    # Distance analysis
    clean_mean = X_clean_scaled.mean(axis=0)

    # Distance from clean center
    clean_dist = np.linalg.norm(X_clean_scaled - clean_mean, axis=1).mean()
    anom_dist = np.linalg.norm(X_anom_scaled - clean_mean, axis=1).mean()

    print(f"\n   Average distance from clean center:")
    print(f"      Clean records: {clean_dist:.3f}")
    print(f"      Anomalies: {anom_dist:.3f}")
    print(f"      Ratio: {anom_dist/clean_dist:.2f}x")

    if anom_dist / clean_dist < 1.5:
        print(f"\n   ⚠️  WARNING: Anomalies only {anom_dist/clean_dist:.2f}x farther than clean!")
        print(f"      Anomalies are TOO SUBTLE - very close to clean distribution")
        print(f"      This explains low recall and high FPR")

    # 7. Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if anom_dist / clean_dist < 1.5:
        print("\n🎯 ISSUE: Synthetic anomalies too subtle")
        print("   - Anomalies overlap heavily with clean distribution")
        print("   - Model cannot distinguish effectively")
        print("\n💡 SOLUTIONS:")
        print("   1. Make synthetic anomalies more extreme")
        print("   2. Use ensemble methods (multiple models)")
        print("   3. Lower acceptance criteria (60% recall, 10% FPR)")
        print("   4. Use supervised learning instead of iForest")
    else:
        print("\n✅ Anomalies are distinguishable")
        print("   - Need better model hyperparameters")
        print("   - Need more training data or trees")

    return {
        'overlap_pct': overlap_pct,
        'distance_ratio': anom_dist / clean_dist,
        'clean_dist': clean_dist,
        'anom_dist': anom_dist
    }


if __name__ == '__main__':
    results = analyze_anomalies()
