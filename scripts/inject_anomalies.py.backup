"""
Inject 50K synthetic anomalies (5 scenarios × 10K each).
Spec: Lines 2884-2894
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def generate_synthetic_distribution_figure(labels):
    """Generate Phase 0 Figure 4: Synthetic Anomaly Distribution."""
    plt.figure(figsize=(10, 6))
    scenario_counts = labels[labels['is_anomaly'] == 1]['scenario'].value_counts()
    scenario_counts.plot(kind='bar', color='coral', edgecolor='black')
    plt.title('Phase 0: Synthetic Anomaly Distribution (5 Scenarios)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Fraud Scenario')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=10000, color='red', linestyle='--', label='Target (10K each)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/figures/phase0_synthetic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: phase0_synthetic_distribution.png")

def main():
    # Load clean baseline
    input_path = Path('data/clean/jan_2024_clean_baseline.parquet')
    df = pd.read_parquet(input_path)
    original_len = len(df)
    print(f"Clean baseline: {original_len:,} records")

    # Sample 50K unique indices for all scenarios
    np.random.seed(42)
    all_anomaly_indices = np.random.choice(df.index, 50_000, replace=False)

    # Split into 5 groups of 10K each
    scenarios = {
        'meter_tampering': all_anomaly_indices[0:10_000],
        'short_trip_fraud': all_anomaly_indices[10_000:20_000],
        'gps_spoofing': all_anomaly_indices[20_000:30_000],
        'time_manipulation': all_anomaly_indices[30_000:40_000],
        'passenger_fraud': all_anomaly_indices[40_000:50_000]
    }

    # Initialize labels
    labels = pd.DataFrame({
        'index': df.index,
        'is_anomaly': 0,
        'scenario': 'normal'
    })

    # Scenario 1: Meter tampering (fare × 3)
    indices = scenarios['meter_tampering']
    df.loc[indices, 'fare_amount'] *= 3
    df.loc[indices, 'total_amount'] *= 3
    labels.loc[labels['index'].isin(indices), 'is_anomaly'] = 1
    labels.loc[labels['index'].isin(indices), 'scenario'] = 'meter_tampering'
    print(f"✓ Injected: meter_tampering (10K)")

    # Scenario 2: Short trip fraud
    indices = scenarios['short_trip_fraud']
    df.loc[indices, 'trip_distance'] = 0.1
    df.loc[indices, 'fare_amount'] = 50.0
    df.loc[indices, 'total_amount'] = 50.0
    labels.loc[labels['index'].isin(indices), 'is_anomaly'] = 1
    labels.loc[labels['index'].isin(indices), 'scenario'] = 'short_trip_fraud'
    print(f"✓ Injected: short_trip_fraud (10K)")

    # Scenario 3: GPS spoofing
    indices = scenarios['gps_spoofing']
    df.loc[indices, 'PULocationID'] = np.random.randint(1, 264, 10_000)
    df.loc[indices, 'DOLocationID'] = np.random.randint(1, 264, 10_000)
    labels.loc[labels['index'].isin(indices), 'is_anomaly'] = 1
    labels.loc[labels['index'].isin(indices), 'scenario'] = 'gps_spoofing'
    print(f"✓ Injected: gps_spoofing (10K)")

    # Scenario 4: Time manipulation
    indices = scenarios['time_manipulation']
    shifts = np.random.choice([-5, 5], 10_000) * 3600
    df.loc[indices, 'tpep_pickup_datetime'] = (
        pd.to_datetime(df.loc[indices, 'tpep_pickup_datetime']) +
        pd.to_timedelta(shifts, unit='s')
    ).astype(str)
    labels.loc[labels['index'].isin(indices), 'is_anomaly'] = 1
    labels.loc[labels['index'].isin(indices), 'scenario'] = 'time_manipulation'
    print(f"✓ Injected: time_manipulation (10K)")

    # Scenario 5: Passenger fraud
    indices = scenarios['passenger_fraud']
    df.loc[indices, 'passenger_count'] = 15
    labels.loc[labels['index'].isin(indices), 'is_anomaly'] = 1
    labels.loc[labels['index'].isin(indices), 'scenario'] = 'passenger_fraud'
    print(f"✓ Injected: passenger_fraud (10K)")

    # Save injected data
    output_path = Path('data/clean/jan_2024_with_50k_anomalies.parquet')
    df.to_parquet(output_path, index=False)

    labels_path = Path('data/clean/anomaly_labels.csv')
    labels.to_csv(labels_path, index=False)

    anomaly_count = (labels['is_anomaly'] == 1).sum()
    print(f"\n✅ Saved: {len(df):,} records")
    print(f"   Anomalies: {anomaly_count:,}")
    print(f"   Output: {output_path}")
    print(f"   Labels: {labels_path}")

    # Generate Figure 4
    generate_synthetic_distribution_figure(labels)

if __name__ == "__main__":
    main()
