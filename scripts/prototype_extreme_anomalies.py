#!/usr/bin/env python3
"""
Prototype Extreme Synthetic Anomalies
Generate contextual impossibilities instead of simple multipliers.

Key improvements:
1. Impossible combinations (not just extreme values)
2. Space-time violations
3. Business logic violations
4. Target: <15% overlap with clean data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import timedelta

sys.path.insert(0, '.')


class ExtremeSyntheticGenerator:
    """Generate extreme contextual anomalies."""

    AIRPORT_ZONES = {132, 138, 1}  # JFK, LaGuardia, Newark

    def __init__(self, seed=42):
        np.random.seed(seed)
        self.scenarios = {
            'meter_tampering_extreme': self.meter_tampering_extreme,
            'gps_spoofing_impossible': self.gps_spoofing_impossible,
            'passenger_fraud_impossible': self.passenger_fraud_impossible,
            'time_manipulation_extreme': self.time_manipulation_extreme,
            'combined_impossibility': self.combined_impossibility,
        }

    def meter_tampering_extreme(self, record):
        """
        Extreme meter tampering: Short distance + normal time + HUGE fare.

        Strategy: fare_per_mile = 10-30x normal ($2.50 → $25-75/mile)
        """
        # Keep short distance and reasonable time
        record['trip_distance'] = np.random.uniform(1.0, 3.0)  # 1-3 miles
        duration_minutes = np.random.uniform(5, 15)  # 5-15 minutes

        # Calculate duration
        pickup_time = record['tpep_pickup_datetime']
        record['tpep_dropoff_datetime'] = pickup_time + timedelta(minutes=duration_minutes)

        # EXTREME fare: charge for 15-30 mile trip!
        normal_fare_per_mile = 2.50
        extreme_multiplier = np.random.uniform(10, 30)  # 10-30x
        record['fare_amount'] = record['trip_distance'] * normal_fare_per_mile * extreme_multiplier

        # Total = fare + extras
        extras = np.random.uniform(5, 10)
        record['total_amount'] = record['fare_amount'] + extras

        # Ratio features (for detection):
        # fare_per_mile = $25-75 (vs normal $2.50)
        # implied_speed = 12-36 mph (normal, not suspicious)

        return record

    def gps_spoofing_impossible(self, record):
        """
        GPS spoofing: Huge distance + impossible short time.

        Strategy: implied_speed = 150-300 mph (physically impossible in NYC)
        """
        # HUGE distance
        record['trip_distance'] = np.random.uniform(50, 100)  # 50-100 miles

        # VERY short time
        duration_minutes = np.random.uniform(10, 20)  # 10-20 minutes
        pickup_time = record['tpep_pickup_datetime']
        record['tpep_dropoff_datetime'] = pickup_time + timedelta(minutes=duration_minutes)

        # Fare: reasonable per mile (not suspicious on its own)
        fare_per_mile = np.random.uniform(2.0, 3.5)
        record['fare_amount'] = record['trip_distance'] * fare_per_mile

        extras = np.random.uniform(5, 15)
        record['total_amount'] = record['fare_amount'] + extras

        # Ratio features (for detection):
        # implied_speed = 150-300 mph (IMPOSSIBLE!)
        # fare_per_mile = $2-3.5 (normal)

        return record

    def passenger_fraud_impossible(self, record):
        """
        Passenger fraud: IMPOSSIBLE passenger count.

        Strategy: passengers = 15-30 (NYC taxi max = 5-6 realistically)
        """
        # Normal distance and time
        record['trip_distance'] = np.random.uniform(3, 10)
        duration_minutes = np.random.uniform(15, 40)

        pickup_time = record['tpep_pickup_datetime']
        record['tpep_dropoff_datetime'] = pickup_time + timedelta(minutes=duration_minutes)

        # IMPOSSIBLE passenger count
        record['passenger_count'] = np.random.randint(15, 31)  # 15-30 people!

        # Normal fare
        fare_per_mile = np.random.uniform(2.0, 3.0)
        record['fare_amount'] = record['trip_distance'] * fare_per_mile

        extras = np.random.uniform(3, 8)
        record['total_amount'] = record['fare_amount'] + extras

        # Ratio features (for detection):
        # passenger_count = 15-30 (IMPOSSIBLE - taxi max ~6)

        return record

    def time_manipulation_extreme(self, record):
        """
        Time manipulation: Long distance + ZERO duration.

        Strategy: duration < 1 minute for 10+ mile trip
        """
        # Long distance
        record['trip_distance'] = np.random.uniform(10, 30)

        # ZERO duration (or <1 min)
        duration_seconds = np.random.uniform(1, 30)  # 1-30 seconds!
        pickup_time = record['tpep_pickup_datetime']
        record['tpep_dropoff_datetime'] = pickup_time + timedelta(seconds=duration_seconds)

        # Normal fare per mile
        fare_per_mile = np.random.uniform(2.0, 3.0)
        record['fare_amount'] = record['trip_distance'] * fare_per_mile

        extras = np.random.uniform(3, 8)
        record['total_amount'] = record['fare_amount'] + extras

        # Ratio features (for detection):
        # implied_speed = INFINITE mph
        # duration = near-zero

        return record

    def combined_impossibility(self, record):
        """
        Combined: Multiple violations at once.

        Strategy: Airport pickup + short time + huge fare + many passengers
        """
        # Airport pickup
        record['PULocationID'] = np.random.choice(list(self.AIRPORT_ZONES))

        # Short distance
        record['trip_distance'] = np.random.uniform(2, 5)

        # Short time
        duration_minutes = np.random.uniform(5, 10)
        pickup_time = record['tpep_pickup_datetime']
        record['tpep_dropoff_datetime'] = pickup_time + timedelta(minutes=duration_minutes)

        # HUGE fare (10-20x normal)
        record['fare_amount'] = np.random.uniform(150, 300)

        # Many passengers
        record['passenger_count'] = np.random.randint(10, 20)

        extras = np.random.uniform(10, 20)
        record['total_amount'] = record['fare_amount'] + extras

        # Multiple impossibilities:
        # - fare_per_mile = $30-150 (EXTREME)
        # - passenger_count = 10-20 (IMPOSSIBLE)
        # - Airport + short trip + huge fare (SUSPICIOUS)

        return record

    def generate(self, clean_df, n_anomalies=5000):
        """
        Generate extreme synthetic anomalies.

        Args:
            clean_df: Clean records to use as templates
            n_anomalies: Number of anomalies to generate

        Returns:
            DataFrame with anomalies, labels DataFrame
        """
        print("="*60)
        print("GENERATING EXTREME SYNTHETIC ANOMALIES")
        print("="*60)

        # Sample base records
        print(f"\n1. Sampling {n_anomalies} base records...")
        base_records = clean_df.sample(n=n_anomalies, replace=True, random_state=42)

        # Generate anomalies
        print(f"\n2. Generating {n_anomalies} extreme anomalies...")
        anomalies = []
        labels = []

        scenarios = list(self.scenarios.keys())
        per_scenario = n_anomalies // len(scenarios)

        for scenario_name, scenario_func in self.scenarios.items():
            print(f"   Generating {per_scenario} × {scenario_name}")

            for idx in range(per_scenario):
                # Get base record
                base_idx = idx + (scenarios.index(scenario_name) * per_scenario)
                if base_idx >= len(base_records):
                    base_idx = base_idx % len(base_records)

                record = base_records.iloc[base_idx].to_dict()

                # Apply scenario
                anomaly_record = scenario_func(record.copy())
                anomalies.append(anomaly_record)

                # Label
                labels.append({
                    'is_anomaly': 1,
                    'scenario': scenario_name
                })

        df_anomalies = pd.DataFrame(anomalies)
        df_labels = pd.DataFrame(labels)

        print(f"\n✓ Generated {len(df_anomalies):,} extreme anomalies")
        print(f"\nScenario breakdown:")
        for scenario in scenarios:
            count = (df_labels['scenario'] == scenario).sum()
            print(f"  {scenario}: {count:,}")

        return df_anomalies, df_labels


def main():
    # Load clean data
    clean_path = 'data/clean/prototype_layer2_clean.parquet'
    print(f"Loading clean data: {clean_path}")
    df_clean = pd.read_parquet(clean_path)
    print(f"✓ Loaded {len(df_clean):,} clean records")

    # Generate anomalies
    generator = ExtremeSyntheticGenerator()
    df_anomalies, df_labels = generator.generate(df_clean, n_anomalies=5000)

    # Combine clean + anomalies
    print(f"\n3. Combining clean + anomalies...")

    # Add labels for clean data
    df_clean_labels = pd.DataFrame({
        'is_anomaly': [0] * len(df_clean),
        'scenario': ['clean'] * len(df_clean)
    })

    # Combine
    df_combined = pd.concat([df_clean, df_anomalies], ignore_index=True)
    df_all_labels = pd.concat([df_clean_labels, df_labels], ignore_index=True)

    # Save
    output_data = 'data/clean/prototype_with_extreme_anomalies.parquet'
    output_labels = 'data/clean/prototype_anomaly_labels.csv'

    df_combined.to_parquet(output_data, index=False)
    df_all_labels.to_csv(output_labels, index=False)

    print(f"\n✓ Saved combined dataset:")
    print(f"  Data: {output_data} ({len(df_combined):,} records)")
    print(f"  Labels: {output_labels}")
    print(f"\nDataset composition:")
    print(f"  Clean: {len(df_clean):,} ({len(df_clean)/len(df_combined)*100:.1f}%)")
    print(f"  Anomalies: {len(df_anomalies):,} ({len(df_anomalies)/len(df_combined)*100:.1f}%)")

    return df_combined, df_all_labels


if __name__ == '__main__':
    main()
