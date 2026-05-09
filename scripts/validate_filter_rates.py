#!/usr/bin/env python3
"""
Validate Sequential Funnel Filter Rates

Simulates Layer 1 (Schema) + Layer 2 (Rules) on Jan 2024 raw data
to measure actual violation rates before ML model training.

Expected:
- Layer 1 (Schema): ~10.08% blocked (on RAW production data)
- Layer 2 (Rules): ~3.41% blocked
- Combined: ~13.49% blocked
- Output for ML: ~86.51% ultra-clean data

NOTE: Jan 2024 baseline is pre-cleaned, so expect low violation rate (~0.16%).
The 13.49% rate applies to RAW production data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class SchemaValidator:
    """Layer 1: Schema validation - reject records with missing/invalid required fields."""

    REQUIRED_FIELDS = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime',
        'trip_distance', 'fare_amount', 'PULocationID', 'DOLocationID',
        'passenger_count',
    ]

    def __init__(self):
        self.stats = {f: 0 for f in ['null', 'zone', 'passenger', 'total']}
        self.total = 0

    def validate(self, record: dict):
        """Returns (is_valid, reason)."""
        self.total += 1

        # Null check
        for field in self.REQUIRED_FIELDS:
            val = record.get(field)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                self.stats['null'] += 1
                return False, f"null:{field}"

        # Zone validation
        pu = record.get('PULocationID', 0)
        do = record.get('DOLocationID', 0)
        if not (1 <= pu <= 265) or not (1 <= do <= 265):
            self.stats['zone'] += 1
            return False, f"invalid_zone:pu={pu},do={do}"

        # Passenger count
        pax = record.get('passenger_count', 0)
        if not (1 <= pax <= 6):
            self.stats['passenger'] += 1
            return False, f"invalid_passenger:{pax}"

        return True, "ok"

    def print_stats(self):
        for k, v in self.stats.items():
            if v > 0:
                print(f"  {k}: {v:,} rejected")


class RuleBasedValidator:
    """Layer 2: Rule-based canary - reject physical impossibilities."""

    def __init__(self):
        self.stats = {k: 0 for k in ['negative_fare', 'zero_dist', 'zero_dur', 'speed', 'total']}
        self.total = 0

    def validate(self, record: dict):
        """Returns (is_valid, reason)."""
        self.total += 1

        # Negative fare
        fare = record.get('fare_amount', 0)
        if fare <= 0:
            self.stats['negative_fare'] += 1
            return False, f"non_positive_fare:{fare}"

        # Zero distance
        dist = record.get('trip_distance', 0)
        if dist <= 0:
            self.stats['zero_dist'] += 1
            return False, f"non_positive_distance:{dist}"

        # Duration calculation
        try:
            pickup = pd.to_datetime(record['tpep_pickup_datetime'])
            dropoff = pd.to_datetime(record['tpep_dropoff_datetime'])
            dur_sec = (dropoff - pickup).total_seconds()
        except Exception:
            self.stats['zero_dur'] += 1
            return False, "datetime_parse_error"

        if dur_sec <= 0:
            self.stats['zero_dur'] += 1
            return False, f"non_positive_duration:{dur_sec}"

        # Speed check (physically impossible > 100 mph in NYC)
        dur_hr = dur_sec / 3600
        speed = dist / dur_hr if dur_hr > 0 else 0
        if speed > 100 or speed <= 0:
            self.stats['speed'] += 1
            return False, f"impossible_speed:{speed:.1f}_mph"

        return True, "ok"

    def print_stats(self):
        for k, v in self.stats.items():
            if v > 0:
                print(f"  {k}: {v:,} rejected")


def validate_filter_rates(input_path: str = 'data/raw/yellow_tripdata_2024-01.parquet'):
    """Measure actual filter rates on Jan 2024 data."""

    print("=" * 60)
    print("VALIDATING SEQUENTIAL FUNNEL FILTER RATES")
    print("=" * 60)

    # Load raw data
    print(f"\n1. Loading raw data: {input_path}")
    df = pd.read_parquet(input_path)
    initial_count = len(df)
    print(f"   Records: {initial_count:,}")

    # Layer 1: Schema Validation
    print(f"\n2. Running Layer 1 (Schema Validation)...")
    schema_validator = SchemaValidator()
    layer1_clean = []

    for idx, row in df.iterrows():
        is_valid, reason = schema_validator.validate(row.to_dict())
        if is_valid:
            layer1_clean.append(row.to_dict())

        if (idx + 1) % 100000 == 0:
            print(f"   Processed: {idx + 1:,} / {len(df):,}")

    schema_validator.print_stats()
    layer1_count = len(layer1_clean)
    layer1_block_rate = (initial_count - layer1_count) / initial_count * 100

    print(f"\n   Layer 1 Results:")
    print(f"   Input:  {initial_count:,}")
    print(f"   Output: {layer1_count:,}")
    print(f"   Blocked: {initial_count - layer1_count:,} ({layer1_block_rate:.2f}%)")

    # Layer 2: Rule-Based Validation
    print(f"\n3. Running Layer 2 (Rule-Based Canary)...")
    rule_validator = RuleBasedValidator()
    layer2_clean = []

    df_layer1 = pd.DataFrame(layer1_clean)
    for idx, row in df_layer1.iterrows():
        is_valid, reason = rule_validator.validate(row.to_dict())
        if is_valid:
            layer2_clean.append(row.to_dict())

        if (idx + 1) % 100000 == 0:
            print(f"   Processed: {idx + 1:,} / {len(df_layer1):,}")

    rule_validator.print_stats()
    layer2_count = len(layer2_clean)
    layer2_block_rate = (layer1_count - layer2_count) / layer1_count * 100

    print(f"\n   Layer 2 Results:")
    print(f"   Input:  {layer1_count:,}")
    print(f"   Output: {layer2_count:,}")
    print(f"   Blocked: {layer1_count - layer2_count:,} ({layer2_block_rate:.2f}%)")

    # Combined Results
    combined_block_rate = (initial_count - layer2_count) / initial_count * 100

    print("\n" + "=" * 60)
    print("SEQUENTIAL FUNNEL SUMMARY")
    print("=" * 60)
    print(f"\nInitial (Raw):        {initial_count:,}")
    print(f"After Layer 1:        {layer1_count:,} (-{layer1_block_rate:.2f}%)")
    print(f"After Layer 2:        {layer2_count:,} (-{layer2_block_rate:.2f}%)")
    print(f"Final Clean:           {layer2_count:,}")
    print(f"\nCombined Block Rate:   {combined_block_rate:.2f}%")
    print(f"Clean Data for ML:    {100 - combined_block_rate:.2f}%")

    # Validation against expected rates
    print("\n" + "=" * 60)
    print("VALIDATION vs EXPECTED RATES")
    print("=" * 60)

    print(f"\nNOTE: Expected rates apply to RAW production data.")
    print(f"Jan 2024 baseline is pre-cleaned, so low violation rate is expected.")

    expected_layer1 = 10.08
    expected_layer2 = 3.41
    expected_combined = 13.49

    print(f"\nExpected (on RAW production):")
    print(f"  Layer 1: {expected_layer1}%")
    print(f"  Layer 2: {expected_layer2}%")
    print(f"  Combined: {expected_combined}%")

    print(f"\nActual (on Jan 2024 baseline):")
    print(f"  Layer 1: {layer1_block_rate:.2f}%")
    print(f"  Layer 2: {layer2_block_rate:.2f}%")
    print(f"  Combined: {combined_block_rate:.2f}%")

    if combined_block_rate < 1.0:
        print(f"\n  Low violation rate expected - baseline is pre-cleaned")
    else:
        print(f"\n  Higher than expected violation rate")

    # Save ultra-clean data for ML training
    output_path = Path('data/clean/jan_2024_clean_baseline.parquet')
    df_clean = pd.DataFrame(layer2_clean)
    df_clean.to_parquet(output_path, index=False)
    print(f"\n  Saved ultra-clean data: {output_path}")
    print(f"  Records: {len(df_clean):,}")
    print(f"  Ready for iForestASD training (Task 2.1)")

    return {
        'initial': initial_count,
        'layer1_output': layer1_count,
        'layer2_output': layer2_count,
        'layer1_block_rate': layer1_block_rate,
        'layer2_block_rate': layer2_block_rate,
        'combined_block_rate': combined_block_rate,
    }


if __name__ == '__main__':
    results = validate_filter_rates()
