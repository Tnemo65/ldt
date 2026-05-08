#!/usr/bin/env python3
"""
Layer 0 Filter: Schema Validation + 7 Hard Business Rules
Task 0.9: Generate clean baseline from raw data

Input: raw parquet (3,066,766 records)
Output: clean baseline (expected ~2.9M records, 95% pass rate)

Usage:
  python scripts/layer0_filter.py \
    --input data/yellow_tripdata_2024-01.parquet \
    --output data/clean/jan_2024_clean_baseline.parquet
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def schema_validation(record):
    """Validate strict schema fields (must not be NULL).

    Strict fields (ML features + keyBy):
    - tpep_pickup_datetime (for watermark)
    - tpep_dropoff_datetime (for duration calc)
    - passenger_count (ML feature)
    - trip_distance (ML feature)
    - PULocationID (keyBy spatial)
    - DOLocationID (ML feature)
    - payment_type (ML feature)
    - fare_amount (ML feature)
    - total_amount (ML feature)
    """
    strict_fields = [
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime',
        'passenger_count',
        'trip_distance',
        'PULocationID',
        'DOLocationID',
        'payment_type',
        'fare_amount',
        'total_amount'
    ]

    for field in strict_fields:
        if pd.isna(record[field]):
            return False, f'null_{field}'

    return True, None


def hard_rule_validation(record):
    """Apply 7 hard business rules.

    Rules (from plan Task 0.9):
    1. negative_fare: fare_amount <= 0
    2. negative_total: total_amount <= 0
    3. zero_distance_with_fare: trip_distance == 0 AND fare_amount > 0
    4. invalid_passengers: passenger_count < 1 OR > 6
    5. invalid_location: PULocationID < 1 OR > 265, DOLocationID < 1 OR > 263
    6. invalid_payment: payment_type < 1 OR > 6
    7. extreme_duration: duration < 60s OR > 24 hours
    """
    violations = []

    # Rule 1: negative_fare
    if record['fare_amount'] <= 0:
        violations.append('negative_fare')

    # Rule 2: negative_total
    if record['total_amount'] <= 0:
        violations.append('negative_total')

    # Rule 3: zero_distance_with_fare
    if record['trip_distance'] == 0 and record['fare_amount'] > 0:
        violations.append('zero_distance_with_fare')

    # Rule 4: invalid_passengers
    if record['passenger_count'] < 1 or record['passenger_count'] > 6:
        violations.append('invalid_passengers')

    # Rule 5: invalid_location
    if record['PULocationID'] < 1 or record['PULocationID'] > 265:
        violations.append('invalid_PU_location')
    if record['DOLocationID'] < 1 or record['DOLocationID'] > 263:
        violations.append('invalid_DO_location')

    # Rule 6: invalid_payment
    if record['payment_type'] < 1 or record['payment_type'] > 6:
        violations.append('invalid_payment')

    # Rule 7: extreme_duration
    try:
        pickup = pd.to_datetime(record['tpep_pickup_datetime'])
        dropoff = pd.to_datetime(record['tpep_dropoff_datetime'])
        duration_sec = (dropoff - pickup).total_seconds()

        if duration_sec < 60:  # < 1 minute
            violations.append('duration_too_short')
        elif duration_sec > 86400:  # > 24 hours
            violations.append('duration_too_long')
    except:
        violations.append('invalid_datetime')

    if violations:
        return False, violations

    return True, None


def layer0_filter(input_path: Path, output_path: Path):
    """Apply Layer 0 filtering: Schema + 7 Hard Rules.

    Args:
        input_path: Raw parquet file
        output_path: Clean baseline parquet file

    Returns:
        Statistics dict
    """
    print("="*80)
    print("Layer 0 Filter: Schema Validation + 7 Hard Business Rules")
    print("="*80)

    # Load raw data
    print(f"\n1. Loading raw data: {input_path}")
    df = pd.read_parquet(input_path)
    initial_count = len(df)
    print(f"   Initial records: {initial_count:,}")

    # Statistics
    stats = {
        'initial': initial_count,
        'schema_violations': 0,
        'hard_rule_violations': 0,
        'clean_output': 0,
        'violation_details': {}
    }

    # Filter
    print("\n2. Applying filters...")
    clean_records = []

    for idx, record in df.iterrows():
        # Schema validation
        schema_ok, schema_violation = schema_validation(record)
        if not schema_ok:
            stats['schema_violations'] += 1
            if schema_violation not in stats['violation_details']:
                stats['violation_details'][schema_violation] = 0
            stats['violation_details'][schema_violation] += 1
            continue

        # Hard rule validation
        rules_ok, rule_violations = hard_rule_validation(record)
        if not rules_ok:
            stats['hard_rule_violations'] += 1
            for v in rule_violations:
                if v not in stats['violation_details']:
                    stats['violation_details'][v] = 0
                stats['violation_details'][v] += 1
            continue

        # Clean record
        clean_records.append(record)

        # Progress
        if (idx + 1) % 100000 == 0:
            print(f"   Processed: {idx + 1:,} / {initial_count:,} "
                  f"({(idx + 1) / initial_count * 100:.1f}%)")

    stats['clean_output'] = len(clean_records)

    # Save clean baseline
    print(f"\n3. Saving clean baseline: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df = pd.DataFrame(clean_records)
    clean_df.to_parquet(output_path, index=False)

    # Report statistics
    print("\n" + "="*80)
    print("LAYER 0 FILTERING RESULTS")
    print("="*80)
    print(f"Initial records:          {stats['initial']:>10,}")
    print(f"Schema violations:        {stats['schema_violations']:>10,} "
          f"({stats['schema_violations']/stats['initial']*100:>5.2f}%)")
    print(f"Hard rule violations:     {stats['hard_rule_violations']:>10,} "
          f"({stats['hard_rule_violations']/stats['initial']*100:>5.2f}%)")
    print(f"Clean output:             {stats['clean_output']:>10,} "
          f"({stats['clean_output']/stats['initial']*100:>5.2f}%)")

    print("\nViolation breakdown:")
    for violation, count in sorted(stats['violation_details'].items(),
                                   key=lambda x: x[1], reverse=True):
        print(f"  {violation:.<35} {count:>8,} "
              f"({count/stats['initial']*100:>5.2f}%)")

    print("\n" + "="*80)
    print("✅ Clean baseline created!")
    print(f"   Output: {output_path}")
    print(f"   Pass rate: {stats['clean_output']/stats['initial']*100:.2f}%")
    print("="*80)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Layer 0 Filter: Generate clean baseline'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input raw parquet file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output clean baseline parquet file'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1

    stats = layer0_filter(input_path, output_path)

    # Validate pass rate (should be ~95%)
    pass_rate = stats['clean_output'] / stats['initial'] * 100
    if pass_rate < 90:
        print(f"\n⚠️  Warning: Pass rate {pass_rate:.2f}% is lower than expected (95%)")
        print("    Check if input data quality is as expected")

    return 0


if __name__ == '__main__':
    exit(main())
