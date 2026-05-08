#!/usr/bin/env python3
"""
Prototype Layer 1: Schema Validation
Part of 3-Layer Sequential Funnel architecture.

Purpose:
- Load 100K subset from Jan 2024
- Validate schema (required fields, data types, ranges)
- Measure violation rate
- Output clean data for Layer 2

Expected: Block ~10% schema violations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import sys

sys.path.insert(0, '.')


class SchemaValidator:
    """Layer 1: Schema validation - first line of defense."""

    REQUIRED_FIELDS = [
        'tpep_pickup_datetime',
        'tpep_dropoff_datetime',
        'trip_distance',
        'fare_amount',
        'total_amount',
        'passenger_count',
        'PULocationID',
        'DOLocationID',
    ]

    FIELD_TYPES = {
        'trip_distance': (int, float),
        'fare_amount': (int, float),
        'total_amount': (int, float),
        'passenger_count': int,
        'PULocationID': int,
        'DOLocationID': int,
    }

    def __init__(self):
        self.stats = {
            'total': 0,
            'missing_fields': 0,
            'wrong_type': 0,
            'null_values': 0,
            'passed': 0,
        }

    def validate(self, record: dict) -> Tuple[bool, str]:
        """
        Validate a single record.

        Returns:
            (is_valid, reason)
        """
        self.stats['total'] += 1

        # Check 1: Required fields present
        for field in self.REQUIRED_FIELDS:
            if field not in record:
                self.stats['missing_fields'] += 1
                return False, f'missing_{field}'

        # Check 2: No null values in critical fields
        for field in self.REQUIRED_FIELDS:
            if pd.isna(record[field]) or record[field] is None:
                self.stats['null_values'] += 1
                return False, f'null_{field}'

        # Check 3: Correct data types
        for field, expected_type in self.FIELD_TYPES.items():
            value = record[field]
            if not isinstance(value, expected_type):
                try:
                    # Try conversion
                    if expected_type == int:
                        int(value)
                    elif expected_type == (int, float):
                        float(value)
                except (ValueError, TypeError):
                    self.stats['wrong_type'] += 1
                    return False, f'type_{field}'

        self.stats['passed'] += 1
        return True, 'valid'

    def print_stats(self):
        """Print validation statistics."""
        total = self.stats['total']
        passed = self.stats['passed']
        failed = total - passed

        print("\n" + "="*60)
        print("LAYER 1: SCHEMA VALIDATION RESULTS")
        print("="*60)
        print(f"\nTotal records:        {total:,}")
        print(f"✅ Passed:            {passed:,} ({passed/total*100:.2f}%)")
        print(f"❌ Failed:            {failed:,} ({failed/total*100:.2f}%)")
        print(f"\nViolation breakdown:")
        print(f"  Missing fields:     {self.stats['missing_fields']:,}")
        print(f"  Wrong types:        {self.stats['wrong_type']:,}")
        print(f"  Null values:        {self.stats['null_values']:,}")


def create_prototype_subset(
    input_path: str = 'data/clean/jan_2024_clean_baseline.parquet',
    output_path: str = 'data/clean/prototype_100k.parquet',
    size: int = 100_000
):
    """Create 100K random subset for prototyping."""
    print("="*60)
    print("CREATING PROTOTYPE SUBSET")
    print("="*60)

    print(f"\n1. Loading data from: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   Total records: {len(df):,}")

    print(f"\n2. Sampling {size:,} records...")
    df_subset = df.sample(n=size, random_state=42)

    print(f"\n3. Saving to: {output_path}")
    df_subset.to_parquet(output_path, index=False)
    print(f"   ✓ Saved {len(df_subset):,} records")

    return df_subset


def validate_subset(df: pd.DataFrame):
    """Run schema validation on subset."""
    print("\n" + "="*60)
    print("RUNNING SCHEMA VALIDATION")
    print("="*60)

    validator = SchemaValidator()
    valid_records = []
    violation_reasons = []

    print(f"\nValidating {len(df):,} records...")
    for idx, row in df.iterrows():
        record = row.to_dict()
        is_valid, reason = validator.validate(record)

        if is_valid:
            valid_records.append(record)
        else:
            violation_reasons.append(reason)

        if (idx + 1) % 25000 == 0:
            print(f"  Validated: {idx+1:,} / {len(df):,}")

    validator.print_stats()

    # Save clean data for Layer 2
    if valid_records:
        df_clean = pd.DataFrame(valid_records)
        output_path = 'data/clean/prototype_layer1_clean.parquet'
        df_clean.to_parquet(output_path, index=False)
        print(f"\n✓ Clean data saved to: {output_path}")
        print(f"  Records: {len(df_clean):,}")

    return df_clean, violation_reasons


def main():
    # Step 1: Create subset if not exists
    subset_path = 'data/clean/prototype_100k.parquet'

    if not Path(subset_path).exists():
        print("Subset not found, creating...")
        df = create_prototype_subset()
    else:
        print(f"Loading existing subset: {subset_path}")
        df = pd.read_parquet(subset_path)
        print(f"✓ Loaded {len(df):,} records")

    # Step 2: Run validation
    df_clean, violations = validate_subset(df)

    print("\n" + "="*60)
    print("✅ LAYER 1 COMPLETE")
    print("="*60)
    print(f"Input:  {len(df):,} records")
    print(f"Output: {len(df_clean):,} clean records")
    print(f"Blocked: {len(df) - len(df_clean):,} violations")

    return df_clean


if __name__ == '__main__':
    main()
