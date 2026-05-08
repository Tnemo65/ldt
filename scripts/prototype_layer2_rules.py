#!/usr/bin/env python3
"""
Prototype Layer 2: Rule-Based Canary Model
Part of 3-Layer Sequential Funnel architecture.

Purpose:
- Apply hard constraints and business rules
- Block physically impossible trips
- Measure actual violation rate
- Output clean data for Layer 3 (ML)

Expected: Block ~3-5% rule violations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys
from datetime import datetime

sys.path.insert(0, '.')


class RuleBasedValidator:
    """Layer 2: Rule-based canary - catch obvious violations."""

    # Hard constraints on individual fields
    HARD_CONSTRAINTS = {
        'fare_amount': (0, 500),       # $0 - $500
        'trip_distance': (0, 100),     # 0 - 100 miles
        'passenger_count': (0, 9),     # 0 - 9 passengers
        'total_amount': (0, 600),      # $0 - $600
    }

    # Airport location IDs (JFK, LaGuardia, Newark)
    AIRPORT_ZONES = {132, 138, 1}

    def __init__(self):
        self.stats = {
            'total': 0,
            'passed': 0,
            'negative_values': 0,
            'out_of_range': 0,
            'impossible_speed': 0,
            'impossible_fare_distance': 0,
            'impossible_duration': 0,
            'airport_fare_too_low': 0,
        }

    def validate(self, record: dict) -> Tuple[bool, str]:
        """
        Validate a single record against business rules.

        Returns:
            (is_valid, reason)
        """
        self.stats['total'] += 1

        # Rule 1: No negative values
        if record.get('fare_amount', 0) < 0:
            self.stats['negative_values'] += 1
            return False, 'negative_fare'

        if record.get('trip_distance', 0) < 0:
            self.stats['negative_values'] += 1
            return False, 'negative_distance'

        # Rule 2: Values within reasonable ranges
        for field, (min_val, max_val) in self.HARD_CONSTRAINTS.items():
            value = record.get(field, 0)
            if not (min_val <= value <= max_val):
                self.stats['out_of_range'] += 1
                return False, f'range_{field}'

        # Rule 3: Calculate implied speed (if possible)
        try:
            distance = record.get('trip_distance', 0)

            # Calculate duration in hours
            pickup = record.get('tpep_pickup_datetime')
            dropoff = record.get('tpep_dropoff_datetime')

            if isinstance(pickup, str):
                pickup = pd.to_datetime(pickup)
            if isinstance(dropoff, str):
                dropoff = pd.to_datetime(dropoff)

            duration_seconds = (dropoff - pickup).total_seconds()
            duration_hours = duration_seconds / 3600

            if duration_hours <= 0:
                self.stats['impossible_duration'] += 1
                return False, 'zero_duration'

            # Check duration range (1 min to 4 hours)
            if not (60 <= duration_seconds <= 14400):
                self.stats['impossible_duration'] += 1
                return False, 'duration_range'

            # Calculate implied speed
            if distance > 0 and duration_hours > 0:
                implied_speed = distance / duration_hours  # mph

                # Speed limit: 100 mph (NYC + highways)
                if implied_speed > 100:
                    self.stats['impossible_speed'] += 1
                    return False, 'speed_limit'

        except Exception:
            # If can't calculate, skip this rule
            pass

        # Rule 4: Fare vs distance sanity check
        fare = record.get('fare_amount', 0)
        distance = record.get('trip_distance', 0)

        if distance > 20 and fare < 20:
            # Long trip (>20 miles) but cheap fare (<$20) = suspicious
            self.stats['impossible_fare_distance'] += 1
            return False, 'fare_too_low_for_distance'

        # Rule 5: Airport minimum fare
        pu_location = record.get('PULocationID', 0)
        if pu_location in self.AIRPORT_ZONES and fare < 15:
            # Airport pickup but fare <$15 = impossible
            self.stats['airport_fare_too_low'] += 1
            return False, 'airport_minimum_fare'

        # Passed all rules
        self.stats['passed'] += 1
        return True, 'valid'

    def print_stats(self):
        """Print validation statistics."""
        total = self.stats['total']
        passed = self.stats['passed']
        failed = total - passed

        print("\n" + "="*60)
        print("LAYER 2: RULE-BASED VALIDATION RESULTS")
        print("="*60)
        print(f"\nTotal records:        {total:,}")
        print(f"✅ Passed:            {passed:,} ({passed/total*100:.2f}%)")
        print(f"❌ Failed:            {failed:,} ({failed/total*100:.2f}%)")
        print(f"\nViolation breakdown:")
        print(f"  Negative values:        {self.stats['negative_values']:,}")
        print(f"  Out of range:           {self.stats['out_of_range']:,}")
        print(f"  Impossible speed:       {self.stats['impossible_speed']:,}")
        print(f"  Impossible fare/dist:   {self.stats['impossible_fare_distance']:,}")
        print(f"  Impossible duration:    {self.stats['impossible_duration']:,}")
        print(f"  Airport fare too low:   {self.stats['airport_fare_too_low']:,}")


def run_layer2_validation(
    input_path: str = 'data/clean/prototype_layer1_clean.parquet'
):
    """Run Layer 2 validation on Layer 1 output."""
    print("="*60)
    print("LAYER 2: RULE-BASED CANARY MODEL")
    print("="*60)

    print(f"\n1. Loading Layer 1 output: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"   Records: {len(df):,}")

    print(f"\n2. Running rule-based validation...")
    validator = RuleBasedValidator()
    valid_records = []
    violation_reasons = []

    for idx, row in df.iterrows():
        record = row.to_dict()
        is_valid, reason = validator.validate(record)

        if is_valid:
            valid_records.append(record)
        else:
            violation_reasons.append(reason)

        if (idx + 1) % 25000 == 0:
            print(f"   Validated: {idx+1:,} / {len(df):,}")

    validator.print_stats()

    # Save clean data for Layer 3
    if valid_records:
        df_clean = pd.DataFrame(valid_records)
        output_path = 'data/clean/prototype_layer2_clean.parquet'
        df_clean.to_parquet(output_path, index=False)
        print(f"\n✓ Clean data saved to: {output_path}")
        print(f"  Records: {len(df_clean):,}")

    print("\n" + "="*60)
    print("✅ LAYER 2 COMPLETE")
    print("="*60)
    print(f"Input:  {len(df):,} records (from Layer 1)")
    print(f"Output: {len(df_clean):,} clean records (to Layer 3)")
    print(f"Blocked: {len(df) - len(df_clean):,} rule violations")
    print(f"Violation rate: {(len(df) - len(df_clean))/len(df)*100:.2f}%")

    return df_clean, violation_reasons


def main():
    df_clean, violations = run_layer2_validation()
    return df_clean


if __name__ == '__main__':
    main()
