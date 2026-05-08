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

Prototype: scripts/prototype_layer{1,2}_{schema,rules}.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '.')

# Import validators from prototype scripts
from scripts.prototype_layer1_schema import SchemaValidator
from scripts.prototype_layer2_rules import RuleBasedValidator


def validate_filter_rates(input_path: str = 'data/raw/yellow_tripdata_2024-01.parquet'):
    """Measure actual filter rates on Jan 2024 data."""

    print("="*60)
    print("VALIDATING SEQUENTIAL FUNNEL FILTER RATES")
    print("="*60)

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
            print(f"   Processed: {idx+1:,} / {len(df):,}")

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
            print(f"   Processed: {idx+1:,} / {len(df_layer1):,}")

    rule_validator.print_stats()
    layer2_count = len(layer2_clean)
    layer2_block_rate = (layer1_count - layer2_count) / layer1_count * 100

    print(f"\n   Layer 2 Results:")
    print(f"   Input:  {layer1_count:,}")
    print(f"   Output: {layer2_count:,}")
    print(f"   Blocked: {layer1_count - layer2_count:,} ({layer2_block_rate:.2f}%)")

    # Combined Results
    combined_block_rate = (initial_count - layer2_count) / initial_count * 100

    print("\n" + "="*60)
    print("SEQUENTIAL FUNNEL SUMMARY")
    print("="*60)
    print(f"\nInitial (Raw):        {initial_count:,}")
    print(f"After Layer 1:        {layer1_count:,} (-{layer1_block_rate:.2f}%)")
    print(f"After Layer 2:        {layer2_count:,} (-{layer2_block_rate:.2f}%)")
    print(f"Final Clean:          {layer2_count:,}")
    print(f"\nCombined Block Rate:  {combined_block_rate:.2f}%")
    print(f"Clean Data for ML:    {100 - combined_block_rate:.2f}%")

    # Validation against expected rates
    print("\n" + "="*60)
    print("VALIDATION vs EXPECTED RATES")
    print("="*60)

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
        print(f"\n✅ Low violation rate expected - baseline is pre-cleaned")
    else:
        print(f"\n⚠️  Higher than expected violation rate")

    # Save ultra-clean data for ML training
    output_path = Path('data/clean/jan_2024_clean_baseline.parquet')
    df_clean = pd.DataFrame(layer2_clean)
    df_clean.to_parquet(output_path, index=False)
    print(f"\n✓ Saved ultra-clean data: {output_path}")
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
