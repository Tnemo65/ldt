"""Baseline sanitization tests.
Spec: Lines 2848-2882 (CRITICAL)
"""

import pytest
import pandas as pd
from pathlib import Path

def test_sanitized_file_exists():
    """Sanitized baseline must exist."""
    path = Path('data/clean/jan_2024_clean_baseline.parquet')
    assert path.exists(), "Sanitized file not found"

def test_sanitized_null_rate():
    """Sanitized data must have null_rate < 0.5%."""
    df = pd.read_parquet('data/clean/jan_2024_clean_baseline.parquet')
    null_rate = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    assert null_rate < 0.5, f"Null rate {null_rate:.2f}% exceeds 0.5%"

def test_sanitized_violation_rate():
    """Sanitized data must have violation_rate < 0.5%."""
    df = pd.read_parquet('data/clean/jan_2024_clean_baseline.parquet')

    violations = (
        (df['fare_amount'] <= 0) |
        (df['trip_distance'] <= 0) |
        (df['passenger_count'] > 6) |
        (df['passenger_count'] == 0)
    )

    viol_rate = violations.sum() / len(df) * 100
    assert viol_rate < 0.5, f"Violation rate {viol_rate:.2f}% exceeds 0.5%"

def test_sanitized_records_count():
    """Sanitized data should have ~2.4-3M records after aggressive filtering."""
    df = pd.read_parquet('data/clean/jan_2024_clean_baseline.parquet')
    assert 2_300_000 <= len(df) <= 3_500_000, \
        f"Record count {len(df):,} outside expected range"
