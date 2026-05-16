#!/usr/bin/env python3
"""Unit tests for warmup coverage validator."""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from validate_warmup_coverage import validate_warmup_coverage, _get_warnings


def make_df(dates, hours):
    """Helper: create test DataFrame from lists of (date_str, hour) tuples."""
    rows = [{'tpep_pickup_datetime': f'{d} {h:02d}:00:00'} for d, h in zip(dates, hours)]
    return pd.DataFrame(rows)


def test_full_week_passes():
    """A complete Mon-Sun week should pass."""
    dates = []
    hours = list(range(0, 24))  # All 24 hours
    for day in range(7):
        for h in hours:
            dates.append((f'2024-01-{(day+1):02d}', h))
    df = make_df([d for d, h in dates], [h for d, h in dates])
    result = validate_warmup_coverage(df)
    assert result['passes_check'], f"Full week should pass but got: {result}"
    assert result['has_full_week']
    assert result['weekday_dates'] >= 5
    assert result['weekend_dates'] >= 2


def test_weekend_only_fails():
    """Weekend-only data should fail."""
    dates = []
    for day in [5, 6]:  # Sat, Sun
        for h in [12, 18]:
            dates.append((f'2024-01-{(day+1):02d}', h))
    df = make_df([d for d, h in dates], [h for d, h in dates])
    result = validate_warmup_coverage(df)
    assert not result['passes_check'], "Weekend-only should not pass coverage check"
    warnings_text = str(result.get('warnings', []))
    assert 'NO WEEKDAY DATA' in warnings_text, \
        f"Expected 'NO WEEKDAY DATA' warning but got: {warnings_text}"


def test_limited_hours_fails():
    """Less than 18 hours should fail."""
    dates = [(f'2024-01-0{i+1}', 12) for i in range(7)]  # Only noon
    df = make_df([d for d, h in dates], [h for d, h in dates])
    result = validate_warmup_coverage(df)
    assert not result['passes_check']
    assert any('hour' in w.lower() for w in result['warnings'])


def test_coverage_score():
    """Coverage score should be between 0-100."""
    df = make_df(
        [f'2024-01-{(i%7)+1:02d}' for i in range(10)],
        [(i % 24) for i in range(100)]
    )
    result = validate_warmup_coverage(df)
    assert 0 <= result['coverage_score'] <= 100


def test_empty_df():
    """Empty DataFrame should return error."""
    df = pd.DataFrame(columns=['tpep_pickup_datetime'])
    result = validate_warmup_coverage(df)
    assert not result['passes_check']


if __name__ == '__main__':
    tests = [
        test_full_week_passes,
        test_weekend_only_fails,
        test_limited_hours_fails,
        test_coverage_score,
        test_empty_df,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            msg = str(e) if e.args else "(no message)"
            print(f"  FAIL: {t.__name__}: {msg}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
