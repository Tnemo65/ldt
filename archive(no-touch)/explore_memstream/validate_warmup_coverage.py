#!/usr/bin/env python3
"""
Warmup Coverage Validator for NYC Taxi Periodic Data.

Ensures warmup data contains at least 1 complete weekly cycle
(Mon-Sun) and adequate hour coverage to avoid false anomalies
during periodic patterns.

Usage:
    python validate_warmup_coverage.py --data /path/to/nyc_taxi.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def validate_warmup_coverage(df_warmup: pd.DataFrame) -> dict:
    """
    Verify warmup contains at least 1 complete cycle.

    Returns dict with:
        - unique_dates: number of unique calendar dates
        - weekday_dates: Mon-Fri dates
        - weekend_dates: Sat-Sun dates  
        - hour_coverage: number of unique hours (0-23)
        - has_full_week: bool
        - passes_check: bool (all requirements met)
        - span_days: total calendar span
        - coverage_score: 0-100 quality score
    """
    df = df_warmup.copy()
    df['dt'] = pd.to_datetime(df.get('tpep_pickup_datetime', df.get('pickup_datetime', None)), errors='coerce')
    
    if df['dt'].isna().all():
        # Try parsing from raw datetime columns
        if 'tpep_pickup_datetime' in df_warmup.columns:
            df['dt'] = pd.to_datetime(df_warmup['tpep_pickup_datetime'], errors='coerce')
        elif 'pickup_datetime' in df_warmup.columns:
            df['dt'] = pd.to_datetime(df_warmup['pickup_datetime'], errors='coerce')
    
    df = df.dropna(subset=['dt'])
    if len(df) == 0:
        return {'error': 'No parseable datetime found', 'passes_check': False}

    df['date'] = df['dt'].dt.date
    df['dow'] = df['dt'].dt.dayofweek  # 0=Mon, 6=Sun
    df['hour'] = df['dt'].dt.hour

    unique_dates = df['date'].nunique()
    weekday_mask = df['dow'] < 5
    weekend_mask = df['dow'] >= 5
    
    weekday_dates = df[weekday_mask]['date'].nunique()
    weekend_dates = df[weekend_mask]['date'].nunique()
    
    # Minimum 5 weekday days and 2 weekend days = 1 complete week
    has_full_week = weekday_dates >= 5 and weekend_dates >= 2
    
    # Hour coverage: need at least 18 of 24 hours represented
    hour_coverage = df['hour'].nunique()
    hour_score = min(hour_coverage / 18.0, 1.0)
    
    # Week coverage score
    week_score = min(weekday_dates / 5.0, 1.0) * 0.5 + min(weekend_dates / 2.0, 1.0) * 0.5
    
    # Date span
    span_days = (df['date'].max() - df['date'].min()).days + 1 if unique_dates > 0 else 0
    
    coverage_score = int((week_score * 0.6 + hour_score * 0.4) * 100)
    
    passes_check = (
        has_full_week and 
        hour_coverage >= 18 and
        weekday_dates >= 5 and
        weekend_dates >= 2
    )
    
    return {
        'unique_dates': unique_dates,
        'weekday_dates': weekday_dates,
        'weekend_dates': weekend_dates,
        'hour_coverage': hour_coverage,
        'has_full_week': has_full_week,
        'hour_score': float(hour_score),
        'week_score': float(week_score),
        'coverage_score': coverage_score,
        'span_days': span_days,
        'passes_check': passes_check,
        'n_records': len(df),
        'warnings': _get_warnings(weekday_dates, weekend_dates, hour_coverage, span_days),
    }


def _get_warnings(weekday_dates: int, weekend_dates: int, 
                  hour_coverage: int, span_days: int) -> list:
    """Generate human-readable warnings."""
    warnings = []
    if weekday_dates < 5:
        warnings.append(f"Only {weekday_dates} weekday days (need >= 5 for complete week)")
    if weekend_dates < 2:
        warnings.append(f"Only {weekend_dates} weekend days (need >= 2 for complete week)")
    if hour_coverage < 18:
        warnings.append(f"Only {hour_coverage} hours covered (need >= 18 for representative coverage)")
    if span_days < 7:
        warnings.append(f"Total span is {span_days} days (need >= 7 for weekly cycle)")
    if weekday_dates > 0 and weekend_dates == 0:
        warnings.append("NO WEEKEND DATA - all weekend trips will be flagged as anomalous!")
    if weekend_dates > 0 and weekday_dates == 0:
        warnings.append("NO WEEKDAY DATA - all weekday trips will be flagged as anomalous!")
    return warnings


def main():
    parser = argparse.ArgumentParser(description='Validate warmup data coverage')
    parser.add_argument('--data', type=str, required=True, help='Warmup CSV path')
    parser.add_argument('--date-col', type=str, default='tpep_pickup_datetime',
                        help='Datetime column name')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Fraction of data to validate (for sampling)')
    args = parser.parse_args()

    print("=" * 60)
    print("Warmup Coverage Validator")
    print("=" * 60)

    df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data)
    if args.frac < 1.0:
        df = df.sample(frac=args.frac, random_state=42)
    print(f"\nLoaded {len(df):,} records")

    result = validate_warmup_coverage(df)

    print(f"\n{'Coverage Analysis':=^60}")
    print(f"  Unique dates:          {result.get('unique_dates', 'N/A')}")
    print(f"  Weekday days:          {result.get('weekday_dates', 'N/A')}")
    print(f"  Weekend days:          {result.get('weekend_dates', 'N/A')}")
    print(f"  Hour coverage:         {result.get('hour_coverage', 'N/A')}/24 hours")
    print(f"  Calendar span:         {result.get('span_days', 'N/A')} days")
    print(f"  Has complete week:     {result.get('has_full_week', 'N/A')}")
    print(f"  Coverage score:        {result.get('coverage_score', 'N/A')}/100")

    if result.get('warnings'):
        print(f"\n{'Warnings':=^60}")
        for w in result['warnings']:
            print(f"  WARNING: {w}")

    print(f"\n{'Result':=^60}")
    if result['passes_check']:
        print("  PASS - Warmup data is representative")
    else:
        print("  FAIL - Warmup data may cause false anomalies")
        print("  Recommendation: Include at least 1 full week (Mon-Sun)")

    print("=" * 60)
    return 0 if result['passes_check'] else 1


if __name__ == '__main__':
    sys.exit(main())
