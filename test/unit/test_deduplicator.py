"""Deduplication tests using KeyedState with TTL.
Spec: Lines 1540-1565 (7-day TTL, OnCreateAndWrite)
"""

import pytest
from src.operators.deduplicator import DeduplicatorFunction


def test_first_record_passes():
    """First occurrence of trip_id should pass through."""
    dedup = DeduplicatorFunction()

    record = {
        'trip_id': 'abc123',
        'VendorID': 1,
        'fare_amount': 15.5
    }

    # Mock runtime context (simplified for unit test)
    # In real Flink, state is managed by RuntimeContext
    result = dedup.map(record)

    assert result is not None, "First record should pass"
    assert result['trip_id'] == 'abc123'


def test_duplicate_record_filtered():
    """Duplicate trip_id should be filtered (returns None)."""
    dedup = DeduplicatorFunction()

    record = {
        'trip_id': 'abc123',
        'VendorID': 1,
        'fare_amount': 15.5
    }

    # First occurrence
    result1 = dedup.map(record)
    assert result1 is not None, "First should pass"

    # Duplicate
    result2 = dedup.map(record)
    assert result2 is None, "Duplicate should be filtered"


def test_different_trip_ids_both_pass():
    """Different trip_ids should both pass."""
    dedup = DeduplicatorFunction()

    record1 = {'trip_id': 'abc123', 'fare_amount': 15.5}
    record2 = {'trip_id': 'xyz789', 'fare_amount': 20.0}

    result1 = dedup.map(record1)
    result2 = dedup.map(record2)

    assert result1 is not None
    assert result2 is not None
    assert result1['trip_id'] == 'abc123'
    assert result2['trip_id'] == 'xyz789'


def test_ttl_config():
    """Deduplicator should have 7-day TTL config."""
    dedup = DeduplicatorFunction()

    # Verify TTL is 7 days (in real Flink, this is tested via state descriptor)
    assert dedup.ttl_days == 7, "TTL should be 7 days"
