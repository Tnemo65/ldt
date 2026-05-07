"""Surrogate key generation tests.
Spec: Lines 1515-1535 (MurmurHash3, not MD5)
"""

import pytest
from src.operators.key_generator import generate_trip_id

def test_murmur_hash_deterministic():
    """Same input → same trip_id."""
    record = {
        'VendorID': 1,
        'tpep_pickup_datetime': '2024-01-15T10:30:00',
        'PULocationID': 161,
        'DOLocationID': 230,
        'fare_amount': 15.5
    }

    id1 = generate_trip_id(record)
    id2 = generate_trip_id(record)

    assert id1 == id2, "Trip ID not deterministic"

def test_murmur_hash_32_chars():
    """Trip ID should be 32-char hex string (128-bit hash)."""
    record = {
        'VendorID': 1,
        'tpep_pickup_datetime': '2024-01-15T10:30:00',
        'PULocationID': 161,
        'DOLocationID': 230,
        'fare_amount': 15.5
    }

    trip_id = generate_trip_id(record)

    assert len(trip_id) == 32, f"Expected 32 chars (128-bit), got {len(trip_id)}"
    assert all(c in '0123456789abcdef' for c in trip_id), "Not hex string"

def test_different_records_different_ids():
    """Different records → different trip_ids."""
    record1 = {'VendorID': 1, 'tpep_pickup_datetime': '2024-01-15T10:30:00',
                'PULocationID': 161, 'DOLocationID': 230, 'fare_amount': 15.5}
    record2 = {'VendorID': 1, 'tpep_pickup_datetime': '2024-01-15T10:30:00',
                'PULocationID': 161, 'DOLocationID': 230, 'fare_amount': 16.0}  # Different fare

    id1 = generate_trip_id(record1)
    id2 = generate_trip_id(record2)

    assert id1 != id2, "Different records have same ID"
