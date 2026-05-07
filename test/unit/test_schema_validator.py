"""Schema validation tests.
Spec: Lines 1570-1590 (Required fields, zone validation)
"""

import pytest
from src.operators.schema_validator import SchemaValidator


def test_valid_record_passes():
    """Valid record with all required fields should pass."""
    validator = SchemaValidator()

    record = {
        'trip_id': 'abc123',
        'VendorID': 1,
        'tpep_pickup_datetime': '2024-01-15T10:30:00',
        'PULocationID': 161,
        'DOLocationID': 230,
        'trip_distance': 5.2,
        'fare_amount': 15.5,
        'passenger_count': 2
    }

    result = validator.filter(record)
    assert result is True, "Valid record should pass"


def test_missing_required_field_rejected():
    """Record missing required field should be rejected."""
    validator = SchemaValidator()

    record = {
        'trip_id': 'abc123',
        'VendorID': 1,
        # Missing fare_amount
        'PULocationID': 161,
        'DOLocationID': 230,
        'trip_distance': 5.2,
        'passenger_count': 2
    }

    result = validator.filter(record)
    assert result is False, "Record with missing field should be rejected"


def test_null_required_field_rejected():
    """Record with null required field should be rejected."""
    validator = SchemaValidator()

    record = {
        'trip_id': 'abc123',
        'VendorID': 1,
        'tpep_pickup_datetime': '2024-01-15T10:30:00',
        'PULocationID': 161,
        'DOLocationID': 230,
        'trip_distance': None,  # Null value
        'fare_amount': 15.5,
        'passenger_count': 2
    }

    result = validator.filter(record)
    assert result is False, "Record with null required field should be rejected"


def test_invalid_zone_id_rejected():
    """Record with invalid zone ID (out of range 1-263) should be rejected."""
    validator = SchemaValidator()

    record = {
        'trip_id': 'abc123',
        'VendorID': 1,
        'tpep_pickup_datetime': '2024-01-15T10:30:00',
        'PULocationID': 999,  # Invalid zone ID
        'DOLocationID': 230,
        'trip_distance': 5.2,
        'fare_amount': 15.5,
        'passenger_count': 2
    }

    result = validator.filter(record)
    assert result is False, "Record with invalid zone ID should be rejected"


def test_zone_id_zero_rejected():
    """Record with zone ID = 0 should be rejected."""
    validator = SchemaValidator()

    record = {
        'trip_id': 'abc123',
        'VendorID': 1,
        'tpep_pickup_datetime': '2024-01-15T10:30:00',
        'PULocationID': 0,  # Invalid zone ID
        'DOLocationID': 230,
        'trip_distance': 5.2,
        'fare_amount': 15.5,
        'passenger_count': 2
    }

    result = validator.filter(record)
    assert result is False, "Zone ID 0 should be rejected"


def test_boundary_zone_ids_valid():
    """Zone IDs 1 and 263 (boundaries) should be valid."""
    validator = SchemaValidator()

    record1 = {
        'trip_id': 'abc123',
        'VendorID': 1,
        'tpep_pickup_datetime': '2024-01-15T10:30:00',
        'PULocationID': 1,  # Lower boundary
        'DOLocationID': 263,  # Upper boundary
        'trip_distance': 5.2,
        'fare_amount': 15.5,
        'passenger_count': 2
    }

    result = validator.filter(record1)
    assert result is True, "Zone IDs 1 and 263 should be valid"
