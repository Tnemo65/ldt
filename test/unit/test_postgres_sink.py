"""PostgreSQL JDBC sink tests.
Spec: Lines 1595-1620 (JDBC sink configuration)
"""

import pytest
from src.sinks.postgres_sink import (
    create_raw_trips_sink,
    create_violations_sink,
    record_to_raw_trips_row,
    record_to_violation_row
)


def test_record_to_raw_trips_row():
    """Convert record dict to tuple for JDBC insertion."""
    record = {
        'trip_id': 'abc123',
        'VendorID': 1,
        'tpep_pickup_datetime': '2024-01-15T10:30:00',
        'tpep_dropoff_datetime': '2024-01-15T10:45:00',
        'passenger_count': 2,
        'trip_distance': 5.2,
        'PULocationID': 161,
        'DOLocationID': 230,
        'payment_type': 1,
        'fare_amount': 15.5,
        'total_amount': 18.8
    }

    row = record_to_raw_trips_row(record)

    assert row[0] == 'abc123'  # trip_id
    assert row[1] == 1  # vendor_id
    assert row[2] == '2024-01-15T10:30:00'  # pickup_datetime
    assert row[3] == '2024-01-15T10:45:00'  # dropoff_datetime
    assert row[4] == 2  # passenger_count
    assert row[5] == 5.2  # trip_distance
    assert row[6] == 161  # pickup_location_id
    assert row[7] == 230  # dropoff_location_id
    assert row[8] == 1  # payment_type
    assert row[9] == 15.5  # fare_amount
    assert row[10] == 18.8  # total_amount
    assert len(row) == 11  # 11 fields


def test_record_to_violation_row():
    """Convert invalid record to violation row."""
    record = {
        'trip_id': 'abc123',
        'VendorID': 1,
        'PULocationID': 999,  # Invalid zone
        'fare_amount': None  # Missing
    }

    row = record_to_violation_row(record, 'INVALID_ZONE')

    assert row[0] == 'abc123'  # trip_id
    assert row[1] == 'INVALID_ZONE'  # violation_type
    assert 'PULocationID' in row[2] or 'zone' in row[2].lower()  # violation_reason
    assert len(row) == 3  # trip_id, type, reason


@pytest.mark.skip(reason="Requires Flink JVM with JDBC connector JAR - tested in integration tests")
def test_create_raw_trips_sink_not_none():
    """Sink factory should return non-None JDBC sink."""
    sink = create_raw_trips_sink()
    assert sink is not None, "Raw trips sink should not be None"


@pytest.mark.skip(reason="Requires Flink JVM with JDBC connector JAR - tested in integration tests")
def test_create_violations_sink_not_none():
    """Violations sink factory should return non-None JDBC sink."""
    sink = create_violations_sink()
    assert sink is not None, "Violations sink should not be None"
