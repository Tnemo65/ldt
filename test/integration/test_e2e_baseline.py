"""
End-to-end integration test for baseline pipeline.
Spec: Lines 1675-1700 (Kafka → Flink → PostgreSQL)

IMPORTANT: Requires running infrastructure:
- Docker Compose services (Kafka, PostgreSQL, PgBouncer)
- Kafka topics created
- PostgreSQL schema initialized

Run with:
  pytest test/integration/test_e2e_baseline.py -v
"""

import pytest
import time
import subprocess
import psycopg2
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.produce_taxi_data import create_producer, read_parquet_records


@pytest.fixture(scope="module")
def kafka_producer():
    """Create Kafka producer for testing."""
    try:
        producer = create_producer('localhost:9092')
        yield producer
        producer.close()
    except Exception as e:
        pytest.skip(f"Kafka not available: {e}")


@pytest.fixture(scope="module")
def postgres_connection():
    """Create PostgreSQL connection for verification."""
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='dq_pipeline',
            user='cadqstream',
            password='cadqstream123'
        )
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available: {e}")


def produce_test_records(producer, topic: str, count: int = 1000):
    """Produce test records to Kafka topic.

    Args:
        producer: KafkaProducer instance
        topic: Kafka topic name
        count: Number of records to send
    """
    data_file = Path('data/clean/jan_2024_clean_baseline.parquet')
    if not data_file.exists():
        pytest.skip(f"Test data not found: {data_file}")

    sent = 0
    for record in read_parquet_records(data_file, limit=count):
        producer.send(topic, value=record)
        sent += 1

    producer.flush()
    return sent


@pytest.mark.integration
@pytest.mark.slow
def test_e2e_baseline_pipeline(kafka_producer, postgres_connection):
    """Test complete data flow: Kafka → Flink → PostgreSQL.

    Flow:
    1. Produce 1000 test records to Kafka
    2. Wait for Flink job to process (assumes job is running)
    3. Verify records appear in PostgreSQL
    4. Check deduplication works (some duplicates expected)
    5. Check schema validation (violations table)
    """

    # 1. Produce test records
    print("\n1. Producing test records to Kafka...")
    sent_count = produce_test_records(kafka_producer, 'taxi-nyc-raw', count=1000)
    print(f"✓ Sent {sent_count} records")

    # 2. Wait for processing
    print("\n2. Waiting for Flink to process (30s)...")
    time.sleep(30)

    # 3. Query PostgreSQL for valid records
    print("\n3. Querying PostgreSQL...")
    cursor = postgres_connection.cursor()

    # Count valid records in taxi_trips_raw
    cursor.execute("SELECT COUNT(*) FROM taxi_trips_raw")
    raw_count = cursor.fetchone()[0]
    print(f"✓ taxi_trips_raw: {raw_count} records")

    # Count violations in schema_violations
    cursor.execute("SELECT COUNT(*) FROM schema_violations")
    violation_count = cursor.fetchone()[0]
    print(f"✓ schema_violations: {violation_count} records")

    # Get recent records (last minute)
    cursor.execute("""
        SELECT COUNT(*)
        FROM taxi_trips_raw
        WHERE ingestion_timestamp > NOW() - INTERVAL '1 minute'
    """)
    recent_count = cursor.fetchone()[0]
    print(f"✓ Recent records (last minute): {recent_count}")

    cursor.close()

    # 4. Assertions
    # Allow for some deduplication (expect at least 90% of sent records)
    assert raw_count >= sent_count * 0.9, \
        f"Too few records in PostgreSQL: {raw_count} < {sent_count * 0.9}"

    # Recent records should be non-zero (job is processing)
    assert recent_count > 0, \
        "No recent records - Flink job may not be running"

    print(f"\n✅ E2E test passed!")
    print(f"   Sent: {sent_count}")
    print(f"   Stored: {raw_count}")
    print(f"   Violations: {violation_count}")
    print(f"   Success rate: {raw_count/sent_count*100:.1f}%")


@pytest.mark.integration
def test_deduplication_works(kafka_producer, postgres_connection):
    """Test that deduplication filters duplicate records."""

    cursor = postgres_connection.cursor()

    # Get count before
    cursor.execute("SELECT COUNT(*) FROM taxi_trips_raw")
    count_before = cursor.fetchone()[0]

    # Send duplicate record (same trip_id)
    test_record = {
        'trip_id': 'test_duplicate_123',
        'VendorID': 1,
        'tpep_pickup_datetime': '2024-01-15T10:30:00',
        'tpep_dropoff_datetime': '2024-01-15T10:45:00',
        'PULocationID': 161,
        'DOLocationID': 230,
        'passenger_count': 2,
        'trip_distance': 5.2,
        'fare_amount': 15.5,
        'total_amount': 18.8,
        'payment_type': 1
    }

    # Send twice
    kafka_producer.send('taxi-nyc-raw', value=test_record)
    kafka_producer.send('taxi-nyc-raw', value=test_record)
    kafka_producer.flush()

    # Wait for processing
    time.sleep(10)

    # Check count after
    cursor.execute("SELECT COUNT(*) FROM taxi_trips_raw WHERE trip_id = 'test_duplicate_123'")
    duplicate_count = cursor.fetchone()[0]

    cursor.close()

    # Should only have 1 record (deduplication worked)
    assert duplicate_count <= 1, \
        f"Deduplication failed: found {duplicate_count} copies of same trip_id"

    print(f"✅ Deduplication test passed (found {duplicate_count} record)")


@pytest.mark.integration
def test_schema_validation_rejects_invalid(kafka_producer, postgres_connection):
    """Test that schema validation rejects invalid records."""

    cursor = postgres_connection.cursor()

    # Get violations count before
    cursor.execute("SELECT COUNT(*) FROM schema_violations")
    violations_before = cursor.fetchone()[0]

    # Send invalid record (missing required field)
    invalid_record = {
        'VendorID': 1,
        'tpep_pickup_datetime': '2024-01-15T10:30:00',
        'PULocationID': 161,
        'DOLocationID': 230,
        # Missing: trip_distance, fare_amount (required fields)
        'passenger_count': 2
    }

    kafka_producer.send('taxi-nyc-raw', value=invalid_record)
    kafka_producer.flush()

    # Wait for processing
    time.sleep(10)

    # Check violations after
    cursor.execute("SELECT COUNT(*) FROM schema_violations")
    violations_after = cursor.fetchone()[0]

    cursor.close()

    # Should have more violations
    assert violations_after > violations_before, \
        "Schema validation didn't reject invalid record"

    print(f"✅ Schema validation test passed ({violations_after - violations_before} new violations)")
