"""
PostgreSQL JDBC sinks for CA-DQStream.
Spec: Lines 1595-1620 (JDBC connector with PgBouncer)
"""

from pyflink.datastream.connectors.jdbc import JdbcSink, JdbcConnectionOptions, JdbcExecutionOptions
from pyflink.common.typeinfo import Types
import os
import json


def create_raw_trips_sink():
    """Create JDBC sink for taxi_trips_raw table.

    V1.9 Spec:
    - Connects via PgBouncer (port 6432) for connection pooling
    - Batch size: 100 for optimal throughput
    - Max retries: 3 with 1s delay
    """

    # Connection through PgBouncer for pooling
    pgbouncer_host = os.getenv('PGBOUNCER_HOST', 'localhost')
    pgbouncer_port = os.getenv('PGBOUNCER_PORT', '6432')
    db_name = os.getenv('POSTGRES_DB', 'dq_pipeline')
    db_user = os.getenv('POSTGRES_USER', 'cadqstream')
    db_password = os.getenv('POSTGRES_PASSWORD', 'cadqstream123')

    conn_opts = (
        JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
        .with_url(f"jdbc:postgresql://{pgbouncer_host}:{pgbouncer_port}/{db_name}")
        .with_driver_name("org.postgresql.Driver")
        .with_user_name(db_user)
        .with_password(db_password)
        .build()
    )

    # Execution options for batching and retries
    exec_opts = (
        JdbcExecutionOptions.builder()
        .with_batch_size(100)
        .with_batch_interval_ms(5000)
        .with_max_retries(3)
        .build()
    )

    # INSERT statement matching schema.sql
    insert_sql = """
        INSERT INTO taxi_trips_raw (
            trip_id, vendor_id, pickup_datetime, dropoff_datetime,
            passenger_count, trip_distance, pickup_location_id, dropoff_location_id,
            payment_type, fare_amount, total_amount
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (trip_id) DO NOTHING
    """

    # Type information for JDBC
    type_info = Types.ROW([
        Types.STRING(),  # trip_id
        Types.INT(),  # vendor_id
        Types.STRING(),  # pickup_datetime (converted to TIMESTAMP by JDBC)
        Types.STRING(),  # dropoff_datetime
        Types.INT(),  # passenger_count
        Types.DOUBLE(),  # trip_distance
        Types.INT(),  # pickup_location_id
        Types.INT(),  # dropoff_location_id
        Types.INT(),  # payment_type
        Types.DOUBLE(),  # fare_amount
        Types.DOUBLE()  # total_amount
    ])

    sink = JdbcSink.sink(
        insert_sql,
        type_info,
        conn_opts,
        exec_opts
    )

    return sink


def create_violations_sink():
    """Create JDBC sink for schema_violations table."""

    pgbouncer_host = os.getenv('PGBOUNCER_HOST', 'localhost')
    pgbouncer_port = os.getenv('PGBOUNCER_PORT', '6432')
    db_name = os.getenv('POSTGRES_DB', 'dq_pipeline')
    db_user = os.getenv('POSTGRES_USER', 'cadqstream')
    db_password = os.getenv('POSTGRES_PASSWORD', 'cadqstream123')

    conn_opts = (
        JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
        .with_url(f"jdbc:postgresql://{pgbouncer_host}:{pgbouncer_port}/{db_name}")
        .with_driver_name("org.postgresql.Driver")
        .with_user_name(db_user)
        .with_password(db_password)
        .build()
    )

    exec_opts = (
        JdbcExecutionOptions.builder()
        .with_batch_size(100)
        .with_batch_interval_ms(5000)
        .with_max_retries(3)
        .build()
    )

    insert_sql = """
        INSERT INTO schema_violations (
            trip_id, violation_type, violation_reason
        ) VALUES (?, ?, ?)
    """

    type_info = Types.ROW([
        Types.STRING(),  # trip_id
        Types.STRING(),  # violation_type
        Types.STRING()  # violation_reason
    ])

    sink = JdbcSink.sink(
        insert_sql,
        type_info,
        conn_opts,
        exec_opts
    )

    return sink


def record_to_raw_trips_row(record: dict) -> tuple:
    """Convert record dict to tuple for taxi_trips_raw insertion.

    Args:
        record: Validated taxi trip record

    Returns:
        Tuple of 11 fields matching INSERT statement order
    """
    return (
        record.get('trip_id', ''),
        int(record.get('VendorID', 0)),
        record.get('tpep_pickup_datetime', ''),
        record.get('tpep_dropoff_datetime', ''),
        int(record.get('passenger_count', 0)),
        float(record.get('trip_distance', 0.0)),
        int(record.get('PULocationID', 0)),
        int(record.get('DOLocationID', 0)),
        int(record.get('payment_type', 0)),
        float(record.get('fare_amount', 0.0)),
        float(record.get('total_amount', 0.0))
    )


def record_to_violation_row(record: dict, violation_type: str) -> tuple:
    """Convert invalid record to violation row.

    Args:
        record: Invalid record that failed validation
        violation_type: Type of violation (e.g., 'INVALID_ZONE', 'MISSING_FIELD')

    Returns:
        Tuple of 3 fields: (trip_id, violation_type, violation_reason)
    """
    trip_id = record.get('trip_id', 'UNKNOWN')

    # Generate reason based on record
    reasons = []
    if record.get('PULocationID', 0) < 1 or record.get('PULocationID', 0) > 263:
        reasons.append(f"Invalid PULocationID: {record.get('PULocationID')}")
    if record.get('DOLocationID', 0) < 1 or record.get('DOLocationID', 0) > 263:
        reasons.append(f"Invalid DOLocationID: {record.get('DOLocationID')}")
    if record.get('fare_amount') is None:
        reasons.append("Missing fare_amount")
    if record.get('trip_distance') is None:
        reasons.append("Missing trip_distance")

    violation_reason = '; '.join(reasons) if reasons else f"Violation: {violation_type}"

    return (trip_id, violation_type, violation_reason)
