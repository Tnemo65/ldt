"""
CA-DQStream Flink Job - Baseline Pipeline.
Spec: Section 3, Lines 1428-1650
"""

from pyflink.datastream import StreamExecutionEnvironment, MapFunction, CheckpointingMode
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
import os
import json

from src.operators.key_generator import generate_trip_id
from src.operators.deduplicator import DeduplicatorFunction
from src.operators.watermark_assigner import create_watermark_strategy
from src.operators.schema_validator import SchemaValidator
from src.sinks.postgres_sink import (
    create_raw_trips_sink,
    create_violations_sink,
    record_to_raw_trips_row,
    record_to_violation_row
)

def create_kafka_source(env, topic: str):
    """Create Kafka source with Avro deserialization."""

    properties = {
        'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        'group.id': 'cadqstream-flink-consumer',
        'auto.offset.reset': 'earliest',
    }

    kafka_source = FlinkKafkaConsumer(
        topics=topic,
        deserialization_schema=SimpleStringSchema(),
        properties=properties
    )

    return kafka_source

class ParseJsonFunction(MapFunction):
    """Parse JSON string to dict."""
    def map(self, value):
        try:
            return json.loads(value)
        except:
            return None


class AddTripIdFunction(MapFunction):
    """Add trip_id to record using MurmurHash3."""
    def map(self, record):
        if record is None:
            return None
        record['trip_id'] = generate_trip_id(record)
        return record


class RecordToRawTripsRowFunction(MapFunction):
    """Convert record dict to tuple for JDBC insertion."""
    def map(self, record):
        if record is None:
            return None
        return record_to_raw_trips_row(record)


class RecordToViolationRowFunction(MapFunction):
    """Convert invalid record to violation tuple."""
    def map(self, record):
        if record is None:
            return None
        return record_to_violation_row(record, 'SCHEMA_VIOLATION')


def main():
    """Main Flink job entry point."""

    # Environment setup
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(4)  # Match TaskManager slots

    # Checkpointing configuration (V1.9)
    # Note: Requires MinIO running and accessible
    checkpoint_config = env.get_checkpoint_config()
    checkpoint_config.set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    checkpoint_config.set_checkpoint_interval(45000)  # 45 seconds
    checkpoint_config.set_min_pause_between_checkpoints(30000)  # 30s minimum pause
    checkpoint_config.set_checkpoint_timeout(300000)  # 5 min timeout
    checkpoint_config.set_max_concurrent_checkpoints(1)

    # Enable checkpoint on job cancellation
    checkpoint_config.enable_externalized_checkpoints(
        checkpoint_config.ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
    )

    # State backend: RocksDB with S3 (MinIO)
    # Note: Requires flink-s3-fs-hadoop JAR and proper S3 credentials
    # Uncomment when running with MinIO:
    # env.set_state_backend(RocksDBStateBackend("s3://cadqstream-state/", True))
    # checkpoint_config.set_checkpoint_storage("s3://cadqstream-checkpoints/")

    print("="*60)
    print("CA-DQStream Flink Job - Baseline Pipeline")
    print("="*60)
    print(f"Checkpointing: EXACTLY_ONCE, interval=45s")
    print(f"State backend: RocksDB (local for now, MinIO when enabled)")
    print("="*60)

    # Kafka source
    kafka_source = create_kafka_source(env, 'taxi-nyc-raw')
    stream = env.add_source(kafka_source)

    # Layer 1: Parse JSON and assign watermarks
    stream = (
        stream
        .map(ParseJsonFunction(), output_type=Types.PICKLED_BYTE_ARRAY())
        .filter(lambda x: x is not None)
        .assign_timestamps_and_watermarks(create_watermark_strategy())
    )

    # Layer 1: Generate trip_id (surrogate key)
    stream = stream.map(AddTripIdFunction(), output_type=Types.PICKLED_BYTE_ARRAY())

    # Layer 1: Deduplication (keyed by trip_id)
    deduplicated_stream = (
        stream
        .key_by(lambda x: x['trip_id'], key_type=Types.STRING())
        .map(DeduplicatorFunction(), output_type=Types.PICKLED_BYTE_ARRAY())
        .filter(lambda x: x is not None)  # Remove duplicates (None values)
    )

    # Layer 1: Schema validation - split into valid/violations
    validator = SchemaValidator()

    # Valid records (pass filter)
    valid_stream = deduplicated_stream.filter(validator)

    # Violations (fail filter) - need to invert the filter
    violation_stream = deduplicated_stream.filter(lambda x: not validator.filter(x))

    # PostgreSQL Sinks
    # Note: JDBC sinks require Flink runtime with JDBC connector JAR
    # For now, we'll print to console. Uncomment below when running with proper Flink setup
    # valid_stream.map(RecordToRawTripsRowFunction()).add_sink(create_raw_trips_sink())
    # violation_stream.map(RecordToViolationRowFunction()).add_sink(create_violations_sink())

    # Debug output
    print("\n[Valid records]")
    valid_stream.print()

    print("\n[Schema violations]")
    violation_stream.print()

    # Execute
    print("\nStarting Flink job...")
    env.execute("CA-DQStream Baseline Pipeline")

if __name__ == "__main__":
    main()
