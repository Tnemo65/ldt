"""
CA-DQStream Flink Job - Baseline Pipeline.
Spec: Section 3, Lines 1428-1650
"""

from pyflink.datastream import StreamExecutionEnvironment, MapFunction
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
import os
import json

from src.operators.key_generator import generate_trip_id
from src.operators.deduplicator import DeduplicatorFunction
from src.operators.watermark_assigner import create_watermark_strategy

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


def main():
    """Main Flink job entry point."""

    # Environment setup
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(4)  # Match TaskManager slots

    print("="*60)
    print("CA-DQStream Flink Job - Baseline Pipeline")
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
    stream = (
        stream
        .key_by(lambda x: x['trip_id'], key_type=Types.STRING())
        .map(DeduplicatorFunction(), output_type=Types.PICKLED_BYTE_ARRAY())
        .filter(lambda x: x is not None)  # Remove duplicates (None values)
    )

    # Debug output
    stream.print()

    # Execute
    print("\nStarting Flink job...")
    env.execute("CA-DQStream Baseline Pipeline")

if __name__ == "__main__":
    main()
