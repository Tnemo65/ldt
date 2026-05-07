"""
CA-DQStream Flink Job - Baseline Pipeline.
Spec: Section 3, Lines 1428-1650
"""

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
import os

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

    # Placeholder: Will add Layer 1 operators in next tasks
    stream.print()

    # Execute
    print("\nStarting Flink job...")
    env.execute("CA-DQStream Baseline Pipeline")

if __name__ == "__main__":
    main()
