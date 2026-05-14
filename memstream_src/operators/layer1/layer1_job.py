"""
Layer 1 Flink Job - Baseline Validation Pipeline.

Standalone Flink job for testing Layer 1 operators.
Pipeline: taxi-nyc-raw → ParseJson → Watermark → key_by(trip_id) → Dedup → Schema

Reference: original_flow.md lines 374-422
"""

import logging
import os
from typing import Optional

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer
from pyflink.datastream.connectors.kafka.config import ConsumerConfig
from pyflink.common.serialization import JsonDeserializationSchema
from pyflink.common import SideOutput
from pyflink.common.watermark_strategy import Duration

from layer1 import (
    ParseJsonFunction,
    AddTripIdFunction,
    DeduplicatorFunction,
    SchemaValidator,
    SideOutputTag,
    create_watermark_strategy,
)
from sinks.kafka_sinks import (
    get_kafka_sink,
    TAXI_NYC_RAW_TOPIC,
    DQ_STREAM_PROCESSED_TOPIC,
    DQ_HARD_RULE_VIOLATIONS_TOPIC,
)

LOGGER = logging.getLogger('cadqstream.layer1_job')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


class Layer1Config:
    """Configuration for Layer 1 job."""
    
    def __init__(self):
        self.parallelism = int(os.getenv('LAYER1_PARALLELISM', '4'))
        self.kafka_bootstrap_servers = os.getenv(
            'KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'
        )
        self.kafka_consumer_group = os.getenv(
            'KAFKA_CONSUMER_GROUP', 'cadqstream-layer1'
        )
        self.dedup_ttl_days = int(os.getenv('DEDUP_TTL_DAYS', '7'))
        self.watermark_bounded_lateness_ms = int(
            os.getenv('WATERMARK_BOUNDED_LATENESS_MS', '10000')
        )
        self.checkpoint_interval_ms = int(
            os.getenv('CHECKPOINT_INTERVAL_MS', '60000')
        )
        self.state_backend = os.getenv(
            'STATE_BACKEND', 'rocksdb'
        )


def create_layer1_job(env: StreamExecutionEnvironment, config: Layer1Config = None):
    """
    Create Layer 1 Flink job.
    
    Pipeline:
        kafka(taxi-nyc-raw)
          → ParseJsonFunction
          → WatermarkAssigner (event time from tpep_pickup_datetime)
          → key_by(trip_id)
          → DeduplicatorFunction (7-day TTL)
          → SchemaValidator
          → Split: valid_stream | violation_stream
          → valid_stream → kafka(dq-stream-processed)
          → violation_stream → kafka(dq-hard-rule-violations)
    
    Args:
        env: Flink StreamExecutionEnvironment
        config: Layer 1 configuration
        
    Returns:
        Configured job graph
    """
    if config is None:
        config = Layer1Config()
    
    LOGGER.info("[Layer1Job] Starting with config: parallelism=%d", config.parallelism)
    
    env.set_parallelism(config.parallelism)
    
    if config.checkpoint_interval_ms > 0:
        env.enable_checkpointing(config.checkpoint_interval_ms)
        env.get_checkpoint_config().set_min_pause_between_checkpoints(
            config.checkpoint_interval_ms // 2
        )
    
    kafka_source = _create_kafka_source(config)
    
    raw_stream = env.from_source(
        source=kafka_source,
        watermark_strategy=create_watermark_strategy(
            bounded_out_of_orderness_ms=config.watermark_bounded_lateness_ms
        ),
        source_name="Kafka-taxi-nyc-raw"
    )
    
    parsed_stream = raw_stream \
        .map(ParseJsonFunction(), name="ParseJson") \
        .filter(lambda r: r is not None, name="FilterInvalidJson")
    
    trip_id_stream = parsed_stream \
        .map(AddTripIdFunction(), name="AddTripId")
    
    keyed_stream = trip_id_stream \
        .key_by(lambda r: r.get('trip_id', ''), name="KeyByTripId")
    
    dedup_stream = keyed_stream \
        .process(
            DeduplicatorFunction(ttl_days=config.dedup_ttl_days),
            name="Deduplicator"
        ) \
        .filter(lambda r: r is not None, name="FilterDuplicates")
    
    from pyflink.datastream import OutputTag
    
    violation_tag = OutputTag(DQ_HARD_RULE_VIOLATIONS_TOPIC, types=Types.PICKLED_BYTE_ARRAY)
    valid_tag = OutputTag("valid", types=Types.PICKLED_BYTE_ARRAY)
    
    split_stream = dedup_stream.process(
        SchemaValidator(),
        name="SchemaValidator"
    )
    
    valid_stream = split_stream.get_side_output(valid_tag) \
        .map(lambda r: r, name="ExtractValid")
    
    violation_stream = split_stream.get_side_output(violation_tag) \
        .map(lambda r: r, name="ExtractViolations")
    
    valid_stream.sink_to(
        get_kafka_sink(DQ_STREAM_PROCESSED_TOPIC, config.kafka_bootstrap_servers),
        name="Sink-ValidRecords"
    )
    
    violation_stream.sink_to(
        get_kafka_sink(DQ_HARD_RULE_VIOLATIONS_TOPIC, config.kafka_bootstrap_servers),
        name="Sink-Violations"
    )
    
    LOGGER.info("[Layer1Job] Pipeline created successfully")
    
    return env


def _create_kafka_source(config: Layer1Config):
    """Create Kafka source for taxi-nyc-raw topic."""
    consumer_props = {
        'bootstrap.servers': config.kafka_bootstrap_servers,
        'group.id': config.kafka_consumer_group,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': 'false',
    }
    
    return KafkaSource.builder() \
        .set_bootstrap_servers(config.kafka_bootstrap_servers) \
        .set_topics(TAXI_NYC_RAW_TOPIC) \
        .set_consumer_group(config.kafka_consumer_group) \
        .set_starting_offsets(KafkaOffsetsInitializer.committed_offsets()) \
        .set_deserialization_schema(JsonDeserializationSchema.builder()
            .type_info_hint(Types.PICKLED_BYTE_ARRAY)
            .build()) \
        .set_properties(consumer_props) \
        .build()


def main():
    """Main entry point for Layer 1 job."""
    LOGGER.info("[Layer1Job] Initializing Layer 1 Flink job...")
    
    config = Layer1Config()
    
    env = StreamExecutionEnvironment.get_execution_environment()
    
    env.set_parallelism(config.parallelism)
    
    if config.state_backend == 'rocksdb':
        try:
            from pyflink.state_backend import RocksDBStateBackend
            rocksdb_path = os.getenv(
                'ROCKSDB_CHECKPOINT_PATH',
                'file:///tmp/cadqstream-rocksdb'
            )
            env.set_state_backend(RocksDBStateBackend(rocksdb_path))
            LOGGER.info("[Layer1Job] Using RocksDB state backend: %s", rocksdb_path)
        except ImportError:
            LOGGER.warning("[Layer1Job] RocksDB not available, using default state backend")
    elif config.state_backend == 'hashmap':
        from pyflink.state_backend import HashMapStateBackend
        env.set_state_backend(HashMapStateBackend())
    
    create_layer1_job(env, config)
    
    LOGGER.info("[Layer1Job] Executing job...")
    env.execute("CA-DQStream Layer 1 - Baseline Validation")


if __name__ == '__main__':
    main()
