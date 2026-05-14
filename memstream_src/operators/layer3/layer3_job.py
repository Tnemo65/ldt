"""
Layer 3 Flink Job - Voting Ensemble + MetaAggregator.

Standalone Flink job for testing Layer 3 operators.
Pipeline:
    - kafka(dq-stream-anomalies) → VotingEnsemble
    - kafka(Layer 2 complex output) → VotingEnsemble
    - Union → VotingEnsemble → MetaAggregator (5-min window)
    - Sink → kafka(dq-meta-stream)

Reference: original_flow.md lines 570-598
"""

import logging
import os
from typing import Optional

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer
from pyflink.datastream.connectors.kafka.config import ConsumerConfig
from pyflink.common.serialization import JsonDeserializationSchema
from pyflink.common.time import Time
from pyflink.common.typeinfo import Types

from layer3 import (
    VotingEnsembleFunction,
    MetaAggregatorProcessWindowFunction,
    MetaAggregatorFactory,
)
from sinks.kafka_sinks import (
    get_kafka_sink,
    get_meta_stream_sink,
    DQ_STREAM_ANOMALIES_TOPIC,
    DQ_META_STREAM_TOPIC,
)

LOGGER = logging.getLogger('cadqstream.layer3_job')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


class Layer3Config:
    """Configuration for Layer 3 job."""
    
    def __init__(self):
        self.parallelism = int(os.getenv('LAYER3_PARALLELISM', '4'))
        self.kafka_bootstrap_servers = os.getenv(
            'KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'
        )
        self.kafka_consumer_group = os.getenv(
            'KAFKA_CONSUMER_GROUP', 'cadqstream-layer3'
        )
        self.window_size_minutes = int(os.getenv('META_WINDOW_MINUTES', '5'))
        self.checkpoint_interval_ms = int(
            os.getenv('CHECKPOINT_INTERVAL_MS', '60000')
        )


def create_layer3_job(env: StreamExecutionEnvironment, config: Layer3Config = None):
    """
    Create Layer 3 Flink job.
    
    Pipeline:
        kafka(dq-stream-anomalies)  ─┐
                                    ├─ Union
        kafka(dq-stream-complex)   ─┘
                ↓
        VotingEnsembleFunction
                ↓
        key_by(neighborhood)
                ↓
        TumblingWindow(5 min)
                ↓
        MetaAggregator
                ↓
        kafka(dq-meta-stream)
    
    Args:
        env: Flink StreamExecutionEnvironment
        config: Layer 3 configuration
        
    Returns:
        Configured job graph
    """
    if config is None:
        config = Layer3Config()
    
    LOGGER.info(
        "[Layer3Job] Starting with config: parallelism=%d, window=%d min",
        config.parallelism, config.window_size_minutes
    )
    
    env.set_parallelism(config.parallelism)
    
    if config.checkpoint_interval_ms > 0:
        env.enable_checkpointing(config.checkpoint_interval_ms)
        env.get_checkpoint_config().set_min_pause_between_checkpoints(
            config.checkpoint_interval_ms // 2
        )
    
    kafka_props = {
        'bootstrap.servers': config.kafka_bootstrap_servers,
        'group.id': config.kafka_consumer_group,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': 'false',
    }
    
    canary_source = _create_kafka_source(
        config,
        DQ_STREAM_ANOMALIES_TOPIC,
        "kafka-canary-source"
    )
    
    complex_source = _create_kafka_source(
        config,
        "dq-stream-complex",
        "kafka-complex-source"
    )
    
    canary_stream = env.from_source(
        source=canary_source,
        watermark_strategy=None,
        source_name="Kafka-dq-stream-anomalies"
    )
    
    complex_stream = env.from_source(
        source=complex_source,
        watermark_strategy=None,
        source_name="Kafka-dq-stream-complex"
    )
    
    union_stream = canary_stream.union(complex_stream)
    
    voting_stream = union_stream \
        .key_by(lambda r: r.get('trip_id', ''), name="KeyByTripId") \
        .process(VotingEnsembleFunction(), name="VotingEnsemble")
    
    neighborhood_keyed_stream = voting_stream \
        .key_by(lambda r: r.get('neighborhood', r.get('context_key', 'unknown')), 
                name="KeyByNeighborhood")
    
    windowed_stream = neighborhood_keyed_stream \
        .window(TumblingEventTimeWindows.of(Time.minutes(config.window_size_minutes)))
    
    meta_stream = windowed_stream.process(
        MetaAggregatorFactory.create(config.window_size_minutes),
        name="MetaAggregator"
    )
    
    meta_stream.sink_to(
        get_meta_stream_sink(config.kafka_bootstrap_servers),
        name="Sink-MetaStream"
    )
    
    LOGGER.info("[Layer3Job] Pipeline created successfully")
    
    return env


def _create_kafka_source(config: Layer3Config, topic: str, name: str):
    """Create Kafka source for given topic."""
    return KafkaSource.builder() \
        .set_bootstrap_servers(config.kafka_bootstrap_servers) \
        .set_topics(topic) \
        .set_consumer_group(config.kafka_consumer_group) \
        .set_starting_offsets(KafkaOffsetsInitializer.committed_offsets()) \
        .set_deserialization_schema(JsonDeserializationSchema.builder()
            .type_info_hint(Types.PICKLED_BYTE_ARRAY)
            .build()) \
        .set_properties({
            'bootstrap.servers': config.kafka_bootstrap_servers,
            'group.id': config.kafka_consumer_group,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': 'false',
        }) \
        .build()


def main():
    """Main entry point for Layer 3 job."""
    LOGGER.info("[Layer3Job] Initializing Layer 3 Flink job...")
    
    config = Layer3Config()
    
    env = StreamExecutionEnvironment.get_execution_environment()
    
    env.set_parallelism(config.parallelism)
    
    create_layer3_job(env, config)
    
    LOGGER.info("[Layer3Job] Executing job...")
    env.execute("CA-DQStream Layer 3 - Voting Ensemble + MetaAggregator")


if __name__ == '__main__':
    main()
