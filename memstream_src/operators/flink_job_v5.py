"""
CA-DQStream + MemStream v5 - Complete Pipeline.

Complete Flink job orchestrating all 4 layers:
- Layer 1: Baseline Validation (Parse, Dedup, Schema, Watermark)
- Layer 2: Dual-Branch (Canary + MemStream ML)
- Layer 3: Voting Ensemble + MetaAggregator
- Layer 4: IEC Feedback

Reference: original_flow.md for full architecture details.
"""

import logging
import os
from typing import Optional, Dict

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer
from pyflink.common.serialization import JsonDeserializationSchema
from pyflink.common.time import Time
from pyflink.common.typeinfo import Types
from pyflink.common.watermark_strategy import Duration

from layer1 import (
    ParseJsonFunction,
    AddTripIdFunction,
    DeduplicatorFunction,
    SchemaValidator,
    create_watermark_strategy,
)
from layer2 import (
    CanaryRulesOperator,
    ViolationFilter,
    CleanRecordFilter,
)
from memstream_scoring_op import MemStreamScoringOperator
from layer3 import (
    VotingEnsembleFunction,
    MetaAggregatorProcessWindowFunction,
    MetaAggregatorFactory,
)
from iec_feedback_op import IECFeedbackOperator
from sinks.kafka_sinks import (
    get_kafka_sink,
    get_processed_sink,
    get_hard_rule_violations_sink,
    get_anomalies_sink,
    get_meta_stream_sink,
    get_clean_stream_sink,
    TAXI_NYC_RAW_TOPIC,
    DQ_STREAM_PROCESSED_TOPIC,
    DQ_HARD_RULE_VIOLATIONS_TOPIC,
    DQ_STREAM_ANOMALIES_TOPIC,
    DQ_META_STREAM_TOPIC,
    DQ_STREAM_PROCESSED_CLEAN_TOPIC,
)

LOGGER = logging.getLogger('cadqstream.v5')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


class FlinkJobV5Config:
    """Configuration for complete v5 pipeline."""
    
    def __init__(self):
        self.parallelism = int(os.getenv('FLINK_PARALLELISM', '4'))
        self.kafka_bootstrap_servers = os.getenv(
            'KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'
        )
        self.kafka_consumer_group = os.getenv(
            'KAFKA_CONSUMER_GROUP', 'cadqstream-v5'
        )
        
        self.dedup_ttl_days = int(os.getenv('DEDUP_TTL_DAYS', '7'))
        self.watermark_bounded_lateness_ms = int(
            os.getenv('WATERMARK_BOUNDED_LATENESS_MS', '10000')
        )
        
        self.meta_window_minutes = int(os.getenv('META_WINDOW_MINUTES', '5'))
        
        self.checkpoint_interval_ms = int(
            os.getenv('CHECKPOINT_INTERVAL_MS', '60000')
        )
        
        self.enable_layer1 = os.getenv('ENABLE_LAYER1', 'true').lower() == 'true'
        self.enable_layer2 = os.getenv('ENABLE_LAYER2', 'true').lower() == 'true'
        self.enable_layer3 = os.getenv('ENABLE_LAYER3', 'true').lower() == 'true'
        self.enable_layer4 = os.getenv('ENABLE_LAYER4', 'false').lower() == 'true'


def create_complete_pipeline(
    env: StreamExecutionEnvironment,
    config: FlinkJobV5Config = None
):
    """
    Create complete CA-DQStream v5 pipeline.
    
    Pipeline:
        kafka(taxi-nyc-raw)
            ↓
        [Layer 1: Baseline Validation]
            → ParseJson → Watermark → key_by(trip_id) → Dedup → Schema
            ↓ valid                          ↓ invalid
            kafka(dq-stream-processed)   kafka(dq-hard-rule-violations)
            ↓
        [Layer 2: Dual-Branch]
            → Split: canary | complex
                ↓                    ↓
            CanaryRules       MemStreamScoring
                ↓                    ↓
            kafka(dq-stream-anomalies) ← Union
            ↓
        [Layer 3: Voting Ensemble]
            → key_by(neighborhood) → Window(5min) → MetaAggregator
            ↓
        kafka(dq-meta-stream)
            ↓
        [Layer 4: IEC Feedback]
            → IECFeedbackOperator
            ↓
        kafka(iec-action-replay)
    
    Args:
        env: Flink StreamExecutionEnvironment
        config: Pipeline configuration
        
    Returns:
        Configured job graph
    """
    if config is None:
        config = FlinkJobV5Config()
    
    LOGGER.info(
        "[V5Pipeline] Starting complete pipeline: L1=%s, L2=%s, L3=%s, L4=%s",
        config.enable_layer1, config.enable_layer2,
        config.enable_layer3, config.enable_layer4
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
    
    raw_source = KafkaSource.builder() \
        .set_bootstrap_servers(config.kafka_bootstrap_servers) \
        .set_topics(TAXI_NYC_RAW_TOPIC) \
        .set_consumer_group(config.kafka_consumer_group) \
        .set_starting_offsets(KafkaOffsetsInitializer.committed_offsets()) \
        .set_deserialization_schema(JsonDeserializationSchema.builder()
            .type_info_hint(Types.PICKLED_BYTE_ARRAY)
            .build()) \
        .set_properties(kafka_props) \
        .build()
    
    raw_stream = env.from_source(
        source=raw_source,
        watermark_strategy=create_watermark_strategy(
            bounded_out_of_orderness_ms=config.watermark_bounded_lateness_ms
        ),
        source_name="Kafka-taxi-nyc-raw"
    )
    
    layer1_stream = _build_layer1(
        raw_stream, config,
        env
    )
    
    layer2_stream = _build_layer2(
        layer1_stream, config,
        env
    )
    
    layer3_stream = _build_layer3(
        layer2_stream, config,
        env
    )
    
    if config.enable_layer4:
        _build_layer4(
            layer3_stream, config,
            env
        )
    else:
        layer3_stream.sink_to(
            get_meta_stream_sink(config.kafka_bootstrap_servers),
            name="Sink-MetaStream"
        )
    
    LOGGER.info("[V5Pipeline] Complete pipeline created successfully")
    
    return env


def _build_layer1(
    raw_stream,
    config: FlinkJobV5Config,
    env: StreamExecutionEnvironment
):
    """Build Layer 1: Baseline Validation."""
    LOGGER.info("[V5Pipeline] Building Layer 1: Baseline Validation")
    
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
    violation_tag = OutputTag(
        DQ_HARD_RULE_VIOLATIONS_TOPIC,
        types=Types.PICKLED_BYTE_ARRAY
    )
    
    valid_tag = OutputTag("valid", types=Types.PICKLED_BYTE_ARRAY)
    
    split_stream = dedup_stream.process(
        SchemaValidator(),
        name="SchemaValidator"
    )
    
    valid_stream = split_stream.get_side_output(valid_tag)
    violation_stream = split_stream.get_side_output(violation_tag)
    
    valid_stream.sink_to(
        get_processed_sink(config.kafka_bootstrap_servers),
        name="Sink-ValidRecords"
    )
    
    violation_stream.sink_to(
        get_hard_rule_violations_sink(config.kafka_bootstrap_servers),
        name="Sink-Violations"
    )
    
    LOGGER.info("[V5Pipeline] Layer 1 completed")
    
    return valid_stream


def _build_layer2(
    valid_stream,
    config: FlinkJobV5Config,
    env: StreamExecutionEnvironment
):
    """Build Layer 2: Dual-Branch Processing."""
    LOGGER.info("[V5Pipeline] Building Layer 2: Dual-Branch")
    
    neighborhood_stream = valid_stream \
        .key_by(lambda r: r.get('PULocationID', 0), name="KeyByPULocationID")
    
    canary_stream = neighborhood_stream \
        .map(CanaryRulesOperator(), name="CanaryRules")
    
    complex_stream = neighborhood_stream \
        .process(MemStreamScoringOperator(), name="MemStreamScoring")
    
    canary_anomalies = canary_stream \
        .filter(lambda r: r.get('has_violation', False), name="FilterCanaryViolations")
    
    union_stream = canary_anomalies.union(complex_stream)
    
    union_stream.sink_to(
        get_anomalies_sink(config.kafka_bootstrap_servers),
        name="Sink-Anomalies"
    )
    
    canary_clean = canary_stream \
        .filter(lambda r: not r.get('has_violation', False), name="FilterCanaryClean")
    
    canary_clean.sink_to(
        get_clean_stream_sink(config.kafka_bootstrap_servers),
        name="Sink-CanaryClean"
    )
    
    LOGGER.info("[V5Pipeline] Layer 2 completed")
    
    return union_stream


def _build_layer3(
    anomalies_stream,
    config: FlinkJobV5Config,
    env: StreamExecutionEnvironment
):
    """Build Layer 3: Voting Ensemble + MetaAggregator."""
    LOGGER.info("[V5Pipeline] Building Layer 3: Voting + MetaAggregator")
    
    voting_stream = anomalies_stream \
        .key_by(lambda r: r.get('trip_id', ''), name="KeyByTripId") \
        .process(VotingEnsembleFunction(), name="VotingEnsemble")
    
    neighborhood_stream = voting_stream \
        .key_by(
            lambda r: r.get('neighborhood', r.get('context_key', 'unknown')),
            name="KeyByNeighborhood"
        )
    
    windowed_stream = neighborhood_stream \
        .window(TumblingEventTimeWindows.of(Time.minutes(config.meta_window_minutes)))
    
    meta_stream = windowed_stream.process(
        MetaAggregatorFactory.create(config.meta_window_minutes),
        name="MetaAggregator"
    )
    
    LOGGER.info("[V5Pipeline] Layer 3 completed")
    
    return meta_stream


def _build_layer4(
    meta_stream,
    config: FlinkJobV5Config,
    env: StreamExecutionEnvironment
):
    """Build Layer 4: IEC Feedback."""
    LOGGER.info("[V5Pipeline] Building Layer 4: IEC Feedback")
    
    iec_stream = meta_stream \
        .key_by(lambda r: r.get('neighborhood', 'global'), name="KeyByNeighborhood") \
        .process(IECFeedbackOperator(), name="IECFeedback")
    
    iec_stream.sink_to(
        get_kafka_sink('iec-action-replay', config.kafka_bootstrap_servers),
        name="Sink-IECActions"
    )
    
    LOGGER.info("[V5Pipeline] Layer 4 completed")
    
    return iec_stream


def main():
    """Main entry point for v5 pipeline."""
    LOGGER.info("[V5Pipeline] Starting CA-DQStream v5 Complete Pipeline...")
    
    config = FlinkJobV5Config()
    
    env = StreamExecutionEnvironment.get_execution_environment()
    
    env.set_parallelism(config.parallelism)
    
    _configure_state_backend(env, config)
    
    create_complete_pipeline(env, config)
    
    LOGGER.info("[V5Pipeline] Executing job...")
    env.execute("CA-DQStream v5 - Complete Pipeline")


def _configure_state_backend(env, config):
    """Configure state backend."""
    state_backend_type = os.getenv('STATE_BACKEND', 'rocksdb')
    
    if state_backend_type == 'rocksdb':
        try:
            from pyflink.state_backend import RocksDBStateBackend
            checkpoint_path = os.getenv(
                'ROCKSDB_CHECKPOINT_PATH',
                'file:///tmp/cadqstream-rocksdb'
            )
            env.set_state_backend(RocksDBStateBackend(checkpoint_path))
            LOGGER.info("[V5Pipeline] Using RocksDB state backend: %s", checkpoint_path)
        except ImportError:
            LOGGER.warning("[V5Pipeline] RocksDB not available, using default")
    elif state_backend_type == 'hashmap':
        from pyflink.state_backend import HashMapStateBackend
        env.set_state_backend(HashMapStateBackend())
        LOGGER.info("[V5Pipeline] Using HashMap state backend")


if __name__ == '__main__':
    main()
