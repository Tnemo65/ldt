# =============================================================================
# Kafka Sinks — CA-DQStream + MemStream
# =============================================================================
#
# Exactly-once sinks for all pipeline Kafka topics using FlinkKafkaProducer.
# Serialization: JSON via JsonSerializationSchema
# Compression: LZ4
# Delivery guarantee: EXACTLY_ONCE (transactional)
#
# Topic reference: original_flow.md lines 284-293
# =============================================================================

from __future__ import annotations

import os
import logging
from typing import Dict, Optional, Any

from pyflink.common.serialization import JsonSerializationSchema
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.kafka import FlinkKafkaProducer

LOGGER = logging.getLogger('cadqstream.sinks')


# =============================================================================
# Topic Constants
# =============================================================================

TAXI_NYC_RAW_TOPIC = 'taxi-nyc-raw'
DQ_HARD_RULE_VIOLATIONS_TOPIC = 'dq-hard-rule-violations'
DQ_STREAM_PROCESSED_TOPIC = 'dq-stream-processed'
DQ_STREAM_ANOMALIES_TOPIC = 'dq-stream-anomalies'
DQ_META_STREAM_TOPIC = 'dq-meta-stream'
DQ_STREAM_PROCESSED_CLEAN_TOPIC = 'dq-stream-processed-clean'
IEC_ACTION_REPLAY_TOPIC = 'iec-action-replay'
IF_MODEL_UPDATES_TOPIC = 'if-model-updates'


# Topic metadata for documentation and health checks
TOPICS: Dict[str, Dict[str, Any]] = {
    'taxi-nyc-raw': {
        'partitions': 4,
        'replication': 1,
        'retention_days': 7,
        'cleanup': 'delete',
        'description': 'NYC Yellow Taxi raw JSON records',
    },
    'dq-hard-rule-violations': {
        'partitions': 4,
        'replication': 1,
        'retention_days': 30,
        'cleanup': 'compact',
        'description': 'L1 schema rule violations',
    },
    'dq-stream-processed': {
        'partitions': 4,
        'replication': 1,
        'retention_days': 7,
        'cleanup': 'delete',
        'description': 'L1 valid enriched records',
    },
    'dq-stream-anomalies': {
        'partitions': 4,
        'replication': 1,
        'retention_days': 30,
        'cleanup': 'compact',
        'description': 'Anomaly scores (canary + ML)',
    },
    'dq-meta-stream': {
        'partitions': 4,
        'replication': 1,
        'retention_days': 7,
        'cleanup': 'delete',
        'description': 'Voting results and meta-metrics',
    },
    'dq-stream-processed-clean': {
        'partitions': 4,
        'replication': 1,
        'retention_days': 7,
        'cleanup': 'delete',
        'description': 'Clean records (canary passed)',
    },
    'iec-action-replay': {
        'partitions': 1,
        'replication': 1,
        'retention_days': 1,
        'cleanup': 'delete',
        'description': 'IEC feedback decisions',
    },
    'if-model-updates': {
        'partitions': 1,
        'replication': 1,
        'retention_days': 7,
        'cleanup': 'compact',
        'description': 'Model update events',
    },
}


# =============================================================================
# Configuration
# =============================================================================

def _get_bootstrap_servers() -> str:
    return os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')


# =============================================================================
# Kafka Producer Config Builder
# =============================================================================

def _build_producer_config(bootstrap_servers: Optional[str] = None) -> Dict[str, str]:
    """Build FlinkKafkaProducer config dict with EXACTLY_ONCE guarantees."""
    servers = bootstrap_servers or _get_bootstrap_servers()
    return {
        'bootstrap.servers': servers,
        'acks': 'all',
        'compression.type': 'lz4',
        'linger.ms': '5',
        'batch.size': '16384',
    }


# =============================================================================
# Sink Factory
# =============================================================================

def get_kafka_sink(
    topic: str,
    bootstrap_servers: Optional[str] = None,
    custom_config: Optional[Dict[str, str]] = None,
) -> FlinkKafkaProducer:
    """
    Build an EXACTLY_ONCE FlinkKafkaProducer for the given topic.

    Args:
        topic: Kafka topic name.
        bootstrap_servers: Kafka broker list (default: from env KAFKA_BOOTSTRAP_SERVERS).
        custom_config: Optional overrides merged into the default producer config.

    Returns:
        Configured FlinkKafkaProducer with EXACTLY_ONCE semantic.

    Example:
        # Sink valid records to dq-stream-processed
        sink = get_kafka_sink(DQ_STREAM_PROCESSED_TOPIC)
        valid_stream.sink_to(sink)

        # Sink anomalies to dq-stream-anomalies
        anomaly_sink = get_kafka_sink(DQ_STREAM_ANOMALIES_TOPIC)
        anomaly_stream.sink_to(anomaly_sink)
    """
    producer_config = _build_producer_config(bootstrap_servers)
    if custom_config:
        producer_config.update(custom_config)

    LOGGER.info(
        "[KafkaSink] Creating EXACTLY_ONCE producer for topic='%s' "
        "bootstrap_servers='%s' compression='lz4'",
        topic, producer_config['bootstrap.servers']
    )

    return FlinkKafkaProducer(
        topic=topic,
        serialization_schema=JsonSerializationSchema(),
        producer_config=producer_config,
        producer_semantic=FlinkKafkaProducer.SEMANTIC.EXACTLY_ONCE,
    )


# =============================================================================
# Convenience Sinks for Common Topics
# =============================================================================

def get_hard_rule_violations_sink(
    bootstrap_servers: Optional[str] = None,
) -> FlinkKafkaProducer:
    """Sink L1 schema/rule violations to dq-hard-rule-violations."""
    return get_kafka_sink(DQ_HARD_RULE_VIOLATIONS_TOPIC, bootstrap_servers)


def get_processed_sink(
    bootstrap_servers: Optional[str] = None,
) -> FlinkKafkaProducer:
    """Sink L1 valid enriched records to dq-stream-processed."""
    return get_kafka_sink(DQ_STREAM_PROCESSED_TOPIC, bootstrap_servers)


def get_anomalies_sink(
    bootstrap_servers: Optional[str] = None,
) -> FlinkKafkaProducer:
    """Sink anomaly scores to dq-stream-anomalies."""
    return get_kafka_sink(DQ_STREAM_ANOMALIES_TOPIC, bootstrap_servers)


def get_meta_stream_sink(
    bootstrap_servers: Optional[str] = None,
) -> FlinkKafkaProducer:
    """Sink voting results to dq-meta-stream."""
    return get_kafka_sink(DQ_META_STREAM_TOPIC, bootstrap_servers)


def get_clean_stream_sink(
    bootstrap_servers: Optional[str] = None,
) -> FlinkKafkaProducer:
    """Sink clean records (canary passed) to dq-stream-processed-clean."""
    return get_kafka_sink(DQ_STREAM_PROCESSED_CLEAN_TOPIC, bootstrap_servers)


def get_iec_action_replay_sink(
    bootstrap_servers: Optional[str] = None,
) -> FlinkKafkaProducer:
    """Sink IEC decisions to iec-action-replay."""
    return get_kafka_sink(IEC_ACTION_REPLAY_TOPIC, bootstrap_servers)


def get_model_updates_sink(
    bootstrap_servers: Optional[str] = None,
) -> FlinkKafkaProducer:
    """Sink model update events to if-model-updates."""
    return get_kafka_sink(IF_MODEL_UPDATES_TOPIC, bootstrap_servers)
