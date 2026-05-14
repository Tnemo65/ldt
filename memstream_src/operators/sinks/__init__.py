# =============================================================================
# CA-DQStream + MemStream Sinks
# =============================================================================
#
# This package contains sink connectors for writing pipeline outputs to:
#   - Kafka topics (exactly-once, LZ4 compressed JSON)
#   - MinIO/S3 object storage (StreamingFileSink with Parquet)
#
# Sinks: MinIO
# =============================================================================

from .kafka_sinks import (
    get_kafka_sink,
    KafkaSinkConfig,
    DQ_HARD_RULE_VIOLATIONS_TOPIC,
    DQ_STREAM_PROCESSED_TOPIC,
    DQ_STREAM_ANOMALIES_TOPIC,
    DQ_META_STREAM_TOPIC,
    DQ_STREAM_PROCESSED_CLEAN_TOPIC,
    IEC_ACTION_REPLAY_TOPIC,
    IF_MODEL_UPDATES_TOPIC,
    TAXI_NYC_RAW_TOPIC,
)

from .minio_sinks import (
    get_minio_sink,
    MinIOSinkConfig,
    CADQSTREAM_RAW_BUCKET,
    CADQSTREAM_VIOLATIONS_BUCKET,
    CADQSTREAM_ANOMALIES_BUCKET,
    CADQSTREAM_METRICS_BUCKET,
    CADQSTREAM_DRIFT_BUCKET,
    CADQSTREAM_RAW_PATH,
    CADQSTREAM_VIOLATIONS_PATH,
    CADQSTREAM_ANOMALIES_PATH,
    CADQSTREAM_METRICS_PATH,
    CADQSTREAM_DRIFT_PATH,
)

__all__ = [
    # Kafka
    "get_kafka_sink",
    "KafkaSinkConfig",
    "DQ_HARD_RULE_VIOLATIONS_TOPIC",
    "DQ_STREAM_PROCESSED_TOPIC",
    "DQ_STREAM_ANOMALIES_TOPIC",
    "DQ_META_STREAM_TOPIC",
    "DQ_STREAM_PROCESSED_CLEAN_TOPIC",
    "IEC_ACTION_REPLAY_TOPIC",
    "IF_MODEL_UPDATES_TOPIC",
    "TAXI_NYC_RAW_TOPIC",
    # MinIO
    "get_minio_sink",
    "MinIOSinkConfig",
    "CADQSTREAM_RAW_BUCKET",
    "CADQSTREAM_VIOLATIONS_BUCKET",
    "CADQSTREAM_ANOMALIES_BUCKET",
    "CADQSTREAM_METRICS_BUCKET",
    "CADQSTREAM_DRIFT_BUCKET",
    "CADQSTREAM_RAW_PATH",
    "CADQSTREAM_VIOLATIONS_PATH",
    "CADQSTREAM_ANOMALIES_PATH",
    "CADQSTREAM_METRICS_PATH",
    "CADQSTREAM_DRIFT_PATH",
]
