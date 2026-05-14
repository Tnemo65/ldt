# =============================================================================
# CA-DQStream + MemStream Operators
# =============================================================================
#
# This package contains PyFlink operators for the CA-DQStream + MemStream
# hybrid anomaly detection system.
#
# Modules:
#   - sinks: Kafka (EXACTLY_ONCE) and MinIO (Parquet) sinks
#   - storage: MinIO/S3 client utilities
#   - layer1: Baseline Validation (Parse, Dedup, Schema, Watermark)
#   - layer2: Dual-Branch (Canary + MemStream)
#   - layer3: Voting Ensemble + MetaAggregator
#   - layer4: IEC Feedback
#   - memstream_scoring_op: MemStream scoring with per-key memory state
#   - iec_feedback_op: IEC feedback broadcast operator
#   - health_server: Flask health/metrics server
#   - traffic_splitter: Shadow/canary/production traffic routing
#
# Sinks: MinIO
# =============================================================================

# Sink exports
from .sinks.kafka_sinks import (
    get_kafka_sink,
    DQ_HARD_RULE_VIOLATIONS_TOPIC,
    DQ_STREAM_PROCESSED_TOPIC,
    DQ_STREAM_ANOMALIES_TOPIC,
    DQ_META_STREAM_TOPIC,
    DQ_STREAM_PROCESSED_CLEAN_TOPIC,
    IEC_ACTION_REPLAY_TOPIC,
    IF_MODEL_UPDATES_TOPIC,
    TAXI_NYC_RAW_TOPIC,
)
from .sinks.minio_sinks import (
    get_minio_sink,
    get_raw_trips_sink,
    get_schema_violations_sink,
    get_canary_violations_sink,
    get_anomaly_scores_sink,
    get_meta_metrics_sink,
    get_drift_events_sink,
)

# Layer 1 exports
from .layer1 import (
    ParseJsonFunction,
    AddTripIdFunction,
    DeduplicatorFunction,
    SchemaValidator,
    ValidFilter,
    InvalidFilter,
    WatermarkAssigner,
    create_watermark_strategy,
)

# Layer 3 exports
from .layer3 import (
    VotingEnsembleFunction,
    VotingDecision,
    decide,
    MetaAggregator,
    MetaAggregatorProcessWindowFunction,
    MetaAggregatorFactory,
)

# Other operator exports
from .memstream_scoring_op import MemStreamScoringOperator
from .iec_feedback_op import IECFeedbackOperator
from .traffic_splitter import TrafficSplitter, TrafficMode, TrafficConfig
from .canary_rules import check_canary, check_canary_rules

__all__ = [
    # Sinks
    "get_kafka_sink",
    "get_minio_sink",
    "get_raw_trips_sink",
    "get_schema_violations_sink",
    "get_canary_violations_sink",
    "get_anomaly_scores_sink",
    "get_meta_metrics_sink",
    "get_drift_events_sink",
    "DQ_HARD_RULE_VIOLATIONS_TOPIC",
    "DQ_STREAM_PROCESSED_TOPIC",
    "DQ_STREAM_ANOMALIES_TOPIC",
    "DQ_META_STREAM_TOPIC",
    "DQ_STREAM_PROCESSED_CLEAN_TOPIC",
    "IEC_ACTION_REPLAY_TOPIC",
    "IF_MODEL_UPDATES_TOPIC",
    "TAXI_NYC_RAW_TOPIC",
    # Layer 1
    "ParseJsonFunction",
    "AddTripIdFunction",
    "DeduplicatorFunction",
    "SchemaValidator",
    "ValidFilter",
    "InvalidFilter",
    "WatermarkAssigner",
    "create_watermark_strategy",
    # Layer 3
    "VotingEnsembleFunction",
    "VotingDecision",
    "decide",
    "MetaAggregator",
    "MetaAggregatorProcessWindowFunction",
    "MetaAggregatorFactory",
    # Other operators
    "MemStreamScoringOperator",
    "IECFeedbackOperator",
    "TrafficSplitter",
    "TrafficMode",
    "TrafficConfig",
    "check_canary",
    "check_canary_rules",
]
