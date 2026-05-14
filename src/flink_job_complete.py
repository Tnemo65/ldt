"""
CA-DQStream Complete Pipeline - All 4 Layers Integrated.
Integration: Layer 1 -> Layer 2 (Canary + Complex) -> Layer 3 (Rendezvous + Meta) -> Layer 4 (IEC)

Pipeline Flow:
Layer 1: Kafka Source (taxi-nyc-raw) -> Parse JSON -> Watermark -> Dedup -> Schema Validation
Layer 2: Canary Branch (7 rules) | Complex Branch (ML scoring via MemStream)
Layer 3: Voting Ensemble -> MetaAggregator
Layer 4: IEC (ADWIN-U drift detection + METER strategy)
Outputs: MinIO (raw-zone, quarantine-zone, clean-zone) + Kafka topics + Prometheus metrics

Usage:
  python src/flink_job_complete.py
"""

from pyflink.datastream import StreamExecutionEnvironment, CheckpointingMode
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream import MapFunction, FilterFunction
from pyflink.datastream.window import TumblingEventTimeWindows
from pyflink.common.time import Time
import os
import json
import logging
import sys

# Import all operators
from src.operators.watermark_assigner import create_watermark_strategy
from src.operators.key_generator import generate_trip_id
from src.operators.deduplicator import DeduplicatorFunction
from src.operators.schema_validator import SchemaValidator
from src.operators.canary_rules import CanaryRulesValidator, ViolationFilter, CleanRecordFilter
from src.operators.memstream_scoring_operator import MemStreamScoringOperator
from src.operators.meta_aggregator import VotingEnsembleFunction, extract_neighborhood_key
from src.operators.iec_operator import IECOperator
from src.sinks.minio_sink import (
    create_raw_trips_sink,
    create_schema_violations_sink,
    create_canary_violations_sink,
    create_anomaly_scores_sink,
    create_meta_metrics_sink,
    create_drift_events_sink,
    create_alerts_sink,
    create_pipeline_stats_sink,
    record_to_raw_trips_json,
    record_to_schema_violation_json,
    record_to_canary_violation_json,
    record_to_anomaly_score_json,
    record_to_drift_event_json,
    record_to_alert_json,
    record_to_pipeline_stats_json,
    record_to_meta_metric_json,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
LOGGER = logging.getLogger('cadqstream-pipeline')


# =============================================================================
# METRIC MAP FUNCTIONS
# Emits metrics via direct HTTP POST to cadqstream-metrics (non-blocking).
# =============================================================================

_MetricsEndpoint = 'http://cadqstream-metrics:9250/internal/metrics'
_MetricsClient = None


def _get_client():
    global _MetricsClient
    if _MetricsClient is None:
        import urllib.request
        _MetricsClient = urllib.request
    return _MetricsClient


def _emit_metric(name, value, labels, metric_type):
    """POST a single metric to cadqstream-metrics. Fails silently on error."""
    try:
        payload = json.dumps({
            'name': name,
            'value': value,
            'labels': labels or {},
            'type': metric_type
        }).encode('utf-8')
        req = _get_client().Request(
            _MetricsEndpoint,
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        _get_client().urlopen(req, timeout=2)
    except Exception:
        pass


class L1ViolationSinkFunction(MapFunction):
    def map(self, value):
        if value is not None:
            _emit_metric('records_violation_total', 1,
                         {'layer': 'L1', 'type': 'schema'}, 'counter')
        return value


class CanaryViolationSinkFunction(MapFunction):
    def map(self, value):
        if value is None:
            return None
        violations = value.get('canary_violations', [])
        if violations:
            _emit_metric('violation_records_total', 1,
                         {'layer': 'L1', 'type': 'canary'}, 'counter')
            for rule in violations:
                _emit_metric('anomalies_canary_total', 1,
                             {'layer': 'L1', 'rule': rule}, 'counter')
        return value


class MLAnomalySinkFunction(MapFunction):
    def map(self, value):
        if value is None:
            return None
        if value.get('is_anomaly', False):
            neighborhood = str(value.get('neighborhood',
                                       value.get('context_key', 'unknown')))
            _emit_metric('anomalies_ml_total', 1,
                         {'layer': 'L2', 'neighborhood': neighborhood}, 'counter')
        return value


class L2ValidSinkFunction(MapFunction):
    def map(self, value):
        if value is not None:
            _emit_metric('records_valid_total', 1, {'layer': 'L2'}, 'counter')
        return value


class L1ValidSinkFunction(MapFunction):
    def map(self, value):
        if value is not None:
            _emit_metric('records_valid_total', 1, {'layer': 'L1'}, 'counter')
        return value


# =============================================================================
# KAFKA SINK FACTORY
# =============================================================================

def make_kafka_sink(topic, bootstrap_servers):
    """Create a Kafka producer sink with optimized throughput settings."""
    props = {
        'bootstrap.servers': bootstrap_servers,
        'linger.ms': '5',
        'batch.size': '32768',
        'buffer.memory': '67108864',
        'compression.type': 'lz4',
        'acks': '1',
    }
    return FlinkKafkaProducer(
        topic=topic,
        serialization_schema=SimpleStringSchema(),
        producer_config=props
    )


# =============================================================================
# MINIO SINK FACTORIES
# All data persisted to MinIO via StreamingFileSink.
# Buckets: raw-zone, quarantine-zone, clean-zone
# Rolling: 60s time-based OR 128 MB size limit (whichever hits first).
# =============================================================================

def make_raw_trips_sink():
    return create_raw_trips_sink()

def make_schema_violations_sink():
    return create_schema_violations_sink()

def make_canary_violations_sink():
    return create_canary_violations_sink()

def make_anomaly_scores_sink():
    return create_anomaly_scores_sink()

def make_meta_metrics_sink():
    return create_meta_metrics_sink()

def make_drift_events_sink():
    return create_drift_events_sink()

def make_alerts_sink():
    return create_alerts_sink()

def make_pipeline_stats_sink():
    return create_pipeline_stats_sink()


# =============================================================================
# HELPER MAP FUNCTIONS
# =============================================================================

class ParseJsonFunction(MapFunction):
    def map(self, value):
        try:
            return json.loads(value)
        except:
            return None


class AddTripIdFunction(MapFunction):
    def map(self, record):
        if record is None:
            return None
        record['trip_id'] = generate_trip_id(record)
        return record


class ExtractNeighborhoodFunction(MapFunction):
    def map(self, record):
        return extract_neighborhood_key(record), record


class ToRawTripJson(MapFunction):
    def map(self, record):
        if record is None:
            return None
        return record_to_raw_trips_json(record)


class ToSchemaViolationJson(MapFunction):
    def __init__(self):
        self.counter = 0
    def map(self, record):
        if record is None:
            return None
        self.counter += 1
        return record_to_schema_violation_json(record, self.counter)


class ToCanaryViolationJson(MapFunction):
    def map(self, record):
        if record is None:
            return None
        return record_to_canary_violation_json(record)


class ToAnomalyScoreJson(MapFunction):
    def map(self, record):
        if record is None:
            return None
        return record_to_anomaly_score_json(record)


class ToDriftEventJson(MapFunction):
    def __init__(self):
        self.counter = 0
    def map(self, record):
        if record is None:
            return None
        self.counter += 1
        return record_to_drift_event_json(record, self.counter)


class ToMetaMetricJson(MapFunction):
    """Serialize meta-metric record from window output to JSON."""
    def map(self, record):
        if record is None:
            return None
        return record_to_meta_metric_json(record)


class ToAlertJson(MapFunction):
    """Serialize alert record to JSON for MinIO clean-zone/alerts.

    Emits only when IEC detects drift OR executes a non-do_nothing strategy.
    """
    def map(self, record):
        if record is None:
            return None
        drifts = record.get('drifts_detected', [])
        strategy = record.get('iec_strategy', 'do_nothing')
        if not drifts and strategy == 'do_nothing':
            return None
        return record_to_alert_json(record)


class SerializeToJson(MapFunction):
    """Serialize record dict to JSON string for Kafka topics."""
    def map(self, value):
        if value is None:
            return None
        if isinstance(value, tuple):
            value = value[1]
        return json.dumps(value, default=str)


# =============================================================================
# KAFKA SOURCE FACTORY
# =============================================================================

def create_kafka_source(env, topic: str):
    properties = {
        'bootstrap.servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        'group.id': 'cadqstream-complete-pipeline',
        'auto.offset.reset': 'earliest',
        'request.timeout.ms': '120000',
        'session.timeout.ms': '60000',
        'heartbeat.interval.ms': '10000',
        'max.poll.interval.ms': '300000',
        'consumer.timeout.ms': '60000',
        'metadata.max.age.ms': '30000',
    }
    return FlinkKafkaConsumer(
        topics=topic,
        deserialization_schema=SimpleStringSchema(),
        properties=properties
    )


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)

    from pyflink.datastream import ExternalizedCheckpointCleanup
    checkpoint_config = env.get_checkpoint_config()
    checkpoint_config.set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    checkpoint_config.set_checkpoint_interval(300000)
    checkpoint_config.set_min_pause_between_checkpoints(150000)
    checkpoint_config.set_checkpoint_timeout(1200000)
    checkpoint_config.set_max_concurrent_checkpoints(1)
    checkpoint_config.enable_externalized_checkpoints(
        ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
    )

    # ── Layer 1: Baseline Validation ───────────────────────────────────
    kafka_source = create_kafka_source(env, 'taxi-nyc-raw')
    stream = env.add_source(kafka_source)

    stream = (
        stream
        .map(ParseJsonFunction(), output_type=Types.PICKLED_BYTE_ARRAY())
        .filter(lambda x: x is not None)
        .assign_timestamps_and_watermarks(create_watermark_strategy())
    )
    stream = stream.map(AddTripIdFunction(), output_type=Types.PICKLED_BYTE_ARRAY())

    deduplicated_stream = (
        stream
        .key_by(lambda x: x['trip_id'], key_type=Types.STRING())
        .map(DeduplicatorFunction(), output_type=Types.PICKLED_BYTE_ARRAY())
        .filter(lambda x: x is not None)
    )

    validator = SchemaValidator()

    class ValidFilter(FilterFunction):
        def filter(self, record):
            return validator.filter(record)

    class InvalidFilter(FilterFunction):
        def filter(self, record):
            return not validator.filter(record)

    valid_stream = deduplicated_stream.filter(ValidFilter())
    violation_stream = deduplicated_stream.filter(InvalidFilter())

    # ── Layer 2: Dual-Branch Processing ────────────────────────────────
    canary_stream = valid_stream.map(
        CanaryRulesValidator(),
        output_type=Types.PICKLED_BYTE_ARRAY()
    )
    complex_stream = valid_stream.map(
        MemStreamScoringOperator(),
        output_type=Types.PICKLED_BYTE_ARRAY()
    )

    # ── Layer 3: Voting Ensemble ────────────────────────────────────────
    merged_stream = canary_stream.union(complex_stream)
    voting_stream = merged_stream.map(
        VotingEnsembleFunction(),
        output_type=Types.PICKLED_BYTE_ARRAY()
    )

    # ── Layer 3b: Meta-Metrics Windowing ────────────────────────────────
    # Window voting stream by neighborhood key, 1-minute tumbling windows
    from src.operators.meta_aggregator import (
        MetaAggregateFunction, MetaWindowProcessFunction, extract_neighborhood_key
    )

    keyed_for_meta = (
        voting_stream
        .filter(lambda x: x is not None)
        .map(lambda r: (extract_neighborhood_key(r), r))
        .key_by(lambda x: x[0], key_type=Types.STRING())
    )

    meta_window_stream = (
        keyed_for_meta
        .window(TumblingEventTimeWindows.of(Time.minutes(1)))
        .aggregate(
            MetaAggregateFunction(),
            MetaWindowProcessFunction(),
            output_type=Types.PICKLED_BYTE_ARRAY()
        )
    )

    # ── Layer 4: IEC ───────────────────────────────────────────────────
    # IEC consumes windowed meta-metrics (not raw records) for drift detection
    iec_stream = meta_window_stream.map(
        IECOperator(),
        output_type=Types.PICKLED_BYTE_ARRAY()
    )

    BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')

    # ── MinIO Sinks ───────────────────────────────────────────────────
    # Layer 1: schema violations
    violation_stream \
        .map(L1ViolationSinkFunction(), output_type=Types.PICKLED_BYTE_ARRAY()) \
        .map(ToSchemaViolationJson()) \
        .add_sink(make_schema_violations_sink())

    # Layer 1: valid raw trips
    valid_stream \
        .map(L1ValidSinkFunction(), output_type=Types.PICKLED_BYTE_ARRAY()) \
        .map(ToRawTripJson()) \
        .add_sink(make_raw_trips_sink())

    # Layer 2: canary violations
    canary_stream.filter(ViolationFilter()) \
        .map(ToCanaryViolationJson()) \
        .add_sink(make_canary_violations_sink())

    # Layer 2: ML anomaly scores
    complex_stream.filter(lambda r: r is not None and r.get('is_anomaly', False)) \
        .map(ToAnomalyScoreJson()) \
        .add_sink(make_anomaly_scores_sink())

    # Layer 3: windowed meta-metrics -> MinIO clean-zone/meta_metrics
    meta_window_stream \
        .map(ToMetaMetricJson()) \
        .add_sink(make_meta_metrics_sink())

    # Layer 4: IEC drift events
    iec_stream \
        .map(ToDriftEventJson()) \
        .add_sink(make_drift_events_sink())

    # Layer 4: IEC alerts (drift detected OR non-do_nothing strategy)
    iec_stream \
        .map(ToAlertJson()) \
        .add_sink(make_alerts_sink())

    # ── Kafka Sinks ───────────────────────────────────────────────────
    # Valid records -> dq-stream-processed
    valid_stream \
        .map(SerializeToJson(), output_type=Types.STRING()) \
        .add_sink(make_kafka_sink('dq-stream-processed', BOOTSTRAP))

    # IEC decisions -> iec-action-replay
    iec_stream \
        .map(SerializeToJson(), output_type=Types.STRING()) \
        .filter(lambda x: x is not None) \
        .add_sink(make_kafka_sink('iec-action-replay', BOOTSTRAP))

    # Canary anomalies -> dq-stream-anomalies
    canary_stream.filter(ViolationFilter()) \
        .map(CanaryViolationSinkFunction(), output_type=Types.PICKLED_BYTE_ARRAY()) \
        .map(SerializeToJson(), output_type=Types.STRING()) \
        .add_sink(make_kafka_sink('dq-stream-anomalies', BOOTSTRAP))

    # Clean canary records -> dq-stream-processed-clean
    canary_stream.filter(CleanRecordFilter()) \
        .map(L2ValidSinkFunction(), output_type=Types.PICKLED_BYTE_ARRAY()) \
        .map(SerializeToJson(), output_type=Types.STRING()) \
        .add_sink(make_kafka_sink('dq-stream-processed-clean', BOOTSTRAP))

    # ML anomalies -> dq-stream-anomalies
    complex_stream.filter(lambda r: r is not None and r.get('is_anomaly', False)) \
        .map(MLAnomalySinkFunction(), output_type=Types.PICKLED_BYTE_ARRAY()) \
        .map(SerializeToJson(), output_type=Types.STRING()) \
        .add_sink(make_kafka_sink('dq-stream-anomalies', BOOTSTRAP))

    # Windowed meta-metrics -> dq-meta-stream (for IEC downstream consumers)
    meta_window_stream \
        .map(SerializeToJson(), output_type=Types.STRING()) \
        .filter(lambda x: x is not None) \
        .add_sink(make_kafka_sink('dq-meta-stream', BOOTSTRAP))

    LOGGER.info("Submitting job to Flink cluster...")
    env.execute("CA-DQStream Complete Pipeline - 4 Layers")


if __name__ == "__main__":
    main()
