"""
CA-DQStream Sequential Pipeline — Phase 3 Refactor.

Pipeline Flow (Sequential, No Dual Branch, No Voting):
    Kafka taxi-nyc-raw-v2
      -> Layer 1 (Parse JSON / Watermark / Dedup / Schema Validation)
      -> Layer 2 (CanaryRulesValidator: 7 rules -> canary_violations)
      -> Layer 2 (MemStreamScoringOperator: AE + Memory, auto-update if score < beta)
      -> Layer 3 (MetaAggregator: 1-min window per neighborhood -> 6 meta-metrics)
      -> Layer 4 (IEC: ADWIN drift detection -> 2 strategies: do_nothing / quick_retrain)

MinIO Buckets:
    cadqstream-raw/taxi_trips_raw/         — Layer 1 valid records
    cadqstream-violations/schema/          — Layer 1 schema failures
    cadqstream-violations/canary/         — Layer 2 canary rule violations
    cadqstream-anomalies/scores/           — MemStream anomaly scores
    cadqstream-metrics/meta/               — MetaAggregator windowed metrics
    cadqstream-drift/iec/                  — IEC decisions + alerts

Kafka Event Types:
    PROCESSED_RECORD, CANARY_VIOLATION, ANOMALY_RECORD,
    META_RECORD, IEC_DECISION, DLQ_RECORD

Usage:
    python src/flink_job_complete.py
"""

from pyflink.datastream import StreamExecutionEnvironment, CheckpointingMode, RuntimeExecutionMode
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common import ConfigOptions
from pyflink.common.typeinfo import Types
from pyflink.datastream import MapFunction, FilterFunction
from pyflink.datastream.window import TumblingProcessingTimeWindows
from pyflink.common.time import Time

from src.operators.watermark_assigner import create_watermark_strategy
import os
import json
import logging
import sys


# =============================================================================
# KAFKA SINK FACTORY
# =============================================================================

def _simple_schema():
    """Return a SimpleStringSchema for Kafka sinks.

    PyFlink 1.18.1 does not support custom Python SerializationSchema objects
    (no _j_serialization_schema JNI bridge). All upstream operators MUST produce
    valid JSON strings so SimpleStringSchema can serialize them as UTF-8 bytes.
    """
    from pyflink.common.serialization import SimpleStringSchema
    return SimpleStringSchema()


from src.operators.key_generator import generate_trip_id
from src.operators.deduplicator import DeduplicatorFunction
from src.operators.consecutive_records_filter import ConsecutiveRecordsFilter, _extract_meter_key
from src.operators.schema_validator import SchemaValidator
from src.operators.canary_rules_operator import CanaryRulesValidator, ViolationFilter
from src.operators.memstream_scoring_operator import MemStreamScoringOperator
from src.operators.meta_aggregator import (
    MetaAggregateFunction,
    MetaWindowProcessFunction,
    SequentialFinalDecisionFunction,
    extract_neighborhood_key,
)
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
        if isinstance(value, dict) and value is not None:
            _emit_metric('records_violation_total', 1,
                         {'layer': 'L1', 'type': 'schema'}, 'counter')
        return value


class CanaryViolationSinkFunction(MapFunction):
    def map(self, value):
        if value is None or not isinstance(value, dict):
            return {'canary_violations': [], 'has_violation': False}
        violations = value.get('canary_violations', [])
        if violations:
            _emit_metric('violation_records_total', 1,
                         {'layer': 'L2', 'type': 'canary'}, 'counter')
            for rule in violations:
                _emit_metric('anomalies_canary_total', 1,
                             {'layer': 'L2', 'rule': rule}, 'counter')
        return value


class MLAnomalySinkFunction(MapFunction):
    def map(self, value):
        if value is None or not isinstance(value, dict):
            return {'is_anomaly': False}
        if value.get('is_anomaly', False):
            neighborhood = str(value.get('neighborhood',
                                       value.get('context_key', 'unknown')))
            _emit_metric('anomalies_ml_total', 1,
                         {'layer': 'L2', 'neighborhood': neighborhood}, 'counter')
        return value


class L1ValidSinkFunction(MapFunction):
    def map(self, value):
        if value is not None:
            _emit_metric('records_valid_total', 1, {'layer': 'L1'}, 'counter')
        return value


class L2ValidSinkFunction(MapFunction):
    def map(self, value):
        if isinstance(value, dict) and value is not None:
            _emit_metric('records_valid_total', 1, {'layer': 'L2'}, 'counter')
        return value


class MetaSinkFunction(MapFunction):
    """Emit cadqstream metrics for MetaAggregator windowed output."""
    def map(self, value):
        if isinstance(value, dict) and value is not None:
            _emit_metric('records_valid_total', 1, {'layer': 'L3'}, 'counter')
            final_decision = value.get('final_decision', 'CLEAN')
            _emit_metric('final_decisions_total', 1,
                        {'layer': 'L3', 'decision': final_decision}, 'counter')
        return value


class L4MetricsFunction(MapFunction):
    """Emit cadqstream L4 metrics per micro-batch from iec_stream."""
    _client = None
    _endpoint = 'http://cadqstream-metrics:9250/internal/metrics'
    _errors = 0

    def open(self, runtime_context):
        import urllib.request
        L4MetricsFunction._client = urllib.request

    def map(self, value):
        if not isinstance(value, dict) or value is None:
            return value
        try:
            strategy = value.get('iec_strategy', 'do_nothing')
            payload = json.dumps({
                'name': 'records_valid_total',
                'value': 1,
                'labels': {'layer': 'L4'},
                'type': 'counter'
            }).encode('utf-8')
            L4MetricsFunction._client.urlopen(
                L4MetricsFunction._endpoint, data=payload, timeout=1
            )
            payload2 = json.dumps({
                'name': 'iec_strategies_total',
                'value': 1,
                'labels': {'layer': 'L4', 'strategy': strategy},
                'type': 'counter'
            }).encode('utf-8')
            L4MetricsFunction._client.urlopen(
                L4MetricsFunction._endpoint, data=payload2, timeout=1
            )
        except Exception:
            L4MetricsFunction._errors += 1
            if L4MetricsFunction._errors <= 3:
                logging.getLogger('cadqstream-l4').warning(
                    "L4 metric emit failed (errors=%d)", L4MetricsFunction._errors
                )
        return value


# =============================================================================
# KAFKA SINK FACTORY
# =============================================================================

def make_unified_kafka_sink(bootstrap_servers):
    """Create a unified Kafka sink for all pipeline outputs.

    All event types are tagged with _event_type and routed through one producer.
    """
    props = {
        'bootstrap.servers': bootstrap_servers,
        'linger.ms': '5',
        'batch.size': '65536',
        'buffer.memory': '134217728',
        'compression.type': 'lz4',
        # CRITICAL FIX #1: acks=all is required for exactly-once checkpointing.
        # acks=0 means fire-and-forget — records may be lost on broker failure,
        # which violates the exactly-once guarantee even if checkpoints succeed.
        # Trade-off: acks=all increases latency by ~1-3ms per record due to
        # waiting for all ISRs to acknowledge. Ensure Kafka broker has
        # min.insync.replicas=1 (default) or higher.
        'acks': 'all',
        'retries': '3',
        'max.in.flight.requests.per.connection': '5',
    }
    return FlinkKafkaProducer(
        topic='dq-stream-unified',
        serialization_schema=_simple_schema(),
        producer_config=props
    )


class UnifiedKafkaSerializer(MapFunction):
    """Tag and serialize every pipeline event into a single Kafka topic."""
    def map(self, value):
        if value is None:
            return json.dumps({
                '_dlq': True,
                '_dlq_reason': 'null_input',
                '_dlq_category': 'VALIDATION_ERROR',
                '_dlq_operator': 'UnifiedKafkaSerializer',
                '_event_type': 'DLQ_RECORD',
            }, default=str)
        if isinstance(value, tuple):
            value = value[1]
        if not isinstance(value, dict):
            return json.dumps({
                '_dlq': True,
                '_dlq_reason': f'not_dict:{type(value).__name__}',
                '_dlq_category': 'VALIDATION_ERROR',
                '_dlq_operator': 'UnifiedKafkaSerializer',
                '_event_type': 'DLQ_RECORD',
            }, default=str)
        record = {**value, '_event_type': value.get('_event_type', 'RECORD')}
        return json.dumps(record, default=str)


def make_kafka_sink(topic, bootstrap_servers):
    """Create a Kafka producer sink with optimized throughput settings."""
    props = {
        'bootstrap.servers': bootstrap_servers,
        'linger.ms': '5',
        'batch.size': '32768',
        'buffer.memory': '67108864',
        'compression.type': 'lz4',
        # CRITICAL FIX #1: acks=all required for exactly-once.
        # See make_unified_kafka_sink for full trade-off documentation.
        'acks': 'all',
        'retries': '3',
        'max.in.flight.requests.per.connection': '5',
    }
    return FlinkKafkaProducer(
        topic=topic,
        serialization_schema=_simple_schema(),
        producer_config=props
    )


# =============================================================================
# MINIO SINK FACTORY FUNCTIONS
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


# =============================================================================
# HELPER MAP FUNCTIONS
# =============================================================================

class SafeParseJsonFunction(MapFunction):
    """Parse JSON with DLQ logging for poison pill detection.

    ROOT CAUSE FIX #2: When json.loads() fails, the raw string value is
    logged to the DLQ metrics endpoint AND a sentinel dict is returned so
    downstream operators never receive None.
    """
    _dlq_client = None
    _dlq_endpoint = 'http://cadqstream-metrics:9250/internal/metrics'
    _parse_errors = 0

    def open(self, runtime_context):
        import urllib.request
        SafeParseJsonFunction._dlq_client = urllib.request

    def map(self, value):
        if value is None:
            self._emit_dlq('null_input', '')
            return self._sentinel('null_input')
        try:
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                self._emit_dlq('non_dict_payload', str(value)[:500])
                return self._sentinel(f'non_dict_payload:{str(value)[:100]}')
            return parsed
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            SafeParseJsonFunction._parse_errors += 1
            if SafeParseJsonFunction._parse_errors <= 10:
                logging.getLogger('cadqstream-l1').error(
                    "[SafeParseJson] FAIL #%d: %s raw=%s",
                    SafeParseJsonFunction._parse_errors,
                    type(e).__name__,
                    str(value)[:200]
                )
            self._emit_dlq(f'parse_error_{type(e).__name__}', str(value)[:500])
            return self._sentinel(f'parse_error:{type(e).__name__}')

    def _sentinel(self, reason: str):
        return {
            '_dlq': True,
            '_dlq_reason': reason,
            '_dlq_category': 'PARSE_ERROR',
            '_dlq_timestamp': '',
            '_dlq_operator': 'SafeParseJsonFunction',
            'trip_id': f'dlq_parse_{reason[:30]}',
            'tpep_pickup_datetime': '',
        }

    def _emit_dlq(self, reason: str, raw_value: str):
        try:
            payload = json.dumps({
                'name': 'dlq_records_total',
                'value': 1,
                'labels': {'reason': reason, 'stage': 'parse_json'},
                'type': 'counter'
            }).encode('utf-8')
            req = SafeParseJsonFunction._dlq_client.Request(
                SafeParseJsonFunction._dlq_endpoint,
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            SafeParseJsonFunction._dlq_client.urlopen(req, timeout=2)
        except Exception:
            pass


ParseJsonFunction = SafeParseJsonFunction


class AddTripIdFunction(MapFunction):
    def map(self, record):
        if record is None:
            return {
                '_dlq': True,
                '_dlq_reason': 'null_input',
                '_dlq_category': 'VALIDATION_ERROR',
                '_dlq_operator': 'AddTripIdFunction',
                'trip_id': 'dlq_null',
            }
        if not isinstance(record, dict):
            logging.getLogger('cadqstream-l1').error(
                "[AddTripId] WARNING: expected dict got %s value=%s",
                type(record).__name__, str(record)[:100]
            )
            return {
                '_dlq': True,
                '_dlq_reason': f'not_dict:{type(record).__name__}',
                '_dlq_category': 'VALIDATION_ERROR',
                '_dlq_operator': 'AddTripIdFunction',
                'trip_id': f'dlq_{type(record).__name__[:20]}',
            }
        record['trip_id'] = generate_trip_id(record)
        return record


class ExtractNeighborhoodFunction(MapFunction):
    """Extract neighborhood key and emit as (key, record) tuple for keyed windowing."""
    def map(self, record):
        if not isinstance(record, dict):
            return {
                '_dlq': True,
                '_dlq_reason': f'not_dict:{type(record).__name__}',
                '_dlq_category': 'VALIDATION_ERROR',
                '_dlq_operator': 'ExtractNeighborhoodFunction',
                'trip_id': f'dlq_{type(record).__name__[:20]}',
                'tpep_pickup_datetime': '',
            }, {
                '_dlq': True,
                '_dlq_reason': f'not_dict:{type(record).__name__}',
                '_dlq_category': 'VALIDATION_ERROR',
            }
        return extract_neighborhood_key(record), record


def _safe_to_json(record, serializer_fn, operator_name: str, fallback_reason: str):
    """Common safe wrapper: never return None from serialization functions.

    Always returns str for Kafka SimpleStringSchema compatibility.
    """
    if not isinstance(record, dict):
        return json.dumps({
            '_dlq': True,
            '_dlq_reason': f'not_dict:{type(record).__name__}',
            '_dlq_category': 'VALIDIZATION_ERROR',
            '_dlq_operator': operator_name,
            'trip_id': f'dlq_{type(record).__name__[:20]}',
            'tpep_pickup_datetime': '',
        }, default=str)
    try:
        result = serializer_fn(record)
        if isinstance(result, bytes):
            return result.decode('utf-8')
        return result
    except (TypeError, ValueError, KeyError) as e:
        return json.dumps({
            '_dlq': True,
            '_dlq_reason': f'serialization_error:{type(e).__name__}:{e}',
            '_dlq_category': 'VALIDIZATION_ERROR',
            '_dlq_operator': operator_name,
            'trip_id': record.get('trip_id', 'unknown'),
            'tpep_pickup_datetime': record.get('tpep_pickup_datetime', ''),
        }, default=str)


class ToRawTripJson(MapFunction):
    def map(self, record):
        return _safe_to_json(record, record_to_raw_trips_json, 'ToRawTripJson', 'raw_trips')


class ToSchemaViolationJson(MapFunction):
    _counter = 0
    def map(self, record):
        if not isinstance(record, dict):
            return json.dumps({
                '_dlq': True,
                '_dlq_reason': 'not_dict',
                '_dlq_category': 'VALIDATION_ERROR',
                '_dlq_operator': 'ToSchemaViolationJson',
                'trip_id': f'dlq_{type(record).__name__[:20]}',
                'violation_type': 'SCHEMA_VALIDATION_FAILED',
                'kafka_offset': 0,
                'kafka_partition': 0,
            }, default=str)
        ToSchemaViolationJson._counter += 1
        return _safe_to_json(
            record,
            lambda r: record_to_schema_violation_json(r, ToSchemaViolationJson._counter),
            'ToSchemaViolationJson',
            'schema_violation'
        )


class ToCanaryViolationJson(MapFunction):
    def map(self, record):
        return _safe_to_json(record, record_to_canary_violation_json, 'ToCanaryViolationJson', 'canary_violation')


class ToAnomalyScoreJson(MapFunction):
    def map(self, record):
        return _safe_to_json(record, record_to_anomaly_score_json, 'ToAnomalyScoreJson', 'anomaly_score')


class ToDriftEventJson(MapFunction):
    def map(self, record):
        return _safe_to_json(record, record_to_drift_event_json, 'ToDriftEventJson', 'drift_event')


class ToMetaMetricJson(MapFunction):
    def map(self, record):
        return _safe_to_json(record, record_to_meta_metric_json, 'ToMetaMetricJson', 'meta_metric')


class ToAlertJson(MapFunction):
    """Serialize alert record to JSON for MinIO cadqstream-drift/alerts."""
    def map(self, record):
        if not isinstance(record, dict):
            return json.dumps({
                '_dlq': True,
                '_dlq_reason': 'not_dict',
                '_dlq_category': 'VALIDATION_ERROR',
                '_dlq_operator': 'ToAlertJson',
                'alert_id': 'dlq_invalid',
                'severity': 'info',
                'category': 'unknown',
                'neighborhood': 'unknown',
                'strategy': 'unknown',
                'drift_count': 0,
                'drifts': [],
                'timestamp': '',
            }, default=str)
        drifts = record.get('drifts_detected', [])
        strategy = record.get('iec_strategy', 'do_nothing')
        if not drifts and strategy == 'do_nothing':
            return json.dumps({
                '_dlq': True,
                '_dlq_reason': 'do_nothing_no_drift',
                '_dlq_category': 'FILTERED',
                '_dlq_operator': 'ToAlertJson',
                'alert_id': 'filtered_no_action',
                'severity': 'info',
                'category': 'no_action',
                'neighborhood': record.get('neighborhood', 'unknown'),
                'strategy': strategy,
                'drift_count': 0,
                'drifts': [],
                'timestamp': record.get('iec_timestamp', ''),
            }, default=str)
        return _safe_to_json(record, record_to_alert_json, 'ToAlertJson', 'alert')


class SerializeToJson(MapFunction):
    """Serialize record dict to JSON string for Kafka topics."""
    def map(self, value):
        if value is None:
            return json.dumps({
                '_dlq': True,
                '_dlq_reason': 'null_input',
                '_dlq_category': 'VALIDATION_ERROR',
                '_dlq_operator': 'SerializeToJson',
                'trip_id': 'dlq_null',
            }, default=str)
        if isinstance(value, tuple):
            value = value[1]
        return _safe_to_json(value, lambda r: json.dumps(r, default=str), 'SerializeToJson', 'generic')


# =============================================================================
# KAFKA SOURCE FACTORY
# =============================================================================

def create_kafka_source(env, topic: str):
    """Create a KafkaSource configured for unbounded streaming."""
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
    consumer_group = os.getenv('KAFKA_CONSUMER_GROUP', 'cadqstream-complete-pipeline')

    properties = {
        'bootstrap.servers': bootstrap_servers,
        'group.id': consumer_group,
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': 'false',
        'fetch.min.bytes': '524288',
        'fetch.max.wait.ms': '500',
        'max.partition.fetch.bytes': '10485760',
        'max.poll.records': '5000',
        'request.timeout.ms': '60000',
        'session.timeout.ms': '30000',
        'heartbeat.interval.ms': '5000',
        'max.poll.interval.ms': '180000',
        'consumer.timeout.ms': '30000',
        'metadata.max.age.ms': '30000',
    }
    consumer = FlinkKafkaConsumer(
        topics=topic,
        deserialization_schema=SimpleStringSchema(),
        properties=properties
    )
    consumer.set_start_from_earliest()
    return consumer


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """CA-DQStream Sequential Pipeline — Phase 3.

    Sequential flow (no dual branch, no voting):
      Layer 1 -> Layer 2 (Canary + MemStream) -> Layer 3 (MetaAggregator) -> Layer 4 (IEC)

    Key changes from Phase 2D:
    - Removed VotingEnsembleFunction (no voting)
    - Removed PerRecordIECOperator (replaced by IECOperator on windowed meta-metrics)
    - Sequential decision: final_decision set directly from has_violation / is_anomaly
    - MetaAggregator uses keyed TumblingEventTimeWindows per neighborhood
    - IECOperator receives windowed format {nb: {metrics}}, not per-record
    """
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.STREAMING)
    env.set_parallelism(4)

    from pyflink.datastream import ExternalizedCheckpointCleanup
    checkpoint_config = env.get_checkpoint_config()
    checkpoint_config.set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    checkpoint_config.set_checkpoint_interval(120000)
    checkpoint_config.set_min_pause_between_checkpoints(60000)
    checkpoint_config.set_checkpoint_timeout(600000)
    checkpoint_config.set_max_concurrent_checkpoints(1)
    # NOTE: State backend (RocksDB) is configured in docker-compose.yml via
    # flink-conf.yaml: state.backend: rocksdb, state.backend.incremental: true.
    # This is picked up automatically by the Flink runtime — no code-level
    # env.set_state_backend() call needed for PyFlink 1.18+.
    checkpoint_config.enable_externalized_checkpoints(
        ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
    )
    # CRITICAL FIX #2: Unaligned checkpoints disabled with parallelism=4.
    # With 4 parallel subtasks, aligned checkpoints can cause head-of-line
    # blocking when one subtask is slow — barriers wait indefinitely for the
    # slowest channel. Options to fix:
    #   (A) Enable unaligned checkpoints for barrierless processing:
    #         checkpoint_config.enable_unaligned_checkpoints()
    #   (B) Keep aligned but increase concurrent checkpoints to overlap them:
    #         checkpoint_config.set_max_concurrent_checkpoints(2)
    # Chosen: keep aligned checkpoints with max_concurrent=1 (safer for RocksDB).
    # NOTE: With parallelism=4 and single TM, unaligned checkpoints are risky
    # because all 4 slots share the same TM process — partial writes can corrupt
    # state. Only enable unaligned checkpoints when running across multiple TMs.
    checkpoint_config.disable_unaligned_checkpoints()

    # ── Layer 1: Ingestion & Validation ─────────────────────────────────
    kafka_source = create_kafka_source(env, 'taxi-nyc-raw-v2')
    stream = env.add_source(kafka_source)

    stream = (
        stream
        .map(SafeParseJsonFunction())
        .filter(lambda x: x is not None)
        .assign_timestamps_and_watermarks(create_watermark_strategy())
    )
    stream = stream.map(AddTripIdFunction())

    deduplicated_stream = (
        stream
        .key_by(lambda x: x['trip_id'], key_type=Types.STRING())
        .map(DeduplicatorFunction())
        .filter(lambda x: x is not None)
    )

    consecutive_dedup_stream = (
        deduplicated_stream
        # TEMP DISABLED: consecutive dedup drops all records when Kafka produces
        # records with identical meter keys (same VendorID, PULocationID, DOLocationID,
        # RatecodeID, passenger_count, trip_distance, payment_type, fare_amount,
        # total_amount). This is common when consumer parallelism < Kafka partitions.
        # After fixing the pipeline flow, re-enable with a smarter dedup key that
        # includes trip_id or a sequence number.
        # .key_by(lambda x: _extract_meter_key(x), key_type=Types.STRING())
        # .process(ConsecutiveRecordsFilter())
        # For now, pass through all deduped records.
        .map(lambda x: x)
    )

    class DlqFilter(FilterFunction):
        def filter(self, record):
            return isinstance(record, dict) and record.get('_dlq') is True

    dlq_from_dedup_stream = consecutive_dedup_stream.filter(DlqFilter())
    valid_after_dedup = consecutive_dedup_stream.filter(
        lambda x: isinstance(x, dict) and x.get('_dlq') is not True
    )

    validator = SchemaValidator()

    class ValidFilter(FilterFunction):
        def filter(self, record):
            if not isinstance(record, dict):
                return False
            return validator.filter(record)

    class InvalidFilter(FilterFunction):
        def filter(self, record):
            if not isinstance(record, dict):
                return False
            return not validator.filter(record)

    valid_stream = valid_after_dedup.filter(ValidFilter())
    violation_stream = valid_after_dedup.filter(InvalidFilter())

    # ── Layer 2: Sequential Processing (Canary -> MemStream) ────────────
    # Canary: appends has_violation, canary_violations
    layer2_stream = valid_stream.map(CanaryRulesValidator())

    # Extract neighborhood key first so we can key the stream for MemStream.
    # MemStreamScoringOperator uses Flink ValueState which requires a KeyedStream.
    layer2_stream = (
        layer2_stream
        .map(ExtractNeighborhoodFunction())
        .key_by(lambda x: x[0], key_type=Types.STRING())
    )

    # MemStream: appends anomaly_score, is_anomaly, neighborhood, etc.
    # Also polls MinIO for retrain signals every RETAIN_SIGNAL_POLL_INTERVAL records
    layer2_stream = layer2_stream.map(MemStreamScoringOperator())

    # Add sequential final_decision (no voting — Canary OR ML anomaly)
    layer2_stream = layer2_stream.map(SequentialFinalDecisionFunction())

    # Layer 2 metrics — all on keyed stream (sink operators handle keyed context)
    layer2_stream.map(L2ValidSinkFunction())

    # ── Layer 3: MetaAggregator (1-min window per neighborhood) ───────────
    # Already keyed by neighborhood, so window operates directly.
    # ExtractNeighborhoodFunction already produced (key, record) tuples.
    # Re-extract neighborhood from the record field (index 1) for windowing.
    meta_window_stream = (
        layer2_stream
        .map(lambda x: (extract_neighborhood_key(x[1]) if isinstance(x, tuple) else x[0], x[1]))
        .key_by(lambda x: x[0], key_type=Types.STRING())
        .window(TumblingProcessingTimeWindows.of(Time.minutes(1)))
        .aggregate(
            MetaAggregateFunction(),
            window_function=MetaWindowProcessFunction(),
        )
    )

    # Layer 3 metrics
    meta_window_stream.map(MetaSinkFunction())

    # ── Layer 4: IEC (windowed meta-metrics -> 2 strategies) ──────────────
    # IECOperator expects {neighborhood: {metrics}} format — meta_window_stream
    # produces flat records with neighborhood_id field, which IECOperator converts
    iec_stream = meta_window_stream.map(IECOperator())

    # Layer 4 metrics
    iec_stream.map(L4MetricsFunction())

    # ── Kafka Unified Sink ────────────────────────────────────────────────
    BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
    unified_sink = make_unified_kafka_sink(BOOTSTRAP)
    unified_serializer = UnifiedKafkaSerializer()

    def _tag_stream(stream, event_type: str):
        return stream \
            .map(lambda r: {**r, '_event_type': event_type} if isinstance(r, dict) else r) \
            .map(unified_serializer, output_type=Types.STRING())

    merged_kafka = _tag_stream(valid_stream, 'PROCESSED_RECORD')
    merged_kafka = merged_kafka.union(
        _tag_stream(layer2_stream.filter(ViolationFilter()), 'CANARY_VIOLATION')
    )
    merged_kafka = merged_kafka.union(
        _tag_stream(
            layer2_stream.filter(lambda r: r is not None and r.get('is_anomaly', False)),
            'ANOMALY_RECORD'
        )
    )
    merged_kafka = merged_kafka.union(_tag_stream(meta_window_stream, 'META_RECORD'))
    merged_kafka = merged_kafka.union(_tag_stream(iec_stream, 'IEC_DECISION'))

    merged_kafka.add_sink(unified_sink)

    # ── MinIO Sinks ───────────────────────────────────────────────────────
    # Layer 1: schema violations
    violation_stream \
        .map(L1ViolationSinkFunction()) \
        .map(ToSchemaViolationJson()) \
        .map(make_schema_violations_sink())

    # Layer 1: valid raw trips
    valid_stream \
        .map(L1ValidSinkFunction()) \
        .map(ToRawTripJson()) \
        .map(make_raw_trips_sink())

    # Layer 2: canary violations
    layer2_stream.filter(ViolationFilter()) \
        .map(CanaryViolationSinkFunction()) \
        .map(ToCanaryViolationJson()) \
        .map(make_canary_violations_sink())

    # Layer 2: ML anomaly scores
    layer2_stream.filter(lambda r: r is not None and r.get('is_anomaly', False)) \
        .map(MLAnomalySinkFunction()) \
        .map(ToAnomalyScoreJson()) \
        .map(make_anomaly_scores_sink())

    # Layer 3: windowed meta-metrics -> MinIO cadqstream-metrics
    meta_window_stream \
        .map(ToMetaMetricJson()) \
        .map(make_meta_metrics_sink())

    # Layer 4: IEC decisions -> MinIO cadqstream-drift
    iec_stream \
        .map(ToDriftEventJson()) \
        .map(make_drift_events_sink())

    # Layer 4: IEC alerts (drift detected OR quick_retrain strategy)
    iec_stream \
        .map(ToAlertJson()) \
        .map(make_alerts_sink())

    LOGGER.info("Submitting job to Flink cluster...")
    env.execute("CA-DQStream Sequential Pipeline - Phase 3")


if __name__ == "__main__":
    main()
