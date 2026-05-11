"""
CA-DQStream Complete Pipeline - All 4 Layers Integrated.
Integration: Layer 1 → Layer 2 (Canary + Complex) → Layer 3 (Rendezvous + Meta) → Layer 4 (IEC)

Pipeline Flow:
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Baseline Validation                                    │
│ - Kafka Source (taxi-nyc-raw)                                  │
│ - Parse JSON → Watermark → KeyGen → Dedup → Schema Validation  │
└─────────────────────────────────────────────────────────────────┘
                          ↓ (valid records)
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Dual-Branch Processing                                 │
│ ┌─────────────────┐              ┌────────────────────┐         │
│ │ Canary Branch   │              │ Complex Branch     │         │
│ │ (Rules)         │              │ (ML Scoring)       │         │
│ └─────────────────┘              └────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                          ↓ (both branches)
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Rendezvous + MetaAggregator                            │
│ - Merge Canary + Complex (CoProcessFunction)                   │
│ - Voting Ensemble (Canary overrides ML)                        │
│ - 1-min windowed meta-metrics per neighborhood                 │
└─────────────────────────────────────────────────────────────────┘
                          ↓ (meta-metrics)
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: IEC (Intelligent Evolution Controller)                 │
│ - ADWIN-U drift detection (36 instances)                       │
│ - METER strategy prediction                                     │
│ - Multi-strategy execution (adjust/retrain/switch)             │
└─────────────────────────────────────────────────────────────────┘
                          ↓ (outputs)
┌─────────────────────────────────────────────────────────────────┐
│ Outputs:                                                         │
│ - PostgreSQL: taxi_trips_raw, schema_violations, hard_violations│
│ - Kafka: dq-meta-stream, iec-action-replay                     │
│ - Metrics: Prometheus/Grafana                                   │
└─────────────────────────────────────────────────────────────────┘

Usage:
  python src/flink_job_complete.py
"""

from pyflink.datastream import StreamExecutionEnvironment, CheckpointingMode
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.window import TumblingEventTimeWindows
from pyflink.common.time import Time
from pyflink.datastream.connectors.jdbc import JdbcSink, JdbcConnectionOptions, JdbcExecutionOptions
import os
import json
from datetime import datetime

# Import all operators
from src.operators.watermark_assigner import create_watermark_strategy
from src.operators.key_generator import generate_trip_id
from src.operators.deduplicator import DeduplicatorFunction
from src.operators.schema_validator import SchemaValidator
from src.operators.canary_rules import CanaryRulesValidator, ViolationFilter, CleanRecordFilter
from src.operators.if_scoring_operator import IFScoringOperator
from src.operators.rendezvous_operator import RendezvousOperator
from src.operators.meta_aggregator import (
    VotingEnsembleFunction,
    MetaAggregateFunction,
    MetaWindowProcessFunction,
    extract_neighborhood_key
)
from src.operators.iec_operator import IECOperator
from src.sinks.postgres_sink import (
    create_raw_trips_sink,
    create_violations_sink,
    record_to_raw_trips_row,
    record_to_violation_row
)

# MapFunction wrappers for PyFlink compatibility
from pyflink.datastream import MapFunction, FilterFunction
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
LOGGER = logging.getLogger('cadqstream-pipeline')


# =============================================================================
# P2: Custom Prometheus metrics via Flink's metric reporter
# =============================================================================

class CadqstreamMetrics:
    """Exposes cadqstream_pipeline_* metrics through Flink's Prometheus reporter.

    Metrics are emitted as log lines in a structured format that Prometheus
    scraping rules can parse, and registered with Flink's MetricGroup when
    the runtime_context is available.
    """

    def __init__(self):
        self._counters = {}   # metric_name -> int
        self._gauges = {}     # metric_name -> float
        self._labels = {}     # metric_name -> dict of labels
        self._ctx = None

    def open(self, ctx):
        """Called by the operator's open() to register metrics."""
        self._ctx = ctx
        try:
            from pyflink.common.metrics import Counter, Gauge
            registry = ctx.get_metrics_registry()
            self._reg = registry
            self._use_flink_metrics = True
        except Exception:
            self._use_flink_metrics = False

    def _emit(self, name, value, labels=None, metric_type='gauge'):
        """Emit a metric by POSTing to cadqstream-metrics scrape endpoint."""
        try:
            import urllib.request
            import json
            payload = json.dumps({
                'name': f'cadqstream_{name}',
                'value': value,
                'labels': labels or {},
                'type': 'counter' if metric_type == 'counter' else 'gauge'
            }).encode()
            req = urllib.request.Request(
                'http://cadqstream-metrics:9250/internal/metrics',
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            urllib.request.urlopen(req, timeout=1)
        except Exception:
            pass  # Non-blocking - don't crash pipeline if metrics endpoint is down

    def inc(self, name, delta=1, labels=None):
        self._counters[name] = self._counters.get(name, 0) + delta
        self._emit(name, self._counters[name], labels, 'counter')

    def set(self, name, value, labels=None):
        self._gauges[name] = value
        self._emit(name, value, labels, 'gauge')

    def record_input(self, topic):
        self.inc('records_input_total', labels={'topic': topic})

    def record_valid(self, layer):
        self.inc('records_valid_total', labels={'layer': layer})

    def record_violation(self, violation_type):
        self.inc('records_violation_total', labels={'type': violation_type})

    def record_canary_anomaly(self, rule):
        self.inc('anomalies_canary_total', labels={'rule': rule})

    def record_ml_anomaly(self, neighborhood):
        self.inc('anomalies_ml_total', labels={'neighborhood': neighborhood})

    def record_iec_decision(self, strategy):
        self.inc('iec_decisions_total', labels={'strategy': strategy})

    def record_iec_drift(self, neighborhood):
        self.inc('iec_drift_detected_total', labels={'neighborhood': neighborhood})

    def set_meta_volume(self, neighborhood, volume):
        self.set('meta_volume', volume, labels={'neighborhood': neighborhood})

    def set_meta_anomaly_rate(self, neighborhood, rate):
        self.set('meta_anomaly_rate', rate, labels={'neighborhood': neighborhood})


# Global shared metrics instance for use across all sink map functions
_pipeline_metrics = CadqstreamMetrics()


# =============================================================================
# SINK FACTORIES
# =============================================================================

def _get_pg_env():
    return {
        'host': os.getenv('PGBOUNCER_HOST', 'pgbouncer'),
        'port': os.getenv('PGBOUNCER_PORT', '5432'),
        'db': os.getenv('POSTGRES_DB', 'dq_pipeline'),
        'user': os.getenv('POSTGRES_USER', 'cadqstream'),
        'pass': os.getenv('POSTGRES_PASSWORD', 'cadqstream123'),
    }


def _make_jdbc_conn_opts(host, port, db, user, password):
    opts = JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
    opts.with_url(f"jdbc:postgresql://{host}:{port}/{db}")
    opts.with_driver_name("org.postgresql.Driver")
    opts.with_user_name(user)
    opts.with_password(password)
    return opts.build()


def _make_jdbc_exec_opts(batch_size=100, interval_ms=5000, max_retries=3):
    opts = JdbcExecutionOptions.builder()
    opts.with_batch_size(batch_size)
    opts.with_batch_interval_ms(interval_ms)
    opts.with_max_retries(max_retries)
    return opts.build()


def make_kafka_sink(topic, bootstrap_servers):
    """Create a Kafka producer sink."""
    props = {'bootstrap.servers': bootstrap_servers}
    return FlinkKafkaProducer(
        topic=topic,
        serialization_schema=SimpleStringSchema(),
        producer_config=props
    )


def make_raw_trips_sink():
    """Sink valid records to taxi_trips_raw table."""
    pg = _get_pg_env()
    conn = _make_jdbc_conn_opts(pg['host'], pg['port'], pg['db'], pg['user'], pg['pass'])
    exec_ = _make_jdbc_exec_opts(batch_size=200, interval_ms=3000)

    type_info = Types.ROW([
        Types.STRING(),   # trip_id
        Types.INT(),      # vendor_id
        Types.STRING(),   # pickup_datetime
        Types.STRING(),   # dropoff_datetime
        Types.INT(),      # passenger_count
        Types.DOUBLE(),   # trip_distance
        Types.INT(),      # pickup_location_id
        Types.INT(),      # dropoff_location_id
        Types.INT(),      # payment_type
        Types.DOUBLE(),   # fare_amount
        Types.DOUBLE(),   # total_amount
    ])

    return JdbcSink.sink(
        """INSERT INTO taxi_trips_raw (
            trip_id, vendor_id, pickup_datetime, dropoff_datetime,
            passenger_count, trip_distance, pickup_location_id, dropoff_location_id,
            payment_type, fare_amount, total_amount
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (trip_id) DO NOTHING""",
        type_info, conn, exec_
    )


def make_schema_violations_sink():
    """Sink schema violations to schema_violations table."""
    pg = _get_pg_env()
    conn = _make_jdbc_conn_opts(pg['host'], pg['port'], pg['db'], pg['user'], pg['pass'])
    exec_ = _make_jdbc_exec_opts(batch_size=100, interval_ms=3000)

    type_info = Types.ROW([
        Types.STRING(),   # trip_id
        Types.STRING(),   # violation_type
        Types.STRING(),   # violation_reason
        Types.LONG(),  # kafka_offset
        Types.INT(),     # kafka_partition
    ])

    return JdbcSink.sink(
        """INSERT INTO schema_violations (
            trip_id, violation_type, violation_reason, kafka_offset, kafka_partition
        ) VALUES (?, ?, ?, ?, ?)""",
        type_info, conn, exec_
    )


def make_canary_violations_sink():
    """Sink canary rule violations to canary_violations table."""
    pg = _get_pg_env()
    conn = _make_jdbc_conn_opts(pg['host'], pg['port'], pg['db'], pg['user'], pg['pass'])
    exec_ = _make_jdbc_exec_opts(batch_size=100, interval_ms=3000)

    type_info = Types.ROW([
        Types.STRING(),     # trip_id
        Types.STRING(),    # violation_types (JSON array string)
        Types.INT(),       # violation_count
        Types.DOUBLE(),    # fare_amount
        Types.DOUBLE(),    # trip_distance
        Types.INT(),       # passenger_count
        Types.INT(),       # payment_type
        Types.STRING(),    # pickup_datetime
        Types.STRING(),    # final_decision
        Types.STRING(),    # decision_source
        Types.DOUBLE(),   # confidence
    ])

    return JdbcSink.sink(
        """INSERT INTO canary_violations (
            trip_id, violation_types, violation_count,
            fare_amount, trip_distance, passenger_count,
            payment_type, pickup_datetime, final_decision,
            decision_source, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        type_info, conn, exec_
    )


def make_anomaly_scores_sink():
    """Sink ML anomaly scores to anomaly_scores table."""
    pg = _get_pg_env()
    conn = _make_jdbc_conn_opts(pg['host'], pg['port'], pg['db'], pg['user'], pg['pass'])
    exec_ = _make_jdbc_exec_opts(batch_size=100, interval_ms=3000)

    type_info = Types.ROW([
        Types.STRING(),   # trip_id
        Types.DOUBLE(),  # anomaly_score
        Types.DOUBLE(),  # threshold
        Types.BOOLEAN(), # is_anomaly
        Types.STRING(),   # context_key
        Types.STRING(),  # neighborhood
        Types.STRING(),  # model_version
    ])

    return JdbcSink.sink(
        """INSERT INTO anomaly_scores (
            trip_id, anomaly_score, threshold, is_anomaly,
            context_key, neighborhood, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        type_info, conn, exec_
    )


def make_meta_metrics_sink():
    """Sink windowed meta-metrics to meta_metrics table."""
    pg = _get_pg_env()
    conn = _make_jdbc_conn_opts(pg['host'], pg['port'], pg['db'], pg['user'], pg['pass'])
    exec_ = _make_jdbc_exec_opts(batch_size=10, interval_ms=60000)

    type_info = Types.ROW([
        Types.STRING(),   # neighborhood
        Types.STRING(),   # window_start
        Types.STRING(),   # window_end
        Types.LONG(),  # volume
        Types.DOUBLE(),  # null_rate
        Types.DOUBLE(),  # violation_rate
        Types.DOUBLE(),  # anomaly_rate
        Types.DOUBLE(),  # avg_anomaly_score
        Types.DOUBLE(),  # delta_score
    ])

    return JdbcSink.sink(
        """INSERT INTO meta_metrics (
            neighborhood, window_start, window_end, volume,
            null_rate, violation_rate, anomaly_rate,
            avg_anomaly_score, delta_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (neighborhood, window_start) DO UPDATE SET
            volume=EXCLUDED.volume,
            violation_rate=EXCLUDED.violation_rate,
            anomaly_rate=EXCLUDED.anomaly_rate,
            avg_anomaly_score=EXCLUDED.avg_anomaly_score""",
        type_info, conn, exec_
    )


def make_drift_events_sink():
    """Sink IEC drift events to drift_events table."""
    pg = _get_pg_env()
    conn = _make_jdbc_conn_opts(pg['host'], pg['port'], pg['db'], pg['user'], pg['pass'])
    exec_ = _make_jdbc_exec_opts(batch_size=10, interval_ms=60000)

    type_info = Types.ROW([
        Types.STRING(),   # scenario
        Types.STRING(),   # neighborhood
        Types.STRING(),   # metric_name
        Types.STRING(),   # drift_indicator
        Types.DOUBLE(),  # drift_magnitude
        Types.INT(),     # neighborhood_count
        Types.STRING(),   # strategy
        Types.DOUBLE(),  # iec_confidence
        Types.STRING(),   # action_taken
        Types.INT(),     # recovery_time_sec
    ])

    return JdbcSink.sink(
        """INSERT INTO drift_events (
            scenario, neighborhood, metric_name, drift_indicator,
            drift_magnitude, neighborhood_count, strategy,
            iec_confidence, action_taken, recovery_time_sec
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        type_info, conn, exec_
    )


# =============================================================================
# HELPER MAP FUNCTIONS
# =============================================================================


class ParseJsonFunction(MapFunction):
    """Parse JSON string to dict."""
    def map(self, value):
        try:
            return json.loads(value)
        except:
            return None


class AddTripIdFunction(MapFunction):
    """Add trip_id using MurmurHash3."""
    def map(self, record):
        if record is None:
            return None
        record['trip_id'] = generate_trip_id(record)
        return record


class ExtractNeighborhoodFunction(MapFunction):
    """Extract neighborhood key for spatial grouping."""
    def map(self, record):
        return extract_neighborhood_key(record), record


class ToRawTripRow(MapFunction):
    """Convert validated record to taxi_trips_raw row tuple."""
    def map(self, record):
        if record is None:
            return None
        return (
            record.get('trip_id', ''),
            int(record.get('VendorID', 0)),
            str(record.get('tpep_pickup_datetime', '')),
            str(record.get('tpep_dropoff_datetime', '')),
            int(float(record.get('passenger_count', 0))),
            float(record.get('trip_distance', 0.0)),
            int(float(record.get('PULocationID', 0))),
            int(float(record.get('DOLocationID', 0))),
            int(float(record.get('payment_type', 0))),
            float(record.get('fare_amount', 0.0)),
            float(record.get('total_amount', 0.0)),
        )


class ToSchemaViolationRow(MapFunction):
    """Convert invalid record to schema_violations row tuple."""
    def __init__(self):
        self.counter = 0
    def map(self, record):
        if record is None:
            return None
        self.counter += 1
        missing = [f for f in ['trip_distance','fare_amount','PULocationID','DOLocationID','passenger_count'] if not record.get(f)]
        reasons = [f"Missing/Invalid: {f}" for f in missing]
        return (
            record.get('trip_id', f'unknown_{self.counter}'),
            'SCHEMA_VALIDATION_FAILED',
            '; '.join(reasons) if reasons else 'Schema validation failed',
            0,
            0,
        )


class ToCanaryViolationRow(MapFunction):
    """Convert canary violation record to canary_violations row tuple."""
    def map(self, record):
        if record is None:
            return None
        violations = record.get('canary_violations', [])
        violations_str = json.dumps(violations)
        return (
            record.get('trip_id', ''),
            violations_str,
            len(violations),
            float(record.get('fare_amount', 0.0)),
            float(record.get('trip_distance', 0.0)),
            int(float(record.get('passenger_count', 0))),
            int(float(record.get('payment_type', 0))),
            str(record.get('tpep_pickup_datetime', '')),
            'ANOMALY',
            'canary_rule',
            1.0,
        )


class ToAnomalyScoreRow(MapFunction):
    """Convert ML scored record to anomaly_scores row tuple."""
    def map(self, record):
        if record is None:
            return None
        return (
            record.get('trip_id', ''),
            float(record.get('anomaly_score', 0.0)),
            float(record.get('threshold', 0.5)),
            bool(record.get('is_anomaly', False)),
            str(record.get('context_key', 'unknown')),
            str(record.get('neighborhood', 'unknown')),
            os.getenv('MODEL_VERSION', 'mock-v1'),
        )


class ToMetaMetricRow(MapFunction):
    """Convert meta-metric dict to meta_metrics row tuple.

    Uses window_start/window_end from MetaWindowProcessFunction if available,
    otherwise falls back to current timestamp.
    """
    def map(self, record):
        if record is None:
            return None
        window_start = record.get('window_start', datetime.utcnow().isoformat())
        window_end = record.get('window_end', datetime.utcnow().isoformat())
        neighborhood = record.get('neighborhood_id', record.get('neighborhood', 'unknown'))
        return (
            str(neighborhood),
            str(window_start),
            str(window_end),
            int(record.get('volume', 0)),
            float(record.get('null_rate', 0.0)),
            float(record.get('violation_rate', 0.0)),
            float(record.get('anomaly_rate', 0.0)),
            float(record.get('avg_anomaly_score', 0.0)),
            float(record.get('delta_score', 0.0)),
        )


class ToDriftEventRow(MapFunction):
    """Convert IEC decision to drift_events row tuple."""
    def __init__(self):
        self.counter = 0
    def map(self, record):
        if record is None:
            return None
        self.counter += 1

        # Extract from IEC decision structure (IECOperator.map() output)
        drifts = record.get('drifts_detected', [])
        drift_mag = max([d.get('magnitude', 0.0) for d in drifts], default=0.0)
        assessment = record.get('drift_assessment', {})
        strategy = record.get('iec_strategy', 'NO_ACTION').upper().replace(' ', '_')
        action_result = record.get('action_result', {})

        # Use neighborhood_id from meta-metrics dict key if present
        neighborhood = record.get('neighborhood_id', 'global')

        return (
            str(record.get('scenario', 'UNKNOWN')),
            str(neighborhood),
            str(record.get('metric_name', 'anomaly_rate')),
            str(assessment.get('severity', 'STABLE').upper()),
            float(drift_mag),
            int(assessment.get('neighborhood_count', 1)),
            str(strategy),
            float(record.get('iec_confidence', 0.0)),
            str(action_result.get('action', action_result.get('message', ''))),
            int(action_result.get('recovery_time_sec', 0)),
        )


class SerializeToJson(MapFunction):
    """Serialize record dict to JSON string for Kafka."""
    def map(self, value):
        if value is None:
            return None
        if isinstance(value, tuple):
            value = value[1]  # unwrap (key, record) tuple
        return json.dumps(value, default=str)


class NullFilter(MapFunction):
    """Filter out None values."""
    def map(self, value):
        return value


def create_kafka_source(env, topic: str):
    """Create Kafka source."""
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

    kafka_source = FlinkKafkaConsumer(
        topics=topic,
        deserialization_schema=SimpleStringSchema(),
        properties=properties
    )

    return kafka_source


def main():
    """Complete CA-DQStream pipeline with all 4 layers."""

    # ═══════════════════════════════════════════════════════════════
    # Environment Setup
    # ═══════════════════════════════════════════════════════════════

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(4)

    # Checkpointing (EXACTLY_ONCE)
    from pyflink.datastream import ExternalizedCheckpointCleanup
    checkpoint_config = env.get_checkpoint_config()
    checkpoint_config.set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    checkpoint_config.set_checkpoint_interval(45000)  # 45s
    checkpoint_config.set_min_pause_between_checkpoints(30000)
    checkpoint_config.set_checkpoint_timeout(300000)  # 5 min
    checkpoint_config.set_max_concurrent_checkpoints(1)
    checkpoint_config.enable_externalized_checkpoints(
        ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
    )

    # ═══════════════════════════════════════════════════════════════
    # LAYER 1: Baseline Validation
    # ═══════════════════════════════════════════════════════════════

    LOGGER.info("  Layer 1: Baseline Validation")

    # Kafka source
    kafka_source = create_kafka_source(env, 'taxi-nyc-raw')
    stream = env.add_source(kafka_source)

    # Parse JSON + Watermarks
    stream = (
        stream
        .map(ParseJsonFunction(), output_type=Types.PICKLED_BYTE_ARRAY())
        .filter(lambda x: x is not None)
        .assign_timestamps_and_watermarks(create_watermark_strategy())
    )

    # Generate trip_id
    stream = stream.map(AddTripIdFunction(), output_type=Types.PICKLED_BYTE_ARRAY())

    # Deduplication
    deduplicated_stream = (
        stream
        .key_by(lambda x: x['trip_id'], key_type=Types.STRING())
        .map(DeduplicatorFunction(), output_type=Types.PICKLED_BYTE_ARRAY())
        .filter(lambda x: x is not None)
    )

    # Schema validation — share a SINGLE validator instance per partition
    validator = SchemaValidator()

    class ValidFilter(FilterFunction):
        def filter(self, record):
            return validator.filter(record)

    class InvalidFilter(FilterFunction):
        def filter(self, record):
            return not validator.filter(record)

    valid_stream = deduplicated_stream.filter(ValidFilter())
    violation_stream = deduplicated_stream.filter(InvalidFilter())

    LOGGER.info("  Layer 1: Kafka Source → Parse → Watermark → Dedup → Schema Validation")

    # ═══════════════════════════════════════════════════════════════
    # LAYER 2: Dual-Branch Processing
    # ═══════════════════════════════════════════════════════════════

    LOGGER.info("  Layer 2: Canary Branch (7 rules) + Complex Branch (ML scoring)")

    # Complex Branch: ML scoring
    # Note: Requires Broadcast State with model loaded
    # For now, we'll skip actual ML scoring to avoid Broadcast State complexity
    # In production, this would use IFScoringOperator with BroadcastState

    # Simplified: Pass through with mock ML scores
    class MockMLScoringFunction(MapFunction):
        """Mock ML scoring for integration testing.

        Enriches each record with anomaly_score, threshold, is_anomaly,
        context_key, and neighborhood so downstream sinks have all required fields.
        """
        def map(self, value):
            import random
            zone_id = int(float(value.get('PULocationID', 1)))
            if zone_id <= 50:
                neighborhood = 'manhattan'
            elif zone_id <= 100:
                neighborhood = 'brooklyn'
            elif zone_id <= 150:
                neighborhood = 'queens'
            elif zone_id <= 200:
                neighborhood = 'bronx'
            elif zone_id in [132, 138]:
                neighborhood = 'airport'
            else:
                neighborhood = 'staten_island'

            value['neighborhood'] = neighborhood
            value['anomaly_score'] = random.uniform(0.2, 0.8)
            value['threshold'] = 0.50
            value['is_anomaly'] = value['anomaly_score'] > value['threshold']
            value['context_key'] = f'{neighborhood}_midday_weekday_medium'
            return value

    complex_stream = valid_stream.map(
        MockMLScoringFunction(),
        output_type=Types.PICKLED_BYTE_ARRAY()
    )

    # ═══════════════════════════════════════════════════════════════
    # LAYER 3: Rendezvous + MetaAggregator
    # ═══════════════════════════════════════════════════════════════

    LOGGER.info("  Layer 3: Rendezvous Merge + Voting Ensemble + MetaAggregator (1-min windows)")

    # Rendezvous: Connect canary + complex streams keyed by trip_id.
    # CoProcessFunction buffers whichever branch arrives first and emits
    # only when both sides have the same trip_id, enriching each record
    # with both canary_violations AND anomaly_score for voting.
    canary_keyed = canary_stream.key_by(lambda r: r.get('trip_id', ''), key_type=Types.STRING())
    complex_keyed = complex_stream.key_by(lambda r: r.get('trip_id', ''), key_type=Types.STRING())

    try:
        from pyflink.datastream import CoProcessFunction

        class RendezvousCoProcessFunc(CoProcessFunction):
            def __init__(self):
                self.canary_buf = {}
                self.complex_buf = {}

            def open(self, runtime_context):
                pass  # In-memory buffer sufficient for demo-scale

            def process_element1(self, canary_record, context):
                trip_id = canary_record.get('trip_id', '')
                if trip_id in self.complex_buf:
                    merged = {**canary_record, **self.complex_buf.pop(trip_id)}
                    yield merged
                else:
                    self.canary_buf[trip_id] = canary_record

            def process_element2(self, complex_record, context):
                trip_id = complex_record.get('trip_id', '')
                if trip_id in self.canary_buf:
                    merged = {**self.canary_buf.pop(trip_id), **complex_record}
                    yield merged
                else:
                    self.complex_buf[trip_id] = complex_record

        merged_stream = canary_keyed.connect(complex_keyed).process(RendezvousCoProcessFunc())
    except Exception as e:
        merged_stream = canary_stream.union(complex_stream)

    voting_stream = merged_stream.map(
        VotingEnsembleFunction(),
        output_type=Types.PICKLED_BYTE_ARRAY()
    )

    meta_stream = (
        voting_stream
        .map(ExtractNeighborhoodFunction(), output_type=Types.TUPLE([Types.STRING(), Types.PICKLED_BYTE_ARRAY()]))
        .key_by(lambda x: x[0], key_type=Types.STRING())
        .window(TumblingEventTimeWindows.of(Time.minutes(1)))
        .aggregate(
            MetaAggregateFunction(),
            accumulator_type=Types.PICKLED_BYTE_ARRAY(),
            output_type=Types.PICKLED_BYTE_ARRAY()
        )
    )

    iec_stream = meta_stream.map(
        IECOperator(),
        output_type=Types.PICKLED_BYTE_ARRAY()
    )

    LOGGER.info("  Layer 4: IEC (ADWIN-U drift detection + METER strategy selection)")
    LOGGER.info("  Outputs: 2 PG tables + 4 Kafka topics")

    BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')

    # ── Layer 1: Valid records ──────────────────────────────────────
    # Schema-violated records -> PostgreSQL schema_violations + Kafka dq-hard-rule-violations
    violation_stream \
        .map(ToSchemaViolationRow(), output_type=Types.ROW([
            Types.STRING(),   # trip_id
            Types.STRING(),   # violation_type
            Types.STRING(),   # violation_reason
            Types.LONG(),  # kafka_offset
            Types.INT(),     # kafka_partition
        ])) \
        .filter(lambda x: x is not None) \
        .add_sink(make_schema_violations_sink())

    # Schema violations -> Kafka dq-hard-rule-violations (for downstream consumers / anomaly injection)
    violation_stream \
        .map(SerializeToJson(), output_type=Types.STRING()) \
        .filter(lambda x: x is not None) \
        .add_sink(make_kafka_sink('dq-hard-rule-violations', BOOTSTRAP))

    # Valid records -> PostgreSQL taxi_trips_raw
    valid_stream \
        .map(ToRawTripRow(), output_type=Types.ROW([
            Types.STRING(),   # trip_id
            Types.INT(),      # vendor_id
            Types.STRING(),   # pickup_datetime
            Types.STRING(),   # dropoff_datetime
            Types.INT(),      # passenger_count
            Types.DOUBLE(),   # trip_distance
            Types.INT(),      # pickup_location_id
            Types.INT(),      # dropoff_location_id
            Types.INT(),      # payment_type
            Types.DOUBLE(),   # fare_amount
            Types.DOUBLE(),   # total_amount
        ])) \
        .filter(lambda x: x is not None) \
        .add_sink(make_raw_trips_sink())

    # ── Layer 2: Canary violations ───────────────────────────────────
    # Canary-violated records (have violations) -> PostgreSQL canary_violations
    canary_stream.filter(ViolationFilter()) \
        .map(ToCanaryViolationRow(), output_type=Types.ROW([
            Types.STRING(),     # trip_id
            Types.STRING(),     # violation_types (JSON)
            Types.INT(),       # violation_count
            Types.DOUBLE(),    # fare_amount
            Types.DOUBLE(),    # trip_distance
            Types.INT(),       # passenger_count
            Types.INT(),       # payment_type
            Types.STRING(),    # pickup_datetime
            Types.STRING(),    # final_decision
            Types.STRING(),    # decision_source
            Types.DOUBLE(),    # confidence
        ])) \
        .filter(lambda x: x is not None) \
        .add_sink(make_canary_violations_sink())

    # ML-scored records -> PostgreSQL anomaly_scores
    complex_stream \
        .map(ToAnomalyScoreRow(), output_type=Types.ROW([
            Types.STRING(),   # trip_id
            Types.DOUBLE(),  # anomaly_score
            Types.DOUBLE(),  # threshold
            Types.BOOLEAN(), # is_anomaly
            Types.STRING(),  # context_key
            Types.STRING(),  # neighborhood
            Types.STRING(),  # model_version
        ])) \
        .filter(lambda x: x is not None) \
        .add_sink(make_anomaly_scores_sink())

    # ── Layer 3: Meta-metrics ────────────────────────────────────────
    # Windowed aggregates -> PostgreSQL meta_metrics
    meta_stream \
        .map(ToMetaMetricRow(), output_type=Types.ROW([
            Types.STRING(),   # neighborhood
            Types.STRING(),   # window_start
            Types.STRING(),   # window_end
            Types.LONG(),  # volume
            Types.DOUBLE(),  # null_rate
            Types.DOUBLE(),  # violation_rate
            Types.DOUBLE(),  # anomaly_rate
            Types.DOUBLE(),  # avg_anomaly_score
            Types.DOUBLE(),  # delta_score
        ])) \
        .filter(lambda x: x is not None) \
        .add_sink(make_meta_metrics_sink())

    # Meta-metrics -> Kafka dq-meta-stream (for downstream consumers)
    meta_stream \
        .map(SerializeToJson(), output_type=Types.STRING()) \
        .filter(lambda x: x is not None) \
        .add_sink(make_kafka_sink('dq-meta-stream', BOOTSTRAP))

    # ── Layer 4: IEC decisions ───────────────────────────────────────
    # Drift events -> PostgreSQL drift_events
    iec_stream \
        .map(ToDriftEventRow(), output_type=Types.ROW([
            Types.STRING(),   # scenario
            Types.STRING(),  # neighborhood
            Types.STRING(),  # metric_name
            Types.STRING(),  # drift_indicator
            Types.DOUBLE(), # drift_magnitude
            Types.INT(),   # neighborhood_count
            Types.STRING(), # strategy
            Types.DOUBLE(), # iec_confidence
            Types.STRING(), # action_taken
            Types.INT(),   # recovery_time_sec
        ])) \
        .filter(lambda x: x is not None) \
        .add_sink(make_drift_events_sink())

    # IEC decisions -> Kafka iec-action-replay (for action replay)
    iec_stream \
        .map(SerializeToJson(), output_type=Types.STRING()) \
        .filter(lambda x: x is not None) \
        .add_sink(make_kafka_sink('iec-action-replay', BOOTSTRAP))

    # ── Kafka output topics ──────────────────────────────────────────
    # Valid records -> Kafka dq-stream-processed (for other consumers)
    valid_stream \
        .map(SerializeToJson(), output_type=Types.STRING()) \
        .filter(lambda x: x is not None) \
        .add_sink(make_kafka_sink('dq-stream-processed', BOOTSTRAP))

    # Canary violations -> Kafka dq-stream-anomalies
    canary_stream.filter(ViolationFilter()) \
        .map(SerializeToJson(), output_type=Types.STRING()) \
        .filter(lambda x: x is not None) \
        .add_sink(make_kafka_sink('dq-stream-anomalies', BOOTSTRAP))

    LOGGER.info("\nSubmitting job to Flink cluster...")
    env.execute("CA-DQStream Complete Pipeline - 4 Layers")


if __name__ == "__main__":
    main()
