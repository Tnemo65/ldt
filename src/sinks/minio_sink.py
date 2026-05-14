"""
MinIO Lakehouse Sinks for CA-DQStream.

Each stream writes Parquet files to a dedicated MinIO bucket via Hadoop S3A.
Flink's StreamingFileSink provides:
  - Exactly-once guarantee via two-phase commit (2PC)
  - In-progress files buffered in memory, flushed on checkpoint
  - Roll condition: size OR time (whichever hits first)

Bucket layout (created by deployment/minio/init-scripts/01-create-buckets.sh):
  raw-zone/         → taxi_trips_raw         (valid records from Layer 1)
  quarantine-zone/  → schema_violations       (Layer 1 parse/schema failures)
                      → canary_violations     (Layer 2 canary rule failures)
  clean-zone/       → anomaly_scores          (Layer 2 ML scoring results)
                      → meta_metrics           (Layer 3 windowed meta-metrics)
                      → drift_events           (Layer 4 IEC decisions)
                      → alerts                (IEC + pipeline alerts)

Env vars:
  S3_ENDPOINT       e.g. http://minio:9000
  AWS_ACCESS_KEY_ID defaults to minioadmin
  AWS_SECRET_ACCESS_KEY defaults to minioadmin123
"""

from pyflink.common.serialization import Encoder
from pyflink.datastream.connectors.file_system import (
    FileSink,
    OutputFileConfig,
    RollingPolicy,
    StreamingFileSink,
)
import os
import json
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Env / Helpers
# ─────────────────────────────────────────────────────────────────────────────

_S3_ENDPOINT = os.getenv('S3_ENDPOINT', 'http://minio:9000')
_S3_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
_S3_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin123')


def _s3_path(bucket: str, table: str) -> str:
    return f"s3://{bucket}/{table}"


# ─────────────────────────────────────────────────────────────────────────────
# Per-table row → JSON string converters  (used by to-json MapFunctions)
# ─────────────────────────────────────────────────────────────────────────────

def record_to_raw_trips_json(record: dict) -> str:
    """Serialize a validated trip record to a single-line JSON string."""
    return json.dumps({
        'trip_id': record.get('trip_id', ''),
        'vendor_id': int(record.get('VendorID', 0)),
        'pickup_datetime': str(record.get('tpep_pickup_datetime', '')),
        'dropoff_datetime': str(record.get('tpep_dropoff_datetime', '')),
        'passenger_count': int(record.get('passenger_count', 0)),
        'trip_distance': float(record.get('trip_distance', 0.0)),
        'pickup_location_id': int(record.get('PULocationID', 0)),
        'dropoff_location_id': int(record.get('DOLocationID', 0)),
        'payment_type': int(record.get('payment_type', 0)),
        'fare_amount': float(record.get('fare_amount', 0.0)),
        'total_amount': float(record.get('total_amount', 0.0)),
    }, default=str)


def record_to_schema_violation_json(record: dict, counter: int = 0) -> str:
    """Serialize a schema-violated record to JSON."""
    missing = [
        f for f in ['trip_distance', 'fare_amount', 'PULocationID',
                     'DOLocationID', 'passenger_count']
        if not record.get(f)
    ]
    reasons = [f"Missing/Invalid: {f}" for f in missing]
    return json.dumps({
        'trip_id': record.get('trip_id', f'unknown_{counter}'),
        'violation_type': 'SCHEMA_VALIDATION_FAILED',
        'violation_reason': '; '.join(reasons) if reasons else 'Schema validation failed',
        'kafka_offset': 0,
        'kafka_partition': 0,
    })


def record_to_canary_violation_json(record: dict) -> str:
    """Serialize a canary rule violation record to JSON."""
    violations = record.get('canary_violations', [])
    return json.dumps({
        'trip_id': record.get('trip_id', ''),
        'violation_types': json.dumps(violations),
        'violation_count': len(violations),
        'fare_amount': float(record.get('fare_amount', 0.0)),
        'trip_distance': float(record.get('trip_distance', 0.0)),
        'passenger_count': int(record.get('passenger_count', 0)),
        'payment_type': int(record.get('payment_type', 0)),
        'pickup_datetime': str(record.get('tpep_pickup_datetime', '')),
        'final_decision': 'ANOMALY',
        'decision_source': 'canary_rule',
        'confidence': 1.0,
    }, default=str)


def record_to_anomaly_score_json(record: dict) -> str:
    """Serialize an ML anomaly score record to JSON."""
    return json.dumps({
        'trip_id': record.get('trip_id', ''),
        'anomaly_score': float(record.get('anomaly_score', 0.0)),
        'threshold': float(record.get('threshold', 0.5)),
        'is_anomaly': bool(record.get('is_anomaly', False)),
        'context_key': str(record.get('context_key', 'unknown')),
        'neighborhood': str(record.get('neighborhood', 'unknown')),
        'model_version': os.getenv('MODEL_VERSION', 'memstream-v1'),
    }, default=str)


def record_to_meta_metric_json(record: dict) -> str:
    """Serialize a windowed meta-metric record to JSON."""
    window_start = record.get('window_start', datetime.utcnow().isoformat())
    window_end = record.get('window_end', datetime.utcnow().isoformat())
    neighborhood = record.get('neighborhood_id', record.get('neighborhood', 'unknown'))
    return json.dumps({
        'neighborhood': str(neighborhood),
        'window_start': str(window_start),
        'window_end': str(window_end),
        'volume': int(record.get('volume', 0)),
        'null_rate': float(record.get('null_rate', 0.0)),
        'violation_rate': float(record.get('violation_rate', 0.0)),
        'anomaly_rate': float(record.get('anomaly_rate', 0.0)),
        'avg_anomaly_score': float(record.get('avg_anomaly_score', 0.0)),
        'delta_score': float(record.get('delta_score', 0.0)),
    }, default=str)


def record_to_drift_event_json(record: dict, counter: int = 0) -> str:
    """Serialize an IEC drift-event record to JSON."""
    del counter
    drifts = record.get('drifts_detected', [])
    drift_mag = max([d.get('magnitude', 0.0) for d in drifts], default=0.0)
    assessment = record.get('drift_assessment', {})
    strategy = record.get('iec_strategy', 'NO_ACTION').upper().replace(' ', '_')
    action_result = record.get('action_result', {})
    neighborhood = record.get('neighborhood_id', 'global')
    return json.dumps({
        'scenario': str(record.get('scenario', 'UNKNOWN')),
        'neighborhood': str(neighborhood),
        'metric_name': str(record.get('metric_name', 'anomaly_rate')),
        'drift_indicator': str(assessment.get('severity', 'STABLE').upper()),
        'drift_magnitude': float(drift_mag),
        'neighborhood_count': int(assessment.get('neighborhood_count', 1)),
        'strategy': str(strategy),
        'iec_confidence': float(record.get('iec_confidence', 0.0)),
        'action_taken': str(action_result.get('action', action_result.get('message', ''))),
        'recovery_time_sec': int(action_result.get('recovery_time_sec', 0)),
    }, default=str)


def record_to_alert_json(record: dict) -> str:
    """Serialize an alert record to JSON.

    An alert is emitted when:
      - IEC detects drift (drifts_detected is non-empty)
      - OR IEC executes a non-do_nothing strategy
      - OR anomaly_rate spikes above threshold

    Each alert includes severity, category, and a human-readable message.
    """
    drifts = record.get('drifts_detected', [])
    strategy = record.get('iec_strategy', 'do_nothing')
    neighborhood = str(record.get('neighborhood_id', record.get('neighborhood', 'global')))
    scenario = str(record.get('scenario', 'UNKNOWN'))

    # ── Determine alert severity ─────────────────────────────────────
    if drifts:
        max_mag = max([d.get('magnitude', 0.0) for d in drifts], default=0.0)
        if max_mag > 0.5:
            severity = 'critical'
        elif max_mag > 0.2:
            severity = 'warning'
        else:
            severity = 'info'
    elif strategy != 'do_nothing':
        severity = {'adjust_threshold': 'info', 'retrain_model': 'warning', 'switch_model': 'critical'}.get(
            strategy, 'warning'
        )
    else:
        severity = 'info'

    # ── Determine alert category ─────────────────────────────────────
    if drifts:
        category = 'drift_detected'
        message = (
            f"IEC detected {len(drifts)} drift(s) in {neighborhood} "
            f"(scenario={scenario}, max_magnitude={max([d.get('magnitude', 0) for d in drifts], default=0):.3f})"
        )
    elif strategy == 'retrain_model':
        category = 'model_retrain'
        message = (
            f"IEC triggered model retraining in {neighborhood} "
            f"(confidence={record.get('iec_confidence', 0):.2f})"
        )
    elif strategy == 'switch_model':
        category = 'model_switch'
        message = f"IEC switching to alternative model in {neighborhood} due to severe drift."
    elif strategy == 'adjust_threshold':
        category = 'threshold_adjust'
        action = record.get('action_result', {}).get('new_threshold', '?')
        message = f"IEC adjusted anomaly threshold to {action} in {neighborhood}."
    else:
        category = 'unknown'
        message = f"IEC decision: {strategy} in {neighborhood}."

    return json.dumps({
        'alert_id': f"{scenario}_{neighborhood}_{record.get('iec_timestamp', datetime.utcnow().isoformat())}",
        'severity': severity,
        'category': category,
        'scenario': scenario,
        'neighborhood': neighborhood,
        'message': message,
        'strategy': strategy.upper().replace(' ', '_'),
        'iec_confidence': float(record.get('iec_confidence', 0.0)),
        'drift_count': len(drifts),
        'drifts': drifts,
        'action_result': record.get('action_result', {}),
        'timestamp': record.get('iec_timestamp', datetime.utcnow().isoformat()),
        # passthrough meta-metrics for context
        'volume': int(record.get('volume', 0)),
        'anomaly_rate': float(record.get('anomaly_rate', 0.0)),
        'violation_rate': float(record.get('violation_rate', 0.0)),
        'delta_score': float(record.get('delta_score', 0.0)),
    }, default=str)


def record_to_pipeline_stats_json(record: dict) -> str:
    """Serialize a periodic pipeline statistics record to JSON.

    Emitted every N records from CadqstreamMetrics pipeline stats.
    """
    return json.dumps({
        'timestamp': record.get('timestamp', datetime.utcnow().isoformat()),
        'window': record.get('window', 'unknown'),
        'records_input': int(record.get('records_input', 0)),
        'records_valid': int(record.get('records_valid', 0)),
        'records_invalid': int(record.get('records_invalid', 0)),
        'canary_violations': int(record.get('canary_violations', 0)),
        'ml_anomalies': int(record.get('ml_anomalies', 0)),
        'iec_decisions': int(record.get('iec_decisions', 0)),
        'drift_events': int(record.get('drift_events', 0)),
        'anomaly_rate': float(record.get('anomaly_rate', 0.0)),
        'throughput_rps': float(record.get('throughput_rps', 0.0)),
    }, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Sink factories
# Each returns a FileSink that writes single-line JSON to S3A/Parquet.
# We use JSON (not Parquet) for maximum compatibility — Flink StreamingFileSink
# natively supports JSON via SimpleStringEncoder without extra serializers.
# Parquet would require pyflink-specific ParquetWriterFactory which is complex
# in pure Python; JSON is the idiomatic choice for PyFlink streaming sinks.
# ─────────────────────────────────────────────────────────────────────────────

def _build_file_sink(
    bucket: str,
    table: str,
) -> StreamingFileSink:
    """Build a StreamingFileSink for a given bucket/table with rolling policy.

    Args:
        bucket:   MinIO bucket name (e.g. 'raw-zone')
        table:    logical table name, used as the path prefix

    Returns:
        StreamingFileSink configured for S3A with default rolling policy.
        Each record is encoded as UTF-8 JSON via Encoder.simple_string_encoder().
    """
    path = _s3_path(bucket, table)

    encoder = Encoder.simple_string_encoder()

    rolling = RollingPolicy.default_rolling_policy()

    output = (
        OutputFileConfig
        .builder()
        .with_part_prefix(f"{table}")
        .with_part_suffix('.json')
        .build()
    )

    return (
        StreamingFileSink
        .for_row_format(path, encoder)
        .with_rolling_policy(rolling)
        .with_output_file_config(output)
        .build()
    )


def create_raw_trips_sink():
    """Sink valid taxi trips to cadqstream-raw/taxi_trips_raw/."""
    return _build_file_sink('cadqstream-raw', 'taxi_trips_raw')


def create_schema_violations_sink():
    """Sink schema violations to cadqstream-violations/schema_violations/."""
    return _build_file_sink('cadqstream-violations', 'schema_violations')


def create_canary_violations_sink():
    """Sink canary rule violations to cadqstream-violations/canary_violations/."""
    return _build_file_sink('cadqstream-violations', 'canary_violations')


def create_anomaly_scores_sink():
    """Sink ML anomaly scores to cadqstream-anomalies/anomaly_scores/."""
    return _build_file_sink('cadqstream-anomalies', 'anomaly_scores')


def create_meta_metrics_sink():
    """Sink windowed meta-metrics to cadqstream-metrics/meta_metrics/."""
    return _build_file_sink('cadqstream-metrics', 'meta_metrics')


def create_drift_events_sink():
    """Sink IEC drift events to cadqstream-drift/drift_events/."""
    return _build_file_sink('cadqstream-drift', 'drift_events')


def create_alerts_sink():
    """Sink IEC/pipeline alerts to cadqstream-drift/alerts/."""
    return _build_file_sink('cadqstream-drift', 'alerts')


def create_pipeline_stats_sink():
    """Sink periodic pipeline statistics to cadqstream-metrics/pipeline_stats/."""
    return _build_file_sink('cadqstream-metrics', 'pipeline_stats')
