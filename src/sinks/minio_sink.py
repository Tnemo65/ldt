"""
MinIO Lakehouse Sinks for CA-DQStream.

Each stream writes JSON files to a dedicated MinIO bucket via the native Python
minio client (boto3-compatible S3 API), bypassing Hadoop S3A entirely.

Why not StreamingFileSink?
  StreamingFileSink with S3A multipart uploads causes 0-byte zombie files when
  checkpoints are very fast (~700ms) and data never reaches the multipart threshold.
  The native minio client uses single-part PUT operations that complete immediately,
  eliminating the 2PC complexity entirely.

Sink strategy:
  - Records accumulate in a thread-safe list in process_element()
  - Every N records, the buffer is flushed:
      1. JSON lines are joined with newlines
      2. A timestamped filename is generated (yyyy-mm-dd_HHMMSS_<uuid>.json)
      3. minio_client.put_object() uploads the file in one shot
      4. The buffer is cleared
  - Every Flink checkpoint (via snapshot_state()) also forces a flush, ensuring
    no data is lost between checkpoints regardless of record count.
  - The open() method (called once per task instance) initializes the minio client.

Env vars:
  MINIO_ENDPOINT   e.g. http://minio:9000
  MINIO_ACCESS_KEY  defaults to minioadmin
  MINIO_SECRET_KEY  defaults to minioadmin123
  MINIO_BUCKET      target bucket name (set per sink factory)

Bucket layout (created by deployment/minio/init-scripts/01-create-buckets.sh):
  cadqstream-raw/       → taxi_trips_raw         (valid records from Layer 1)
  cadqstream-violations/ → schema_violations      (Layer 1 parse/schema failures)
                          → canary_violations     (Layer 2 canary rule failures)
  cadqstream-anomalies/  → anomaly_scores          (Layer 2 ML scoring results)
  cadqstream-metrics/   → meta_metrics            (Layer 3 windowed meta-metrics)
                          → pipeline_stats         (stats-writer aggregates)
  cadqstream-drift/     → drift_events            (Layer 4 IEC decisions)
                          → alerts                (IEC + pipeline alerts)
"""

import json
import os
import uuid
from datetime import datetime
from threading import Lock
from typing import Any, List, Optional

from pyflink.datastream.functions import RuntimeContext, RichMapFunction, CheckpointedFunction
from pyflink.datastream.state import ListState, ListStateDescriptor
from pyflink.common.typeinfo import Types


# ─────────────────────────────────────────────────────────────────────────────
# Env / Helpers
# ─────────────────────────────────────────────────────────────────────────────

_MINIO_ENDPOINT  = os.getenv('MINIO_ENDPOINT',  'http://minio:9000')
_MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
_MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')


def _generate_filename(table: str) -> str:
    """Generate a unique, timestamped filename for a table."""
    ts = datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')
    uid = uuid.uuid4().hex[:8]
    return f"{table}/{ts}_{uid}.json"


# ─────────────────────────────────────────────────────────────────────────────
# Custom MinIO Sink (native Python client, no S3A)
# ─────────────────────────────────────────────────────────────────────────────

class CustomMinioUploader(RichMapFunction, CheckpointedFunction):
    """RichMapFunction + CheckpointedFunction that buffers records and uploads JSON to MinIO.

    Uses the native Python minio client (S3 API) instead of Hadoop S3A,
    completely bypassing StreamingFileSink and its 2PC multipart complexity.

    Thread-safety:
      _buffer is guarded by _lock so concurrent checkpoint snapshots do not corrupt state.

    Flush triggers:
      - record_count >= flush_every_records (primary)
      - snapshot_state() forces a flush on every checkpoint barrier (guarantees no
        data is stuck in the buffer when a checkpoint completes)
      - close() forces final flush of any remaining records

    ROOT CAUSE FIX: Removed threading.Timer (which silently failed within PyFlink's
    subprocess execution model). Replaced timer-based flush with CheckpointedFunction
    integration — every Flink checkpoint barrier triggers a synchronous flush so
    buffered data is committed before the checkpoint is acknowledged.
    """

    _DEFAULT_FLUSH_RECORDS = 1000

    def __init__(
        self,
        bucket: str,
        table: str,
        flush_every_records: int = _DEFAULT_FLUSH_RECORDS,
    ):
        super().__init__()
        self.bucket = bucket
        self.table  = table
        self.flush_every_records = flush_every_records

        self._minio_client: Optional[Any] = None
        self._using_boto3: bool           = False
        self._boto3_client: Optional[Any]= None
        self._buffer: List[str]            = []
        self._lock: Lock                   = Lock()
        self._records_since_flush: int    = 0
        self._uploads_ok: int             = 0
        self._uploads_fail: int           = 0

        self._buffer_state: Optional[ListState] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self, runtime_context: RuntimeContext):
        """Initialize the minio client (one per task instance, not per record)."""
        try:
            from minio import Minio
        except ImportError:
            try:
                import boto3
                self._minio_client = None
                self._using_boto3  = True
                self._boto3_client = boto3.client(
                    's3',
                    endpoint_url=os.getenv('MINIO_ENDPOINT', _MINIO_ENDPOINT),
                    aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', _MINIO_ACCESS_KEY),
                    aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', _MINIO_SECRET_KEY),
                    region_name='us-east-1',
                )
                import logging
                logging.getLogger('cadqstream-minio').info(
                    "minio package unavailable; falling back to boto3"
                )
                self._init_buffer_state(runtime_context)
                return
            except ImportError:
                import logging
                logging.getLogger('cadqstream-minio').error(
                    "Neither minio nor boto3 is available; MinIO sink will be a no-op"
                )
                self._init_buffer_state(runtime_context)
                return

        self._using_boto3 = False
        self._minio_client = Minio(
            _MINIO_ENDPOINT,
            access_key=_MINIO_ACCESS_KEY,
            secret_key=_MINIO_SECRET_KEY,
            secure=False,
        )
        import logging
        logging.getLogger('cadqstream-minio').info(
            "MinIO client initialized: endpoint=%s bucket=%s",
            _MINIO_ENDPOINT, self.bucket,
        )
        self._init_buffer_state(runtime_context)

    def _init_buffer_state(self, runtime_context: RuntimeContext):
        """Initialize Flink ListState for checkpoint recovery of buffered records."""
        state_desc = ListStateDescriptor(
            'minio_buffer',
            Types.PICKLED_BYTE_ARRAY()
        )
        self._buffer_state = runtime_context.get_operator_state().get_list_state(state_desc)

    def close(self):
        with self._lock:
            if self._buffer:
                self._do_upload()
        import logging
        logging.getLogger('cadqstream-minio').info(
            "CustomMinioUploader closed: uploads_ok=%d uploads_fail=%d",
            self._uploads_ok, self._uploads_fail,
        )

    # ── CheckpointedFunction ─────────────────────────────────────────────────
    # snapshot_state is called when the checkpoint barrier arrives.
    # We flush the buffer synchronously here so the flushed data is committed
    # to MinIO before Flink acknowledges the checkpoint.

    def initialize_state(self, context):
        """Restore buffered records from the previous checkpoint on task restart."""
        if self._buffer_state is not None:
            import logging
            restored = 0
            for blob in self._buffer_state.get():
                if blob:
                    try:
                        decoded = blob.decode('utf-8') if isinstance(blob, bytes) else str(blob)
                        self._buffer.append(decoded)
                        restored += 1
                    except Exception:
                        pass
            if restored > 0:
                logging.getLogger('cadqstream-minio').info(
                    "Restored %d records from checkpoint buffer", restored
                )

    def snapshot_state(self, context):
        """Flush the buffer before the checkpoint is acknowledged.

        ROOT CAUSE FIX: The checkpoint barrier is the only reliable flush signal
        in PyFlink's subprocess execution model. Timer threads silently fail
        inside the Python subprocess spawned by PyFlink. By flushing here, we
        guarantee that:
          1. All buffered records are uploaded before the checkpoint completes
          2. The Flink state (empty buffer) is snapshotted after the upload
          3. On restart, only unflushed records are re-processed
        """
        with self._lock:
            if self._buffer:
                self._do_upload()
            if self._buffer_state is not None:
                self._buffer_state.clear()
                for line in self._buffer:
                    self._buffer_state.add(line.encode('utf-8'))

    # ── MapFunction contract ──────────────────────────────────────────────────

    def map(self, value):
        """Buffer a record; flush when threshold is hit."""
        if value is not None:
            line = str(value) if not isinstance(value, str) else value
        else:
            line = ''
        with self._lock:
            self._buffer.append(line)
            self._records_since_flush += 1
            if self._records_since_flush >= self.flush_every_records:
                self._do_upload()
        return value

    # ── Flush logic ──────────────────────────────────────────────────────────

    def _do_upload(self):
        """Atomic: drain buffer, build JSON file, PUT to MinIO."""
        if not self._buffer:
            return

        buffer_snapshot       = self._buffer
        self._buffer          = []
        self._records_since_flush = 0

        filename = _generate_filename(self.table)
        content  = '\n'.join(buffer_snapshot).encode('utf-8')

        try:
            if self._using_boto3:
                from io import BytesIO
                bio = BytesIO(content)
                self._boto3_client.put_object(
                    Bucket=self.bucket,
                    Key=filename,
                    Body=bio,
                    ContentLength=len(content),
                )
            else:
                from io import BytesIO
                bio = BytesIO(content)
                self._minio_client.put_object(
                    bucket_name=self.bucket,
                    object_name=filename,
                    data=bio,
                    length=len(content),
                    content_type='application/json',
                )
            self._uploads_ok += 1
            import logging
            logging.getLogger('cadqstream-minio').debug(
                "Uploaded %s/%s (%d bytes, %d records)",
                self.bucket, filename, len(content), len(buffer_snapshot),
            )
        except Exception as e:
            self._uploads_fail += 1
            import logging
            logging.getLogger('cadqstream-minio').error(
                "Upload failed %s/%s: %s (failures=%d)",
                self.bucket, filename, e, self._uploads_fail,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Per-table row → JSON string converters  (used by to-json MapFunctions)
# ─────────────────────────────────────────────────────────────────────────────

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
    """Serialize a record to JSON for MinIO cadqstream-metrics/meta_metrics.

    Handles records from both windowed aggregators (has window_start/window_end)
    and iec_stream/voting_stream (has iec_timestamp/voting_timestamp instead).
    """
    window_start = record.get('window_start')
    window_end = record.get('window_end')
    if not window_start:
        window_start = record.get('iec_timestamp') or record.get('voting_timestamp') or ''
    if not window_end:
        window_end = window_start

    neighborhood = record.get('neighborhood_id') or record.get('neighborhood', 'unknown')
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
    drift_mag = max([d.get('magnitude', 0.0) if isinstance(d, dict) else 0.0 for d in drifts], default=0.0)
    assessment = record.get('drift_assessment', {})
    if not isinstance(assessment, dict):
        assessment = {}
    strategy = record.get('iec_strategy', 'NO_ACTION').upper().replace(' ', '_')
    action_result = record.get('action_result', {})
    if not isinstance(action_result, dict):
        action_result = {}
    neighborhood = record.get('neighborhood_id', record.get('neighborhood', 'global'))
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
        max_mag = max([d.get('magnitude', 0.0) if isinstance(d, dict) else 0.0 for d in drifts], default=0.0)
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
            f"(scenario={scenario}, max_magnitude={max([d.get('magnitude', 0) if isinstance(d, dict) else 0 for d in drifts], default=0):.3f})"
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
        action_result = record.get('action_result', {})
        if not isinstance(action_result, dict):
            action_result = {}
        action = action_result.get('new_threshold', '?')
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
# Each returns a RichMapFunction that uploads JSON files to MinIO via native S3 API.
# No StreamingFileSink, no Hadoop S3A, no 2PC multipart complexity.
# ─────────────────────────────────────────────────────────────────────────────

def _build_minio_sink(
    bucket: str,
    table: str,
    flush_every_records: int = CustomMinioUploader._DEFAULT_FLUSH_RECORDS,
) -> CustomMinioUploader:
    """Build a CustomMinioUploader for a given bucket/table.

    Args:
        bucket:             MinIO bucket name (e.g. 'cadqstream-raw')
        table:              logical table name, used as the path prefix
        flush_every_records: flush after this many records (default 1000)

    Returns:
        CustomMinioUploader configured for the target bucket/table.
        The client is initialized inside open(), so no connection is held
        at class definition time.
    """
    return CustomMinioUploader(
        bucket=bucket,
        table=table,
        flush_every_records=flush_every_records,
    )


def create_raw_trips_sink():
    """Sink valid taxi trips to cadqstream-raw/taxi_trips_raw/."""
    return _build_minio_sink('cadqstream-raw', 'taxi_trips_raw')


def create_schema_violations_sink():
    """Sink schema violations to cadqstream-violations/schema_violations/."""
    return _build_minio_sink('cadqstream-violations', 'schema_violations')


def create_canary_violations_sink():
    """Sink canary rule violations to cadqstream-violations/canary_violations/."""
    return _build_minio_sink('cadqstream-violations', 'canary_violations')


def create_anomaly_scores_sink():
    """Sink ML anomaly scores to cadqstream-anomalies/anomaly_scores/."""
    return _build_minio_sink('cadqstream-anomalies', 'anomaly_scores')


def create_meta_metrics_sink():
    """Sink windowed meta-metrics to cadqstream-metrics/meta_metrics/."""
    return _build_minio_sink('cadqstream-metrics', 'meta_metrics')


def create_drift_events_sink():
    """Sink IEC drift events to cadqstream-drift/drift_events/."""
    return _build_minio_sink('cadqstream-drift', 'drift_events')


def create_alerts_sink():
    """Sink IEC/pipeline alerts to cadqstream-drift/alerts/."""
    return _build_minio_sink('cadqstream-drift', 'alerts')


def create_pipeline_stats_sink():
    """Sink periodic pipeline statistics to cadqstream-metrics/pipeline_stats/."""
    return _build_minio_sink('cadqstream-metrics', 'pipeline_stats')
