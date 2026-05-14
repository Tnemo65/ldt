# =============================================================================
# MinIO Bucket Configuration — CA-DQStream + MemStream
# =============================================================================
#
# Centralized bucket and path definitions for all MinIO storage.
# Used by sinks and storage utilities.
#
# Reference: original_flow.md lines 1397-1420 (MinIO Buckets)
# =============================================================================

from __future__ import annotations

from typing import Dict

# =============================================================================
# Bucket Names
# =============================================================================

BUCKETS: Dict[str, str] = {
    'models': 'cadqstream-models',
    'checkpoints': 'cadqstream-checkpoints',
    'raw': 'cadqstream-raw',
    'violations': 'cadqstream-violations',
    'anomalies': 'cadqstream-anomalies',
    'metrics': 'cadqstream-metrics',
    'drift': 'cadqstream-drift',
    'dlq': 'cadqstream-dlq',
}

# Aliases for convenience
CADQSTREAM_RAW_BUCKET = BUCKETS['raw']
CADQSTREAM_VIOLATIONS_BUCKET = BUCKETS['violations']
CADQSTREAM_ANOMALIES_BUCKET = BUCKETS['anomalies']
CADQSTREAM_METRICS_BUCKET = BUCKETS['metrics']
CADQSTREAM_DRIFT_BUCKET = BUCKETS['drift']
CADQSTREAM_MODELS_BUCKET = BUCKETS['models']
CADQSTREAM_CHECKPOINTS_BUCKET = BUCKETS['checkpoints']
CADQSTREAM_DLQ_BUCKET = BUCKETS['dlq']


# =============================================================================
# Path Prefixes within Buckets
# =============================================================================

BUCKET_PATHS: Dict[str, str] = {
    'models': 'memstream/',
    'checkpoints': 'flink/{job_id}/',
    'raw': 'taxi_trips/dt={date}/',
    'violations_schema': 'schema/dt={date}/',
    'violations_canary': 'canary/dt={date}/',
    'anomalies': 'scores/dt={date}/',
    'metrics': 'meta/dt={date}/hour={hour}/',
    'drift': 'events/dt={date}/',
    'dlq': 'records/dt={date}/',
}


# =============================================================================
# Convenience Path Constants
# =============================================================================

CADQSTREAM_RAW_PATH = BUCKET_PATHS['raw']
CADQSTREAM_VIOLATIONS_SCHEMA_PATH = BUCKET_PATHS['violations_schema']
CADQSTREAM_VIOLATIONS_CANARY_PATH = BUCKET_PATHS['violations_canary']
CADQSTREAM_ANOMALIES_PATH = BUCKET_PATHS['anomalies']
CADQSTREAM_METRICS_PATH = BUCKET_PATHS['metrics']
CADQSTREAM_DRIFT_PATH = BUCKET_PATHS['drift']
CADQSTREAM_DLQ_PATH = BUCKET_PATHS['dlq']

# raw_taxi_trips             cadqstream-raw/taxi_trips/
# dlq_records                cadqstream-dlq/records/
# flink_checkpoints          cadqstream-checkpoints/
# model_artifacts            cadqstream-models/

DATABASE_TABLES_MAPPING: Dict[str, Dict[str, str]] = {
    'anomaly_scores': {
        'old_table': 'anomaly_scores',
        'new_bucket': BUCKETS['anomalies'],
        'new_path': BUCKET_PATHS['anomalies'],
        'description': 'ML anomaly scores (MemStream output)',
    },
    'meta_metrics': {
        'old_table': 'meta_metrics',
        'new_bucket': BUCKETS['metrics'],
        'new_path': BUCKET_PATHS['metrics'],
        'description': 'Voting ensemble meta metrics',
    },
    'drift_events': {
        'old_table': 'drift_events',
        'new_bucket': BUCKETS['drift'],
        'new_path': BUCKET_PATHS['drift'],
        'description': 'Concept/data drift detection events',
    },
    'schema_violations': {
        'old_table': 'schema_violations',
        'new_bucket': BUCKETS['violations'],
        'new_path': BUCKET_PATHS['violations_schema'],
        'description': 'L1 schema validation violations',
    },
    'canary_violations': {
        'old_table': 'canary_violations',
        'new_bucket': BUCKETS['violations'],
        'new_path': BUCKET_PATHS['violations_canary'],
        'description': 'Canary rule violations',
    },
    'raw_taxi_trips': {
        'old_table': 'raw_taxi_trips',
        'new_bucket': BUCKETS['raw'],
        'new_path': BUCKET_PATHS['raw'],
        'description': 'Raw taxi trip records',
    },
    'dlq_records': {
        'old_table': 'dlq_records',
        'new_bucket': BUCKETS['dlq'],
        'new_path': BUCKET_PATHS['dlq'],
        'description': 'Dead letter queue records',
    },
}
