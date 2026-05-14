# =============================================================================
# CA-DQStream + MemStream Storage
# =============================================================================
#
# This package provides MinIO/S3 storage utilities:
#   - minio_client: boto3-based S3 client for MinIO
#   - bucket_config: Centralized bucket and path definitions
#
# Storage: MinIO
# All data: MinIO S3-compatible API
# =============================================================================

from .minio_client import (
    MinIOClient,
    MinIOConfig,
    ensure_bucket_exists,
    list_objects,
    upload_file,
    download_file,
    delete_object,
    get_presigned_url,
)

from .bucket_config import (
    BUCKETS,
    BUCKET_PATHS,
    CADQSTREAM_RAW_BUCKET,
    CADQSTREAM_VIOLATIONS_BUCKET,
    CADQSTREAM_ANOMALIES_BUCKET,
    CADQSTREAM_METRICS_BUCKET,
    CADQSTREAM_DRIFT_BUCKET,
    CADQSTREAM_MODELS_BUCKET,
    CADQSTREAM_CHECKPOINTS_BUCKET,
    CADQSTREAM_DLQ_BUCKET,
    CADQSTREAM_RAW_PATH,
    CADQSTREAM_VIOLATIONS_SCHEMA_PATH,
    CADQSTREAM_VIOLATIONS_CANARY_PATH,
    CADQSTREAM_ANOMALIES_PATH,
    CADQSTREAM_METRICS_PATH,
    CADQSTREAM_DRIFT_PATH,
    CADQSTREAM_DLQ_PATH,
    DATABASE_TABLES_MAPPING,
)

__all__ = [
    # MinIO client
    "MinIOClient",
    "MinIOConfig",
    "ensure_bucket_exists",
    "list_objects",
    "upload_file",
    "download_file",
    "delete_object",
    "get_presigned_url",
    # Bucket config
    "BUCKETS",
    "BUCKET_PATHS",
    "CADQSTREAM_RAW_BUCKET",
    "CADQSTREAM_VIOLATIONS_BUCKET",
    "CADQSTREAM_ANOMALIES_BUCKET",
    "CADQSTREAM_METRICS_BUCKET",
    "CADQSTREAM_DRIFT_BUCKET",
    "CADQSTREAM_MODELS_BUCKET",
    "CADQSTREAM_CHECKPOINTS_BUCKET",
    "CADQSTREAM_DLQ_BUCKET",
    "CADQSTREAM_RAW_PATH",
    "CADQSTREAM_VIOLATIONS_SCHEMA_PATH",
    "CADQSTREAM_VIOLATIONS_CANARY_PATH",
    "CADQSTREAM_ANOMALIES_PATH",
    "CADQSTREAM_METRICS_PATH",
    "CADQSTREAM_DRIFT_PATH",
    "CADQSTREAM_DLQ_PATH",
    "DATABASE_TABLES_MAPPING",
]
