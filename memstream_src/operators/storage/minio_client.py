# =============================================================================
# MinIO Client — CA-DQStream + MemStream
# =============================================================================
#
# S3-compatible client for MinIO using boto3.
# Used for:
#   - Ensuring buckets exist before Flink writes
#   - Reading/writing Parquet files offline (e.g., stats-writer, ml-service)
#   - Admin operations (list, delete, presigned URLs)
#
# Endpoint: http://minio:9000 (in docker network)
# Credentials: from env MINIO_ACCESS_KEY / MINIO_SECRET_KEY (or AWS_* vars)
#
# MinIO reference: deployment/docker-compose.yml (minio section)
# =============================================================================

from __future__ import annotations

import os
import io
import logging
import hashlib
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, BotoCoreError

LOGGER = logging.getLogger('cadqstream.storage')


# =============================================================================
# Configuration
# =============================================================================

def _get_env(key: str, fallback: str) -> str:
    val = os.getenv(key)
    if val:
        return val
    # Fallback aliases used by docker-compose
    if key == 'MINIO_ENDPOINT':
        return os.getenv('S3_ENDPOINT', fallback)
    if key == 'MINIO_ACCESS_KEY':
        return os.getenv('AWS_ACCESS_KEY_ID', fallback)
    if key == 'MINIO_SECRET_KEY':
        return os.getenv('AWS_SECRET_ACCESS_KEY', fallback)
    return fallback


@dataclass
class MinIOConfig:
    """MinIO / S3 connection configuration."""

    endpoint_url: str = field(
        default_factory=lambda: _get_env('MINIO_ENDPOINT', 'http://minio:9000')
    )
    aws_access_key_id: str = field(
        default_factory=lambda: _get_env('MINIO_ACCESS_KEY', 'minioadmin')
    )
    aws_secret_access_key: str = field(
        default_factory=lambda: _get_env('MINIO_SECRET_KEY', 'minioadmin123')
    )
    region_name: str = field(default_factory=lambda: os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
    path_style_access: bool = field(default_factory=lambda: True)
    max_attempts: int = 3
    connect_timeout: float = 5.0
    read_timeout: float = 30.0

    def to_boto_config(self) -> BotoConfig:
        return BotoConfig(
            retries={'max_attempts': self.max_attempts, 'mode': 'standard'},
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            s3={'addressing_style': 'path'},
        )


# =============================================================================
# MinIO Client
# =============================================================================

class MinIOClient:
    """
    Thread-safe MinIO / S3 client wrapper.

    Wraps boto3 s3 resource to provide a stable API for common operations:
    - ensure_bucket_exists()
    - upload_file() / upload_bytes()
    - download_file() / download_bytes()
    - list_objects()
    - delete_object()
    - get_presigned_url()

    Usage:
        config = MinIOConfig()
        client = MinIOClient(config)

        # Ensure bucket exists
        client.ensure_bucket_exists('cadqstream-raw')

        # Upload a file
        client.upload_file('/local/data.parquet', 'cadqstream-raw', 'data/2026-05.parquet')

        # List objects
        for obj in client.list_objects('cadqstream-raw', prefix='data/'):
            print(obj['Key'], obj['Size'])

        # Download
        data = client.download_bytes('cadqstream-raw', 'data/2026-05.parquet')
    """

    def __init__(self, config: Optional[MinIOConfig] = None):
        self.config = config or MinIOConfig()
        self._client = None
        self._resource = None

    @property
    def client(self):
        """Lazy boto3 S3 client."""
        if self._client is None:
            self._client = boto3.client(
                's3',
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                region_name=self.config.region_name,
                config=self.config.to_boto_config(),
            )
        return self._client

    @property
    def resource(self):
        """Lazy boto3 S3 resource."""
        if self._resource is None:
            self._resource = boto3.resource(
                's3',
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.aws_access_key_id,
                aws_secret_access_key=self.config.aws_secret_access_key,
                region_name=self.config.region_name,
                config=self.config.to_boto_config(),
            )
        return self._resource

    # -------------------------------------------------------------------------
    # Bucket Operations
    # -------------------------------------------------------------------------

    def ensure_bucket_exists(self, bucket: str) -> bool:
        """
        Create the bucket if it does not exist (no-op if already exists).

        Returns True if bucket exists or was created successfully.
        Raises on unexpected errors.
        """
        try:
            self.client.head_bucket(Bucket=bucket)
            LOGGER.debug("[MinIO] Bucket '%s' already exists", bucket)
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ('404', 'NoSuchBucket'):
                LOGGER.info("[MinIO] Creating bucket: %s", bucket)
                self.client.create_bucket(Bucket=bucket)
                LOGGER.info("[MinIO] Bucket '%s' created", bucket)
                return True
            raise

    def list_buckets(self) -> List[Dict[str, Any]]:
        """Return all buckets with creation date."""
        response = self.client.list_buckets()
        return [{'Name': b['Name'], 'CreationDate': b['CreationDate']}
                for b in response.get('Buckets', [])]

    # -------------------------------------------------------------------------
    # Object Operations
    # -------------------------------------------------------------------------

    def upload_file(self, local_path: str, bucket: str, key: str) -> bool:
        """
        Upload a local file to MinIO.

        Args:
            local_path: Path to the local file.
            bucket: Target bucket name.
            key: Object key (path within bucket).

        Returns True on success.
        """
        try:
            self.ensure_bucket_exists(bucket)
            self.client.upload_file(local_path, bucket, key)
            LOGGER.info("[MinIO] Uploaded: %s -> s3://%s/%s", local_path, bucket, key)
            return True
        except (BotoCoreError, ClientError) as e:
            LOGGER.error("[MinIO] Upload failed: s3://%s/%s: %s", bucket, key, e)
            raise

    def upload_bytes(
        self,
        data: bytes,
        bucket: str,
        key: str,
        content_type: str = 'application/octet-stream',
    ) -> bool:
        """
        Upload raw bytes to MinIO.

        Args:
            data: Bytes to upload.
            bucket: Target bucket name.
            key: Object key.
            content_type: MIME type (default: binary).

        Returns True on success.
        """
        try:
            self.ensure_bucket_exists(bucket)
            self.client.put_object(
                Body=data,
                Bucket=bucket,
                Key=key,
                ContentType=content_type,
            )
            LOGGER.info("[MinIO] Uploaded %d bytes: s3://%s/%s", len(data), bucket, key)
            return True
        except (BotoCoreError, ClientError) as e:
            LOGGER.error("[MinIO] Upload failed: s3://%s/%s: %s", bucket, key, e)
            raise

    def upload_parquet(
        self,
        df,  # pd.DataFrame
        bucket: str,
        key: str,
    ) -> bool:
        """
        Convert a pandas DataFrame to Parquet in memory and upload.

        Requires: pandas, pyarrow or fastparquet.

        Args:
            df: pandas DataFrame.
            bucket: Target bucket name.
            key: Object key (should end in .parquet).

        Returns True on success.
        """
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine='pyarrow')
        buffer.seek(0)
        return self.upload_bytes(
            buffer.getvalue(),
            bucket,
            key,
            content_type='application/x-parquet',
        )

    def download_file(self, bucket: str, key: str, local_path: str) -> bool:
        """
        Download a MinIO object to a local file.

        Returns True on success.
        """
        try:
            self.ensure_bucket_exists(bucket)
            self.client.download_file(bucket, key, local_path)
            LOGGER.info("[MinIO] Downloaded: s3://%s/%s -> %s", bucket, key, local_path)
            return True
        except (BotoCoreError, ClientError) as e:
            LOGGER.error("[MinIO] Download failed: s3://%s/%s: %s", bucket, key, e)
            raise

    def download_bytes(self, bucket: str, key: str) -> bytes:
        """
        Download a MinIO object and return as bytes.

        Returns the object's content.
        """
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            data = response['Body'].read()
            LOGGER.info("[MinIO] Downloaded %d bytes: s3://%s/%s", len(data), bucket, key)
            return data
        except (BotoCoreError, ClientError) as e:
            LOGGER.error("[MinIO] Download failed: s3://%s/%s: %s", bucket, key, e)
            raise

    def download_parquet(self, bucket: str, key: str):
        """
        Download a Parquet object and return as a pandas DataFrame.

        Requires: pandas, pyarrow.
        """
        data = self.download_bytes(bucket, key)
        return __import__('pandas').read_parquet(io.BytesIO(data), engine='pyarrow')

    def list_objects(
        self,
        bucket: str,
        prefix: str = '',
        max_keys: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        List objects in a bucket with optional prefix.

        Returns list of object summaries (Key, Size, LastModified, ETag).
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys,
            )
            return response.get('Contents', [])
        except (BotoCoreError, ClientError) as e:
            LOGGER.error("[MinIO] List failed: s3://%s/%s: %s", bucket, prefix, e)
            raise

    def delete_object(self, bucket: str, key: str) -> bool:
        """Delete a single object from a bucket."""
        try:
            self.client.delete_object(Bucket=bucket, Key=key)
            LOGGER.info("[MinIO] Deleted: s3://%s/%s", bucket, key)
            return True
        except (BotoCoreError, ClientError) as e:
            LOGGER.error("[MinIO] Delete failed: s3://%s/%s: %s", bucket, key, e)
            raise

    def delete_objects_batch(self, bucket: str, keys: List[str]) -> bool:
        """
        Delete multiple objects in a single request (more efficient than
        individual delete_object calls for batch operations).
        """
        if not keys:
            return True
        try:
            objects = [{'Key': k} for k in keys]
            self.client.delete_objects(
                Bucket=bucket,
                Delete={'Objects': objects},
            )
            LOGGER.info("[MinIO] Batch deleted %d objects from s3://%s/", len(keys), bucket)
            return True
        except (BotoCoreError, ClientError) as e:
            LOGGER.error("[MinIO] Batch delete failed: s3://%s/: %s", bucket, e)
            raise

    def get_presigned_url(
        self,
        bucket: str,
        key: str,
        expires_in_seconds: int = 3600,
    ) -> str:
        """
        Generate a presigned URL for temporary direct access.

        Args:
            bucket: Bucket name.
            key: Object key.
            expires_in_seconds: URL validity window (default: 1 hour).

        Returns the presigned URL string.
        """
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expires_in_seconds,
            )
            LOGGER.debug("[MinIO] Presigned URL generated: s3://%s/%s (expires=%ds)",
                         bucket, key, expires_in_seconds)
            return url
        except (BotoCoreError, ClientError) as e:
            LOGGER.error("[MinIO] Presigned URL failed: s3://%s/%s: %s", bucket, key, e)
            raise

    def get_object_metadata(self, bucket: str, key: str) -> Dict[str, Any]:
        """Return HEAD object metadata (size, content-type, last-modified)."""
        try:
            response = self.client.head_object(Bucket=bucket, Key=key)
            return {
                'ContentLength': response.get('ContentLength'),
                'ContentType': response.get('ContentType'),
                'LastModified': response.get('LastModified'),
                'ETag': response.get('ETag'),
            }
        except (BotoCoreError, ClientError) as e:
            LOGGER.error("[MinIO] HEAD failed: s3://%s/%s: %s", bucket, key, e)
            raise

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """
        Run a lightweight health check against MinIO.

        Returns dict with status, endpoint, and bucket count.
        """
        try:
            buckets = self.list_buckets()
            return {
                'status': 'healthy',
                'endpoint': self.config.endpoint_url,
                'bucket_count': len(buckets),
                'buckets': [b['Name'] for b in buckets],
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'endpoint': self.config.endpoint_url,
                'error': str(e),
            }


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

# Default client instance (lazy)
_default_client: Optional[MinIOClient] = None


def _get_default_client() -> MinIOClient:
    global _default_client
    if _default_client is None:
        _default_client = MinIOClient()
    return _default_client


def ensure_bucket_exists(bucket: str) -> bool:
    """Create bucket if it does not exist (uses default client)."""
    return _get_default_client().ensure_bucket_exists(bucket)


def list_objects(bucket: str, prefix: str = '', max_keys: int = 1000) -> List[Dict[str, Any]]:
    """List objects in a bucket (uses default client)."""
    return _get_default_client().list_objects(bucket, prefix, max_keys)


def upload_file(local_path: str, bucket: str, key: str) -> bool:
    """Upload a local file (uses default client)."""
    return _get_default_client().upload_file(local_path, bucket, key)


def download_file(bucket: str, key: str, local_path: str) -> bool:
    """Download an object to a local file (uses default client)."""
    return _get_default_client().download_file(bucket, key, local_path)


def delete_object(bucket: str, key: str) -> bool:
    """Delete a single object (uses default client)."""
    return _get_default_client().delete_object(bucket, key)


def get_presigned_url(bucket: str, key: str, expires_in_seconds: int = 3600) -> str:
    """Generate a presigned URL (uses default client)."""
    return _get_default_client().get_presigned_url(bucket, key, expires_in_seconds)
