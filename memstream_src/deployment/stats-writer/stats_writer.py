"""
Stats Writer for CA-DQStream
Writes statistics from Redis to MinIO for archival and analysis.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

import redis
from minio import Minio
from minio.error import S3Error
import pandas as pd
import io

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StatsWriter:
    """Writes statistics from Redis to MinIO."""

    def __init__(self):
        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))
        self.redis_password = os.getenv('REDIS_PASSWORD', '')

        # MinIO configuration
        self.minio_endpoint = os.getenv('MINIO_ENDPOINT', 'minio:9000')
        self.minio_access = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        self.minio_secret = os.getenv('MINIO_SECRET_KEY', '')
        self.minio_bucket = os.getenv('MINIO_BUCKET_STATS', 'cadqstream-metrics')

        # Write interval
        self.write_interval = int(os.getenv('WRITE_INTERVAL', '60'))

        # Clients
        self.redis_client = None
        self.minio_client = None

    def initialize(self):
        """Initialize Redis and MinIO clients."""
        # Connect to Redis
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                decode_responses=True,
                socket_timeout=5
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        # Connect to MinIO
        try:
            self.minio_client = Minio(
                self.minio_endpoint,
                access_key=self.minio_access,
                secret_key=self.minio_secret,
                secure=False
            )

            # Ensure bucket exists
            if not self.minio_client.bucket_exists(self.minio_bucket):
                self.minio_client.make_bucket(self.minio_bucket)
                logger.info(f"Created MinIO bucket: {self.minio_bucket}")
            else:
                logger.info(f"MinIO bucket exists: {self.minio_bucket}")

            logger.info(f"Connected to MinIO at {self.minio_endpoint}")

        except Exception as e:
            logger.error(f"Failed to connect to MinIO: {e}")
            raise

    def collect_stats(self) -> Dict[str, Any]:
        """Collect statistics from Redis."""
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'records_processed': 0,
            'violations': {},
            'anomalies': {},
            'latency_p50': 0,
            'latency_p95': 0,
            'latency_p99': 0
        }

        try:
            # Get key patterns from Redis
            keys = [
                'stats:records_processed',
                'stats:violations:*',
                'stats:anomalies:*',
                'stats:latency:*'
            ]

            for pattern in keys:
                if '*' in pattern:
                    matching_keys = self.redis_client.keys(pattern)
                    for key in matching_keys:
                        value = self.redis_client.get(key)
                        if value:
                            stat_name = key.replace('stats:', '')
                            try:
                                stats[stat_name] = json.loads(value)
                            except:
                                stats[stat_name] = value
                else:
                    value = self.redis_client.get(pattern)
                    if value:
                        try:
                            stats[pattern.replace('stats:', '')] = json.loads(value)
                        except:
                            pass

        except Exception as e:
            logger.warning(f"Error collecting stats: {e}")

        return stats

    def write_to_minio(self, stats: Dict[str, Any]):
        """Write stats to MinIO as Parquet file."""
        try:
            # Create timestamp-based filename
            timestamp = datetime.utcnow()
            date_str = timestamp.strftime('%Y-%m-%d')
            hour_str = timestamp.strftime('%H')
            filename = f"stats/{date_str}/{hour_str}/{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"

            # Convert to DataFrame
            df = pd.DataFrame([stats])

            # Write to buffer
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)

            # Upload to MinIO
            self.minio_client.put_object(
                bucket_name=self.minio_bucket,
                object_name=filename,
                data=buffer,
                length=buffer.getbuffer().nbytes,
                content_type='application/octet-stream'
            )

            logger.info(f"Wrote stats to MinIO: {filename}")

        except S3Error as e:
            logger.error(f"MinIO error: {e}")
        except Exception as e:
            logger.error(f"Error writing to MinIO: {e}")

    def run(self):
        """Main loop."""
        logger.info("Starting Stats Writer...")

        self.initialize()

        logger.info(f"Writing stats every {self.write_interval} seconds")

        while True:
            try:
                # Collect stats
                stats = self.collect_stats()

                # Write to MinIO
                self.write_to_minio(stats)

                logger.debug(f"Stats written: {json.dumps(stats, default=str)}")

            except Exception as e:
                logger.error(f"Error in main loop: {e}")

            time.sleep(self.write_interval)


def main():
    writer = StatsWriter()
    writer.run()


if __name__ == '__main__':
    main()
