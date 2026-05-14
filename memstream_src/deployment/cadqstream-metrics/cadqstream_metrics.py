"""
CA-DQStream Metrics Exporter
Exposes custom metrics for Prometheus scraping.
"""

import os
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
)
import redis
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== METRICS ====================

# Processing metrics
RECORDS_PROCESSED = Counter(
    'cadqstream_records_processed_total',
    'Total records processed',
    ['topic']
)

VIOLATIONS_DETECTED = Counter(
    'cadqstream_violations_total',
    'Total data quality violations detected',
    ['rule_type', 'severity']
)

ANOMALIES_DETECTED = Counter(
    'cadqstream_anomalies_detected_total',
    'Total anomalies detected by MemStream',
    ['anomaly_type']
)

# Latency metrics
PROCESSING_LATENCY = Histogram(
    'cadqstream_processing_latency_seconds',
    'Record processing latency',
    ['stage'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
)

# Status gauges
REDIS_CONNECTION_STATUS = Gauge(
    'cadqstream_redis_connected',
    'Redis connection status (1=connected, 0=disconnected)'
)

KAFKA_CONNECTION_STATUS = Gauge(
    'cadqstream_kafka_connected',
    'Kafka connection status (1=connected, 0=disconnected)'
)

# Throughput
THROUGHPUT = Gauge(
    'cadqstream_throughput_per_second',
    'Current throughput in records per second'
)

# Model metrics
MODEL_INFERENCE_TIME = Histogram(
    'cadqstream_model_inference_seconds',
    'MemStream model inference time',
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
)

RECONSTRUCTION_ERROR = Histogram(
    'cadqstream_reconstruction_error',
    'MemStream reconstruction error',
    buckets=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
)


class MetricsCollector:
    """Collects and exposes metrics."""

    def __init__(self):
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))
        self.redis_password = os.getenv('REDIS_PASSWORD', '')
        self.last_throughput_check = time.time()
        self.last_record_count = 0

    def collect_redis_metrics(self):
        """Collect metrics from Redis."""
        try:
            r = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                socket_timeout=5
            )
            r.ping()
            REDIS_CONNECTION_STATUS.set(1)

            # Get key counts
            keys = r.dbsize()
            logger.debug(f"Redis keys: {keys}")

        except Exception as e:
            logger.warning(f"Redis metrics collection failed: {e}")
            REDIS_CONNECTION_STATUS.set(0)

    def update_throughput(self):
        """Calculate current throughput."""
        current_time = time.time()
        elapsed = current_time - self.last_throughput_check

        if elapsed >= 10:
            self.last_throughput_check = current_time

    def run(self):
        """Run the metrics collection loop."""
        while True:
            try:
                self.collect_redis_metrics()
                self.update_throughput()
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")

            time.sleep(5)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus scraping."""

    def do_GET(self):
        if self.path == '/metrics':
            output = generate_latest()
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(output)
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        elif self.path.startswith('/webhook'):
            self._handle_webhook()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_webhook(self):
        """Handle incoming alert webhooks."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            alert = json.loads(body)
            logger.info(f"Received alert: {alert.get('alerts', [{}])[0].get('labels', {}).get('alertname', 'unknown')}")
        except Exception as e:
            logger.warning(f"Webhook parse error: {e}")

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "received"}')

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def main():
    port = int(os.getenv('METRICS_PORT', '9250'))

    # Start metrics collector in background
    collector = MetricsCollector()

    # Start HTTP server
    server = HTTPServer(('0.0.0.0', port), MetricsHandler)
    logger.info(f"Metrics exporter listening on port {port}")

    server.serve_forever()


if __name__ == '__main__':
    main()
