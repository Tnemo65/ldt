#!/usr/bin/env python3
"""
Stats Writer — polls cadqstream-metrics and writes pipeline_stats to MinIO.

Runs as a sidecar service. Every 60 seconds:
1. GET cadqstream-metrics:9250/metrics
2. Parse Prometheus text format for cadqstream_* counters
3. Write aggregated stats as JSONL to cadqstream-metrics/pipeline_stats in MinIO

Also runs as a one-shot via environment variable for Flink checkpoint events.

Environment:
  MINIO_ENDPOINT (default: minio:9000)
  MINIO_ACCESS_KEY (default: minioadmin)
  MINIO_SECRET_KEY (default: minioadmin123)
  METRICS_URL (default: http://cadqstream-metrics:9250/metrics)
  INTERVAL_SECONDS (default: 60)
"""

import os
import sys
import json
import time
import threading
import logging
from datetime import datetime
from io import StringIO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
LOGGER = logging.getLogger('stats-writer')


# ── Health check HTTP server ──────────────────────────────────────────────────
def run_health_server(port=9252):
    """Simple HTTP health server on a separate thread."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/health':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            else:
                self.send_response(404)
                self.end_headers()
        def log_message(self, fmt, *args):
            pass  # silence default logging
    server = HTTPServer(('0.0.0.0', port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    LOGGER.info("Health server listening on :%d", port)


def parse_prometheus_text(content: str) -> dict:
    """Parse Prometheus text format into a dict of metric_name -> value."""
    metrics = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Handle simple metrics: name value
        parts = line.split(None, 1)
        if len(parts) == 2:
            name, val = parts
            try:
                metrics[name] = float(val)
            except ValueError:
                pass
    return metrics


def fetch_metrics(url: str) -> dict:
    """Fetch metrics from cadqstream-metrics endpoint."""
    try:
        import urllib.request
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return parse_prometheus_text(resp.read().decode('utf-8'))
    except Exception as e:
        LOGGER.warning("Failed to fetch metrics: %s", e)
        return {}


def write_to_minio(stats: dict, endpoint: str, access_key: str, secret_key: str):
    """Write a JSON stats record to MinIO cadqstream-metrics/pipeline_stats."""
    try:
        from minio import Minio
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )
        bucket = 'cadqstream-metrics'
        # Ensure bucket exists (use bucket_exists, not stat_object which checks objects)
        if not client.bucket_exists(bucket):
            LOGGER.warning("Bucket %s does not exist", bucket)

        record = json.dumps({**stats, 'written_at': datetime.utcnow().isoformat()}, default=str)
        data = record.encode('utf-8')
        import io
        reader = io.BytesIO(data)
        import uuid
        key = f"pipeline_stats/stats_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jsonl"
        client.put_object(bucket, key, reader, len(data))
        LOGGER.info("Written stats to s3://%s/%s (%d bytes)", bucket, key, len(data))
    except ImportError:
        LOGGER.warning("minio package not installed - stats not written to MinIO")
    except Exception as e:
        LOGGER.warning("Failed to write to MinIO: %s", e)


def build_stats_summary(metrics: dict) -> dict:
    """Build a pipeline stats record from raw metrics dict."""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'records_input': int(metrics.get('cadqstream_records_input_total', 0)),
        'records_valid': int(metrics.get('cadqstream_records_valid_total', 0)),
        'records_invalid': int(metrics.get('cadqstream_records_violation_total', 0)),
        'canary_violations': int(metrics.get('cadqstream_anomalies_canary_total', 0)),
        'ml_anomalies': int(metrics.get('cadqstream_anomalies_ml_total', 0)),
        'iec_decisions': int(metrics.get('cadqstream_iec_decisions_total', 0)),
        'drift_events': int(metrics.get('cadqstream_iec_drift_detected_total', 0)),
    }


def run_loop(interval: int, metrics_url: str, minio_endpoint: str, access_key: str, secret_key: str):
    """Main polling loop."""
    LOGGER.info("Stats writer started. Polling every %ds.", interval)
    prev_metrics = {}

    while True:
        try:
            metrics = fetch_metrics(metrics_url)
            if metrics:
                # Compute deltas for rates
                stats = build_stats_summary(metrics)
                write_to_minio(stats, minio_endpoint, access_key, secret_key)
                prev_metrics = metrics
            else:
                LOGGER.debug("No metrics fetched this round.")
        except Exception as e:
            LOGGER.error("Error in stats loop: %s", e)

        time.sleep(interval)


def main():
    interval = int(os.getenv('INTERVAL_SECONDS', '60'))
    metrics_url = os.getenv('METRICS_URL', 'http://cadqstream-metrics:9250/metrics')
    minio_endpoint = os.getenv('MINIO_ENDPOINT', 'minio:9000')
    access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
    secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')

    run_health_server(int(os.getenv('HEALTH_PORT', '9252')))

    # One-shot mode: write once and exit (for testing / on-demand use)
    if os.getenv('ONE_SHOT') == '1':
        metrics = fetch_metrics(metrics_url)
        if metrics:
            stats = build_stats_summary(metrics)
            write_to_minio(stats, minio_endpoint, access_key, secret_key)
            print(json.dumps(stats, indent=2, default=str))
        return

    run_loop(interval, metrics_url, minio_endpoint, access_key, secret_key)


if __name__ == '__main__':
    main()
