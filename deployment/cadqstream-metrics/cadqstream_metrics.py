#!/usr/bin/env python3
"""
CA-DQStream Metrics Service.
Exposes:
  POST /internal/metrics  - receive metric payloads from Flink pipeline
  GET  /metrics           - Prometheus scrape endpoint
  GET  /health            - health check
"""

from flask import Flask, Response, request
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
import threading
import time
import requests
import logging
import os

app = Flask(__name__)

# Quiet werkzeug logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
LOGGER = logging.getLogger('cadqstream-metrics')

# Thread-safe metric registry
_metrics_lock = threading.Lock()
_metric_collectors = {}  # (name, type) -> prometheus Metric


def _get_or_create_metric(name: str, metric_type: str) -> object:
    """Look up or create a prometheus Counter/Gauge by (name, type)."""
    key = (name, metric_type)
    with _metrics_lock:
        if key not in _metric_collectors:
            if metric_type == 'counter':
                collector = Counter(name, f'Auto-created counter: {name}')
            else:
                collector = Gauge(name, f'Auto-created gauge: {name}')
            _metric_collectors[key] = collector
        return _metric_collectors[key]


# ── Built-in gauges ────────────────────────────────────────────────────────────

FLINK_JM_URL = os.getenv('FLINK_JM_URL', 'http://flink-jobmanager:8081')

cadqstream_pipeline_up = Gauge(
    'cadqstream_pipeline_up',
    '1 if Flink pipeline is reachable, 0 otherwise',
    ['job_name']
)

cadqstream_flink_job_running = Gauge(
    'cadqstream_flink_job_running',
    '1 if Flink REST /overview returns HTTP 200'
)


def _poll_flink():
    """Background thread: poll Flink overview every 30 s and update built-in gauges."""
    while True:
        try:
            resp = requests.get(f'{FLINK_JM_URL}/overview', timeout=5)
            running = 1 if resp.status_code == 200 else 0
            cadqstream_flink_job_running.set(running)
            cadqstream_pipeline_up.labels(job_name='cadqstream-complete').set(running)
            LOGGER.info("Flink overview poll: status=%d", running)
        except Exception as exc:
            cadqstream_flink_job_running.set(0)
            cadqstream_pipeline_up.labels(job_name='cadqstream-complete').set(0)
            LOGGER.warning("Flink overview poll failed: %s", exc)
        time.sleep(30)


# Start background thread
_poll_thread = threading.Thread(target=_poll_flink, daemon=True)
_poll_thread.start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/internal/metrics', methods=['POST'])
def receive_metric():
    """Receive a metric payload from CadqstreamMetrics._emit() and update the metric."""
    try:
        payload = request.get_json(force=True)
    except Exception:
        return {'error': 'invalid JSON'}, 400

    name = payload.get('name')
    value = payload.get('value')
    labels = payload.get('labels', {})
    metric_type = payload.get('type', 'gauge')

    if name is None or value is None:
        return {'error': 'missing name or value'}, 400

    try:
        collector = _get_or_create_metric(name, metric_type)
        labels_dict = dict(labels)  # ensure dict
        if metric_type == 'counter':
            collector.labels(**labels_dict).inc(value if value else 1)
        else:
            collector.labels(**labels_dict).set(value)
        LOGGER.debug("Metric updated: name=%s type=%s value=%s labels=%s",
                     name, metric_type, value, labels)
    except Exception as exc:
        LOGGER.error("Failed to update metric %s: %s", name, exc)
        return {'error': str(exc)}, 500

    return {'status': 'ok'}, 200


@app.route('/metrics')
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/health')
def health():
    """Health check."""
    with _metrics_lock:
        count = len(_metric_collectors)
    return {'status': 'healthy', 'metrics_count': count}, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9250)
