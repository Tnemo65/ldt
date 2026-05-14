#!/usr/bin/env python3
"""
CA-DQStream Metrics Service.
Exposes:
  POST /internal/metrics  - receive metric payloads from Flink pipeline
  GET  /metrics           - Prometheus scrape endpoint
  GET  /health            - health check
"""

from flask import Flask, Response, request
from prometheus_client import Counter, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
import threading
import time
import requests
import logging
import os

app = Flask(__name__)

logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
LOGGER = logging.getLogger('cadqstream-metrics')

# Use multi-threaded registry to allow concurrent updates
REGISTRY = CollectorRegistry(auto_describe=True)

# Thread-safe metric registry: name -> (metric_type, collector)
_registered = {}
_lock = threading.Lock()

# ── Flink health polling ─────────────────────────────────────────────────────

FLINK_JM_URL = os.getenv('FLINK_JM_URL', 'http://flink-jobmanager:8081')


def _poll_flink():
    """Poll Flink overview every 30 s and update health gauges."""
    while True:
        try:
            resp = requests.get(f'{FLINK_JM_URL}/overview', timeout=5)
            running = 1 if resp.status_code == 200 else 0
            with _lock:
                for name, (mtype, col) in _registered.items():
                    if name == 'cadqstream_flink_job_running':
                        col.set(running)
                    elif name == 'cadqstream_pipeline_up':
                        col.labels(job_name='cadqstream-complete').set(running)
            LOGGER.info("Flink overview poll: status=%d", running)
        except Exception as exc:
            with _lock:
                for name, (mtype, col) in _registered.items():
                    if name == 'cadqstream_flink_job_running':
                        col.set(0)
                    elif name == 'cadqstream_pipeline_up':
                        col.labels(job_name='cadqstream-complete').set(0)
            LOGGER.warning("Flink overview poll failed: %s", exc)
        time.sleep(30)


_poll_thread = threading.Thread(target=_poll_flink, daemon=True)
_poll_thread.start()


# ── Metric registration and update ─────────────────────────────────────────────

def _ensure_metric(name, label_keys, metric_type):
    """Register a metric if not already registered. Thread-safe."""
    with _lock:
        key = (name, metric_type)
        if key in _registered:
            return _registered[key][1]
        if metric_type == 'counter':
            col = Counter(name, f'CA-DQStream {name}', list(label_keys) or None, registry=REGISTRY)
        else:
            col = Gauge(name, f'CA-DQStream {name}', list(label_keys) or None, registry=REGISTRY)
        _registered[key] = (metric_type, col)
        return col


def _update_metric(name, value, labels_dict, metric_type):
    """Update a single metric. Creates it dynamically if needed."""
    label_keys = list(labels_dict.keys()) if labels_dict else []
    col = _ensure_metric(name, label_keys, metric_type)

    try:
        if metric_type == 'counter':
            col.labels(**labels_dict).inc(value if value else 1)
        else:
            col.labels(**labels_dict).set(value)
    except ValueError:
        pass
    return True


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/internal/metrics', methods=['POST'])
def receive_metric():
    """Receive a metric payload from CadqstreamMetrics._emit() and update the metric."""
    try:
        payload = request.get_json(force=True)
    except Exception:
        return {'error': 'invalid JSON'}, 400

    if payload.get('type') == 'batch':
        counters = payload.get('counters', [])
        gauges = payload.get('gauges', [])
        updated = 0
        for item in counters:
            _update_metric(item['Name'], item.get('value', 1), item.get('labels', {}), 'counter')
            updated += 1
        for item in gauges:
            _update_metric(item['Name'], item.get('value', 0), item.get('labels', {}), 'gauge')
            updated += 1
        return {'status': 'ok', 'updated': updated}, 200

    name = payload.get('name')
    value = payload.get('value', 1)
    labels = payload.get('labels', {})
    metric_type = payload.get('type', 'gauge')

    if not name:
        return {'error': 'missing name'}, 400

    full_name = f'cadqstream_{name}'
    labels_dict = dict(labels) if labels else {}
    _update_metric(full_name, value, labels_dict, metric_type)
    return {'status': 'ok'}, 200


@app.route('/metrics')
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)


@app.route('/health')
def health():
    """Health check."""
    with _lock:
        count = len(_registered)
    return {'status': 'healthy', 'metrics_count': count}, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9250)
