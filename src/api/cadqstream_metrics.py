#!/usr/bin/env python3
"""
CA-DQStream Metrics Collector - Exposes pipeline metrics for Prometheus scraping.

This service receives metric emissions from the Flink pipeline (via HTTP POST)
and exposes them at /metrics for Prometheus to scrape.

Usage:
    uvicorn src.api.cadqstream_metrics:app --host 0.0.0.0 --port 9250
"""

from flask import Flask, request, Response
import threading
import time

app = Flask(__name__)

# Thread-safe metric storage
_metrics_lock = threading.Lock()
_counters = {}   # name -> (labels_hash -> value)
_gauges = {}     # name -> (labels_hash -> value)

# prometheus_client collectors
try:
    from prometheus_client import Counter, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
    _registry = CollectorRegistry()
    _counters_prom = {}
    _gauges_prom = {}
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Pre-create all known cadqstream metrics so they always exist with correct type
_KNOWN_COUNTERS = [
    'cadqstream_records_input_total',
    'cadqstream_records_valid_total',
    'cadqstream_records_violation_total',
    'cadqstream_anomalies_canary_total',
    'cadqstream_anomalies_ml_total',
    'cadqstream_iec_decisions_total',
    'cadqstream_iec_drift_detected_total',
]

_KNOWN_GAUGES = [
    'cadqstream_meta_volume',
    'cadqstream_meta_anomaly_rate',
    'cadqstream_meta_null_rate',
    'cadqstream_meta_delta_score',
    'cadqstream_iec_confidence',
]

# Register known metrics at startup so they always appear in /metrics output
for name in _KNOWN_COUNTERS:
    _counters_prom[name] = Counter(name, 'CA-DQStream counter metric', [], registry=_registry)
for name in _KNOWN_GAUGES:
    _gauges_prom[name] = Gauge(name, 'CA-DQStream gauge metric', [], registry=_registry)

def _hash_labels(labels):
    import json
    return json.dumps(labels, sort_keys=True)

def _full_metric_name(name, labels):
    label_str = ''
    if labels:
        label_str = '{' + ','.join(f'{k}="{v}"' for k, v in sorted(labels.items())) + '}'
    return f"{name}{label_str}"

@app.route('/internal/metrics', methods=['POST'])
def receive_metric():
    """Receive metric from Flink pipeline."""
    import json
    try:
        data = request.get_json()
        name = data.get('name', '')
        value = data.get('value', 0)
        labels = data.get('labels', {})
        metric_type = data.get('type', 'gauge')

        with _metrics_lock:
            key = _hash_labels(labels)
            if metric_type == 'counter':
                if name not in _counters:
                    _counters[name] = {}
                if key not in _counters[name]:
                    _counters[name][key] = 0
                _counters[name][key] += value

                if PROMETHEUS_AVAILABLE:
                    if name not in _counters_prom:
                        _counters_prom[name] = Counter(name, '', list(labels.keys()), registry=_registry)
                    _counters_prom[name].labels(**labels).inc(value)
            else:
                if name not in _gauges:
                    _gauges[name] = {}
                _gauges[name][key] = value

                if PROMETHEUS_AVAILABLE:
                    if name not in _gauges_prom:
                        _gauges_prom[name] = Gauge(name, '', list(labels.keys()), registry=_registry)
                    _gauges_prom[name].labels(**labels).set(value)

        return {'status': 'ok'}, 200
    except Exception as e:
        return {'error': str(e)}, 400

@app.route('/metrics')
def metrics():
    """Prometheus scraping endpoint."""
    if PROMETHEUS_AVAILABLE:
        return Response(generate_latest(_registry), mimetype=CONTENT_TYPE_LATEST)
    else:
        # Fallback: manual text format
        lines = ['# HELP cadqstream_metrics CA-DQStream pipeline metrics', '# TYPE cadqstream_metrics untyped']
        with _metrics_lock:
            for name, label_map in _counters.items():
                for lhash, val in label_map.items():
                    import json
                    labels = json.loads(lhash)
                    lstr = '{' + ','.join(f'{k}="{v}"' for k, v in sorted(labels.items())) + '}' if labels else ''
                    lines.append(f'{name}{lstr} {val}')
            for name, label_map in _gauges.items():
                for lhash, val in label_map.items():
                    import json
                    labels = json.loads(lhash)
                    lstr = '{' + ','.join(f'{k}="{v}"' for k, v in sorted(labels.items())) + '}' if labels else ''
                    lines.append(f'{name}{lstr} {val}')
        return Response('\n'.join(lines) + '\n', mimetype='text/plain; version=0.0.4')

@app.route('/health')
def health():
    return {'status': 'ok', 'uptime': time.time()}

if __name__ == '__main__':
    from werkzeug.serving import make_server
    server = make_server('0.0.0.0', 9250, app)
    server.serve_forever()
