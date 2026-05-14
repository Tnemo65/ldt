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

# Pre-create all cadqstream metrics so HELP/TYPE lines always appear in /metrics output,
# with the correct label sets for each metric.
if PROMETHEUS_AVAILABLE:
    _counters_prom['cadqstream_records_input_total'] = Counter(
        'cadqstream_records_input_total', 'CA-DQStream input record counter',
        ['topic'], registry=_registry)
    _counters_prom['cadqstream_records_valid_total'] = Counter(
        'cadqstream_records_valid_total', 'CA-DQStream valid record counter',
        ['layer'], registry=_registry)
    _counters_prom['cadqstream_records_violation_total'] = Counter(
        'cadqstream_records_violation_total', 'CA-DQStream violation counter',
        ['layer', 'type'], registry=_registry)
    _counters_prom['cadqstream_violation_records_total'] = Counter(
        'cadqstream_violation_records_total', 'CA-DQStream records-with-violation counter',
        ['layer', 'type'], registry=_registry)
    _counters_prom['cadqstream_anomalies_canary_total'] = Counter(
        'cadqstream_anomalies_canary_total', 'CA-DQStream canary rule anomaly counter',
        ['layer', 'rule'], registry=_registry)
    _counters_prom['cadqstream_anomalies_ml_total'] = Counter(
        'cadqstream_anomalies_ml_total', 'CA-DQStream ML anomaly counter',
        ['layer', 'neighborhood'], registry=_registry)
    _counters_prom['cadqstream_iec_decisions_total'] = Counter(
        'cadqstream_iec_decisions_total', 'CA-DQStream IEC decision counter',
        ['strategy'], registry=_registry)
    _counters_prom['cadqstream_iec_drift_detected_total'] = Counter(
        'cadqstream_iec_drift_detected_total', 'CA-DQStream IEC drift counter',
        ['neighborhood'], registry=_registry)
    _gauges_prom['cadqstream_meta_volume'] = Gauge(
        'cadqstream_meta_volume', 'CA-DQStream meta volume gauge',
        ['neighborhood'], registry=_registry)
    _gauges_prom['cadqstream_meta_anomaly_rate'] = Gauge(
        'cadqstream_meta_anomaly_rate', 'CA-DQStream meta anomaly rate gauge',
        ['neighborhood'], registry=_registry)
    _gauges_prom['cadqstream_meta_null_rate'] = Gauge(
        'cadqstream_meta_null_rate', 'CA-DQStream meta null rate gauge',
        ['neighborhood'], registry=_registry)
    _gauges_prom['cadqstream_meta_delta_score'] = Gauge(
        'cadqstream_meta_delta_score', 'CA-DQStream meta delta score gauge',
        ['neighborhood'], registry=_registry)
    _gauges_prom['cadqstream_iec_confidence'] = Gauge(
        'cadqstream_iec_confidence', 'CA-DQStream IEC confidence gauge',
        ['neighborhood'], registry=_registry)

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
