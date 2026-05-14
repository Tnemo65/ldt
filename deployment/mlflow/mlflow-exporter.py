#!/usr/bin/env python3
"""
MLflow Prometheus Metrics Exporter
Exposes MLflow tracking metrics in Prometheus format.

Metrics exposed:
- mlflow_experiments_total: Total number of MLflow experiments
- mlflow_runs_active: Number of active MLflow runs
- mlflow_runs_completed: Number of completed MLflow runs
- mlflow_exporter_requests_total: Total exporter scrape requests
"""

import os
import time
from flask import Flask, Response
import requests
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')

# Prometheus metrics
mlflow_experiments_total = Gauge(
    'mlflow_experiments_total',
    'Total number of MLflow experiments'
)
mlflow_runs_active = Gauge(
    'mlflow_runs_active',
    'Number of active MLflow runs'
)
mlflow_runs_completed = Gauge(
    'mlflow_runs_completed',
    'Number of completed MLflow runs'
)
mlflow_exporter_requests = Counter(
    'mlflow_exporter_requests_total',
    'Total exporter scrape requests',
    ['status']
)


def fetch_mlflow_metrics():
    """Fetch metrics from MLflow tracking API."""
    experiments_count = 0
    active_runs = 0
    completed_runs = 0

    try:
        # List experiments
        resp = requests.get(
            f'{MLFLOW_TRACKING_URI}/api/2.0/preview/mlflow/experiments/list',
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            experiments = data.get('experiments', [])
            # Exclude default experiment
            experiments_count = len([
                e for e in experiments
                if e.get('name') != 'Default'
            ])

            # Count runs per experiment
            for exp in experiments:
                exp_id = exp.get('experiment_id')
                if exp_id:
                    try:
                        runs_resp = requests.post(
                            f'{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/search',
                            json={'experiment_ids': [exp_id], 'max_results': 1000},
                            timeout=5
                        )
                        if runs_resp.status_code == 200:
                            runs = runs_resp.json().get('runs', [])
                            for run in runs:
                                status = run.get('info', {}).get('status', '')
                                if status == 'RUNNING':
                                    active_runs += 1
                                elif status == 'FINISHED':
                                    completed_runs += 1
                    except Exception:
                        pass
    except Exception as e:
        print(f"Error fetching MLflow metrics: {e}")

    return experiments_count, active_runs, completed_runs


@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    try:
        experiments, active, completed = fetch_mlflow_metrics()
        mlflow_experiments_total.set(experiments)
        mlflow_runs_active.set(active)
        mlflow_runs_completed.set(completed)
        mlflow_exporter_requests.labels(status='success').inc()
    except Exception as e:
        mlflow_exporter_requests.labels(status='error').inc()
        print(f"Error updating metrics: {e}")

    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


@app.route('/health')
def health():
    """Health check endpoint."""
    return 'OK'


@app.route('/')
def index():
    """Root endpoint with usage info."""
    return '''
    <html>
    <head><title>MLflow Prometheus Exporter</title></head>
    <body>
        <h1>MLflow Prometheus Exporter</h1>
        <p>Metrics endpoint: <a href="/metrics">/metrics</a></p>
        <p>Health endpoint: <a href="/health">/health</a></p>
    </body>
    </html>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9251)
