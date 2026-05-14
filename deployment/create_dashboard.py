#!/usr/bin/env python3
"""Create Grafana dashboard via API."""
import json
import base64
import urllib.request

GRAFANA_URL = "http://localhost:3000"
AUTH = ("admin", "admin123")

dashboard = {
    "dashboard": {
        "title": "CA-DQStream Overview",
        "uid": "cadqstream-overview",
        "refresh": "10s",
        "panels": [
            {
                "id": 1, "title": "Services Online", "type": "stat",
                "gridPos": {"h": 4, "w": 4, "x": 0, "y": 0},
                "targets": [{"expr": 'count(up{job!="prometheus"})', "legendFormat": "Services Up", "refId": "A"}]
            },
            {
                "id": 2, "title": "Kafka Consumer Lag", "type": "stat",
                "gridPos": {"h": 4, "w": 4, "x": 4, "y": 0},
                "targets": [{"expr": 'sum(kafka_consumergroup_lag{topic="taxi-nyc-raw"})', "legendFormat": "Total Lag", "refId": "A"}]
            },
            {
                "id": 3, "title": "MinIO API Requests", "type": "stat",
                "gridPos": {"h": 4, "w": 4, "x": 8, "y": 0},
                "targets": [{"expr": 'sum(rate(s3_requests_success_total[5m]))', "legendFormat": "Requests/sec", "refId": "A"}]
            },
            {
                "id": 4, "title": "CPU Usage", "type": "stat",
                "gridPos": {"h": 4, "w": 4, "x": 12, "y": 0},
                "targets": [{"expr": '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)', "legendFormat": "CPU %", "refId": "A"}]
            },
            {
                "id": 5, "title": "Memory Usage", "type": "stat",
                "gridPos": {"h": 4, "w": 4, "x": 16, "y": 0},
                "targets": [{"expr": '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100', "legendFormat": "Mem %", "refId": "A"}]
            },
            {
                "id": 10, "title": "Kafka Consumer Lag (per partition)", "type": "timeseries",
                "gridPos": {"h": 6, "w": 12, "x": 0, "y": 4},
                "targets": [{"expr": 'kafka_consumergroup_lag{topic="taxi-nyc-raw"}', "legendFormat": "{{consumergroup}} P{{partition}}", "refId": "A"}]
            },
            {
                "id": 11, "title": "Kafka Topic Offsets", "type": "timeseries",
                "gridPos": {"h": 6, "w": 12, "x": 12, "y": 4},
                "targets": [{"expr": 'kafka_topic_partition_current_offset{topic="taxi-nyc-raw"}', "legendFormat": "P{{partition}}", "refId": "A"}]
            },
            {
                "id": 20, "title": "MinIO S3 Request Rate", "type": "timeseries",
                "gridPos": {"h": 5, "w": 8, "x": 0, "y": 10},
                "targets": [
                    {"expr": 'rate(s3_requests_success_total[5m])', "legendFormat": "Success/sec", "refId": "A"},
                    {"expr": 'rate(s3_requests_error_total[5m])', "legendFormat": "Error/sec", "refId": "B"}
                ]
            },
            {
                "id": 21, "title": "MinIO S3 Latency", "type": "timeseries",
                "gridPos": {"h": 5, "w": 8, "x": 8, "y": 10},
                "targets": [
                    {"expr": 'histogram_quantile(0.99, rate(s3_request_duration_seconds_bucket[5m]))', "legendFormat": "P99 Latency", "refId": "A"},
                    {"expr": 'histogram_quantile(0.50, rate(s3_request_duration_seconds_bucket[5m]))', "legendFormat": "P50 Latency", "refId": "B"}
                ]
            },
            {
                "id": 22, "title": "Under-replicated Partitions", "type": "stat",
                "gridPos": {"h": 5, "w": 4, "x": 16, "y": 10},
                "targets": [{"expr": 'count(kafka_topic_partition_under_replicated_partition > 0)', "legendFormat": "URP Count", "refId": "A"}]
            },
            {
                "id": 23, "title": "Kafka Brokers", "type": "stat",
                "gridPos": {"h": 5, "w": 4, "x": 20, "y": 10},
                "targets": [{"expr": 'kafka_broker_info', "legendFormat": "Brokers", "refId": "A"}]
            },
            {
                "id": 30, "title": "CPU Usage %", "type": "timeseries",
                "gridPos": {"h": 5, "w": 8, "x": 0, "y": 15},
                "targets": [{"expr": '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)', "legendFormat": "CPU %", "refId": "A"}]
            },
            {
                "id": 31, "title": "Memory Usage", "type": "timeseries",
                "gridPos": {"h": 5, "w": 8, "x": 8, "y": 15},
                "targets": [
                    {"expr": 'node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes', "legendFormat": "Used", "refId": "A"},
                    {"expr": 'node_memory_MemAvailable_bytes', "legendFormat": "Available", "refId": "B"}
                ]
            },
            {
                "id": 32, "title": "Network Traffic (bytes/s)", "type": "timeseries",
                "gridPos": {"h": 5, "w": 8, "x": 16, "y": 15},
                "targets": [
                    {"expr": 'rate(node_network_receive_bytes_total{device!="lo"}[1m])', "legendFormat": "{{device}} RX", "refId": "A"},
                    {"expr": 'rate(node_network_transmit_bytes_total{device!="lo"}[1m])', "legendFormat": "{{device}} TX", "refId": "B"}
                ]
            },
            {
                "id": 40, "title": "System Load", "type": "timeseries",
                "gridPos": {"h": 5, "w": 8, "x": 0, "y": 20},
                "targets": [
                    {"expr": 'node_load1', "legendFormat": "Load 1m", "refId": "A"},
                    {"expr": 'node_load5', "legendFormat": "Load 5m", "refId": "B"}
                ]
            },
            {
                "id": 41, "title": "Disk I/O", "type": "timeseries",
                "gridPos": {"h": 5, "w": 8, "x": 8, "y": 20},
                "targets": [
                    {"expr": 'rate(node_disk_read_bytes_total[5m])', "legendFormat": "Read", "refId": "A"},
                    {"expr": 'rate(node_disk_written_bytes_total[5m])', "legendFormat": "Write", "refId": "B"}
                ]
            },
            {
                "id": 42, "title": "MinIO Disk Usage", "type": "timeseries",
                "gridPos": {"h": 5, "w": 8, "x": 16, "y": 20},
                "targets": [{"expr": 'node_filesystem_size_bytes{mountpoint="/"} - node_filesystem_avail_bytes{mountpoint="/"}', "legendFormat": "Used", "refId": "A"}]
            }
        ]
    },
    "overwrite": True,
    "message": "CA-DQStream overview dashboard created"
}

payload = json.dumps(dashboard).encode('utf-8')
auth_str = base64.b64encode(b'admin:admin123').decode('utf-8')
req = urllib.request.Request(
    f"{GRAFANA_URL}/api/dashboards/db",
    data=payload,
    headers={
        'Authorization': f'Basic {auth_str}',
        'Content-Type': 'application/json'
    },
    method='POST'
)

try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode('utf-8'))
        print(f"Status: {result.get('status')}")
        print(f"Message: {result.get('message')}")
        print(f"Dashboard URL: /d/{result.get('uid')}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
