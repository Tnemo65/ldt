#!/usr/bin/env python3
"""Run E2E pipeline and capture errors."""
import subprocess
import sys

result = subprocess.run(
    ["python3", "e2e_pipeline_submit.py"],
    cwd="/opt/flink/e2e",
    capture_output=True,
    text=True,
    timeout=120,
    env={
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": "/opt/flink/e2e:/opt/flink/e2e/src:/tmp:/tmp/src",
        "KAFKA_BOOTSTRAP_SERVERS": "kafka:9092",
        "POSTGRES_HOST": "postgres",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "dq_pipeline",
        "POSTGRES_USER": "cadqstream",
        "POSTGRES_PASSWORD": "cadqstream123",
        "PYFLINK_SESSION_JOB_ENABLED": "true",
    }
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("RC:", result.returncode)
sys.exit(result.returncode)
