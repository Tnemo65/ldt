#!/bin/bash
PYTHONPATH=/tmp:/tmp/src KAFKA_BOOTSTRAP_SERVERS=ldt-kafka-1:9092 python3 /tmp/flink_job.py >> /tmp/flink_job_run.log 2>&1
echo "Exit code: $?" >> /tmp/flink_job_run.log
