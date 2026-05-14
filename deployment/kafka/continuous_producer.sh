#!/bin/bash
# Continuous producer wrapper
while true; do
    echo "[$(date)] Starting Kafka producer..."
    python3 /app/fast_producer.py 100 kafka:9092 taxi-nyc-raw || true
    echo "[$(date)] Producer exited, restarting in 5s..."
    sleep 5
done
