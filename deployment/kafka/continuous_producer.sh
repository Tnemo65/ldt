#!/bin/bash
# Continuous producer wrapper using continuous_data_simulator.py
# Reads from local parquet/CSV and loops indefinitely
while true; do
    echo "[$(date)] Starting continuous data simulator..."
    python3 /app/continuous_data_simulator.py \
        --input /data/nyc_taxi_300k.parquet \
        --topic "${TOPIC_RAW:-taxi-nyc-raw-v2}" \
        --delay 0.05 \
        || true
    echo "[$(date)] Simulator exited, restarting in 5s..."
    sleep 5
done
