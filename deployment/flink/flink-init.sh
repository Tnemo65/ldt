#!/bin/bash
set -e

FLINK_REST="${FLINK_REST:-http://flink-jobmanager:8081}"
FLINK_TIMEOUT=120
PYTHON_PATH="/opt/flink/pyflink_extracted:/opt/flink/opt/python/py4j-0.10.9.7-src.zip:/opt/flink/opt/python/cloudpickle-2.2.0-src.zip:/opt/flink/e2e"

echo "[flink-init] Waiting for Flink REST API at ${FLINK_REST}..."
ELAPSED=0
while ! curl -sf "${FLINK_REST}/overview" > /dev/null 2>&1; do
    echo "[flink-init] Flink not ready (${ELAPSED}s/${FLINK_TIMEOUT}s)..."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $ELAPSED -ge $FLINK_TIMEOUT ]; then
        echo "[flink-init] ERROR: Flink did not become ready within ${FLINK_TIMEOUT}s"
        exit 1
    fi
done
echo "[flink-init] Flink REST API is ready!"

echo "[flink-init] Checking cluster status..."
curl -s "${FLINK_REST}/overview" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  TaskManagers: {d.get(\"taskmanagers\",0)}'); print(f'  Slots total: {d.get(\"taskmanager\",{}).get(\"totalTaskManagerSlotNumber\",0)}'); print(f'  Slots free: {d.get(\"taskmanager\",{}).get(\"totalAvailableSlotNumber\",0)}')"

echo "[flink-init] Checking for existing jobs..."
EXISTING=$(curl -s "${FLINK_REST}/jobs" | python3 -c "import sys,json; d=json.load(sys.stdin); jobs=d.get('jobs',[]); print(len(jobs))" 2>/dev/null || echo "0")
echo "[flink-init] Existing jobs: ${EXISTING}"

if [ "$EXISTING" -gt 0 ]; then
    echo "[flink-init] Jobs already running - skipping submission"
    curl -s "${FLINK_REST}/jobs" | python3 -c "
import sys,json
d=json.load(sys.stdin)
for j in d.get('jobs',[]):
    state = j.get('state', 'UNKNOWN')
    print(f\"  - {j['id'][:16]}... [{state}]\")
" || echo "[flink-init] WARNING: Could not parse jobs response"
    exit 0
fi

echo "[flink-init] Setting up environment variables..."
export PYTHONPATH="${PYTHON_PATH}"
export KAFKA_BOOTSTRAP_SERVERS="kafka:9092"
export MINIO_ENDPOINT="minio:9000"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin123"
export FLINK_ENV="production"

cd /opt/flink/e2e

echo "[flink-init] Activating Python venv and verifying dependencies..."
source /opt/venv/bin/activate

# Verify critical dependencies are installed
python -c "import mmh3; print('mmh3 OK')" 2>&1 || echo "WARNING: mmh3 not found"
python -c "import kafka; print('kafka-python OK')" 2>&1 || echo "WARNING: kafka-python not found"
python -c "import pandas; print('pandas OK')" 2>&1 || echo "WARNING: pandas not found"
python -c "import river; print('river OK')" 2>&1 || echo "WARNING: river not found"
python -c "import boto3; print('boto3 OK')" 2>&1 || echo "WARNING: boto3 not found"
python -c "import minio; print('minio OK')" 2>&1 || echo "WARNING: minio not found"

echo "[flink-init] PYTHONPATH: $PYTHONPATH"
echo "[flink-init] Python: $(which python)"
echo "[flink-init] Submitting Flink job via flink CLI..."

# Submit with explicit -m flag to specify JobManager address
# This fixes the "Connection refused: /0.0.0.0:8081" error
flink run -m flink-jobmanager:8081 -d \
    -py /opt/flink/e2e/src/flink_job_complete.py \
    2>&1 || true

echo "[flink-init] Monitoring for 90 seconds..."
sleep 90

NEW_JOBS=$(curl -s "${FLINK_REST}/jobs" | python3 -c "import sys,json; d=json.load(sys.stdin); jobs=d.get('jobs',[]); print(len(jobs))" 2>/dev/null || echo "0")
echo "[flink-init] Jobs after submission: ${NEW_JOBS}"

if [ "$NEW_JOBS" -gt 0 ]; then
    echo "[flink-init] SUCCESS: Job is running!"
    curl -s "${FLINK_REST}/jobs" | python3 -c "
import sys,json
d=json.load(sys.stdin)
for j in d.get('jobs',[]):
    state = j.get('state', 'UNKNOWN')
    print(f\"  - {j['id']} [{state}]\")
" || echo "[flink-init] WARNING: Could not parse jobs response"
else
    echo "[flink-init] WARNING: No jobs found after submission. Check Flink UI logs."
fi

echo "[flink-init] Init complete."
