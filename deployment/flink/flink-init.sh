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
    print(f\"  - {j['id'][:16]}... [{j['state']}]\")
"
    exit 0
fi

echo "[flink-init] Submitting Flink job via REST API..."
export PYTHONPATH="${PYTHON_PATH}"
export KAFKA_BOOTSTRAP_SERVERS="kafka:9092"
export PGBOUNCER_HOST="pgbouncer"
export PGBOUNCER_PORT="5432"
export POSTGRES_DB="dq_pipeline"
export POSTGRES_USER="cadqstream"
export POSTGRES_PASSWORD="cadqstream123"
export MINIO_ENDPOINT="minio:9000"
export MINIO_ACCESS_KEY="minioadmin"
export MINIO_SECRET_KEY="minioadmin123"
export FLINK_ENV="production"

cd /opt/flink/e2e

RESPONSE=$(curl -s -X POST "${FLINK_REST}/jobs" \
  -H "Content-Type: application/json" \
  -d "{
    \"programArgs\": \"\",
    \"entryClass\": \"\",
    \"savepointPath\": null,
    \"allowNonRestoredState\": false
  }")

JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id',''))" 2>/dev/null || echo "")

if [ -z "$JOB_ID" ]; then
    echo "[flink-init] REST submission may have failed. Response: $RESPONSE"
    echo "[flink-init] Trying flink run command..."

    flink run -d \
        -py /opt/flink/e2e/src/flink_job_complete.py \
        2>&1 || true

    sleep 5
else
    echo "[flink-init] Job submitted with ID: ${JOB_ID}"
fi

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
    print(f\"  - {j['id']} [{j['state']}]\")
"
else
    echo "[flink-init] WARNING: No jobs found after submission. Check Flink UI logs."
fi

echo "[flink-init] Init complete."
