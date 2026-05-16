#!/bin/bash
# Flink job submission script with full idempotency.
#
# Idempotency guarantees:
#   1. Only submits if NO jobs are in RUNNING state (RESTARTING and CREATED are not considered healthy).
#   2. If the job name already exists in RUNNING state, skip submission.
#   3. If submission fails, retries up to 3 times with 30s backoff.
#   4. Reports job status after submission attempt.
#
# ROOT CAUSE FIX #3: Prevents restart loops caused by repeated submission
# of already-running jobs on container rebuild.

set -e

FLINK_REST="${FLINK_REST:-http://flink-jobmanager:8081}"
FLINK_TIMEOUT=120
FLINK_SUBMIT_RETRIES=3
FLINK_SUBMIT_BACKOFF=30
PYTHON_PATH="/opt/flink/pyflink_extracted:/opt/flink/opt/python/py4j-0.10.9.7-src.zip:/opt/flink/opt/python/cloudpickle-2.2.0-src.zip:/opt/flink/e2e"
FLINK_JOB_NAME="CA-DQStream Complete Pipeline - 4 Layers"

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

# ─── Idempotency check: look for RUNNING jobs with the same name ───────────
echo "[flink-init] Checking for existing RUNNING jobs..."
EXISTING_JOBS=$(curl -s "${FLINK_REST}/jobs" | python3 -c "
import sys,json
d=json.load(sys.stdin)
running = [j for j in d.get('jobs',[]) if j.get('state') == 'RUNNING']
print(f'RUNNING={len(running)}')
for j in running:
    print(f'  - {j[\"id\"]} [{j.get(\"state\",\"?\")}] name={j.get(\"name\",\"?\")}')
" 2>/dev/null || echo "RUNNING=0")

RUNNING_COUNT=$(echo "$EXISTING_JOBS" | grep '^RUNNING=' | cut -d= -f2)
if [ "${RUNNING_COUNT:-0}" -gt 0 ]; then
    echo "[flink-init] Found ${RUNNING_COUNT} RUNNING job(s) — checking for name match..."
    JOB_RUNNING=$(echo "$EXISTING_JOBS" | grep -c "name=${FLINK_JOB_NAME}" || true)
    if [ "${JOB_RUNNING:-0}" -gt 0 ]; then
        echo "[flink-init] Job '${FLINK_JOB_NAME}' is already RUNNING — skipping submission (idempotent)"
        echo "$EXISTING_JOBS" | grep "name=${FLINK_JOB_NAME}" || true
        exit 0
    fi
    # Other jobs are running — skip to avoid resource contention
    echo "[flink-init] Other jobs are RUNNING but not '${FLINK_JOB_NAME}' — skipping submission"
    echo "$EXISTING_JOBS" | grep '^  -' || true
    exit 0
fi

echo "[flink-init] No RUNNING jobs found — proceeding with submission..."

echo "[flink-init] Setting up environment variables..."
export PYTHONPATH="${PYTHON_PATH}"
export KAFKA_BOOTSTRAP_SERVERS="kafka:9092"
export MINIO_ENDPOINT="minio:9000"
export MINIO_ACCESS_KEY="${MINIO_ROOT_USER}"
export MINIO_SECRET_KEY="${MINIO_ROOT_PASSWORD}"
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

# ─── Submission with retry ─────────────────────────────────────────────────
SUBMIT_STATUS=1
for attempt in $(seq 1 $FLINK_SUBMIT_RETRIES); do
    echo "[flink-init] Submitting Flink job (attempt ${attempt}/${FLINK_SUBMIT_RETRIES})..."

    if flink run -m flink-jobmanager:8081 -d \
        -py /opt/flink/e2e/src/flink_job_complete.py \
        2>&1 | tee /tmp/flink_submit_${attempt}.log; then
        echo "[flink-init] Submission command completed."
        SUBMIT_STATUS=0
        break
    else
        SUBMIT_STATUS=$?
        echo "[flink-init] Submission attempt ${attempt} failed (exit ${SUBMIT_STATUS})"
        if [ $attempt -lt $FLINK_SUBMIT_RETRIES ]; then
            echo "[flink-init] Waiting ${FLINK_SUBMIT_BACKOFF}s before retry..."
            sleep $FLINK_SUBMIT_BACKOFF
        fi
    fi
done

if [ $SUBMIT_STATUS -ne 0 ]; then
    echo "[flink-init] ERROR: All submission attempts failed. Check logs above."
    for attempt in $(seq 1 $FLINK_SUBMIT_RETRIES); do
        if [ -f "/tmp/flink_submit_${attempt}.log" ]; then
            echo "=== Submission log ${attempt} ==="
            tail -30 /tmp/flink_submit_${attempt}.log
        fi
    done
fi

echo "[flink-init] Verifying job status after submission..."
sleep 10

FINAL_JOBS=$(curl -s "${FLINK_REST}/jobs" | python3 -c "
import sys,json
d=json.load(sys.stdin)
jobs = d.get('jobs',[])
print(f'Total jobs: {len(jobs)}')
for j in jobs:
    print(f'  - {j[\"id\"][:16]}... [{j.get(\"state\",\"?\")}] name={j.get(\"name\",\"?\")}')
" 2>/dev/null || echo "Could not fetch job list")

echo "$FINAL_JOBS"

RUNNING_AFTER=$(echo "$FINAL_JOBS" | grep -c '\[RUNNING\]' || echo 0)
if [ "${RUNNING_AFTER:-0}" -gt 0 ]; then
    echo "[flink-init] SUCCESS: ${RUNNING_AFTER} job(s) now in RUNNING state."
else
    echo "[flink-init] WARNING: No jobs in RUNNING state after submission. Check Flink UI logs."
fi

echo "[flink-init] Init complete."
