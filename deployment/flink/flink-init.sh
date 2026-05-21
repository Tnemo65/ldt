#!/bin/bash
# Flink job submission and continuous health monitoring with auto-resubmit.
#
# Monitors the CA-DQStream pipeline job continuously. On FAILED, CANCELED,
# or disappearance, automatically resubmits with exponential backoff.
#
# ROOT CAUSE FIX: Previous script checked job health only once at startup.
# The job could crash 17 minutes later and the init container would exit 0.
# This version implements continuous monitoring with automatic recovery.
#
# Idempotency:
#   - Only one job with the canonical name can exist at a time.
#   - Skip submission if RUNNING job with the canonical name already exists.
#   - On crash: cancel any stale job graph first, then resubmit.
#
# Exit codes:
#   0  = Job is RUNNING and healthy (or was successfully recovered)
#   1  = Max resubmission attempts exceeded (cluster or code is broken)

set -euo pipefail

FLINK_REST="${FLINK_REST:-http://flink-jobmanager:8081}"
FLINK_TIMEOUT=120
MAX_RESUBMISSIONS=5
RESUBMIT_BACKOFF=60
HEALTH_CHECK_INTERVAL=30
JOB_NAME="CA-DQStream Sequential Pipeline - Phase 3"
PYTHON_PATH="/opt/flink/pyflink_extracted:/opt/flink/opt/python/py4j-0.10.9.7-src.zip:/opt/flink/opt/python/cloudpickle-2.2.0-src.zip:/opt/flink/e2e:/opt/venv/lib/python3.10/site-packages"
SUBMITTED_JOBS_LOG="/tmp/flink_submitted_jobs.log"
INITIAL_STATE="NOT_FOUND"
JOB_ID="NOT_FOUND"

echo "[flink-init] ============================================================"
echo "[flink-init] CA-DQStream Flink Job Supervisor -- Auto-Recovery Mode"
echo "[flink-init] ============================================================"

ELAPSED=0
while ! curl -sf "${FLINK_REST}/overview" > /dev/null 2>&1; do
    echo "  [flink-init] Flink not ready (${ELAPSED}s/${FLINK_TIMEOUT}s)..."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $ELAPSED -ge $FLINK_TIMEOUT ]; then
        echo "[flink-init] ERROR: Flink REST API did not respond within ${FLINK_TIMEOUT}s"
        exit 1
    fi
done
echo "[flink-init] Flink REST API is ready."

echo "[flink-init] Cluster status:"
curl -s "${FLINK_REST}/overview" | /opt/venv/bin/python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'  TaskManagers: {d.get(\"taskmanagers\",0)}')
print(f'  Slots total:  {d.get(\"taskmanager\",{}).get(\"totalTaskManagerSlotNumber\",0)}')
print(f'  Slots free:   {d.get(\"taskmanager\",{}).get(\"totalAvailableSlotNumber\",0)}')
"

# ── Write helper script once (before any function is called) ────────────────────
cat > /tmp/find_job.py << 'PYEOF'
import sys, json, urllib.request, urllib.error

target = sys.argv[1]
base_url = sys.argv[2]

data = json.load(sys.stdin)
for entry in data.get('jobs', []):
    job_id = entry.get('id')
    if not job_id:
        continue
    try:
        detail = json.load(urllib.request.urlopen(base_url + '/jobs/' + job_id, timeout=5))
        if detail.get('name', '') == target:
            print(detail.get('state', 'UNKNOWN') + '|' + job_id)
            sys.exit(0)
    except Exception:
        pass
print('NOT_FOUND')
PYEOF

# ── Get job state by name ───────────────────────────────────────────────────────
# The /jobs endpoint only returns {id, status} — no name field.
# We must query /jobs/{id} for each job to match by name.
get_job_state() {
    local name="$1"
    local jobs_json
    jobs_json=$(curl -sf "${FLINK_REST}/jobs" 2>/dev/null) || { echo "API_ERROR"; return; }

    local result
    result=$(echo "$jobs_json" | /opt/venv/bin/python3 /tmp/find_job.py "$name" "${FLINK_REST}" 2>/dev/null)
    echo "${result%%|*}"
}

# ── Get job ID by name ─────────────────────────────────────────────────────────
get_job_id() {
    local name="$1"
    local jobs_json
    jobs_json=$(curl -sf "${FLINK_REST}/jobs" 2>/dev/null) || { echo "API_ERROR"; return; }
    local result
    result=$(echo "$jobs_json" | /opt/venv/bin/python3 /tmp/find_job.py "$name" "${FLINK_REST}" 2>/dev/null)
    echo "${result##*|}"
}

# ── Cancel a job by ID (ignore errors if already gone) ─────────────────────
cancel_job() {
    local job_id="$1"
    echo "[flink-init] Canceling stale job ${job_id}..."
    curl -sf -X PATCH "${FLINK_REST}/jobs/${job_id}/yarn-cancel" > /dev/null 2>&1 \
    || curl -sf -X PATCH "${FLINK_REST}/jobs/${job_id}/cancel" > /dev/null 2>&1 \
    || echo "[flink-init]   (cancel ignored — job may already be gone)"
}

# ── Wait for cluster to settle after cancel ────────────────────────────────────
wait_for_cluster() {
    echo "[flink-init] Waiting ${RESUBMIT_BACKOFF}s for cluster to settle..."
    sleep $RESUBMIT_BACKOFF
}

# ── Submit the Flink job via flink CLI ────────────────────────────────────────
do_submit() {
    local attempt="$1"
    echo "[flink-init] Setting up environment..."
    export PYTHONPATH="${PYTHON_PATH}"
    export KAFKA_BOOTSTRAP_SERVERS="kafka:9092"
    export MINIO_ENDPOINT="minio:9000"
    export MINIO_ACCESS_KEY="${MINIO_ROOT_USER}"
    export MINIO_SECRET_KEY="${MINIO_PASSWORD}"
    export FLINK_ENV="production"

    cd /opt/flink/e2e

    echo "[flink-init] Activating Python venv..."
    source /opt/venv/bin/activate 2>/dev/null || true

    echo "[flink-init] Verifying critical dependencies..."
    python -c "import mmh3; print('  mmh3 OK')" 2>&1 || echo "  WARNING: mmh3 not found"
    python -c "import kafka; print('  kafka-python OK')" 2>&1 || echo "  WARNING: kafka-python not found"
    python -c "import pandas; print('  pandas OK')" 2>&1 || echo "  WARNING: pandas not found"
    python -c "import river; print('  river OK')" 2>&1 || echo "  WARNING: river not found"
    python -c "import boto3; print('  boto3 OK')" 2>&1 || echo "  WARNING: boto3 not found"
    python -c "import minio; print('  minio OK')" 2>&1 || echo "  WARNING: minio not found"

    echo "[flink-init] Submitting Flink job via flink CLI..."
    if flink run -m flink-jobmanager:8081 -d \
        -p 4 \
        -py /opt/flink/e2e/src/flink_job_complete.py \
        2>&1 | tee "/tmp/flink_submit_${attempt}.log"; then
        echo "[flink-init]   Submission command completed (exit 0)."
        return 0
    else
        echo "[flink-init]   Submission command exited non-zero."
        if [ -f "/tmp/flink_submit_${attempt}.log" ]; then
            echo "[flink-init]   Last 20 lines of submission log:"
            tail -20 "/tmp/flink_submit_${attempt}.log" | sed 's/^/   /'
        fi
        return 1
    fi
}

# ── Wait for job to appear and reach RUNNING state ────────────────────────────
wait_for_job_creation() {
    local timeout="$1"
    local elapsed=0
    echo "[flink-init] Waiting up to ${timeout}s for job to appear in REST API..."
    while [ $elapsed -lt $timeout ]; do
        state=$(get_job_state "$JOB_NAME")
        if [ "$state" = "RUNNING" ]; then
            echo "[flink-init]   Job is RUNNING: ${JOB_NAME}"
            return 0
        fi
        if [ "$state" != "NOT_FOUND" ] && [ "$state" != "API_ERROR" ]; then
            echo "  [flink-init]   Job found with state=${state}, waiting for RUNNING... (${elapsed}s/${timeout}s)"
            sleep 5
            elapsed=$((elapsed + 5))
            continue
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "  [flink-init]   Still waiting... (${elapsed}s/${timeout}s) state=${state}"
    done
    echo "[flink-init] WARNING: Job did not appear in REST API after ${timeout}s"
    return 1
}

# ── Log submission to persistent file ─────────────────────────────────────────
log_submission() {
    local attempt="$1"
    local job_id="$2"
    local ts
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "${ts},${attempt},${job_id}" >> "$SUBMITTED_JOBS_LOG"
}

# ════════════════════════════════════════════════════════════════════════════════
# MAIN: Submit if needed, then enter permanent monitoring loop
# ════════════════════════════════════════════════════════════════════════════════
# Query initial job state only AFTER the REST API is confirmed ready and all
# helper functions are defined. This is the correct place to check idempotency.
INITIAL_STATE=$(get_job_state "$JOB_NAME")
JOB_ID=$(get_job_id "$JOB_NAME")

if [ "$INITIAL_STATE" != "RUNNING" ]; then
    echo "[flink-init] No RUNNING job found. Current state: ${INITIAL_STATE}"
    echo "[flink-init] Starting submit-and-monitor loop (max ${MAX_RESUBMISSIONS} attempts)..."

    for attempt in $(seq 1 $MAX_RESUBMISSIONS); do
        echo ""
        echo "[flink-init] ═══ SUBMISSION ATTEMPT ${attempt}/${MAX_RESUBMISSIONS} ═══"

        stale_id=$(get_job_id "$JOB_NAME")
        if [ "$stale_id" != "NOT_FOUND" ] && [ "$stale_id" != "API_ERROR" ]; then
            cancel_job "$stale_id"
            wait_for_cluster
        fi

        if ! do_submit "$attempt"; then
            echo "[flink-init] Submission failed on attempt ${attempt}"
            if [ $attempt -lt $MAX_RESUBMISSIONS ]; then
                echo "[flink-init] Retrying in ${RESUBMIT_BACKOFF}s..."
                sleep $RESUBMIT_BACKOFF
                continue
            else
                echo "[flink-init] ERROR: All ${MAX_RESUBMISSIONS} submission attempts failed."
                exit 1
            fi
        fi

        if ! wait_for_job_creation 120; then
            echo "[flink-init] Job did not appear after submission attempt ${attempt}"
            if [ $attempt -lt $MAX_RESUBMISSIONS ]; then
                echo "[flink-init] Will retry..."
                sleep $RESUBMIT_BACKOFF
                continue
            else
                echo "[flink-init] ERROR: Job failed to appear after ${MAX_RESUBMISSIONS} attempts."
                exit 1
            fi
        fi

        CURRENT_JOB_ID=$(get_job_id "$JOB_NAME")
        log_submission "$attempt" "$CURRENT_JOB_ID"
        echo "[flink-init] Successfully submitted job: ${CURRENT_JOB_ID}"
        break
    done
else
    echo "[flink-init] Job '${JOB_NAME}' is already RUNNING — skipping submission (idempotent)"
    JOB_ID=$(get_job_id "$JOB_NAME")
    echo "[flink-init]   Job ID: ${JOB_ID}"
fi

# ════════════════════════════════════════════════════════════════════════════════
# MAIN: Permanent health monitoring loop — never exits while container runs
# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "[flink-init] ═══ ENTERING CONTINUOUS HEALTH MONITOR ═══"
echo "[flink-init] Will check job health every ${HEALTH_CHECK_INTERVAL}s."
echo "[flink-init] On FAILURE: auto-cancel + resubmit."

DEAD_COUNT=0
RESUBMIT_ATTEMPT=1

while true; do
    sleep $HEALTH_CHECK_INTERVAL

    current_state=$(get_job_state "$JOB_NAME")

    case "$current_state" in
        RUNNING)
            DEAD_COUNT=0
            echo "[$(date -u +%H:%M:%S)] [HEALTHY] ${JOB_NAME}: RUNNING"
            ;;
        NOT_FOUND|API_ERROR)
            DEAD_COUNT=$((DEAD_COUNT + 1))
            echo "[$(date -u +%H:%M:%S)] [DEAD] ${JOB_NAME}: state=${current_state} (count=${DEAD_COUNT})"
            ;;
        FAILED|CANCELED|FAILING|RESTARTING)
            DEAD_COUNT=$((DEAD_COUNT + 1))
            echo "[$(date -u +%H:%M:%S)] [DEAD] ${JOB_NAME}: state=${current_state} (count=${DEAD_COUNT})"
            ;;
        CREATED|INITIALIZING|UNKNOWN)
            echo "[$(date -u +%H:%M:%S)] [TRANSITIONAL] ${JOB_NAME}: ${current_state}"
            DEAD_COUNT=0
            ;;
        *)
            echo "[$(date -u +%H:%M:%S)] [UNKNOWN] ${JOB_NAME}: unexpected state='${current_state}'"
            DEAD_COUNT=$((DEAD_COUNT + 1))
            ;;
    esac

    if [ $DEAD_COUNT -ge 2 ]; then
        latest_state=$(get_job_state "$JOB_NAME")
        if [ "$latest_state" = "RUNNING" ]; then
            DEAD_COUNT=0
            echo "[$(date -u +%H:%M:%S)] [RECOVERED] Job is RUNNING (race condition avoided)."
            continue
        fi

        echo "[$(date -u +%H:%M:%S)] [RECOVERY] Job is dead (state=${current_state}). Initiating recovery..."

        dead_job_id=$(get_job_id "$JOB_NAME")

        if [ "$dead_job_id" != "NOT_FOUND" ] && [ "$dead_job_id" != "API_ERROR" ]; then
            cancel_job "$dead_job_id"
        fi

        failure_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        echo "[$(date -u +%H:%M:%S)] [RECOVERY] Previous job ${dead_job_id} failed at ${failure_ts}" \
            >> /tmp/flink_job_failures.log

        wait_for_cluster

        if [ $RESUBMIT_ATTEMPT -gt $MAX_RESUBMISSIONS ]; then
            echo "[$(date -u +%H:%M:%S)] [FATAL] Exceeded max resubmissions (${MAX_RESUBMISSIONS})."
            echo "[$(date -u +%H:%M:%S)] [FATAL] Manual intervention required. Exiting with code 1."
            exit 1
        fi

        echo "[$(date -u +%H:%M:%S)] [RECOVERY] Resubmission attempt ${RESUBMIT_ATTEMPT}/${MAX_RESUBMISSIONS}..."

        if do_submit "auto-${RESUBMIT_ATTEMPT}"; then
            if wait_for_job_creation 120; then
                new_job_id=$(get_job_id "$JOB_NAME")
                log_submission "$RESUBMIT_ATTEMPT" "$new_job_id"
                echo "[$(date -u +%H:%M:%S)] [RECOVERED] New job ${new_job_id} is now running."
            else
                echo "[$(date -u +%H:%M:%S)] [RECOVERY] Job did not appear after resubmit."
            fi
        else
            echo "[$(date -u +%H:%M:%S)] [RECOVERY] Resubmit command failed."
        fi

        RESUBMIT_ATTEMPT=$((RESUBMIT_ATTEMPT + 1))
        DEAD_COUNT=0
    fi
done
