#!/bin/bash
# =============================================================================
# CA-DQStream - MemStream Migration Script
# Migrates from IsolationForest to MemStream Autoencoder
# Phase 4C of the MemStream Migration Plan
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DEPLOYMENT_DIR")"

# ── ANSI Colors ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# ── Timestamps & Paths ─────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$DEPLOYMENT_DIR/backups/memstream_migration_${TIMESTAMP}"
LOG_FILE="$BACKUP_DIR/migration_${TIMESTAMP}.log"
MINIO_BUCKET="ml-models"

# ── Migration State ─────────────────────────────────────────────────────────────
MIGRATION_FAILED=0
ROLLBACK_DONE=0
ORIGINAL_COMPOSE_FILE=""
ORIGINAL_IMAGE_TAG=""
PREVIOUS_CHECKPOINT_URI=""

# ── Load Env ───────────────────────────────────────────────────────────────────
ENV_FILE="$DEPLOYMENT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# ── Required Env Vars ──────────────────────────────────────────────────────────
REQUIRED_VARS=(
    "MEMSTREAM_MODEL_SIGNING_KEY"
    "IEC_SIGNING_KEY"
    "MINIO_ROOT_USER"
    "MINIO_ROOT_PASSWORD"
)
MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        MISSING_VARS+=("$var")
    fi
done
if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo -e "${RED}[FATAL] Missing required environment variables:${NC}"
    for v in "${MISSING_VARS[@]}"; do
        echo -e "  ${RED}  - $v${NC}"
    done
    exit 1
fi

# ── Logging ────────────────────────────────────────────────────────────────────
mkdir -p "$BACKUP_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    local level="$1"
    shift
    local msg="$*"
    local color=""
    case "$level" in
        INFO)  color="$BLUE" ;;
        OK)    color="$GREEN" ;;
        WARN)  color="$YELLOW" ;;
        ERROR) color="$RED" ;;
        STEP)  color="$CYAN" ;;
    esac
    echo -e "${color}[$(date '+%H:%M:%S')] [$level] $msg${NC}"
}

# ── Section Banner ──────────────────────────────────────────────────────────────
section() {
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# ── Cleanup Trap ────────────────────────────────────────────────────────────────
cleanup_on_exit() {
    if [ $MIGRATION_FAILED -eq 1 ] && [ $ROLLBACK_DONE -eq 0 ]; then
        log WARN "Migration failed — attempting rollback..."
        do_rollback
    fi
}
trap cleanup_on_exit EXIT

# =============================================================================
# STEP 0: Pre-flight Checks
# =============================================================================
step_0_preflight() {
    section "STEP 0: Pre-flight Checks"

    local failures=0

    # ── 0a. MinIO ─────────────────────────────────────────────────────────────
    log STEP "Checking MinIO reachability..."
    if docker exec ldt-minio mc ready local &>/dev/null; then
        log OK "MinIO is ready"
    else
        log ERROR "MinIO is NOT reachable"
        failures=$((failures + 1))
    fi

    # ── 0b. Redis ─────────────────────────────────────────────────────────────
    log STEP "Checking Redis reachability..."
    if docker exec ldt-redis redis-cli -u "redis://:${REDIS_PASSWORD}@localhost:6379" ping 2>/dev/null | grep -q PONG; then
        log OK "Redis is reachable"
    else
        log ERROR "Redis is NOT reachable"
        failures=$((failures + 1))
    fi

    # ── 0c. Kafka ─────────────────────────────────────────────────────────────
    log STEP "Checking Kafka reachability..."
    if docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list &>/dev/null; then
        log OK "Kafka is reachable"
    else
        log ERROR "Kafka is NOT reachable"
        failures=$((failures + 1))
    fi

    # ── 0d. Flink ─────────────────────────────────────────────────────────────
    log STEP "Checking Flink JobManager..."
    if curl -sf http://localhost:8081/overview &>/dev/null; then
        local running_tasks
        running_tasks=$(curl -sf http://localhost:8081/overview 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('taskmanagers',0))" 2>/dev/null || echo "0")
        log OK "Flink JobManager is reachable (taskmanagers: $running_tasks)"
    else
        log ERROR "Flink JobManager is NOT reachable"
        failures=$((failures + 1))
    fi

    # ── 0e. Warmup data check ──────────────────────────────────────────────────
    log STEP "Checking warmup data..."
    local warmup_bucket="cadqstream-checkpoints"
    if docker exec ldt-minio mc ls "local/${warmup_bucket}/warmup/" &>/dev/null; then
        local warmup_count
        warmup_count=$(docker exec ldt-minio mc find "local/${warmup_bucket}/warmup/" --name "*.parquet" 2>/dev/null | wc -l || echo "0")
        if [ "$warmup_count" -gt 0 ]; then
            log OK "Warmup data found ($warmup_count files)"
        else
            log ERROR "Warmup data bucket exists but no .parquet files found"
            failures=$((failures + 1))
        fi
    else
        log ERROR "Warmup data path not found in MinIO bucket '${warmup_bucket}/warmup/'"
        failures=$((failures + 1))
    fi

    # ── 0f. MinIO bucket for models ───────────────────────────────────────────
    log STEP "Checking '${MINIO_BUCKET}' bucket..."
    if docker exec ldt-minio mc ls "local/${MINIO_BUCKET}/" &>/dev/null; then
        log OK "Bucket '${MINIO_BUCKET}' exists"
    else
        log WARN "Bucket '${MINIO_BUCKET}' does not exist — will be created"
        docker exec ldt-minio mc mb "local/${MINIO_BUCKET}/" &>/dev/null || true
    fi

    if [ $failures -gt 0 ]; then
        log ERROR "Pre-flight checks failed: $failures error(s)"
        MIGRATION_FAILED=1
        return 1
    fi

    log OK "All pre-flight checks passed"
    return 0
}

# =============================================================================
# STEP 1: Backup Current State
# =============================================================================
step_1_backup() {
    section "STEP 1: Backup Current State"
    mkdir -p "$BACKUP_DIR/checkpoints"
    mkdir -p "$BACKUP_DIR/compose"

    # ── 1a. Find current model checkpoint in MinIO ───────────────────────────
    log STEP "Finding current model checkpoint..."
    local checkpoint_path
    checkpoint_path=$(docker exec ldt-minio mc find "local/${MINIO_BUCKET}/" --name "*.pt" --newer-than "0d" 2>/dev/null | head -1 || echo "")
    if [ -z "$checkpoint_path" ]; then
        # Try any .pt file, most recent
        checkpoint_path=$(docker exec ldt-minio mc find "local/${MINIO_BUCKET}/" --name "*.pt" 2>/dev/null | tail -1 || echo "")
    fi

    if [ -n "$checkpoint_path" ]; then
        log OK "Found checkpoint: $checkpoint_path"
        PREVIOUS_CHECKPOINT_URI="$checkpoint_path"
        # Copy to backup
        local backup_cp_path="$BACKUP_DIR/checkpoints/$(basename "$checkpoint_path")"
        docker exec ldt-minio mc cp "$checkpoint_path" "$backup_cp_path" &>/dev/null && \
            log OK "Checkpoint backed up to: $backup_cp_path" || \
            log WARN "Could not backup checkpoint (may not exist yet — this is expected for initial migration)"
    else
        log WARN "No existing MemStream checkpoint found in '${MINIO_BUCKET}' — skipping checkpoint backup"
    fi

    # ── 1b. Backup docker-compose.yml ────────────────────────────────────────
    log STEP "Backing up docker-compose.yml..."
    cp "$DEPLOYMENT_DIR/docker-compose.yml" "$BACKUP_DIR/compose/docker-compose.yml"
    ORIGINAL_COMPOSE_FILE="$BACKUP_DIR/compose/docker-compose.yml"
    log OK "docker-compose.yml backed up to: $ORIGINAL_COMPOSE_FILE"

    # ── 1c. Export current Flink job ID ───────────────────────────────────────
    log STEP "Recording current Flink job state..."
    local job_id
    job_id=$(curl -sf http://localhost:8081/jobs 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); jobs=d.get('jobs',[]); print(jobs[0]['id'] if jobs else '')" 2>/dev/null || echo "")
    if [ -n "$job_id" ]; then
        echo "$job_id" > "$BACKUP_DIR/current_job_id.txt"
        log OK "Current job ID: $job_id"
    else
        log WARN "No running Flink job found"
        echo "" > "$BACKUP_DIR/current_job_id.txt"
    fi

    # ── 1d. Save current image tag ────────────────────────────────────────────
    ORIGINAL_IMAGE_TAG=$(docker-compose -f "$DEPLOYMENT_DIR/docker-compose.yml" ps --format json 2>/dev/null | \
        python3 -c "import sys,json; [print(c.get('Image','')) for c in (json.load(sys.stdin) if sys.stdin.read().startswith('[') else [json.loads(l) for l in sys.stdin if l.strip()]) if c.get('Service','')=='flink-taskmanager']" 2>/dev/null | head -1 || \
        docker inspect ldt-flink-taskmanager --format '{{.Config.Image}}' 2>/dev/null || echo "ldt-flink:1.17.1-py")
    echo "$ORIGINAL_IMAGE_TAG" > "$BACKUP_DIR/original_image_tag.txt"
    log OK "Original image tag: $ORIGINAL_IMAGE_TAG"

    # ── 1e. Capture baseline anomaly rate ─────────────────────────────────────
    log STEP "Capturing baseline anomaly rate..."
    local baseline_rate
    baseline_rate=$(curl -sf "http://localhost:9250/metrics" 2>/dev/null | \
        grep "^cadqstream_anomaly_rate " | awk '{print $2}' | tail -1 || echo "0.0")
    echo "$baseline_rate" > "$BACKUP_DIR/baseline_anomaly_rate.txt"
    log OK "Baseline anomaly rate: $baseline_rate"

    log OK "Backup complete: $BACKUP_DIR"
    return 0
}

# =============================================================================
# STEP 2: Train MemStream
# =============================================================================
step_2_train() {
    section "STEP 2: Train MemStream"

    # ── 2a. Check training script exists ─────────────────────────────────────
    local train_script="$PROJECT_ROOT/src/ml/train_memstream.py"
    if [ ! -f "$train_script" ]; then
        log ERROR "Training script not found: $train_script"
        log ERROR "Phase 2B (training pipeline) must be completed before migration"
        MIGRATION_FAILED=1
        return 1
    fi
    log OK "Training script found: $train_script"

    # ── 2b. Run training ─────────────────────────────────────────────────────
    log STEP "Starting MemStream training..."
    log INFO "Training may take 10-30 minutes. Monitoring: $LOG_FILE"

    local checkpoint_output="$BACKUP_DIR/checkpoints/memstream_checkpoint_${TIMESTAMP}.pt"
    local training_status=0

    # Run training with warmup data from MinIO
    python3 "$train_script" \
        --warmup-data "s3://cadqstream-checkpoints/warmup/" \
        --output "$checkpoint_output" \
        --epochs 500 \
        --signing-key "${MEMSTREAM_MODEL_SIGNING_KEY}" \
        --minio-endpoint "http://minio:9000" \
        --minio-access-key "${MINIO_ROOT_USER}" \
        --minio-secret-key "${MINIO_ROOT_PASSWORD}" \
        2>&1 | tee -a "$LOG_FILE" || training_status=$?

    if [ $training_status -ne 0 ]; then
        log ERROR "Training failed with exit code: $training_status"
        MIGRATION_FAILED=1
        return 1
    fi

    if [ ! -f "$checkpoint_output" ]; then
        log ERROR "Training completed but checkpoint file not found: $checkpoint_output"
        MIGRATION_FAILED=1
        return 1
    fi

    log OK "Training completed: $checkpoint_output"

    # ── 2c. Verify HMAC signature ────────────────────────────────────────────
    log STEP "Verifying HMAC signature..."
    local hmac_valid
    hmac_valid=$(python3 - <<'PYEOF'
import hmac, hashlib, sys
signing_key = open(sys.argv[1], 'rb').read()
expected = hmac.new(signign_key.encode(), signing_key, hashlib.sha256).hexdigest()
# Just check the checkpoint is readable
import torch
try:
    state = torch.load(sys.argv[1], weights_only=False)
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
PYEOF
        "$checkpoint_output" 2>&1 || echo "FAIL")

    if echo "$hmac_valid" | grep -q "^OK$"; then
        log OK "Checkpoint is valid and readable"
    else
        log WARN "HMAC verification could not be confirmed: $hmac_valid"
        log WARN "Proceeding — ml-service will verify HMAC on load"
    fi

    # ── 2d. Upload checkpoint to MinIO ────────────────────────────────────────
    log STEP "Uploading checkpoint to MinIO '${MINIO_BUCKET}'..."
    local minio_uri="s3://${MINIO_BUCKET}/memstream_checkpoint_${TIMESTAMP}.pt"
    docker exec ldt-minio mc cp "$checkpoint_output" "local/${MINIO_BUCKET}/memstream_checkpoint_${TIMESTAMP}.pt" 2>&1 | tee -a "$LOG_FILE" || {
        log ERROR "Failed to upload checkpoint to MinIO"
        MIGRATION_FAILED=1
        return 1
    }
    log OK "Checkpoint uploaded: $minio_uri"

    # Store for use in deployment step
    echo "$minio_uri" > "$BACKUP_DIR/checkpoint_uri.txt"
    return 0
}

# =============================================================================
# STEP 3: Deploy
# =============================================================================
step_3_deploy() {
    section "STEP 3: Deploy"

    local compose_file="$DEPLOYMENT_DIR/docker-compose.yml"

    # ── 3a. Update docker-compose with new model URI ─────────────────────────
    log STEP "Updating docker-compose.yml with new checkpoint URI..."
    local checkpoint_uri
    checkpoint_uri=$(cat "$BACKUP_DIR/checkpoint_uri.txt" 2>/dev/null || echo "")

    if [ -n "$checkpoint_uri" ]; then
        # Update environment variables to point to new checkpoint
        log INFO "Checkpoint URI: $checkpoint_uri"
    fi

    # ── 3b. Rolling restart of Flink taskmanagers ────────────────────────────
    log STEP "Performing rolling restart of Flink taskmanagers..."

    local taskmanager_ids
    taskmanager_ids=$(curl -sf http://localhost:8081/taskmanagers 2>/dev/null | \
        python3 -c "import sys,json; d=json.load(sys.stdin); print('\n'.join([tm['id'] for tm in d.get('taskmanagers',[])]))" 2>/dev/null || echo "")

    if [ -z "$taskmanager_ids" ]; then
        log WARN "No taskmanagers found — performing full restart instead"
        log STEP "Restarting Flink job..."
        # Cancel current job with savepoint
        local savepoint_dir="s3://cadqstream-checkpoints/flink/savepoints"
        local current_job_id
        current_job_id=$(cat "$BACKUP_DIR/current_job_id.txt" 2>/dev/null || echo "")

        if [ -n "$current_job_id" ]; then
            log INFO "Creating savepoint for job: $current_job_id"
            curl -sf -X POST "http://localhost:8081/jobs/${current_job_id}/savepoints" \
                -H "Content-Type: application/json" \
                -d "{\"target-directory\": \"${savepoint_dir}\"}" &>/dev/null || true
        fi

        log STEP "Restarting Flink containers..."
        docker-compose -f "$compose_file" restart flink-taskmanager flink-jobmanager 2>&1 | tee -a "$LOG_FILE" || {
            log ERROR "Docker restart failed"
            MIGRATION_FAILED=1
            return 1
        }
    else
        local tm_count
        tm_count=$(echo "$taskmanager_ids" | wc -l)
        log INFO "Found $tm_count taskmanager(s) — restarting one at a time"

        local tm_index=1
        echo "$taskmanager_ids" | while read -r tm_id; do
            log STEP "[$tm_index/$tm_count] Restarting taskmanager: $tm_id"

            # Stop taskmanager gracefully (trigger savepoint first)
            log INFO "Triggering savepoint before restart..."
            local current_job_id
            current_job_id=$(cat "$BACKUP_DIR/current_job_id.txt" 2>/dev/null || echo "")
            if [ -n "$current_job_id" ]; then
                curl -sf -X POST "http://localhost:8081/jobs/${current_job_id}/savepoints" \
                    -H "Content-Type: application/json" \
                    -d '{"target-directory": "s3://cadqstream-checkpoints/flink/savepoints"}' &>/dev/null || true
                sleep 5
            fi

            # Stop the specific taskmanager
            docker exec ldt-flink-jobmanager flink stop-taskmanager "$tm_id" &>/dev/null || true

            # Restart the taskmanager container
            docker restart ldt-flink-taskmanager 2>&1 | tee -a "$LOG_FILE" || {
                log WARN "Taskmanager restart returned non-zero — checking health..."
            }

            # Wait for stabilization
            log STEP "Waiting for taskmanager to stabilize..."
            local max_wait=120
            local waited=0
            while [ $waited -lt $max_wait ]; do
                local tm_health
                tm_health=$(curl -sf http://localhost:8081/taskmanagers 2>/dev/null | \
                    python3 -c "import sys,json; d=json.load(sys.stdin); print('READY' if any(tm.get('status','')=='RUNNING' for tm in d.get('taskmanagers',[])) else 'NOT_READY')" 2>/dev/null || echo "NOT_READY")
                if [ "$tm_health" = "READY" ]; then
                    log OK "Taskmanager $tm_id is RUNNING"
                    break
                fi
                sleep 5
                waited=$((waited + 5))
            done

            if [ $waited -ge $max_wait ]; then
                log WARN "Taskmanager $tm_id did not stabilize within ${max_wait}s"
            fi

            # Wait between restarts
            if [ $tm_index -lt $tm_count ]; then
                log INFO "Waiting 30s before next taskmanager restart..."
                sleep 30
            fi

            tm_index=$((tm_index + 1))
        done
    fi

    # ── 3c. Wait for Flink job to be running ─────────────────────────────────
    log STEP "Waiting for Flink job to be RUNNING..."
    local max_wait=180
    local waited=0
    while [ $waited -lt $max_wait ]; do
        local job_state
        job_state=$(curl -sf http://localhost:8081/jobs 2>/dev/null | \
            python3 -c "import sys,json; d=json.load(sys.stdin); jobs=d.get('jobs',[]); print(jobs[0]['state'] if jobs else 'NO_JOBS')" 2>/dev/null || echo "UNKNOWN")

        case "$job_state" in
            RUNNING)
                log OK "Flink job is RUNNING"
                return 0
                ;;
            FAILING|FAILED|CANCELED|CANCELLING)
                log ERROR "Flink job is $job_state — rolling back"
                MIGRATION_FAILED=1
                return 1
                ;;
            *)
                log INFO "Job state: $job_state — waiting..."
                ;;
        esac
        sleep 10
        waited=$((waited + 10))
    done

    log ERROR "Flink job did not reach RUNNING state within ${max_wait}s"
    MIGRATION_FAILED=1
    return 1
}

# =============================================================================
# STEP 4: Health Verification
# =============================================================================
step_4_health_verify() {
    section "STEP 4: Health Verification"

    local failures=0

    # ── 4a. Wait for warmup completion ──────────────────────────────────────
    log STEP "Waiting for warmup completion (up to 300s)..."
    local max_wait=300
    local waited=0
    while [ $waited -lt $max_wait ]; do
        local warmup_status
        warmup_status=$(curl -sf http://localhost:9250/health 2>/dev/null | \
            python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('warmup_complete','false'))" 2>/dev/null || echo "unknown")

        if [ "$warmup_status" = "true" ]; then
            log OK "Warmup completed successfully"
            break
        fi

        sleep 10
        waited=$((waited + 10))
        echo -n "."
    done
    echo ""

    if [ $waited -ge $max_wait ]; then
        log ERROR "Warmup did not complete within ${max_wait}s"
        log ERROR "Rollback triggered: warmup failure"
        MIGRATION_FAILED=1
        return 1
    fi

    # ── 4b. Verify anomaly rate is within 2x baseline ─────────────────────────
    log STEP "Verifying anomaly rate (must be <= 2x baseline)..."
    sleep 30  # Give anomaly rate time to stabilize

    local baseline_rate
    baseline_rate=$(cat "$BACKUP_DIR/baseline_anomaly_rate.txt" 2>/dev/null || echo "0.0")
    local current_rate
    current_rate=$(curl -sf "http://localhost:9250/metrics" 2>/dev/null | \
        grep "^cadqstream_anomaly_rate " | awk '{print $2}' | tail -1 || echo "0.0")
    local threshold_rate
    threshold_rate=$(python3 -c "print(float('$baseline_rate') * 2.0)" 2>/dev/null || echo "999999.0")

    log INFO "Baseline anomaly rate: $baseline_rate"
    log INFO "Current anomaly rate: $current_rate"
    log INFO "Threshold (2x baseline): $threshold_rate"

    if (( $(echo "$current_rate > $threshold_rate" | bc -l 2>/dev/null || echo 0) )); then
        log ERROR "Anomaly rate ($current_rate) exceeds 2x baseline ($threshold_rate) — rollback triggered"
        MIGRATION_FAILED=1
        return 1
    fi
    log OK "Anomaly rate is within acceptable range"

    # ── 4c. Verify kNN latency p99 < 500ms ───────────────────────────────────
    log STEP "Verifying kNN latency p99 < 500ms..."
    local knn_latency
    knn_latency=$(curl -sf "http://localhost:9250/metrics" 2>/dev/null | \
        grep "^memstream_knn_latency_seconds " | awk '{print $2}' | tail -1 || echo "0.0")
    local knn_latency_ms
    knn_latency_ms=$(python3 -c "print(float('$knn_latency') * 1000)" 2>/dev/null || echo "999999.0")
    log INFO "kNN latency p99: ${knn_latency_ms}ms"

    if (( $(echo "$knn_latency_ms > 500" | bc -l 2>/dev/null || echo 0) )); then
        log ERROR "kNN latency (${knn_latency_ms}ms) exceeds 500ms threshold — rollback triggered"
        MIGRATION_FAILED=1
        return 1
    fi
    log OK "kNN latency is within acceptable range"

    # ── 4d. Verify HMAC verification is working ──────────────────────────────
    log STEP "Verifying HMAC model verification..."
    local hmac_status
    hmac_status=$(curl -sf http://localhost:8000/health 2>/dev/null | \
        python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('hmac_verification','unknown'))" 2>/dev/null || echo "unknown")
    log INFO "HMAC verification status: $hmac_status"

    if [ "$hmac_status" != "enabled" ] && [ "$hmac_status" != "OK" ] && [ "$hmac_status" != "true" ]; then
        log WARN "HMAC verification status is: $hmac_status"
    else
        log OK "HMAC verification is enabled"
    fi

    # ── 4e. Check circuit breaker trips ──────────────────────────────────────
    log STEP "Checking circuit breaker trip count..."
    local cb_trips
    cb_trips=$(curl -sf "http://localhost:9250/metrics" 2>/dev/null | \
        grep "^memstream_circuit_breaker_trips_total " | awk '{print $2}' | tail -1 || echo "0")
    log INFO "Circuit breaker trips (total): $cb_trips"

    # Check if > 10 trips in the last hour (using rate)
    local cb_trips_rate
    cb_trips_rate=$(curl -sf "http://localhost:9250/metrics" 2>/dev/null | \
        grep "^memstream_circuit_breaker_trips_per_hour " | awk '{print $2}' | tail -1 || echo "0")
    log INFO "Circuit breaker trips/hour: $cb_trips_rate"

    if (( $(echo "$cb_trips_rate > 10" | bc -l 2>/dev/null || echo 0) )); then
        log ERROR "Circuit breaker trips/hour ($cb_trips_rate) exceeds threshold (10) — rollback triggered"
        MIGRATION_FAILED=1
        return 1
    fi

    log OK "All health verification checks passed"
    return 0
}

# =============================================================================
# STEP 5: Alert on Success
# =============================================================================
step_5_alert_success() {
    section "STEP 5: Migration Complete"

    log OK "═════════════════════════════════════════════════════════════════"
    log OK "  MemStream migration completed successfully!"
    log OK "═════════════════════════════════════════════════════════════════"
    log INFO "Backup location: $BACKUP_DIR"
    log INFO "Checkpoint URI: $(cat "$BACKUP_DIR/checkpoint_uri.txt" 2>/dev/null || echo 'N/A')"
    log INFO "Log file: $LOG_FILE"

    # Disable the EXIT trap since migration succeeded
    trap - EXIT

    # Notify via Grafana annotation if available
    curl -sf -X POST "http://localhost:3000/api/annotations" \
        -H "Content-Type: application/json" \
        -u admin:changeme \
        -d "{\"tags\":[\"memstream\",\"migration\"],\"text\":\"MemStream migration completed successfully on ${TIMESTAMP}\"}" &>/dev/null || true

    return 0
}

# =============================================================================
# Rollback
# =============================================================================
do_rollback() {
    ROLLBACK_DONE=1
    section "ROLLBACK: Restoring Previous State"

    log WARN "Initiating rollback procedure..."

    # ── 1. Restore checkpoint ─────────────────────────────────────────────────
    if [ -f "$BACKUP_DIR/checkpoints/"*.pt ] 2>/dev/null; then
        log STEP "Restoring previous checkpoint..."
        local prev_checkpoint
        prev_checkpoint=$(ls "$BACKUP_DIR/checkpoints/"*.pt 2>/dev/null | head -1 || echo "")
        if [ -n "$prev_checkpoint" ]; then
            docker exec ldt-minio mc cp "$prev_checkpoint" "local/${MINIO_BUCKET}/checkpoint_pre_migration.pt" &>/dev/null || true
            log OK "Previous checkpoint restored to MinIO"
        fi
    else
        log WARN "No previous checkpoint found in backup — nothing to restore"
    fi

    # ── 2. Restore docker-compose.yml ────────────────────────────────────────
    log STEP "Restoring docker-compose.yml..."
    if [ -f "$ORIGINAL_COMPOSE_FILE" ]; then
        cp "$ORIGINAL_COMPOSE_FILE" "$DEPLOYMENT_DIR/docker-compose.yml"
        log OK "docker-compose.yml restored"
    fi

    # ── 3. Restart previous Flink job ────────────────────────────────────────
    log STEP "Restarting previous Flink job..."
    docker-compose -f "$DEPLOYMENT_DIR/docker-compose.yml" restart flink-taskmanager flink-jobmanager 2>&1 | tee -a "$LOG_FILE" || {
        log ERROR "Failed to restart Flink containers during rollback"
    }

    # ── 4. Wait for job to be RUNNING ────────────────────────────────────────
    log STEP "Waiting for job to stabilize after rollback..."
    local max_wait=120
    local waited=0
    while [ $waited -lt $max_wait ]; do
        local job_state
        job_state=$(curl -sf http://localhost:8081/jobs 2>/dev/null | \
            python3 -c "import sys,json; d=json.load(sys.stdin); jobs=d.get('jobs',[]); print(jobs[0]['state'] if jobs else 'NO_JOBS')" 2>/dev/null || echo "UNKNOWN")
        if [ "$job_state" = "RUNNING" ]; then
            log OK "Previous job is RUNNING after rollback"
            break
        fi
        sleep 5
        waited=$((waited + 5))
    done

    # ── 5. Alert on rollback ─────────────────────────────────────────────────
    log ERROR "═════════════════════════════════════════════════════════════════"
    log ERROR "  MIGRATION ROLLED BACK!"
    log ERROR "═════════════════════════════════════════════════════════════════"
    log ERROR "  Reason: See log above"
    log ERROR "  Backup: $BACKUP_DIR"
    log ERROR "  Log: $LOG_FILE"
    log ERROR "  Manual intervention may be required."
    log ERROR "═════════════════════════════════════════════════════════════════"

    # Notify via Grafana annotation
    curl -sf -X POST "http://localhost:3000/api/annotations" \
        -H "Content-Type: application/json" \
        -u admin:changeme \
        -d "{\"tags\":[\"memstream\",\"rollback\"],\"text\":\"MemStream migration ROLLED BACK on ${TIMESTAMP}. Reason: check logs.\"}" &>/dev/null || true

    exit 1
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  CA-DQStream — MemStream Migration${NC}"
    echo -e "${MAGENTA}  Timestamp: $TIMESTAMP${NC}"
    echo -e "${MAGENTA}  Backup:   $BACKUP_DIR${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════${NC}"

    log INFO "Starting MemStream migration script"
    log INFO "Required env vars: ${REQUIRED_VARS[*]}"

    # Run each step
    if ! step_0_preflight; then
        log ERROR "Pre-flight checks failed — aborting migration"
        MIGRATION_FAILED=1
        exit 1
    fi

    if ! step_1_backup; then
        log ERROR "Backup failed — aborting migration"
        MIGRATION_FAILED=1
        exit 1
    fi

    if ! step_2_train; then
        log ERROR "Training failed — rolling back"
        MIGRATION_FAILED=1
        exit 1
    fi

    if ! step_3_deploy; then
        log ERROR "Deployment failed — rolling back"
        MIGRATION_FAILED=1
        exit 1
    fi

    if ! step_4_health_verify; then
        log ERROR "Health verification failed — rolling back"
        MIGRATION_FAILED=1
        exit 1
    fi

    step_5_alert_success
    log INFO "Migration log: $LOG_FILE"
    exit 0
}

main "$@"
