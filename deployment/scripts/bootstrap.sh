#!/bin/bash
# =============================================================================
# CA-DQStream - Bootstrap Script
# Initializes the running stack: waits for health, runs init containers, verifies
# Usage: bash deployment/scripts/bootstrap.sh [--timeout SECONDS] [--skip-init]
# Assumes: containers are already running (via docker compose up -d)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$DEPLOYMENT_DIR/docker-compose.yml"

# ── Default configuration ──────────────────────────────────────────────────
TIMEOUT_DEFAULT=120
TIMEOUT_FAST=30
TIMEOUT_INIT=180
SKIP_INIT=false
FAILED=0

# ── Color codes ─────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# ── Logging helpers ──────────────────────────────────────────────────────────
log_info()  { echo -e "${BLUE}[INFO]${NC}   $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}     $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" >&2; }
log_err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $*"; }
log_phase() { echo -e "\n${MAGENTA}${BOLD}═══ $* ═══${NC}"; }
log_div()   { echo -e "${CYAN}─────────────────────────────────────────────────────────────────────────────${NC}"; }

# ── Usage ───────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --timeout SECONDS   Global timeout for health waits (default: $TIMEOUT_DEFAULT)
  --skip-init         Skip init container execution (verify-only mode)
  -h, --help         Show this help

Bootstrap Phases:
  1. Verify containers are running
  2. Wait for infrastructure health (Zookeeper, Kafka, Schema Registry)
  3. Wait for storage health (Redis, MinIO)
  4. Wait for Flink health (JobManager, TaskManager)
  5. Run init containers (kafka-init → minio-init → flink-init)
  6. Wait for observability (Prometheus, Grafana)
  7. Final verification and summary

Exit codes:
  0  Bootstrap succeeded
  1  Bootstrap failed (critical service unreachable or init container failed)
EOF
    exit 0
}

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout) TIMEOUT_DEFAULT="$2"; shift 2 ;;
        --skip-init) SKIP_INIT=true; shift ;;
        -h|--help) usage ;;
        *) log_err "Unknown option: $1"; exit 1 ;;
    esac
done

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Wait for a service to be healthy using exponential backoff + linear fallback
# Args: name, timeout, interval_base, health_cmd, container_name
wait_for_health() {
    local name="$1"
    local timeout="$2"
    local interval_base="$3"
    local health_cmd="$4"
    local container_name="${5:-ldt-$name}"

    local elapsed=0
    local attempt=0
    local interval=$interval_base

    echo -n "    $name: waiting "
    while [ $elapsed -lt $timeout ]; do
        if eval "$health_cmd" &>/dev/null; then
            echo -e "${GREEN}healthy${NC} (${elapsed}s)"
            return 0
        fi
        attempt=$((attempt + 1))
        echo -n "."
        sleep $interval
        elapsed=$((elapsed + interval))
        # Exponential backoff capped at 10s
        if [ $interval -lt 10 ]; then
            interval=$((interval * 2 < 10 ? interval * 2 : 10))
        fi
    done
    echo -e "${RED}FAILED${NC} (timeout after ${timeout}s)"
    return 1
}

# Wait for a Docker healthcheck to report healthy
# Args: service_name, timeout
wait_docker_health() {
    local service="$1"
    local timeout="$2"
    local container_name="ldt-$service"
    local elapsed=0
    local interval=5

    echo -n "    $service: waiting for Docker healthcheck "
    while [ $elapsed -lt $timeout ]; do
        local status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "none")
        if [ "$status" = "healthy" ]; then
            echo -e "${GREEN}healthy${NC} (${elapsed}s)"
            return 0
        fi
        # Also check if container is running but health is starting
        local running=$(docker inspect --format='{{.State.Running}}' "$container_name" 2>/dev/null || echo "false")
        if [ "$running" != "true" ]; then
            echo -e "${RED}STOPPED${NC}"
            return 1
        fi
        echo -n "."
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    echo -e "${RED}TIMEOUT${NC} (Docker healthcheck did not become healthy in ${timeout}s)"
    return 1
}

# Wait for an init container to exit successfully
# Args: service_name, timeout
wait_init_complete() {
    local service="$1"
    local timeout="$2"
    local container_name="ldt-$service-init"
    local elapsed=0
    local interval=3

    echo -n "    $service-init: waiting for completion "
    while [ $elapsed -lt $timeout ]; do
        local status=$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null || echo "notfound")
        if [ "$status" = "exited" ]; then
            local exit_code=$(docker inspect --format='{{.ExitCode}}' "$container_name" 2>/dev/null || echo "-1")
            if [ "$exit_code" = "0" ]; then
                echo -e "${GREEN}SUCCESS${NC} (exit code 0)"
                return 0
            else
                echo -e "${RED}FAILED${NC} (exit code $exit_code)"
                return 1
            fi
        elif [ "$status" = "notfound" ]; then
            echo -n "?"
        else
            echo -n "($status)"
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
        echo -n "."
    done
    echo -e "${RED}TIMEOUT${NC} (did not complete in ${timeout}s)"
    return 1
}

# Run an init container and wait for it to complete
# Args: service_name, timeout
run_init_container() {
    local service="$1"
    local timeout="$2"

    echo ""
    log_info "Running $service-init..."
    if ! docker compose -f "$COMPOSE_FILE" up -d "$service-init" 2>&1; then
        log_err "Failed to start $service-init container"
        return 1
    fi

    if ! wait_init_complete "$service" "$timeout"; then
        log_err "Logs from $service-init:"
        docker logs "ldt-$service-init" 2>&1 | tail -20 | sed 's/^/      /'
        return 1
    fi

    # Show success logs (last 10 lines)
    local logs=$(docker logs "ldt-$service-init" 2>&1 | tail -10)
    if [ -n "$logs" ]; then
        echo "$logs" | sed 's/^/      /'
    fi

    return 0
}

# =============================================================================
# PHASE 0: HEADER & PREFLIGHT
# =============================================================================
echo ""
echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}${BOLD}  CA-DQStream Bootstrap${NC}"
echo -e "${MAGENTA}${BOLD}  Waiting for health → Running init containers → Verifying bootstrap${NC}"
echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
log_info "Compose file: $COMPOSE_FILE"
log_info "Timeout (slow services): ${TIMEOUT_DEFAULT}s"
log_info "Timeout (fast services): ${TIMEOUT_FAST}s"
log_info "Skip init: $SKIP_INIT"
echo ""

# =============================================================================
# PHASE 1: VERIFY CONTAINERS ARE RUNNING
# =============================================================================
log_phase "PHASE 1: Verifying containers are running"
log_div

REQUIRED_CONTAINERS=(
    "ldt-zookeeper"
    "ldt-kafka"
    "ldt-schema-registry"
    "ldt-redis"
    "ldt-minio"
    "ldt-flink-jobmanager"
    "ldt-flink-taskmanager"
    "ldt-prometheus"
    "ldt-grafana"
)

MISSING=0
for container in "${REQUIRED_CONTAINERS[@]}"; do
    if docker ps --filter "name=$container" --filter "status=running" --format "{{.Names}}" | grep -q "^${container}$"; then
        echo -e "    $container: ${GREEN}running${NC}"
    else
        echo -e "    $container: ${RED}NOT RUNNING${NC}"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    log_err "$MISSING required container(s) not running. Run 'docker compose up -d' first."
    exit 1
fi
log_ok "All required containers are running"

# =============================================================================
# PHASE 2: INFRASTRUCTURE HEALTH WAITS
# =============================================================================
log_phase "PHASE 2: Waiting for infrastructure to be healthy"
log_div

# 2a. Zookeeper
echo -e "${BLUE}[Infrastructure]${NC}"
wait_for_health "zookeeper" "$TIMEOUT_DEFAULT" 3 \
    "docker exec ldt-zookeeper bash -c 'echo ruok | nc localhost 2181 | grep -q imok'" \
    "ldt-zookeeper" || FAILED=$((FAILED + 1))

# 2b. Kafka
wait_for_health "kafka" "$TIMEOUT_DEFAULT" 5 \
    "docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null | grep -q ." \
    "ldt-kafka" || FAILED=$((FAILED + 1))

# 2c. Schema Registry
wait_for_health "schema-registry" "$TIMEOUT_DEFAULT" 3 \
    "curl -sf http://localhost:8081/subjects" \
    "ldt-schema-registry" || FAILED=$((FAILED + 1))

# 2d. Kafka UI (optional — don't fail on it)
echo -n "    kafka-ui: "
if curl -sf http://localhost:8080 &>/dev/null; then
    echo -e "${GREEN}healthy${NC}"
else
    echo -e "${YELLOW}not ready (non-critical)${NC}"
fi

# 2e. Kafka Exporter (optional)
echo -n "    kafka-exporter: "
if curl -sf http://localhost:9308/metrics &>/dev/null; then
    echo -e "${GREEN}healthy${NC}"
else
    echo -e "${YELLOW}not ready (non-critical)${NC}"
fi

if [ $FAILED -gt 0 ]; then
    log_err "Infrastructure health wait failed ($FAILED failure(s)). Aborting."
    exit 1
fi
log_ok "All infrastructure services are healthy"

# =============================================================================
# PHASE 3: STORAGE HEALTH WAITS
# =============================================================================
log_phase "PHASE 3: Waiting for storage services to be healthy"
log_div

# 3a. Redis
echo -e "${BLUE}[Storage]${NC}"
REDIS_PASSWORD="${REDIS_PASSWORD}"
wait_for_health "redis" "$TIMEOUT_DEFAULT" 3 \
    "docker exec ldt-redis redis-cli -a \"$REDIS_PASSWORD\" ping 2>/dev/null | grep -q PONG" \
    "ldt-redis" || FAILED=$((FAILED + 1))

# 3b. MinIO
wait_for_health "minio" "$TIMEOUT_DEFAULT" 3 \
    "docker exec ldt-minio mc ready local" \
    "ldt-minio" || FAILED=$((FAILED + 1))

if [ $FAILED -gt 0 ]; then
    log_err "Storage health wait failed. Aborting."
    exit 1
fi
log_ok "All storage services are healthy"

# =============================================================================
# PHASE 4: FLINK HEALTH WAITS
# =============================================================================
log_phase "PHASE 4: Waiting for Flink cluster to be healthy"
log_div

# 4a. Flink JobManager REST API
echo -e "${BLUE}[Flink]${NC}"
wait_for_health "flink-jobmanager" "$TIMEOUT_DEFAULT" 5 \
    "curl -sf http://localhost:8081/overview" \
    "ldt-flink-jobmanager" || FAILED=$((FAILED + 1))

# 4b. Flink TaskManager (check it's registered with JobManager)
echo -n "    flink-taskmanager: waiting for registration "
FLINK_TM_TIMEOUT=0
FLINK_TM_INTERVAL=5
while [ $FLINK_TM_TIMEOUT -lt $TIMEOUT_DEFAULT ]; do
    if curl -sf http://localhost:8081/taskmanagers 2>/dev/null | grep -q '"id"'; then
        local tm_count=$(curl -sf http://localhost:8081/taskmanagers 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('taskmanagers',[])))" 2>/dev/null || echo "0")
        echo -e "${GREEN}registered${NC} ($tm_count taskmanager(s))"
        break
    fi
    echo -n "."
    sleep $FLINK_TM_INTERVAL
    FLINK_TM_TIMEOUT=$((FLINK_TM_TIMEOUT + FLINK_TM_INTERVAL))
done
if [ $FLINK_TM_TIMEOUT -ge $TIMEOUT_DEFAULT ]; then
    echo -e "${YELLOW}TIMEOUT${NC} (TaskManager not registered yet — may still be starting)"
fi

if [ $FAILED -gt 0 ]; then
    log_err "Flink health wait failed. Aborting."
    exit 1
fi
log_ok "Flink cluster is healthy"

# =============================================================================
# PHASE 5: RUN INIT CONTAINERS
# =============================================================================
log_phase "PHASE 5: Running init containers in dependency order"
log_div

if [ "$SKIP_INIT" = true ]; then
    log_warn "Skipping init containers (--skip-init mode)"
else
    # 5a. kafka-init: create topics
    echo -e "${BLUE}[Init Containers]${NC}"
    run_init_container "kafka" "$TIMEOUT_INIT" || FAILED=$((FAILED + 1))

    # 5b. minio-init: create buckets
    run_init_container "minio" "$TIMEOUT_INIT" || FAILED=$((FAILED + 1))

    # 5c. flink-init: submit Flink job
    # Must run after: flink-jobmanager healthy + flink-taskmanager healthy + kafka-init complete
    echo ""
    log_info "Checking prerequisites for flink-init..."
    local kafka_init_status=$(docker inspect --format='{{.State.Status}}' "ldt-kafka-init" 2>/dev/null || echo "notfound")
    echo -e "    kafka-init status: $kafka_init_status"

    if [ "$kafka_init_status" != "exited" ]; then
        log_warn "kafka-init has not exited. Ensure kafka-init completed before flink-init."
    fi

    run_init_container "flink" "$TIMEOUT_INIT" || FAILED=$((FAILED + 1))

    if [ $FAILED -gt 0 ]; then
        log_err "Init container(s) failed. Aborting bootstrap."
        exit 1
    fi
    log_ok "All init containers completed successfully"
fi

# =============================================================================
# PHASE 6: OBSERVABILITY HEALTH WAITS
# =============================================================================
log_phase "PHASE 6: Waiting for observability stack to be healthy"
log_div

# Prometheus
echo -e "${BLUE}[Observability]${NC}"
wait_for_health "prometheus" "$TIMEOUT_DEFAULT" 3 \
    "wget --spider -q http://localhost:9090/-/healthy" \
    "ldt-prometheus" || FAILED=$((FAILED + 1))

# Grafana (depends on Prometheus)
wait_for_health "grafana" "$TIMEOUT_DEFAULT" 3 \
    "curl -sf http://localhost:3000/api/health" \
    "ldt-grafana" || FAILED=$((FAILED + 1))

if [ $FAILED -gt 0 ]; then
    log_warn "Observability stack has failures — non-critical for data pipeline"
fi

# =============================================================================
# PHASE 7: BOOTSTRAP VERIFICATION
# =============================================================================
log_phase "PHASE 7: Bootstrap verification"
log_div

echo -e "${BLUE}[Kafka Topics]${NC}"
TOPIC_LIST=$(docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null || echo "")
if [ -n "$TOPIC_LIST" ]; then
    TOPIC_COUNT=$(echo "$TOPIC_LIST" | grep -c .)
    echo "$TOPIC_LIST" | sed 's/^/    /'
    echo -e "    ${GREEN}$TOPIC_COUNT topic(s)${NC}"
else
    echo -e "    ${YELLOW}No topics found (kafka-init may not have run)${NC}"
fi

echo ""
echo -e "${BLUE}[MinIO Buckets]${NC}"
BUCKET_LIST=$(docker exec ldt-minio mc ls local/ 2>/dev/null || echo "")
if [ -n "$BUCKET_LIST" ]; then
    BUCKET_COUNT=$(echo "$BUCKET_LIST" | grep -c .)
    echo "$BUCKET_LIST" | sed 's/^/    /'
    echo -e "    ${GREEN}$BUCKET_COUNT bucket(s)${NC}"
else
    echo -e "    ${YELLOW}No buckets found (minio-init may not have run)${NC}"
fi

echo ""
echo -e "${BLUE}[Flink Jobs]${NC}"
if FLINK_JOBS=$(curl -sf http://localhost:8081/jobs 2>/dev/null); then
    JOB_COUNT=$(echo "$FLINK_JOBS" | python3 -c "import sys,json; d=json.load(sys.stdin); jobs=d.get('jobs',[]); [print(f'    {j[\"id\"][:16]}... [{j[\"state\"]}] - {j.get(\"name\",\"unknown\")}') for j in jobs]; print(len(jobs))" 2>/dev/null)
    if [ -n "$JOB_COUNT" ]; then
        RUNNING_COUNT=$(echo "$JOB_COUNT" | grep -c "RUNNING" || echo 0)
        echo -e "    ${GREEN}$RUNNING_COUNT job(s) running${NC}"
    fi
else
    echo -e "    ${YELLOW}Flink REST API not accessible${NC}"
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}${BOLD}  Bootstrap Complete${NC}"
echo -e "${MAGENTA}${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Service URLs${NC}"
echo -e "  ─────────────────────────────────────────────────────────────────────────"
echo -e "  Kafka UI         ${GREEN}http://localhost:8080${NC}"
echo -e "  Flink UI         ${GREEN}http://localhost:8081${NC}"
echo -e "  Grafana          ${GREEN}http://localhost:3000${NC}   (admin / \${GRAFANA_PASSWORD})"
echo -e "  Prometheus       ${GREEN}http://localhost:9090${NC}"
echo -e "  MinIO Console    ${GREEN}http://localhost:9001${NC}   (\${MINIO_ROOT_USER} / \${MINIO_ROOT_PASSWORD})"
echo ""
echo -e "  ${CYAN}Next Steps${NC}"
echo -e "  ─────────────────────────────────────────────────────────────────────────"
echo "  Check health:       bash deployment/scripts/healthcheck.sh"
echo "  View Flink logs:    docker compose -f deployment/docker-compose.yml logs flink-jobmanager"
echo "  View Kafka logs:    docker compose -f deployment/docker-compose.yml logs kafka"
echo ""
echo -e "  ${CYAN}Restart bootstrap:${NC}  bash deployment/scripts/bootstrap.sh"
echo ""

if [ $FAILED -gt 0 ]; then
    log_err "Bootstrap completed with $FAILED error(s). Check logs above."
    exit 1
fi

log_ok "Bootstrap completed successfully."
exit 0
