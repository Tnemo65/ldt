#!/bin/bash
# =============================================================================
# CA-DQStream Production Deploy Script
# Single-command idempotent deployment for CA-DQStream streaming pipeline.
# Environment: Windows 10 + Docker Desktop + Git Bash / WSL2
# Run: bash deployment/scripts/deploy-prod.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DEPLOYMENT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"

# ── Colour codes ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log_step() { echo -e "\n${CYAN}[STEP]${NC}  $*"; }
log_ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_err()  { echo -e "${RED}[ERR]${NC}   $*" >&2; }
log_info() { echo -e "${BLUE}[INFO]${NC}  $*"; }

# ── Load .env if exists ─────────────────────────────────────────────────────
load_env() {
    if [ -f "$ENV_FILE" ]; then
        log_info "Loading env from $ENV_FILE"
        set -a; source "$ENV_FILE"; set +a
    else
        log_err ".env not found at $ENV_FILE"
        exit 1
    fi
}

# ── Pre-flight checks ────────────────────────────────────────────────────────
check_docker() {
    if ! command -v docker &>/dev/null; then
        log_err "docker not found in PATH"
        exit 1
    fi
    if ! docker info &>/dev/null; then
        log_err "Docker daemon not running"
        exit 1
    fi
    log_ok "Docker daemon running"
}

check_env_file() {
    load_env
    local missing=""
    for var in MINIO_ROOT_USER MINIO_ROOT_PASSWORD GRAFANA_PASSWORD REDIS_PASSWORD \
               MEMSTREAM_MODEL_SIGNING_KEY IEC_SIGNING_KEY \
               INTERNAL_API_KEY METRICS_API_KEY; do
        if [ -z "${!var}" ] || [[ "${!var}" == changeme* ]]; then
            missing="$missing $var"
        fi
    done
    if [ -n "$missing" ]; then
        log_warn "Some secrets are placeholders:$missing"
        log_warn "Update $ENV_FILE before production use"
    else
        log_ok "All required secrets present"
    fi
}

# ── Check demo data exists ───────────────────────────────────────────────────
check_demo_data() {
    local demo_file="$DEPLOYMENT_DIR/data/demo_trips.parquet"
    if [ -f "$demo_file" ]; then
        local size_mb
        size_mb=$(du -h "$demo_file" 2>/dev/null | cut -f1 || echo "?")
        log_ok "Demo data found: $demo_file ($size_mb)"
    else
        log_warn "Demo data not found: $demo_file"
        log_info "Generating demo dataset..."
        "$PROJECT_ROOT/.venv/Scripts/python.exe" \
            "$PROJECT_ROOT/HP_benchmark_v5/data/inject_anomalies_memstream.py" \
            --slice demo
        log_ok "Demo data generated"
    fi
}

# ── Stop existing containers ─────────────────────────────────────────────────
stop_existing() {
    log_step "Stopping existing containers..."
    docker compose -f "$DEPLOYMENT_DIR/docker-compose.yml" down 2>/dev/null || true
    log_info "Existing containers stopped"
}

# ── Build Flink image ────────────────────────────────────────────────────────
build_flink() {
    local image="ldt-flink:1.18.1-py"
    if docker image inspect "$image" &>/dev/null; then
        log_ok "Flink image $image already exists (using cached layers)"
    else
        log_step "Building Flink image $image (~5-15 min first time)..."
        docker build -t "$image" -f "$DEPLOYMENT_DIR/flink/Dockerfile" "$PROJECT_ROOT"
    fi
}

# ── Build kafka-producer image ───────────────────────────────────────────────
build_producer() {
    local image="ldt-kafka-producer:latest"
    if docker image inspect "$image" &>/dev/null; then
        log_ok "Producer image already exists"
    else
        log_step "Building kafka-producer image..."
        docker build -t "$image" -f "$DEPLOYMENT_DIR/kafka/Dockerfile.producer" \
            "$DEPLOYMENT_DIR/kafka"
    fi
}

# ── Full docker compose up ───────────────────────────────────────────────────
deploy_all() {
    log_step "Deploying all services via docker compose..."
    docker compose -f "$DEPLOYMENT_DIR/docker-compose.yml" up -d
    log_ok "All services started"
}

# ── Wait for critical services ───────────────────────────────────────────────
wait_healthy() {
    log_step "Waiting for services to become healthy..."

    local timeout=120 interval=5 elapsed=0
    local services=("zookeeper" "kafka" "minio" "redis" "flink-jobmanager")

    for svc in "${services[@]}"; do
        log_info "Waiting for $svc..."
        elapsed=0
        while ! docker ps --filter "name=ldt-$svc" --filter "status=running" --format '{{.Names}}' | grep -q "ldt-$svc"; do
            sleep $interval
            elapsed=$((elapsed + interval))
            if [ $elapsed -ge $timeout ]; then
                log_err "$svc did not become healthy within ${timeout}s"
                log_info "Check: docker compose -f $DEPLOYMENT_DIR/docker-compose.yml logs $svc"
            fi
        done
        log_ok "$svc is running"
    done

    log_info "Waiting extra 30s for dependent services to initialise..."
    sleep 30
}

# ── Upload model to MinIO ─────────────────────────────────────────────────────
upload_model() {
    log_step "Uploading MemStream model to MinIO ml-models/ bucket..."

    local model_path="$PROJECT_ROOT/models/memstream/memstream_checkpoint_v1.pt"
    local hmac_path="$model_path.hmac"

    if [ ! -f "$model_path" ]; then
        log_warn "Model not found at $model_path"
        log_info "Searching for .pt files..."
        find "$PROJECT_ROOT" -name "*.pt" -not -path "*/.venv/*" 2>/dev/null | head -5
        return
    fi

    # Upload model files via minio-init container's mc client
    # Start a temp container using minio/mc to copy files
    docker run --rm --network cadqstream-net \
        -e MC_HOST_local="http://minio:9000" \
        minio/mc:latest sh -c "
            mc alias set local http://minio:9000 '$MINIO_ROOT_USER' '$MINIO_ROOT_PASSWORD' 2>/dev/null || true
            mc cp '$model_path' local/ml-models/ 2>/dev/null || log_err 'Model upload failed'
            [ -f '$hmac_path' ] && mc cp '$hmac_path' local/ml-models/ 2>/dev/null || true
            mc ls local/ml-models/
        " || true

    # Alternative: upload via minio bucket init if ml-models already exists
    docker exec ldt-minio mc ls local/ml-models/ 2>/dev/null || \
        docker exec ldt-minio mc mb local/ml-models 2>/dev/null || true

    if [ -f "$model_path" ]; then
        docker cp "$model_path" ldt-minio:/tmp/memstream_checkpoint_v1.pt 2>/dev/null || true
        docker exec ldt-minio mc cp /tmp/memstream_checkpoint_v1.pt local/ml-models/ 2>/dev/null || \
            log_warn "Could not copy model via docker cp (MinIO may not be reachable yet)"
    fi

    docker exec ldt-minio mc ls local/ml-models/ 2>/dev/null && \
        log_ok "Model uploaded to MinIO" || \
        log_warn "Model upload pending — MinIO may still be initializing"
}

# ── Health check ─────────────────────────────────────────────────────────────
run_healthcheck() {
    log_step "Running healthcheck..."
    bash "$SCRIPT_DIR/healthcheck.sh"
}

# ── Report URLs ──────────────────────────────────────────────────────────────
report_urls() {
    echo ""
    echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${GREEN}  CA-DQStream Deployment Complete${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}Service${NC}              ${CYAN}URL${NC}                                    ${CYAN}Credentials${NC}"
    echo "  ─────────────────────────────────────────────────────────────────────────────────"
    echo "  Kafka UI             http://localhost:8080               (no auth)"
    echo "  Flink UI            http://localhost:8081               (no auth)"
    echo "  Grafana             http://localhost:3000               admin / ${GRAFANA_PASSWORD}"
    echo "  Prometheus          http://localhost:9090               (no auth)"
    echo "  MinIO Console       http://localhost:9001               ${MINIO_ROOT_USER} / ${MINIO_ROOT_PASSWORD}"
    echo ""
    echo "  ${CYAN}Kafka Topics${NC}"
    echo "  ─────────────────────────────────────────────────────────────────────────────────"
    docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null | sed 's/^/    /' || echo "    (not available)"
    echo ""
    echo "  ${CYAN}MinIO Buckets${NC}"
    echo "  ─────────────────────────────────────────────────────────────────────────────────"
    docker exec ldt-minio mc ls local/ 2>/dev/null | awk '{print "    "$NF}' || echo "    (not available)"
    echo ""
    echo -e "  ${CYAN}Next Steps:${NC}"
    echo "  ─────────────────────────────────────────────────────────────────────────────────"
    echo "  1. Verify Flink job: http://localhost:8081"
    echo "  2. Monitor Grafana:  http://localhost:3000"
    echo "  3. Inject anomalies:  bash deployment/scripts/inject-test-data.ps1"
    echo "  4. Full healthcheck: bash deployment/scripts/healthcheck.sh"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  CA-DQStream Production Deploy${NC}"
    echo -e "${CYAN}  Kafka + Flink + MemStream + Prometheus + Grafana + MinIO${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""

    load_env
    check_docker
    check_env_file
    check_demo_data
    stop_existing
    build_flink
    build_producer
    deploy_all
    wait_healthy
    upload_model
    run_healthcheck
    report_urls

    echo -e "${GREEN}Deploy complete.${NC}"
}

main "$@"
