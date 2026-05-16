#!/bin/bash
# =============================================================================
# CA-DQStream Production Deployment - Startup Script
# Full reset, build, and deployment of the complete streaming pipeline
# Storage: MinIO only
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step() { echo -e "${CYAN}[STEP]${NC}  $*"; }

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  CA-DQStream Production Deployment${NC}"
echo -e "${CYAN}  4-Layer Streaming Pipeline on Apache Flink 1.17.1${NC}"
echo -e "${CYAN}  Storage: MinIO only${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 0: DOCKER RESET — Clean slate
# ═══════════════════════════════════════════════════════════════════════════
log_step "STEP 0: Docker Reset — Cleaning environment"

if command -v docker-compose &> /dev/null; then
    log_info "Stopping existing containers via docker-compose..."
    docker-compose -f "$DEPLOYMENT_DIR/docker-compose.yml" down -v --remove-orphans 2>/dev/null || true
fi

log_info "Stopping CA-DQStream containers..."
docker stop $(docker ps -q --filter "name=ldt-") 2>/dev/null || true

log_info "Removing CA-DQStream containers..."
docker rm -f $(docker ps -aq --filter "name=ldt-") 2>/dev/null || true

log_info "Removing CA-DQStream networks..."
docker network rm cadqstream-net 2>/dev/null || true
docker network rm ldt_cadqstream-net 2>/dev/null || true
docker network rm deployment_cadqstream-net 2>/dev/null || true
docker network rm deployment-cadqstream-net 2>/dev/null || true

log_info "Pruning unused Docker resources (images, build cache only)..."
docker image prune -af 2>/dev/null || true

log_ok "Docker reset complete."

# ═══════════════════════════════════════════════════════════════════════════
# STEP 0b: Verify ports are free
# ═══════════════════════════════════════════════════════════════════════════
log_step "STEP 0b: Verifying required ports are available..."
REQUIRED_PORTS=(2181 9092 8081 8080 8082 3000 9090 9100 9308 9000 9001)
PORT_NAMES=("Zookeeper" "Kafka" "Flink UI" "Kafka UI" "Schema Registry" "Grafana" "Prometheus" "Node Exporter" "Kafka Exporter" "MinIO API" "MinIO Console")

for i in "${!REQUIRED_PORTS[@]}"; do
    port="${REQUIRED_PORTS[$i]}"
    name="${PORT_NAMES[$i]}"
    if netstat -tlnp 2>/dev/null | grep -q ":${port} "; then
        log_err "Port $port ($name) is still in use!"
        netstat -tlnp 2>/dev/null | grep ":${port} " || true
        exit 1
    else
        log_ok "Port $port ($name) is free"
    fi
done

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Hardware Check
# ═══════════════════════════════════════════════════════════════════════════
log_step "STEP 1: Hardware Resource Audit"
if command -v nproc &> /dev/null; then
    CPU_CORES=$(nproc)
    log_info "CPU cores: $CPU_CORES"
fi

if command -v free &> /dev/null; then
    TOTAL_MEM=$(free -h 2>/dev/null | grep Mem | awk '{print $2}')
    log_info "Total RAM: $TOTAL_MEM"
fi

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)
    if [ -n "$GPU_INFO" ]; then
        log_ok "GPU detected: $GPU_INFO"
        log_info "GPU will be allocated to Flink TaskManager"
    else
        log_info "No NVIDIA GPU detected — running CPU-only"
    fi
else
    log_info "nvidia-smi not found — CPU-only deployment"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Build Custom Flink Image
# ═══════════════════════════════════════════════════════════════════════════
log_step "STEP 2: Building custom Flink image (ldt-flink:1.17.1-py)"

if [ -f "$DEPLOYMENT_DIR/flink/Dockerfile" ]; then
    log_info "Flink Dockerfile found at $DEPLOYMENT_DIR/flink/Dockerfile"

    FLINK_LIB="$DEPLOYMENT_DIR/flink"
    REQUIRED_JARS=(
        "flink-connector-kafka-1.17.1.jar"
        "flink-connector-jdbc-3.1.1-1.17.jar"
        "kafka-clients-3.5.1.jar"
    )

    for jar in "${REQUIRED_JARS[@]}"; do
        if [ -f "$FLINK_LIB/$jar" ]; then
            log_ok "JAR found: $jar"
        else
            log_warn "JAR NOT found: $jar — image build may fail if not present in Dockerfile"
        fi
    done

    log_info "Building image (this may take several minutes)..."
    if docker build -f "$DEPLOYMENT_DIR/flink/Dockerfile" \
        -t ldt-flink:1.17.1-py \
        "$DEPLOYMENT_DIR" 2>&1; then
        log_ok "Flink image built successfully"
    else
        log_err "Flink image build failed!"
        exit 1
    fi
else
    log_err "Flink Dockerfile not found at $DEPLOYMENT_DIR/flink/Dockerfile"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Create Network
# ═══════════════════════════════════════════════════════════════════════════
log_step "STEP 3: Docker network managed by docker-compose (auto-created on up)"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Start Infrastructure Services
# ═══════════════════════════════════════════════════════════════════════════
log_step "STEP 4: Starting infrastructure services"

docker compose -f "$DEPLOYMENT_DIR/docker-compose.yml" up -d zookeeper kafka schema-registry
log_info "Infrastructure services started (Zookeeper, Kafka, Schema Registry)"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Start Storage Services (MinIO only)
# ═══════════════════════════════════════════════════════════════════════════
log_step "STEP 5: Starting storage services (MinIO only)"

docker compose -f "$DEPLOYMENT_DIR/docker-compose.yml" up -d minio
log_info "Storage services started (MinIO)"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Wait for Services to be Healthy
# ═══════════════════════════════════════════════════════════════════════════
log_step "STEP 6: Waiting for services to become healthy"

log_info "Waiting for Kafka (kafka:9092)..."
wait_for_timeout=120
wait_for_interval=5
wait_for_elapsed=0

while ! docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list &>/dev/null; do
    sleep $wait_for_interval
    wait_for_elapsed=$((wait_for_elapsed + wait_for_interval))
    echo -ne "\r${BLUE}[INFO]${NC}  Kafka not ready (${wait_for_elapsed}s/${wait_for_timeout}s)... "
    if [ $wait_for_elapsed -ge $wait_for_timeout ]; then
        echo ""
        log_err "Kafka did not become ready within ${wait_for_timeout}s"
        exit 1
    fi
done
echo ""
log_ok "Kafka is ready"

log_info "Waiting for MinIO (minio:9000)..."
wait_for_elapsed=0
while ! docker exec ldt-minio mc ready local &>/dev/null 2>&1; do
    sleep $wait_for_interval
    wait_for_elapsed=$((wait_for_elapsed + wait_for_interval))
    echo -ne "\r${BLUE}[INFO]${NC}  MinIO not ready (${wait_for_elapsed}s/${wait_for_timeout}s)... "
    if [ $wait_for_elapsed -ge $wait_for_timeout ]; then
        echo ""
        log_err "MinIO did not become ready within ${wait_for_timeout}s"
        exit 1
    fi
done
echo ""
log_ok "MinIO is ready"

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Run Init Containers
# ═══════════════════════════════════════════════════════════════════════════
log_step "STEP 7: Running initialization containers"

docker compose -f "$DEPLOYMENT_DIR/docker-compose.yml" up -d kafka-init
log_info "Kafka init container ran (topic creation)"
docker logs ldt-kafka-init 2>&1 | tail -20

docker compose -f "$DEPLOYMENT_DIR/docker-compose.yml" up -d minio-init
log_info "MinIO init container ran (bucket creation)"
docker logs ldt-minio-init 2>&1 | tail -20

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: Start Application Services (Flink)
# ═══════════════════════════════════════════════════════════════════════════
log_step "STEP 8: Starting Flink services"

docker compose -f "$DEPLOYMENT_DIR/docker-compose.yml" up -d flink-jobmanager flink-taskmanager
log_info "Flink JobManager and TaskManagers started"

log_info "Waiting for Flink REST API..."
wait_for_elapsed=0
while ! curl -sf http://localhost:8081/overview &>/dev/null; do
    sleep $wait_for_interval
    wait_for_elapsed=$((wait_for_elapsed + wait_for_interval))
    echo -ne "\r${BLUE}[INFO]${NC}  Flink REST API not ready (${wait_for_elapsed}s/120s)... "
    if [ $wait_for_elapsed -ge 120 ]; then
        echo ""
        log_warn "Flink REST API did not become ready within 120s (may still be starting)"
        break
    fi
done
echo ""
if curl -sf http://localhost:8081/overview &>/dev/null; then
    log_ok "Flink REST API is ready"
else
    log_warn "Flink REST API may still be starting — check http://localhost:8081 manually"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: Initialize Flink Artifacts
# ═══════════════════════════════════════════════════════════════════════════
log_step "STEP 9: Initializing Flink artifacts (submitting pipeline job)"

if curl -sf http://localhost:8081/overview &>/dev/null; then
    docker compose -f "$DEPLOYMENT_DIR/docker-compose.yml" up flink-init
    docker logs ldt-flink-init 2>&1 | tail -30
else
    log_warn "Skipping Flink init — REST API not yet available"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STEP 10: Start Monitoring Stack
# ═══════════════════════════════════════════════════════════════════════════
log_step "STEP 10: Starting monitoring stack"

docker compose -f "$DEPLOYMENT_DIR/docker-compose.yml" up -d prometheus grafana kafka-exporter node-exporter
log_info "Monitoring services started (Prometheus, Grafana, exporters)"

log_info "Waiting for Prometheus..."
wait_for_elapsed=0
while ! curl -sf http://localhost:9090/-/healthy &>/dev/null; do
    sleep $wait_for_interval
    wait_for_elapsed=$((wait_for_elapsed + wait_for_interval))
    echo -ne "\r${BLUE}[INFO]${NC}  Prometheus not ready (${wait_for_elapsed}s/60s)... "
    if [ $wait_for_elapsed -ge 60 ]; then
        echo ""
        log_warn "Prometheus may still be starting"
        break
    fi
done
echo ""

log_info "Waiting for Grafana..."
wait_for_elapsed=0
while ! curl -sf http://localhost:3000/api/health &>/dev/null; do
    sleep $wait_for_interval
    wait_for_elapsed=$((wait_for_elapsed + wait_for_interval))
    echo -ne "\r${BLUE}[INFO]${NC}  Grafana not ready (${wait_for_elapsed}s/60s)... "
    if [ $wait_for_elapsed -ge 60 ]; then
        echo ""
        log_warn "Grafana may still be starting"
        break
    fi
done
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# STEP 11: Final Status
# ═══════════════════════════════════════════════════════════════════════════
log_step "STEP 11: Deployment verification"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  CA-DQStream Deployment Complete!${NC}"
echo -e "${GREEN}  MinIO-only storage${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  ${CYAN}Service${NC}              ${CYAN}URL / Endpoint${NC}                              ${CYAN}Credentials${NC}"
echo "  ─────────────────────────────────────────────────────────────────────────────────"
echo "  Kafka UI              http://localhost:8080                   (no auth)"
echo "  Flink UI             http://localhost:8081                   (no auth)"
echo "  Grafana              http://localhost:3000                    admin / \${GRAFANA_PASSWORD}"
echo "  Prometheus           http://localhost:9090                   (no auth)"
echo "  MinIO Console        http://localhost:9001                    \${MINIO_ROOT_USER} / \${MINIO_ROOT_PASSWORD}"
echo ""
echo "  ${CYAN}Data Connections${NC}"
echo "  ─────────────────────────────────────────────────────────────────────────────────"
echo "  Kafka PLAINTEXT      localhost:9092"
echo "  Schema Registry      localhost:8082"
echo "  MinIO API            localhost:9000"
echo ""
echo "  ${CYAN}Kafka Topics${NC}"
echo "  ─────────────────────────────────────────────────────────────────────────────────"
docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null | sed 's/^/  /' || echo "  (not available)"
echo ""
echo "  ${CYAN}MinIO Buckets${NC}"
echo "  ─────────────────────────────────────────────────────────────────────────────────"
docker exec ldt-minio mc ls local/ 2>/dev/null | sed 's/^/  /' || echo "  (not available)"
echo ""
echo "  ${CYAN}Running Containers${NC}"
echo "  ─────────────────────────────────────────────────────────────────────────────────"
docker ps --filter "name=ldt-" --format "  {{.Names}} ({{.Status}})" 2>/dev/null | head -20
echo ""

log_ok "Deployment complete. Check Grafana dashboards for pipeline monitoring."
echo ""
echo "  To check health:    bash deployment/scripts/check-health.sh"
echo "  To stop stack:      bash deployment/scripts/stop.sh"
echo "  To view logs:       docker compose -f deployment/docker-compose.yml logs -f [service]"
echo "  To restart:         bash deployment/scripts/start.sh"
echo ""
