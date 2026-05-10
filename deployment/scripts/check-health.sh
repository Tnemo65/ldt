#!/bin/bash
# =============================================================================
# CA-DQStream - Health Check Script
# Verifies all services are running and healthy
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

HEALTHY=0
UNHEALTHY=0
WARNINGS=0

check_service() {
    local name="$1"
    local host_port="$2"
    local health_cmd="$3"

    echo -n "  $name: "

    if docker ps --filter "name=$name" --filter "status=running" --format "{{.Names}}" | grep -q "^${name}$"; then
        if [ -n "$health_cmd" ]; then
            if eval "$health_cmd" &>/dev/null; then
                echo -e "${GREEN}HEALTHY${NC}"
                HEALTHY=$((HEALTHY + 1))
            else
                echo -e "${YELLOW}RUNNING (app not ready)${NC}"
                WARNINGS=$((WARNINGS + 1))
            fi
        else
            echo -e "${GREEN}RUNNING${NC}"
            HEALTHY=$((HEALTHY + 1))
        fi
    elif docker ps --filter "name=$name" --format "{{.Names}}" | grep -q "^${name}$"; then
        echo -e "${RED}STOPPED${NC}"
        UNHEALTHY=$((UNHEALTHY + 1))
    else
        echo -e "${RED}NOT FOUND${NC}"
        UNHEALTHY=$((UNHEALTHY + 1))
    fi
}

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  CA-DQStream Health Check${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Kafka infrastructure
echo -e "${BLUE}[Kafka Infrastructure]${NC}"
check_service "ldt-zookeeper" "zookeeper:2181" "docker exec ldt-zookeeper bash -c 'echo ruok | nc localhost 2181 | grep -q imok'"
check_service "ldt-kafka" "kafka:9092" "docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list"
check_service "ldt-schema-registry" "schema-registry:8081" "curl -sf http://localhost:8082/subjects"
check_service "ldt-kafka-ui" "kafka-ui:8080" "curl -sf http://localhost:8080"
check_service "ldt-kafka-exporter" "kafka-exporter:9308" "curl -sf http://localhost:9308/metrics"

# Database
echo ""
echo -e "${BLUE}[Database]${NC}"
check_service "ldt-postgres" "postgres:5432" "docker exec ldt-postgres pg_isready -U cadqstream -d dq_pipeline"
check_service "ldt-pgbouncer" "pgbouncer:6432" ""
check_service "ldt-postgres-exporter" "postgres-exporter:9187" "curl -sf http://localhost:9187/metrics"

# Storage
echo ""
echo -e "${BLUE}[Storage]${NC}"
check_service "ldt-minio" "minio:9000" "docker exec ldt-minio mc ready local"

# Streaming
echo ""
echo -e "${BLUE}[Streaming (Flink)]${NC}"
check_service "ldt-flink-jobmanager" "flink-jobmanager:8081" "curl -sf http://localhost:8081/overview"
check_service "ldt-flink-taskmanager" "flink-taskmanager:8081" ""

# ML Platform
echo ""
echo -e "${BLUE}[ML Platform]${NC}"
check_service "ldt-mlflow" "mlflow:5000" "curl -sf http://localhost:5000"

# Observability
echo ""
echo -e "${BLUE}[Observability]${NC}"
check_service "ldt-prometheus" "prometheus:9090" "curl -sf http://localhost:9090/-/healthy"
check_service "ldt-grafana" "grafana:3000" "curl -sf http://localhost:3000/api/health"
check_service "ldt-node-exporter" "node-exporter:9100" "curl -sf http://localhost:9100/metrics"

# Kafka topics check
echo ""
echo -e "${BLUE}[Kafka Topics]${NC}"
if docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null | grep -q .; then
    docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null | sed 's/^/  /'
else
    echo -e "  ${YELLOW}(no topics found)${NC}"
fi

# Flink jobs check
echo ""
echo -e "${BLUE}[Flink Jobs]${NC}"
if curl -sf http://localhost:8081/jobs 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); jobs=d.get('jobs',[]); print(f'  {len(jobs)} job(s)'); [print(f'  - {j[\"id\"][:16]}... [{j[\"state\"]}]') for j in jobs[:5]]" 2>/dev/null; then
    :
else
    echo -e "  ${YELLOW}(could not query Flink jobs)${NC}"
fi

# PostgreSQL tables check
echo ""
echo -e "${BLUE}[PostgreSQL Tables]${NC}"
if docker exec ldt-postgres psql -U cadqstream -d dq_pipeline -c '\dt' 2>/dev/null | grep -q .; then
    docker exec ldt-postgres psql -U cadqstream -d dq_pipeline -c '\dt' 2>/dev/null | grep "^ " | head -10 | sed 's/^/  /'
else
    echo -e "  ${YELLOW}(could not query PostgreSQL tables)${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo "  Health Summary: ${GREEN}$HEALTHY healthy${NC}, ${YELLOW}$WARNINGS warnings${NC}, ${RED}$UNHEALTHY unhealthy${NC}"

if [ $UNHEALTHY -gt 0 ]; then
    echo -e "  ${RED}Some services are unhealthy. Run './scripts/stop.sh && ./scripts/start.sh' to restart.${NC}"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "  ${YELLOW}All containers running but some services may still be initializing.${NC}"
    exit 0
else
    echo -e "  ${GREEN}All services are healthy!${NC}"
    exit 0
fi
