#!/bin/bash
# =============================================================================
# CA-DQStream + MemStream v5 - Health Check Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================"
echo "  CA-DQStream + MemStream v5"
echo "  Health Check"
echo "============================================${NC}"
echo ""

check_service() {
    local name=$1
    local url=$2
    echo -n "  $name: "

    if curl -sf "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        return 1
    fi
}

check_exec() {
    local name=$1
    local container=$2
    local cmd=$3
    echo -n "  $name: "

    if docker exec "$container" $cmd > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        return 1
    fi
}

FAILURES=0

echo "=== Kafka Cluster ==="
check_service "Kafka" "http://localhost:8080/api/clusters" || ((FAILURES++))
check_service "Schema Registry" "http://localhost:8082/subjects" || ((FAILURES++))
check_exec "Kafka Topics" "cadqstream-kafka" "kafka-topics.sh --list --bootstrap-server localhost:9092" || ((FAILURES++))

echo ""
echo "=== Flink Cluster ==="
check_service "JobManager" "http://localhost:8081/overview" || ((FAILURES++))
check_service "TaskManager RPC" "http://localhost:6122" || ((FAILURES++))

echo ""
echo "=== Storage (MinIO) ==="
check_service "MinIO API" "http://localhost:9000/minio/health/live" || ((FAILURES++))
check_service "MinIO Console" "http://localhost:9001" || ((FAILURES++))

echo ""
echo "=== Redis ==="
check_exec "Redis" "cadqstream-redis" "redis-cli -a ${REDIS_PASSWORD:-password} ping" || ((FAILURES++))

echo ""
echo "=== Observability ==="
check_service "Prometheus" "http://localhost:9090/-/healthy" || ((FAILURES++))
check_service "Grafana" "http://localhost:3000/api/health" || ((FAILURES++))
check_service "Alertmanager" "http://localhost:9093/-/healthy" || ((FAILURES++))
check_service "Node Exporter" "http://localhost:9100" || ((FAILURES++))

echo ""
echo "=== ML Service ==="
check_service "ML Service" "http://localhost:8000/health" || ((FAILURES++))

echo ""
echo "=== Metrics Exporters ==="
check_service "Cadqstream Metrics" "http://localhost:9250/metrics" || ((FAILURES++))
check_service "Kafka Exporter" "http://localhost:9308" || ((FAILURES++))

echo ""
echo "=== Kafka Topics ==="
docker exec cadqstream-kafka kafka-topics.sh --list --bootstrap-server localhost:9092 2>/dev/null || echo "  Unable to list topics"

echo ""
echo "=== MinIO Buckets ==="
docker exec cadqstream-minio mc ls local/ 2>/dev/null || echo "  Unable to list buckets"

echo ""
echo -e "${BLUE}============================================${NC}"

if [ $FAILURES -eq 0 ]; then
    echo -e "  ${GREEN}All services are healthy!${NC}"
    echo -e "${BLUE}============================================${NC}"
    exit 0
else
    echo -e "  ${RED}$FAILURES service(s) failed health check${NC}"
    echo -e "${BLUE}============================================${NC}"
    exit 1
fi
