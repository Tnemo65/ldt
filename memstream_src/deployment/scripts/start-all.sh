#!/bin/bash
# =============================================================================
# CA-DQStream + MemStream v5 - Startup Script
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

echo -e "${BLUE}=========================================="
echo "  CA-DQStream + MemStream v5"
echo "  Complete Deployment"
echo "==========================================${NC}"
echo ""

# Check for .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found!${NC}"
    echo "Creating from .env.example..."
    cp .env.example .env
    echo ""
    echo -e "${RED}Please edit .env and set all required secrets!${NC}"
    echo "Required variables:"
    echo "  - MEMSTREAM_MODEL_SIGNING_KEY"
    echo "  - IEC_SIGNING_KEY"
    echo "  - REDIS_PASSWORD"
    echo "  - MINIO_SECRET_KEY"
    echo ""
    echo "Generate keys with: openssl rand -hex 32"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed!${NC}"
    exit 1
fi

if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed!${NC}"
    exit 1
fi

echo -e "${GREEN}Starting CA-DQStream + MemStream v5...${NC}"
echo ""

# Create necessary directories
mkdir -p logs checkpoints 2>/dev/null || true

# Start services
docker compose up -d

echo ""
echo "Waiting for services to be ready..."

# Wait for Kafka
echo -n "  Kafka: "
until docker exec cadqstream-kafka kafka-broker-api-versions --bootstrap-server localhost:9092 &>/dev/null 2>&1; do
    sleep 2
done
echo "OK"

# Wait for MinIO
echo -n "  MinIO: "
until curl -sf http://localhost:9000/minio/health/live &>/dev/null 2>&1; do
    sleep 2
done
echo "OK"

# Wait for Flink
echo -n "  Flink: "
until curl -sf http://localhost:8081/overview &>/dev/null 2>&1; do
    sleep 5
done
echo "OK"

echo ""
echo -e "${GREEN}=========================================="
echo "  All Services Started Successfully!"
echo "==========================================${NC}"
echo ""
echo "Access URLs:"
echo ""
echo "  Kafka Cluster:"
echo "    - Kafka UI:        http://localhost:8080"
echo "    - Schema Registry: http://localhost:8082"
echo ""
echo "  Flink:"
echo "    - JobManager UI:   http://localhost:8081"
echo ""
echo "  Storage (MinIO):"
echo "    - MinIO API:       http://localhost:9000"
echo "    - MinIO Console:   http://localhost:9001"
echo ""
echo "  Observability:"
echo "    - Grafana:        http://localhost:3000"
echo "    - Prometheus:     http://localhost:9090"
echo "    - Alertmanager:   http://localhost:9093"
echo ""
echo "  ML Service:"
echo "    - API:            http://localhost:8000"
echo ""
echo "Default credentials:"
echo "  - Grafana: admin/admin123"
echo "  - MinIO:   minioadmin/minioadmin"
echo ""
echo "Kafka Topics:"
docker exec cadqstream-kafka kafka-topics.sh --list --bootstrap-server localhost:9092 2>/dev/null || true
echo ""
echo "To view logs: docker compose logs -f"
echo "To stop:     ./scripts/stop-all.sh"
echo ""
