#!/bin/bash
# Run Complete 4-Layer Pipeline
# Prerequisite: Docker services must be running (from smoke test)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "CA-DQStream Complete Pipeline (4 Layers)"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Verify Docker services are running
echo -e "${YELLOW}Checking Docker services...${NC}"
RUNNING=$(docker compose ps --services --filter "status=running" | wc -l)
if [ "$RUNNING" -lt 4 ]; then
    echo "⚠️  Docker services not running. Start with:"
    echo "   docker compose up -d"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker services running"

# Verify models exist
echo -e "\n${YELLOW}Checking ML models...${NC}"
if [ ! -f "$PROJECT_ROOT/models/iforest_model_v2.pkl" ]; then
    echo "❌ ML model not found. Train with:"
    echo "   python src/ml/train_iforest.py --n-trees 200 --height 10 --window-size 512"
    exit 1
fi
echo -e "${GREEN}✓${NC} ML models ready"

# Check if Kafka topics exist
echo -e "\n${YELLOW}Checking Kafka topics...${NC}"
TOPICS=$(docker exec cadqstream_kafka_1 kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null | grep -c "taxi-nyc-raw" || echo "0")
if [ "$TOPICS" -eq 0 ]; then
    echo "⚠️  Kafka topics not created. Create with:"
    echo "   ./scripts/create_topics.sh"
    exit 1
fi
echo -e "${GREEN}✓${NC} Kafka topics ready"

echo ""
echo "=========================================="
echo "Starting Complete Pipeline"
echo "=========================================="
echo ""
echo "Pipeline includes:"
echo "  Layer 1: Parse → Watermark → KeyGen → Dedup → Schema"
echo "  Layer 2: Canary (7 rules) + Complex (ML scoring)"
echo "  Layer 3: Rendezvous → Voting → MetaAggregator"
echo "  Layer 4: IEC (ADWIN-U + METER)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run complete pipeline
cd "$PROJECT_ROOT"
python src/flink_job_complete.py
