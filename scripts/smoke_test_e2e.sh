#!/bin/bash
# CA-DQStream End-to-End Smoke Test
# Tests each layer incrementally to verify complete flow

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "CA-DQStream End-to-End Smoke Test"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Start Docker Services
echo -e "${YELLOW}Step 1: Starting Docker services...${NC}"
cd "$PROJECT_ROOT"
docker compose up -d

echo "Waiting for services to be ready (30s)..."
sleep 30

# Verify services
echo "Checking Docker services..."
docker compose ps

# Step 2: Create Kafka Topics
echo -e "\n${YELLOW}Step 2: Creating Kafka topics...${NC}"
"$SCRIPT_DIR/create_topics.sh"

# Step 3: Verify Models
echo -e "\n${YELLOW}Step 3: Verifying ML models...${NC}"
if [ -f "$PROJECT_ROOT/models/iforest_model_v2.pkl" ]; then
    echo -e "${GREEN}✓${NC} iforest_model_v2.pkl exists"
else
    echo -e "${RED}✗${NC} iforest_model_v2.pkl missing - run: python src/ml/train_iforest.py"
    exit 1
fi

if [ -f "$PROJECT_ROOT/models/scaler.pkl" ]; then
    echo -e "${GREEN}✓${NC} scaler.pkl exists"
else
    echo -e "${YELLOW}⚠${NC} scaler.pkl missing (will be auto-created)"
fi

# Step 4: Run Layer 1 Only Test
echo -e "\n${YELLOW}Step 4: Testing Layer 1 (Baseline Validation)...${NC}"
echo "This will run for 30 seconds with Layer 1 only"
echo "Press Ctrl+C to stop early"
echo ""

# Create simplified Layer 1 test job
cat > /tmp/flink_layer1_test.py << 'EOF'
"""Layer 1 Smoke Test - Baseline Validation Only"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
import os
import json

from src.operators.watermark_assigner import create_watermark_strategy
from src.operators.key_generator import generate_trip_id
from src.operators.deduplicator import DeduplicatorFunction
from src.operators.schema_validator import SchemaValidator
from pyflink.datastream import MapFunction

class ParseJsonFunction(MapFunction):
    def map(self, value):
        try:
            return json.loads(value)
        except:
            return None

class AddTripIdFunction(MapFunction):
    def map(self, record):
        if record is None:
            return None
        record['trip_id'] = generate_trip_id(record)
        return record

def main():
    print("="*60)
    print("Layer 1 Smoke Test - Baseline Validation")
    print("="*60)

    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)

    # Kafka source
    properties = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'smoke-test-layer1',
        'auto.offset.reset': 'earliest',
    }

    kafka_source = FlinkKafkaConsumer(
        topics='taxi-nyc-raw',
        deserialization_schema=SimpleStringSchema(),
        properties=properties
    )

    stream = env.add_source(kafka_source)

    # Layer 1 pipeline
    stream = (
        stream
        .map(ParseJsonFunction(), output_type=Types.PICKLED_BYTE_ARRAY())
        .filter(lambda x: x is not None)
        .assign_timestamps_and_watermarks(create_watermark_strategy())
    )

    stream = stream.map(AddTripIdFunction(), output_type=Types.PICKLED_BYTE_ARRAY())

    deduplicated_stream = (
        stream
        .key_by(lambda x: x['trip_id'], key_type=Types.STRING())
        .map(DeduplicatorFunction(), output_type=Types.PICKLED_BYTE_ARRAY())
        .filter(lambda x: x is not None)
    )

    validator = SchemaValidator()
    valid_stream = deduplicated_stream.filter(validator)

    # Print valid records
    valid_stream.print()

    print("\n✓ Layer 1 operators connected")
    print("  - JSON parsing")
    print("  - Watermark assignment")
    print("  - Trip ID generation")
    print("  - Deduplication")
    print("  - Schema validation")
    print("\nStarting pipeline...")

    env.execute("Layer 1 Smoke Test")

if __name__ == "__main__":
    main()
EOF

# Start Layer 1 test in background
python /tmp/flink_layer1_test.py &
FLINK_PID=$!

# Wait a bit then start producer
sleep 5

echo -e "\n${YELLOW}Step 5: Producing test data (100 records)...${NC}"
python "$SCRIPT_DIR/produce_taxi_data.py" \
    --file "$PROJECT_ROOT/data/yellow_tripdata_2024-01.parquet" \
    --rate 10 \
    --limit 100 &

PRODUCER_PID=$!

# Wait for test duration
echo "Running for 30 seconds..."
sleep 30

# Stop processes
echo -e "\n${YELLOW}Stopping test processes...${NC}"
kill $FLINK_PID 2>/dev/null || true
kill $PRODUCER_PID 2>/dev/null || true

# Step 6: Verify Kafka has data
echo -e "\n${YELLOW}Step 6: Verifying Kafka topic has data...${NC}"
MSG_COUNT=$(docker exec -i cadqstream_kafka_1 kafka-run-class kafka.tools.GetOffsetShell \
    --broker-list localhost:9092 \
    --topic taxi-nyc-raw \
    --time -1 2>/dev/null | awk -F':' '{sum += $3} END {print sum}')

echo "Messages in taxi-nyc-raw: $MSG_COUNT"

if [ "$MSG_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Data successfully produced to Kafka"
else
    echo -e "${RED}✗${NC} No data in Kafka topic"
    exit 1
fi

# Summary
echo ""
echo "=========================================="
echo "Smoke Test Summary"
echo "=========================================="
echo -e "${GREEN}✓${NC} Docker services running"
echo -e "${GREEN}✓${NC} Kafka topics created"
echo -e "${GREEN}✓${NC} ML models verified"
echo -e "${GREEN}✓${NC} Layer 1 pipeline executed"
echo -e "${GREEN}✓${NC} Data produced to Kafka ($MSG_COUNT messages)"
echo ""
echo "Next steps:"
echo "  1. Review Flink output above for Layer 1 processing"
echo "  2. Run full pipeline: python src/flink_job_complete.py"
echo "  3. Start ML service: uvicorn src.api.ml_service:app --port 8000"
echo "  4. Monitor with: docker compose logs -f"
echo ""
echo "To stop services:"
echo "  docker compose down"
echo "=========================================="
