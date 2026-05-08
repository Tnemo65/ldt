#!/bin/bash
# Start Data Producer for Testing
# Produces NYC Taxi data to Kafka topic

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default parameters
RATE=${1:-1000}        # Events per second (default: 1000)
LIMIT=${2:-10000}      # Max records (default: 10000)
DATA_FILE="$PROJECT_ROOT/data/yellow_tripdata_2024-01.parquet"

echo "=========================================="
echo "Starting Data Producer"
echo "=========================================="
echo "Rate: $RATE events/sec"
echo "Limit: $LIMIT records"
echo "File: $DATA_FILE"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "⚠️  Data file not found: $DATA_FILE"
    echo "Available files:"
    ls -lh "$PROJECT_ROOT/data/"*.parquet 2>/dev/null || echo "No parquet files found"
    exit 1
fi

# Run producer
cd "$PROJECT_ROOT"
python scripts/produce_taxi_data.py \
    --file "$DATA_FILE" \
    --rate "$RATE" \
    --limit "$LIMIT"
