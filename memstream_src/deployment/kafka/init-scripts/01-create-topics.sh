#!/bin/bash

# Kafka Topic Initialization Script
# This script is run by the kafka-init container to create all required topics

set -e

BOOTSTRAP_SERVER="${KAFKA_BOOTSTRAP_SERVERS:-kafka:9092}"
PARTITIONS="${KAFKA_PARTITIONS:-4}"
RETENTION_HOURS="${KAFKA_RETENTION_MS:-604800000}"

echo "=========================================="
echo "Kafka Topic Initialization"
echo "=========================================="
echo "Bootstrap Servers: ${BOOTSTRAP_SERVER}"
echo "Partitions: ${PARTITIONS}"
echo "Retention: ${RETENTION_HOURS}ms"
echo "=========================================="

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
until kafka-broker-api-versions --bootstrap-server "${BOOTSTRAP_SERVER}" --timeout 10 > /dev/null 2>&1; do
    echo "Kafka not ready, waiting..."
    sleep 5
done
echo "Kafka is ready!"

# Function to create topic
create_topic() {
    local TOPIC_NAME="$1"
    local NUM_PARTITIONS="$2"
    local RETENTION_MS="$3"

    echo ""
    echo "Creating topic: ${TOPIC_NAME}"

    kafka-topics --bootstrap-server "${BOOTSTRAP_SERVER}" \
        --create \
        --if-not-exists \
        --topic "${TOPIC_NAME}" \
        --partitions "${NUM_PARTITIONS}" \
        --replication-factor 1 \
        --config retention.ms="${RETENTION_MS}"

    echo "Topic ${TOPIC_NAME} created successfully!"
}

# Create all required topics
echo ""
echo "Creating Kafka topics..."

# Main data stream
create_topic "taxi-nyc-raw" "${PARTITIONS}" "${RETENTION_HOURS}"

# Data quality topics
create_topic "dq-stream-anomalies" "${PARTITIONS}" "${RETENTION_HOURS}"
create_topic "dq-meta-stream" "${PARTITIONS}" "${RETENTION_HOURS}"
create_topic "dq-hard-rule-violations" "${PARTITIONS}" "${RETENTION_HOURS}"
create_topic "dq-stream-processed" "${PARTITIONS}" "${RETENTION_HOURS}"

# IEC action replay
create_topic "iec-action-replay" "1" "2592000000"

echo ""
echo "=========================================="
echo "All topics created successfully!"
echo "=========================================="

# List all topics
echo ""
echo "Listing all topics:"
kafka-topics --bootstrap-server "${BOOTSTRAP_SERVER}" --list

echo ""
echo "Topic initialization complete!"
