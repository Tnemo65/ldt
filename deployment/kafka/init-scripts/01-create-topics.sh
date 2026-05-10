#!/bin/bash
set -e

KAFKA_HOST="${KAFKA_HOST:-kafka}"
KAFKA_PORT="${KAFKA_PORT:-9092}"
BOOTSTRAP="${KAFKA_HOST}:${KAFKA_PORT}"

echo "[kafka-init] Waiting for Kafka at ${BOOTSTRAP}..."
until kafka-topics --bootstrap-server "${BOOTSTRAP}" --list > /dev/null 2>&1; do
    echo "[kafka-init] Kafka not ready, waiting 5s..."
    sleep 5
done
echo "[kafka-init] Kafka is ready!"

echo "[kafka-init] Creating Kafka topics..."

# Input topic: raw taxi events from data producer
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic taxi-nyc-raw \
    --partitions 4 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete

# Output topic: processed/deduplicated records
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dq-stream-processed \
    --partitions 4 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete

# Anomaly topic: detected violations
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dq-stream-anomalies \
    --partitions 4 --replication-factor 1 \
    --config retention.ms=2592000000 \
    --config cleanup.policy=compact

# Meta metrics topic: windowed aggregates per neighborhood
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dq-meta-stream \
    --partitions 4 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete

# IEC replay topic: drift action replay buffer
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic iec-action-replay \
    --partitions 1 --replication-factor 1 \
    --config retention.ms=86400000 \
    --config cleanup.policy=delete

# Pipeline metrics topic
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dq-metrics \
    --partitions 1 --replication-factor 1 \
    --config retention.ms=2592000000 \
    --config cleanup.policy=compact

# Schema violations topic
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dq-hard-rule-violations \
    --partitions 4 --replication-factor 1 \
    --config retention.ms=2592000000 \
    --config cleanup.policy=compact

echo "[kafka-init] All topics created successfully!"
kafka-topics --bootstrap-server "${BOOTSTRAP}" --list
