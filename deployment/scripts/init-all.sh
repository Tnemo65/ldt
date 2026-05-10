#!/bin/bash
# =============================================================================
# CA-DQStream - Run All Initialization Scripts in Order
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$DEPLOYMENT_DIR"

echo "Running all initialization scripts in order..."

# Kafka init
echo "[kafka-init] Creating topics..."
docker compose -f docker-compose.yml up -d kafka-init
docker wait ldt-kafka-init || true
docker logs ldt-kafka-init | tail -10

# PostgreSQL init (handled by docker-compose volume mount)
echo "[postgres-init] Schema loaded via volume mount"
docker compose -f docker-compose.yml restart postgres
sleep 5

# MinIO init
echo "[minio-init] Creating buckets..."
docker compose -f docker-compose.yml up -d minio-init
docker wait ldt-minio-init || true
docker logs ldt-minio-init | tail -10

# Flink init
echo "[flink-init] Initializing Flink pipeline..."
docker compose -f docker-compose.yml up flink-init
docker logs ldt-flink-init | tail -30

echo "All init scripts complete."
