#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAFKA_HOST="${KAFKA_HOST:-kafka}"
KAFKA_PORT="${KAFKA_PORT:-9092}"
BOOTSTRAP="${KAFKA_HOST}:${KAFKA_PORT}"

echo "[init.sh] Kafka init pipeline starting..."

# Step 1: Create topics (idempotent --if-not-exists)
echo "[init.sh] Step 1: Creating topics..."
bash "${SCRIPT_DIR}/01-create-topics.sh"

# Step 2: Fix compact topics (idempotent -- reads current state, only changes if wrong)
echo "[init.sh] Step 2: Fixing compact topic cleanup policies..."
bash "${SCRIPT_DIR}/02-fix-compact-topics.sh" --apply

echo "[init.sh] Kafka initialization pipeline complete."
