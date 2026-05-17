#!/bin/bash
# Fix Kafka topic configuration: change compact topics to delete policy.
# Idempotent: first run applies fixes, subsequent runs detect correct state and skip.
#
# Usage:
#   ./02-fix-compact-topics.sh              # dry-run (safe)
#   ./02-fix-compact-topics.sh --apply      # actually apply fixes

set -e

KAFKA_HOST="${KAFKA_HOST:-kafka}"
KAFKA_PORT="${KAFKA_PORT:-9092}"
BOOTSTRAP="${KAFKA_HOST}:${KAFKA_PORT}"

ACTION="${1:-}"
if [ "$ACTION" = "--apply" ]; then
    APPLY=true
else
    APPLY=false
    echo "[fix-topics] DRY RUN — pass --apply to actually modify topics"
fi

echo "[fix-topics] Waiting for Kafka at ${BOOTSTRAP}..."
until kafka-topics --bootstrap-server "${BOOTSTRAP}" --list > /dev/null 2>&1; do
    echo "[fix-topics] Kafka not ready, waiting 5s..."
    sleep 5
done
echo "[fix-topics] Kafka is ready!"

declare -A TOPIC_CONFIGS
TOPIC_CONFIGS["dq-stream-anomalies"]="delete|2592000000|4"
TOPIC_CONFIGS["iec-action-dlq"]="delete|604800000|1"

fix_topic() {
    local topic="$1"
    local expected_policy="$2"
    local expected_retention="$3"
    local expected_partitions="$4"

    # Idempotent fix: use kafka-configs --alter instead of delete/recreate.
    # Delete+recreate loses data and consumer offsets on every run.
    # kafka-configs --alter only modifies config, preserving topic state.

    local config_output
    config_output=$(kafka-configs --bootstrap-server "${BOOTSTRAP}" \
        --entity-type topics \
        --entity-name "${topic}" \
        --describe 2>/dev/null || true)

    local current_policy
    current_policy=$(echo "$config_output" | grep "cleanup.policy" | awk -F'=' '{print $3}' || echo "")
    local current_retention
    current_retention=$(echo "$config_output" | grep "retention.ms" | awk -F'=' '{print $3}' || echo "")

    if [ "$current_policy" = "$expected_policy" ]; then
        echo "[fix-topics] ${topic}: cleanup.policy=${current_policy} (OK, no change needed)"
        return 0
    fi

    if [ -z "$current_policy" ]; then
        # Topic does not exist — create it with the correct policy
        if [ "$APPLY" = "false" ]; then
            echo "[fix-topics] ${topic}: WOULD create with cleanup.policy=${expected_policy} (dry-run)"
        else
            echo "[fix-topics] ${topic}: creating with cleanup.policy=${expected_policy}..."
            kafka-topics --bootstrap-server "${BOOTSTRAP}" \
                --create --if-not-exists \
                --topic "${topic}" \
                --partitions "${expected_partitions}" --replication-factor 1 \
                --config "cleanup.policy=${expected_policy}" \
                --config "retention.ms=${expected_retention}" \
                > /dev/null 2>&1
            echo "[fix-topics] ${topic}: created with cleanup.policy=${expected_policy}"
        fi
        return 0
    fi

    # Topic exists but has wrong policy — alter the config instead of deleting
    echo "[fix-topics] ${topic}: current cleanup.policy=${current_policy}, setting to ${expected_policy}"

    if [ "$APPLY" = "false" ]; then
        echo "[fix-topics] ${topic}: WOULD alter cleanup.policy to ${expected_policy} (dry-run)"
        return 0
    fi

    kafka-configs --bootstrap-server "${BOOTSTRAP}" \
        --entity-type topics \
        --entity-name "${topic}" \
        --alter --add-config "cleanup.policy=${expected_policy}" \
        > /dev/null 2>&1 || {
        echo "[fix-topics] ${topic}: kafka-configs alter failed — trying create-if-not-exists..."
        kafka-topics --bootstrap-server "${BOOTSTRAP}" \
            --create --if-not-exists \
            --topic "${topic}" \
            --partitions "${expected_partitions}" --replication-factor 1 \
            --config "cleanup.policy=${expected_policy}" \
            --config "retention.ms=${expected_retention}" \
            > /dev/null 2>&1
    }

    echo "[fix-topics] ${topic}: cleanup.policy set to ${expected_policy}"
}

echo "[fix-topics] Scanning topics for incorrect cleanup policies..."
for topic in "${!TOPIC_CONFIGS[@]}"; do
    IFS='|' read -r expected_policy expected_retention expected_partitions <<< "${TOPIC_CONFIGS[$topic]}"
    fix_topic "$topic" "$expected_policy" "$expected_retention" "$expected_partitions"
done

echo "[fix-topics] Done."
