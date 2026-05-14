#!/usr/bin/env bash
# =============================================================================
# Kafka Topic Creation Script for CA-DQStream + MemStream v5
# =============================================================================
# Creates Kafka topics with proper replication factor and ISR settings
#
# Usage: ./create_topics.sh [bootstrap-server]
# Default: kafka:9092
#
# Reference: original_flow.md lines 279-293
# =============================================================================

set -euo pipefail

# Configuration
BOOTSTRAP_SERVER="${1:-${KAFKA_BOOTSTRAP_SERVERS:-kafka:9092}}"
REPLICATION_FACTOR="${KAFKA_REPLICATION_FACTOR:-1}"
MIN_INSYNC_REPLICAS="${KAFKA_MIN_INSYNC_REPLICAS:-1}"
NUM_PARTITIONS="${KAFKA_NUM_PARTITIONS:-4}"

# Retention settings (milliseconds)
RETENTION_7_DAYS=604800000
RETENTION_30_DAYS=2592000000
RETENTION_1_DAY=86400000

echo "=========================================="
echo "CA-DQStream + MemStream Kafka Topic Setup"
echo "=========================================="
echo "Bootstrap Server: $BOOTSTRAP_SERVER"
echo "Replication Factor: $REPLICATION_FACTOR"
echo "Min In-Sync Replicas: $MIN_INSYNC_REPLICAS"
echo "Partitions: $NUM_PARTITIONS"

# Topic definitions: name:partitions:retention_ms:cleanup
# Reference: original_flow.md lines 284-293
TOPICS=(
    "taxi-nyc-raw:$NUM_PARTITIONS:$RETENTION_7_DAYS:delete"                    # Input data
    "dq-stream-processed:$NUM_PARTITIONS:$RETENTION_7_DAYS:delete"               # Valid records after L1
    "dq-stream-anomalies:$NUM_PARTITIONS:$RETENTION_30_DAYS:compact"            # Canary violations + ML anomalies
    "dq-meta-stream:$NUM_PARTITIONS:$RETENTION_7_DAYS:delete"                   # Voting results + meta metrics
    "dq-hard-rule-violations:$NUM_PARTITIONS:$RETENTION_30_DAYS:compact"         # L1 schema violations
    "dq-stream-processed-clean:1:$RETENTION_7_DAYS:delete"                       # Clean records (canary passed)
    "iec-action-replay:1:$RETENTION_1_DAY:delete"                              # IEC decisions
    "if-model-updates:1:$RETENTION_7_DAYS:compact"                             # Model update events
)

# Dead Letter Queue topics
DLQ_TOPICS=(
    "dq-dlq:1:$RETENTION_7_DAYS:delete"                                         # General DLQ
    "dq-hard-rule-violations-dlq:1:$RETENTION_30_DAYS:delete"                  # L1 violations DLQ
    "dq-stream-anomalies-dlq:1:$RETENTION_30_DAYS:delete"                      # Anomalies DLQ
    "iec-action-replay-dlq:1:$RETENTION_1_DAY:delete"                           # IEC DLQ
)

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $*"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

check_prerequisites() {
    if ! command -v kafka-topics.sh &> /dev/null; then
        log_error "kafka-topics.sh not found. Please install Kafka and add it to PATH."
        exit 1
    fi
    log_info "Kafka tools found"
}

wait_for_kafka() {
    local max_attempts=30
    local attempt=1

    log_info "Waiting for Kafka to be available at ${BOOTSTRAP_SERVER}..."

    while [ $attempt -le $max_attempts ]; do
        if kafka-broker-api-versions.sh --bootstrap-server "${BOOTSTRAP_SERVER}" &> /dev/null; then
            log_info "Kafka is available"
            return 0
        fi
        log_info "Attempt ${attempt}/${max_attempts} - Kafka not ready, waiting 5 seconds..."
        sleep 5
        ((attempt++))
    done

    log_error "Kafka did not become available after ${max_attempts} attempts"
    exit 1
}

create_topic() {
    local topic_name="$1"
    local partitions="$2"
    local retention_ms="$3"
    local cleanup_policy="$4"

    log_info "Creating topic: $topic_name (partitions=$partitions, retention=${retention_ms}ms, cleanup=$cleanup_policy)"

    kafka-topics.sh \
        --create \
        --bootstrap-server "${BOOTSTRAP_SERVER}" \
        --topic "${topic_name}" \
        --partitions "$partitions" \
        --replication-factor "${REPLICATION_FACTOR}" \
        --config "retention.ms=${retention_ms}" \
        --config "cleanup.policy=${cleanup_policy}" \
        --config "compression.type=snappy" \
        --if-not-exists

    if [ $? -eq 0 ]; then
        log_info "Topic '$topic_name' created successfully"

        # Set min.insync.replicas
        kafka-configs.sh \
            --alter \
            --bootstrap-server "${BOOTSTRAP_SERVER}" \
            --topic "${topic_name}" \
            --add-config "min.insync.replicas=${MIN_INSYNC_REPLICAS}" \
            2>/dev/null || true
    else
        log_error "Failed to create topic '$topic_name'"
        return 1
    fi
}

describe_topic() {
    local topic_name="$1"

    kafka-topics.sh \
        --bootstrap-server "${BOOTSTRAP_SERVER}" \
        --describe \
        --topic "${topic_name}"
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    check_prerequisites
    wait_for_kafka

    # Create main topics
    log_info ""
    log_info "=========================================="
    log_info "Creating Main Topics"
    log_info "=========================================="

    for entry in "${TOPICS[@]}"; do
        IFS=':' read -r name partitions retention cleanup <<< "$entry"
        create_topic "$name" "$partitions" "$retention" "$cleanup"
    done

    # Create DLQ topics
    log_info ""
    log_info "=========================================="
    log_info "Creating Dead Letter Queue Topics"
    log_info "=========================================="

    for entry in "${DLQ_TOPICS[@]}"; do
        IFS=':' read -r name partitions retention cleanup <<< "$entry"
        create_topic "$name" "$partitions" "$retention" "$cleanup"
    done

    # Describe all topics
    log_info ""
    log_info "=========================================="
    log_info "Topic Summary"
    log_info "=========================================="

    kafka-topics.sh \
        --bootstrap-server "${BOOTSTRAP_SERVER}" \
        --list | grep -E "^(taxi-nyc|dq-|iec-|if-)" | sort | while read -r topic; do
            describe_topic "$topic"
            echo ""
        done

    log_info ""
    log_info "=========================================="
    log_info "Kafka topics created successfully!"
    log_info "=========================================="
    kafka-topics.sh --list --bootstrap-server "${BOOTSTRAP_SERVER}"
}

# Run main function
main "$@"
