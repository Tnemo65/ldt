#!/bin/bash
set -e

# =============================================================================
# Kafka Topic Initialization — Sequential Pipeline Phase 3
# Architecture: Layer1 -> Canary -> MemStream -> MetaAggregator -> IEC
#
# Active topics: taxi-nyc-raw-v2, iec-action-replay, iec-action-dlq,
#               memstream-model-updates, dq-stream-unified
# Dead topics removed: dq-stream-processed, dq-stream-anomalies, dq-meta-stream,
#                      dq-stream-processed-clean, dq-metrics, dq-hard-rule-violations,
#                      dlq-parse-error, dlq-schema-error, dlq-algorithm-error,
#                      dlq-validation-error
# =============================================================================

KAFKA_HOST="${KAFKA_HOST:-kafka}"
KAFKA_PORT="${KAFKA_PORT:-9092}"
BOOTSTRAP="${KAFKA_HOST}:${KAFKA_PORT}"

echo "[kafka-init] Waiting for Kafka at ${BOOTSTRAP}..."
until kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --list > /dev/null 2>&1; do
    echo "[kafka-init] Kafka not ready, waiting 5s..."
    sleep 5
done
echo "[kafka-init] Kafka is ready!"

echo "[kafka-init] Creating Kafka topics..."

# ── Active topics ──────────────────────────────────────────────────────────

# Input topic v2: canonical topic consumed by flink_job_complete.py
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic taxi-nyc-raw-v2 \
    --partitions 8 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete

# Legacy input topic: used by e2e_pipeline_submit.py baseline job only
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic taxi-nyc-raw \
    --partitions 8 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete

# IEC replay topic: drift action buffer -> consumed by action-replay-worker
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic iec-action-replay \
    --partitions 1 --replication-factor 1 \
    --config retention.ms=86400000 \
    --config cleanup.policy=delete

# IEC dead letter queue: exhausted retries from action-replay-worker
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic iec-action-dlq \
    --partitions 1 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete

# MemStream retrained model broadcast (compact — latest checkpoint wins)
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic memstream-model-updates \
    --partitions 4 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=compact

# Unified pipeline output: all event types (PROCESSED_RECORD, CANARY_VIOLATION,
# ANOMALY_RECORD, META_RECORD, IEC_DECISION) merged into one topic
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dq-stream-unified \
    --partitions 4 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete

# ── Dead topics (documented for potential re-addition) ─────────────────────
# dq-stream-processed: no producer (flink_job_complete writes to dq-stream-unified)
# dq-stream-anomalies: no producer (anomaly records written to MinIO)
# dq-meta-stream: no producer (meta records written to MinIO cadqstream-metrics)
# dq-stream-processed-clean: never produced or consumed
# dq-metrics: no producer (metrics via HTTP to cadqstream-metrics:9250)
# dq-hard-rule-violations: no producer (violations written to MinIO cadqstream-violations)
# dlq-parse-error, dlq-schema-error, dlq-algorithm-error, dlq-validation-error:
#   no producer (DLQ records embedded in dq-stream-unified with _dlq fields)

echo "[kafka-init] All topics created successfully!"
kafka-topics --bootstrap-server "${BOOTSTRAP}" --list

# ─── Schema Registry Integration ───────────────────────────────────────────
# Pipeline uses SimpleStringSchema (raw JSON) at runtime.
# Schemas registered here for documentation and future Avro migration.

SCHEMA_REGISTRY_URL="${SCHEMA_REGISTRY_URL:-http://schema-registry:8081}"

echo "[kafka-init] Waiting for Schema Registry at ${SCHEMA_REGISTRY_URL}..."
until curl -sf "${SCHEMA_REGISTRY_URL}/subjects" > /dev/null 2>&1; do
    echo "[kafka-init] Schema Registry not ready, waiting 5s..."
    sleep 5
done
echo "[kafka-init] Schema Registry is ready!"

register_schema() {
    local subject="$1"
    local schema_file="$2"
    echo "[kafka-init] Registering schema for subject '${subject}'..."
    curl -s -X POST "${SCHEMA_REGISTRY_URL}/subjects/${subject}/versions" \
        -H "Content-Type: application/vnd.schemaregistry.v1+json" \
        --data @"${schema_file}" \
        | grep -o '"id":[0-9]*' || echo "[kafka-init]   (already registered or skipped)"
}

# ── Active topic schemas ────────────────────────────────────────────────────

cat > /tmp/taxi-nyc-raw-schema.json << 'SCHEMA_EOF'
{
  "type": "record",
  "name": "TaxiNYCRaw",
  "namespace": "com.cadqstream.events",
  "doc": "NYC Yellow Taxi trip record from taxi-nyc-raw-v2 Kafka topic",
  "fields": [
    {"name": "VendorID",       "type": ["null", "int"],    "default": null},
    {"name": "tpep_pickup_datetime",  "type": "string",     "doc": "ISO8601 pickup timestamp"},
    {"name": "tpep_dropoff_datetime", "type": "string",    "doc": "ISO8601 dropoff timestamp"},
    {"name": "passenger_count", "type": ["null", "double"], "default": null},
    {"name": "trip_distance",  "type": ["null", "double"], "default": null},
    {"name": "RatecodeID",     "type": ["null", "double"], "default": null},
    {"name": "store_and_fwd_flag", "type": "string"},
    {"name": "PULocationID",  "type": ["null", "double"], "default": null, "doc": "NYC TLC zone 1-263"},
    {"name": "DOLocationID",  "type": ["null", "double"], "default": null, "doc": "NYC TLC zone 1-263"},
    {"name": "payment_type",   "type": ["null", "double"], "default": null},
    {"name": "fare_amount",   "type": ["null", "double"], "default": null},
    {"name": "extra",         "type": ["null", "double"], "default": null},
    {"name": "mta_tax",       "type": ["null", "double"], "default": null},
    {"name": "tip_amount",   "type": ["null", "double"], "default": null},
    {"name": "tolls_amount",  "type": ["null", "double"], "default": null},
    {"name": "improvement_surcharge", "type": ["null", "double"], "default": null},
    {"name": "total_amount",  "type": ["null", "double"], "default": null},
    {"name": "congestion_surcharge", "type": ["null", "double"], "default": null},
    {"name": "trip_duration", "type": ["null", "double"], "default": null, "doc": "Duration in hours"},
    {"name": "speed_mph",     "type": ["null", "double"], "default": null}
  ]
}
SCHEMA_EOF

cat > /tmp/iec-action-replay-schema.json << 'SCHEMA_EOF'
{
  "type": "record",
  "name": "IEMActionReplay",
  "namespace": "com.cadqstream.events",
  "doc": "IEC operator decision for action replay",
  "fields": [
    {"name": "scenario", "type": "string", "doc": "DRIFT_DETECTED|STABLE"},
    {"name": "neighborhood", "type": "string"},
    {"name": "metric_name", "type": "string"},
    {"name": "drift_indicator", "type": "string"},
    {"name": "drift_magnitude", "type": "double"},
    {"name": "neighborhood_count", "type": "int"},
    {"name": "strategy", "type": "string", "doc": "do_nothing|quick_retrain"},
    {"name": "iec_confidence", "type": "double"},
    {"name": "action_result", "type": {
      "type": "record",
      "name": "IECActionResult",
      "fields": [
        {"name": "action", "type": "string"},
        {"name": "message", "type": "string"},
        {"name": "new_threshold", "type": ["null", "double"], "default": null}
      ]
    }},
    {"name": "drifts_detected", "type": {"type": "array", "items": "string"}},
    {"name": "iec_timestamp", "type": "string"}
  ]
}
SCHEMA_EOF

cat > /tmp/dq-stream-unified-schema.json << 'SCHEMA_EOF'
{
  "type": "record",
  "name": "DQStreamUnified",
  "namespace": "com.cadqstream.events",
  "doc": "Unified pipeline output with event type tagging",
  "fields": [
    {"name": "_event_type", "type": "string", "doc": "PROCESSED_RECORD|CANARY_VIOLATION|ANOMALY_RECORD|META_RECORD|IEC_DECISION|DLQ_RECORD"},
    {"name": "_raw_value", "type": ["null", "string"], "default": null}
  ]
}
SCHEMA_EOF

register_schema "taxi-nyc-raw-value"      /tmp/taxi-nyc-raw-schema.json
register_schema "iec-action-replay-value"  /tmp/iec-action-replay-schema.json
register_schema "dq-stream-unified-value"  /tmp/dq-stream-unified-schema.json

# ── Dead topic schemas (commented — re-add if topics are re-implemented) ────
# DQ Processed record schema (dead topic: dq-stream-processed)
# cat > /tmp/dq-stream-processed-schema.json << 'SCHEMA_EOF'
# {
#   "type": "record",
#   "name": "DQStreamProcessed",
#   "namespace": "com.cadqstream.events",
#   "doc": "Processed taxi trip record from dq-stream-processed topic",
#   "fields": [
#     {"name": "trip_id", "type": "string"},
#     {"name": "has_violation", "type": "boolean"},
#     {"name": "canary_violations", "type": {"type": "array", "items": "string"}},
#     {"name": "anomaly_score", "type": "double"},
#     {"name": "is_anomaly", "type": "boolean"},
#     {"name": "final_decision", "type": "string"},
#     {"name": "neighborhood", "type": "string"}
#   ]
# }
# SCHEMA_EOF
# register_schema "dq-stream-processed-value" /tmp/dq-stream-processed-schema.json

# DQ Meta-Metrics schema (dead topic: dq-meta-stream)
# cat > /tmp/dq-meta-stream-schema.json << 'SCHEMA_EOF'
# {
#   "type": "record",
#   "name": "DQMetaStream",
#   "namespace": "com.cadqstream.events",
#   "doc": "Windowed meta-metrics from MetaAggregator",
#   "fields": [
#     {"name": "neighborhood", "type": "string"},
#     {"name": "window_start", "type": "string"},
#     {"name": "window_end", "type": "string"},
#     {"name": "volume", "type": "long"},
#     {"name": "violation_rate", "type": "double"},
#     {"name": "anomaly_rate", "type": "double"},
#     {"name": "avg_anomaly_score", "type": "double"},
#     {"name": "delta_score", "type": "double"}
#   ]
# }
# SCHEMA_EOF
# register_schema "dq-meta-stream-value" /tmp/dq-meta-stream-schema.json

# DLQ record schema (dead topics: dlq-parse-error, dlq-schema-error, etc.)
# cat > /tmp/dlq-record-schema.json << 'SCHEMA_EOF'
# {
#   "type": "record",
#   "name": "DQStreamDLQ",
#   "namespace": "com.cadqstream.events",
#   "doc": "Dead Letter Queue record with failure metadata",
#   "fields": [
#     {"name": "_dlq", "type": "boolean"},
#     {"name": "_dlq_reason", "type": "string"},
#     {"name": "_dlq_category", "type": "string"},
#     {"name": "_dlq_timestamp", "type": "string"},
#     {"name": "_dlq_operator", "type": "string"},
#     {"name": "_dlq_original", "type": "string"},
#     {"name": "trip_id", "type": "string"}
#   ]
# }
# SCHEMA_EOF
# register_schema "dlq-parse-error-value"       /tmp/dlq-record-schema.json
# register_schema "dlq-schema-error-value"     /tmp/dlq-record-schema.json
# register_schema "dlq-algorithm-error-value"  /tmp/dlq-record-schema.json
# register_schema "dlq-validation-error-value" /tmp/dlq-record-schema.json

echo "[kafka-init] Schema Registry registration complete."
echo "[kafka-init] Registered subjects:"
curl -s "${SCHEMA_REGISTRY_URL}/subjects" | python3 -m json.tool 2>/dev/null || \
    curl -s "${SCHEMA_REGISTRY_URL}/subjects"

rm -f /tmp/taxi-nyc-raw-schema.json /tmp/iec-action-replay-schema.json \
       /tmp/dq-stream-unified-schema.json /tmp/dq-stream-processed-schema.json \
       /tmp/dq-meta-stream-schema.json /tmp/dlq-record-schema.json

echo "[kafka-init] === Kafka initialization complete ==="
