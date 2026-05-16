#!/bin/bash
set -e

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

# Input topic: raw taxi events from data producer
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic taxi-nyc-raw \
    --partitions 8 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete

# Input topic v2: canonical topic consumed by flink_job_complete.py
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic taxi-nyc-raw-v2 \
    --partitions 8 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete

# Output topic: processed/deduplicated records (increased from 4 → 8)
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dq-stream-processed \
    --partitions 8 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete

# Anomaly topic: detected violations (event log — use delete policy, NOT compact)
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dq-stream-anomalies \
    --partitions 4 --replication-factor 1 \
    --config retention.ms=2592000000 \
    --config cleanup.policy=delete

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

# IEC Dead Letter Queue: failed IEC actions after max retries (event log)
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic iec-action-dlq \
    --partitions 1 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete

# Clean canary records output topic
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dq-stream-processed-clean \
    --partitions 4 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete

# Pipeline metrics topic
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dq-metrics \
    --partitions 1 --replication-factor 1 \
    --config retention.ms=2592000000 \
    --config cleanup.policy=delete

# Schema violations topic
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dq-hard-rule-violations \
    --partitions 4 --replication-factor 1 \
    --config retention.ms=2592000000 \
    --config cleanup.policy=delete

# ML model broadcast topic (compacted for hot-swappable model updates, increased from 1 → 4)
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic if-model-updates \
    --partitions 4 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=compact

# MemStream retrained model broadcast topic (Phase 3D: for retrained weight broadcast)
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic memstream-model-updates \
    --partitions 4 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=compact

# Unified pipeline output topic (consolidates all separate output topics)
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dq-stream-unified \
    --partitions 4 --replication-factor 1 \
    --config retention.ms=604800000 \
    --config cleanup.policy=delete

# ─── DLQ Topics (Dead Letter Queue — ROOT CAUSE FIX #2) ─────────────────────
# Records that fail parsing, validation, or algorithm processing are routed here.
# Each DLQ category gets its own topic for independent downstream analysis.
kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dlq-parse-error \
    --partitions 2 --replication-factor 1 \
    --config retention.ms=2592000000 \
    --config cleanup.policy=delete

kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dlq-schema-error \
    --partitions 2 --replication-factor 1 \
    --config retention.ms=2592000000 \
    --config cleanup.policy=delete

kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dlq-algorithm-error \
    --partitions 2 --replication-factor 1 \
    --config retention.ms=2592000000 \
    --config cleanup.policy=delete

kafka-topics --bootstrap-server "${BOOTSTRAP}" \
    --create --if-not-exists \
    --topic dlq-validation-error \
    --partitions 2 --replication-factor 1 \
    --config retention.ms=2592000000 \
    --config cleanup.policy=delete

echo "[kafka-init] All topics created successfully!"
kafka-topics --bootstrap-server "${BOOTSTRAP}" --list

# ─── P6: Schema Registry Integration ─────────────────────────────────────────
# Register JSON schemas for pipeline topics via Schema Registry REST API.
# The Schema Registry runs on schema-registry:8081 inside the Docker network.

SCHEMA_REGISTRY_URL="${SCHEMA_REGISTRY_URL:-http://schema-registry:8081}"

echo "[kafka-init] Waiting for Schema Registry at ${SCHEMA_REGISTRY_URL}..."
until curl -sf "${SCHEMA_REGISTRY_URL}/subjects" > /dev/null 2>&1; do
    echo "[kafka-init] Schema Registry not ready, waiting 5s..."
    sleep 5
done
echo "[kafka-init] Schema Registry is ready!"

# Helper to register a schema (suppresses error if already registered)
register_schema() {
    local subject="$1"
    local schema_file="$2"
    echo "[kafka-init] Registering schema for subject '${subject}'..."
    curl -s -X POST "${SCHEMA_REGISTRY_URL}/subjects/${subject}/versions" \
        -H "Content-Type: application/vnd.schemaregistry.v1+json" \
        --data @"${schema_file}" \
        | grep -o '"id":[0-9]*' || echo "[kafka-init]   (already registered or skipped)"
}

# Taxi NYC Raw record schema (input topic)
cat > /tmp/taxi-nyc-raw-schema.json << 'SCHEMA_EOF'
{
  "type": "record",
  "name": "TaxiNYCRaw",
  "namespace": "com.cadqstream.events",
  "doc": "NYC Yellow Taxi trip record from taxi-nyc-raw Kafka topic",
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

# DQ Processed record schema (output topic)
cat > /tmp/dq-stream-processed-schema.json << 'SCHEMA_EOF'
{
  "type": "record",
  "name": "DQStreamProcessed",
  "namespace": "com.cadqstream.events",
  "doc": "Processed and validated taxi trip record from dq-stream-processed topic",
  "fields": [
    {"name": "trip_id", "type": "string", "doc": "MurmurHash3 of trip key"},
    {"name": "VendorID", "type": ["null", "int"], "default": null},
    {"name": "tpep_pickup_datetime", "type": "string"},
    {"name": "tpep_dropoff_datetime", "type": "string"},
    {"name": "passenger_count", "type": "int"},
    {"name": "trip_distance", "type": "double"},
    {"name": "PULocationID", "type": "int", "doc": "NYC TLC zone 1-263"},
    {"name": "DOLocationID", "type": "int", "doc": "NYC TLC zone 1-263"},
    {"name": "payment_type", "type": "int"},
    {"name": "fare_amount", "type": "double"},
    {"name": "total_amount", "type": "double"},
    {"name": "has_violation", "type": "boolean", "doc": "True if canary rules violated"},
    {"name": "canary_violations", "type": {"type": "array", "items": "string"}, "doc": "List of violated canary rules"},
    {"name": "anomaly_score", "type": "double", "doc": "ML anomaly score 0-1"},
    {"name": "is_anomaly", "type": "boolean", "doc": "ML anomaly flag"},
    {"name": "final_decision", "type": "string", "doc": "ANOMALY or CLEAN"},
    {"name": "neighborhood", "type": "string", "doc": "manhattan|brooklyn|queens|bronx|airport|staten_island"},
    {"name": "_producer_ts", "type": ["null", "string"], "default": null}
  ]
}
SCHEMA_EOF

# DQ Meta-Metrics schema (output topic)
cat > /tmp/dq-meta-stream-schema.json << 'SCHEMA_EOF'
{
  "type": "record",
  "name": "DQMetaStream",
  "namespace": "com.cadqstream.events",
  "doc": "Windowed meta-metrics from MetaAggregator for IEC drift detection",
  "fields": [
    {"name": "neighborhood", "type": "string"},
    {"name": "neighborhood_id", "type": "string"},
    {"name": "window_start", "type": "string", "doc": "ISO8601 window start"},
    {"name": "window_end", "type": "string", "doc": "ISO8601 window end"},
    {"name": "volume", "type": "long", "doc": "Record count in window"},
    {"name": "null_rate", "type": "double", "doc": "Fraction of records with null fields"},
    {"name": "violation_rate", "type": "double", "doc": "Fraction of Canary violations"},
    {"name": "anomaly_rate", "type": "double", "doc": "Fraction of ML anomalies"},
    {"name": "avg_anomaly_score", "type": "double", "doc": "Mean ML anomaly score"},
    {"name": "delta_score", "type": "double", "doc": "Change in anomaly_rate from previous window"}
  ]
}
SCHEMA_EOF

# IEC Action Replay schema
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
    {"name": "strategy", "type": "string", "doc": "do_nothing|adjust_threshold|retrain_model|switch_model"},
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

# Register schemas (suppress errors for idempotency)
register_schema "taxi-nyc-raw-value"         /tmp/taxi-nyc-raw-schema.json
register_schema "dq-stream-processed-value"  /tmp/dq-stream-processed-schema.json
register_schema "dq-meta-stream-value"       /tmp/dq-meta-stream-schema.json
register_schema "iec-action-replay-value"    /tmp/iec-action-replay-schema.json
# ROOT CAUSE FIX #3: Register unified topic schema
cat > /tmp/dq-stream-unified-schema.json << 'SCHEMA_EOF'
{
  "type": "record",
  "name": "DQStreamUnified",
  "namespace": "com.cadqstream.events",
  "doc": "Unified pipeline output with event type tagging",
  "fields": [
    {"name": "_event_type", "type": "string"},
    {"name": "_raw_value", "type": ["null", "string"], "default": null}
  ]
}
SCHEMA_EOF
register_schema "dq-stream-unified-value" /tmp/dq-stream-unified-schema.json

# ROOT CAUSE FIX #2: Register DLQ topic schemas
cat > /tmp/dlq-record-schema.json << 'SCHEMA_EOF'
{
  "type": "record",
  "name": "DQStreamDLQ",
  "namespace": "com.cadqstream.events",
  "doc": "Dead Letter Queue record with failure metadata",
  "fields": [
    {"name": "_dlq", "type": "boolean", "doc": "Always true for DLQ records"},
    {"name": "_dlq_reason", "type": "string", "doc": "Human-readable failure reason"},
    {"name": "_dlq_category", "type": "string", "doc": "PARSE_ERROR|SCHEMA_ERROR|ALGORITHM_ERROR|VALIDATION_ERROR"},
    {"name": "_dlq_timestamp", "type": "string", "doc": "Event time from original record"},
    {"name": "_dlq_operator", "type": "string", "doc": "Operator that produced the failure"},
    {"name": "_dlq_original", "type": "string", "doc": "Serialized original record (JSON)"},
    {"name": "trip_id", "type": "string"},
    {"name": "tpep_pickup_datetime", "type": "string"}
  ]
}
SCHEMA_EOF
register_schema "dlq-parse-error-value"       /tmp/dlq-record-schema.json
register_schema "dlq-schema-error-value"     /tmp/dlq-record-schema.json
register_schema "dlq-algorithm-error-value"  /tmp/dlq-record-schema.json
register_schema "dlq-validation-error-value" /tmp/dlq-record-schema.json

echo "[kafka-init] Schema Registry registration complete."
echo "[kafka-init] Registered subjects:"
curl -s "${SCHEMA_REGISTRY_URL}/subjects" | python3 -m json.tool 2>/dev/null || \
    curl -s "${SCHEMA_REGISTRY_URL}/subjects"

# Cleanup temp schemas
rm -f /tmp/taxi-nyc-raw-schema.json /tmp/dq-stream-processed-schema.json \
       /tmp/dq-meta-stream-schema.json /tmp/iec-action-replay-schema.json \
       /tmp/dq-stream-unified-schema.json /tmp/dlq-record-schema.json

echo "[kafka-init] === Kafka initialization complete ==="
