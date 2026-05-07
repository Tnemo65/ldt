#!/bin/bash
# scripts/create_topics.sh

KAFKA_CONTAINER="ldt-kafka-1"

topics=(
  "taxi-nyc-raw:12:delete:604800000"
  "dq-schema-violations:12:delete:604800000"
  "dq-hard-rule-violations:12:delete:604800000"
  "dq-anomaly-scores:12:delete:604800000"
  "dq-meta-stream:12:delete:604800000"
  "if-model-updates:1:compact:0"
  "iec-action-replay:12:delete:604800000"
)

echo "Creating Kafka topics..."

for topic_spec in "${topics[@]}"; do
  IFS=':' read -r topic partitions cleanup retention <<< "$topic_spec"

  echo "  Creating: $topic ($partitions partitions, $cleanup policy)"

  docker exec $KAFKA_CONTAINER kafka-topics \
    --create \
    --bootstrap-server localhost:9092 \
    --topic "$topic" \
    --partitions "$partitions" \
    --replication-factor 1 \
    --config cleanup.policy="$cleanup" \
    --config retention.ms="$retention" \
    --if-not-exists
done

echo "✅ Topics created"

echo ""
echo "Listing topics:"
docker exec $KAFKA_CONTAINER kafka-topics --list --bootstrap-server localhost:9092
