#!/bin/bash
# Create Kafka topics - Laptop version (reduced partitions)

docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create \
  --topic taxi-nyc-raw \
  --partitions 2 \
  --replication-factor 1 \
  --config retention.ms=3600000

docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create \
  --topic dq-schema-violations \
  --partitions 1 \
  --replication-factor 1

docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create \
  --topic dq-hard-rule-violations \
  --partitions 1 \
  --replication-factor 1

docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create \
  --topic dq-anomaly-scores \
  --partitions 1 \
  --replication-factor 1

docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create \
  --topic dq-meta-stream \
  --partitions 1 \
  --replication-factor 1

docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create \
  --topic if-model-updates \
  --partitions 1 \
  --replication-factor 1 \
  --config cleanup.policy=compact

docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create \
  --topic iec-action-replay \
  --partitions 1 \
  --replication-factor 1

echo "✅ Topics created (laptop version - 1-2 partitions)"
