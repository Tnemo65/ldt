# Kafka Engineer Review — PLAN_v3.md

**Reviewer:** Principal Kafka Engineer (12 yrs Kafka, Confluent, Kafka Streams)
**Date:** 2026-05-12
**Files Reviewed:** `src/flink_job_complete.py`, `01-create-topics.sh`, `anomaly_producer.py`, `fast_producer.py`, `broadcast_state_loader.py`, `deployment/docker-compose.yml`, `deployment/.env`, `kafka-overview.json`

---

## 1. Kafka Source Configuration

### 1.1 Consumer Configuration Assessment

**Strengths:**
- `max.poll.interval.ms: 300000` (5 min) — appropriate for ML scoring workload
- `session.timeout.ms: 30000` — reasonable
- `heartbeat.interval.ms: 10000` — correct (1/3 of session)

**Issues:**

| Issue | Severity | Description |
|-------|----------|-------------|
| `enable.auto.commit` not set | HIGH | Must be `false` for exactly-once processing. Default is `true`. |
| `fetch.min.bytes` not set | MEDIUM | Default 1 byte causes excessive network round-trips. Set to `1024` for batching. |
| `max.poll.records` not set | MEDIUM | Default 500 records per poll. For 1000 records/sec, 500 is fine, but 1000+ needs tuning. |
| `isolation.level` not set | MEDIUM | For exactly-once, should be `read_committed` to avoid uncommitted records. |

### 1.2 Consumer Group Strategy

**CURRENT:** `group.id: cadqstream-flink` — single consumer group for all partitions.

**ISSUE (MEDIUM):** With 4 partitions and parallelism=4, each Flink source instance consumes 1 partition. Adding more Flink TMs won't increase throughput beyond 4.

**FIX:** Increase partitions and Flink parallelism to match:
```bash
kafka-topics --bootstrap-server kafka:9092 \
  --alter --topic taxi-nyc-raw --partitions 12
```

### 1.3 Auto-Offset Reset

```properties
auto.offset.reset: earliest
```

**ISSUE (MEDIUM):** `earliest` means replaying all historical data on consumer group reset. For a production pipeline, `latest` is safer. `earliest` should only be used during initial backfill.

---

## 2. Exactly-Once vs At-Least-Once

### 2.1 Kafka Sink — Exactly-Once NOT Configured — CRITICAL

`flink_job_complete.py`:
```python
return FlinkKafkaProducer(
    topic=topic,
    serialization_schema=SimpleStringSchema(),
    producer_config=props
)
```

**Missing:** `producer_semantic=FlinkKafkaProducer.SEMANTIC.EXACTLY_ONCE`.

Without exactly-once, PostgreSQL sink may receive duplicate anomaly records after failover. For fraud detection, duplicates = duplicate investigations.

### 2.2 Missing Idempotence Configuration — HIGH

No idempotence configuration on Kafka producer:
```properties
enable.idempotence=true
max.in.flight.requests.per.connection=5
retries=2147483647
acks=all
```

Without idempotence, retried produces can create duplicates.

### 2.3 Transaction Timeout — HIGH

```properties
transaction.timeout.ms=900000  # 15 min
```

Transactions must complete within this window. With checkpointing every 45s, a 15-min transaction timeout is adequate. But ensure this is consistent with Flink checkpoint interval.

### 2.4 `isolation.level` on Consumer — HIGH

Consumer must read committed transactions only:
```properties
isolation.level: read_committed
```

Without this, consumer reads uncommitted transaction data, causing inconsistency.

---

## 3. Consumer Group & Partition Strategy

### 3.1 Partition Count = 4 — Bottleneck

```properties
KAFKA_NUM_PARTITIONS: 4
```

With 4 partitions, max Flink source parallelism = 4. For 1M records/hour (~278/sec), 4 partitions × ~70 records/sec = throughput ceiling.

**FIX:** Set partitions to 12–24:
```bash
kafka-topics --bootstrap-server kafka:9092 \
  --alter --topic taxi-nyc-raw --partitions 24
kafka-topics --bootstrap-server kafka:9092 \
  --alter --topic dq-stream-anomalies --partitions 24
kafka-topics --bootstrap-server kafka:9092 \
  --alter --topic dq-stream-violations --partitions 12
```

### 3.2 Replication Factor = 1 — CRITICAL

All topics created with `replication-factor: 1`:
```bash
kafka-topics --create --topic taxi-nyc-raw \
  --bootstrap-server kafka:9092 \
  --partitions 4 --replication-factor 1
```

Broker failure = data loss and pipeline outage. NYC taxi data is critical infrastructure.

**FIX:**
```bash
kafka-topics --create --topic taxi-nyc-raw \
  --bootstrap-server kafka:9092 \
  --partitions 12 --replication-factor 3 \
  --config min.insync.replicas=2
```

### 3.3 No Sticky Partitioner

Default `DefaultPartitioner` (round-robin) used. For per-neighborhood locality, use sticky partitioning:
```properties
partitioner.class=org.apache.kafka.clients.producer.StickyPartitioner
```

Or use `wax.partitioner.class` for custom locality-aware partitioning.

---

## 4. Schema Registry

### 4.1 Schema Registry Deployed But Never Used — HIGH

Schema Registry is deployed and schemas are registered, but serialization is raw JSON:
```python
return FlinkKafkaProducer(
    serialization_schema=SimpleStringSchema(),  # NOT Avro!
    producer_config=props
)
```

Schema Registry is providing zero value.

### 4.2 No Schema Evolution Strategy

No compatibility mode configured (`GLOBAL`, `BACKWARD`, `FORWARD`, `FULL`, `NONE`). Without compatibility, schema changes break consumers.

```bash
curl -X PUT http://schema-registry:8081/config \
  -H "Content-Type: application/vnd.schemaregistry.v1+json" \
  -d '{"compatibility": "BACKWARD"}'
```

### 4.3 No Schema Validation on Producer

`anomaly_producer.py` does not validate against Schema Registry before producing. Malformed records may be produced.

---

## 5. Dead Letter Queue

### 5.1 Violation Records Sent to DLQ

**GOOD:** `ToSchemaViolationRow` function exists and logs to violation stream.

**ISSUE (MEDIUM):** DLQ records have hardcoded Kafka offsets and partitions (always 0):
```python
yield (
    record.get('trip_id', ''),     # 0 = kafka_offset (WRONG)
    0,                             # 0 = kafka_partition (WRONG)
    ...
)
```

For selective replay, DLQ records must contain the **original** offset and partition where the record failed.

### 5.2 DLQ Monitoring

No Prometheus metrics for DLQ volume. Should expose:
```python
dlq_records_total{topic, error_type, partition}
dlq_bytes_total{topic}
```

---

## 6. Throughput Optimization

### 6.1 Producer Batching Not Configured — MEDIUM

```properties
linger.ms=10  # Batch for 10ms before sending
batch.size=16384  # 16KB batch size
compression.type=lz4  # Fast compression
```

Default `linger.ms=0` sends immediately — no batching. For anomaly detection at 1000 records/sec, batching reduces network overhead by 10–50x.

### 6.2 No Compression

Default no compression. With JSON payloads, compression saves 3–10x bandwidth.

### 6.3 Consumer Fetch Size

```properties
fetch.max.bytes=52428800  # 50MB — good
fetch.min.bytes=1024        # Should be 1KB minimum
```

---

## 7. Consumer Lag & Monitoring

### 7.1 No Consumer Lag Alerts — HIGH

Prometheus alerts for Kafka lag are missing:
```yaml
- alert: KafkaConsumerLagHigh
  expr: kafka_consumergroup_lag_sum{group="cadqstream-flink"} > 100000
  for: 5m
  labels:
    severity: warning
```

### 7.2 Lag Per-Partition Monitoring

Current metric is aggregate `kafka_consumergroup_lag_sum`. Should also track per-partition lag to detect hot partitions:
```yaml
kafka_consumergroup_lag{group, topic, partition}
```

### 7.3 No Lag Burn Rate Alert

No alerting on lag growth rate (lag delta / time). A slowly growing lag won't hit threshold but will eventually overflow.

---

## 8. Kafka Connect Sink

### 8.1 JDBC Sink — Exactly-Once with PostgreSQL

**GOOD:** `FlinkKafkaProducer` with EXACTLY_ONCE writes to PostgreSQL with exactly-once semantics.

**ISSUE (MEDIUM):** The PostgreSQL sink does not use `INSERT ... ON CONFLICT DO NOTHING` or idempotent keys. With exactly-once Kafka transactions, each record is produced exactly once. The PostgreSQL sink must also handle idempotency.

### 8.2 No Dead Letter Queue for Kafka Sink

If PostgreSQL is unavailable, records are re-tried indefinitely. No DLQ for records that permanently fail.

---

## 9. Security

### 9.1 PLAINTEXT Only — CRITICAL

```properties
KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
```

No TLS, no SASL. Any pod can produce/consume any topic.

### 9.2 No ACLs

No Kafka ACLs configured. Any authenticated client can access any topic.

---

## 10. CRITICAL Issues

### CR-1: Exactly-Once NOT Configured on Kafka Sink

**File:** `flink_job_complete.py`

**What is wrong:** `FlinkKafkaProducer` without `producer_semantic=EXACTLY_ONCE`. Pipeline claims exactly-once but delivers at-least-once.

**Fix:**
```python
return FlinkKafkaProducer(
    topic=topic,
    serialization_schema=SimpleStringSchema(),
    producer_config={
        'bootstrap.servers': bootstrap_servers,
        'transaction.timeout.ms': '900000',
        'enable.idempotence': 'true',
        'max.in.flight.requests.per.connection': '5',
        'retries': '2147483647',
        'acks': 'all',
    },
    producer_semantic=FlinkKafkaProducer.SEMANTIC.EXACTLY_ONCE
)
```

### CR-2: Replication Factor = 1 for All Topics

**File:** `01-create-topics.sh`

**What is wrong:** `replication-factor: 1` everywhere. Broker failure = data loss.

**Fix:**
```bash
kafka-topics --create --topic taxi-nyc-raw \
  --bootstrap-server kafka:9092 \
  --partitions 12 --replication-factor 3 \
  --config min.insync.replicas=2 \
  --config retention.ms=604800000

kafka-topics --create --topic dq-stream-anomalies \
  --bootstrap-server kafka:9092 \
  --partitions 12 --replication-factor 3 \
  --config min.insync.replicas=2

kafka-topics --create --topic dq-stream-violations \
  --bootstrap-server kafka:9092 \
  --partitions 12 --replication-factor 3 \
  --config min.insync.replicas=2
```

### CR-3: PLAINTEXT Security Protocol

**File:** `deployment/docker-compose.yml`

**Fix — Enable TLS + SASL/SCRAM:**
```yaml
KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: SSL:SSL,SSL_HOST:SSL
KAFKA_SSL_KEYSTORE_LOCATION: /etc/kafka/secrets/kafka.keystore.jks
KAFKA_SSL_KEYSTORE_PASSWORD: ${KAFKA_SSL_KEYSTORE_PASSWORD}
KAFKA_SSL_TRUSTSTORE_LOCATION: /etc/kafka/secrets/kafka.truststore.jks
KAFKA_SSL_TRUSTSTORE_PASSWORD: ${KAFKA_SSL_TRUSTSTORE_PASSWORD}
KAFKA_SSL_CLIENT_AUTH: required
KAFKA_PRODUCER_SASL_MECHANISM: SCRAM-SHA-512
KAFKA_PRODUCER_SASL_JAAS_CONFIG: |
  org.apache.kafka.common.security.scram.ScramLoginModule required \
    username="cadqstream" \
    password="${KAFKA_PASSWORD}";
```

### CR-4: DLQ Records Have Wrong Offset/Partition (0)

**File:** `flink_job_complete.py` line 456

**What is wrong:** Returns `0` for both `kafka_offset` and `kafka_partition`. Cannot selectively replay failed records.

**Fix:** Extract from Kafka record metadata:
```python
class ToSchemaViolationRow(KeyedProcessFunction):
    def map(self, record, context):
        kafka_meta = context.get_current_key()  # Must come from Kafka source
        trip_id, pickup_dt, violation_type = record.get('trip_id', ''), \
            record.get('tpep_pickup_datetime', ''), \
            record.get('violation_type', 'unknown')
        yield (
            trip_id,
            pickup_dt,
            violation_type,
            str(record.get('raw_record', '')),  # Full original record
            float(record.get('anomaly_score', -1.0)),
            str(record.get('violation_reason', 'schema_error')),
            str(context.timestamp()),  # Event time as offset proxy
            datetime.now().isoformat(),
        )
```

---

## 11. HIGH Issues

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `flink_job_complete.py` | No idempotence config | Add `enable.idempotence=true`, `acks=all` |
| 2 | `flink_job_complete.py` | Consumer `isolation.level` not set | Add `isolation.level=read_committed` |
| 3 | `01-create-topics.sh` | `KAFKA_NUM_PARTITIONS=4` bottleneck | Increase to 12–24 partitions |
| 4 | `01-create-topics.sh` | Schema Registry unused | Switch to `JsonSchemaSerializationSchema` or Avro |
| 5 | `prometheus_alerts.yml` | No Kafka lag alerts | Add `kafka_consumergroup_lag > 100000` alert |
| 6 | `01-create-topics.sh` | No compatibility mode | Set `COMPATIBILITY: BACKWARD` on Schema Registry |

---

## 12. MEDIUM/LOW Issues

| # | File | Issue |
|---|------|-------|
| 1 | `01-create-topics.sh` | `fetch.min.bytes` not configured |
| 2 | `anomaly_producer.py` | No schema validation before producing |
| 3 | `flink_job_complete.py` | No DLQ monitoring metrics |
| 4 | `deployment/docker-compose.yml` | No Kafka ACLs |
| 5 | `01-create-topics.sh` | `auto.offset.reset=earliest` should be `latest` for production |

---

## 13. Priority Fixes

### Fix 1: Exactly-Once Kafka Sink

```python
def make_kafka_sink(topic, bootstrap_servers):
    props = {
        'bootstrap.servers': bootstrap_servers,
        'transaction.timeout.ms': '900000',
        'enable.idempotence': 'true',
        'max.in.flight.requests.per.connection': '5',
        'retries': '2147483647',
        'acks': 'all',
        'compression.type': 'lz4',
        'linger.ms': '10',
        'batch.size': '16384',
    }
    return FlinkKafkaProducer(
        topic=topic,
        serialization_schema=SimpleStringSchema(),
        producer_config=props,
        producer_semantic=FlinkKafkaProducer.SEMANTIC.EXACTLY_ONCE
    )
```

### Fix 2: Consumer with Exactly-Once

```python
consumer_props = {
    'bootstrap.servers': bootstrap_servers,
    'group.id': 'cadqstream-flink',
    'enable.auto.commit': 'false',
    'isolation.level': 'read_committed',
    'auto.offset.reset': 'earliest',  # Only for initial backfill; change to 'latest' for prod
    'fetch.min.bytes': '1024',
    'fetch.max.wait.ms': '500',
}
```

### Fix 3: Prometheus Alerts for Kafka

```yaml
- alert: KafkaConsumerLagCritical
  expr: kafka_consumergroup_lag_sum{group="cadqstream-flink"} > 100000
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Consumer lag > 100K messages"
    runbook: "https://wiki.internal/runbooks/kafka-lag"

- alert: KafkaPartitionLagHigh
  expr: kafka_consumergroup_lag{group="cadqstream-flink"} > 50000
  for: 10m
  labels:
    severity: warning
```

---

*Reviewed by: Principal Kafka Engineer | 2026-05-12*
