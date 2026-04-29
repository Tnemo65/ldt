import requests
import psycopg2

print("=" * 50)
print("CA-DQStream Docker Stack Verification")
print("=" * 50)

# Prometheus
try:
    r = requests.get('http://localhost:9090/api/v1/status/config', timeout=5)
    print(f"[OK] Prometheus API: {r.status_code}")
except Exception as e:
    print(f"[FAIL] Prometheus: {e}")

# Grafana
try:
    r2 = requests.get('http://localhost:3000/api/health', timeout=5)
    print(f"[OK] Grafana: {r2.status_code}")
except Exception as e:
    print(f"[FAIL] Grafana: {e}")

# Kafka (via broker list)
try:
    from kafka import KafkaConsumer
    consumer = KafkaConsumer(
        bootstrap_servers=['localhost:9092'],
        client_id='verify-script',
        consumer_timeout_ms=2000
    )
    print(f"[OK] Kafka: connected, topics = {consumer.topics()}")
    consumer.close()
except Exception as e:
    print(f"[WARN] Kafka consumer: {e}")

# PostgreSQL
try:
    conn = psycopg2.connect(
        host='localhost', port='5432',
        user='taxi_dq_user', password='taxi_dq_pass',
        database='taxi_dq'
    )
    cur = conn.cursor()

    cur.execute("SELECT count(*) FROM taxi_clean")
    count_clean = cur.fetchone()[0]
    print(f"[OK] PostgreSQL taxi_clean: {count_clean:,} rows")

    cur.execute("SELECT count(*) FROM ml_anomalies")
    count_anomalies = cur.fetchone()[0]
    print(f"[OK] PostgreSQL ml_anomalies: {count_anomalies:,} rows")

    cur.execute("SELECT count(*) FROM quality_metrics")
    count_metrics = cur.fetchone()[0]
    print(f"[OK] PostgreSQL quality_metrics: {count_metrics:,} rows")

    cur.execute("SELECT count(*) FROM rule_violations")
    count_rules = cur.fetchone()[0]
    print(f"[OK] PostgreSQL rule_violations: {count_rules:,} rows")

    conn.close()
except Exception as e:
    print(f"[FAIL] PostgreSQL: {e}")

# Kafka topic check
try:
    from kafka import KafkaProducer
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        client_id='verify-script',
        acks=1,
        max_block_ms=5000
    )
    topic = 'taxi-trips'
    metadata = producer.listTopics(timeout=5)
    print(f"[OK] Kafka topics: {list(metadata.keys())}")
    producer.close()
except Exception as e:
    print(f"[WARN] Kafka topic check: {e}")

print("=" * 50)
print("Stack verification complete.")
