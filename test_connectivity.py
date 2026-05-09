#!/usr/bin/env python3
"""Test Kafka connectivity from Flink container."""
import os
import sys

# Test Kafka connectivity
print("=" * 60)
print("CA-DQStream E2E - Service Connectivity Test")
print("=" * 60)

# Test 1: Kafka
print("\n[1] Testing Kafka connectivity...")
try:
    from kafka import KafkaProducer
    # Try kafka:9092 first (Docker network), then localhost:9092
    for server in ['kafka:9092', 'ldt-kafka:9092', 'localhost:9092']:
        try:
            p = KafkaProducer(bootstrap_servers=server, request_timeout_ms=5000)
            print(f"  Kafka OK: {server}")
            p.close()
            kafka_server = server
            break
        except Exception as e:
            print(f"  Kafka {server}: {e}")
            kafka_server = None
except Exception as e:
    print(f"  Kafka import error: {e}")
    kafka_server = None

# Test 2: PostgreSQL
print("\n[2] Testing PostgreSQL connectivity...")
try:
    import psycopg2
    for host in ['postgres', 'ldt-postgres', 'localhost']:
        try:
            conn = psycopg2.connect(
                host=host, port=5432, dbname='dq_pipeline',
                user='cadqstream', password='cadqstream123', connect_timeout=5
            )
            cur = conn.cursor()
            cur.execute("SELECT version()")
            print(f"  PostgreSQL OK: {host}")
            print(f"  Version: {cur.fetchone()[0][:60]}...")
            conn.close()
            pg_host = host
            break
        except Exception as e:
            print(f"  PostgreSQL {host}: {e}")
            pg_host = None
except Exception as e:
    print(f"  PostgreSQL import error: {e}")
    pg_host = None

# Test 3: ML Models
print("\n[3] Testing ML models...")
import os
WORK = "/opt/flink/e2e"
for model_path in [
    f"{WORK}/models/iforest_model.pkl",
    f"{WORK}/models/scaler.pkl",
    f"{WORK}/models/context_thresholds.json",
    f"{WORK}/models/neighborhood_mapping.json"
]:
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"  {os.path.basename(model_path)}: {size:,} bytes")
    else:
        print(f"  {os.path.basename(model_path)}: NOT FOUND")

# Test 4: Source code
print("\n[4] Testing source code imports...")
src_modules = [
    "operators.key_generator",
    "operators.deduplicator",
    "operators.schema_validator",
    "operators.canary_rules",
    "operators.if_scoring_operator",
    "operators.rendezvous_operator",
    "operators.meta_aggregator",
    "operators.iec_operator",
    "sinks.postgres_sink",
]
sys.path.insert(0, "/opt/flink/e2e")
sys.path.insert(0, "/opt/flink/e2e/src")
for mod in src_modules:
    try:
        m = __import__(mod, fromlist=[""])
        print(f"  {mod}: OK")
    except Exception as e:
        print(f"  {mod}: ERROR - {e}")

print("\n" + "=" * 60)
print("Connectivity Test Complete")
print("=" * 60)
