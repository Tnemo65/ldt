#!/usr/bin/env python3
"""
Throughput benchmark for CA-DQStream baseline pipeline.
Spec: Lines 1705-1730 (Target: 1-5K events/sec, <500ms p99 latency)

IMPORTANT: Requires running infrastructure:
- Kafka broker(s)
- Flink job running
- PostgreSQL + PgBouncer

Usage:
  python scripts/benchmark_throughput.py --rate 1000 --duration 60
  python scripts/benchmark_throughput.py --rate 5000 --duration 300
"""

import argparse
import time
import json
from datetime import datetime
from pathlib import Path
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic
import pandas as pd
import psycopg2


def create_benchmark_producer(bootstrap_servers='localhost:9092'):
    """Create high-performance Kafka producer for benchmarking."""
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks=1,  # Leader acknowledgement (faster than acks='all')
        retries=0,  # No retries for throughput test
        linger_ms=10,  # Batch for 10ms
        batch_size=32768,  # 32KB batch size
        compression_type='snappy',
        max_in_flight_requests_per_connection=10  # Higher parallelism
    )
    return producer


def generate_sample_record(idx: int):
    """Generate synthetic taxi record for benchmarking.

    Args:
        idx: Record index (for uniqueness)

    Returns:
        Record dict
    """
    now = datetime.now().isoformat()

    return {
        'VendorID': (idx % 2) + 1,
        'tpep_pickup_datetime': now,
        'tpep_dropoff_datetime': now,
        'passenger_count': (idx % 6) + 1,
        'trip_distance': 5.0 + (idx % 20),
        'PULocationID': 100 + (idx % 163),
        'DOLocationID': 100 + ((idx + 50) % 163),
        'payment_type': (idx % 4) + 1,
        'fare_amount': 15.0 + (idx % 30),
        'total_amount': 20.0 + (idx % 40)
    }


def benchmark_producer_throughput(
    target_eps: int,
    duration_sec: int,
    topic: str = 'taxi-nyc-raw',
    bootstrap_servers: str = 'localhost:9092'
):
    """Benchmark producer throughput.

    Args:
        target_eps: Target events per second
        duration_sec: Test duration in seconds
        topic: Kafka topic
        bootstrap_servers: Kafka brokers

    Returns:
        Dict with benchmark results
    """
    print(f"=== Producer Throughput Benchmark ===")
    print(f"Target: {target_eps} events/sec")
    print(f"Duration: {duration_sec}s")
    print(f"Topic: {topic}")
    print("")

    producer = create_benchmark_producer(bootstrap_servers)

    sent = 0
    start_time = time.time()
    interval = 1.0 / target_eps

    try:
        while time.time() - start_time < duration_sec:
            record = generate_sample_record(sent)
            producer.send(topic, value=record)
            sent += 1

            # Rate limiting (micro-sleep every 10 records for smoothness)
            if sent % 10 == 0:
                elapsed = time.time() - start_time
                expected = sent * interval
                if elapsed < expected:
                    time.sleep(expected - elapsed)

            # Progress update every second
            if sent % target_eps == 0:
                current_rate = sent / (time.time() - start_time)
                print(f"Sent {sent} records ({current_rate:.0f} eps)")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted")

    finally:
        producer.flush()
        producer.close()

    elapsed = time.time() - start_time
    actual_rate = sent / elapsed

    results = {
        'target_eps': target_eps,
        'duration': elapsed,
        'sent': sent,
        'actual_rate': actual_rate,
        'success': actual_rate >= target_eps * 0.95  # 95% of target
    }

    print(f"\n✓ Producer benchmark complete:")
    print(f"  Sent: {sent} records")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Actual rate: {actual_rate:.0f} eps")
    print(f"  Target: {target_eps} eps")
    print(f"  Achievement: {actual_rate/target_eps*100:.1f}%")

    return results


def check_consumer_lag(
    topic: str = 'taxi-nyc-raw',
    group: str = 'cadqstream-flink-consumer',
    bootstrap_servers: str = 'localhost:9092'
):
    """Check consumer group lag.

    Args:
        topic: Kafka topic
        group: Consumer group ID
        bootstrap_servers: Kafka brokers

    Returns:
        Total lag across all partitions
    """
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            group_id=group,
            enable_auto_commit=False
        )

        # Get partition assignments
        partitions = consumer.partitions_for_topic(topic)
        if not partitions:
            print(f"⚠️  No partitions found for topic '{topic}'")
            return None

        total_lag = 0
        for partition in partitions:
            tp = (topic, partition)

            # Get latest offset (high water mark)
            consumer.assign([tp])
            consumer.seek_to_end(tp)
            latest_offset = consumer.position(tp)

            # Get committed offset (consumer position)
            committed = consumer.committed(tp)
            if committed is None:
                committed = 0

            lag = latest_offset - committed
            total_lag += lag

        consumer.close()

        print(f"\n=== Consumer Lag Check ===")
        print(f"Topic: {topic}")
        print(f"Group: {group}")
        print(f"Partitions: {len(partitions)}")
        print(f"Total lag: {total_lag} messages")

        return total_lag

    except Exception as e:
        print(f"❌ Error checking consumer lag: {e}")
        return None


def check_postgres_ingestion_rate(
    duration_sec: int = 60,
    host: str = 'localhost',
    port: int = 5432,
    database: str = 'dq_pipeline',
    user: str = 'cadqstream',
    password: str = 'cadqstream123'
):
    """Measure PostgreSQL ingestion rate.

    Args:
        duration_sec: Measurement window in seconds

    Returns:
        Records per second ingestion rate
    """
    print(f"\n=== PostgreSQL Ingestion Rate ===")
    print(f"Measurement window: {duration_sec}s")

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        cursor = conn.cursor()

        # Count at start
        cursor.execute("SELECT COUNT(*) FROM taxi_trips_raw")
        count_start = cursor.fetchone()[0]
        print(f"Start count: {count_start}")

        # Wait
        time.sleep(duration_sec)

        # Count at end
        cursor.execute("SELECT COUNT(*) FROM taxi_trips_raw")
        count_end = cursor.fetchone()[0]
        print(f"End count: {count_end}")

        cursor.close()
        conn.close()

        ingested = count_end - count_start
        rate = ingested / duration_sec

        print(f"✓ Ingested: {ingested} records in {duration_sec}s")
        print(f"✓ Rate: {rate:.0f} records/sec")

        return rate

    except Exception as e:
        print(f"❌ Error measuring ingestion rate: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Benchmark CA-DQStream throughput')
    parser.add_argument(
        '--rate',
        type=int,
        default=1000,
        help='Target events per second (default: 1000)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Test duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--topic',
        type=str,
        default='taxi-nyc-raw',
        help='Kafka topic (default: taxi-nyc-raw)'
    )
    parser.add_argument(
        '--bootstrap-servers',
        type=str,
        default='localhost:9092',
        help='Kafka bootstrap servers (default: localhost:9092)'
    )
    parser.add_argument(
        '--check-lag',
        action='store_true',
        help='Check consumer lag after benchmark'
    )
    parser.add_argument(
        '--check-postgres',
        action='store_true',
        help='Check PostgreSQL ingestion rate'
    )

    args = parser.parse_args()

    # 1. Benchmark producer throughput
    producer_results = benchmark_producer_throughput(
        args.rate,
        args.duration,
        args.topic,
        args.bootstrap_servers
    )

    # 2. Check consumer lag (optional)
    if args.check_lag:
        time.sleep(5)  # Wait for consumers to catch up
        lag = check_consumer_lag(args.topic, bootstrap_servers=args.bootstrap_servers)

        if lag is not None and lag > args.rate * 2:
            print(f"\n⚠️  WARNING: High consumer lag detected ({lag} messages)")
            print("   Flink job may not be keeping up with producer rate")

    # 3. Check PostgreSQL ingestion (optional)
    if args.check_postgres:
        ingestion_rate = check_postgres_ingestion_rate(duration_sec=30)

        if ingestion_rate and ingestion_rate < args.rate * 0.8:
            print(f"\n⚠️  WARNING: Low ingestion rate ({ingestion_rate:.0f} < {args.rate*0.8:.0f} eps)")
            print("   PostgreSQL sink may be bottleneck")

    # Summary
    print(f"\n{'='*50}")
    print("BENCHMARK SUMMARY")
    print('='*50)
    print(f"Producer: {producer_results['actual_rate']:.0f} eps (target: {args.rate} eps)")
    if producer_results['success']:
        print("✅ PASS: Producer achieved ≥95% of target rate")
    else:
        print("❌ FAIL: Producer did not achieve 95% of target rate")

    return 0 if producer_results['success'] else 1


if __name__ == '__main__':
    exit(main())
