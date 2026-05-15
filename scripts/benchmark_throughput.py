#!/usr/bin/env python3
"""
Throughput benchmark for CA-DQStream baseline pipeline.
Spec: Lines 1705-1730 (Target: 1-5K events/sec, <500ms p99 latency)

IMPORTANT: Requires running infrastructure:
- Kafka broker(s)
- Flink job running
- MinIO

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


def create_benchmark_producer(bootstrap_servers='localhost:9092'):
    """Create high-performance Kafka producer for benchmarking."""
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks=1,
        retries=0,
        linger_ms=10,
        batch_size=32768,
        compression_type='snappy',
        max_in_flight_requests_per_connection=10
    )
    return producer


def generate_sample_record(idx: int):
    """Generate synthetic taxi record for benchmarking."""
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
    """Benchmark producer throughput."""
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

            if sent % 10 == 0:
                elapsed = time.time() - start_time
                expected = sent * interval
                if elapsed < expected:
                    time.sleep(expected - elapsed)

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
        'success': actual_rate >= target_eps * 0.95
    }

    print(f"\nProducer benchmark complete:")
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
    """Check consumer group lag."""
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            group_id=group,
            enable_auto_commit=False
        )

        partitions = consumer.partitions_for_topic(topic)
        if not partitions:
            print(f"No partitions found for topic '{topic}'")
            return None

        total_lag = 0
        for partition in partitions:
            tp = (topic, partition)
            consumer.assign([tp])
            consumer.seek_to_end(tp)
            latest_offset = consumer.position(tp)
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
        print(f"Error checking consumer lag: {e}")
        return None


def check_minio_write_rate(
    duration_sec: int = 30,
    bucket: str = 'cadqstream-anomalies'
):
    """Measure MinIO write rate via Kafka consumer throughput."""
    print(f"\n=== MinIO Write Rate (via Kafka consumer throughput) ===")
    print(f"Measurement window: {duration_sec}s")
    print(f"Note: Flink writes to MinIO via StreamingFileSink (S3A)")
    print(f"  Records should appear in MinIO bucket: {bucket}")

    try:
        consumer = KafkaConsumer(
            'dq-stream-processed',
            bootstrap_servers='localhost:9092',
            group_id='cadqstream-throughput-check',
            auto_offset_reset='latest',
            consumer_timeout_ms=duration_sec * 1000
        )

        records = []
        for message in consumer:
            records.append(message)
            if len(records) >= 10000:
                break

        consumer.close()

        rate = len(records) / duration_sec
        print(f"Ingested: {len(records)} records in {duration_sec}s")
        print(f"Rate: {rate:.0f} records/sec")
        return rate

    except Exception as e:
        print(f"Error measuring MinIO write rate: {e}")
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
        time.sleep(5)
        lag = check_consumer_lag(args.topic, bootstrap_servers=args.bootstrap_servers)

        if lag is not None and lag > args.rate * 2:
            print(f"\nWARNING: High consumer lag detected ({lag} messages)")
            print("   Flink job may not be keeping up with producer rate")

    # Summary
    print(f"\n{'='*50}")
    print("BENCHMARK SUMMARY")
    print('='*50)
    print(f"Producer: {producer_results['actual_rate']:.0f} eps (target: {args.rate} eps)")
    if producer_results['success']:
        print("PASS: Producer achieved >=95% of target rate")
    else:
        print("FAIL: Producer did not achieve 95% of target rate")

    return 0 if producer_results['success'] else 1


if __name__ == '__main__':
    exit(main())
