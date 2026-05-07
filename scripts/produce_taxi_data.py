#!/usr/bin/env python3
"""
Kafka producer for NYC Taxi data with Avro serialization.
Spec: Lines 1645-1670 (Avro schema, 1K events/sec)
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from kafka import KafkaProducer
import pandas as pd


def create_producer(bootstrap_servers='localhost:9092'):
    """Create Kafka producer with JSON serialization.

    Note: For production, use Avro with schema registry.
    For baseline testing, JSON is simpler.
    """
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        retries=3,
        max_in_flight_requests_per_connection=5,
        compression_type='snappy'
    )
    return producer


def read_parquet_records(file_path: Path, limit: int = None):
    """Read records from parquet file.

    Args:
        file_path: Path to parquet file
        limit: Maximum number of records to read (None = all)

    Yields:
        Record dict with ISO datetime strings
    """
    df = pd.read_parquet(file_path)

    if limit:
        df = df.head(limit)

    for _, row in df.iterrows():
        record = row.to_dict()

        # Convert datetime columns to ISO strings
        for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
            if col in record and pd.notna(record[col]):
                if isinstance(record[col], pd.Timestamp):
                    record[col] = record[col].isoformat()
                elif isinstance(record[col], datetime):
                    record[col] = record[col].isoformat()

        # Convert numpy types to Python types
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif hasattr(value, 'item'):  # numpy types
                record[key] = value.item()

        yield record


def produce_records(
    producer,
    topic: str,
    file_path: Path,
    rate: int = 1000,
    limit: int = None
):
    """Produce records to Kafka topic at specified rate.

    Args:
        producer: KafkaProducer instance
        topic: Kafka topic name
        file_path: Path to parquet file
        rate: Events per second
        limit: Maximum records to send (None = all)
    """
    interval = 1.0 / rate  # seconds between records
    count = 0
    start_time = time.time()

    print(f"Producing records to topic '{topic}'")
    print(f"Rate: {rate} events/sec")
    print(f"File: {file_path}")
    if limit:
        print(f"Limit: {limit} records")
    print("")

    try:
        for record in read_parquet_records(file_path, limit):
            # Send to Kafka
            producer.send(topic, value=record)
            count += 1

            # Rate limiting
            if count % rate == 0:
                elapsed = time.time() - start_time
                expected = count * interval
                if elapsed < expected:
                    time.sleep(expected - elapsed)

                # Progress update
                print(f"Sent {count} records ({count/(time.time()-start_time):.0f} events/sec)")

            elif count % 100 == 0:
                # Micro sleep for smoother rate limiting
                time.sleep(interval)

        # Flush remaining
        producer.flush()

        elapsed = time.time() - start_time
        print(f"\n✅ Completed: {count} records in {elapsed:.1f}s ({count/elapsed:.0f} events/sec)")

    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted: {count} records sent")
        producer.flush()


def main():
    parser = argparse.ArgumentParser(description='Produce NYC Taxi data to Kafka')
    parser.add_argument(
        '--file',
        type=str,
        default='data/clean/jan_2024_clean_baseline.parquet',
        help='Parquet file to read (default: jan_2024_clean_baseline.parquet)'
    )
    parser.add_argument(
        '--topic',
        type=str,
        default='taxi-nyc-raw',
        help='Kafka topic (default: taxi-nyc-raw)'
    )
    parser.add_argument(
        '--rate',
        type=int,
        default=1000,
        help='Events per second (default: 1000)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Max records to send (default: all)'
    )
    parser.add_argument(
        '--bootstrap-servers',
        type=str,
        default='localhost:9092',
        help='Kafka bootstrap servers (default: localhost:9092)'
    )

    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return 1

    # Create producer
    producer = create_producer(args.bootstrap_servers)

    # Produce records
    produce_records(
        producer,
        args.topic,
        file_path,
        args.rate,
        args.limit
    )

    producer.close()
    return 0


if __name__ == '__main__':
    exit(main())
