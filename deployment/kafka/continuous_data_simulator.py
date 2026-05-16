#!/usr/bin/env python3
"""
Continuous NYC Taxi Data Simulator — unbounded Kafka producer.

Task 2: Reads rows from a local parquet/CSV file and publishes them to Kafka
as a continuous unbounded stream, looping forever.

Requirements:
    pip install pandas kafka-python

Usage (local dev):
    python deployment/kafka/continuous_data_simulator.py \
        --input data/nyc_taxi_300k.parquet \
        --bootstrap kafka:9092 \
        --topic taxi-nyc-raw-v2 \
        --delay 0.05

Usage (Docker, via docker-compose kafka-producer service):
    Built into deployment/kafka/Dockerfile.producer
    Default entrypoint: continuous_data_simulator.py
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

LOGGER = logging.getLogger("continuous-data-simulator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


TOPIC_RAW = os.getenv("TOPIC_RAW", "taxi-nyc-raw-v2")
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
DEFAULT_DELAY = float(os.getenv("PRODUCE_DELAY", "0.05"))
DEFAULT_INPUT = os.getenv("INPUT_FILE", "data/nyc_taxi_300k.parquet")
DEFAULT_LOOP = os.getenv("LOOP_INDEFINITELY", "true").lower() == "true"


def _parse_row(record: dict) -> dict:
    """Normalize a parquet row into the canonical taxi trip dict.

    Adds trip_duration (hours) and speed_mph (miles per hour) so downstream
    operators do not need to recompute them.
    """
    pickup_str = record.get("tpep_pickup_datetime", "")
    dropoff_str = record.get("tpep_dropoff_datetime", "")

    pickup_ts = _to_timestamp(pickup_str)
    dropoff_ts = _to_timestamp(dropoff_str)

    if pickup_ts is not None and dropoff_ts is not None:
        duration_sec = (dropoff_ts - pickup_ts).total_seconds()
    else:
        duration_sec = 0.0
    duration_h = duration_sec / 3600.0

    distance = float(record.get("trip_distance") or 0.0)
    speed_mph = round(distance / duration_h, 6) if duration_h > 0 else 0.0

    return {
        "VendorID": int(record.get("VendorID") or 0),
        "tpep_pickup_datetime": _iso(pickup_ts),
        "tpep_dropoff_datetime": _iso(dropoff_ts),
        "passenger_count": float(record.get("passenger_count") or 0.0),
        "trip_distance": distance,
        "RatecodeID": float(record.get("RatecodeID") or 1.0),
        "store_and_fwd_flag": str(record.get("store_and_fwd_flag") or "N"),
        "PULocationID": float(record.get("PULocationID") or 0.0),
        "DOLocationID": float(record.get("DOLocationID") or 0.0),
        "payment_type": float(record.get("payment_type") or 0.0),
        "fare_amount": float(record.get("fare_amount") or 0.0),
        "extra": float(record.get("extra") or 0.0),
        "mta_tax": float(record.get("mta_tax") or 0.0),
        "tip_amount": float(record.get("tip_amount") or 0.0),
        "tolls_amount": float(record.get("tolls_amount") or 0.0),
        "improvement_surcharge": float(record.get("improvement_surcharge") or 0.0),
        "total_amount": float(record.get("total_amount") or 0.0),
        "congestion_surcharge": float(record.get("congestion_surcharge") or 0.0),
        "Airport_fee": float(record.get("Airport_fee") or 0.0),
        "trip_duration": duration_h,
        "speed_mph": speed_mph,
    }


def _to_timestamp(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _iso(ts):
    if ts is None:
        return ""
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def _load_records(input_path: str):
    """Load records from parquet or CSV, returning a list of normalized dicts."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    import pandas as pd

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .parquet or .csv")

    LOGGER.info("Loaded %d rows from %s", len(df), input_path)
    return [_parse_row(row) for _, row in df.iterrows()]


def _create_topic_if_not_exists(admin_client, topic_name, num_partitions=8):
    """Create Kafka topic idempotently (no-op if it already exists)."""
    existing = admin_client.list_topics(timeout=10)
    if topic_name not in {t.name for t in existing.values}:
        from kafka.admin import NewTopic

        new_topic = NewTopic(
            name=topic_name,
            num_partitions=num_partitions,
            replication_factor=1,
            topic_configs={
                "retention.ms": "604800000",
                "cleanup.policy": "delete",
            },
        )
        admin_client.create_topics([new_topic], validate_only=False)
        LOGGER.info("Created topic: %s", topic_name)
    else:
        LOGGER.info("Topic already exists: %s", topic_name)


def _build_producer(bootstrap_servers):
    from kafka import KafkaProducer
    from kafka.admin import KafkaAdminClient

    LOGGER.info("Connecting to Kafka at %s...", bootstrap_servers)

    try:
        admin = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id="continuous-data-simulator-admin",
            request_timeout_ms=30000,
        )
    except Exception as e:
        LOGGER.error("Cannot connect to Kafka admin: %s", e)
        raise

    _create_topic_if_not_exists(admin, TOPIC_RAW)
    admin.close()

    for backoff in [1, 2, 4, 8, 16]:
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                acks="all",
                retries=5,
                max_in_flight_requests_per_connection=5,
                compression_type="lz4",
                linger_ms=10,
                batch_size=16384,
                buffer_memory=67108864,
                reconnect_backoff_ms=1000,
                reconnect_backoff_max_ms=30000,
            )
            LOGGER.info("KafkaProducer connected successfully.")
            return producer
        except Exception as e:
            LOGGER.warning("KafkaProducer connection attempt failed: %s. Retrying in %ds...", e, backoff)
            time.sleep(backoff)
    raise RuntimeError("Could not connect to Kafka after multiple retries.")


def run_loop(input_path, bootstrap_servers, topic, delay, loop_indefinitely):
    records = _load_records(input_path)
    total_records = len(records)

    if total_records == 0:
        LOGGER.error("No records loaded from %s. Exiting.", input_path)
        sys.exit(1)

    producer = _build_producer(bootstrap_servers)

    loop_num = 0
    sent_total = 0
    start_time = time.time()

    try:
        while True:
            loop_num += 1
            LOGGER.info("Loop %d: publishing %d records to %s...", loop_num, total_records, topic)

            for i, record in enumerate(records):
                future = producer.send(topic, record)
                try:
                    future.get(timeout=5)
                except Exception as e:
                    LOGGER.warning("Send failed for record %d: %s", i, e)

                sent_total += 1

                if delay > 0:
                    time.sleep(delay)

                if sent_total % 5000 == 0:
                    elapsed = time.time() - start_time
                    rate = sent_total / elapsed if elapsed > 0 else 0
                    LOGGER.info(
                        "Progress: %d records sent (loop %d, rate=%.1f/sec)",
                        sent_total, loop_num, rate,
                    )

            LOGGER.info("Loop %d complete: %d records sent.", loop_num, total_records)

            if not loop_indefinitely:
                LOGGER.info("Single-pass mode complete. Exiting.")
                break

    except KeyboardInterrupt:
        LOGGER.info("Interrupted. Shutting down...")
    finally:
        producer.flush()
        producer.close()
        elapsed = time.time() - start_time
        LOGGER.info(
            "Producer stopped. Total: %d records in %.1fs (%.1f records/sec avg).",
            sent_total, elapsed, sent_total / max(elapsed, 0.1),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Continuous NYC Taxi data simulator — unbounded Kafka producer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help="Path to parquet or CSV file containing taxi records.",
    )
    parser.add_argument(
        "--bootstrap", "-b",
        default=BOOTSTRAP,
        help="Kafka bootstrap servers (comma-separated).",
    )
    parser.add_argument(
        "--topic", "-t",
        default=TOPIC_RAW,
        help="Kafka topic to publish records to.",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=DEFAULT_DELAY,
        help="Inter-record delay in seconds (0.05 = 20 records/sec per partition).",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Disable infinite looping (publish each row exactly once and exit).",
    )
    args = parser.parse_args()

    loop = not args.no_loop
    LOGGER.info(
        "Starting continuous data simulator: input=%s topic=%s delay=%.3fs loop=%s",
        args.input, args.topic, args.delay, loop,
    )

    run_loop(
        input_path=args.input,
        bootstrap_servers=args.bootstrap,
        topic=args.topic,
        delay=args.delay,
        loop_indefinitely=loop,
    )


if __name__ == "__main__":
    main()
