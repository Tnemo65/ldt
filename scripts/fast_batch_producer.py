#!/usr/bin/env python3
"""
Fast batch producer: reads parquet in chunks and sends to Kafka.
Uses kafka-python-ng for Python 3.12+ compatibility.
Run inside Docker network: docker run --rm --network=cadqstream-net -v c:/proj/ldt/deployment/data:/data ldt-kafka-producer python3 /tmp/fast_batch_producer.py
"""
import sys
import time
import json
import argparse
from datetime import datetime

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

def parse_row(record):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/demo_trips.parquet")
    parser.add_argument("--bootstrap", default="kafka:9092")
    parser.add_argument("--topic", default="taxi-nyc-raw-v2")
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()

    try:
        from kafka import KafkaProducer
    except ImportError:
        from kafka.kafka import KafkaProducer

    print(f"Loading {args.input}...")
    import pandas as pd
    df = pd.read_parquet(args.input)
    total = len(df)
    print(f"Loaded {len(df)} rows, sending to {args.topic} at {args.bootstrap}...")

    producer = KafkaProducer(
        bootstrap_servers=[args.bootstrap],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks=1,
        compression_type='gzip',
        linger_ms=50,
        batch_size=131072,
        buffer_memory=67108864,
        max_in_flight_requests_per_connection=10,
    )

    start = time.time()
    count = 0
    for idx, row in df.iterrows():
        record = parse_row(row.to_dict())
        producer.send(args.topic, value=record)
        count += 1
        if (idx + 1) % args.batch_size == 0:
            producer.flush()
            elapsed = time.time() - start
            rate = count / elapsed
            remaining = total - count
            eta = remaining / rate if rate > 0 else 0
            print(f"  Sent {count}/{total} ({100*count/total:.1f}%) | {rate:.0f} rec/s | ETA {eta:.0f}s")

    producer.flush()
    producer.close()
    elapsed = time.time() - start
    rate = count / elapsed if elapsed > 0 else 0
    print(f"DONE: Sent {count} records in {elapsed:.1f}s ({rate:.0f} rec/s)")

if __name__ == "__main__":
    main()
