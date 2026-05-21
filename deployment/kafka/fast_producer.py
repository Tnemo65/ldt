#!/usr/bin/env python3
"""Fast Kafka producer: reads parquet, sends JSON to taxi-nyc-raw-v2, async flush."""

import sys
import json
import time
import os
import signal

sys.path.insert(0, r'c:\proj\ldt\.venv\Lib\site-packages')
sys.path.insert(0, r'c:\proj\ldt\deployment')

import pandas as pd
from kafka import KafkaProducer

BOOTSTRAP = '172.18.0.11:9092'
TOPIC = os.getenv('TOPIC_RAW', 'taxi-nyc-raw-v2')
INPUT = r'c:\proj\ldt\deployment\data\demo_trips.parquet'
DELAY = float(os.getenv('PRODUCE_DELAY', '0.0'))
FLUSH_EVERY = 5000

def parse_row(row):
    pickup_str = row.get("tpep_pickup_datetime", "")
    dropoff_str = row.get("tpep_dropoff_datetime", "")
    pickup_ts = None
    dropoff_ts = None
    if pickup_str:
        try:
            pickup_ts = pd.Timestamp(pickup_str)
        except Exception:
            pass
    if dropoff_str:
        try:
            dropoff_ts = pd.Timestamp(dropoff_str)
        except Exception:
            pass
    duration_h = 0.0
    if pickup_ts and dropoff_ts:
        duration_sec = (dropoff_ts - pickup_ts).total_seconds()
        duration_h = duration_sec / 3600.0
    distance = float(row.get("trip_distance") or 0.0)
    speed_mph = round(distance / duration_h, 6) if duration_h > 0 else 0.0
    return {
        "VendorID": int(row.get("VendorID") or 0),
        "tpep_pickup_datetime": pickup_ts.strftime("%Y-%m-%dT%H:%M:%S") if pickup_ts else "",
        "tpep_dropoff_datetime": dropoff_ts.strftime("%Y-%m-%dT%H:%M:%S") if dropoff_ts else "",
        "passenger_count": float(row.get("passenger_count") or 0.0),
        "trip_distance": distance,
        "RatecodeID": float(row.get("RatecodeID") or 1.0),
        "store_and_fwd_flag": str(row.get("store_and_fwd_flag") or "N"),
        "PULocationID": float(row.get("PULocationID") or 0.0),
        "DOLocationID": float(row.get("DOLocationID") or 0.0),
        "payment_type": float(row.get("payment_type") or 0.0),
        "fare_amount": float(row.get("fare_amount") or 0.0),
        "extra": float(row.get("extra") or 0.0),
        "mta_tax": float(row.get("mta_tax") or 0.0),
        "tip_amount": float(row.get("tip_amount") or 0.0),
        "tolls_amount": float(row.get("tolls_amount") or 0.0),
        "improvement_surcharge": float(row.get("improvement_surcharge") or 0.0),
        "total_amount": float(row.get("total_amount") or 0.0),
        "congestion_surcharge": float(row.get("congestion_surcharge") or 0.0),
        "Airport_fee": float(row.get("Airport_fee") or 0.0),
        "trip_duration": duration_h,
        "speed_mph": speed_mph,
    }

print(f"Loading {INPUT}...", flush=True)
df = pd.read_parquet(INPUT)
total = len(df)
print(f"Loaded {total} rows", flush=True)

# Connect via Docker host network - find Kafka IP
producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
    acks=1,  # leader acks only - faster
    retries=3,
    max_in_flight_requests_per_connection=5,
    compression_type='lz4',
    linger_ms=50,
    batch_size=65536,
    buffer_memory=134217728,
)
print("Producer connected", flush=True)

sent = 0
start = time.time()
for i, row in df.iterrows():
    record = parse_row(row)
    future = producer.send(TOPIC, record)
    sent += 1
    if DELAY > 0:
        time.sleep(DELAY)
    if sent % FLUSH_EVERY == 0:
        producer.flush()
        elapsed = time.time() - start
        rate = sent / elapsed if elapsed > 0 else 0
        print(f"Sent {sent}/{total} ({sent/total*100:.1f}%) - rate={rate:.0f}/sec", flush=True)

producer.flush()
elapsed = time.time() - start
print(f"ALL DONE: {sent} records in {elapsed:.1f}s ({sent/elapsed:.0f}/sec avg)", flush=True)
producer.close()
