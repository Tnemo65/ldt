#!/usr/bin/env python3
"""Continuous Kafka producer for testing."""
import json, time, random
from kafka import KafkaProducer

def create_producer():
    return KafkaProducer(
        bootstrap_servers='kafka:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        retries=3,
    )

def generate_trip():
    return {
        "VendorID": random.choice([1, 2]),
        "tpep_pickup_datetime": "2024-01-01 12:00:00",
        "tpep_dropoff_datetime": "2024-01-01 12:15:00",
        "passenger_count": float(random.randint(1, 4)),
        "trip_distance": round(random.uniform(0.5, 10.0), 2),
        "RatecodeID": 1.0,
        "store_and_fwd_flag": "N",
        "PULocationID": random.randint(1, 263),
        "DOLocationID": random.randint(1, 263),
        "payment_type": random.choice([1, 2, 3, 4]),
        "fare_amount": round(random.uniform(5, 30), 2),
        "extra": 1.0,
        "mta_tax": 0.5,
        "tip_amount": round(random.uniform(0, 8), 2),
        "tolls_amount": 0.0,
        "improvement_surcharge": 1.0,
        "total_amount": round(random.uniform(10, 50), 2),
        "congestion_surcharge": 2.5,
        "Airport_fee": 0.0,
    }

producer = create_producer()
print("Producer started, sending messages continuously...")
count = 0
try:
    while True:
        for i in range(100):
            trip = generate_trip()
            producer.send('taxi-nyc-raw', trip)
        producer.flush()
        count += 100
        print(f"Sent {count} messages total...")
        time.sleep(2)
except KeyboardInterrupt:
    producer.flush()
    producer.close()
    print(f"Done! Sent {count} messages total.")
