#!/usr/bin/env python3
"""Simple Kafka producer for testing."""
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
        "passenger_count": random.randint(1, 4),
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
    }

producer = create_producer()
print("Producer created, sending messages...")
for i in range(100):
    trip = generate_trip()
    future = producer.send('taxi-nyc-raw', trip)
    if i % 20 == 0:
        print(f"Sent {i} messages...")
producer.flush()
producer.close()
print("Done! Sent 100 messages.")
