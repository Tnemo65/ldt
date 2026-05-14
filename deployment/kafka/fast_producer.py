#!/usr/bin/env python3
"""Continuous Kafka producer using synthetic NYC taxi data."""
import json
import time
import sys
import random
import logging
from kafka import KafkaProducer

LOGGER = logging.getLogger('cadqstream-producer')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

VENDOR_IDS = [1, 2]
RATECODE_IDS = [1, 2, 3, 4, 5, 6, 99]
STORE_FLAGS = ["N", "Y"]
PAYMENT_TYPES = [1, 2, 3, 4, 5, 6]
PULOCS = list(range(1, 266))
DOLOCS = list(range(1, 266))
PASSENGER_COUNTS = [1, 2, 3, 4, 5, 6]

# name, d_min, d_max, f_min, f_max, du_min, du_max
TRIP_PROFILES = [
    ("short",   0.5,  3.0,   5.0,  20.0,   300, 1800),
    ("medium",  2.0, 10.0,  15.0,  60.0,   900, 3600),
    ("long",    8.0, 30.0,  40.0, 200.0,  1800, 5400),
    ("airport", 15.0, 35.0,  50.0, 150.0,  2400, 5400),
]

def gen_trip(hour):
    p = random.choice(TRIP_PROFILES)
    name = p[0]
    d_min = p[1]
    d_max = p[2]
    f_min = p[3]
    f_max = p[4]
    du_min = p[5]
    du_max = p[6]

    distance = round(random.uniform(d_min, d_max), 2)
    fare = round(random.uniform(f_min, f_max), 2)
    dur_sec = random.uniform(du_min, du_max)

    extra = round(random.uniform(0, 3.5), 2)
    mta_tax = 0.5
    tip = round(fare * random.uniform(0, 0.25), 2) if random.random() > 0.3 else 0.0
    tolls = round(random.uniform(0, 20), 2) if random.random() > 0.9 else 0.0
    imp_surcharge = 1.0
    cong_surcharge = 2.5
    airport_fee = 2.5 if name == "airport" else 0.0
    total = round(fare + extra + mta_tax + tip + tolls + imp_surcharge + cong_surcharge + airport_fee, 2)

    dur_h = dur_sec / 3600.0
    pu_hour = hour % 24
    pu_min = random.randint(0, 59)
    pu_sec = random.randint(0, 59)
    do_min_total = int(pu_min + dur_sec / 60.0)
    do_hour = (pu_hour + do_min_total // 60) % 24
    do_min = do_min_total % 60
    day = (hour // 24) + 1

    return {
        "VendorID": random.choice(VENDOR_IDS),
        "tpep_pickup_datetime": "2024-01-%02dT%02d:%02d:%02d" % (day, pu_hour, pu_min, pu_sec),
        "tpep_dropoff_datetime": "2024-01-%02dT%02d:%02d:%02d" % (day, do_hour, do_min, pu_sec),
        "passenger_count": float(random.choice(PASSENGER_COUNTS)),
        "trip_distance": distance,
        "RatecodeID": float(random.choice(RATECODE_IDS)),
        "store_and_fwd_flag": random.choice(STORE_FLAGS),
        "PULocationID": float(random.choice(PULOCS)),
        "DOLocationID": float(random.choice(DOLOCS)),
        "payment_type": float(random.choice(PAYMENT_TYPES)),
        "fare_amount": fare,
        "extra": extra,
        "mta_tax": mta_tax,
        "tip_amount": tip,
        "tolls_amount": tolls,
        "improvement_surcharge": imp_surcharge,
        "total_amount": total,
        "congestion_surcharge": cong_surcharge,
        "Airport_fee": airport_fee,
        "trip_duration": dur_h,
        "speed_mph": round(distance / dur_h, 6) if dur_h > 0 else 0.0,
    }

if __name__ == "__main__":
    rate = int(sys.argv[1]) if len(sys.argv) > 1 else 100  # messages per second
    bootstrap = sys.argv[2] if len(sys.argv) > 2 else "kafka:9092"
    topic = sys.argv[3] if len(sys.argv) > 3 else "taxi-nyc-raw"

    LOGGER.info("Connecting to Kafka at %s...", bootstrap)
    try:
        producer = KafkaProducer(
            bootstrap_servers=[bootstrap],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
            max_in_flight_requests_per_connection=5,
            compression_type="gzip",
            linger_ms=5,
            batch_size=65536,
        )
    except Exception as e:
        LOGGER.error("Error connecting: %s", e)
        sys.exit(1)

    interval = 1.0 / rate if rate > 0 else 0.01
    LOGGER.info("Starting continuous producer at ~%d msg/sec to topic '%s'...", rate, topic)
    
    hour = 0
    sent = 0
    start = time.time()
    last_log = start
    
    try:
        while True:
            trip = gen_trip(hour)
            producer.send(topic, trip)
            sent += 1
            
            # Log every 10 seconds
            now = time.time()
            if now - last_log >= 10:
                elapsed = now - start
                actual_rate = sent / elapsed
                LOGGER.info("Sent %d messages total, avg rate: %.0f/sec", sent, actual_rate)
                last_log = now
            
            hour += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        LOGGER.info("Shutting down...")
    finally:
        producer.flush()
        producer.close()
        elapsed = time.time() - start
        LOGGER.info("DONE: %d messages sent in %.1fs (%.0f/sec avg)", sent, elapsed, sent / max(elapsed, 0.1))
