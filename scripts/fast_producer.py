#!/usr/bin/env python3
"""Fast Kafka producer using synthetic NYC taxi data - no pandas needed"""
import json, time, sys, random
from kafka import KafkaProducer

WORK = "/opt/flink/e2e"
OUT = WORK + "/fast_producer.log"

def log(msg):
    with open(OUT, "a") as f:
        f.write(msg + "\n")
    print(msg)

VENDOR_IDS = [1, 2]
RATECODE_IDS = [1, 2, 3, 4, 5, 6, 99]
STORE_FLAGS = ["N", "Y"]
PAYMENT_TYPES = [1, 2, 3, 4, 5, 6]
PULOCS = list(range(1, 266))
DOLOCS = list(range(1, 266))
PASSENGER_COUNTS = [1, 2, 3, 4, 5, 6]

# Base features for realistic trips
TRIP_PROFILES = [
    # (distance_range, fare_range, duration_range_min)
    ("short", (0.5, 3.0), (0.05, 0.3), (5, 30)),      # short trips
    ("medium", (2.0, 10.0), (10.0, 35.0), (15, 45)),   # medium trips
    ("long", (8.0, 30.0), (30.0, 100.0), (30, 90)),   # long trips
    ("airport", (15.0, 35.0), (45.0, 150.0), (40, 90)), # airport trips
]

def gen_trip(hour, is_weekend):
    profile = random.choice(TRIP_PROFILES)
    distance = round(random.uniform(*profile[1][0]), 2)
    duration_h = random.uniform(*profile[2][0])
    fare = round(random.uniform(*profile[1][1]), 2)
    speed = distance / duration_h if duration_h > 0 else 0

    extra = round(random.uniform(0, 3.5), 2)
    mta_tax = 0.5
    tip = round(fare * random.uniform(0, 0.25), 2) if random.random() > 0.3 else 0
    tolls = round(random.uniform(0, 20), 2) if random.random() > 0.9 else 0
    imp_surcharge = 1.0
    cong_surcharge = 2.5
    airport_fee = 2.5 if profile[0] == "airport" else 0.0
    total = round(fare + extra + mta_tax + tip + tolls + imp_surcharge + cong_surcharge + airport_fee, 2)

    pu_hour = hour
    pu_min = random.randint(0, 59)
    pu_sec = random.randint(0, 59)
    do_min = int(pu_min + duration_h * 60)
    do_hour = pu_hour + (pu_min + int(duration_h * 60)) // 60
    do_hour = do_hour % 24

    pickup_dt = f"2024-01-{(hour // 24) + 1:02d}-{(hour % 24):02d}:{pu_min:02d}:{pu_sec:02d}"
    dropoff_dt = f"2024-01-{(do_hour // 24) + (hour // 24) + 1:02d}-{do_hour % 24:02d}:{do_min % 60:02d}:{pu_sec:02d}"

    return {
        "VendorID": random.choice(VENDOR_IDS),
        "tpep_pickup_datetime": pickup_dt,
        "tpep_dropoff_datetime": dropoff_dt,
        "passenger_count": float(random.choice(PASSENGER_COUNTS)),
        "trip_distance": distance,
        "RatecodeID": random.choice(RATECODE_IDS),
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
        "trip_duration": duration_h,
        "speed_mph": round(speed, 6),
    }

try:
    producer = KafkaProducer(
        bootstrap_servers=["ldt-kafka-1:9092"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        retries=3,
        batch_size=65536,
        linger_ms=10,
        buffer_memory=134217728,
        compression_type="gzip",
    )
except:
    producer = KafkaProducer(
        bootstrap_servers=["localhost:9092"],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        retries=3,
    )

log(f"[{time.strftime('%H:%M:%S')}] Producer starting...")
total = int(sys.argv[1])
log(f"[{time.strftime('%H:%M:%S')}] Sending {total} as fast as possible...")
sent = 0
start = time.time()
# Pre-generate hours for realistic timestamps
hours = [i % 24 for i in range(total // 100 + 1)]
for i in range(total):
    trip = gen_trip(hours[i % len(hours)], random.random() > 0.7)
    producer.send("taxi-nyc-raw", trip)
    sent += 1
    if sent % 50000 == 0:
        elapsed = time.time() - start
        log(f"[{time.strftime('%H:%M:%S')}] Sent {sent}/{total} ({sent/elapsed:.0f}/sec)")

producer.flush()
producer.close()
elapsed = time.time() - start
log(f"[{time.strftime('%H:%M:%S')}] DONE: {sent} sent, {elapsed:.1f}s, {sent/elapsed:.0f}/sec avg")
