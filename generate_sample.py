#!/usr/bin/env python3
"""Generate a small sample of NYC taxi data for quick Kafka producer testing."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
output = sys.argv[2] if len(sys.argv) > 2 else "data/taxi_sample_10k.parquet"

np.random.seed(42)
random.seed(42)

zones = list(range(1, 264))
base_date = datetime(2024, 1, 1, 0, 0, 0)

records = []
for i in range(N):
    pickup = base_date + timedelta(minutes=i * 5 + random.randint(-2, 2))
    trip_duration_min = random.randint(3, 60)
    dropoff = pickup + timedelta(minutes=trip_duration_min)
    distance = round(random.uniform(0.5, 25.0), 1)
    fare = round(random.uniform(5.0, 75.0) + distance * 2.5, 2)
    tip = round(random.uniform(0.0, fare * 0.2), 2) if random.random() > 0.3 else 0.0
    tolls = round(random.uniform(0.0, 15.0), 2) if random.random() > 0.85 else 0.0

    records.append({
        "VendorID": random.choice([1, 2]),
        "tpep_pickup_datetime": pickup.strftime("%Y-%m-%d %H:%M:%S"),
        "tpep_dropoff_datetime": dropoff.strftime("%Y-%m-%d %H:%M:%S"),
        "passenger_count": float(random.randint(1, 4)),
        "trip_distance": float(distance),
        "RatecodeID": 1.0,
        "store_and_fwd_flag": "N",
        "PULocationID": float(random.choice(zones)),
        "DOLocationID": float(random.choice(zones)),
        "payment_type": float(random.choice([1, 2, 3])),
        "fare_amount": float(fare),
        "extra": round(random.uniform(0.0, 2.5), 2),
        "mta_tax": 0.5,
        "tip_amount": float(tip),
        "tolls_amount": float(tolls),
        "improvement_surcharge": 0.3,
        "total_amount": float(round(fare + 0.5 + 0.3 + tip + tolls, 2)),
    })

df = pd.DataFrame(records)
df.to_parquet(output, index=False)
print(f"Generated {N} rows -> {output}")
