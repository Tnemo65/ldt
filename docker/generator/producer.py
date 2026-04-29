# Kafka Data Generator for CA-DQStream
# Produces synthetic NYC taxi records to Kafka topic "taxi-trips"

import json
import random
import time
import os
import threading
from datetime import datetime, timedelta
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'taxi-trips')
EMIT_RATE = int(os.getenv('EMIT_RATE', '10000'))  # records per second

VENDOR_IDS = [1, 2]
RATECODE_IDS = [1, 2, 3, 4, 5, 6, 99]
PAYMENT_TYPES = [1, 2, 3, 4, 5, 6]
PULOCATION_IDS = list(range(1, 264))   # NYC taxi zones
DOLOCATION_IDS = list(range(1, 264))

def generate_record(base_time=None):
    if base_time is None:
        base_time = datetime.now()

    vendor = random.choice(VENDOR_IDS)
    pickup = base_time - timedelta(seconds=random.randint(60, 3600))
    dropoff = pickup + timedelta(seconds=random.randint(180, 7200))

    trip_distance = max(0.0, random.gauss(3.5, 4.0))
    duration_sec = random.randint(180, 7200)
    fare_amount = max(2.5, random.gauss(15.0, 12.0))

    ratecode = random.choice(RATECODE_IDS)
    if ratecode == 2:  # JFK
        fare_amount = random.uniform(52, 70)
        trip_distance = random.uniform(17, 25)
    elif ratecode == 3:  # Newark
        fare_amount = random.uniform(55, 80)
        trip_distance = random.uniform(18, 28)
    elif ratecode == 4:  # Negotiated
        fare_amount = random.uniform(20, 200)

    passenger = random.choice([1, 1, 1, 2, 2, 3, 4])
    extra = random.choice([0, 0, 0, 2.5, 3.0])
    mta_tax = 0.5
    tip_amount = 0.0
    tolls = 0.0
    imp_surcharge = 1.0
    congestion_surcharge = random.choice([0, 2.5])
    airport_fee = random.choice([0, 1.25, 2.5]) if ratecode in [2, 3] else 0

    total = fare_amount + extra + mta_tax + tip_amount + tolls + imp_surcharge + congestion_surcharge + airport_fee

    payment = random.choice([1, 1, 1, 1, 2, 3, 4])
    if payment == 1:  # Credit card
        tip_amount = round(random.uniform(0, fare_amount * 0.3), 2)
        total += tip_amount

    return {
        'VendorID': vendor,
        'tpep_pickup_datetime': pickup.isoformat(),
        'tpep_dropoff_datetime': dropoff.isoformat(),
        'passenger_count': passenger,
        'trip_distance': round(trip_distance, 2),
        'RatecodeID': ratecode,
        'store_and_fwd_flag': random.choice(['N', 'N', 'N', 'Y']),
        'PULocationID': random.choice(PULOCATION_IDS),
        'DOLocationID': random.choice(DOLOCATION_IDS),
        'payment_type': payment,
        'fare_amount': round(fare_amount, 2),
        'extra': round(extra, 2),
        'mta_tax': round(mta_tax, 2),
        'tip_amount': round(tip_amount, 2),
        'tolls_amount': round(tolls, 2),
        'improvement_surcharge': round(imp_surcharge, 2),
        'total_amount': round(total, 2),
        'congestion_surcharge': round(congestion_surcharge, 2),
        'Airport_fee': round(airport_fee, 2),
    }

def generate_batch(batch_size, base_time=None):
    return [generate_record(base_time) for _ in range(batch_size)]

class Producer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=5,
            max_in_flight_requests_per_connection=5,
            compression_type='snappy',
            linger_ms=10,
            batch_size=32768,
        )
        self.total_sent = 0
        self.running = True
        log.info(f'Connected to Kafka at {KAFKA_BOOTSTRAP}, topic: {KAFKA_TOPIC}')
        log.info(f'Emit rate: {EMIT_RATE:,} records/sec')

    def send_batch(self, batch):
        futures = []
        for record in batch:
            future = self.producer.send(KAFKA_TOPIC, record)
            futures.append(future)
        try:
            self.producer.flush(timeout=5)
        except KafkaError as e:
            log.error(f'Flush error: {e}')
        self.total_sent += len(batch)

    def run(self):
        interval = 1.0 / (EMIT_RATE / 1000)  # seconds per 1000 records
        batch_size = 1000

        while self.running:
            t0 = time.time()
            batch = generate_batch(batch_size)
            self.send_batch(batch)
            elapsed = time.time() - t0
            sleep_time = max(0, (interval * batch_size / 1000) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            if self.total_sent % 100000 < batch_size:
                log.info(f'Progress: {self.total_sent:,} records sent to {KAFKA_TOPIC}')

    def stop(self):
        self.running = False
        self.producer.close()
        log.info(f'Stopped. Total sent: {self.total_sent:,}')

if __name__ == '__main__':
    producer = Producer()
    try:
        producer.run()
    except KeyboardInterrupt:
        producer.stop()
