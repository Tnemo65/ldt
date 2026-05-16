#!/usr/bin/env python3
"""
Anomaly Simulation Producer for CA-DQStream.

Injects synthetic anomalies into Kafka topics to simulate real-world data quality
issues. This populates the dq-hard-rule-violations topic, triggers canary rule
detections, fires Prometheus alerts, and populates MinIO output buckets.

Anomaly Types:
  INVALID_ZONE        PULocationID/DOLocationID outside 1-263
  NEGATIVE_FARE      fare_amount < 0
  ZERO_PASSENGER     passenger_count == 0
  IMPOSSIBLE_SPEED    fare_amount/trip_distance > 200 mph implied
  MISSING_FIELD       Required field null
  DRIFT_SPIKE         10x normal fare for burst window
  SEASONAL_SHIFT      Gradual fare increase over 30 min

Usage:
  python anomaly_producer.py [--bootstrap KAFKA_BOOTSTRAP] [--mode MODE]
  Modes: continuous (default), burst, drift_inject

Environment Variables:
  KAFKA_BOOTSTRAP_SERVERS  Default: kafka:9092
  ANOMALY_RATE             Base anomaly injection rate (0.0-1.0), default: 0.08
  ENABLE_ALERTS            Whether to emit alert-promoting violations, default: true
"""

import json
import time
import random
import argparse
import threading
import os
from datetime import datetime, timedelta
from collections import deque
from kafka import KafkaProducer
from kafka.errors import KafkaError


# ─── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
BASE_ANOMALY_RATE = float(os.getenv('ANOMALY_RATE', '0.08'))
# FIX: Bounded rate limiting to prevent unbounded accumulation and consumer lag
# Set to 0 for unlimited (as-fast-as-possible). Recommended: 500-1000 records/sec.
PRODUCE_RATE_LIMIT = int(os.getenv('KAFKA_PRODUCE_RATE_LIMIT', '0'))  # 0 = unlimited
PRODUCE_SLEEP_INTERVAL = 1.0 / float(os.getenv('KAFKA_PRODUCE_INTERVAL', '1.0')) if PRODUCE_RATE_LIMIT > 0 else 0.0

NYC_ZONE_MIN, NYC_ZONE_MAX = 1, 263

PAYMENT_TYPES = [1, 2, 3, 4, 5, 6]
TRIP_PROFILES = [
    ("short",   0.5,  3.0,   5.0,  20.0,   300, 1800),
    ("medium",  2.0, 10.0,  15.0,  60.0,   900, 3600),
    ("long",    8.0, 30.0,  40.0, 200.0,  1800, 5400),
    ("airport", 15.0, 35.0,  50.0, 150.0,  2400, 5400),
]


# ─── Anomaly Generators ────────────────────────────────────────────────────────

def gen_base_trip(counter=0, elapsed_seconds=0):
    """Generate a normal NYC taxi trip record."""
    profile = random.choice(TRIP_PROFILES)
    name, d_min, d_max, f_min, f_max, du_min, du_max = profile

    distance = round(random.uniform(d_min, d_max), 2)
    fare = round(random.uniform(f_min, f_max), 2)
    dur_sec = random.uniform(du_min, du_max)
    extra = round(random.uniform(0, 3.5), 2)
    mta_tax = 0.5
    tip = round(fare * random.uniform(0, 0.25), 2) if random.random() > 0.3 else 0.0
    tolls = round(random.uniform(0, 20), 2) if random.random() > 0.9 else 0.0
    total = round(fare + extra + mta_tax + tip + tolls + 1.0 + 2.5, 2)
    dur_h = dur_sec / 3600.0

    base_date = datetime(2024, 1, 1) + timedelta(seconds=elapsed_seconds)
    pickup_str = base_date.strftime("%Y-%m-%dT%H:%M:%S")
    dropoff_date = base_date + timedelta(seconds=int(dur_sec))
    dropoff_str = dropoff_date.strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "VendorID": random.choice([1, 2]),
        "tpep_pickup_datetime": pickup_str,
        "tpep_dropoff_datetime": dropoff_str,
        "passenger_count": float(random.choice([1, 2, 3, 4, 5, 6])),
        "trip_distance": distance,
        "RatecodeID": 1.0,
        "store_and_fwd_flag": "N",
        "PULocationID": float(random.randint(1, NYC_ZONE_MAX)),
        "DOLocationID": float(random.randint(1, NYC_ZONE_MAX)),
        "payment_type": float(random.choice(PAYMENT_TYPES)),
        "fare_amount": fare,
        "extra": extra,
        "mta_tax": mta_tax,
        "tip_amount": tip,
        "tolls_amount": tolls,
        "improvement_surcharge": 1.0,
        "total_amount": total,
        "congestion_surcharge": 2.5,
        "trip_duration": dur_h,
        "speed_mph": round(distance / dur_h, 6) if dur_h > 0 else 0.0,
        "_trip_seq": counter,
    }


def inject_invalid_zone(trip):
    """PULocationID or DOLocationID outside 1-263."""
    trip = dict(trip)
    if random.random() < 0.5:
        trip['PULocationID'] = float(random.randint(300, 500))
    else:
        trip['DOLocationID'] = float(random.randint(300, 500))
    trip['_anomaly_type'] = 'INVALID_ZONE'
    trip['_anomaly_severity'] = 'warning'
    return trip


def inject_negative_fare(trip):
    """Negative or zero fare amount."""
    trip = dict(trip)
    trip['fare_amount'] = round(random.uniform(-50.0, -0.01), 2)
    trip['total_amount'] = round(trip['fare_amount'] + random.uniform(0, 5), 2)
    trip['_anomaly_type'] = 'NEGATIVE_FARE'
    trip['_anomaly_severity'] = 'critical'
    return trip


def inject_zero_passenger(trip):
    """Passenger count set to zero."""
    trip = dict(trip)
    trip['passenger_count'] = 0.0
    trip['_anomaly_type'] = 'ZERO_PASSENGER'
    trip['_anomaly_severity'] = 'warning'
    return trip


def inject_impossible_speed(trip):
    """Fare implies speed > 200 mph."""
    trip = dict(trip)
    trip['trip_distance'] = 0.5
    trip['fare_amount'] = round(random.uniform(200.0, 500.0), 2)
    trip['_anomaly_type'] = 'IMPOSSIBLE_SPEED'
    trip['_anomaly_severity'] = 'critical'
    return trip


def inject_missing_field(trip):
    """Null/missing required field."""
    trip = dict(trip)
    field = random.choice(['fare_amount', 'trip_distance', 'PULocationID', 'DOLocationID'])
    trip[field] = None
    trip['_anomaly_type'] = 'MISSING_FIELD'
    trip['_anomaly_severity'] = 'warning'
    return trip


def inject_extreme_fare(trip):
    """Fare > $1000."""
    trip = dict(trip)
    trip['fare_amount'] = round(random.uniform(1001.0, 5000.0), 2)
    trip['total_amount'] = trip['fare_amount'] + 5.0
    trip['_anomaly_type'] = 'EXTREME_FARE'
    trip['_anomaly_severity'] = 'warning'
    return trip


def inject_zero_distance_with_fare(trip):
    """Zero distance with positive fare."""
    trip = dict(trip)
    trip['trip_distance'] = 0.0
    trip['fare_amount'] = round(random.uniform(10.0, 50.0), 2)
    trip['total_amount'] = trip['fare_amount'] + 5.0
    trip['_anomaly_type'] = 'ZERO_DISTANCE_WITH_FARE'
    trip['_anomaly_severity'] = 'warning'
    return trip


# Registry of injectors: (weight, function)
# Weights sum to 1.0; a random draw selects which anomaly type to inject.
ANOMALY_INJECTORS = [
    (0.20, inject_invalid_zone),
    (0.15, inject_negative_fare),
    (0.15, inject_zero_passenger),
    (0.15, inject_impossible_speed),
    (0.10, inject_missing_field),
    (0.10, inject_extreme_fare),
    (0.15, inject_zero_distance_with_fare),
]


def choose_anomaly():
    """Choose an anomaly injector based on weights."""
    r = random.random()
    cumulative = 0.0
    for weight, injector in ANOMALY_INJECTORS:
        cumulative += weight
        if r < cumulative:
            return injector
    return ANOMALY_INJECTORS[0][1]


# ─── Metrics Tracking ──────────────────────────────────────────────────────────

class AnomalyMetrics:
    """Track metrics published as log lines for Prometheus scraping."""

    def __init__(self):
        self.total_sent = 0
        self.normal_sent = 0
        self.anomalies_sent = 0
        self.by_type = {inj.__name__: 0 for _, inj in ANOMALY_INJECTORS}
        self._metrics = {
            'anomaly_producer_records_total': 0,
            'anomaly_producer_records_normal': 0,
            'anomaly_producer_records_anomaly': 0,
            'anomaly_producer_records_drift': 0,
        }

    def emit(self):
        """Emit metrics as structured log lines for Prometheus scrape."""
        ts = datetime.utcnow().isoformat()
        print(f"[metrics][{ts}] anomaly_producer_records_total {self._metrics['anomaly_producer_records_total']}")
        print(f"[metrics][{ts}] anomaly_producer_records_normal {self._metrics['anomaly_producer_records_normal']}")
        print(f"[metrics][{ts}] anomaly_producer_records_anomaly {self._metrics['anomaly_producer_records_anomaly']}")
        print(f"[metrics][{ts}] anomaly_producer_records_drift {self._metrics['anomaly_producer_records_drift']}")
        for atype, count in self.by_type.items():
            print(f"[metrics][{ts}] anomaly_producer_type{{type=\"{atype}\"}} {count}")

    def record_normal(self):
        self.total_sent += 1
        self.normal_sent += 1
        self._metrics['anomaly_producer_records_total'] += 1
        self._metrics['anomaly_producer_records_normal'] += 1

    def record_anomaly(self, injector):
        self.total_sent += 1
        self.anomalies_sent += 1
        self._metrics['anomaly_producer_records_total'] += 1
        self._metrics['anomaly_producer_records_anomaly'] += 1
        name = injector.__name__
        self.by_type[name] = self.by_type.get(name, 0) + 1

    def record_drift(self):
        self._metrics['anomaly_producer_records_drift'] += 1


# ─── Kafka Producer ────────────────────────────────────────────────────────────

def create_producer(bootstrap):
    """Create a Kafka producer with gzip compression."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=[bootstrap],
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            acks='all',
            retries=3,
            max_in_flight_requests_per_connection=5,
            compression_type='lz4',
            linger_ms=5,
            batch_size=65536,
        )
        return producer
    except Exception as e:
        print(f"[producer] Failed to connect to Kafka at {bootstrap}: {e}")
        raise


# ─── Mode: Continuous ─────────────────────────────────────────────────────────

def run_continuous(bootstrap, anomaly_rate, run_forever=True, max_records=None,
                    raw_topic=os.getenv('TOPIC_RAW', 'taxi-nyc-raw-v2'), violation_topic='dq-hard-rule-violations'):
    """Send records continuously, injecting anomalies at anomaly_rate."""
    print(f"[producer] Starting continuous mode (rate={anomaly_rate:.1%})")
    producer = create_producer(bootstrap)
    metrics = AnomalyMetrics()
    counter = [0]

    def emit_metrics_periodically():
        while True:
            time.sleep(30)
            metrics.emit()

    thread = threading.Thread(target=emit_metrics_periodically, daemon=True)
    thread.start()

    start = time.time()
    last_report = start

    while run_forever or (max_records and counter[0] < max_records):
        counter[0] += 1
        if max_records and counter[0] > max_records:
            break

        trip = gen_base_trip(counter=counter[0], elapsed_seconds=counter[0] / 10.0)
        trip['_producer_ts'] = datetime.utcnow().isoformat()
        trip['_producer_hour'] = counter[0] // 100

        if random.random() < anomaly_rate:
            injector = choose_anomaly()
            trip = injector(trip)
            metrics.record_anomaly(injector)
            # Send to violation topic (Flink reads this as dq-hard-rule-violations)
            try:
                producer.send(violation_topic, trip)
            except KafkaError:
                pass
        else:
            metrics.record_normal()

        # Always send to raw topic so Flink pipeline processes it
        try:
            producer.send(raw_topic, trip)
        except KafkaError:
            pass

        if counter[0] % 5000 == 0:
            now = time.time()
            elapsed = now - last_report
            rate = 5000 / elapsed if elapsed > 0 else 0
            print(f"[producer] Sent {counter[0]:,} records ({rate:.0f}/sec), "
                  f"anomalies: {metrics.anomalies_sent:,} ({metrics.anomalies_sent/max(1,counter[0]):.1%})")
            last_report = now

        # FIX: Bounded rate limiting to prevent unbounded Kafka accumulation
        if PRODUCE_RATE_LIMIT > 0:
            time.sleep(PRODUCE_SLEEP_INTERVAL)

    producer.flush()
    producer.close()
    metrics.emit()
    elapsed = time.time() - start
    print(f"[producer] DONE: {counter[0]:,} records in {elapsed:.1f}s "
          f"({counter[0]/elapsed:.0f}/sec avg), anomalies: {metrics.anomalies_sent:,}")


# ─── Mode: Burst ───────────────────────────────────────────────────────────────

def run_burst(bootstrap, burst_count=200, normal_before=500, normal_after=500,
              raw_topic=os.getenv('TOPIC_RAW', 'taxi-nyc-raw-v2'), violation_topic='dq-hard-rule-violations'):
    """Send normal records, then a burst of anomalies, then more normal records."""
    print(f"[producer] Starting burst mode: {normal_before} normal, {burst_count} anomaly burst, {normal_after} normal")
    producer = create_producer(bootstrap)
    metrics = AnomalyMetrics()

    for i in range(normal_before):
        trip = gen_base_trip(counter=i, elapsed_seconds=i / 10.0)
        trip['_producer_ts'] = datetime.utcnow().isoformat()
        metrics.record_normal()
        producer.send(raw_topic, trip)

    print(f"[producer] Sending {burst_count} anomaly burst...")
    burst_start = time.time()
    for i in range(burst_count):
        trip = gen_base_trip(counter=normal_before + i, elapsed_seconds=(normal_before + i) / 10.0)
        trip['_producer_ts'] = datetime.utcnow().isoformat()
        injector = choose_anomaly()
        trip = injector(trip)
        metrics.record_anomaly(injector)
        producer.send(raw_topic, trip)
        producer.send(violation_topic, trip)

    burst_elapsed = time.time() - burst_start
    print(f"[producer] Burst complete: {burst_count} anomalies in {burst_elapsed:.1f}s")

    for i in range(normal_after):
        trip = gen_base_trip(counter=normal_before + burst_count + i, elapsed_seconds=(normal_before + burst_count + i) / 10.0)
        trip['_producer_ts'] = datetime.utcnow().isoformat()
        metrics.record_normal()
        producer.send(raw_topic, trip)

    producer.flush()
    producer.close()
    metrics.emit()
    print(f"[producer] Burst mode complete. Total: {normal_before + burst_count + normal_after}")


# ─── Mode: Drift Inject ───────────────────────────────────────────────────────

def run_drift_inject(bootstrap, base_fare=15.0, drift_multiplier=10.0,
                     drift_start_record=1000, drift_duration=300,
                     raw_topic=os.getenv('TOPIC_RAW', 'taxi-nyc-raw-v2')):
    """Inject a drift spike: 10x fare increase for a window of records."""
    print(f"[producer] Starting drift inject mode: base_fare={base_fare}, "
          f"drift_start={drift_start_record}, drift_multiplier={drift_multiplier}x")
    producer = create_producer(bootstrap)
    metrics = AnomalyMetrics()
    metrics.record_drift()

    for i in range(drift_start_record):
        trip = gen_base_trip(counter=i, elapsed_seconds=i / 10.0)
        trip['_producer_ts'] = datetime.utcnow().isoformat()
        metrics.record_normal()
        producer.send(raw_topic, trip)

    print(f"[producer] DRIFT SPIKE START at record {drift_start_record}")
    for i in range(drift_duration):
        trip = gen_base_trip(counter=drift_start_record + i, elapsed_seconds=(drift_start_record + i) / 10.0)
        trip['_producer_ts'] = datetime.utcnow().isoformat()
        # Multiply fare by drift_multiplier
        trip['fare_amount'] = round(trip['fare_amount'] * drift_multiplier, 2)
        trip['total_amount'] = round(trip['total_amount'] * drift_multiplier, 2)
        trip['_anomaly_type'] = 'DRIFT_SPIKE'
        trip['_anomaly_severity'] = 'critical'
        trip['_drift_inject'] = True
        trip['_drift_multiplier'] = drift_multiplier
        metrics.record_anomaly(inject_extreme_fare)
        producer.send(raw_topic, trip)
        if i % 50 == 0:
            print(f"[producer] Drift spike: {i}/{drift_duration} records injected "
                  f"(fare ~${trip['fare_amount']:.2f})")

    print(f"[producer] DRIFT SPIKE END at record {drift_start_record + drift_duration}")

    for i in range(100):
        trip = gen_base_trip(counter=drift_start_record + drift_duration + i, elapsed_seconds=(drift_start_record + drift_duration + i) / 10.0)
        trip['_producer_ts'] = datetime.utcnow().isoformat()
        trip['_post_drift'] = True
        metrics.record_normal()
        producer.send(raw_topic, trip)

    producer.flush()
    producer.close()
    metrics.emit()
    print("[producer] Drift inject complete.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CA-DQStream Anomaly Simulation Producer')
    parser.add_argument('--bootstrap', default=DEFAULT_BOOTSTRAP,
                        help=f'Kafka bootstrap servers (default: {DEFAULT_BOOTSTRAP})')
    parser.add_argument('--mode', choices=['continuous', 'burst', 'drift_inject'],
                        default='continuous', help='Production mode')
    parser.add_argument('--rate', type=float, default=BASE_ANOMALY_RATE,
                        help=f'Base anomaly injection rate 0.0-1.0 (default: {BASE_ANOMALY_RATE})')
    parser.add_argument('--records', type=int, default=None,
                        help='Max records for continuous mode (default: unlimited)')
    parser.add_argument('--raw-topic', default=os.getenv('TOPIC_RAW', 'taxi-nyc-raw-v2'),
                        help=f'Raw data topic (default: taxi-nyc-raw-v2)')
    parser.add_argument('--violation-topic', default='dq-hard-rule-violations',
                        help='Violation topic (default: dq-hard-rule-violations)')

    args = parser.parse_args()

    if PRODUCE_RATE_LIMIT > 0:
        print(f"[producer] Rate limiting: max {PRODUCE_RATE_LIMIT} records/sec (KAFKA_PRODUCE_RATE_LIMIT set)")
    else:
        print(f"[producer] Rate limiting: DISABLED (unlimited, KAFKA_PRODUCE_RATE_LIMIT=0 or unset)")

    print(f"[producer] Bootstrap: {args.bootstrap}")
    print(f"[producer] Mode: {args.mode}")

    if args.mode == 'continuous':
        run_continuous(
            bootstrap=args.bootstrap,
            anomaly_rate=args.rate,
            max_records=args.records,
            raw_topic=args.raw_topic,
            violation_topic=args.violation_topic,
        )
    elif args.mode == 'burst':
        run_burst(
            bootstrap=args.bootstrap,
            raw_topic=args.raw_topic,
            violation_topic=args.violation_topic,
        )
    elif args.mode == 'drift_inject':
        run_drift_inject(
            bootstrap=args.bootstrap,
            raw_topic=args.raw_topic,
        )


if __name__ == '__main__':
    main()
