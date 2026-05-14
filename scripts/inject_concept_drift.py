"""
CA-DQStream: Full Concept Drift Injection & Testing Suite
===========================================================
Script này inject tất cả các loại concept drift vào Kafka để test
toàn bộ pipeline từ ingest -> L1 -> L2 -> L3 -> L4 (IEC).

Các loại drift được test:
1. ABRUPT_DRIFT    - Thay đổi đột ngột (ví dụ: fare tăng 10x)
2. GRADUAL_DRIFT   - Thay đổi từ từ theo thời gian (seasonal shift)
3. TRANSIENT_DRIFT - Spike ngắn rồi biến mất (burst window)
4. RECURRING_DRIFT - Drift xuất hiện theo chu kỳ
5. FEATURE_DRIFT   - Chỉ một số feature thay đổi
6. LABEL_DRIFT     - Distribution của nhãn thay đổi (anomaly rate tăng)
7. DISTRIBUTION_SHIFT - Toàn bộ phân phối dữ liệu thay đổi

Usage:
    python scripts/inject_concept_drift.py --drift-type ABRUPT_DRIFT --duration 300
    python scripts/inject_concept_drift.py --drift-type ALL --run-benchmark

Environment:
    KAFKA_BOOTSTRAP_SERVERS=kafka:9092 (hoặc localhost:9092)
"""

import json
import time
import random
import argparse
import threading
import os
import sys
import math
from datetime import datetime, timedelta
from collections import deque
from kafka import KafkaProducer
from kafka.errors import KafkaError

# ─── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
RAW_TOPIC = 'taxi-nyc-raw'

NYC_ZONE_MIN, NYC_ZONE_MAX = 1, 263
PAYMENT_TYPES = [1, 2, 3, 4, 5, 6]

TRIP_PROFILES = [
    ("short",   0.5,  3.0,   5.0,  20.0,   300, 1800),
    ("medium",  2.0, 10.0,  15.0,  60.0,   900, 3600),
    ("long",    8.0, 30.0,  40.0, 200.0,  1800, 5400),
    ("airport", 15.0, 35.0,  50.0, 150.0,  2400, 5400),
]

# ─── Drift Type Definitions ────────────────────────────────────────────────────

DRIFT_TYPES = {
    'ABRUPT_DRIFT': {
        'description': 'Thay đổi đột ngột trong distribution (ví dụ: fare tăng 10x)',
        'detection_expectation': 'ADWIN phát hiện ngay trong 1-2 window',
        'severity': 'HIGH',
        'metrics_affected': ['anomaly_rate', 'avg_anomaly_score', 'violation_rate'],
        'multiplier': 10.0,
        'duration_type': 'fixed'
    },
    'GRADUAL_DRIFT': {
        'description': 'Thay đổi từ từ theo thời gian (seasonal pricing shift)',
        'detection_expectation': 'ADWIN phát hiện sau nhiều window, delta_score tăng dần',
        'severity': 'MODERATE',
        'metrics_affected': ['anomaly_rate', 'delta_score', 'violation_rate'],
        'multiplier': 1.0,  # Sẽ được điều chỉnh từ từ
        'duration_type': 'gradual'
    },
    'TRANSIENT_DRIFT': {
        'description': 'Spike ngắn rồi biến mất (burst anomaly window)',
        'detection_expectation': 'ADWIN phát hiện rồi recovery, có thể miss nếu quá nhanh',
        'severity': 'LOW-MODERATE',
        'metrics_affected': ['anomaly_rate', 'volume'],
        'multiplier': 5.0,
        'duration_type': 'burst'
    },
    'RECURRING_DRIFT': {
        'description': 'Drift xuất hiện theo chu kỳ (ví dụ: rush hour mỗi ngày)',
        'detection_expectation': 'ADWIN quen dần nhưng delta_score vẫn tăng',
        'severity': 'LOW',
        'metrics_affected': ['volume', 'violation_rate'],
        'multiplier': 3.0,
        'duration_type': 'cyclic'
    },
    'FEATURE_DRIFT': {
        'description': 'Chỉ một số feature thay đổi (ví dụ: chỉ fare tăng)',
        'detection_expectation': 'Phụ thuộc vào feature importance trong model',
        'severity': 'MODERATE-HIGH',
        'metrics_affected': ['anomaly_rate', 'avg_anomaly_score'],
        'multiplier': 1.0,
        'duration_type': 'targeted'
    },
    'LABEL_DRIFT': {
        'description': 'Anomaly rate tăng đột ngột (nhiều fraud hơn)',
        'detection_expectation': 'ADWIN phát hiện qua anomaly_rate metric',
        'severity': 'HIGH',
        'metrics_affected': ['anomaly_rate', 'violation_rate'],
        'multiplier': 1.0,
        'duration_type': 'fraud_injection'
    },
    'DISTRIBUTION_SHIFT': {
        'description': 'Toàn bộ phân phối dữ liệu thay đổi (ví dụ: airport traffic)',
        'detection_expectation': 'ADWIN phát hiện trên nhiều metrics',
        'severity': 'HIGH',
        'metrics_affected': ['volume', 'null_rate', 'violation_rate', 'anomaly_rate'],
        'multiplier': 1.0,
        'duration_type': 'global'
    }
}

# ─── Trip Generator ────────────────────────────────────────────────────────────

def gen_base_trip(hour=12, counter=0, neighborhood_bias=None):
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

    pu_hour = hour % 24
    pu_min = random.randint(0, 59)
    pu_sec = random.randint(0, 59)
    do_min_total = int(pu_min + dur_sec / 60.0)
    do_hour = (pu_hour + do_min_total // 60) % 24
    do_min = do_min_total % 60
    day = (hour // 24) + 1

    # Neighborhood bias for distribution shift
    if neighborhood_bias == 'manhattan':
        pu_loc = float(random.randint(1, 50))
        do_loc = float(random.randint(1, 50))
    elif neighborhood_bias == 'airport':
        pu_loc = float(random.choice([132, 138]))  # JFK, LGA
        do_loc = float(random.randint(1, 263))
    else:
        pu_loc = float(random.randint(1, NYC_ZONE_MAX))
        do_loc = float(random.randint(1, NYC_ZONE_MAX))

    return {
        "VendorID": random.choice([1, 2]),
        "tpep_pickup_datetime": f"2024-01-{day:02d}T{pu_hour:02d}:{pu_min:02d}:{pu_sec:02d}",
        "tpep_dropoff_datetime": f"2024-01-{day:02d}T{do_hour:02d}:{do_min:02d}:{pu_sec:02d}",
        "passenger_count": float(random.choice([1, 2, 3, 4, 5, 6])),
        "trip_distance": distance,
        "RatecodeID": 1.0,
        "store_and_fwd_flag": "N",
        "PULocationID": pu_loc,
        "DOLocationID": do_loc,
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


# ─── Drift Injectors ───────────────────────────────────────────────────────────

class DriftInjector:
    """Base class for drift injection strategies."""

    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier
        self.start_time = None

    def apply(self, trip, progress=0.0):
        """Apply drift to trip. progress = 0.0 to 1.0 within drift window."""
        raise NotImplementedError

    def reset(self):
        self.start_time = None


class AbruptDriftInjector(DriftInjector):
    """Sudden change: multiplier applied immediately at start."""

    def apply(self, trip, progress=0.0):
        trip = dict(trip)
        trip['fare_amount'] = round(trip['fare_amount'] * self.multiplier, 2)
        trip['total_amount'] = round(trip['total_amount'] * self.multiplier, 2)
        trip['_drift_type'] = 'ABRUPT_DRIFT'
        trip['_drift_severity'] = 'HIGH'
        return trip


class GradualDriftInjector(DriftInjector):
    """Gradual change: multiplier increases over time."""

    def apply(self, trip, progress=0.0):
        trip = dict(trip)
        # Linear increase from 1.0 to multiplier over duration
        current_mult = 1.0 + (self.multiplier - 1.0) * min(progress, 1.0)
        trip['fare_amount'] = round(trip['fare_amount'] * current_mult, 2)
        trip['total_amount'] = round(trip['total_amount'] * current_mult, 2)
        trip['_drift_type'] = 'GRADUAL_DRIFT'
        trip['_drift_severity'] = 'MODERATE'
        trip['_drift_progress'] = round(progress, 3)
        return trip


class TransientDriftInjector(DriftInjector):
    """Short burst: spike then disappear."""

    def apply(self, trip, progress=0.0):
        trip = dict(trip)
        # Spike in middle of window, fade at edges
        if 0.2 <= progress <= 0.8:
            trip['fare_amount'] = round(trip['fare_amount'] * self.multiplier, 2)
            trip['total_amount'] = round(trip['total_amount'] * self.multiplier, 2)
            trip['_drift_type'] = 'TRANSIENT_DRIFT'
            trip['_drift_severity'] = 'MODERATE'
        else:
            trip['_drift_type'] = 'TRANSIENT_DRIFT'
            trip['_drift_severity'] = 'NORMAL'
        return trip


class FeatureDriftInjector(DriftInjector):
    """Only specific features change (e.g., fare_per_mile ratio)."""

    def apply(self, trip, progress=0.0):
        trip = dict(trip)
        # Only change fare, not distance (creates ratio drift)
        trip['fare_amount'] = round(trip['fare_amount'] * self.multiplier, 2)
        trip['_drift_type'] = 'FEATURE_DRIFT'
        trip['_drift_severity'] = 'MODERATE-HIGH'
        return trip


class LabelDriftInjector(DriftInjector):
    """Inject more fraud/anomalies into canary-clean records."""

    def __init__(self, multiplier=1.0, fraud_rate=0.5):
        super().__init__(multiplier)
        self.fraud_rate = fraud_rate

    def apply(self, trip, progress=0.0):
        trip = dict(trip)
        # Inject realistic fraud: short trip with inflated fare
        if random.random() < self.fraud_rate:
            trip['trip_distance'] = round(random.uniform(0.3, 1.0), 2)
            trip['fare_amount'] = round(random.uniform(40.0, 80.0), 2)
            trip['total_amount'] = trip['fare_amount'] + 5.0
            trip['_drift_type'] = 'LABEL_DRIFT'
            trip['_drift_severity'] = 'HIGH'
        else:
            trip['_drift_type'] = 'NORMAL'
        return trip


class DistributionShiftInjector(DriftInjector):
    """Entire data distribution shifts (e.g., airport traffic changes)."""

    def __init__(self, multiplier=1.0, target_neighborhood='airport'):
        super().__init__(multiplier)
        self.target_neighborhood = target_neighborhood

    def apply(self, trip, progress=0.0):
        trip = dict(trip)

        if self.target_neighborhood == 'airport':
            # Shift traffic to airport zones
            trip['PULocationID'] = float(random.choice([132, 138]))
            trip['fare_amount'] = round(trip['fare_amount'] * 2.0, 2)  # Airport premium
            trip['trip_distance'] = round(random.uniform(15.0, 35.0), 2)
            trip['RatecodeID'] = 2.0  # JFK flat fare
            trip['_drift_type'] = 'DISTRIBUTION_SHIFT'
            trip['_drift_severity'] = 'HIGH'
            trip['_neighborhood'] = 'airport'

        return trip


# ─── Kafka Producer ────────────────────────────────────────────────────────────

def create_producer(bootstrap):
    """Create a Kafka producer."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=[bootstrap],
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            acks='all',
            retries=3,
            compression_type='gzip',
            linger_ms=5,
            batch_size=65536,
        )
        return producer
    except Exception as e:
        print(f"[producer] Failed to connect to Kafka at {bootstrap}: {e}")
        raise


# ─── Drift Runners ─────────────────────────────────────────────────────────────

def run_abrupt_drift(bootstrap, duration=300, pre_records=500, post_records=100, multiplier=10.0):
    """Run abrupt drift test: normal -> 10x fare -> normal."""
    print(f"\n{'='*60}")
    print(f"DRIFT TYPE: ABRUPT_DRIFT")
    print(f"{'='*60}")
    print(f"  Multiplier: {multiplier}x")
    print(f"  Pre-drift records: {pre_records}")
    print(f"  Drift duration: {duration} records")
    print(f"  Post-drift records: {post_records}")

    producer = create_producer(bootstrap)
    injector = AbruptDriftInjector(multiplier=multiplier)

    # Pre-drift: normal records
    print(f"\n[Phase 1] Sending {pre_records} normal records...")
    for i in range(pre_records):
        trip = gen_base_trip(hour=(i // 100) % 24, counter=i)
        trip['_phase'] = 'pre_drift'
        producer.send(RAW_TOPIC, trip)
        if i % 100 == 0:
            print(f"  Sent {i}/{pre_records}")

    # Drift: abrupt change
    print(f"\n[Phase 2] DRIFT START - Sending {duration} drifted records...")
    for i in range(duration):
        trip = gen_base_trip(hour=((pre_records + i) // 100) % 24, counter=pre_records + i)
        trip = injector.apply(trip, progress=0.5)  # Full drift
        trip['_phase'] = 'drift'
        trip['_record_in_drift'] = i
        producer.send(RAW_TOPIC, trip)
        if i % 50 == 0:
            print(f"  Drift record {i}/{duration} (fare ~${trip['fare_amount']:.2f})")

    # Post-drift: return to normal
    print(f"\n[Phase 3] DRIFT END - Sending {post_records} normal records...")
    for i in range(post_records):
        trip = gen_base_trip(hour=((pre_records + duration + i) // 100) % 24,
                            counter=pre_records + duration + i)
        trip['_phase'] = 'post_drift'
        producer.send(RAW_TOPIC, trip)

    producer.flush()
    producer.close()
    print(f"\n[COMPLETE] Abrupt drift test done!")


def run_gradual_drift(bootstrap, duration=600, pre_records=500, post_records=100, multiplier=2.5):
    """Run gradual drift test: fare increases slowly over time."""
    print(f"\n{'='*60}")
    print(f"DRIFT TYPE: GRADUAL_DRIFT")
    print(f"{'='*60}")
    print(f"  Final multiplier: {multiplier}x")
    print(f"  Duration: {duration} records (gradual increase)")

    producer = create_producer(bootstrap)
    injector = GradualDriftInjector(multiplier=multiplier)

    # Pre-drift
    print(f"\n[Phase 1] Sending {pre_records} normal records...")
    for i in range(pre_records):
        trip = gen_base_trip(hour=(i // 100) % 24, counter=i)
        trip['_phase'] = 'pre_drift'
        producer.send(RAW_TOPIC, trip)

    # Gradual drift: increase fare over time
    print(f"\n[Phase 2] DRIFT START - Gradual increase over {duration} records...")
    for i in range(duration):
        progress = i / duration  # 0.0 to 1.0
        trip = gen_base_trip(hour=((pre_records + i) // 100) % 24,
                            counter=pre_records + i)
        trip = injector.apply(trip, progress=progress)
        trip['_phase'] = 'drift'
        trip['_record_in_drift'] = i
        trip['_drift_progress'] = round(progress, 3)
        producer.send(RAW_TOPIC, trip)
        if i % 100 == 0:
            print(f"  Progress {int(progress*100)}% - fare ~${trip['fare_amount']:.2f}")

    # Post-drift: maintain new level
    print(f"\n[Phase 3] DRIFT SUSTAINED - Sending {post_records} records at new level...")
    for i in range(post_records):
        trip = gen_base_trip(hour=((pre_records + duration + i) // 100) % 24,
                            counter=pre_records + duration + i)
        trip = injector.apply(trip, progress=1.0)
        trip['_phase'] = 'post_drift'
        producer.send(RAW_TOPIC, trip)

    producer.flush()
    producer.close()
    print(f"\n[COMPLETE] Gradual drift test done!")


def run_transient_drift(bootstrap, duration=200, pre_records=500, post_records=100, multiplier=5.0):
    """Run transient drift test: short spike in middle."""
    print(f"\n{'='*60}")
    print(f"DRIFT TYPE: TRANSIENT_DRIFT (Burst)")
    print(f"{'='*60}")
    print(f"  Multiplier: {multiplier}x")
    print(f"  Spike duration: {duration} records (20%-80% is spike)")

    producer = create_producer(bootstrap)
    injector = TransientDriftInjector(multiplier=multiplier)

    # Pre-drift
    print(f"\n[Phase 1] Sending {pre_records} normal records...")
    for i in range(pre_records):
        trip = gen_base_trip(hour=(i // 100) % 24, counter=i)
        trip['_phase'] = 'pre_drift'
        producer.send(RAW_TOPIC, trip)

    # Transient drift: spike in middle
    print(f"\n[Phase 2] DRIFT START - Transient spike over {duration} records...")
    for i in range(duration):
        progress = i / duration
        trip = gen_base_trip(hour=((pre_records + i) // 100) % 24,
                            counter=pre_records + i)
        trip = injector.apply(trip, progress=progress)
        trip['_phase'] = 'drift'
        trip['_record_in_drift'] = i
        producer.send(RAW_TOPIC, trip)
        if i % 40 == 0:
            phase = 'NORMAL' if trip.get('_drift_severity') == 'NORMAL' else 'SPIKE'
            print(f"  Record {i}/{duration} - Phase: {phase}")

    # Post-drift: back to normal
    print(f"\n[Phase 3] DRIFT END - Sending {post_records} normal records...")
    for i in range(post_records):
        trip = gen_base_trip(hour=((pre_records + duration + i) // 100) % 24,
                            counter=pre_records + duration + i)
        trip['_phase'] = 'post_drift'
        producer.send(RAW_TOPIC, trip)

    producer.flush()
    producer.close()
    print(f"\n[COMPLETE] Transient drift test done!")


def run_recurring_drift(bootstrap, cycles=3, cycle_duration=100, pre_records=200, post_records=100, multiplier=3.0):
    """Run recurring drift test: drift appears in cycles."""
    print(f"\n{'='*60}")
    print(f"DRIFT TYPE: RECURRING_DRIFT")
    print(f"{'='*60}")
    print(f"  Cycles: {cycles}")
    print(f"  Cycle duration: {cycle_duration} records")
    print(f"  Multiplier: {multiplier}x")

    producer = create_producer(bootstrap)
    injector = AbruptDriftInjector(multiplier=multiplier)

    counter = 0

    # Pre-drift
    print(f"\n[Phase 1] Sending {pre_records} normal records...")
    for i in range(pre_records):
        trip = gen_base_trip(hour=(i // 100) % 24, counter=counter)
        trip['_phase'] = 'pre_drift'
        producer.send(RAW_TOPIC, trip)
        counter += 1

    # Recurring cycles
    print(f"\n[Phase 2] DRIFT CYCLES - {cycles} cycles...")
    for cycle in range(cycles):
        print(f"\n  Cycle {cycle + 1}/{cycles}:")

        # Normal period
        print(f"    Normal period: {cycle_duration} records")
        for i in range(cycle_duration):
            trip = gen_base_trip(hour=((counter) // 100) % 24, counter=counter)
            trip['_phase'] = 'normal'
            trip['_cycle'] = cycle
            producer.send(RAW_TOPIC, trip)
            counter += 1

        # Drift period
        print(f"    Drift period: {cycle_duration} records")
        for i in range(cycle_duration):
            trip = gen_base_trip(hour=((counter) // 100) % 24, counter=counter)
            trip = injector.apply(trip, progress=0.5)
            trip['_phase'] = 'drift'
            trip['_cycle'] = cycle
            producer.send(RAW_TOPIC, trip)
            counter += 1

    # Post-drift
    print(f"\n[Phase 3] Sending {post_records} normal records...")
    for i in range(post_records):
        trip = gen_base_trip(hour=((counter) // 100) % 24, counter=counter)
        trip['_phase'] = 'post_drift'
        producer.send(RAW_TOPIC, trip)
        counter += 1

    producer.flush()
    producer.close()
    print(f"\n[COMPLETE] Recurring drift test done!")


def run_feature_drift(bootstrap, duration=300, pre_records=500, post_records=100, multiplier=5.0):
    """Run feature drift test: only specific features change."""
    print(f"\n{'='*60}")
    print(f"DRIFT TYPE: FEATURE_DRIFT")
    print(f"{'='*60}")
    print(f"  Only fare changes (ratio drift)")
    print(f"  Multiplier: {multiplier}x")

    producer = create_producer(bootstrap)
    injector = FeatureDriftInjector(multiplier=multiplier)

    # Pre-drift
    print(f"\n[Phase 1] Sending {pre_records} normal records...")
    for i in range(pre_records):
        trip = gen_base_trip(hour=(i // 100) % 24, counter=i)
        trip['_phase'] = 'pre_drift'
        producer.send(RAW_TOPIC, trip)

    # Feature drift
    print(f"\n[Phase 2] DRIFT START - Feature drift over {duration} records...")
    for i in range(duration):
        trip = gen_base_trip(hour=((pre_records + i) // 100) % 24,
                            counter=pre_records + i)
        trip = injector.apply(trip, progress=0.5)
        trip['_phase'] = 'drift'
        trip['_record_in_drift'] = i
        producer.send(RAW_TOPIC, trip)
        if i % 50 == 0:
            print(f"  Record {i}/{duration} - fare/distance ratio drift")

    # Post-drift
    print(f"\n[Phase 3] DRIFT END - Sending {post_records} normal records...")
    for i in range(post_records):
        trip = gen_base_trip(hour=((pre_records + duration + i) // 100) % 24,
                            counter=pre_records + duration + i)
        trip['_phase'] = 'post_drift'
        producer.send(RAW_TOPIC, trip)

    producer.flush()
    producer.close()
    print(f"\n[COMPLETE] Feature drift test done!")


def run_label_drift(bootstrap, duration=400, pre_records=500, post_records=100, fraud_rate=0.5):
    """Run label drift test: more anomalies injected."""
    print(f"\n{'='*60}")
    print(f"DRIFT TYPE: LABEL_DRIFT")
    print(f"{'='*60}")
    print(f"  Fraud injection rate: {fraud_rate*100}%")
    print(f"  Duration: {duration} records")

    producer = create_producer(bootstrap)
    injector = LabelDriftInjector(fraud_rate=fraud_rate)

    # Pre-drift
    print(f"\n[Phase 1] Sending {pre_records} normal records...")
    for i in range(pre_records):
        trip = gen_base_trip(hour=(i // 100) % 24, counter=i)
        trip['_phase'] = 'pre_drift'
        producer.send(RAW_TOPIC, trip)

    # Label drift: inject more fraud
    print(f"\n[Phase 2] DRIFT START - High fraud injection over {duration} records...")
    fraud_count = 0
    for i in range(duration):
        trip = gen_base_trip(hour=((pre_records + i) // 100) % 24,
                            counter=pre_records + i)
        original_trip = dict(trip)
        trip = injector.apply(trip, progress=0.5)
        trip['_phase'] = 'drift'
        trip['_record_in_drift'] = i
        if trip.get('_drift_type') == 'LABEL_DRIFT':
            fraud_count += 1
        producer.send(RAW_TOPIC, trip)
        if i % 80 == 0:
            print(f"  Record {i}/{duration} - fraud injected: {fraud_count} so far")

    # Post-drift: return to normal fraud rate
    print(f"\n[Phase 3] DRIFT END - Sending {post_records} normal records...")
    for i in range(post_records):
        trip = gen_base_trip(hour=((pre_records + duration + i) // 100) % 24,
                            counter=pre_records + duration + i)
        trip['_phase'] = 'post_drift'
        producer.send(RAW_TOPIC, trip)

    producer.flush()
    producer.close()
    print(f"\n[COMPLETE] Label drift test done! Total fraud injected: {fraud_count}")


def run_distribution_shift(bootstrap, duration=400, pre_records=500, post_records=100):
    """Run distribution shift test: neighborhood distribution changes."""
    print(f"\n{'='*60}")
    print(f"DRIFT TYPE: DISTRIBUTION_SHIFT")
    print(f"{'='*60}")
    print(f"  Target: Airport traffic shift")
    print(f"  Duration: {duration} records")

    producer = create_producer(bootstrap)
    injector = DistributionShiftInjector(target_neighborhood='airport')

    # Pre-drift: normal distribution
    print(f"\n[Phase 1] Sending {pre_records} normal distribution records...")
    for i in range(pre_records):
        trip = gen_base_trip(hour=(i // 100) % 24, counter=i)
        trip['_phase'] = 'pre_drift'
        producer.send(RAW_TOPIC, trip)

    # Distribution shift: all traffic to airport
    print(f"\n[Phase 2] DRIFT START - Distribution shift over {duration} records...")
    for i in range(duration):
        trip = gen_base_trip(hour=((pre_records + i) // 100) % 24,
                            counter=pre_records + i)
        trip = injector.apply(trip, progress=0.5)
        trip['_phase'] = 'drift'
        trip['_record_in_drift'] = i
        producer.send(RAW_TOPIC, trip)
        if i % 80 == 0:
            print(f"  Record {i}/{duration} - shifted to airport distribution")

    # Post-drift: return to normal
    print(f"\n[Phase 3] DRIFT END - Sending {post_records} normal records...")
    for i in range(post_records):
        trip = gen_base_trip(hour=((pre_records + duration + i) // 100) % 24,
                            counter=pre_records + duration + i)
        trip['_phase'] = 'post_drift'
        producer.send(RAW_TOPIC, trip)

    producer.flush()
    producer.close()
    print(f"\n[COMPLETE] Distribution shift test done!")


def run_all_drift_types(bootstrap, records_per_drift=800):
    """Run all drift types sequentially for comprehensive testing."""
    print(f"\n{'#'*80}")
    print(f"# FULL DRIFT BENCHMARK: ALL DRIFT TYPES")
    print(f"{'#'*80}")

    drift_functions = [
        ('ABRUPT_DRIFT', run_abrupt_drift),
        ('GRADUAL_DRIFT', run_gradual_drift),
        ('TRANSIENT_DRIFT', run_transient_drift),
        ('RECURRING_DRIFT', run_recurring_drift),
        ('FEATURE_DRIFT', run_feature_drift),
        ('LABEL_DRIFT', run_label_drift),
        ('DISTRIBUTION_SHIFT', run_distribution_shift),
    ]

    total_start = time.time()

    for drift_name, drift_func in drift_functions:
        drift_info = DRIFT_TYPES.get(drift_name, {})
        print(f"\n\n{'#'*80}")
        print(f"# Starting: {drift_name}")
        print(f"# {drift_info.get('description', '')}")
        print(f"{'#'*80}\n")

        try:
            if drift_name == 'ABRUPT_DRIFT':
                drift_func(bootstrap, duration=200, pre_records=200, post_records=50)
            elif drift_name == 'GRADUAL_DRIFT':
                drift_func(bootstrap, duration=300, pre_records=200, post_records=50, multiplier=2.5)
            elif drift_name == 'TRANSIENT_DRIFT':
                drift_func(bootstrap, duration=150, pre_records=200, post_records=50)
            elif drift_name == 'RECURRING_DRIFT':
                drift_func(bootstrap, cycles=2, cycle_duration=80, pre_records=100, post_records=50)
            elif drift_name == 'FEATURE_DRIFT':
                drift_func(bootstrap, duration=200, pre_records=200, post_records=50)
            elif drift_name == 'LABEL_DRIFT':
                drift_func(bootstrap, duration=200, pre_records=200, post_records=50, fraud_rate=0.4)
            elif drift_name == 'DISTRIBUTION_SHIFT':
                drift_func(bootstrap, duration=200, pre_records=200, post_records=50)

            print(f"\n[OK] {drift_name} completed successfully")

        except Exception as e:
            print(f"\n[ERROR] {drift_name} failed: {e}")
            import traceback
            traceback.print_exc()

        # Pause between drift types
        print(f"\n[Pause] Waiting 5 seconds before next drift type...")
        time.sleep(5)

    total_elapsed = time.time() - total_start
    print(f"\n\n{'#'*80}")
    print(f"# FULL DRIFT BENCHMARK COMPLETE")
    print(f"# Total time: {total_elapsed:.1f} seconds")
    print(f"{'#'*80}")


# ─── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='CA-DQStream Concept Drift Injection Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Drift Types Available:
  ABRUPT_DRIFT      - Sudden change (e.g., 10x fare spike)
  GRADUAL_DRIFT     - Slow change over time (seasonal shift)
  TRANSIENT_DRIFT   - Short burst then disappear (spike window)
  RECURRING_DRIFT   - Cyclic drift (rush hour patterns)
  FEATURE_DRIFT     - Only specific features change
  LABEL_DRIFT       - More anomalies injected (fraud rate increase)
  DISTRIBUTION_SHIFT - Entire data distribution changes
  ALL               - Run all drift types sequentially

Examples:
  # Run specific drift type
  python scripts/inject_concept_drift.py --drift-type ABRUPT_DRIFT

  # Run with custom parameters
  python scripts/inject_concept_drift.py --drift-type GRADUAL_DRIFT --duration 600 --multiplier 3.0

  # Run all drift types
  python scripts/inject_concept_drift.py --drift-type ALL

  # Run with custom Kafka
  python scripts/inject_concept_drift.py --drift-type ABRUPT_DRIFT --bootstrap localhost:9092
"""
    )

    parser.add_argument('--bootstrap', default=DEFAULT_BOOTSTRAP,
                       help=f'Kafka bootstrap servers (default: {DEFAULT_BOOTSTRAP})')
    parser.add_argument('--drift-type', required=True,
                       choices=list(DRIFT_TYPES.keys()) + ['ALL'],
                       help='Type of drift to inject')
    parser.add_argument('--duration', type=int, default=300,
                       help='Duration of drift in records (default: 300)')
    parser.add_argument('--multiplier', type=float, default=10.0,
                       help='Drift multiplier (default: 10.0)')
    parser.add_argument('--pre-records', type=int, default=500,
                       help='Records before drift (default: 500)')
    parser.add_argument('--post-records', type=int, default=100,
                       help='Records after drift (default: 100)')
    parser.add_argument('--fraud-rate', type=float, default=0.5,
                       help='Fraud injection rate for LABEL_DRIFT (default: 0.5)')

    args = parser.parse_args()

    print(f"[CONFIG]")
    print(f"  Bootstrap: {args.bootstrap}")
    print(f"  Drift Type: {args.drift_type}")
    print(f"  Duration: {args.duration}")
    print(f"  Multiplier: {args.multiplier}")

    if args.drift_type == 'ALL':
        run_all_drift_types(args.bootstrap)
    elif args.drift_type == 'ABRUPT_DRIFT':
        run_abrupt_drift(args.bootstrap, duration=args.duration,
                        pre_records=args.pre_records, post_records=args.post_records,
                        multiplier=args.multiplier)
    elif args.drift_type == 'GRADUAL_DRIFT':
        run_gradual_drift(args.bootstrap, duration=args.duration,
                         pre_records=args.pre_records, post_records=args.post_records,
                         multiplier=args.multiplier)
    elif args.drift_type == 'TRANSIENT_DRIFT':
        run_transient_drift(args.bootstrap, duration=args.duration,
                           pre_records=args.pre_records, post_records=args.post_records,
                           multiplier=args.multiplier)
    elif args.drift_type == 'RECURRING_DRIFT':
        run_recurring_drift(args.bootstrap, cycles=3, cycle_duration=args.duration//2,
                           pre_records=args.pre_records, post_records=args.post_records,
                           multiplier=args.multiplier)
    elif args.drift_type == 'FEATURE_DRIFT':
        run_feature_drift(args.bootstrap, duration=args.duration,
                         pre_records=args.pre_records, post_records=args.post_records,
                         multiplier=args.multiplier)
    elif args.drift_type == 'LABEL_DRIFT':
        run_label_drift(args.bootstrap, duration=args.duration,
                       pre_records=args.pre_records, post_records=args.post_records,
                       fraud_rate=args.fraud_rate)
    elif args.drift_type == 'DISTRIBUTION_SHIFT':
        run_distribution_shift(args.bootstrap, duration=args.duration,
                             pre_records=args.pre_records, post_records=args.post_records)

    print(f"\n[DONE] Drift injection complete!")


if __name__ == '__main__':
    main()
