#!/usr/bin/env python3
"""
CA-DQStream Concept Drift Injection & Verification Tool.

Covers ALL concept drift types that the pipeline can detect and handle.
Uses the existing anomaly_producer.py framework to inject realistic drift patterns.

Drift Types:
  1. SUDDEN_FARE_SPIKE   - All fares 10x for a window (abrupt change)
  2. GRADUAL_FARE_RISE   - Fares increase +20% to +100% over 30 min (gradual)
  3. ZONE_SHIFT          - All trips move to single neighborhood (spatial)
  4. FEATURE_CORRUPTION  - Multiple features change simultaneously (covariate)
  5. VOLUME_SPIKE        - Sudden 5x volume in one neighborhood (population)
  6. NO_DRIFT_BASELINE   - Normal data, should NOT trigger drift alerts

Expected System Response:
  - Canary violations increase (extreme_fare, zero_distance, etc.)
  - ML anomaly rate spikes (MemStream sees unusual patterns)
  - MetaAggregator delta_score diverges (canary vs ML disagreement)
  - ADWIN-U detects drift in Layer 4
  - IEC executes strategy (adjust_threshold -> retrain_model -> switch_model)
  - Prometheus alerts fire (IECDriftRateHigh, DriftDetected, etc.)
  - Alerts published to MinIO

Usage:
  python drift_injector.py --scenario SUDDEN_FARE_SPIKE --dry-run
  python drift_injector.py --scenario GRADUAL_FARE_RISE --duration 30 --records 5000
  python drift_injector.py --scenario ZONE_SHIFT --neighborhood airport
  python drift_injector.py --scenario ALL --sequential
"""

import json
import time
import random
import argparse
import sys
import os
import threading
import statistics
from datetime import datetime
from collections import deque
from kafka import KafkaProducer
from kafka.errors import KafkaError

# ─── Kafka Config ────────────────────────────────────────────────────────────
DEFAULT_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
RAW_TOPIC = os.getenv('RAW_TOPIC', 'taxi-nyc-raw-v2')

NYC_ZONE_MIN, NYC_ZONE_MAX = 1, 263
PAYMENT_TYPES = [1, 2, 3, 4, 5, 6]

# Trip profiles: (name, d_min, d_max, f_min, f_max, du_min, du_max)
TRIP_PROFILES = [
    ("short",   0.5,  3.0,   5.0,  20.0,   300, 1800),
    ("medium",  2.0, 10.0,  15.0,  60.0,   900, 3600),
    ("long",    8.0, 30.0,  40.0, 200.0,  1800, 5400),
    ("airport", 15.0, 35.0,  50.0, 150.0,  2400, 5400),
]

# Neighborhood zone ranges
MANHATTAN_ZONES = list(range(1, 51))
BROOKLYN_ZONES  = list(range(51, 101))
QUEENS_ZONES     = list(range(101, 151))
BRONX_ZONES      = list(range(151, 201))
AIRPORT_ZONES    = [132, 138]
STATEN_ISLAND    = list(range(201, 264))
ALL_ZONES        = list(range(1, 264))

NEIGHBORHOOD_ZONES = {
    'manhattan': MANHATTAN_ZONES,
    'brooklyn':  BROOKLYN_ZONES,
    'queens':    QUEENS_ZONES,
    'bronx':     BRONX_ZONES,
    'airport':   AIRPORT_ZONES,
    'staten_island': STATEN_ISLAND,
}


# ─── Baseline Trip Generator ───────────────────────────────────────────────────

def gen_base_trip(counter=0, hour_offset=0):
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

    hour = (hour_offset + counter) % 24
    pu_min = random.randint(0, 59)
    pu_sec = random.randint(0, 59)
    do_min_total = int(pu_min + dur_sec / 60.0)
    do_hour = (hour + do_min_total // 60) % 24
    do_min = do_min_total % 60
    day = (counter // 100) % 28 + 1

    return {
        "VendorID": random.choice([1, 2]),
        "tpep_pickup_datetime": f"2024-01-{day:02d}T{hour:02d}:{pu_min:02d}:{pu_sec:02d}",
        "tpep_dropoff_datetime": f"2024-01-{day:02d}T{do_hour:02d}:{do_min:02d}:{pu_sec:02d}",
        "passenger_count": float(random.choice([1, 2, 3, 4, 5, 6])),
        "trip_distance": distance,
        "RatecodeID": 1.0,
        "store_and_fwd_flag": "N",
        "PULocationID": float(random.choice(ALL_ZONES)),
        "DOLocationID": float(random.choice(ALL_ZONES)),
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
        "_producer_ts": datetime.utcnow().isoformat(),
    }


# ─── Scenario Injectors ───────────────────────────────────────────────────────

def apply_sudden_fare_spike(trip, multiplier=10.0):
    """SCENARIO 1: Sudden Fare Spike - all fares 10x."""
    trip = dict(trip)
    trip['fare_amount'] = round(trip['fare_amount'] * multiplier, 2)
    trip['total_amount'] = round(trip['total_amount'] * multiplier, 2)
    trip['_drift_type'] = 'SUDDEN_FARE_SPIKE'
    trip['_drift_multiplier'] = multiplier
    return trip


def apply_gradual_fare_rise(trip, progress, max_multiplier=3.0):
    """SCENARIO 2: Gradual Fare Rise - fares increase over time."""
    trip = dict(trip)
    # progress: 0.0 to 1.0 over the drift window
    multiplier = 1.0 + (max_multiplier - 1.0) * progress
    trip['fare_amount'] = round(trip['fare_amount'] * multiplier, 2)
    trip['total_amount'] = round(trip['total_amount'] * multiplier, 2)
    trip['_drift_type'] = 'GRADUAL_FARE_RISE'
    trip['_drift_progress'] = round(progress, 3)
    return trip


def apply_zone_shift(trip, target_neighborhood):
    """SCENARIO 3: Zone Shift - all trips move to one neighborhood."""
    trip = dict(trip)
    zones = NEIGHBORHOOD_ZONES.get(target_neighborhood, ALL_ZONES)
    trip['PULocationID'] = float(random.choice(zones))
    trip['DOLocationID'] = float(random.choice(zones))
    trip['_drift_type'] = 'ZONE_SHIFT'
    trip['_target_neighborhood'] = target_neighborhood
    return trip


def apply_feature_corruption(trip, severity=0.5):
    """SCENARIO 4: Feature Corruption - multiple features shift."""
    trip = dict(trip)
    # Multiply distance by 0.1x to 0.3x
    trip['trip_distance'] = round(trip['trip_distance'] * random.uniform(0.1, 0.3), 2)
    # Multiply fare by 2x to 5x
    trip['fare_amount'] = round(trip['fare_amount'] * random.uniform(2.0, 5.0), 2)
    trip['total_amount'] = round(trip['fare_amount'] + trip['extra'] + trip['mta_tax'] + trip['tip_amount'] + trip['tolls_amount'] + 1.0 + 2.5, 2)
    # Corrupt speed
    dur_h = trip['trip_duration']
    trip['speed_mph'] = round(trip['trip_distance'] / dur_h, 6) if dur_h > 0 else 0.0
    # Set passenger to invalid
    trip['passenger_count'] = 0.0
    trip['_drift_type'] = 'FEATURE_CORRUPTION'
    trip['_severity'] = severity
    return trip


def apply_volume_spike(trip, target_neighborhood, spike_ratio=0.95):
    """SCENARIO 5: Volume Spike - most trips target one neighborhood."""
    trip = dict(trip)
    if random.random() < spike_ratio:
        zones = NEIGHBORHOOD_ZONES.get(target_neighborhood, ALL_ZONES)
        trip['PULocationID'] = float(random.choice(zones))
        trip['DOLocationID'] = float(random.choice(zones))
        trip['_drift_type'] = 'VOLUME_SPIKE'
        trip['_target_neighborhood'] = target_neighborhood
    return trip


# ─── Kafka Producer ───────────────────────────────────────────────────────────

def create_producer(bootstrap):
    try:
        return KafkaProducer(
            bootstrap_servers=[bootstrap],
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            acks='all',
            retries=3,
            compression_type='gzip',
        )
    except Exception as e:
        print(f"[FATAL] Cannot connect to Kafka at {bootstrap}: {e}")
        sys.exit(1)


# ─── Dry-Run Mode ─────────────────────────────────────────────────────────────

def dry_run_scenario(scenario, records, kwargs):
    """Print what would be sent without actually sending."""
    print(f"\n{'='*70}")
    print(f"DRIFT INJECTION DRY RUN: {scenario}")
    print(f"{'='*70}")
    print(f"Records to send: {records}")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")
    print(f"\nExpected system response:")
    expected = {
        'SUDDEN_FARE_SPIKE': [
            "Canary: extreme_fare violations spike (>20% of records)",
            "ML: anomaly rate spikes (MemStream sees unusual patterns)",
            "MetaAggregator: violation_rate + delta_score diverge",
            "ADWIN-U: drift detected in null_rate or anomaly_rate",
            "IEC: execute adjust_threshold -> retrain_model",
            "Alert: DriftDetected, IECDriftRateHigh",
        ],
        'GRADUAL_FARE_RISE': [
            "Canary: gradual increase in extreme_fare violations",
            "ML: gradual anomaly score increase",
            "MetaAggregator: violation_rate slowly climbs",
            "ADWIN-U: sustained drift in violation_rate",
            "IEC: execute retrain_model (moderate severity)",
            "Alert: IECRetrainTriggered",
        ],
        'ZONE_SHIFT': [
            "Canary: minimal impact (rules still pass)",
            "ML: anomaly rate spikes (unusual zone distribution)",
            "MetaAggregator: volume shifts to target neighborhood",
            "ADWIN-U: drift detected in volume metric",
            "IEC: execute adjust_threshold or retrain_model",
            "Alert: DriftDetected (zone-specific)",
        ],
        'FEATURE_CORRUPTION': [
            "Canary: invalid_passengers + extreme_fare + impossible patterns",
            "ML: very high anomaly rate (>30%)",
            "MetaAggregator: violation_rate spikes, delta_score high",
            "ADWIN-U: drift in multiple metrics",
            "IEC: execute retrain_model -> switch_model (high severity)",
            "Alert: IECSwitchTriggered, DriftDetected",
        ],
        'VOLUME_SPIKE': [
            "Canary: no direct impact",
            "ML: neighborhood-specific anomaly rate change",
            "MetaAggregator: volume metric spikes in target neighborhood",
            "ADWIN-U: drift in volume for target neighborhood",
            "IEC: execute do_nothing or adjust_threshold",
            "Alert: DriftDetected (volume metric only)",
        ],
        'NO_DRIFT_BASELINE': [
            "Canary: normal ~5-8% baseline violations",
            "ML: normal ~5-10% baseline anomaly rate",
            "MetaAggregator: stable meta-metrics",
            "ADWIN-U: NO drift detected",
            "IEC: execute do_nothing (zero drift events)",
            "Alert: NO alerts (baseline only)",
        ],
    }
    for resp in expected.get(scenario, []):
        print(f"  - {resp}")
    print()


# ─── Injection Routines ──────────────────────────────────────────────────────

def inject_sudden_fare_spike(bootstrap, records=3000, pre_records=500, post_records=500,
                               multiplier=10.0, rate=100):
    """Inject sudden fare spike: normal -> 10x fares -> normal."""
    print(f"\n{'#'*70}")
    print(f"# SCENARIO 1: SUDDEN FARE SPIKE (multiplier={multiplier}x)")
    print(f"# Pre: {pre_records} normal | Drift: {records} | Post: {post_records}")
    print(f"# Rate: ~{rate}/sec | Expected: Canary extreme_fare + ML anomaly spike")
    print(f"{'#'*70}")
    prod = create_producer(bootstrap)
    interval = 1.0 / rate if rate > 0 else 0.01

    # Phase 1: Pre-drift (normal)
    print(f"[1/3] Pre-drift: sending {pre_records} normal records...")
    for i in range(pre_records):
        trip = gen_base_trip(counter=i)
        trip['_phase'] = 'pre_drift'
        prod.send(RAW_TOPIC, trip)
        if i % 100 == 0:
            print(f"  sent {i}/{pre_records} (fare ~${trip['fare_amount']:.2f})")
        time.sleep(interval)
    prod.flush()
    print(f"  Pre-drift done. Fare range: $0-$200")
    time.sleep(5)

    # Phase 2: Sudden spike (all 10x)
    print(f"[2/3] DRIFT SPIKE: sending {records} records with {multiplier}x fare...")
    drift_start = time.time()
    for i in range(records):
        trip = gen_base_trip(counter=pre_records + i)
        trip = apply_sudden_fare_spike(trip, multiplier=multiplier)
        prod.send(RAW_TOPIC, trip)
        if i % 100 == 0:
            elapsed = time.time() - drift_start
            pct = i / records * 100
            print(f"  [{pct:5.1f}%] sent {i}/{records} | fare ~${trip['fare_amount']:.2f} | elapsed {elapsed:.1f}s")
        time.sleep(interval)
    prod.flush()
    print(f"  Spike done. Fare range: ${multiplier}x baseline ($0-${200*multiplier})")
    time.sleep(5)

    # Phase 3: Post-drift (normal)
    print(f"[3/3] Post-drift: sending {post_records} normal records...")
    for i in range(post_records):
        trip = gen_base_trip(counter=pre_records + records + i)
        trip['_phase'] = 'post_drift'
        prod.send(RAW_TOPIC, trip)
        if i % 100 == 0:
            print(f"  sent {i}/{post_records} (fare ~${trip['fare_amount']:.2f})")
        time.sleep(interval)
    prod.flush()
    prod.close()
    print(f"COMPLETE: {pre_records + records + post_records} records sent")
    print(f"Expected alerts: DriftDetected, IECRetrainTriggered, IECDriftRateHigh")


def inject_gradual_fare_rise(bootstrap, records=3000, pre_records=500, post_records=500,
                               max_multiplier=3.0, rate=100):
    """Inject gradual fare rise: fares increase from +20% to +100%."""
    print(f"\n{'#'*70}")
    print(f"# SCENARIO 2: GRADUAL FARE RISE (max {max_multiplier}x over {records} records)")
    print(f"# Pre: {pre_records} normal | Drift: {records} gradual | Post: {post_records}")
    print(f"# Rate: ~{rate}/sec | Expected: Gradual canary + ML rise, sustained IEC")
    print(f"{'#'*70}")
    prod = create_producer(bootstrap)
    interval = 1.0 / rate if rate > 0 else 0.01

    # Phase 1: Pre-drift
    print(f"[1/3] Pre-drift: {pre_records} normal records...")
    for i in range(pre_records):
        trip = gen_base_trip(counter=i)
        trip['_phase'] = 'pre_drift'
        prod.send(RAW_TOPIC, trip)
        time.sleep(interval)
    prod.flush()
    print(f"  Done. Normal baseline established.")
    time.sleep(3)

    # Phase 2: Gradual drift
    print(f"[2/3] GRADUAL RISE: {records} records with progressive fare increase...")
    drift_start = time.time()
    for i in range(records):
        progress = i / max(1, records - 1)  # 0.0 to 1.0
        trip = gen_base_trip(counter=pre_records + i)
        trip = apply_gradual_fare_rise(trip, progress, max_multiplier=max_multiplier)
        prod.send(RAW_TOPIC, trip)
        if i % 300 == 0:
            elapsed = time.time() - drift_start
            current_mult = 1.0 + (max_multiplier - 1.0) * progress
            print(f"  [{i}/{records}] progress={progress*100:.1f}% | fare ~${trip['fare_amount']:.2f} ({current_mult:.1f}x) | {elapsed:.1f}s")
        time.sleep(interval)
    prod.flush()
    time.sleep(5)

    # Phase 3: Post-drift
    print(f"[3/3] Post-drift: {post_records} normal records...")
    for i in range(post_records):
        trip = gen_base_trip(counter=pre_records + records + i)
        trip['_phase'] = 'post_drift'
        prod.send(RAW_TOPIC, trip)
        time.sleep(interval)
    prod.flush()
    prod.close()
    print(f"COMPLETE: {pre_records + records + post_records} records sent")
    print(f"Expected alerts: IECRetrainTriggered (moderate drift, sustained)")


def inject_zone_shift(bootstrap, records=2000, pre_records=500, post_records=500,
                       target='airport', rate=100):
    """Inject zone shift: all trips target one neighborhood."""
    print(f"\n{'#'*70}")
    print(f"# SCENARIO 3: ZONE SHIFT -> {target.upper()}")
    print(f"# Pre: {pre_records} normal | Shift: {records} | Post: {post_records}")
    print(f"# Rate: ~{rate}/sec | Expected: ML anomaly spike, volume drift, IEC adjust")
    print(f"{'#'*70}")
    prod = create_producer(bootstrap)
    interval = 1.0 / rate if rate > 0 else 0.01

    print(f"[1/3] Pre-drift: {pre_records} normal records...")
    for i in range(pre_records):
        trip = gen_base_trip(counter=i)
        trip['_phase'] = 'pre_drift'
        prod.send(RAW_TOPIC, trip)
        time.sleep(interval)
    prod.flush()
    time.sleep(3)

    print(f"[2/3] ZONE SHIFT: {records} records targeting {target}...")
    drift_start = time.time()
    for i in range(records):
        trip = gen_base_trip(counter=pre_records + i)
        trip = apply_zone_shift(trip, target)
        prod.send(RAW_TOPIC, trip)
        if i % 400 == 0:
            elapsed = time.time() - drift_start
            print(f"  [{i}/{records}] zone={trip['PULocationID']:.0f} | {target} zones: {NEIGHBORHOOD_ZONES[target]} | {elapsed:.1f}s")
        time.sleep(interval)
    prod.flush()
    time.sleep(5)

    print(f"[3/3] Post-drift: {post_records} normal records...")
    for i in range(post_records):
        trip = gen_base_trip(counter=pre_records + records + i)
        trip['_phase'] = 'post_drift'
        prod.send(RAW_TOPIC, trip)
        time.sleep(interval)
    prod.flush()
    prod.close()
    print(f"COMPLETE")
    print(f"Expected: ML anomaly spike (unusual zone pattern), volume drift in {target}")


def inject_feature_corruption(bootstrap, records=2000, pre_records=500, post_records=500,
                               rate=100):
    """Inject feature corruption: distance drops, fare spikes, passengers invalid."""
    print(f"\n{'#'*70}")
    print(f"# SCENARIO 4: FEATURE CORRUPTION (multiple features)")
    print(f"# Pre: {pre_records} | Corruption: {records} | Post: {post_records}")
    print(f"# Rate: ~{rate}/sec | Expected: Heavy canary + ML spike, IEC switch_model")
    print(f"{'#'*70}")
    prod = create_producer(bootstrap)
    interval = 1.0 / rate if rate > 0 else 0.01

    print(f"[1/3] Pre-drift: {pre_records} normal records...")
    for i in range(pre_records):
        trip = gen_base_trip(counter=i)
        trip['_phase'] = 'pre_drift'
        prod.send(RAW_TOPIC, trip)
        time.sleep(interval)
    prod.flush()
    time.sleep(3)

    print(f"[2/3] CORRUPTION: {records} corrupted records...")
    drift_start = time.time()
    for i in range(records):
        trip = gen_base_trip(counter=pre_records + i)
        trip = apply_feature_corruption(trip)
        prod.send(RAW_TOPIC, trip)
        if i % 400 == 0:
            elapsed = time.time() - drift_start
            print(f"  [{i}/{records}] dist={trip['trip_distance']:.2f}mi | fare=${trip['fare_amount']:.2f} | pax={trip['passenger_count']:.0f} | {elapsed:.1f}s")
        time.sleep(interval)
    prod.flush()
    time.sleep(5)

    print(f"[3/3] Post-drift: {post_records} normal records...")
    for i in range(post_records):
        trip = gen_base_trip(counter=pre_records + records + i)
        trip['_phase'] = 'post_drift'
        prod.send(RAW_TOPIC, trip)
        time.sleep(interval)
    prod.flush()
    prod.close()
    print(f"COMPLETE")
    print(f"Expected: Heavy canary violations (invalid_passengers, extreme_fare), IEC switch_model")


def inject_volume_spike(bootstrap, records=2000, pre_records=500, post_records=500,
                          target='airport', spike_ratio=0.95, rate=100):
    """Inject volume spike: 95% of trips go to one neighborhood."""
    print(f"\n{'#'*70}")
    print(f"# SCENARIO 5: VOLUME SPIKE -> {target.upper()} ({spike_ratio*100:.0f}% of traffic)")
    print(f"# Pre: {pre_records} | Spike: {records} | Post: {post_records}")
    print(f"# Rate: ~{rate}/sec | Expected: Volume metric drift, IEC do_nothing/adjust")
    print(f"{'#'*70}")
    prod = create_producer(bootstrap)
    interval = 1.0 / rate if rate > 0 else 0.01

    print(f"[1/3] Pre-drift: {pre_records} normal records...")
    for i in range(pre_records):
        trip = gen_base_trip(counter=i)
        trip['_phase'] = 'pre_drift'
        prod.send(RAW_TOPIC, trip)
        time.sleep(interval)
    prod.flush()
    time.sleep(3)

    print(f"[2/3] VOLUME SPIKE: {records} records ({spike_ratio*100:.0f}% to {target})...")
    drift_start = time.time()
    for i in range(records):
        trip = gen_base_trip(counter=pre_records + i)
        trip = apply_volume_spike(trip, target, spike_ratio)
        prod.send(RAW_TOPIC, trip)
        if i % 400 == 0:
            elapsed = time.time() - drift_start
            print(f"  [{i}/{records}] target={target} | PULoc={trip['PULocationID']:.0f} | {elapsed:.1f}s")
        time.sleep(interval)
    prod.flush()
    time.sleep(5)

    print(f"[3/3] Post-drift: {post_records} normal records...")
    for i in range(post_records):
        trip = gen_base_trip(counter=pre_records + records + i)
        trip['_phase'] = 'post_drift'
        prod.send(RAW_TOPIC, trip)
        time.sleep(interval)
    prod.flush()
    prod.close()
    print(f"COMPLETE")
    print(f"Expected: Volume drift in {target} neighborhood, IEC do_nothing (low severity)")


def inject_no_drift_baseline(bootstrap, records=2000, rate=100):
    """Baseline: normal data only - should NOT trigger drift alerts."""
    print(f"\n{'#'*70}")
    print(f"# SCENARIO 6: NO DRIFT BASELINE ({records} normal records)")
    print(f"# Expected: NO drift alerts, stable meta-metrics")
    print(f"{'#'*70}")
    prod = create_producer(bootstrap)
    interval = 1.0 / rate if rate > 0 else 0.01

    print(f"Sending {records} normal records...")
    start = time.time()
    for i in range(records):
        trip = gen_base_trip(counter=i)
        trip['_phase'] = 'baseline'
        prod.send(RAW_TOPIC, trip)
        if i % 500 == 0:
            elapsed = time.time() - start
            rate_actual = i / max(elapsed, 0.1)
            print(f"  [{i}/{records}] fare=${trip['fare_amount']:.2f} | {rate_actual:.0f}/sec")
        time.sleep(interval)
    prod.flush()
    prod.close()
    print(f"COMPLETE: {records} normal records")
    print(f"Expected: Stable ~5-8% canary violations, ~5-10% ML anomalies, NO drift")


def inject_sequential_all(bootstrap, rate=100):
    """Run ALL scenarios sequentially with wait times between."""
    print(f"\n{'='*70}")
    print(f"SEQUENTIAL FULL DRIFT TEST: ALL 5 SCENARIOS")
    print(f"{'='*70}\n")

    scenarios = [
        ("NO_DRIFT_BASELINE", inject_no_drift_baseline, {'records': 1000, 'rate': rate}),
        ("SUDDEN_FARE_SPIKE", inject_sudden_fare_spike, {'records': 2000, 'pre_records': 300, 'post_records': 300, 'multiplier': 10.0, 'rate': rate}),
        ("GRADUAL_FARE_RISE", inject_gradual_fare_rise, {'records': 2000, 'pre_records': 300, 'post_records': 300, 'max_multiplier': 3.0, 'rate': rate}),
        ("ZONE_SHIFT", inject_zone_shift, {'records': 1500, 'pre_records': 300, 'post_records': 300, 'target': 'airport', 'rate': rate}),
        ("FEATURE_CORRUPTION", inject_feature_corruption, {'records': 1500, 'pre_records': 300, 'post_records': 300, 'rate': rate}),
        ("VOLUME_SPIKE", inject_volume_spike, {'records': 1500, 'pre_records': 300, 'post_records': 300, 'target': 'manhattan', 'spike_ratio': 0.95, 'rate': rate}),
    ]

    for name, func, kwargs in scenarios:
        print(f"\n\n{'#'*70}")
        print(f"# STARTING: {name}")
        print(f"{'#'*70}")
        try:
            func(bootstrap=bootstrap, **kwargs)
        except KeyboardInterrupt:
            print("Interrupted!")
            break
        print(f"\nWaiting 15 seconds before next scenario...")
        time.sleep(15)

    print(f"\n{'='*70}")
    print(f"ALL SCENARIOS COMPLETE")
    print(f"{'='*70}")
    print(f"\nNow check Grafana dashboards for:")
    print(f"  - cadqstream_iec_drift_detected_total (should be > 0)")
    print(f"  - cadqstream_iec_decisions_total (should have retrain_model + adjust_threshold)")
    print(f"  - cadqstream_meta_delta_score (should spike during drift)")
    print(f"  - cadqstream_anomalies_ml_total (should spike during drift)")
    print(f"  - Prometheus alerts: DriftDetected, IECRetrainTriggered")


# ─── Main CLI ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='CA-DQStream Concept Drift Injector')
    p.add_argument('--bootstrap', default=DEFAULT_BOOTSTRAP, help='Kafka bootstrap')
    p.add_argument('--scenario', required=True,
                   choices=['SUDDEN_FARE_SPIKE','GRADUAL_FARE_RISE','ZONE_SHIFT',
                           'FEATURE_CORRUPTION','VOLUME_SPIKE','NO_DRIFT_BASELINE','ALL'],
                   help='Drift scenario')
    p.add_argument('--dry-run', action='store_true', help='Print plan without sending')
    p.add_argument('--records', type=int, default=3000, help='Records during drift phase')
    p.add_argument('--pre-records', type=int, default=500, help='Records before drift')
    p.add_argument('--post-records', type=int, default=500, help='Records after drift')
    p.add_argument('--multiplier', type=float, default=10.0, help='Fare multiplier')
    p.add_argument('--max-multiplier', type=float, default=3.0, help='Max gradual fare multiplier')
    p.add_argument('--target', default='airport', help='Target neighborhood')
    p.add_argument('--spike-ratio', type=float, default=0.95, help='Volume spike ratio')
    p.add_argument('--rate', type=int, default=100, help='Records per second')

    args = p.parse_args()

    print(f"\nCA-DQStream Concept Drift Injection Tool")
    print(f"  Bootstrap: {args.bootstrap}")
    print(f"  Scenario:  {args.scenario}")
    print(f"  Rate:      {args.rate}/sec")

    if args.dry_run:
        dry_run_scenario(args.scenario, args.records, vars(args))
        return

    if args.scenario == 'ALL':
        inject_sequential_all(args.bootstrap, rate=args.rate)
        return

    dispatch = {
        'SUDDEN_FARE_SPIKE':   inject_sudden_fare_spike,
        'GRADUAL_FARE_RISE':   inject_gradual_fare_rise,
        'ZONE_SHIFT':          inject_zone_shift,
        'FEATURE_CORRUPTION':  inject_feature_corruption,
        'VOLUME_SPIKE':         inject_volume_spike,
        'NO_DRIFT_BASELINE':    inject_no_drift_baseline,
    }

    func = dispatch[args.scenario]

    if args.scenario == 'NO_DRIFT_BASELINE':
        func(bootstrap=args.bootstrap, records=args.records, rate=args.rate)
    elif args.scenario == 'SUDDEN_FARE_SPIKE':
        func(bootstrap=args.bootstrap, records=args.records,
             pre_records=args.pre_records, post_records=args.post_records,
             multiplier=args.multiplier, rate=args.rate)
    elif args.scenario == 'GRADUAL_FARE_RISE':
        func(bootstrap=args.bootstrap, records=args.records,
             pre_records=args.pre_records, post_records=args.post_records,
             max_multiplier=args.max_multiplier, rate=args.rate)
    elif args.scenario == 'ZONE_SHIFT':
        func(bootstrap=args.bootstrap, records=args.records,
             pre_records=args.pre_records, post_records=args.post_records,
             target=args.target, rate=args.rate)
    elif args.scenario == 'FEATURE_CORRUPTION':
        func(bootstrap=args.bootstrap, records=args.records,
             pre_records=args.pre_records, post_records=args.post_records, rate=args.rate)
    elif args.scenario == 'VOLUME_SPIKE':
        func(bootstrap=args.bootstrap, records=args.records,
             pre_records=args.pre_records, post_records=args.post_records,
             target=args.target, spike_ratio=args.spike_ratio, rate=args.rate)


if __name__ == '__main__':
    main()
