#!/usr/bin/env python3
"""
CA-DQStream End-to-End Layer Injection Test Suite.

Injects targeted test data to verify each pipeline layer triggers correctly.

Test 1: Hard Rule Violations (L1 Schema + L2 Canary)
  - Records with missing/null required fields → L1 drops → cadqstream-violations/schema/
  - Records with negative fare / zero distance / invalid passengers → L2 drops → cadqstream-violations/canary/

Test 2: Point Anomaly (L3 MemStream)
  - Valid records with extreme contextual values → passes L1/L2 → scored by ML
  - High anomaly score → cadqstream-anomalies/ + unified Kafka topic

Test 3: Canary Violation (L2 only, passes to violations)
  - Valid schema but violates business rules → cadqstream-violations/canary/

Usage:
  # Run both tests (default):
  python e2e_test_injector.py

  # Run only L1/L2 violations:
  python e2e_test_injector.py --test l1-l2

  # Run only L3 anomaly:
  python e2e_test_injector.py --test l3

  # Run with custom Kafka:
  python e2e_test_injector.py --bootstrap localhost:29092

Environment:
  KAFKA_BOOTSTRAP_SERVERS  Default: kafka:9092 (use localhost:29092 for host)
  MINIO_ENDPOINT           Default: http://localhost:9000
  MINIO_ACCESS_KEY         (required)
  MINIO_SECRET_KEY         (required)
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False


# ─── Constants ─────────────────────────────────────────────────────────────────

RAW_TOPIC = "taxi-nyc-raw-v2"
NYC_ZONE_MIN, NYC_ZONE_MAX = 1, 263

# Manhattan zones (1-43) → neighborhood idx 0, to exercise real MemStream scoring
MANHATTAN_ZONES = list(range(1, 44))

# Neighborhood names (for reference)
NB_NAMES = [
    'manhattan', 'brooklyn', 'queens_lower', 'queens_upper',
    'bronx', 'staten_island', 'ewr', 'jfk', 'nalp', 'unknown'
]

# ─── Helpers ───────────────────────────────────────────────────────────────────

def make_producer(bootstrap: str) -> Optional[object]:
    if not KAFKA_AVAILABLE:
        print("[ERROR] kafka-python not installed. Run: pip install kafka-python")
        return None
    try:
        return KafkaProducer(
            bootstrap_servers=[bootstrap],
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            acks="all",
            retries=3,
            max_in_flight_requests_per_connection=5,
            compression_type="lz4",
            linger_ms=5,
            batch_size=32768,
        )
    except Exception as e:
        print(f"[ERROR] Cannot connect to Kafka at {bootstrap}: {e}")
        return None


def make_minio_client(endpoint: str, access_key: str, secret_key: str) -> Optional[object]:
    if not MINIO_AVAILABLE:
        print("[WARN] minio package not installed. Skipping MinIO bucket verification.")
        return None
    try:
        return Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )
    except Exception as e:
        print(f"[ERROR] Cannot connect to MinIO at {endpoint}: {e}")
        return None


def _gen_base_trip(counter: int, puloc: int, dolic: int,
                    distance: float, fare: float,
                    passengers: int = 1,
                    payment: int = 1,
                    pickup_hour: int = 12,
                    pickup_minute: int = 0,
                    extra: float = 2.5,
                    mta_tax: float = 0.5,
                    tip: float = 0.0,
                    tolls: float = 0.0,
                    imp_surcharge: float = 1.0,
                    cong_surcharge: float = 2.5,
                    ratecode: int = 1) -> Dict:
    """Generate a minimal valid NYC taxi trip record with full field coverage."""
    dur_sec = 900.0  # 15 min
    if distance > 0:
        dur_sec = (distance / 15.0) * 3600  # ~15 mph
    dur_h = dur_sec / 3600.0

    total = round(fare + extra + mta_tax + tip + tolls + imp_surcharge + cong_surcharge, 2)

    base_date = datetime(2024, 6, 15, pickup_hour, pickup_minute, 0)
    pickup_str = base_date.strftime("%Y-%m-%dT%H:%M:%S")
    dropoff = base_date + timedelta(seconds=int(dur_sec))
    dropoff_str = dropoff.strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "VendorID": random.choice([1, 2]),
        "tpep_pickup_datetime": pickup_str,
        "tpep_dropoff_datetime": dropoff_str,
        "passenger_count": float(passengers),
        "trip_distance": round(distance, 2),
        "RatecodeID": float(ratecode),
        "store_and_fwd_flag": "N",
        "PULocationID": float(puloc),
        "DOLocationID": float(dolic),
        "payment_type": float(payment),
        "fare_amount": round(fare, 2),
        "extra": round(extra, 2),
        "mta_tax": round(mta_tax, 2),
        "tip_amount": round(tip, 2),
        "tolls_amount": round(tolls, 2),
        "improvement_surcharge": round(imp_surcharge, 2),
        "total_amount": round(total, 2),
        "congestion_surcharge": round(cong_surcharge, 2),
        "trip_duration": round(dur_h, 6),
        "speed_mph": round(distance / dur_h, 6) if dur_h > 0 else 0.0,
        "_test_seq": counter,
    }


# ─── Test 1: L1 Schema Violations ──────────────────────────────────────────────
# Trigger: Records missing/null required fields (trip_distance, fare_amount,
#          PULocationID, DOLocationID, passenger_count)
# Expected: L1SchemaValidator.filter() returns False → InvalidFilter → MinIO cadqstream-violations/schema/

def send_l1_schema_violations(producer, topic: str, n: int = 5) -> int:
    """Send records that fail L1 schema validation (missing required fields)."""
    print(f"\n{'='*60}")
    print("TEST 1: L1 Schema Violations")
    print(f"{'='*60}")
    print(f"  Target: MinIO cadqstream-violations/schema/")
    print(f"  Trigger: Missing/null required fields → SchemaValidator.filter() returns False")
    print(f"  Sending {n} malformed records to {topic}...")

    sent = 0
    for i in range(n):
        # Vary which required field is missing
        choices = [
            # Missing fare_amount
            {**_gen_base_trip(i, 10, 20, 2.0, 15.0), "fare_amount": None},
            # Missing trip_distance
            {**_gen_base_trip(i, 10, 20, 2.0, 15.0), "trip_distance": None},
            # Missing PULocationID
            {**_gen_base_trip(i, 10, 20, 2.0, 15.0), "PULocationID": None},
            # Missing DOLocationID
            {**_gen_base_trip(i, 10, 20, 2.0, 15.0), "DOLocationID": None},
            # Missing passenger_count
            {**_gen_base_trip(i, 10, 20, 2.0, 15.0), "passenger_count": None},
            # All required fields missing
            {"trip_id": f"l1-test-{i}", "VendorID": 1},
            # Invalid zone (outside 1-263)
            {**_gen_base_trip(i, 300, 20, 2.0, 15.0)},
            # PULocationID = 0 (outside 1-263)
            {**_gen_base_trip(i, 0, 20, 2.0, 15.0)},
        ]
        record = choices[i % len(choices)]
        record["_test_case"] = "L1_SCHEMA_VIOLATION"
        record["_test_seq"] = i

        try:
            future = producer.send(topic, record)
            future.get(timeout=5)
            sent += 1
            print(f"  [{i+1}/{n}] Sent L1 violation: {json.dumps(record, default=str)[:120]}...")
        except KafkaError as e:
            print(f"  [ERROR] Failed to send L1 violation record {i}: {e}")

    print(f"  Sent {sent}/{n} L1 schema violation records")
    return sent


# ─── Test 2: L2 Canary Rule Violations ─────────────────────────────────────────
# Trigger: Records that pass L1 schema validation but violate CanaryRulesValidator
# Expected: has_violation=True → ViolationFilter → MinIO cadqstream-violations/canary/

def send_l2_canary_violations(producer, topic: str, n: int = 10) -> int:
    """Send records that pass L1 but fail L2 canary rules."""
    print(f"\n{'='*60}")
    print("TEST 2: L2 Canary Rule Violations")
    print(f"{'='*60}")
    print(f"  Target: MinIO cadqstream-violations/canary/")
    print(f"  Trigger: Valid schema + business rule violation")
    print(f"  Sending {n} canary-violation records to {topic}...")

    # Use Manhattan zones to ensure neighborhood mapping works
    puloc = random.choice(MANHATTAN_ZONES)
    dolic = random.choice(MANHATTAN_ZONES)
    if dolic == puloc:
        dolic = puloc + 1 if puloc < 43 else puloc - 1

    sent = 0
    for i in range(n):
        # Generate a valid base record
        base = _gen_base_trip(i + 1000, puloc, dolic, 2.0, 15.0, passengers=1)

        # Apply different canary violations
        if i % 7 == 0:
            # Rule 1: Negative fare → negative_fare
            record = {**base, "fare_amount": -25.50, "total_amount": -20.00}
            label = "negative_fare"
        elif i % 7 == 1:
            # Rule 2: Zero distance with positive fare → zero_distance_with_fare
            record = {**base, "trip_distance": 0.0, "fare_amount": 25.00, "total_amount": 30.00}
            label = "zero_distance_with_fare"
        elif i % 7 == 2:
            # Rule 3: Invalid passenger count (0) → invalid_passengers
            record = {**base, "passenger_count": 0.0}
            label = "invalid_passengers"
        elif i % 7 == 3:
            # Rule 3b: Invalid passenger count (> 6) → invalid_passengers
            record = {**base, "passenger_count": 15.0}
            label = "invalid_passengers (>6)"
        elif i % 7 == 4:
            # Rule 4: Invalid payment type → invalid_payment
            record = {**base, "payment_type": 9.0}
            label = "invalid_payment"
        elif i % 7 == 5:
            # Rule 5: Extreme fare (> $1000) → extreme_fare
            record = {**base, "fare_amount": 2500.00, "total_amount": 2510.00}
            label = "extreme_fare"
        else:
            # Rule 7: total_amount < fare_amount → total_less_than_fare
            record = {**base, "fare_amount": 50.00, "total_amount": 30.00}
            label = "total_less_than_fare"

        record["_test_case"] = "L2_CANARY_VIOLATION"
        record["_test_seq"] = i + 1000

        try:
            future = producer.send(topic, record)
            future.get(timeout=5)
            sent += 1
            print(f"  [{i+1}/{n}] Sent canary violation ({label}): "
                  f"fare={record['fare_amount']}, distance={record['trip_distance']}, "
                  f"passengers={record['passenger_count']}")
        except KafkaError as e:
            print(f"  [ERROR] Failed to send canary violation record {i}: {e}")

    print(f"  Sent {sent}/{n} L2 canary violation records")
    return sent


# ─── Test 3: L3 MemStream Anomaly (Global Outlier) ────────────────────────────
# Trigger: Valid records that pass L1+L2 but have extreme contextual values
#          designed to score high on the 34D MemStream feature vector.
# Expected: is_anomaly=True → MinIO cadqstream-anomalies/ + unified Kafka ANOMALY_RECORD

def send_l3_anomaly_outliers(producer, topic: str, n: int = 15) -> int:
    """Send 'The Scammer' + 'Global Outlier' records that pass L1/L2 but score high on MemStream."""
    print(f"\n{'='*60}")
    print("TEST 3: L3 MemStream Anomaly (Global Outlier)")
    print(f"{'='*60}")
    print(f"  Target: MinIO cadqstream-anomalies/scores/")
    print(f"  Trigger: Valid L1/L2 + extreme 34D feature vector → MemStream score > 1.0")
    print(f"  Strategy: Short distance + astronomical fare (fare_per_mile >> normal range)")
    print(f"  Sending {n} anomaly records to {topic}...")

    sent = 0
    for i in range(n):
        puloc = random.choice(MANHATTAN_ZONES)
        dolic = random.choice(MANHATTAN_ZONES)
        if dolic == puloc:
            dolic = puloc + 1 if puloc < 43 else puloc - 1

        # "The Scammer": 0.1 mile but $500 fare
        # This produces:
        #   fare_per_mile ≈ $5000/mile  (normal ≈ $3-8/mile)
        #   fare_per_min  ≈ $3000/min   (normal ≈ $1-3/min)
        #   speed_mph     ≈ 0.4 mph     (normal ≈ 10-20 mph)
        # All ratios are 100-1000x outside normal range → extreme anomaly score
        record = _gen_base_trip(
            counter=i + 2000,
            puloc=puloc,
            dolic=dolic,
            distance=0.1,          # ~100 meters
            fare=500.0,            # $500 for 100m
            passengers=1,
            payment=1,
            pickup_hour=14,         # daytime
            extra=0.0,
            mta_tax=0.0,
            tip=0.0,
            tolls=0.0,
            imp_surcharge=0.0,
            cong_surcharge=0.0,
        )
        record["total_amount"] = 500.0

        # Also try a second variant: passenger_count=15 (impossible)
        # Note: passenger_count=15 would be caught by canary rule (invalid_passengers),
        # so we stick with the fare/distance extreme

        # Third variant: extremely high speed (short time, long distance)
        if i % 3 == 2:
            record = _gen_base_trip(
                counter=i + 2000,
                puloc=puloc,
                dolic=dolic,
                distance=50.0,        # 50 miles
                fare=30.0,             # $30 for 50 miles (low fare for long trip)
                passengers=1,
                payment=1,
                pickup_hour=12,
                extra=0.0,
                mta_tax=0.0,
                tip=0.0,
                tolls=0.0,
                imp_surcharge=0.0,
                cong_surcharge=0.0,
            )
            # This will have very low fare_per_mile (~$0.60/mile) and high speed

        record["_test_case"] = "L3_MEMSTREAM_ANOMALY"
        record["_test_seq"] = i + 2000

        try:
            future = producer.send(topic, record)
            future.get(timeout=5)
            sent += 1
            d = record["trip_distance"]
            f = record["fare_amount"]
            fpm = f / d if d > 0 else 0
            print(f"  [{i+1}/{n}] Sent anomaly outlier: "
                  f"distance={d:.1f}mi, fare=${f:.2f}, fare/mi=${fpm:.1f}")
        except KafkaError as e:
            print(f"  [ERROR] Failed to send anomaly record {i}: {e}")

    print(f"  Sent {sent}/{n} L3 anomaly records")
    return sent


# ─── Interleave with normal records ─────────────────────────────────────────────
# MemStream warmup requires normal records to build memory.
# Send a small batch of normal records before anomalies so ML has context.

def send_normal_warmup_records(producer, topic: str, n: int = 20) -> int:
    """Send normal records for MemStream warmup context."""
    print(f"\n{'='*60}")
    print("WARMUP: Normal Records (for MemStream context)")
    print(f"{'='*60}")
    print(f"  Purpose: Build normal memory buffer for MemStream warmup")
    print(f"  Sending {n} normal records to {topic}...")

    profiles = [
        # name, d_min, d_max, f_min, f_max
        ("short",   0.5,  3.0,  5.0, 20.0),
        ("medium",  2.0, 10.0, 15.0, 60.0),
        ("long",    8.0, 30.0, 40.0, 200.0),
        ("airport", 15.0, 35.0, 50.0, 150.0),
    ]

    sent = 0
    for i in range(n):
        profile = profiles[i % len(profiles)]
        d = round(random.uniform(profile[1], profile[2]), 2)
        f = round(random.uniform(profile[3], profile[4]), 2)
        puloc = random.choice(MANHATTAN_ZONES)
        dolic = random.choice(MANHATTAN_ZONES)
        if dolic == puloc:
            dolic = puloc + 1 if puloc < 43 else puloc - 1

        record = _gen_base_trip(
            counter=i + 5000,
            puloc=puloc,
            dolic=dolic,
            distance=d,
            fare=f,
            passengers=random.choice([1, 2, 3]),
            payment=random.choice([1, 2]),
            pickup_hour=random.randint(6, 22),
        )
        record["_test_case"] = "NORMAL_WARMUP"

        try:
            producer.send(topic, record).get(timeout=5)
            sent += 1
            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{n}] Sent normal record: {profile[0]}")
        except KafkaError as e:
            print(f"  [ERROR] Failed to send normal record {i}: {e}")

    print(f"  Sent {sent}/{n} normal warmup records")
    return sent


# ─── MinIO Verification ─────────────────────────────────────────────────────────

def verify_minio_buckets(client, sleep_after_send: float = 5.0) -> Dict:
    """Poll MinIO buckets after a brief wait for Flink to write files."""
    print(f"\n{'='*60}")
    print(f"VERIFICATION: Checking MinIO buckets after {sleep_after_send}s...")
    print(f"{'='*60}")

    results = {
        "cadqstream-violations/schema": False,
        "cadqstream-violations/canary": False,
        "cadqstream-anomalies/scores": False,
        "cadqstream-raw/taxi_trips_raw": False,
    }

    if client is None:
        print("  [SKIP] MinIO client not available")
        return results

    print(f"  Waiting {sleep_after_send}s for Flink to flush files to MinIO...")
    time.sleep(sleep_after_send)

    buckets_to_check = [
        ("cadqstream-violations", "schema_violations"),
        ("cadqstream-violations", "canary_violations"),
        ("cadqstream-anomalies", "anomaly_scores"),
        ("cadqstream-raw", "taxi_trips_raw"),
    ]

    for bucket, prefix in buckets_to_check:
        key = f"{bucket}/{prefix}"
        try:
            objects = list(client.list_objects(bucket, prefix=prefix + "/", recursive=True))
            count = len(objects)
            results[key] = count > 0
            if count > 0:
                print(f"  [FOUND] {bucket}/{prefix}/ → {count} file(s)")
                for obj in objects[:3]:
                    print(f"          {obj.object_name} ({obj.size} bytes)")
                if count > 3:
                    print(f"          ... and {count - 3} more")
            else:
                print(f"  [EMPTY] {bucket}/{prefix}/ → 0 files")
        except S3Error as e:
            print(f"  [ERROR] {bucket}/{prefix}/: {e}")

    return results


def print_summary(test_results: Dict, verification_results: Dict):
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    print("\nTest Execution:")
    for name, count in test_results.items():
        status = "SENT" if count > 0 else "SKIPPED"
        print(f"  [{status}] {name}: {count} records")

    print("\nMinIO Bucket Verification:")
    for bucket, found in verification_results.items():
        status = "PASS" if found else "EXPECTED (Flushing...)"
        print(f"  [{status}] {bucket}/")

    print("\nExpected Layer Behavior:")
    print("  L1 Schema Violations  → cadqstream-violations/schema/ (SchemaValidator rejects)")
    print("  L2 Canary Violations  → cadqstream-violations/canary/ (CanaryRulesValidator rejects)")
    print("  L3 Anomaly Outliers   → cadqstream-anomalies/scores/ + unified Kafka ANOMALY_RECORD")
    print("  Normal Warmup         → cadqstream-raw/taxi_trips_raw/ (valid → Window)")

    print("\nNext Steps:")
    print("  1. Check cadqstream-metrics:9250/metrics for counter increments")
    print("     - cadqstream_records_violation_total{layer='L1'}")
    print("     - cadqstream_violation_records_total{layer='L2'}")
    print("     - cadqstream_anomalies_ml_total{layer='L2'}")
    print("     - cadqstream_final_decisions_total{layer='L3',decision='ANOMALY'}")
    print("  2. Check ML service:8000/metrics for inference activity")
    print("     - ml_service_inference_requests_total")
    print("  3. Check MinIO console (localhost:9001) for written files")
    print("  4. Run Flink job if not already running:")
    print("     docker compose -f deployment/docker-compose-minimal.yml up -d")
    print("     python src/flink_job_complete.py")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CA-DQStream End-to-End Layer Injection Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full suite (L1 + L2 + L3):
  python e2e_test_injector.py

  # L1/L2 violations only:
  python e2e_test_injector.py --test l1-l2

  # L3 anomaly only:
  python e2e_test_injector.py --test l3

  # Custom Kafka endpoint (host machine):
  python e2e_test_injector.py --bootstrap localhost:29092

  # Custom MinIO endpoint:
  python e2e_test_injector.py --minio http://localhost:9000
        """
    )
    parser.add_argument(
        "--bootstrap", default="kafka:9092",
        help="Kafka bootstrap servers (default: kafka:9092, use localhost:29092 from host)"
    )
    parser.add_argument(
        "--minio", default="localhost:9000",
        help="MinIO endpoint without http:// (default: localhost:9000)"
    )
    parser.add_argument(
        "--minio-user", required=True,
        help="MinIO access key"
    )
    parser.add_argument(
        "--minio-pass", required=True,
        help="MinIO secret key"
    )
    parser.add_argument(
        "--topic", default=RAW_TOPIC,
        help=f"Raw Kafka topic (default: {RAW_TOPIC})"
    )
    parser.add_argument(
        "--test", choices=["all", "l1-l2", "l3", "warmup"], default="all",
        help="Which test to run (default: all)"
    )
    parser.add_argument(
        "--l1-count", type=int, default=5,
        help="Number of L1 schema violation records (default: 5)"
    )
    parser.add_argument(
        "--l2-count", type=int, default=10,
        help="Number of L2 canary violation records (default: 10)"
    )
    parser.add_argument(
        "--l3-count", type=int, default=15,
        help="Number of L3 anomaly records (default: 15)"
    )
    parser.add_argument(
        "--warmup-count", type=int, default=20,
        help="Number of normal warmup records (default: 20)"
    )
    parser.add_argument(
        "--verify-wait", type=float, default=10.0,
        help="Seconds to wait before verifying MinIO buckets (default: 10.0)"
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip MinIO bucket verification"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CA-DQStream End-to-End Layer Injection Test")
    print("=" * 60)
    print(f"  Kafka:       {args.bootstrap}")
    print(f"  MinIO:       {args.minio}")
    print(f"  Raw Topic:   {args.topic}")
    print(f"  Test Mode:   {args.test}")
    print(f"  L1 count:    {args.l1_count}")
    print(f"  L2 count:    {args.l2_count}")
    print(f"  L3 count:    {args.l3_count}")
    print(f"  Warmup count:{args.warmup_count}")

    # Connect to Kafka
    producer = make_producer(args.bootstrap)
    if producer is None:
        print("[FATAL] Kafka producer unavailable. Exiting.")
        sys.exit(1)
    print(f"[OK] Connected to Kafka at {args.bootstrap}")

    # Connect to MinIO
    minio_client = None
    if not args.no_verify:
        minio_client = make_minio_client(args.minio, args.minio_user, args.minio_pass)

    # Track results
    test_results = {}

    # ── Warmup: normal records first ──────────────────────────────────────────
    if args.test in ("all", "warmup", "l3"):
        warmup_count = send_normal_warmup_records(producer, args.topic, args.warmup_count)
        test_results["L3_Warmup (normal records)"] = warmup_count
        time.sleep(1)  # Brief pause between batches

    # ── Test 1: L1 Schema Violations ─────────────────────────────────────────
    if args.test in ("all", "l1-l2"):
        l1_count = send_l1_schema_violations(producer, args.topic, args.l1_count)
        test_results["L1 Schema Violations"] = l1_count
        time.sleep(1)

    # ── Test 2: L2 Canary Rule Violations ────────────────────────────────────
    if args.test in ("all", "l1-l2"):
        l2_count = send_l2_canary_violations(producer, args.topic, args.l2_count)
        test_results["L2 Canary Violations"] = l2_count
        time.sleep(1)

    # ── Test 3: L3 MemStream Anomaly ────────────────────────────────────────
    if args.test in ("all", "l3"):
        l3_count = send_l3_anomaly_outliers(producer, args.topic, args.l3_count)
        test_results["L3 MemStream Anomaly"] = l3_count
        time.sleep(1)

    # ── More warmup after anomalies ──────────────────────────────────────────
    if args.test == "l3":
        # Send more normal records so MemStream can distinguish anomaly from normal
        extra_warmup = send_normal_warmup_records(producer, args.topic, 10)
        test_results["L3 Post-Anomaly Warmup"] = extra_warmup

    # Cleanup
    producer.flush(timeout=10)
    producer.close(timeout=10)
    print("\n[OK] Kafka producer closed.")

    # ── Verify MinIO buckets ──────────────────────────────────────────────────
    verification_results = {
        "cadqstream-violations/schema": False,
        "cadqstream-violations/canary": False,
        "cadqstream-anomalies/scores": False,
        "cadqstream-raw/taxi_trips_raw": False,
    }

    if not args.no_verify:
        verification_results = verify_minio_buckets(minio_client, args.verify_wait)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(test_results, verification_results)


if __name__ == "__main__":
    main()
