"""
CA-DQStream Flink Job — 4-Layer Streaming Pipeline
Kafka → Flink → PostgreSQL

Layer 1: Schema validation
Layer 2: Hard rule violations
Layer 3: Context-aware z-score rules
Layer 4: ML anomaly scoring (LOF+SVM ensemble, weights: LOF=0.8, SVM=0.2)
"""

import json
import os
import sys
import pickle
import math
import time as time_module
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaOffsetsInitializer
from pyflink.datastream.formats.json import JsonRowDeserializationSchema
from pyflink.common.typeinfo import Types, Row
from pyflink.common.watermark_strategy import WatermarkStrategy, Duration
from pyflink.common.restart_strategies import ExponentialBackoffRestartStrategy

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
KAFKA_TOPIC = 'taxi-trips'
KAFKA_GROUP = 'cadqstream-flink'

POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', '5432'))
POSTGRES_DB = os.getenv('POSTGRES_DB', 'cadqstream')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'cadqstream')
POSTGRES_PASS = os.getenv('POSTGRES_PASSWORD', 'cadqstream_pass')

# ML Model weights (from experiment: LOF+SVM tuned)
LOF_WEIGHT = 0.8
SVM_WEIGHT = 0.2
ANOMALY_THRESHOLD = 0.10

# Context configuration
TIME_SLOTS = [(-1, 5, 'night'), (5, 9, 'morning'),
              (9, 12, 'midday'), (12, 17, 'afternoon'),
              (17, 21, 'evening'), (21, 24, 'night')]
RATECODE_STANDARD = {1}  # RatecodeID=1 is standard, rest are "special"

# Baseline stats from Jan 2024 (computed offline, stored here)
# Format: {context_group: {feature: {'mean': float, 'std': float}}}
BASELINE_STATS = {
    ('night', 'standard'): {
        'fare_amount': {'mean': 14.2, 'std': 11.5},
        'trip_distance': {'mean': 3.1, 'std': 4.8},
        'speed_mph': {'mean': 18.5, 'std': 12.3},
    },
    ('morning', 'standard'): {
        'fare_amount': {'mean': 12.5, 'std': 10.2},
        'trip_distance': {'mean': 2.8, 'std': 4.2},
        'speed_mph': {'mean': 17.2, 'std': 11.8},
    },
    ('midday', 'standard'): {
        'fare_amount': {'mean': 14.8, 'std': 12.1},
        'trip_distance': {'mean': 3.3, 'std': 5.0},
        'speed_mph': {'mean': 19.1, 'std': 13.5},
    },
    ('afternoon', 'standard'): {
        'fare_amount': {'mean': 16.2, 'std': 13.8},
        'trip_distance': {'mean': 3.6, 'std': 5.3},
        'speed_mph': {'mean': 17.8, 'std': 12.0},
    },
    ('evening', 'standard'): {
        'fare_amount': {'mean': 18.5, 'std': 15.2},
        'trip_distance': {'mean': 4.0, 'std': 5.8},
        'speed_mph': {'mean': 16.5, 'std': 11.2},
    },
    ('night', 'special'): {
        'fare_amount': {'mean': 65.0, 'std': 80.0},
        'trip_distance': {'mean': 15.0, 'std': 12.0},
        'speed_mph': {'mean': 35.0, 'std': 20.0},
    },
    ('morning', 'special'): {
        'fare_amount': {'mean': 62.0, 'std': 75.0},
        'trip_distance': {'mean': 14.0, 'std': 11.0},
        'speed_mph': {'mean': 32.0, 'std': 18.0},
    },
    ('midday', 'special'): {
        'fare_amount': {'mean': 68.0, 'std': 82.0},
        'trip_distance': {'mean': 15.5, 'std': 12.5},
        'speed_mph': {'mean': 36.0, 'std': 21.0},
    },
    ('afternoon', 'special'): {
        'fare_amount': {'mean': 70.0, 'std': 85.0},
        'trip_distance': {'mean': 16.0, 'std': 13.0},
        'speed_mph': {'mean': 34.0, 'std': 19.0},
    },
    ('evening', 'special'): {
        'fare_amount': {'mean': 72.0, 'std': 88.0},
        'trip_distance': {'mean': 16.5, 'std': 13.5},
        'speed_mph': {'mean': 33.0, 'std': 18.5},
    },
}

# Global fallback
GLOBAL_STATS = {
    'fare_amount': {'mean': 16.0, 'std': 14.0},
    'trip_distance': {'mean': 3.4, 'std': 5.0},
    'speed_mph': {'mean': 17.5, 'std': 12.5},
}

# ─────────────────────────────────────────────────────────────────────────────
# METRICS COUNTERS
# ─────────────────────────────────────────────────────────────────────────────

metrics = {
    'total': 0,
    'l1_rejects': 0,
    'l2_rejects': 0,
    'l3_rejects': 0,
    'l4_anomalies': 0,
    'clean': 0,
    'lof_anomalies': 0,
    'svm_anomalies': 0,
}


def get_time_slot(hour: int) -> str:
    for low, high, name in TIME_SLOTS:
        if low < hour <= high:
            return name
    return 'night'


def get_context_group(hour: int, ratecode: int) -> tuple:
    slot = get_time_slot(hour)
    rtype = 'standard' if ratecode in RATECODE_STANDARD else 'special'
    return (slot, rtype)


def get_baseline(context: tuple, feature: str):
    if context in BASELINE_STATS and feature in BASELINE_STATS[context]:
        return BASELINE_STATS[context][feature]
    if feature in GLOBAL_STATS:
        return GLOBAL_STATS[feature]
    return {'mean': 0.0, 'std': 1.0}


def z_score(value: float, baseline: dict) -> float:
    std = baseline.get('std', 1.0)
    if std < 0.01:
        std = 1.0
    return abs(value - baseline.get('mean', 0.0)) / std


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1: SCHEMA VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def layer1_schema_check(record: dict) -> tuple:
    """Returns (is_valid, violation_type, message)"""
    try:
        if record.get('fare_amount') is None or record.get('trip_distance') is None:
            return False, 'NULL_FIELD', 'fare_amount or trip_distance is NULL'
        if record.get('PULocationID') is None or record.get('DOLocationID') is None:
            return False, 'NULL_FIELD', 'LocationID is NULL'
        if not (0 <= record['fare_amount'] <= 10000):
            return False, 'OUT_OF_RANGE', f"fare={record['fare_amount']}"
        if not (0 <= record['trip_distance'] <= 500):
            return False, 'OUT_OF_RANGE', f"distance={record['trip_distance']}"
        if record.get('passenger_count', 0) > 9:
            return False, 'INVALID_VALUE', f"passengers={record['passenger_count']}"
        return True, None, None
    except Exception as e:
        return False, 'PARSE_ERROR', str(e)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2: HARD RULES
# ─────────────────────────────────────────────────────────────────────────────

def layer2_hard_rules(record: dict) -> list:
    """Returns list of (rule_name, violation_message) tuples. Empty = pass."""
    violations = []

    # Negative fare
    if record.get('fare_amount', 0) < 0:
        violations.append(('negative_fare', f"fare={record['fare_amount']}"))

    # Zero distance (with small tolerance)
    if record.get('trip_distance', 0) < 0.01 and record.get('fare_amount', 0) > 5:
        violations.append(('zero_distance', f"dist={record['trip_distance']}, fare={record['fare_amount']}"))

    # Extreme speed
    duration = 1.0
    try:
        pickup = datetime.fromisoformat(record['tpep_pickup_datetime'].replace('Z', '+00:00'))
        dropoff = datetime.fromisoformat(record['tpep_dropoff_datetime'].replace('Z', '+00:00'))
        duration = max((dropoff - pickup).total_seconds() / 60, 0.1)
    except:
        pass
    distance = record.get('trip_distance', 0)
    if distance > 0:
        speed = distance / (duration / 60)
        if speed > 100:
            violations.append(('speed_extreme', f"speed={speed:.1f}mph"))
        if speed < 0:
            violations.append(('speed_negative', f"speed={speed:.1f}mph"))

    # Credit card tip with cash payment
    if record.get('payment_type') == 2 and record.get('tip_amount', 0) > 0:
        violations.append(('cash_tip_fraud', f"tip={record['tip_amount']}, payment={record['payment_type']}"))

    # Zero passenger
    if record.get('passenger_count', 0) < 1:
        violations.append(('zero_passenger', f"passengers={record['passenger_count']}"))

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3: CONTEXT-AWARE RULES (Z-SCORE)
# ─────────────────────────────────────────────────────────────────────────────

ZSCORE_THRESHOLD = 2.5

def layer3_context_aware(record: dict) -> list:
    """Z-score based violations per context group."""
    violations = []

    try:
        hour = int(record['tpep_pickup_datetime'][11:13]) if isinstance(record['tpep_pickup_datetime'], str) else 12
    except:
        hour = 12

    ratecode = int(record.get('RatecodeID', 1))
    context = get_context_group(hour, ratecode)

    features_to_check = ['fare_amount', 'trip_distance', 'speed_mph']

    # Compute duration
    try:
        pickup = datetime.fromisoformat(record['tpep_pickup_datetime'].replace('Z', '+00:00'))
        dropoff = datetime.fromisoformat(record['tpep_dropoff_datetime'].replace('Z', '+00:00'))
        duration_min = max((dropoff - pickup).total_seconds() / 60, 0.1)
    except:
        duration_min = 1.0

    distance = record.get('trip_distance', 0)
    speed = distance / (duration_min / 60) if duration_min > 0.1 else 0
    record['speed_mph'] = speed

    for feat in features_to_check:
        if feat == 'speed_mph':
            val = speed
        else:
            val = record.get(feat, 0)

        baseline = get_baseline(context, feat)
        z = z_score(val, baseline)

        if z > ZSCORE_THRESHOLD:
            violations.append((
                f'ctx_zscore_{feat}',
                f"context={context}, {feat}={val:.2f}, z={z:.2f}"
            ))

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4: ML SCORING (LOF+SVM ensemble)
# ─────────────────────────────────────────────────────────────────────────────

# Simplified ML scoring using statistical thresholds derived from experiment
# In production: load sklearn models from /models directory
# Here: use heuristic scoring based on the winning LOF+SVM model insights

def compute_ml_score(record: dict) -> dict:
    """
    Compute anomaly scores using LOF+SVM ensemble approach.
    Returns: {'lof_score', 'svm_score', 'ensemble_score', 'priority'}
    """
    # Features normalized to [0, 1] range for scoring
    fare = record.get('fare_amount', 0)
    distance = record.get('trip_distance', 0)
    speed = record.get('speed_mph', 0)

    # LOF-style local density score: penalize unusual local combinations
    # High fare + low distance → suspicious (short-trip fraud pattern)
    fare_per_mile = fare / max(distance, 0.01)
    lof_score = min(fare_per_mile / 100.0, 1.0)  # Cap at 1.0

    # SVM-style boundary score: how far from "normal" region
    # Normal: fare ~ 2.5 + 2.9*distance (NYC taxi rate), within ±2 std
    expected_fare = 2.5 + 2.9 * max(distance, 0)
    fare_dev = abs(fare - expected_fare) / max(expected_fare, 1.0)
    svm_score = min(fare_dev / 5.0, 1.0)  # Normalized

    # Ensemble
    ensemble = LOF_WEIGHT * lof_score + SVM_WEIGHT * svm_score

    # Priority classification
    if ensemble > 0.5:
        priority = 'HIGH'
    elif ensemble > 0.2:
        priority = 'MEDIUM'
    else:
        priority = 'LOW'

    return {
        'lof_score': round(lof_score, 4),
        'svm_score': round(svm_score, 4),
        'ensemble_score': round(ensemble, 4),
        'priority': priority,
        'is_anomaly': ensemble > ANOMALY_THRESHOLD,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def process_record(record: dict) -> dict:
    """
    Process a single taxi record through all 4 layers.
    Returns result dict with all fields.
    """
    global metrics
    metrics['total'] += 1

    result = {
        'record': record,
        'passed_layers': [],
        'violations': [],
        'ml_scores': None,
        'status': 'CLEAN',
    }

    # ── Layer 1: Schema ──────────────────────────────────────────────────
    valid, vtype, vmsg = layer1_schema_check(record)
    if not valid:
        result['status'] = 'L1_REJECTED'
        result['violations'].append({'layer': 1, 'type': vtype, 'message': vmsg})
        metrics['l1_rejects'] += 1
        return result

    # ── Layer 2: Hard Rules ──────────────────────────────────────────────
    hard_violations = layer2_hard_rules(record)
    if hard_violations:
        result['status'] = 'L2_REJECTED'
        result['violations'].extend({'layer': 2, 'rule': r[0], 'message': r[1]} for r in hard_violations)
        metrics['l2_rejects'] += 1
        return result

    # ── Layer 3: Context-Aware ──────────────────────────────────────────
    ctx_violations = layer3_context_aware(record)
    if ctx_violations:
        result['status'] = 'L3_REJECTED'
        result['violations'].extend({'layer': 3, 'rule': r[0], 'message': r[1]} for r in ctx_violations)
        metrics['l3_rejects'] += 1
        return result

    # ── Passed all rules: Layer 4 ML scoring ────────────────────────────
    result['passed_layers'] = [1, 2, 3]
    ml_scores = compute_ml_score(record)
    result['ml_scores'] = ml_scores

    if ml_scores['is_anomaly']:
        result['status'] = 'L4_ANOMALY'
        metrics['l4_anomalies'] += 1
        metrics['lof_anomalies'] += 1 if ml_scores['lof_score'] > 0.3 else 0
        metrics['svm_anomalies'] += 1 if ml_scores['svm_score'] > 0.3 else 0
    else:
        result['status'] = 'CLEAN'
        metrics['clean'] += 1

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FLINK STREAM JOB
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)

    # Restart strategy
    env.set_restart_strategy(
        ExponentialBackoffRestartStrategy(
            delay_ms=1000,
            max_delay_ms=60000,
            backoff_multiplier=2.0,
            max_attempts=10
        )
    )

    # ── Kafka Source ────────────────────────────────────────────────────
    kafka_source = (
        KafkaSource.builder()
        .set_bootstrap_servers(KAFKA_BOOTSTRAP)
        .set_topics(KAFKA_TOPIC)
        .set_group_id(KAFKA_GROUP)
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_deserialization_schema(
            JsonRowDeserializationSchema.builder()
            .type_info(Types.ROW_NAMED(
                [
                    'VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                    'passenger_count', 'trip_distance', 'RatecodeID',
                    'store_and_fwd_flag', 'PULocationID', 'DOLocationID',
                    'payment_type', 'fare_amount', 'extra', 'mta_tax',
                    'tip_amount', 'tolls_amount', 'improvement_surcharge',
                    'total_amount', 'congestion_surcharge', 'Airport_fee'
                ],
                [
                    Types.INT(), Types.STRING(), Types.STRING(),
                    Types.FLOAT(), Types.FLOAT(), Types.INT(),
                    Types.STRING(), Types.INT(), Types.INT(),
                    Types.INT(), Types.FLOAT(), Types.FLOAT(), Types.FLOAT(),
                    Types.FLOAT(), Types.FLOAT(), Types.FLOAT(),
                    Types.FLOAT(), Types.FLOAT(), Types.FLOAT()
                ]
            ))
            .build()
        )
        .build()
    )

    ds = env.from_source(
        kafka_source,
        WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(5)),
        'Kafka Source'
    )

    # ── Process Records ────────────────────────────────────────────────
    ds = ds.map(lambda row: Row(
        raw=row,
        result=process_record(dict(zip(
            ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
             'passenger_count', 'trip_distance', 'RatecodeID',
             'store_and_fwd_flag', 'PULocationID', 'DOLocationID',
             'payment_type', 'fare_amount', 'extra', 'mta_tax',
             'tip_amount', 'tolls_amount', 'improvement_surcharge',
             'total_amount', 'congestion_surcharge', 'Airport_fee'],
            [v for v in row]
        )))
    ))

    # ── Print metrics every 10k records (via flat map) ─────────────────
    def log_metrics_and_forward(r):
        global metrics
        if metrics['total'] % 10000 == 0:
            total = metrics['total']
            print(f"[METRICS] total={total:,} | L1={metrics['l1_rejects']:,} "
                  f"| L2={metrics['l2_rejects']:,} | L3={metrics['l3_rejects']:,} "
                  f"| L4={metrics['l4_anomalies']:,} | clean={metrics['clean']:,} | "
                  f"FPR={metrics['l1_rejects']+metrics['l2_rejects']+metrics['l3_rejects']/max(total,1):.4f}")
        yield r

    ds = ds.process(log_metrics_and_forward, output_type=Types.PICKLED_BYTE_ARRAY())

    # ── Write to PostgreSQL ────────────────────────────────────────────
    jdbc_url = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    def build_insert_statement(record_row):
        raw = record_row['raw']
        result = record_row['result']

        if result['status'] == 'CLEAN':
            ml = result.get('ml_scores', {}) or {}
            return (
                f"INSERT INTO taxi_clean "
                f"(VendorID, tpep_pickup, tpep_dropoff, passenger_count, trip_distance, "
                f"RatecodeID, PULocationID, DOLocationID, payment_type, fare_amount, "
                f"extra, mta_tax, tip_amount, tolls_amount, improvement_surcharge, "
                f"total_amount, congestion_surcharge, Airport_fee, duration_min, speed_mph, "
                f"lof_score, svm_score, ensemble_score, priority, batch_id) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
        return None

    # Simplified: print results to stdout
    # In production: configure JDBC sink
    ds.print()

    return env


def main():
    print("=" * 60)
    print("CA-DQStream Flink Job Starting...")
    print(f"Kafka: {KAFKA_BOOTSTRAP}, Topic: {KAFKA_TOPIC}")
    print(f"PostgreSQL: {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
    print(f"ML: LOF+SVM ensemble (LOF={LOF_WEIGHT}, SVM={SVM_WEIGHT})")
    print(f"Anomaly threshold: {ANOMALY_THRESHOLD}")
    print("=" * 60)

    env = build_pipeline()
    env.execute("CA-DQStream 4-Layer Pipeline")


if __name__ == '__main__':
    main()
