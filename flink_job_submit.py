#!/usr/bin/env python3
"""CA-DQStream Flink Job - Full pipeline with PostgreSQL sink."""
import sys
sys.path.insert(0, '/opt/flink/pyflink_extracted')
sys.path.insert(1, '/opt/flink/opt/python/py4j-0.10.9.7-src.zip')
sys.path.insert(2, '/opt/flink/opt/python/cloudpickle-2.2.0-src.zip')

import os, importlib, pickle, json, random, traceback

WORK = "/opt/flink/e2e"
sys.path.insert(0, WORK)
sys.path.insert(0, WORK + "/src")

os.environ['KAFKA_BOOTSTRAP_SERVERS'] = 'kafka:9092'
os.environ['POSTGRES_HOST'] = 'postgres'
os.environ['POSTGRES_PORT'] = '5432'
os.environ['POSTGRES_DB'] = 'dq_pipeline'
os.environ['POSTGRES_USER'] = 'cadqstream'
os.environ['POSTGRES_PASSWORD'] = 'cadqstream123'

print("=" * 70)
print("CA-DQStream - Flink Job with PostgreSQL Sink")
print("=" * 70)
print("Kafka: " + os.environ['KAFKA_BOOTSTRAP_SERVERS'])
print("Postgres: " + os.environ['POSTGRES_HOST'] + ":" + os.environ['POSTGRES_PORT'])

from pyflink.datastream import StreamExecutionEnvironment, CheckpointingMode, MapFunction
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.common import Row

import psycopg2
from functools import lru_cache

# ---- Custom PostgreSQL Sink via psycopg2 ----
@lru_cache(maxsize=4)
def _get_pg_conn():
    try:
        conn = psycopg2.connect(
            host='postgres', port=5432, dbname='dq_pipeline',
            user='cadqstream', password='cadqstream123'
        )
        conn.autocommit = False
        sys.stderr.write("[PG] Connected to PostgreSQL\n")
        sys.stderr.flush()
        return conn
    except Exception as e:
        sys.stderr.write(f"[PG] Connection failed: {e}\n")
        sys.stderr.flush()
        raise


def _safe_get(value, field, default=None):
    if hasattr(value, field):
        return getattr(value, field)
    if isinstance(value, dict):
        return value.get(field, default)
    return default


class AnomalySink(MapFunction):
    """Write anomaly scores to PostgreSQL."""
    _total = 0

    def map(self, value):
        if value is None:
            return value
        try:
            trip_id = str(_safe_get(value, 'trip_id', ''))
            anomaly_score = float(_safe_get(value, 'anomaly_score', 0.5))
            threshold = float(_safe_get(value, 'threshold', 0.5))
            is_anomaly = bool(_safe_get(value, 'is_anomaly', False))
            context_key = str(_safe_get(value, 'context_key', 'unknown'))
            trip_distance = float(_safe_get(value, 'trip_distance', 0.0))
            pu_id = int(_safe_get(value, 'PULocationID', 0) or _safe_get(value, 'pickup_location_id', 0))
            do_id = int(_safe_get(value, 'DOLocationID', 0) or _safe_get(value, 'dropoff_location_id', 0))

            conn = _get_pg_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO anomaly_scores (trip_id, anomaly_score, threshold, is_anomaly, context_key, trip_distance, pickup_location_id, dropoff_location_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (trip_id) DO UPDATE SET
                        anomaly_score = EXCLUDED.anomaly_score,
                        threshold = EXCLUDED.threshold,
                        is_anomaly = EXCLUDED.is_anomaly,
                        context_key = EXCLUDED.context_key
                """, (trip_id, anomaly_score, threshold, is_anomaly, context_key, trip_distance, pu_id, do_id))
            conn.commit()
            self._total += 1
            if self._total % 500 == 0:
                sys.stderr.write(f"[AnomalySink] Written {self._total}\n")
                sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[AnomalySink] ERROR: {e}\n")
            sys.stderr.flush()
            try:
                conn = _get_pg_conn()
                conn.rollback()
            except Exception:
                pass
        return value


class ViolationSink(MapFunction):
    """Write schema violations to PostgreSQL."""
    _total = 0

    def map(self, value):
        if value is None:
            return value
        try:
            trip_id = str(_safe_get(value, 'trip_id', ''))
            violations = _safe_get(value, 'canary_violations', [])
            if not violations:
                return value

            conn = _get_pg_conn()
            with conn.cursor() as cur:
                for v in violations:
                    cur.execute("""
                        INSERT INTO schema_violations (trip_id, violation_type, violation_reason)
                        VALUES (%s, %s, %s)
                    """, (trip_id, 'CANARY_RULE', str(v)))
            conn.commit()
            self._total += 1
            if self._total % 100 == 0:
                sys.stderr.write(f"[ViolationSink] Written {self._total}\n")
                sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[ViolationSink] ERROR: {e}\n")
            sys.stderr.flush()
            try:
                conn = _get_pg_conn()
                conn.rollback()
            except Exception:
                pass
        return value

# ---- Execution Environment ----
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(4)
print("Parallelism: " + str(env.get_parallelism()))

# ---- Kafka Source ----
kafka_props = {
    'bootstrap.servers': os.environ['KAFKA_BOOTSTRAP_SERVERS'],
    'group.id': 'cadqstream-flink-consumer',
    'auto.offset.reset': 'earliest',
}
kafka_source = FlinkKafkaConsumer(
    topics='taxi-nyc-raw',
    deserialization_schema=SimpleStringSchema(),
    properties=kafka_props
)
stream = env.add_source(kafka_source)
print("Kafka source: OK")

# ---- Layer 1: Parse JSON ----
class ParseJson(MapFunction):
    def map(self, value):
        try:
            return json.loads(value)
        except:
            return None

stream = stream.map(ParseJson()).filter(lambda x: x is not None)
print("JSON parsing: OK")

# ---- Layer 1: Add trip_id ----
from operators.key_generator import generate_trip_id
class AddTripId(MapFunction):
    def map(self, record):
        if record is None:
            return None
        record['trip_id'] = generate_trip_id(record)
        return record

stream = stream.map(AddTripId())
print("Trip ID: OK")

# ---- Layer 1: Deduplication ----
from operators.deduplicator import DeduplicatorFunction
dedup_stream = (
    stream
    .key_by(lambda x: x.get('trip_id', ''), key_type=Types.STRING())
    .map(DeduplicatorFunction())
    .filter(lambda x: x is not None)
)
print("Deduplication: OK")

# ---- Layer 1: Schema Validation ----
from operators.schema_validator import SchemaValidator
validator = SchemaValidator()
valid_stream = dedup_stream.filter(validator)
print("Schema validation: OK")

# ---- Layer 2: Canary Rules ----
from operators.canary_rules import CanaryRulesValidator
canary_stream = valid_stream.map(CanaryRulesValidator())
print("Canary rules: OK")

# ---- Layer 2: ML Scoring ----
try:
    features = importlib.import_module("features.vectorizer")
    VecCls = getattr(features, "FeatureVectorizer")
    if_op = importlib.import_module("operators.if_scoring_operator")
    get_ctx = getattr(if_op, "get_context_key")
    _m = pickle.load(open(WORK + "/models/iforest_model.pkl", "rb"))
    _s = pickle.load(open(WORK + "/models/scaler.pkl", "rb"))
    _t = json.load(open(WORK + "/models/context_thresholds_v2.json"))
    _v = VecCls()
    n_buckets = len(_t.get('thresholds', {}))
    print("ML model: " + str(_m.n_estimators) + " trees, " + str(n_buckets) + " context buckets")

    class MLScoring(MapFunction):
        def map(self, r):
            if not r:
                return None
            try:
                f = _v.transform(r)
                fs = _s.transform([f])[0]
                raw = _m.score_samples(fs.reshape(1, -1))[0]
                score = -raw
                ctx = get_ctx(r)
                thr = _t.get("thresholds", {}).get(ctx, _t.get("global_threshold", 0.5))
                return dict(**r, anomaly_score=float(score), threshold=float(thr),
                           is_anomaly=bool(score > thr), context_key=ctx)
            except Exception:
                return dict(**r, anomaly_score=0.5, is_anomaly=False, context_key="err")

    ml_stream = valid_stream.map(MLScoring())
    print("ML scoring: OK")
except Exception as e:
    print("ML scoring skipped: " + str(e))

    class MockML(MapFunction):
        def map(self, r):
            if not r:
                return None
            return dict(**r, anomaly_score=random.uniform(0.2, 0.8), is_anomaly=False, context_key="mock")

    ml_stream = valid_stream.map(MockML())

# ---- PostgreSQL Sinks ----
print("\nSetting up PostgreSQL sinks...")

# Anomaly scores sink
ml_stream.filter(lambda x: x is not None).map(AnomalySink()).filter(lambda x: x is not None)
print("Anomaly scores sink: OK")

# Canary violations sink
canary_stream.filter(lambda x: x and x.get('has_violation')).map(ViolationSink())
print("Canary violations sink: OK")

# Console sinks for monitoring
valid_stream.filter(lambda x: x).print()
ml_stream.filter(lambda x: x and x.get("is_anomaly")).print()

print("\n" + "=" * 70)
print("Starting CA-DQStream Flink Job...")
print("=" * 70)
sys.stdout.flush()

env.execute("CA-DQStream Pipeline v4.0")
