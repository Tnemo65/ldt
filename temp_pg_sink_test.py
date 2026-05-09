#!/usr/bin/env python3
"""Kafka -> PostgreSQL via psycopg2 with proper Row attribute access."""
import sys
sys.path.insert(0, '/opt/flink/pyflink_extracted')
sys.path.insert(1, '/opt/flink/opt/python/py4j-0.10.9.7-src.zip')
sys.path.insert(2, '/opt/flink/opt/python/cloudpickle-2.2.0-src.zip')
import os, json, traceback
os.environ['KAFKA_BOOTSTRAP_SERVERS'] = 'kafka:9092'

from pyflink.datastream import StreamExecutionEnvironment, MapFunction
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common import Row

import psycopg2
from functools import lru_cache

# Module-level connection cache
@lru_cache(maxsize=4)
def _get_pg_conn():
    try:
        conn = psycopg2.connect(
            host='postgres', port=5432, dbname='dq_pipeline',
            user='cadqstream', password='cadqstream123'
        )
        conn.autocommit = False
        sys.stderr.write("[PGSink] Connected to PostgreSQL\n")
        sys.stderr.flush()
        return conn
    except Exception as e:
        sys.stderr.write(f"[PGSink] Connection failed: {e}\n")
        sys.stderr.flush()
        raise


def _safe_get(value, field, default=None):
    """Get value from Row (attribute) or dict."""
    if hasattr(value, field):
        return getattr(value, field)
    if isinstance(value, dict):
        return value.get(field, default)
    return default


class PostgresSink(MapFunction):
    """Per-record PostgreSQL insert using cached connection."""
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
            if self._total % 100 == 0:
                sys.stderr.write(f"[PGSink] Written {self._total} records\n")
                sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[PGSink] ERROR: {e}\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            try:
                conn = _get_pg_conn()
                conn.rollback()
            except Exception:
                pass
        return value


print("Setting up env...")
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(2)

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

class ParseAndScore(MapFunction):
    def map(self, value):
        try:
            data = json.loads(value)
            trip_id = (str(data.get('VendorID', 'X')) + '_' +
                       str(data.get('PULocationID', 0)) + '_' +
                       str(data.get('tpep_pickup_datetime', '')))
            return Row(
                trip_id=trip_id,
                anomaly_score=0.5, threshold=0.5, is_anomaly=False, context_key='test',
                trip_distance=float(data.get('trip_distance', 0.0)),
                PULocationID=int(data.get('PULocationID', 0)),
                DOLocationID=int(data.get('DOLocationID', 0))
            )
        except Exception:
            return None

parsed = stream.map(ParseAndScore()).filter(lambda x: x is not None)
print("Parse: OK")

pg_sink = PostgresSink()
sink_stream = parsed.map(pg_sink)
print("Postgres sink: OK")
sink_stream.filter(lambda r: r is not None).print()

print("Starting...")
sys.stdout.flush()
env.execute("Test Postgres Sink v5")
