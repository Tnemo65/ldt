#!/usr/bin/env python3
"""Minimal test: Kafka -> PostgreSQL via JdbcSink."""
import sys
sys.path.insert(0, '/opt/flink/pyflink_extracted')
sys.path.insert(1, '/opt/flink/opt/python/py4j-0.10.9.7-src.zip')
sys.path.insert(2, '/opt/flink/opt/python/cloudpickle-2.2.0-src.zip')
import os, json
os.environ['KAFKA_BOOTSTRAP_SERVERS'] = 'kafka:9092'

from pyflink.datastream import StreamExecutionEnvironment, MapFunction
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.common import Row
from pyflink.datastream.connectors.jdbc import JdbcSink, JdbcConnectionOptions, JdbcExecutionOptions

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
            return Row(
                trip_id=str(data.get('VendorID', 'X')) + '_' + str(data.get('PULocationID', 0)) + '_' + str(data.get('tpep_pickup_datetime', '')),
                anomaly_score=0.5,
                threshold=0.5,
                is_anomaly=False,
                context_key='test',
                trip_distance=float(data.get('trip_distance', 0.0)),
                pickup_location_id=int(data.get('PULocationID', 0)),
                dropoff_location_id=int(data.get('DOLocationID', 0))
            )
        except Exception as e:
            return None

parsed = stream.map(ParseAndScore()).filter(lambda x: x is not None)
print("Parse: OK")

conn_opts = (
    JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
    .with_url("jdbc:postgresql://postgres:5432/dq_pipeline")
    .with_driver_name("org.postgresql.Driver")
    .with_user_name("cadqstream")
    .with_password("cadqstream123")
    .build()
)
exec_opts = (
    JdbcExecutionOptions.builder()
    .with_batch_size(10)
    .with_batch_interval_ms(5000)
    .with_max_retries(3)
    .build()
)

anomaly_type_info = Types.ROW_NAMED(
    ['trip_id', 'anomaly_score', 'threshold', 'is_anomaly', 'context_key', 'trip_distance', 'pickup_location_id', 'dropoff_location_id'],
    [Types.STRING(), Types.DOUBLE(), Types.DOUBLE(), Types.BOOLEAN(), Types.STRING(), Types.DOUBLE(), Types.INT(), Types.INT()]
)

insert_sql = """
INSERT INTO anomaly_scores (trip_id, anomaly_score, threshold, is_anomaly, context_key, trip_distance, pickup_location_id, dropoff_location_id)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (trip_id) DO UPDATE SET
    anomaly_score = EXCLUDED.anomaly_score,
    threshold = EXCLUDED.threshold,
    is_anomaly = EXCLUDED.is_anomaly
"""

anomaly_sink = JdbcSink.sink(insert_sql, anomaly_type_info, conn_opts, exec_opts)
parsed.add_sink(anomaly_sink)
print("JDBC sink: OK")

# Also print to see data flowing
parsed.filter(lambda r: r is not None).print()

print("Starting...")
sys.stdout.flush()
env.execute("Test JDBC Sink")
