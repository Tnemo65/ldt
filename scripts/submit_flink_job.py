#!/usr/bin/env python3
"""
CA-DQStream Flink Job Submission.
Runs inside the JobManager container and submits the job to the cluster.
"""
import os
import sys
import signal
import time

# Setup paths
sys.path.insert(0, '/tmp')
sys.path.insert(0, '/tmp/src')
sys.path.insert(0, '/tmp')

# Environment variables for Kafka, etc.
os.environ['KAFKA_BOOTSTRAP_SERVERS'] = 'ldt-kafka:9092'
os.environ['PYTHONPATH'] = '/tmp:/tmp/src'
os.environ['PYTHONPATH'] = '/tmp:/tmp/src'

print("=" * 70)
print("CA-DQStream - Flink Job Submission")
print("=" * 70)
print(f"Working directory: {os.getcwd()}")
print(f"Python: {sys.version}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")
print(f"KAFKA: {os.environ['KAFKA_BOOTSTRAP_SERVERS']}")
print("=" * 70)

# Import Flink
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream import MapFunction

print("\n[1] Setting up StreamExecutionEnvironment...")
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(4)
print(f"    Parallelism: {env.get_parallelism()}")

# Checkpointing disabled for demo (uncomment to enable with proper state backend)
# ck_config = env.get_checkpoint_config()
# ck_config.set_checkpointing_mode(CheckpointingMode.AT_LEAST_ONCE)
# ck_config.set_checkpoint_interval(60000)
# ck_config.set_checkpoint_timeout(120000)
print("\n[2] Checkpointing: DISABLED (add MinIO/S3 backend for production)")

# Kafka Source
print("\n[3] Setting up Kafka source (taxi-nyc-raw)...")
kafka_props = {
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'cadqstream-flink-consumer',
    'auto.offset.reset': 'earliest',
}
kafka_source = FlinkKafkaConsumer(
    topics='taxi-nyc-raw',
    deserialization_schema=SimpleStringSchema(),
    properties=kafka_props
)
stream = env.add_source(kafka_source)
print("    Kafka source connected")

# Layer 1: Parse JSON
import json
print("\n[4] Layer 1: JSON parsing...")
class ParseJson(MapFunction):
    def map(self, value):
        try:
            return json.loads(value)
        except:
            return None

stream = stream.map(ParseJson(), output_type=Types.PICKLED_BYTE_ARRAY()).filter(lambda x: x is not None)
print("    JSON parsing: OK")

# Layer 1: Add trip_id
print("\n[5] Layer 1: Trip ID generation (MurmurHash3)...")
from src.operators.key_generator import generate_trip_id
class AddTripId(MapFunction):
    def map(self, record):
        if record is None:
            return None
        record['trip_id'] = generate_trip_id(record)
        return record

stream = stream.map(AddTripId(), output_type=Types.PICKLED_BYTE_ARRAY())
print("    Trip ID: OK")

# Layer 1: Deduplication
print("\n[6] Layer 1: Deduplication (7-day TTL)...")
from src.operators.deduplicator import DeduplicatorFunction
dedup_stream = (
    stream
    .key_by(lambda x: x.get('trip_id', ''), key_type=Types.STRING())
    .map(DeduplicatorFunction(), output_type=Types.PICKLED_BYTE_ARRAY())
    .filter(lambda x: x is not None)
)
print("    Deduplication: OK")

# Layer 1: Schema Validation
print("\n[7] Layer 1: Schema Validation (19 fields)...")
from src.operators.schema_validator import SchemaValidator
validator = SchemaValidator()
valid_stream = dedup_stream.filter(validator)
violation_stream = dedup_stream.filter(lambda x: not validator.filter(x))
print("    Schema validation: OK")

# Layer 2: Canary (Rule-based)
print("\n[8] Layer 2: Canary Branch (Business Rules)...")
from src.operators.canary_rules import CanaryRulesValidator
canary_stream = valid_stream.map(CanaryRulesValidator(), output_type=Types.PICKLED_BYTE_ARRAY())
print("    Canary rules: OK")

# Layer 2: Complex (ML Scoring)
print("\n[9] Layer 2: Complex Branch (IsolationForest Scoring)...")
from src.operators.if_scoring_operator import IFScoringOperator
ml_stream = valid_stream.map(IFScoringOperator(), output_type=Types.PICKLED_BYTE_ARRAY())
print("    ML scoring: OK")

# Print streams for monitoring
print("\n[10] Setting up output sinks...")
canary_stream.filter(lambda x: x is not None).print()
ml_stream.filter(lambda x: x is not None).filter(lambda x: x.get('is_anomaly', False)).print()
print("    Sinks: Console print (valid + anomalies)")

print("\n" + "=" * 70)
print("Starting CA-DQStream Flink Job...")
print("=" * 70)

# Execute (this blocks)
try:
    env.execute("CA-DQStream Pipeline v1.0")
except Exception as e:
    print(f"Job error: {e}")
    import traceback
    traceback.print_exc()
