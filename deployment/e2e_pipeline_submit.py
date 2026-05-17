#!/usr/bin/env python3
"""
CA-DQStream E2E Pipeline Submission.
Runs inside the Flink JobManager container with full environment setup.
"""
import os
import sys
import time
import signal
import traceback

# Setup Python paths
sys.path.insert(0, '/opt/flink/e2e')
sys.path.insert(0, '/opt/flink/e2e/src')
sys.path.insert(0, '/tmp')
sys.path.insert(0, '/tmp/src')

# Environment
os.environ['KAFKA_BOOTSTRAP_SERVERS'] = 'kafka:9092'
os.environ['PYTHONPATH'] = '/opt/flink/e2e:/opt/flink/e2e/src:/tmp:/tmp/src'
os.environ['PYFLINK_SESSION_JOB_ENABLED'] = 'true'

WORK = "/opt/flink/e2e"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    sys.stdout.flush()

log("=" * 70)
log("CA-DQStream E2E Pipeline Submission")
log("=" * 70)
log(f"Python: {sys.version.split()[0]}")
log(f"KAFKA: {os.environ['KAFKA_BOOTSTRAP_SERVERS']}")
log(f"WORK: {WORK}")
log("=" * 70)

# Import PyFlink
log("\n[1] Importing PyFlink...")
from pyflink.datastream import StreamExecutionEnvironment, CheckpointingMode, MapFunction
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
log("  PyFlink imported successfully")

# Import operators
log("\n[2] Importing operators...")
import json

from src.operators.key_generator import generate_trip_id
log("  - key_generator: OK")

from src.operators.deduplicator import DeduplicatorFunction
log("  - deduplicator: OK")

from src.operators.schema_validator import SchemaValidator
log("  - schema_validator: OK")

from src.operators.canary_rules_operator import CanaryRulesValidator, ViolationFilter
log("  - canary_rules: OK")

from src.features.vectorizer import FeatureVectorizer
log("  - vectorizer: OK")

# Create environment
log("\n[4] Setting up StreamExecutionEnvironment...")
env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(4)
env.get_checkpoint_config().set_checkpointing_mode(CheckpointingMode.AT_LEAST_ONCE)
log("  Environment configured (parallelism=4, AT_LEAST_ONCE)")

# Watermark strategy for event-time processing
log("\n[4b] Setting up Watermark Strategy...")
from src.operators.watermark_assigner import create_watermark_strategy
watermark_strategy = create_watermark_strategy()
log("  Watermark Strategy: bounded_out_of_orderness(10s) + idleness(30s)")

# Kafka source
log("\n[5] Setting up Kafka source...")
kafka_props = {
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'cadqstream-e2e-full',
    'auto.offset.reset': 'earliest',
}
kafka_source = FlinkKafkaConsumer(
    topics='taxi-nyc-raw',
    deserialization_schema=SimpleStringSchema(),
    properties=kafka_props
)
log("  Kafka source: taxi-nyc-raw (kafka:9092)")

# Layer 1: Parse JSON
log("\n[6] Layer 1: Baseline Validation...")

class ParseJson(MapFunction):
    _cnt = 0
    def map(self, value):
        ParseJson._cnt += 1
        if ParseJson._cnt % 50000 == 0:
            log(f"  [Parse] {ParseJson._cnt} records processed")
        try:
            return json.loads(value)
        except:
            return None

stream = env.add_source(kafka_source)
stream = stream.assign_timestamps_and_watermarks(watermark_strategy)
log("  - Watermark Strategy applied: OK")

stream = stream.map(ParseJson(), output_type=Types.PICKLED_BYTE_ARRAY()).filter(lambda x: x is not None)
log("  - JSON parsing: OK")

# Add trip_id
class AddTripId(MapFunction):
    _cnt = 0
    def map(self, record):
        if record is None:
            return None
        AddTripId._cnt += 1
        record['trip_id'] = generate_trip_id(record)
        return record

stream = stream.map(AddTripId(), output_type=Types.PICKLED_BYTE_ARRAY())
log("  - Trip ID generation (MurmurHash3): OK")

# Deduplication
dedup_stream = (
    stream
    .key_by(lambda x: x.get('trip_id', ''), key_type=Types.STRING())
    .map(DeduplicatorFunction(), output_type=Types.PICKLED_BYTE_ARRAY())
    .filter(lambda x: x is not None)
)
log("  - Deduplication (7-day TTL): OK")

# Schema validation
validator = SchemaValidator()
valid_stream = dedup_stream.filter(validator)
violation_stream = dedup_stream.filter(lambda x: not validator.filter(x))
log("  - Schema validation (19 fields): OK")

# Layer 2: Dual-Branch
log("\n[7] Layer 2: Dual-Branch Processing...")

# Canary Branch
canary_stream = valid_stream.map(CanaryRulesValidator(), output_type=Types.PICKLED_BYTE_ARRAY())
log("  - Canary Branch (7 rules): OK")

# Layer 3: Voting Ensemble
log("\n[8] Layer 3: Voting Ensemble + MetaAggregator...")

class VotingFunction(MapFunction):
    _cnt = 0
    def map(self, record):
        if record is None:
            return None
        VotingFunction._cnt += 1
        canary_violates = record.get('canary_violates', False)
        if canary_violates:
            record['final_verdict'] = 'VIOLATION'
            record['voter'] = 'canary'
        else:
            record['final_verdict'] = 'CLEAN'
            record['voter'] = 'none'
        if VotingFunction._cnt % 50000 == 0:
            log(f"  [Voting] {VotingFunction._cnt} records: {record['final_verdict']}")
        return record

voting_stream = valid_stream.map(VotingFunction(), output_type=Types.PICKLED_BYTE_ARRAY())
log("  - Voting Ensemble: OK")

# Layer 4: Intelligent Evolution Controller (IEC)
# NOTE: Full Layer 4 requires MetaAggregator with windowed meta-metrics output.
# This placeholder demonstrates IEC integration. In production, iec_stream receives
# meta-metrics from MetaAggregator.windowed_all() with 1-minute tumbling windows.
log("\n[8b] Layer 4: IEC (placeholder - requires MetaAggregator windowed output)")

# Import IEC components
from src.operators.iec_operator import IECOperator
log("  - IEC Operator import: OK")

# NOTE: In full implementation, IECOperator processes windowed meta-metrics:
#   meta_stream = voting_stream.window(...).apply(MetaAggregator())
#   iec_stream = meta_stream.map(IECOperator())
#
# Current e2e_pipeline uses stateless VotingFunction without windowing.
# For production: Use src/operators/meta_aggregator.py with 1-min TumblingEventTimeWindow.
#
# Layer 4 components available in src/operators/iec_operator.py:
#   - MultiInstanceADWIN (36 instances: 6 neighborhoods × 6 metrics)
#   - DriftAggregator with 4 strategies: do_nothing, adjust_threshold, retrain_model, switch_model
#   - METER hypernetwork integration (sklearn MLPClassifier)
#   - Action Replay Worker with exponential backoff retry

# Placeholder: IE controller would be applied to meta-metrics stream in production
log("  - Layer 4 (IEC): Requires MetaAggregator with 1-min window (see src/operators/iec_operator.py)")

# Outputs
log("\n[9] Setting up outputs...")

# Print valid records (every 5000th)
class PrintSample(MapFunction):
    _cnt = 0
    def map(self, r):
        PrintSample._cnt += 1
        if PrintSample._cnt % 5000 == 0 and r:
            verdict = r.get('final_verdict', 'N/A')
            score = r.get('anomaly_score', 0)
            log(f"  >>> [{verdict}] score={score:.4f} trip_id={r.get('trip_id','')[:16]}...")
        return r

valid_stream_sample = valid_stream.map(PrintSample(), output_type=Types.PICKLED_BYTE_ARRAY())
valid_stream_sample.filter(lambda x: x).print()
log("  - Valid records: print()")

violation_stream.filter(lambda x: x).print()
log("  - Schema violations: print()")

canary_stream.filter(ViolationFilter()).print()
log("  - Canary violations: print()")

# Voting
voting_stream.filter(lambda x: x and x.get('final_verdict') != 'CLEAN').print()
log("  - Voting decisions (violations+anomalies): print()")

log("\n" + "=" * 70)
log("STARTING CA-DQSTREAM E2E PIPELINE")
log("=" * 70)
log(f"Job: CA-DQStream E2E Full Pipeline")
log(f"Parallelism: {env.get_parallelism()}")
log(f"Checkpointing: AT_LEAST_ONCE")
log(f"Kafka: taxi-nyc-raw @ kafka:9092")
log("=" * 70)

try:
    result = env.execute("CA-DQStream E2E Full Pipeline")
    log(f"\nJob finished: {result}")
except Exception as e:
    log(f"\nJob error: {e}")
    traceback.print_exc()
