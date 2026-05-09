#!/usr/bin/env python3
"""
CA-DQStream End-to-End Demo.
Submit Flink job and run producer from host machine.
"""
import os
import sys
import time
import signal
import subprocess
import threading
import json
import requests
from datetime import datetime

# Change to project directory
os.chdir('c:/proj/ldt')
sys.path.insert(0, 'c:/proj/ldt')

FLINK_REST = "http://localhost:8081"

def wait_for_flink():
    """Wait for Flink to be ready."""
    print("Waiting for Flink cluster...")
    for i in range(30):
        try:
            resp = requests.get(f"{FLINK_REST}/config", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                print(f"  Flink {data.get('flink-version')} ready!")
                return True
        except:
            pass
        time.sleep(2)
    return False

def check_jobmanagers():
    """Check job manager status."""
    try:
        resp = requests.get(f"{FLINK_REST}/jobmanager/config", timeout=10)
        if resp.status_code == 200:
            config = resp.json()
            print(f"  JobManager config: {len(config)} entries")
            return True
    except Exception as e:
        print(f"  Error: {e}")
    return False

def get_jobs():
    """Get all jobs."""
    try:
        resp = requests.get(f"{FLINK_REST}/jobs", timeout=10)
        if resp.status_code == 200:
            return resp.json().get('jobs', [])
    except:
        pass
    return []

def submit_job_docker():
    """Submit the Flink job inside the JobManager container."""
    print("\nSubmitting Flink job from inside container...")
    submit_script = """
import os, sys, signal
sys.path.insert(0, '/tmp')
sys.path.insert(0, '/tmp/src')
os.environ['KAFKA_BOOTSTRAP_SERVERS'] = 'ldt-kafka:9092'
os.environ['POSTGRES_HOST'] = 'ldt-postgres'
os.environ['POSTGRES_PORT'] = '5432'
os.environ['POSTGRES_DB'] = 'dq_pipeline'
os.environ['POSTGRES_USER'] = 'cadqstream'
os.environ['POSTGRES_PASSWORD'] = 'cadqstream123'
os.environ['PYTHONPATH'] = '/tmp:/tmp/src'

from pyflink.datastream import StreamExecutionEnvironment, MapFunction, CheckpointingMode
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from src.operators.key_generator import generate_trip_id
from src.operators.deduplicator import DeduplicatorFunction
from src.operators.schema_validator import SchemaValidator
from src.operators.canary_rules import CanaryRulesValidator
from src.operators.if_scoring_operator import IFScoringOperator
import json

env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(4)
ck = env.get_checkpoint_config()
ck.set_checkpointing_mode(CheckpointingMode.AT_LEAST_ONCE)
ck.set_checkpoint_interval(60000)

kafka_src = FlinkKafkaConsumer(
    topics='taxi-nyc-raw',
    deserialization_schema=SimpleStringSchema(),
    properties={'bootstrap.servers': 'ldt-kafka:9092', 'group.id': 'cadqstream', 'auto.offset.reset': 'earliest'}
)
stream = env.add_source(kafka_src, name='Kafka')

class PJ(MapFunction):
    def map(self, v):
        try: return json.loads(v)
        except: return None

stream = stream.map(PJ(), output_type=Types.PICKLED_BYTE_ARRAY()).filter(lambda x: x is not None)

class AT(MapFunction):
    def map(self, r):
        if r: r['trip_id'] = generate_trip_id(r)
        return r

stream = stream.map(AT(), output_type=Types.PICKLED_BYTE_ARRAY())
dedup = stream.key_by(lambda x: x.get('trip_id',''), key_type=Types.STRING()).map(DeduplicatorFunction(), output_type=Types.PICKLED_BYTE_ARRAY()).filter(lambda x: x is not None)

val = SchemaValidator()
valid = dedup.filter(val)
canary = valid.map(CanaryRulesValidator(), output_type=Types.PICKLED_BYTE_ARRAY())
canary.print()

env.execute('CA-DQStream Pipeline v1.0')
"""
    # Write the submission script to a temp file
    with open('c:/proj/ldt/temp_submit.py', 'w') as f:
        f.write(submit_script)

    # Copy to container
    os.system('docker cp c:/proj/ldt/temp_submit.py ldt-flink-jobmanager:/tmp/submit.py')

    # Run it inside the container
    cmd = 'docker exec -d ldt-flink-jobmanager bash -c "PYTHONPATH=/tmp:/tmp/src KAFKA_BOOTSTRAP_SERVERS=ldt-kafka:9092 python3 /tmp/submit.py 2>&1 | tee /tmp/flink_job.log"'
    os.system(cmd)
    print("  Job submitted in background (check with: docker logs ldt-flink-jobmanager | tail -50)")
    return True

def main():
    print("=" * 70)
    print("CA-DQStream - End-to-End Demo")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check Flink
    if not wait_for_flink():
        print("Flink not ready!")
        return

    check_jobmanagers()

    # Check current jobs
    jobs = get_jobs()
    print(f"\nCurrent jobs: {len(jobs)}")
    for j in jobs:
        print(f"  {j['id']}: {j['name']} ({j['state']})")

    # Submit job
    print()
    submit_job_docker()

    # Wait for job to start
    print("\nWaiting for job to start...")
    for i in range(30):
        jobs = get_jobs()
        running = [j for j in jobs if j['state'] in ['RUNNING', 'CREATED', 'INITIALIZING']]
        if running:
            print(f"\nJob is {running[0]['state']}!")
            print(f"Job ID: {running[0]['id']}")
            break
        time.sleep(2)

    print("\nCheck job status at: http://localhost:8081/#/job/list")
    print("Check logs with: docker logs ldt-flink-jobmanager")

if __name__ == "__main__":
    main()
