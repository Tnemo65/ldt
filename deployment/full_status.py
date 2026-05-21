import subprocess
import json
import os

# Load env vars from .env if present
try:
    with open(os.path.join(os.path.dirname(__file__), '../.env')) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k, v)
except Exception:
    pass

JOB_ID = "817e44e0a9ddd87608c86dfc80f8bccf"

# 1. Check Kafka topic consumer lag
script = '''
from kafka import KafkaConsumer
from kafka.admin import KafkaAdminClient
admin = KafkaAdminClient(bootstrap_servers='ldt-kafka:9092')
c = KafkaConsumer('taxi-nyc-raw', 'dq-stream-processed', 'dq-meta-stream', 
                 bootstrap_servers='ldt-kafka:9092', auto_offset_reset='earliest',
                 consumer_timeout_ms=5000, enable_auto_commit=False)
print("Partitions for taxi-nyc-raw:")
for tp in c.assignment():
    end = c.end_offsets([tp])[tp]
    committed = c.committed(tp)
    print(f"  {tp.topic}:{tp.partition} -> end={end}")
'''
result = subprocess.run(
    ['docker', 'exec', 'ldt-kafka-producer', 'python3', '-c', script],
    capture_output=True, text=True
)
print('Kafka consumer offsets:')
print(result.stdout)
if result.stderr and 'Traceback' in result.stderr:
    print(f'ERR: {result.stderr[:200]}')

# 2. Check current job metrics
result2 = subprocess.run(
    ['docker', 'exec', 'ldt-flink-jobmanager', 'curl', '-s',
     f'http://localhost:8081/jobs/{JOB_ID}'],
    capture_output=True, text=True
)
d = json.loads(result2.stdout)
print(f'\nJob State: {d.get("state")}')
print(f'Duration: {d.get("duration",0)//1000}s')
for v in d.get('vertices', []):
    m = v.get('metrics', {})
    print(f"  {v.get('name','?')[:60]}: read={m.get('read-records','?')} write={m.get('write-records','?')}")

# 3. Check checkpoint
result3 = subprocess.run(
    ['docker', 'exec', 'ldt-flink-jobmanager', 'curl', '-s',
     f'http://localhost:8081/jobs/{JOB_ID}/checkpoints'],
    capture_output=True, text=True
)
cp = json.loads(result3.stdout)
counts = cp.get('counts', {})
print(f'\nCheckpoints: {counts}')
inp = cp.get('latest', {}).get('in_progress')
if inp:
    elapsed = json.loads(result2.stdout).get('now', 0) - inp.get('trigger_timestamp', 0)
    print(f'  In-progress: id={inp.get("id")} elapsed={elapsed//1000}s ack={inp.get("num_acknowledged_subtasks","?")}/{inp.get("num_subtasks","?")}')

# 4. Check MinIO data
result4 = subprocess.run(
    ['docker', 'exec', 'ldt-kafka-producer', 'python3', '-c', '''
import boto3
s3 = boto3.client("s3", endpoint_url="http://ldt-minio:9000", aws_access_key_id=os.environ.get("MINIO_ROOT_USER","cadqstream"), aws_secret_access_key=os.environ.get("MINIO_ROOT_PASSWORD","CADQStream2026!"))
for b in ["cadqstream-raw", "cadqstream-anomalies", "cadqstream-metrics"]:
    r = s3.list_objects_v2(Bucket=b, MaxKeys=5)
    print(f"{b}: {r.get(\\"KeyCount\\", 0)} objects")
    for obj in r.get("Contents", [])[:3]:
        print(f"  {obj[\\"Key\\"]} ({obj[\\"Size\\"]} bytes)")
'''],
    capture_output=True, text=True
)
print(f'\nMinIO data:')
print(result4.stdout)
