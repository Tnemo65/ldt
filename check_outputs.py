import json, subprocess, time

def cmd(args, timeout=10):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "(timeout)"
    except Exception as e:
        return f"(error: {e})"

# Check MinIO buckets
print("=== MinIO CadQStream Lakehouse ===")
for bucket in ['raw', 'violations', 'canary-violations', 'anomaly-scores', 'drift-events', 'alerts']:
    result = cmd(["docker", "exec", "ldt-minio", "mc", "ls", f"minio/cadqstream/{bucket}/year=2024/month=12/day=15/hour=11/", "-r", "--prefix"])
    if "error" in result.lower():
        print(f"  {bucket}: error or not found")
    else:
        lines = [l for l in result.splitlines() if 'part' in l]
        print(f"  {bucket}: {len(lines)} files in hour 11")

# Check Kafka output topics
print("\n=== Kafka Output Topics ===")
for topic in ['dq-stream-processed', 'dq-stream-anomalies', 'dq-stream-processed-clean', 'dq-meta-stream', 'iec-action-replay']:
    r = cmd(["docker", "exec", "ldt-kafka", "kafka-console-consumer",
             "--bootstrap-server", "localhost:9092", "--topic", topic,
             "--max-messages", "1", "--from-beginning"], timeout=5)
    has_data = 'error' not in r.lower() and len(r.strip()) > 50
    preview = r.strip()[:80].replace('\n', ' ')
    print(f"  {topic}: {'HAS DATA' if has_data else 'NO DATA'} | {preview}")

# Check checkpoint status
result = cmd(["docker", "exec", "ldt-flink-jobmanager", "curl", "-s",
              "http://localhost:8081/jobs/1f504e8e5e0ecb76ab36ec4adbf41cc5/checkpoints"])
try:
    cp = json.loads(result)
    h = cp.get('history', [])
    print(f"\n=== Checkpoint Status ===")
    print(f"  total={cp['counts']['total']} completed={cp['counts']['completed']} failed={cp['counts']['failed']}")
    if h:
        last = h[-1]
        print(f"  Latest: id={last['id']} status={last['status']} acks={last['num_acknowledged_subtasks']}/{last['num_subtasks']} type={last.get('checkpoint_type','N/A')}")
    else:
        print("  No history")
except Exception as e:
    print(f"\n=== Checkpoint === Could not parse: {str(e)}")
