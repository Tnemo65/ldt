import json, sys, subprocess, os

# Get job details
result = subprocess.run(
    ["docker", "exec", "ldt-flink-jobmanager", "curl", "-s",
     "http://localhost:8081/jobs/1f504e8e5e0ecb76ab36ec4adbf41cc5"],
    capture_output=True, text=True
)
d = json.loads(result.stdout)

print(f"\n{'='*80}")
print(f"Job: {d['jid']} State: {d['state']}")
print(f"{'='*80}")
for i, v in enumerate(d['vertices']):
    vid = v['id'][:16]
    name = v['name'].replace('\n', ' ')[:80]
    wr = v['metrics']['write-records']
    rr = v['metrics']['read-records']
    print(f"V{i+1}({vid}): write={wr:6d} read={rr:6d} | {name}")

# Check checkpoints
result2 = subprocess.run(
    ["docker", "exec", "ldt-flink-jobmanager", "curl", "-s",
     "http://localhost:8081/jobs/1f504e8e5e0ecb76ab36ec4adbf41cc5/checkpoints"],
    capture_output=True, text=True
)
cp = json.loads(result2.stdout)
print(f"\nCheckpoints: total={cp['counts']['total']} completed={cp['counts']['completed']} failed={cp['counts']['failed']}")
if cp.get('history'):
    h = cp['history'][-1]
    print(f"  Latest: id={h['id']} status={h['status']} acks={h['num_acknowledged_subtasks']}/{h['num_subtasks']}")

# Consumer lag
result3 = subprocess.run(
    ["docker", "exec", "ldt-kafka", "kafka-consumer-groups",
     "--bootstrap-server", "localhost:9092",
     "--group", "cadqstream-complete-pipeline",
     "--describe"],
    capture_output=True, text=True
)
print(f"\nConsumer Group Status:")
print(result3.stdout[-500:] if len(result3.stdout) > 500 else result3.stdout)
