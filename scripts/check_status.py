#!/usr/bin/env python3
import json, urllib.request

# Check Flink jobs
try:
    resp = urllib.request.urlopen("http://localhost:8081/jobs", timeout=5)
    d = json.loads(resp.read())
    jobs = d.get("jobs", [])
    print(f"Flink Jobs: {len(jobs)}")
    for j in jobs:
        print(f"  {j['id']} - {j['status']}")
except Exception as e:
    print(f"Flink check error: {e}")

# Check Kafka topic
import subprocess
try:
    r = subprocess.run(["docker", "exec", "ldt-kafka", "kafka-topics", "--list", "--bootstrap-server", "localhost:9092"],
                       capture_output=True, text=True, timeout=10)
    print(f"\nKafka Topics:\n{r.stdout}")
    if r.stderr:
        print(f"Kafka stderr: {r.stderr[:200]}")
except Exception as e:
    print(f"Kafka check error: {e}")
