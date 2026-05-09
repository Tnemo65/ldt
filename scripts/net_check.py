#!/usr/bin/env python3
import subprocess, json

containers = ["ldt-kafka-1", "ldt-flink-jobmanager", "ldt-flink-taskmanager-1"]
for name in containers:
    result = subprocess.run(["docker", "inspect", name, "--format", "{{json .NetworkSettings.Networks}}"],
                          capture_output=True, text=True, timeout=10)
    nets = json.loads(result.stdout)
    net_names = list(nets.keys())
    print(f"{name}: {net_names}")
