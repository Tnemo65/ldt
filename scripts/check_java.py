#!/usr/bin/env python3
import subprocess
r = subprocess.run(["ps", "aux"], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if "java" in line.lower() or "flink" in line.lower():
        print(line.strip()[:100])
