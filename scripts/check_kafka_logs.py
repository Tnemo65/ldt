#!/usr/bin/env python3
import subprocess
r = subprocess.run(["docker", "logs", "ldt-kafka-1"], capture_output=True, text=True, timeout=30)
print("STDOUT:")
print(r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout)
print("\nSTDERR:")
print(r.stderr[-3000:] if len(r.stderr) > 3000 else r.stderr)
