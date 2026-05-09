#!/usr/bin/env python3
with open("/opt/kafka/config/server.properties") as f:
    for line in f:
        line = line.strip()
        if "listener" in line.lower() or "advertised" in line.lower():
            print(line)
