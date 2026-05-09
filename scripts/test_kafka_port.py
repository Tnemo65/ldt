#!/usr/bin/env python3
import socket
s = socket.socket()
s.settimeout(5)
r = s.connect_ex(('kafka', 9092))
s.close()
if r == 0:
    print("Port 9092 reachable from Flink JM to Kafka")
else:
    print(f"Port 9092 NOT reachable, error={r}")
