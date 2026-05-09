#!/usr/bin/env python3
for line in open("/proc/net/tcp"):
    if " 23A " in line or " 0BDA " in line:
        print(line.strip())
