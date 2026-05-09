#!/usr/bin/env python3
with open("/proc/net/tcp") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) > 9 and parts[3] == "0A":  # LISTEN state
            local = parts[1]
            hex_port = local.split(":")[1]
            port = int(hex_port, 16)
            print(f"Port {port} ({hex_port}) - {line.strip()[:80]}")
