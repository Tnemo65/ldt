#!/usr/bin/env python3
import os
for k, v in os.environ.items():
    if 'KAFKA' in k:
        print(f"{k}={v}")
