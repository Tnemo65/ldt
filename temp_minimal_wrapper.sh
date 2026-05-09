#!/bin/bash
echo "Starting minimal3 at $(date)" >> /tmp/minimal3.log
PYTHONPATH=/tmp:/tmp/src python3 /tmp/minimal3.py >> /tmp/minimal3.log 2>&1
echo "Exit code: $?" >> /tmp/minimal3.log
