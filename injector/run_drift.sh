#!/bin/bash
pip install -q kafka-python 2>&1
python3 /src/drift_injector.py --scenario ALL --rate 500 2>&1
echo "EXIT: $?"
