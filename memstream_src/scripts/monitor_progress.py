#!/usr/bin/env python3
"""Monitor benchmark progress by checking CSV file."""
import sys, os
from pathlib import Path
import time

results_file = Path('c:/proj/ldt/results/rigorous_v1/results_detailed.csv')
results_file.parent.mkdir(parents=True, exist_ok=True)

last_count = 0
last_time = time.time()

print("Monitoring benchmark progress...")
print(f"Output dir: {results_file.parent}")
print(f"Expected: 8 algos x 33 folds x 5 seeds = 1320 rows")
print()

while True:
    if results_file.exists():
        with open(results_file) as f:
            lines = f.readlines()
        count = len(lines) - 1  # subtract header
        elapsed = time.time() - last_time

        if count != last_count:
            delta = count - last_count
            rate = delta / elapsed if elapsed > 0 else 0
            eta = (1320 - count) / rate / 60 if rate > 0 else 0
            print(f"[{time.strftime('%H:%M:%S')}] Rows: {count:4d}/1320 ({count/1320*100:5.1f}%) "
                  f"+{delta:3d} ({rate:.1f}/s) ETA={eta:.1f}min")
            sys.stdout.flush()
            last_count = count
            last_time = time.time()

        if count >= 1320:
            print(f"\nDONE! All {count} rows collected.")
            break
    else:
        if int(time.time()) % 30 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Waiting for results file...")
            sys.stdout.flush()

    time.sleep(5)
