"""
Simple ADWIN Test - Debug drift detection
"""

import numpy as np
from river.drift import ADWIN

np.random.seed(42)

print("=" * 60)
print("ADWIN Debug Test")
print("=" * 60)

# Test 1: Simple step change
print("\nTest 1: Simple step change (mean 10 -> 20)")
adwin1 = ADWIN(delta=0.001)

# First 100: mean 10
data1 = np.random.normal(10, 1, 100)
for v in data1:
    adwin1.update(v)

print(f"After warmup (100 samples, mean=10): drift_detected = {adwin1.drift_detected}")

# Next 50: mean 20 (step change)
drift_detected_at = None
for i, v in enumerate(np.random.normal(20, 1, 50)):
    detected = adwin1.update(v)
    if detected and drift_detected_at is None:
        drift_detected_at = i + 100
        print(f"Drift detected at sample {drift_detected_at}!")

if drift_detected_at is None:
    print("NO DRIFT DETECTED!")

# Test 2: Gradual change
print("\n\nTest 2: Gradual change (10 -> 15 over 200 samples)")
adwin2 = ADWIN(delta=0.001)

# Warmup
for v in np.random.normal(10, 1, 100):
    adwin2.update(v)

print(f"After warmup: drift_detected = {adwin2.drift_detected}")

# Gradual
first_det = None
for i in range(200):
    mean = 10 + (i / 200) * 5  # 10 -> 15
    detected = adwin2.update(np.random.normal(mean, 1))
    if detected and first_det is None:
        first_det = 100 + i
        print(f"Drift detected at sample {first_det} (mean ~{mean:.1f})")

if first_det is None:
    print("NO DRIFT DETECTED!")

# Test 3: Bigger drift
print("\n\nTest 3: Large drift (10 -> 50, 5x)")
adwin3 = ADWIN(delta=0.001)

# Warmup
for v in np.random.normal(10, 3, 200):
    adwin3.update(v)

print(f"After warmup (200 samples, std=3): drift_detected = {adwin3.drift_detected}")
print(f"ADWIN window size estimate: {len(adwin3._w) if hasattr(adwin3, '_w') else 'N/A'}")

# Big step
first_det = None
for i, v in enumerate(np.random.normal(50, 3, 100)):
    detected = adwin3.update(v)
    if detected and first_det is None:
        first_det = 200 + i
        print(f"Drift detected at sample {first_det}!")

if first_det is None:
    print("NO DRIFT DETECTED!")

# Test 4: Compare with raw metric comparison
print("\n\nTest 4: Direct mean comparison (baseline)")
data_before = np.random.normal(10, 3, 200)
data_after = np.random.normal(50, 3, 100)

print(f"Mean before: {np.mean(data_before):.2f}")
print(f"Mean after: {np.mean(data_after):.2f}")
print(f"Ratio: {np.mean(data_after)/np.mean(data_before):.1f}x")
print(f"Change: {abs(np.mean(data_after) - np.mean(data_before)):.2f}")
print(f"Std before: {np.std(data_before):.2f}")
print(f"Change/Std: {abs(np.mean(data_after) - np.mean(data_before))/np.std(data_before):.1f}")

print("\n\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
ADWIN requires:
1. Warmup period (>100 samples)
2. Consistent drift signal
3. Low variance relative to drift magnitude

Recommendation: For production, ADWIN delta should be tuned per metric.
For now, we can use simpler change detection:
- Rolling mean comparison
- Statistical tests (t-test, KS test)
""")
