# CA-DQStream: 4 Critical Flaws - Diagnosis & Fix

> **Date:** 2026-05-13
> **Status:** CRITICAL - Must fix before deployment
> **Reviewed by:** Architecture Review Board

---

## EXECUTIVE SUMMARY

Bản thiết kế CA-DQStream có 4 lỗi kiến trúc nghiêm trọng khi ghép MemStream vào Flink. Nếu đem đi code hoặc bảo vệ trước hội đồng, hệ thống sẽ bị reject ngay lập tức.

| # | Flaw | Severity | Impact |
|---|------|----------|---------|
| 1 | Feature Dimension Mismatch (25D vs 30D) | **CRITICAL** | Type 3 Fraud undetectable |
| 2 | METER Retrain Redundancy | **CRITICAL** | Conflicting signals to MemStream |
| 3 | Warmup Inside Flink Operator | **CRITICAL** | TaskManager timeout, pipeline crash |
| 4 | Double ADWIN Collision | **HIGH** | Competing drift signals |

---

## FLAW 1: Feature Dimension Mismatch

### 1.1 Problem

```python
# Current: src/ml/memstream_core.py, line 57
class MemStreamConfig:
    def __init__(self):
        self.in_dim: int = 25   # ← WRONG: Missing RatecodeID
```

```python
# Current: src/operators/memstream_scoring_operator.py, line 41
DEFAULT_CONFIG = {
    'in_dim': 25,    # ← WRONG
    'hidden_dim': 50,
    'out_dim': 25,
}
```

### 1.2 Root Cause

MemStream được configure 25D nhưng benchmark v10 đã chứng minh rằng **30D là bắt buộc**:

| Feature | Dimensions | Lý do |
|---------|------------|--------|
| Base features (raw + derived + temporal) | 15D | Core trip attributes |
| **Grid X/Y spatial** | 4D | Micro-location context |
| **RatecodeID one-hot** | 5D | JFK flat fare detection |
| Normalized ratios | 6D | Context-aware scoring |
| **TOTAL** | **30D** | **Minimum for production** |

**Type 3 Fraud** (JFK Ratecode + Manhattan Zone + $70 fare) sẽ **không bị phát hiện** nếu thiếu RatecodeID one-hot. Không có 5 cột one-hot này, mô hình không biết zone 161 (Manhattan) kết hợp với RatecodeID=2 (JFK) là bất thường.

### 1.3 Fix

**File: `src/ml/memstream_core.py`**

```python
class MemStreamConfig:
    def __init__(self):
        # Architecture - FIXED to 30D
        self.in_dim: int = 30      # 25D base + 4D Grid + 5D Ratecode one-hot
        self.hidden_dim: int = 60   # 2x compression (was 50)
        self.out_dim: int = 30     # Match input dimension
```

**File: `src/operators/memstream_scoring_operator.py`**

```python
DEFAULT_CONFIG = {
    'in_dim': 30,      # FIXED: 30D (was 25D)
    'hidden_dim': 60,   # FIXED: 2x compression (was 50)
    'out_dim': 30,     # FIXED: Match input (was 25)
    'memory_len': 50000,
    # ... rest unchanged
}
```

**30D Feature Vector Specification:**

```
Index 0-14:  Base features (15D)
  [0]  trip_distance
  [1]  duration_minutes
  [2]  fare_amount
  [3]  passenger_count
  [4]  total_amount
  [5]  speed_mph
  [6]  fare_per_mile
  [7]  fare_per_minute
  [8]  fare_per_passenger
  [9]  hour
  [10] day_of_week
  [11] is_weekend
  [12] is_night
  [13] month
  [14] distance_squared

Index 15-18: Grid spatial (4D) — Micro-location context
  [15] pu_grid_x
  [16] pu_grid_y
  [17] do_grid_x
  [18] do_grid_y

Index 19-23: RatecodeID one-hot (5D) — CRITICAL for Type 3 fraud
  [19] ratecode_1 (Standard)
  [20] ratecode_2 (JFK)       ← JFK flat fare detection
  [21] ratecode_3 (Newark)
  [22] ratecode_4 (Negotiated)
  [23] ratecode_5 (Group)

Index 24-29: Normalized ratios (6D)
  [24] fare_per_mile_norm
  [25] fare_per_min_norm
  [26] speed_norm
  [27] pax_per_mile
  [28] inter_borough_rough
  [29] log_fare
```

### 1.4 Verification

```python
# Verify in unit test
def test_memstream_30d():
    cfg = MemStreamConfig()
    assert cfg.in_dim == 30, "Must be 30D for RatecodeID"
    assert cfg.hidden_dim == 60, "Must be 2x for compression"

    ae = MemStreamAE(in_dim=30, hidden_dim=60)
    x = torch.randn(5, 30)
    recon = ae(x)
    assert recon.shape == (5, 30)
```

---

## FLAW 2: METER Retrain Redundancy

### 2.1 Problem

```python
# Current: src/operators/iec_operator.py, line 193-198
strategy_names = {
    0: 'do_nothing',
    1: 'adjust_threshold',
    2: 'retrain_model',   # ← REDUNDANT: MemStream already online-learns
    3: 'switch_model'
}
```

```python
# Current: src/operators/iec_operator.py, line 249-256
elif strategy == 'retrain_model':
    # In production: Trigger async model retraining
    return {
        'action': 'retrain_triggered',
        'message': 'Model retraining initiated',
        'neighborhood': meta_metrics.get('neighborhood_id')
    }
```

### 2.2 Root Cause

**METER được thiết kế cho mô hình TĨNH** (IsolationForest). Khi có drift:
- IsolationForest **không có online learning** → Cần retrain
- **MemStream đã tự nó là online** → Retrain là THỪA

```
┌─────────────────────────────────────────────────────────────────┐
│  IsolationForest (Static)     vs     MemStream (Online)      │
│  ─────────────────────────           ───────────────────────  │
│  • Offline batch training           • Online inference         │
│  • No adaptation                    • Memory module updates    │
│  • Need retrain on drift     ←→     • Self-evolving          │
│  • METER: "Retrain!"         ✓     • METER: "Retrain!" ✗    │
└─────────────────────────────────────────────────────────────────┘

Retrain command to MemStream = Conflicting signal!
MemStream will ignore (online learning is already adaptive)
OR MemStream will retrain unnecessarily (waste resources)
```

### 2.3 Fix

**Redesign METER strategies for MemStream:**

```python
# REVISED strategy_names for MemStream
strategy_names = {
    0: 'do_nothing',          # No drift → continue
    1: 'adjust_threshold',    # Minor drift → adjust beta
    2: 'lock_memory',        # [NEW] Poisoning detected → freeze memory
    3: 'canary_only'         # [NEW] Severe drift → fallback to Canary
}
```

**File: `src/operators/iec_operator.py`**

```python
# Lines 193-198: Replace retrain_model
strategy_names = {
    0: 'do_nothing',      # Normal operation
    1: 'adjust_threshold', # Minor drift → beta adjustment
    2: 'lock_memory',     # Poisoning → freeze memory updates
    3: 'canary_only'      # Severe drift → Canary-only mode
}

# Lines 225-267: Replace _execute_strategy
def _execute_strategy(self, strategy: str, meta_metrics: dict, drift_assessment: dict):
    if strategy == 'do_nothing':
        return {'action': 'none', 'message': 'Normal operation'}

    elif strategy == 'adjust_threshold':
        # Adjust MemStream beta threshold
        anomaly_rate = meta_metrics.get('anomaly_rate', 0.05)
        if anomaly_rate > 0.15:
            new_beta = 0.55  # Higher = less sensitive
        elif anomaly_rate < 0.03:
            new_beta = 0.45  # Lower = more sensitive
        else:
            new_beta = 0.50
        return {
            'action': 'threshold_adjusted',
            'new_beta': new_beta,
            'emit_to_kafka': 'memstream-beta-updates',  # Signal to operator
            'message': f'Beta adjusted to {new_beta}'
        }

    elif strategy == 'lock_memory':
        # Poisoning detected → freeze memory updates
        return {
            'action': 'memory_locked',
            'emit_to_kafka': 'memstream-control',
            'lock_reason': 'poisoning_detected',
            'message': 'Memory updates frozen to prevent poisoning'
        }

    elif strategy == 'canary_only':
        # Severe drift → switch to Canary-only mode
        return {
            'action': 'canary_fallback',
            'emit_to_kafka': 'memstream-control',
            'fallback_reason': 'severe_drift',
            'message': 'Switched to Canary-only mode'
        }
```

**New Kafka topic for MemStream control:**

```yaml
# deployment/kafka/init-scripts/01-create-topics.sh
memstream-control:
  partitions: 1
  replication: 1
  retention: 1 day
  cleanup: delete

memstream-beta-updates:
  partitions: 1
  replication: 1
  retention: 1 day
  cleanup: delete
```

**MemStreamScoringOperator listens for control signals:**

```python
# In MemStreamScoringOperator.map()
def map(self, value):
    # Check for control signals from IEC
    control_signal = self._check_control_channel()
    if control_signal:
        if control_signal['action'] == 'memory_locked':
            self._bar_controller.freeze_memory()
        elif control_signal['action'] == 'canary_fallback':
            self._mode = 'canary_only'
        elif control_signal['action'] == 'beta_adjusted':
            self._ms_core.set_beta(control_signal['new_beta'])

    # Normal scoring...
```

### 2.4 Why `retrain_model` must be removed

| Scenario | Old Behavior (with retrain) | New Behavior (MemStream-native) |
|----------|----------------------------|---------------------------------|
| Gradual drift | METER triggers retrain → Job runs 500 epochs offline → New model uploaded | MemStream updates memory online → Continues seamlessly |
| Abrupt spike | METER triggers retrain → System waits for retrain | MemStream detects via ADWIN → Adjusts memory dynamically |
| Poisoning | No protection → Memory corrupted | `lock_memory` freezes updates → Canary-only fallback |
| Recovery | Need to reload model | `unlock_memory` → MemStream adapts naturally |

---

## FLAW 3: Warmup Inside Flink Operator

### 3.1 Problem

```python
# Current: src/operators/memstream_scoring_operator.py, lines 193-198
def open(self, runtime_context):
    # ...
    elif self.warmup_data is not None:
        LOGGER.info(f"[MemStreamScoring] Warming up with {len(self.warmup_data)} samples...")
        self._ms_core.warmup(self.warmup_data, verbose=True)  # ← BLOCKS TASKMANAGER!
        self._ms_core.set_beta(self.config['default_beta'])
        LOGGER.info("[MemStreamScoring] Warmup complete")
```

```python
# Current: Warmup runs backpropagation
# src/ml/memstream_core.py, MemStreamCore.warmup()
for epoch in range(epochs):  # ← 500 epochs!
    for batch in training_data:
        loss = criterion(recon, original)
        loss.backward()  # ← Gradient computation
        optimizer.step() # ← Weights update
```

### 3.2 Root Cause

**Apache Flink là streaming engine với milisecond latency SLA.**

```
Timeline của Flink TaskManager:
─────────────────────────────────────────────────────────────────→

JobManager gửi: "Start operator"
        │
        ▼
TaskManager: open() được gọi
        │
        ├─→ Initialize model (ms)
        ├─→ Load weights from checkpoint (ms)
        ├─→ Load memory from checkpoint (ms)
        │
        ├─→ ⚠️ warmup() ← CHẠY 500 EPOCHS BACKPROPAGATION!
        │       │
        │       ├─→ Epoch 1: Forward + Backward (100ms)
        │       ├─→ Epoch 2: Forward + Backward (100ms)
        │       ├─→ ...
        │       └─→ Epoch 500: Forward + Backward (100ms)
        │       │
        │       └─→ TOTAL: ~50,000ms = 50 SECONDS BLOCKED!
        │
        ├─→ After 30s: Watermark timeout ⚠️
        ├─→ After 60s: Checkpoint timeout ⚠️
        └─→ After 120s: Job marked FAILED ❌
```

**Flink checkpoint interval = 10 phút (600s)**
**Flink TaskManager timeout = 60 giây (default)**
**Warmup 500 epochs ≈ 50-100 giây** → TaskManager bị timeout → Pipeline CRASH

### 3.3 Fix

**Thiết kế: 2-Phase Separation**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 0: OFFLINE (Before Flink job starts)                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  scripts/warmup_memstream.py                                      │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  1. Load clean baseline data (e.g., Jan 2024)                  │   │
│  │  2. Extract 30D features (FeatureVectorizer)                │   │
│  │  3. Train Denoising Autoencoder: 500 epochs, MSE loss         │   │
│  │  4. Initialize memory from last 10% of clean data                │   │
│  │  5. Save to checkpoint:                                        │   │
│  │     ├── memstream_weights.pt  (AE weights)                       │   │
│  │     ├── memstream_scaler.pkl (StandardScaler)                  │   │
│  │     ├── memstream_memory.pt  (Initial memory state)             │   │
│  │     └── memstream_config.json (30D config)                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: ONLINE (Flink job running)                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  MemStreamScoringOperator.open()                                 │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  1. Load weights from checkpoint (torch.load) ← MILLISECONDS    │   │
│  │  2. Load memory from checkpoint (tensor copy) ← MILLISECONDS   │   │
│  │  3. Set beta threshold                                         │   │
│  │  4. START INFERENCE IMMEDIATELY ← NO TRAINING INSIDE FLINK!    │   │
│  │                                                                  │   │
│  │  Incoming record → Score (kNN + recon) → Update memory (if BAR) │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

**File: `scripts/warmup_memstream.py` (NEW)**

```python
#!/usr/bin/env python3
"""
Offline Warmup Script for MemStream.

MUST run BEFORE Flink job starts. This script:
1. Loads clean baseline data
2. Trains Denoising Autoencoder (500 epochs)
3. Initializes Memory Module
4. Saves checkpoint for Flink to load

Usage:
    python scripts/warmup_memstream.py --data data/clean/jan_2024.parquet --output models/
"""

import argparse
import pickle
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ml.memstream_core import MemStreamCore, MemStreamConfig, set_determinism
from src.features.vectorizer import FeatureVectorizer


def warmup_offline(data_path: str, output_dir: str):
    """Offline warmup: Train AE, initialize memory, save checkpoint."""
    print("=" * 60)
    print("MEMSTREAM OFFLINE WARMUP")
    print("=" * 60)

    # 1. Load and vectorize data
    print(f"\n[1/5] Loading data from {data_path}...")
    import pandas as pd
    df = pd.read_parquet(data_path)
    print(f"    Loaded {len(df):,} records")

    print(f"\n[2/5] Vectorizing to 30D features...")
    vectorizer = FeatureVectorizer()  # 30D vectorizer
    X = vectorizer.transform_batch(df)
    print(f"    Features: {X.shape}")

    # 2. Configure MemStream for 30D
    print(f"\n[3/5] Configuring MemStream (30D → 60D → 30D)...")
    set_determinism(42)
    cfg = MemStreamConfig()
    cfg.in_dim = 30      # 30D (was 25D)
    cfg.hidden_dim = 60  # 2x compression
    cfg.out_dim = 30
    cfg.warmup_epochs = 500
    cfg.warmup_noise_std = 0.1
    cfg.memory_len = 50000
    cfg.default_beta = 0.5

    # 3. Initialize and warmup
    print(f"\n[4/5] Training Denoising Autoencoder (500 epochs)...")
    ms = MemStreamCore(cfg=cfg, device='cpu')
    ms.warmup(X, epochs=500, verbose=True)

    # 4. Save checkpoint
    print(f"\n[5/5] Saving checkpoint to {output_dir}...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(ms.ae.state_dict(), output_path / 'memstream_weights.pt')

    # Save scaler
    with open(output_path / 'memstream_scaler.pkl', 'wb') as f:
        pickle.dump({
            'mean': ms.mean.numpy(),
            'std': ms.std.numpy(),
        }, f)

    # Save memory state
    torch.save({
        'memory': ms.memory.memory.cpu(),
        'memory_count': ms.memory.count,
        'memory_ptr': ms.memory.mem_ptr,
    }, output_path / 'memstream_memory.pt')

    # Save config
    config_json = {
        'in_dim': 30,
        'hidden_dim': 60,
        'out_dim': 30,
        'memory_len': 50000,
        'default_beta': 0.5,
        'warmup_epochs': 500,
        'warmup_date': datetime.now().isoformat(),
        'n_training_samples': len(X),
    }
    with open(output_path / 'memstream_config.json', 'w') as f:
        json.dump(config_json, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"WARMUP COMPLETE")
    print(f"  Output: {output_dir}")
    print(f"  Files: weights, scaler, memory, config")
    print(f"  30D model ready for Flink!")
    print(f"{'=' * 60}")
```

**File: `src/operators/memstream_scoring_operator.py` (FIXED)**

```python
def open(self, runtime_context):
    # ... imports ...

    # Load pre-trained weights from checkpoint (FAST - milliseconds)
    if self._try_load_checkpoint():
        LOGGER.info("[MemStreamScoring] Loaded pre-trained weights from checkpoint")
        LOGGER.info(f"[MemStreamScoring] Warmup was done OFFLINE by scripts/warmup_memstream.py")
        LOGGER.info(f"[MemStreamScoring] Ready for inference: {self._total_scored} records processed")
    else:
        LOGGER.error("[MemStreamScoring] FATAL: No checkpoint found!")
        LOGGER.error("[MemStreamScoring] Run scripts/warmup_memstream.py BEFORE starting Flink job")
        raise RuntimeError("MemStream checkpoint missing - run warmup script first")
```

### 3.4 Deployment Checklist

```bash
# Step 1: Offline warmup (run once, before Flink)
python scripts/warmup_memstream.py \
    --data data/clean/jan_2024_baseline.parquet \
    --output models/memstream_checkpoint/

# Step 2: Verify checkpoint
ls -la models/memstream_checkpoint/
# Expected:
#   memstream_weights.pt      (~2MB)
#   memstream_scaler.pkl     (~1KB)
#   memstream_memory.pt      (~5MB for 50K slots)
#   memstream_config.json    (~200B)

# Step 3: Start Flink job (warmup already done)
flink run -c src.jobs.ca_dqstream_job \
    target/ca-dqstream.jar \
    --memstream-checkpoint models/memstream_checkpoint/
```

---

## FLAW 4: Double ADWIN Collision

### 4.1 Problem

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CURRENT: 2 ADWIN systems "fighting" each other                        │
│                                                                         │
│  Layer 2B: MemStreamScoringOperator                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  MemStreamCore                                                    │   │
│  │  ├── BARController                                                │   │
│  │  │   └── ADWIN (per neighborhood)  ← Monitors: anomaly_score  │   │
│  │  │       Purpose: BAR budget decision                          │   │
│  │  └── MemoryModule (FIFO queue)                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                         │
│                              │ drift_detected signal                   │
│                              ▼                                         │
│  Layer 4: IECOperator                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  MultiInstanceADWIN (36 instances)                              │   │
│  │   └── ADWIN[neighborhood × metric]  ← Monitors: 6 meta-metrics │   │
│  │       Purpose: Global drift detection for METER               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ⚠️ CONFLICT: Both detect "drift" but for different purposes!       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Root Cause

| Aspect | Local ADWIN (MemStream) | Global ADWIN-U (IEC) |
|--------|-------------------------|----------------------|
| **Scope** | Micro-level (single operator) | Macro-level (system-wide) |
| **Target** | Anomaly score stream | 6 meta-metrics from MetaAggregator |
| **Purpose** | BAR budget for memory update | Strategy decision for METER |
| **Output** | `should_update_memory: bool` | `drifts: [{metric, value}]` |
| **Who listens** | BARController | IECOperator |
| **Action** | Update memory or not | Emit strategy signal |

Nếu không phân tách rõ, 2 hệ thống sẽ:
1. Cùng phát hiện drift
2. Cùng trigger actions khác nhau
3. gây ra conflicting signals

### 4.3 Fix

**Document clearly in architecture:**

```python
# File: src/operators/memstream_scoring_operator.py
# At top of file, add clear documentation:

"""
ADWIN SCOPE DEFINITION:
═══════════════════════════════════════════════════════════════════

This operator uses LOCAL ADWIN (inside MemStreamCore) for:
  Purpose: BAR (Budget Allocation Rate) Controller decision
  Target: anomaly_score from MemStream scoring
  Scope: Micro-level (per neighborhood)
  Output: should_update_memory (bool)
  Action: Add new record to Memory Module or not

This is DIFFERENT from Global ADWIN-U in IECOperator:
  Global ADWIN-U monitors: volume, null_rate, violation_rate,
                            anomaly_rate, avg_anomaly_score, delta_score
  Purpose: METER strategy decision
  Scope: Macro-level (system-wide)
  Output: drift events per (neighborhood, metric)
  Action: Emit strategy signal (adjust_threshold, lock_memory, etc.)

These two ADWIN systems do NOT interfere with each other.
═══════════════════════════════════════════════════════════════════
"""
```

**Update architecture diagram:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ADWIN SCOPE SEPARATION (CORRECTED)                                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Layer 2B: LOCAL ADWIN (Inside MemStreamCore)                 │    │
│  │  ─────────────────────────────────────────────────────────────  │    │
│  │  Scope: MICRO (per neighborhood, per record stream)          │    │
│  │  Metric monitored: anomaly_score                              │    │
│  │  Purpose: BAR Controller → Memory Module decision             │    │
│  │  │                                                           │    │
│  │  │ drift_detected → should_update_memory = True             │    │
│  │  │                                                             │    │
│  │  │ Example: "Anomaly score in manhattan suddenly jumped"    │    │
│  │  │           → BAR grants budget → Memory updates             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Layer 4: GLOBAL ADWIN-U (Inside IECOperator)                 │    │
│  │  ─────────────────────────────────────────────────────────────  │    │
│  │  Scope: MACRO (system-wide, per 1-min window)               │    │
│  │  Metrics: 6 meta-metrics from MetaAggregator                │    │
│  │  Purpose: METER strategy decision                            │    │
│  │  │                                                           │    │
│  │  │ drift_detected → METER.predict() → Strategy              │    │
│  │  │                                                             │    │
│  │  │ Example: "null_rate in brooklyn increased 5x over         │    │
│  │  │           3 consecutive windows"                            │    │
│  │  │           → METER: lock_memory → Canary-only fallback      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════   │
│  LOCAL ADWIN ────────────────────────────────────────── GLOBAL ADWIN   │
│  "Should I remember this?"                     "Should I change strategy?" │
│  Answers: BAR budget                              Answers: METER action  │
│  ════════════════════════════════════════════════════════════════════   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Add explicit field to distinguish:**

```python
# In MemStreamScoringOperator.map() output
result = {
    **value,
    # ... existing fields ...
    'local_drift_detected': reason == "drift_detected",  # From LOCAL ADWIN
    'local_adwin_scope': 'micro',  # BAR budget decision
}
```

```python
# In IECOperator output
iec_decision = {
    **meta_metrics,
    'drifts_detected': drifts,  # From GLOBAL ADWIN-U
    'global_adwin_scope': 'macro',  # Strategy decision
}
```

---

## SUMMARY: All 4 Fixes

| # | Fix | Files to Change | Lines |
|---|-----|-----------------|-------|
| 1 | Change `in_dim` from 25 to 30 | `src/ml/memstream_core.py`, `src/operators/memstream_scoring_operator.py` | Config class, DEFAULT_CONFIG |
| 2 | Replace `retrain_model` with `lock_memory` + `canary_only` | `src/operators/iec_operator.py` | strategy_names, _execute_strategy |
| 3 | Move warmup to offline script | `scripts/warmup_memstream.py` (NEW), `src/operators/memstream_scoring_operator.py` | open() method |
| 4 | Document ADWIN scope separation | `src/operators/memstream_scoring_operator.py` | Add docstring block |

---

## ACTION CHECKLIST

### Pre-Code (Architecture)

- [ ] Update MemStreamConfig to 30D → 60D → 30D
- [ ] Update DEFAULT_CONFIG in MemStreamScoringOperator
- [ ] Create FeatureVectorizer30D (extends 25D with Grid + Ratecode)
- [ ] Replace METER strategies: remove `retrain_model`, add `lock_memory`, `canary_only`
- [ ] Add Kafka topics: `memstream-control`, `memstream-beta-updates`
- [ ] Add scope documentation for ADWIN systems

### Implementation

- [ ] Create `scripts/warmup_memstream.py` (offline warmup)
- [ ] Modify `MemStreamScoringOperator.open()` to load checkpoint only (no warmup)
- [ ] Modify `IECOperator._execute_strategy()` for new strategies
- [ ] Add control signal listener to MemStreamScoringOperator
- [ ] Update unit tests for 30D architecture
- [ ] Update integration tests for new control flow

### Testing

- [ ] Test: Warmup script produces valid checkpoint
- [ ] Test: Flink operator loads checkpoint in < 1 second
- [ ] Test: `lock_memory` freezes memory updates
- [ ] Test: `canary_only` switches operator to pass-through mode
- [ ] Test: LOCAL and GLOBAL ADWIN do not interfere
- [ ] Test: Type 3 fraud (JFK ratecode + Manhattan zone) is detected

### Documentation

- [ ] Update `docs/full_flow_explained.md` with 30D architecture
- [ ] Update `docs/concept_drift_analysis.md` with new METER strategies
- [ ] Update `docs/architecture_diagram.md` with ADWIN scope separation
- [ ] Add warmup deployment instructions

---

*Document version: 1.1 - Fixed 4 Critical Flaws*
*Review status: APPROVED for implementation*
*Next: Code implementation*
