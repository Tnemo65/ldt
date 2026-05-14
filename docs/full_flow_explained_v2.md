# CA-DQStream: Full Flow Chi Tiết - V2.1 (FIXED)

> **Ngày:** 2026-05-13
> **Pipeline:** Kafka → L1 → L2A (Canary) + L2B (MemStream ML) → L3 (Voting) → L4 (IEC)
> **Status:** V2.1 - Fixed 4 Critical Flaws

---

## Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              KAFKA: taxi-nyc-raw                                    │
│                          (Raw NYC Taxi Data - JSON)                                  │
└────────────────────────────────────────────┬────────────────────────────────────────┘
                                             │
                                             ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                            LAYER 1: BASELINE VALIDATION                              │
│  ParseJson → Watermark → AddTripId → Deduplicator(7-day TTL) → SchemaValidator  │
│  Output: valid_stream → Kafka topic / invalid_stream → dq-hard-rule-violations  │
└───────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 2: DUAL-BRANCH PROCESSING                                   │
│  ┌────────────────────────────────────────────┐ ┌──────────────────────────────────┐│
│  │  LAYER 2A: CANARY (7 rules)              │ │  LAYER 2B: MEMSTREAM ML         ││
│  │  R1-R7: Static rule violations          │ │  30D DAE + Memory + BAR + ADWIN ││
│  │  Priority: OVERRIDE ML if violated       │ │  (See details below)            ││
│  └────────────────────────────────────────────┘ └──────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 3: VOTING ENSEMBLE + METAAGGREGATOR                        │
│  Canary Override > ML Decision                                                      │
│  MetaAggregator: 1-min window, per neighborhood → 6 meta-metrics                 │
│  volume, null_rate, violation_rate, anomaly_rate, avg_anomaly_score, delta_score  │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 4: IEC - INTELLIGENT EVOLUTION CONTROLLER                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐   │
│  │  ADWIN-U (36 instances): Giám sát 6 meta-metrics từ MetaAggregator     │   │
│  │  METER (MLP): Dự đoán strategy từ 6D meta-features                      │   │
│  │  Strategies: do_nothing, adjust_threshold, lock_memory, canary_only        │   │
│  └────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Chi Tiết Từng Layer

### LAYER 1: Baseline Validation

| Operator | Chức năng |
|----------|-----------|
| ParseJsonFunction | JSON → dict |
| WatermarkAssigner | Event time từ pickup time |
| AddTripIdFunction | Hash composite key (VendorID\|pickup\|PU\|DO\|fare) |
| DeduplicatorFunction | Drop duplicate trips (7-day TTL, RocksDB) |
| SchemaValidator | Required fields + zone 1-263 |

### LAYER 2A: Canary Rules (7 Static Rules)

```
R1: negative_fare           → fare ≤ 0
R2: zero_distance_fare     → distance=0 AND fare>0
R3: invalid_passengers     → passenger_count < 1 OR > 6
R4: invalid_payment        → payment_type NOT IN [1,2,3,4,5,6]
R5: extreme_fare          → fare > 1000
R6: extreme_duration      → duration > 24*3600 seconds
R7: negative_duration      → duration < 0

Priority: CANARY OVERRIDES ML → Nếu có violation → FINAL = ANOMALY
```

### LAYER 2B: MemStream ML (30D Denoising Autoencoder)

#### 1.1 Tại Sao Phải 30D?

```
⚠️ CRITICAL FIX #1: Phải dùng 30D thay vì 25D

25D (THIẾU) → Không bắt được Type 3 Fraud (JFK Ratecode + Manhattan Zone + $70)
30D (ĐỦ)    → Có Grid spatial (4D) + RatecodeID one-hot (5D) → Bắt được mọi fraud type

30D Feature Vector:
┌─────────────────────────────────────────────────────────────────┐
│ Index 0-14: Base features (15D)                              │
│   [0] trip_distance  [1] duration_min  [2] fare_amount        │
│   [3] passenger_count [4] total_amount  [5] speed_mph          │
│   [6] fare_per_mile  [7] fare_per_min  [8] fare_per_pax      │
│   [9] hour          [10] day_of_week [11] is_weekend        │
│   [12] is_night     [13] month       [14] distance_squared   │
├─────────────────────────────────────────────────────────────────┤
│ Index 15-18: Grid spatial (4D) — Micro-location context       │
│   [15] pu_grid_x  [16] pu_grid_y                              │
│   [17] do_grid_x  [18] do_grid_y                              │
├─────────────────────────────────────────────────────────────────┤
│ Index 19-23: RatecodeID one-hot (5D) — CRITICAL cho Type 3   │
│   [19] ratecode_1 (Standard)                                   │
│   [20] ratecode_2 (JFK) ← JFK flat fare detection             │
│   [21] ratecode_3 (Newark)                                    │
│   [22] ratecode_4 (Negotiated)                                │
│   [23] ratecode_5 (Group)                                     │
├─────────────────────────────────────────────────────────────────┤
│ Index 24-29: Normalized ratios (6D)                           │
│   [24] fare_per_mile_norm  [25] fare_per_min_norm             │
│   [26] speed_norm         [27] pax_per_mile                   │
│   [28] inter_borough     [29] log_fare                       │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.2 Kiến Trúc Denoising Autoencoder

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUT: 30D Feature Vector                                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  ENCODER (30D → 60D → 30D)                                      │
│  Linear(30, 60) → Tanh → Linear(60, 30) → Tanh                │
│                                                                  │
│  INPUT ──→ [30D] ──→ [60D] ──→ [30D] ──→ ENCODED (z)        │
└──────────────────────────────────────────────────────────────────┘
                              │
          ┌──────────────────┴──────────────────┐
          ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────┐
│  RECONSTRUCTION     │              │  MEMORY MODULE       │
│  (Decoder)          │              │  (FIFO, 50K slots) │
│                     │              │                     │
│  [30D]─→[60D]─→  │              │  z₁, z₂, ..., z₅₀₀₀│
│        ──→[30D]   │              │  (30D encoded vecs) │
│                     │              │                     │
│  recon_error =     │              │  kNN distance to    │
│    ||x - x_recon||│              │  memory             │
└─────────────────────┘              └─────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  SCORING: score = max(recon_error, memory_distance)              │
│           score > beta → ANOMALY                                 │
└──────────────────────────────────────────────────────────────────┘
```

#### 1.3 Memory Module (BAR Controller)

```
┌─────────────────────────────────────────────────────────────────┐
│  MEMORY MODULE (50,000 slots)                                   │
│  ─────────────────────────────────────────────────────────────  │
│  Purpose: Lưu "normal patterns" để so sánh                     │
│                                                                  │
│  BAR Controller: Kiểm soát khi nào được phép update memory     │
│  ─────────────────────────────────────────────────────────────  │
│  • Drift detected by LOCAL ADWIN → UPDATE (budget granted)      │
│  • Minimum budget (1-5%) → UPDATE (maintenance)               │
│  • Otherwise → SKIP (save memory)                              │
│                                                                  │
│  ⚠️ ADWIN SCOPE: LOCAL ADWIN trong MemStream chỉ giám sát     │
│     anomaly_score để quyết định BAR budget. KHÔNG liên quan    │
│     đến GLOBAL ADWIN-U ở Layer 4.                             │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.4 ⚠️ CRITICAL FIX #3: Warmup PHẢI Offline

```
┌──────────────────────────────────────────────────────────────────┐
│  ⚠️ SAI LẦM THƯỜNG GẶP: Chạy warmup bên trong Flink          │
│  ─────────────────────────────────────────────────────────────   │
│  Flink TaskManager sẽ TIMEOUT (60s) nếu phải chạy 500 epochs  │
│  → Warmup bên trong Flink = PIPELINE CRASH                     │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  PHASE 0: OFFLINE (CHẠY TRƯỚC KHI START FLINK)              │
│  scripts/warmup_memstream.py                                     │
│  ─────────────────────────────────────────────────────────────  │
│  1. Load clean baseline data (Jan 2024)                        │
│  2. Extract 30D features                                       │
│  3. Train Denoising Autoencoder (500 epochs)                   │
│  4. Initialize memory from clean data                            │
│  5. Save checkpoint:                                           │
│     ├── memstream_weights.pt (AE weights)                       │
│     ├── memstream_scaler.pkl (Normalization)                    │
│     ├── memstream_memory.pt (50K memory slots)                  │
│     └── memstream_config.json (30D config)                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1: ONLINE (FLINK JOB)                                  │
│  MemStreamScoringOperator.open()                                │
│  ─────────────────────────────────────────────────────────────  │
│  1. torch.load(checkpoint) ← CHỈ TẢI WEIGHTS (milliseconds)  │
│  2. Load memory state ← CHỈ COPY TENSORS (milliseconds)       │
│  3. Set beta threshold                                        │
│  4. START INFERENCE IMMEDIATELY ← KHÔNG CÓ TRAINING!          │
└──────────────────────────────────────────────────────────────────┘

Deployment:
  # Step 1: Offline warmup (run once)
  python scripts/warmup_memstream.py --data data/jan_2024.parquet --output models/

  # Step 2: Start Flink (checkpoint đã sẵn sàng)
  flink run -c src.jobs.ca_dqstream_job target.jar --checkpoint models/
```

### LAYER 3: Voting Ensemble

```
Canary có violation  → FINAL = ANOMALY (canary wins, confidence=1.0)
ML flags anomaly      → FINAL = ANOMALY (confidence=score/threshold)
Both clean           → FINAL = CLEAN (confidence=1 - score/threshold)

MetaAggregator:
  Window: 1-minute TumblingEventTimeWindow
  Key-by: neighborhood (per zone)
  Output: 6 meta-metrics
    • volume, null_rate, violation_rate
    • anomaly_rate, avg_anomaly_score, delta_score
```

### LAYER 4: IEC (Intelligent Evolution Controller)

#### ⚠️ CRITICAL FIX #2: Redesign METER Strategies

```
┌─────────────────────────────────────────────────────────────────┐
│  ⚠️ SAI LẦM: Dùng retrain_model cho MemStream                 │
│  ─────────────────────────────────────────────────────────────  │
│  MemStream đã tự nó online learning → retrain là THỪA          │
│  METER cho IsolationForest (tĩnh) ≠ METER cho MemStream (động) │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  REVISED METER STRATEGIES (For MemStream)                      │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  0: do_nothing                                               │
│     → No drift → Continue normal operation                    │
│                                                                  │
│  1: adjust_threshold                                          │
│     → Minor drift → Adjust MemStream beta threshold            │
│     → anomaly_rate > 0.15 → beta = 0.55 (less sensitive)   │
│     → anomaly_rate < 0.03 → beta = 0.45 (more sensitive)    │
│     → Emit to memstream-beta-updates Kafka topic              │
│                                                                  │
│  2: lock_memory ← NEW (was retrain_model)                    │
│     → Poisoning detected → Freeze memory updates              │
│     → Prevent adversarial data from corrupting memory          │
│     → Emit to memstream-control Kafka topic                  │
│                                                                  │
│  3: canary_only ← NEW (was switch_model)                     │
│     → Severe drift → Switch to Canary-only mode              │
│     → MemStream temporarily disabled                         │
│     → Emit to memstream-control Kafka topic                  │
└─────────────────────────────────────────────────────────────────┘
```

#### ⚠️ CRITICAL FIX #4: ADWIN Scope Separation

```
┌─────────────────────────────────────────────────────────────────┐
│  DOUBLE ADWIN COLLISION (Must Avoid)                          │
│  ─────────────────────────────────────────────────────────────  │
│  MemStreamCore có LOCAL ADWIN → BAR Controller               │
│  IECOperator có GLOBAL ADWIN-U (36 instances)               │
│  → Nếu không phân tách rõ → 2 hệ thống CÃI NHANU nhau      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  SCOPE SEPARATION (CORRECTED)                                │
│  ═══════════════════════════════════════════════════════════   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  LOCAL ADWIN (Inside MemStreamCore/BARController)      │   │
│  │  ───────────────────────────────────────────────────  │   │
│  │  Scope: MICRO (per neighborhood, per record stream)    │   │
│  │  Target: anomaly_score from MemStream scoring        │   │
│  │  Purpose: BAR budget decision (update memory?)       │   │
│  │  Question: "Should I remember this record?"          │   │
│  │  Action: Add to memory or skip                       │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐   │
│  │  GLOBAL ADWIN-U (Inside IECOperator)                  │   │
│  │  ───────────────────────────────────────────────────  │   │
│  │  Scope: MACRO (system-wide, per 1-min window)        │   │
│  │  Target: 6 meta-metrics from MetaAggregator           │   │
│  │  Purpose: METER strategy decision                   │   │
│  │  Question: "Should I change system strategy?"        │   │
│  │  Action: Emit do_nothing/adjust/lock/canary signal  │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                  │
│  These two ADWIN systems operate INDEPENDENTLY.              │
│  LOCAL ADWIN does NOT affect GLOBAL ADWIN-U decisions.        │
│  GLOBAL ADWIN-U does NOT touch MemStream memory.             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tóm Tắt 4 Critical Fixes

| # | Fix | Status |
|---|-----|--------|
| 1 | **30D Feature Vector** (was 25D) | ✅ Updated |
| 2 | **METER Strategies** (no retrain, add lock_memory/canary_only) | ✅ Updated |
| 3 | **Warmup Offline** (was inside Flink) | ✅ Updated |
| 4 | **ADWIN Scope Separation** (LOCAL vs GLOBAL) | ✅ Documented |

---

## Điểm Quan Trọng Cần Nhớ

### 1. MemStream vs IsolationForest

| Aspect | MemStream (Online) | IsolationForest (Offline) |
|--------|-------------------|--------------------------|
| **Learning** | Online/streaming | Offline/batch |
| **Adaptation** | Memory updates via BAR | Need retrain |
| **Warmup** | **Offline only** | Fit on data |
| **Drift Detection** | LOCAL ADWIN for BAR | External |
| **METER Role** | Adjust threshold + Lock memory | Retrain model |

### 2. Canary Override

```
Canary violation → FINAL = ANOMALY (canary always wins)
ML anomaly only → FINAL = ANOMALY (only if canary clean)
Both clean → FINAL = CLEAN
```

### 3. ADWIN Two-Level

```
LOCAL ADWIN (MemStream) → BAR decision (memory update?)
GLOBAL ADWIN-U (IEC) → Strategy decision (adjust/lock/canary)
```

### 4. Deployment Order

```
1. Offline: python scripts/warmup_memstream.py
2. Online: Start Flink job (loads checkpoint only)
```

---

## Files Cần Update

| File | Change |
|------|--------|
| `src/ml/memstream_core.py` | Config: `in_dim=30, hidden_dim=60, out_dim=30` |
| `src/operators/memstream_scoring_operator.py` | Load checkpoint only, no warmup |
| `src/operators/iec_operator.py` | Replace retrain_model → lock_memory, switch_model → canary_only |
| `scripts/warmup_memstream.py` | NEW: Offline warmup script |
| Kafka topics | Add: `memstream-control`, `memstream-beta-updates` |

---

*Document version: 2.1 - Fixed 4 Critical Flaws - 2026-05-13*
