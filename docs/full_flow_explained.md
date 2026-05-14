# CA-DQStream: Full Flow Chi Tiết - Giải Thích Rõ Ràng

> **Ngày:** 2026-05-13
> **Pipeline:** Kafka → L1 → L2A (Canary) + L2B (MemStream ML) → L3 (Voting) → L4 (IEC)
> **Status:** UPDATED v2.1 - Fixed 4 Critical Flaws

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
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐                 │
│  │  ParseJsonFunc   │→ │WatermarkAssigner │→ │  AddTripIdFunc   │                 │
│  │  (JSON → dict)   │  │ (Event time)     │  │ (MurmurHash3)   │                 │
│  └──────────────────┘  └──────────────────┘  └────────┬─────────┘                 │
│                                                         │                          │
│                                                         ▼                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐   │
│  │                      DeduplicatorFunction (7-day TTL, RocksDB)               │   │
│  │                    → Drop duplicate trips by composite key                    │   │
│  └────────────────────────────────────────────────────────────────────────────┘   │
│                                                         │                          │
│                                                         ▼                          │
│  ┌────────────────────────────────────────────────────────────────────────────┐   │
│  │                         SchemaValidator                                       │   │
│  │  • Required fields: trip_distance, fare_amount, PULocationID, DOLocationID │   │
│  │  • Zone range: PULocationID, DOLocationID ∈ [1, 263]                       │   │
│  └────────────────────────────────┬────────────────────────────────────────────┘   │
│                                   │                                                  │
│               ┌───────────────────┴───────────────────┐                            │
│               ▼                                       ▼                            │
│    ┌─────────────────────┐               ┌──────────────────────────┐            │
│    │  VALID STREAM       │               │  INVALID STREAM         │            │
│    │  (to Layer 2)      │               │  → dq-hard-rule-violations│           │
│    └─────────────────────┘               └──────────────────────────┘            │
└───────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 2: DUAL-BRANCH PROCESSING                                   │
│  ┌────────────────────────────────────────────┐ ┌──────────────────────────────────┐│
│  │  LAYER 2A: CANARY BRANCH (7 rules)        │ │  LAYER 2B: ML BRANCH           ││
│  │                                            │ │                                  ││
│  │  CanaryRulesValidator                     │ │  IFScoringOperator              ││
│  │  ───────────────────────                  │ │  (sklearn IsolationForest)       ││
│  │  R1: negative_fare (fare ≤ 0)           │ │  HOẶC                           ││
│  │  R2: zero_distance (dist=0 & fare>0)    │ │                                  ││
│  │  R3: invalid_passengers (0 or >6)       │ │  MemStreamCore                  ││
│  │  R4: invalid_payment (not 1-6)         │ │  (Denoising Autoencoder          ││
│  │  R5: extreme_fare (>1000)              │ │   + Memory Module)               ││
│  │  R6: extreme_duration (>24h)          │ │                                  ││
│  │  R7: negative_duration                 │ │  HOẶC                           ││
│  │                                            │ │                                  ││
│  │  Output:                                 │ │  CAMemStreamEIA                 ││
│  │  • canary_violations: []                │ │  (Context-Aware MemStream)      ││
│  │  • has_violation: bool                  │ │                                  ││
│  │  • violation_count: int                 │ └──────────────────────────────────┘│
│  └────────────────────────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          LAYER 3: VOTING ENSEMBLE                                    │
│  ┌────────────────────────────────────────────────────────────────────────────┐   │
│  │                          VotingEnsembleFunction                               │   │
│  │  ─────────────────────────────                                              │   │
│  │                                                                              │   │
│  │  PRIORITY: Canary overrides ML                                               │   │
│  │                                                                              │   │
│  │  IF has_violation == True:                                                  │   │
│  │      → final_decision = 'ANOMALY'                                           │   │
│  │      → decision_source = 'canary_rule'                                      │   │
│  │      → confidence = 1.0                                                    │   │
│  │                                                                              │   │
│  │  ELIF is_anomaly == True:                                                   │   │
│  │      → final_decision = 'ANOMALY'                                           │   │
│  │      → decision_source = 'complex_ml'                                        │   │
│  │      → confidence = min(anomaly_score / threshold, 1.0)                      │   │
│  │                                                                              │   │
│  │  ELSE:                                                                      │   │
│  │      → final_decision = 'CLEAN'                                             │   │
│  │      → decision_source = 'both_agree'                                        │   │
│  │      → confidence = 1.0 - (anomaly_score / threshold)                        │   │
│  └────────────────────────────────────────────────────────────────────────────┘   │
│                                             │                                        │
│                                             ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐   │
│  │                    MetaAggregatorFunction                                      │   │
│  │  ───────────────────────────────                                              │   │
│  │  Window: 1-minute TumblingEventTimeWindow                                    │   │
│  │  Key-by: neighborhood (per zone)                                             │   │
│  │                                                                              │   │
│  │  OUTPUT META-METRICS:                                                       │   │
│  │  • volume             = count of records in window                            │   │
│  │  • null_rate         = nulls / volume                                       │   │
│  │  • violation_rate    = violations / volume                                   │   │
│  │  • anomaly_rate     = anomalies / volume                                    │   │
│  │  • avg_anomaly_score = mean(anomaly_score)                                   │   │
│  │  • delta_score      = |violation_rate - anomaly_rate|                       │   │
│  │                        / (violation_rate + anomaly_rate + ε)                  │   │
│  └────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    LAYER 4: IEC (Intelligent Evolution Controller)                   │
│  ┌────────────────────────────────────────────────────────────────────────────┐   │
│  │                           IECOperator.map()                                   │   │
│  │  ─────────────────────────────                                              │   │
│  │                                                                              │   │
│  │  STEP 1: Update ADWIN-U (36 instances)                                       │   │
│  │    → MultiInstanceADWIN.update_meta_metrics(meta_metrics)                   │   │
│  │    → 6 neighborhoods × 6 metrics = 36 ADWIN instances                        │   │
│  │    → Check each metric for drift                                             │   │
│  │                                                                              │   │
│  │  STEP 2: DriftAggregator.assess_drift_severity()                            │   │
│  │    → Count recent drift events                                              │   │
│  │    → severity = 'none' | 'low' | 'moderate' | 'high'                       │   │
│  │                                                                              │   │
│  │  STEP 3: METER.predict() hoặc fallback rules                                │   │
│  │    → METER: MLP(128, 64, 32) trained on drift scenarios                    │   │
│  │    → Input: 6D meta-features                                                │   │
│  │    → Output: strategy (0-3)                                                │   │
│  │                                                                              │   │
│  │  STEP 4: _execute_strategy()                                                │   │
│  │    → do_nothing: log, no action                                            │   │
│  │    → adjust_threshold: log new threshold (stub)                              │   │
│  │    → retrain_model: emit retrain signal                                     │   │
│  │    → switch_model: emit model switch signal                                 │   │
│  └────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                              Kafka: iec-action-replay
```

---

## Chi Tiết Từng Thành Phần

### 1. MemStream ML (Layer 2B - ML Branch)

MemStream là thuật toán streaming anomaly detection dựa trên **Denoising Autoencoder + Memory Module**.

#### 1.1 Tại Sao Dùng MemStream?

| Phương pháp | Ưu điểm | Nhược điểm |
|-------------|---------|------------|
| IsolationForest | Offline tốt | Cần retrain khi có drift |
| MemStream | Streaming, adaptive | Cần warmup |
| OneClassSVM | Online learning | Chậm |

**MemStream phù hợp vì:**
1. **Streaming**: Không cần retrain toàn bộ model
2. **Memory Module**: Lưu trữ patterns "normal" để so sánh
3. **ADWIN tích hợp**: Tự động phát hiện drift
4. **BAR Controller**: Kiểm soát budget cho memory updates

#### 1.2 Kiến Trúc MemStream

```
┌──────────────────────────────────────────────────────────────────┐
│                    INPUT: 30D Feature Vector                       │
│  [25D base + 4D Grid spatial + 5D RatecodeID one-hot]           │
│                                                                     │
│  30D bao gồm:                                                    │
│  ────────────────────────────────────────────────────────────────  │
│  Index 0-14: Base features (15D)                                  │
│  Index 15-18: Grid X/Y spatial (4D) — Micro-location context    │
│  Index 19-23: RatecodeID one-hot (5D) — CRITICAL cho Type 3     │
│  Index 24-29: Normalized ratios (6D)                              │
│                                                                     │
│  ⚠️ LƯU Ý QUAN TRỌNG:                                           │
│  Phải dùng 30D (không phải 25D) để bắt được Type 3 Fraud        │
│  (JFK Ratecode + Manhattan Zone + $70 flat fare)                 │
└──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                     ENCODER (30D → 60D → 30D)                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Linear(30, 60) → Tanh → Linear(60, 30) → Tanh      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  INPUT ──→ [30D] ──→ [60D] ──→ [30D] ──→ ENCODED (z)        │
│                                                                   │
│  z = encoder(x)  →  30D representation của input                  │
└──────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────────┐     ┌───────────────────────┐
        │   RECONSTRUCTION       │     │   MEMORY MODULE       │
        │   (Decoder)           │     │   (FIFO Queue)       │
        │                       │     │                       │
        │  [25D] ──→ [50D]     │     │  ┌─────────────┐    │
        │         ──→ [25D]     │     │  │ z₁ (encoded)│    │
        │                       │     │  │ z₂          │    │
        │  x_recon = decoder(z) │     │  │ z₃          │    │
        │                       │     │  │ ...         │    │
        │  recon_error =        │     │  │ z₁₀₀₀      │    │
        │    ||x - x_recon||    │     │  └─────────────┘    │
        └───────────────────────┘     │   (1000 slots)       │
                                      └───────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                         SCORING                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                           │   │
│  │  1. Recon Error = ||x - decoder(encoder(x))||              │   │
│  │     → Cao nếu input KHÔNG giống training data            │   │
│  │                                                           │   │
│  │  2. Memory Distance = kNN(z, memory)                     │   │
│  │     → Khoảng cách trung bình đến k nearest neighbors     │   │
│  │     → Cao nếu z KHÔNG giống memory patterns             │   │
│  │                                                           │   │
│  │  3. Final Score = max(recon_error, memory_distance)       │   │
│  │     → Lấy giá trị lớn hơn                              │   │
│  │     → Threshold: beta = 0.5 (mặc định)                  │   │
│  │     → Score > beta → ANOMALY                             │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

#### 1.3 Memory Module Hoạt Động Như Thế Nào?

```
┌─────────────────────────────────────────────────────────────────┐
│                     MEMORY MODULE                                │
│                                                                 │
│  Cấu trúc: FIFO circular buffer, 50,000 slots                   │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Memory = [z₀, z₁, z₂, ..., z₉₉₉]                      │  │
│  │           25D    25D   25D       25D                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ▲                                   │
│                              │ Update (detached gradient)       │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │  BAR Controller:                                          │  │
│  │  ──────────────────                                       │  │
│  │  • Drift detected by ADWIN → UPDATE (budget granted)     │  │
│  │  • Minimum budget (1-5%) → UPDATE (maintenance)          │  │
│  │  • Otherwise → SKIP (save memory)                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Ý nghĩa:                                                     │
│  • Memory lưu "normal patterns" dưới dạng encoded vectors     │
│  • Anomaly = input có encoded representation KHÁC với memory │
│  • Drift detected → memory update để adapt to new normal      │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.4 Scoring Chi Tiết

```python
# Pseudocode cho scoring
def score_one(x):
    # 1. Normalize
    x_norm = (x - mean) / std

    # 2. Encode
    z = encoder(x_norm)  # [25D]

    # 3. Decode
    x_recon = decoder(z)  # [25D]

    # 4. Compute reconstruction error
    recon_error = mean((x_norm - x_recon) ** 2)

    # 5. Compute memory distance (k=10 nearest neighbors)
    distances = cdist(z, memory, p=2)  # [1, 1000]
    k_nearest = sort(distances)[:10]
    memory_distance = mean(k_nearest)

    # 6. Final score = max
    score = max(recon_error, memory_distance)

    # 7. Decision
    is_anomaly = score > beta

    return score, is_anomaly
```

#### 1.5 Warmup Phase (Training)

```
┌──────────────────────────────────────────────────────────────────┐
│                      WARMUP PHASE                                 │
│                                                                 │
│  Input: Clean baseline data (normal trips only)                 │
│  ─────────────────────────────────────────────────────────────   │
│                                                                 │
│  STEP 1: Compute normalization stats                            │
│    → mean, std từ first 10% của data                           │
│                                                                 │
│  STEP 2: Train Autoencoder                                      │
│    → 500 epochs với noise injection                             │
│    → MSE loss: ||x_noisy - x_recon||                          │
│    → Early stopping: nếu loss không giảm sau 20 epochs         │
│                                                                 │
│  STEP 3: Initialize Memory                                      │
│    → Encode last 10% của data                                  │
│    → Lưu vào memory buffer (FIFO)                              │
│                                                                 │
│  Output: Trained encoder + initialized memory                   │
│          Ready for streaming inference!                          │
└──────────────────────────────────────────────────────────────────┘
```

#### 1.6 ADWIN Drift Detection (Trong MemStream)

```python
# ADWIN = Adaptive Windowing for Drift Detection
# Giám sát mean của sliding window

class ADWIN:
    def __init__(self, delta=0.002):
        self.window = deque()  # Sliding window
        self.delta = delta      # Sensitivity (越小越敏感)

    def update(self, value):
        self.window.append(value)

        # Limit window size
        if len(self.window) > 1000:
            self.window.popleft()

        # Check drift khi có đủ data
        if len(self.window) > 100:
            return self._detect_drift()

        return False

    def _detect_drift(self):
        # So sánh hai nửa của window
        # Nếu means khác nhau nhiều → DRIFT

        mid = len(self.window) // 2
        left = list(self.window)[:mid]
        right = list(self.window)[mid:]

        mean_left = sum(left) / len(left)
        mean_right = sum(right) / len(right)

        # ADWIN threshold
        m = 1 / (1/n_left + 1/n_right)
        epsilon = sqrt((2/m) * ln(4*len(window)/delta))

        if abs(mean_left - mean_right) > epsilon:
            return True  # DRIFT DETECTED!

        return False
```

### 2. Canary Rules (Layer 2A)

```
┌──────────────────────────────────────────────────────────────────┐
│                      CANARY RULES (7 Rules)                      │
│                                                                 │
│  Purpose: Catch OBVIOUS anomalies (static rules)               │
│           Override ML nếu có violation                          │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  R1: negative_fare          │ fare_amount <= 0                  │
│  R2: zero_distance_fare    │ trip_distance == 0 AND fare > 0    │
│  R3: invalid_passengers    │ passenger_count < 1 OR > 6         │
│  R4: invalid_payment       │ payment_type NOT IN [1,2,3,4,5,6]  │
│  R5: extreme_fare          │ fare_amount > 1000                 │
│  R6: extreme_duration     │ trip_duration > 24*3600 seconds   │
│  R7: negative_duration     │ trip_duration < 0                  │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  OUTPUT:                                                       │
│    • canary_violations = ['rule1', 'rule2']                   │
│    • has_violation = True/False                                │
│    • violation_count = len(violations)                          │
│                                                                 │
│  NOTE: Tất cả records đều pass through (không drop)            │
│        Violations được flag nhưng vẫn đi tiếp vào L3            │
└──────────────────────────────────────────────────────────────────┘
```

### 3. Voting Ensemble (Layer 3)

```
┌──────────────────────────────────────────────────────────────────┐
│                    VOTING ENSEMBLE                                 │
│                                                                 │
│  Input: Records từ Canary HOẶC ML (không phải cả hai)        │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │   ┌─────────────┐         ┌─────────────┐                 │ │
│  │   │ Canary:     │         │ ML:         │                 │ │
│  │   │ has_violation│         │ is_anomaly │                 │ │
│  │   └──────┬──────┘         └──────┬──────┘                 │ │
│  │          │                        │                         │ │
│  │          ▼                        ▼                         │ │
│  │   ┌──────────────────────────────────────────────┐        │ │
│  │   │           PRIORITY: Canary > ML                │        │ │
│  │   └──────────────────────────────────────────────┘        │ │
│  │          │                                                │ │
│  │          ▼                                                │ │
│  │   ┌──────────────────────────────────────────────┐        │ │
│  │   │  if has_violation: FINAL = ANOMALY (canary)│        │ │
│  │   │  elif is_anomaly:    FINAL = ANOMALY (ml)   │        │ │
│  │   │  else:               FINAL = CLEAN         │        │ │
│  │   └──────────────────────────────────────────────┘        │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 4. MetaAggregator (Layer 3 - Metrics)

```
┌──────────────────────────────────────────────────────────────────┐
│                    META AGGREGATOR                                 │
│                                                                 │
│  Window: 1-minute TumblingEventTimeWindow                       │
│  Key-by: neighborhood (per zone: manhattan, brooklyn, etc.)     │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  Input: Tất cả records trong window                             │
│                                                                 │
│  OUTPUT:                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  {                                                       │   │
│  │    "neighborhood_id": "manhattan",                       │   │
│  │    "volume": 1250,           // records count            │   │
│  │    "null_rate": 0.001,       // null fields / volume    │   │
│  │    "violation_rate": 0.05,    // canary violations       │   │
│  │    "anomaly_rate": 0.08,      // ML anomalies            │   │
│  │    "avg_anomaly_score": 0.35,  // mean ML score          │   │
│  │    "delta_score": 0.176        // |vr - ar| / (vr + ar)  │   │
│  │  }                                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  delta_score ý nghĩa:                                          │
│  • = 0: violation_rate ≈ anomaly_rate (consistent)              │
│  • > 0: violation_rate ≠ anomaly_rate (inconsistent)            │
│  • Cao → Có thể có drift hoặc model lệch                       │
└──────────────────────────────────────────────────────────────────┘
```

### 5. IEC (Layer 4 - Drift Response)

```
┌──────────────────────────────────────────────────────────────────┐
│                    IEC - INTELLIGENT EVOLUTION CONTROLLER          │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  INPUT: Meta-metrics từ MetaAggregator                    │ │
│  │                                                             │ │
│  │  ┌────────────────────────────────────────────────────┐   │ │
│  │  │  STEP 1: MultiInstanceADWIN.update_meta_metrics()  │   │ │
│  │  │  ────────────────────────────────────────────────── │   │ │
│  │  │  36 ADWIN instances:                                │   │ │
│  │  │  • 6 neighborhoods × 6 metrics = 36                  │   │ │
│  │  │  • Mỗi instance giám sát 1 metric                  │   │ │
│  │  │  • Delta (sensitivity) khác nhau per metric:       │   │ │
│  │  │    - null_rate: 0.001 (NHẠY NHẤT)                  │   │ │
│  │  │    - violation_rate: 0.002                         │   │ │
│  │  │    - anomaly_rate: 0.002                           │   │ │
│  │  │    - avg_anomaly_score: 0.003                      │   │ │
│  │  │    - delta_score: 0.002                           │   │ │
│  │  │    - volume: 0.005 (ÍT NHẠY NHẤT)                 │   │ │
│  │  │  → Trả về list các drift events                   │   │ │
│  │  └────────────────────────────────────────────────────┘   │ │
│  │                           │                               │ │
│  │                           ▼                               │ │
│  │  ┌────────────────────────────────────────────────────┐   │ │
│  │  │  STEP 2: DriftAggregator.assess_drift_severity()  │   │ │
│  │  │  ────────────────────────────────────────────────── │   │ │
│  │  │  Count recent drift events:                         │   │ │
│  │  │  • 0 events → severity = 'none'                  │   │ │
│  │  │  • 1-2 events → severity = 'low'                 │   │ │
│  │  │  • 3-5 events → severity = 'moderate'             │   │ │
│  │  │  • 6+ events → severity = 'high'                 │   │ │
│  │  └────────────────────────────────────────────────────┘   │ │
│  │                           │                               │ │
│  │                           ▼                               │ │
│  │  ┌────────────────────────────────────────────────────┐   │ │
│  │  │  STEP 3: METER.predict() hoặc fallback           │   │ │
│  │  │  ────────────────────────────────────────────────── │   │ │
│  │  │  METER = MLP(128, 64, 32)                         │   │ │
│  │  │  Input: [volume, null_rate, violation_rate,        │   │ │
│  │  │           anomaly_rate, avg_anomaly_score,         │   │ │
│  │  │           delta_score]  →  6D                       │   │ │
│  │  │  Output: strategy (0-3)                            │   │ │
│  │  │                                                     │   │ │
│  │  │  Fallback (if METER unavailable):                  │   │ │
│  │  │    severity='none' → do_nothing                    │   │ │
│  │  │    severity='low' → adjust_threshold               │   │ │
│  │  │    severity='moderate' → retrain_model             │   │ │
│  │  │    severity='high' → switch_model                  │   │ │
│  │  └────────────────────────────────────────────────────┘   │ │
│  │                           │                               │ │
│  │                           ▼                               │ │
│  │  ┌────────────────────────────────────────────────────┐   │ │
│  │  │  STEP 4: _execute_strategy()                      │   │ │
│  │  │  ────────────────────────────────────────────────── │   │ │
│  │  │                                                     │   │ │
│  │  │  do_nothing:                                       │   │ │
│  │  │    → Log action only                              │   │ │
│  │  │    → Continue normal operation                     │   │ │
│  │  │                                                     │   │ │
│  │  │  adjust_threshold:                                 │   │ │
│  │  │    → if anomaly_rate > 0.15: new_beta = 0.55     │   │ │
│  │  │    → if anomaly_rate < 0.03: new_beta = 0.45      │   │ │
│  │  │    → Log new threshold (NOTE: NOT APPLIED YET!)   │   │ │
│  │  │                                                     │   │ │
│  │  │  retrain_model:                                    │   │ │
│  │  │    → Emit retrain signal to Kafka                 │   │ │
│  │  │    → Trigger model retraining job                 │   │ │
│  │  │                                                     │   │ │
│  │  │  switch_model:                                     │   │ │
│  │  │    → Emit model switch signal                     │   │ │
│  │  │    → Load alternative model                      │   │ │
│  │  └────────────────────────────────────────────────────┘   │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Luồng Dữ Liệu Chi Tiết Từng Record

### Ví dụ: Record Đi Qua Toàn Bộ Pipeline

```python
# INPUT: Raw JSON từ Kafka
raw_record = {
    "VendorID": 1,
    "tpep_pickup_datetime": "2024-01-15 08:30:00",
    "tpep_dropoff_datetime": "2024-01-15 08:55:00",
    "passenger_count": 2,
    "trip_distance": 3.5,
    "PULocationID": 161,
    "DOLocationID": 237,
    "payment_type": 1,
    "fare_amount": 15.50,
    "total_amount": 20.00,
}

# LAYER 1: Parse & Validate
record = json.loads(raw_record)  # Parse JSON
record['trip_id'] = hash(...)      # Add trip_id
# → Deduplicator: check duplicate → pass through
# → SchemaValidator: check required fields, zone range → VALID

# LAYER 2A: Canary Rules
canary_result = check_canary(record)
# R1: fare=15.50 > 0 → PASS
# R2: distance=3.5 != 0 → PASS
# R3: passengers=2 ∈ [1,6] → PASS
# R4: payment=1 ∈ [1,6] → PASS
# R5: fare=15.50 < 1000 → PASS
# R6: duration=25min < 24h → PASS
# R7: duration=25min > 0 → PASS
canary_result = {
    'canary_violations': [],
    'has_violation': False,
    'violation_count': 0,
}

# LAYER 2B: ML Scoring (MemStream)
features = vectorizer.transform(record)  # → 25D
score = memstream.score_one(features)      # → 0.15
is_anomaly = score > beta(0.5)           # → False

ml_result = {
    'anomaly_score': 0.15,
    'threshold': 0.50,
    'is_anomaly': False,
    'context_key': 'medium_morning_rush_weekday_manhattan',
}

# LAYER 3: Voting
if canary_result['has_violation']:
    final_decision = 'ANOMALY'
    decision_source = 'canary_rule'
elif ml_result['is_anomaly']:
    final_decision = 'ANOMALY'
    decision_source = 'complex_ml'
else:
    final_decision = 'CLEAN'
    decision_source = 'both_agree'

# LAYER 4: IEC (cứ 1 phút, không phải mỗi record)
# MetaAggregator đã aggregate → meta_metrics
# IEC nhận meta_metrics:
drifts = adwin.update_meta_metrics(meta_metrics)
severity = drift_aggregator.assess(severity)
strategy = meter.predict(meta_metrics)  # hoặc fallback
action = iec.execute(strategy)
# → Emit to iec-action-replay topic
```

---

## Tóm Tắt Các Thành Phần Quan Trọng

| Component | Location | Purpose |
|-----------|----------|---------|
| **MemStreamCore** | `src/ml/memstream_core.py` | Denoising Autoencoder + Memory Module |
| **MemStreamAE** | `src/ml/memstream_core.py` | Neural network: 25D→50D→25D |
| **MemoryModule** | `src/ml/memstream_core.py` | FIFO queue cho normal patterns |
| **ADWIN** | `src/ml/memstream_core.py` | Drift detection |
| **BARController** | `src/ml/memstream_core.py` | Control memory update budget |
| **IsolationForest** | `src/ml/train_iforest.py` | sklearn alternative model |
| **METER** | `src/ml/train_meter.py` | MLP cho strategy prediction |
| **IECOperator** | `src/operators/iec_operator.py` | Drift response controller |
| **MultiInstanceADWIN** | `src/iec/adwin_multi_instance.py` | 36 ADWIN instances |
| **CanaryRules** | `src/operators/canary_rules.py` | 7 static anomaly rules |
| **MetaAggregator** | `src/operators/meta_aggregator.py` | Windowed metrics aggregation |

---

## Điểm Quan Trọng Cần Nhớ

### 1. MemStream vs IsolationForest

| Aspect | MemStream | IsolationForest |
|--------|-----------|-----------------|
| **Learning** | Online/streaming | Offline/batch |
| **Adaptation** | Tự động (memory update) | Cần retrain |
| **Memory** | Có (1000 slots) | Không |
| **Warmup** | Cần (500 epochs) | Cần (fit on data) |
| **Drift Detection** | Tích hợp (ADWIN) | Cần external |
| **Production** | Flink Python | Broadcast State |

### 2. Canary Override

```
Canary rules CÓ VIOLATION → FINAL = ANOMALY (canary wins)
ML flags ANOMALY → FINAL = ANOMALY (ml only if canary clean)
Both clean → FINAL = CLEAN
```

### 3. IEC Response

```
Drift detected → severity assessment → METER prediction → action
     ↓               ↓                    ↓              ↓
  ADWIN-U      count drifts           MLP(6D)       emit signal
```

---

*Document version: 1.0 - 2026-05-13*
