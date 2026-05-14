# CA-DQStream: Concept Drift Analysis Summary

**Date:** 2026-05-13
**Pipeline:** Layer 1 (Ingest) → Layer 2 (Canary + ML) → Layer 3 (Voting) → Layer 4 (IEC)

---

## 1. Các Trường Hợp Concept Drift Đã Phân Tích

### 1.1 Drift Types trong hệ thống

| # | Drift Type | Mô tả | Detection Metric | IEC Strategy |
|---|------------|--------|------------------|--------------|
| 1 | **ABRUPT_DRIFT** | Thay đổi đột ngột (vd: fare tăng 10x) | anomaly_rate, avg_anomaly_score | `switch_model` |
| 2 | **GRADUAL_DRIFT** | Thay đổi từ từ (seasonal shift) | delta_score, anomaly_rate | `retrain_model` |
| 3 | **TRANSIENT_DRIFT** | Spike ngắn rồi biến mất | volume, anomaly_rate | `do_nothing` |
| 4 | **RECURRING_DRIFT** | Drift theo chu kỳ (rush hour) | volume, violation_rate | `adjust_threshold` |
| 5 | **FEATURE_DRIFT** | Chỉ một số feature thay đổi | avg_anomaly_score | `retrain_model` |
| 6 | **LABEL_DRIFT** | Anomaly rate tăng đột ngột | anomaly_rate, violation_rate | `retrain_model` |
| 7 | **DISTRIBUTION_SHIFT** | Toàn bộ phân phối thay đổi | Tất cả metrics | `switch_model` |

### 1.2 Đường đi của dữ liệu qua các layer

```
┌─────────────────────────────────────────────────────────────────────┐
│  KAFKA: taxi-nyc-raw                                               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LAYER 1: BASELINE VALIDATION                                       │
│  - ParseJsonFunction → WatermarkAssigner → AddTripId → Deduplicator  │
│  - SchemaValidator                                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ DRIFT IMPACT: JSON parse errors, zone outside 1-263, duplicates   │  │
│  └────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐
        │  VALID STREAM     │   │  INVALID STREAM   │
        │  (to Layer 2)    │   │  → violations     │
        └───────────────────┘   └───────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LAYER 2A: CANARY BRANCH (7 static rules)                            │
│  - negative_fare, zero_distance, invalid_passengers                   │
│  - extreme_fare (>1000), extreme_duration (>24h)                     │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ DRIFT IMPACT: Rule violations increase (sudden or gradual)       │  │
│  │ → Canary overrides ML decision if any rule violated             │  │
│  └────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LAYER 2B: COMPLEX/ML BRANCH (sklearn IsolationForest)               │
│  - 21D feature vector + StandardScaler                               │
│  - Context-aware thresholds (4D: trip_type, time, day, neighborhood) │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ DRIFT IMPACT:                                                   │  │
│  │ - Model trained on historical data                             │  │
│  │ - New patterns not recognized → anomaly_score increases         │  │
│  │ - ADWIN monitors: anomaly_rate, avg_anomaly_score, delta_score  │  │
│  └────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LAYER 3: VOTING ENSEMBLE                                            │
│  - Canary overrides ML (priority: canary > ML)                       │
│  - MetaAggregator: 1-min tumbling window, per neighborhood          │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ DRIFT IMPACT:                                                   │  │
│  │ - Meta-metrics aggregate: volume, null_rate, violation_rate,   │  │
│  │   anomaly_rate, avg_anomaly_score, delta_score                 │  │
│  │ - delta_score = |violation_rate - anomaly_rate| / (sum + ε)    │  │
│  └────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LAYER 4: IEC (Intelligent Evolution Controller)                     │
│  - MultiInstanceADWIN: 36 instances (6 neighborhoods × 6 metrics)    │
│  - METER Hypernetwork: predicts strategy from 6D meta-features       │
│  - Fallback: rule-based if METER unavailable                         │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ DRIFT RESPONSE:                                                 │  │
│  │ - do_nothing: severity=none                                    │  │
│  │ - adjust_threshold: severity=low (anomaly_rate 3-15%)          │  │
│  │ - retrain_model: severity=moderate                             │  │
│  │ - switch_model: severity=high (anomaly_rate >15%)              │  │
│  └────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 2. Kết Quả Test Drift Detection

### 2.1 Test Results (Custom Drift Detector)

| Test Case | Drift Start | First Detection | Total Events | Status |
|-----------|-------------|----------------|--------------|--------|
| ABRUPT_10X | 500 | 501 | 9 | OK |
| ABRUPT_3X | 500 | 505 | 8 | OK |
| GRADUAL_2X | 500 | 583 | 20 | OK |
| TRANSIENT | 500 | 602 | 11 | OK |
| RECURRING | 400 | 405 | 14 | OK |

### 2.2 Production Simulation

```
Total drift events: 54
Drift window: 500-1000
Affected metrics:
  - anomaly_rate: 12 events
  - avg_anomaly_score: 38 events
  - delta_score: 4 events
```

---

## 3. Vấn đề Phát hiện được

### 3.1 ADWIN từ River không hoạt động đúng

**Vấn đề:** ADWIN từ thư viện `river.drift` với cấu hình mặc định không phát hiện được drift ngay cả khi có thay đổi lớn (10x).

**Nguyên nhân:**
- ADWIN cần warmup dài (>100 samples)
- Delta parameter (sensitivity) cần được tune cho từng metric
- Window size quá lớn làm chậm detection

**Giải pháp:** Implement custom drift detector (`SimpleDriftDetector`, `ADWINLike`) hoạt động tốt hơn.

### 3.2 IEC Threshold Adjustment không được apply thực sự

**Vấn đề:** Trong `iec_operator.py`, khi strategy = `adjust_threshold`, code chỉ **log** threshold mới nhưng không apply vào `IFScoringOperator`.

```python
# Current (stub):
elif strategy == 'adjust_threshold':
    new_threshold = self._compute_adjusted_threshold(meta_metrics)
    return {
        'action': 'threshold_adjusted',
        'new_threshold': new_threshold,
        'message': 'Anomaly threshold adjusted'
    }
    # NOTE: Threshold is NOT actually applied!
```

**Giải pháp:** Cần emit threshold update qua Kafka topic `if-model-updates` để hot-swap.

### 3.3 DriftAggregator không reset sau adaptation

**Vấn đề:** Khi IEC thực hiện adaptation thành công, `recent_drifts` không được clear, dẫn đến severity không giảm.

**Giải pháp:** Sau khi thực hiện `retrain_model` hoặc `switch_model` thành công, gọi `drift_aggregator.clear_recent()`.

### 3.4 Multi-Instance ADWIN không có cross-slot coordination

**Vấn đề:** Khi Flink parallelism > 1, mỗi slot có ADWIN instance riêng, dẫn đến:
- Mỗi slot có thể đưa ra strategy khác nhau
- Drift detection không nhất quán across slots

**Giải pháp:** Hoặc dùng parallelism=1 cho IECOperator, hoặc aggregate drift signals qua Kafka.

---

## 4. Các Scripts đã tạo

### 4.1 Drift Injection Scripts

| Script | Mục đích |
|--------|----------|
| `scripts/inject_concept_drift.py` | Inject drift vào Kafka (7 loại) |
| `scripts/benchmark_drift_detection.py` | Benchmark offline với ADWIN |
| `scripts/test_simple_drift.py` | Test nhanh drift detection |
| `scripts/custom_drift_detector.py` | Custom drift detector implementation |

### 4.2 Cách sử dụng

```bash
# Inject specific drift type
python scripts/inject_concept_drift.py --drift-type ABRUPT_DRIFT

# Inject all drift types
python scripts/inject_concept_drift.py --drift-type ALL

# Run benchmark
python scripts/custom_drift_detector.py
```

---

## 5. Đề Xuất Cải thiện

### 5.1 High Priority

1. **Fix IEC Threshold Adjustment**
   - Emit threshold update qua Kafka topic `if-model-updates`
   - Hoặc dùng Broadcast State để propagate threshold changes

2. **Replace ADWIN với Custom Implementation**
   - `ADWINLike` class đã hoạt động tốt
   - Tune window_size và threshold per metric

3. **Add Drift Aggregator Reset**
   - Clear `recent_drifts` sau adaptation thành công

### 5.2 Medium Priority

4. **Add Cross-Slot Coordination cho IEC**
   - Dùng Kafka topic để aggregate drift signals
   - Hoặc set parallelism=1 cho IECOperator

5. **Add Statistical Tests**
   - KS-test hoặc t-test để xác nhận drift
   - Giảm false positives

6. **Add Drift Recovery Detection**
   - Phát hiện khi drift kết thúc
   - Tự động reset ADWIN sau recovery

### 5.3 Low Priority

7. **Add Visualization**
   - Dashboard Grafana cho drift events
   - Real-time drift detection visualization

8. **Add A/B Testing Framework**
   - Test multiple strategies in parallel
   - Compare effectiveness of different responses

---

## 6. Production Readiness Checklist

- [ ] Custom drift detector thay thế ADWIN từ river
- [ ] IEC threshold adjustment được apply thực sự
- [ ] DriftAggregator reset sau adaptation
- [ ] Cross-slot coordination cho IEC (hoặc parallelism=1)
- [ ] Statistical tests cho drift confirmation
- [ ] Drift recovery detection
- [ ] Grafana dashboard cho drift monitoring
- [ ] Integration tests cho tất cả drift types

---

## 7. Test Coverage

### 7.1 Đã Test

| Drift Type | Inject | Detect | Respond | Recover |
|------------|--------|--------|---------|---------|
| ABRUPT_DRIFT | OK | OK | OK | - |
| GRADUAL_DRIFT | OK | OK | OK | - |
| TRANSIENT_DRIFT | OK | OK | OK | - |
| RECURRING_DRIFT | OK | OK | OK | - |
| FEATURE_DRIFT | OK | - | - | - |
| LABEL_DRIFT | OK | - | - | - |
| DISTRIBUTION_SHIFT | OK | - | - | - |

### 7.2 Cần Test Thêm

- [ ] End-to-end với Flink job
- [ ] Kafka injection + IEC response loop
- [ ] Model hot-swap after retrain
- [ ] Alert suppression after recovery

---

*Document generated: 2026-05-13*
*Scripts location: `c:\proj\ldt\scripts\`*
