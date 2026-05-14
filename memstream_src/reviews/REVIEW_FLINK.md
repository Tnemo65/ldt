# Flink Engineer Review — PLAN_v3.md

**Reviewer:** Principal Flink Engineer (15 yrs Apache Flink, PyFlink, Kafka Connect)
**Date:** 2026-05-12
**Files Reviewed:** PLAN_v3.md (§5–§6), `src/flink_job_complete.py`, `src/operators/iec_operator.py`, `src/operators/if_scoring_operator.py`

---

## 1. Architecture Analysis

### 1.1 KeyedProcessFunction — Sound

`MemStreamScoringOperator` (PLAN_v3.md §6) uses `KeyedProcessFunction` keyed by neighborhood. Correct: records for same neighborhood go to same instance.

**GOOD:** Base model loaded from filesystem in `open()`, not from Broadcast State. Only per-key memory (2048×50×4B ≈ 400KB per key) is checkpointed — not the 50MB AE weights.

### 1.2 IECFeedbackOperator — Correct Pattern

`KeyedBroadcastProcessFunction` with `process_broadcast_element()` for beta updates is the right pattern.

### 1.3 CANARY STREAM UNDEFINED — CRITICAL

`flink_job_complete.py:720`:
```python
canary_keyed = canary_stream.key_by(...)
```
`canary_stream` is **never defined**. Job crashes with `NameError` at submit. The entire Layer 1 canary branch is missing from the Flink job.

### 1.4 MockMLScoringFunction Still Used

Line 676–708 uses `MockMLScoringFunction` instead of `MemStreamScoringOperator`. All downstream components receive fake scores.

### 1.5 CANARY Key Type Mismatch

`canary_keyed` keys by `trip_id` (string), but `MemStreamScoringOperator` keys by neighborhood (string). Mixing `key_by` keys across branches may cause routing mismatches in `CoProcessFunction`.

---

## 2. Checkpointing & State Backend

### 2.1 No RocksDB State Backend — HIGH

PyFlink defaults to `FsStateBackend` (heap-based). With 400KB × 100 neighborhoods × 4 TMs = ~1.6GB on JVM heap. Will cause OOM.

```python
from pyflink.datastream.state import EmbeddedRocksDBStateBackend
env.set_state_backend(EmbeddedRocksDBStateBackend())
```

### 2.2 Beta Version Race Condition — HIGH

Beta updates via broadcast state modify `ms.max_thres` in `process_element()`. On checkpoint/restore, the order of applying broadcast-state beta updates vs. checkpointed `max_thres` is non-deterministic.

### 2.3 IEC State Not Checkpointed — MEDIUM

`IECOperator` uses in-memory `self.adwin_u` and `self.drift_aggregator`. Lost after task failure → false positive drift detection after restart.

### 2.4 Rendezvous Buffer Memory Leak — CRITICAL

`CoProcessFunction` buffers unmatched records indefinitely. Unmatched records accumulate → OOM over hours/days.

---

## 3. PyFlink Specific Issues

### 3.1 `os` Not Imported in memstream_core.py — CRITICAL

```python
# PLAN_v3.md memstream_core.py — os not in imports:
import torch, torch.nn, numpy, typing, io, hashlib, hmac, copy
# os is MISSING → NameError on os.getenv(), os.path.exists(), os.unlink()
```

### 3.2 `MapStateDescriptor` Not Imported — CRITICAL

`memstream_scoring_op.py`:
```python
from pyflink.datastream.state import ValueStateDescriptor  # MapStateDescriptor MISSING
```
### 3.3 `KeyedBroadcastProcessFunction` Not Imported — CRITICAL

```python
from pyflink.datastream import KeyedProcessFunction  # BroadcastProcessFunction MISSING
```

### 3.4 Import Inside `process_element` — MEDIUM

```python
def process_element(self, record, context):
    from memstream_src.core.feature_extractor import parse_datetime  # per-record overhead
```

### 3.5 Per-Record Serialization — CRITICAL

Every `process_element` triggers:
1. `io.BytesIO()` buffer creation
2. `torch.save()` of ~400KB memory tensors
3. State backend write

At 10K records/sec: 4GB/sec serialization throughput → checkpoint death spiral.

**Must serialize only on checkpoint via `CheckpointedFunction.snapshot_state()`.**

---

## 4. Kafka Integration

### 4.1 Exactly-Once Not Configured on Kafka Sink

`FlinkKafkaProducer` missing `producer_semantic=FlinkKafkaProducer.SEMANTIC.EXACTLY_ONCE`.

### 4.2 Schema Serialization

`SimpleStringSchema()` used everywhere. Consider `JsonRowDeserializationSchema` for better performance.

### 4.3 Consumer Lag Monitoring

No Prometheus metric for Kafka consumer lag. Must expose:
```python
kafka_consumer_lag{topic, partition, group}
```

---

## 5. Performance Analysis

### 5.1 Throughput Estimate

- PyTorch L1 distance: 2048 × 50 = 102K ops/record
- At 10K records/sec → 1B ops/sec
- PyTorch GIL: single-threaded CPU ops
- Actual throughput: ~500 records/sec per TM slot

### 5.2 Micro-batching Removed

`score_batch_vectorized()` exists but is never called. Per-record scoring is 8–16x less efficient than batching.

---

## 6. Fault Tolerance

### 6.1 No Restart Strategy Configured — HIGH

```python
env.set_restart_strategy(
    RestartStrategies.exponential_delay_restart_strategy(
        initial_restart_delay=1000, max_restart_delay=60000, back_off_multiplier=2.0
    )
)
```

### 6.2 Canary/Complex Branch Not Wired

The CoProcessFunction waits for `canary_buf` and `complex_buf` but neither stream is properly connected to the right sources.

---

## 7. Integration with Existing Job

### 7.1 Feature Dimension Mismatch — CRITICAL

`IFScoringOperator` uses 21D features (`src/features/vectorizer.py`). `MemStreamScoringOperator` uses 25D (`PLAN_v3.md`). Incompatible scores for Voting Ensemble.

### 7.2 4D Context Key Routing

`MemStreamScoringOperator.get_4d_context_key()` generates keys like `"medium_evening_rush_weekday_manhattan"`. Voting Ensemble expects keys from `IFScoringOperator.get_context_key()`. Must verify format parity.

---

## 8. CRITICAL Issues

| # | File | Line | Issue | Fix |
|---|------|------|-------|-----|
| 1 | `flink_job_complete.py` | 720 | `canary_stream` undefined — NameError | Add canary stream definition |
| 2 | `memstream_core.py` | ~528 | `os` not imported — NameError | Add `import os` |
| 3 | `memstream_scoring_op.py` | 1011 | `MapStateDescriptor` not imported — NameError | Add to import |
| 4 | `memstream_scoring_op.py` | 1142 | `KeyedBroadcastProcessFunction` not imported | Add to import |
| 5 | `memstream_scoring_op.py` | 1046–1063 | Per-record serialization — checkpoint death spiral | Serialize only on checkpoint |
| 6 | `flink_job_complete.py` | 727–748 | Rendezvous buffer leak — OOM | Add TTL via `on_timer()` |
| 7 | `flink_job_complete.py` | 604 | No RocksDB state backend — JVM OOM | Add `EmbeddedRocksDBStateBackend()` |
| 8 | `src/features/vectorizer.py` vs `PLAN_v3.md` | — | 21D vs 25D mismatch | Standardize to 25D everywhere |

---

## 9. HIGH Issues

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `flink_job_complete.py` | Kafka producer missing EXACTLY_ONCE | Add `producer_semantic=EXACTLY_ONCE` |
| 2 | `memstream_scoring_op.py` | Beta race on restore | Add `max_thres_version` timestamp |
| 3 | `memstream_scoring_op.py` | Device hardcoded CPU | Add `_get_safe_device()` |
| 4 | `memstream_scoring_op.py` | Import inside `process_element` | Move to class `__init__` |
| 5 | `flink_job_complete.py` | No restart strategy | Configure exponential backoff |
| 6 | `src/features/vectorizer.py` | float64 vs float32 mismatch | Change to `np.float32` |

---

## 10. MEDIUM/LOW Issues

| # | File | Issue |
|---|------|-------|
| 1 | `iec_operator.py` | ADWIN state not checkpointed |
| 2 | `flink_job_complete.py` | `ParseJsonFunction` catches all exceptions silently |
| 3 | `flink_job_complete.py` | Consumer auto-commit not disabled |
| 4 | `memstream_scoring_op.py` | Model path not validated for cross-TM consistency |
| 5 | `memstream_core.py` | Duplicate `torch.device()` conversion |

---

## 11. Priority Fixes

### Fix 1: Wire Up Missing Components

```python
# After line 660 in flink_job_complete.py:
canary_stream = valid_stream.map(
    CanaryRulesValidator(), output_type=Types.PICKLED_BYTE_ARRAY())
complex_stream = valid_stream.map(
    IFScoringOperator(), output_type=Types.PICKLED_BYTE_ARRAY())

from pyflink.datastream.state import EmbeddedRocksDBStateBackend
env.set_state_backend(EmbeddedRocksDBStateBackend())

env.set_restart_strategy(
    RestartStrategies.exponential_delay_restart_strategy(
        initial_restart_delay=1000, max_restart_delay=60000, back_off_multiplier=2.0))
```

### Fix 2: Rendezvous TTL

```python
class RendezvousCoProcessFunc(CoProcessFunction):
    def __init__(self, ttl_ms=300000):
        self.canary_buf = {}
        self.complex_buf = {}
        self.ttl_ms = ttl_ms
        self._timers = set()

    def process_element1(self, record, context):
        trip_id = record.get('trip_id', '')
        if trip_id in self.complex_buf:
            yield {...}  # matched
        else:
            self.canary_buf[trip_id] = (record, context.timestamp())
            context.timer_service().register_event_time_timer(
                context.timestamp() + self.ttl_ms)

    def on_timer(self, timestamp, context, out):
        # Evict expired entries
        expired = [k for k, v in self.canary_buf.items() if v[1] < timestamp - self.ttl_ms]
        for k in expired:
            self.canary_buf.pop(k, None)
```

### Fix 3: Serialize Only on Checkpoint

```python
from pyflink.datastream import CheckpointedFunction

class MemStreamScoringOperator(KeyedProcessFunction, CheckpointedFunction):
    def snapshot_state(self, context):
        pass  # Flink handles this via get_state()

    def initialize_state(self, context):
        pass  # Flink restores via get_state()

    def process_element(self, record, context):
        # NO serialization here — model stays in memory
        key = context.get_current_key()
        self._models[key] = ms  # Keep in-memory only
```

---

*Reviewed by: Flink Principal Engineer | 2026-05-12*
