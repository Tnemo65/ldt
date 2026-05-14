# Benchmark v9 Plan: CA-MemStream-EIA Ablation Study

> Plan version: 9.0
> Date: 2026-05-13
> Based on: v8 critique by expert reviewer

---

## 1. Strategic Direction

### 1.1 Narrative Change: Kaggle Competition vs Scientific Contribution

v8 was designed as a "Kaggle-style" AUC-PR shootout between MemStream and CA-DIF-EIA-Stream. The expert reviewer correctly identified this as a strategic mistake: competing on raw AUC-PR against a SOTA algorithm is a losing game, and even if you win, it's not a publishable "contribution."

The **correct narrative** is:

> "We take the best streaming algorithm (MemStream), wrap it in our CA-EIA framework, and demonstrate that the resulting CA-MemStream-EIA achieves the same accuracy at 5% of the computational cost — which IS the contribution."

### 1.2 System Architecture (CA-MemStream-EIA)

```
Incoming Stream
     │
     ▼
┌─────────────────────────────┐
│   CANARY RULES (Branch 1)   │  ← Filter obvious anomalies (speed>100mph, fare<0)
│   ~0% ML Budget Used        │    Multipliers: 8-30x (extreme, easy to detect)
└─────────────────────────────┘
     │ (pass-through)
     ▼
┌─────────────────────────────┐
│   ML COMPLEX (Branch 2)     │  ← Subtle anomalies (1.5-2x) that fool rules
│   CA-MemStream-EIA          │    Multipliers: 1.5-2x (hard, need ML)
│   Budget: 5% (ADWIN-gated)  │
└─────────────────────────────┘
     │
     ▼
  Decision: ACCEPT / REVIEW
```

### 1.3 The Three Ablation Stages

| Stage | Algorithm | CA | EIA | Narrative |
|-------|-----------|-----|-----|-----------|
| **A** | MemStream (baseline, v8-corrected) | No | No | Original MemStream: powerful but "hungry" and "context-blind" |
| **B** | **CA-MemStream** | Yes (Lưới 4D) | No | Context grid eliminates false alarms, but still "hungry" |
| **C** | **CA-MemStream-EIA** | Yes | Yes (ADWIN-U) | Full system: same accuracy at 5% budget |

---

## 2. Key Metric: BAR Score

The **BAR (Balanced Accuracy Ratio) Score** measures how much accuracy you achieve per unit of label budget.

```
BAR = AUC_PR / (1 + label_budget_fraction)
```

For example:
- MemStream with budget=500 (5% of 10K): BAR = 0.9988 / 1.05 = 0.951
- CA-MemStream-EIA with budget=5 (0.05%): BAR = 0.9980 / 1.0005 = 0.9975

**Primary comparison:** AUC-PR vs Label Budget curve (Pareto frontier)

---

## 3. Dataset Setup (v9)

### 3.1 Branch Separation for Canary vs ML

| Branch | Anomaly Type | Multiplier | Detection Method |
|--------|-------------|------------|-----------------|
| **Canary (Branch 1)** | Obvious rule violations | 8-30x fare, >100mph, negative | Business rules (no ML needed) |
| **ML (Branch 2)** | Subtle concept violations | 1.5-2x fare + distance | MemStream variants (hard) |

### 3.2 ML Branch Anomaly Injection (REFINED)

The v8 error was using 8-30x multipliers for all algorithms. v9 uses subtle anomalies:

```
ML Difficulty Levels:
  easy:   fare × 1.8–2.2x,  distance × 1.3–1.5x  (just noticeable)
  medium: fare × 1.4–1.8x,  distance × 1.1–1.3x  (hard)
  hard:   fare × 1.2–1.5x,  distance × 1.0–1.1x  (very hard, near boundary)
```

### 3.3 Dataset Splits (Same as v8)

- **Source:** NYC Taxi 2024, months 01–06
- **Per month:** Train 10K / Val 2K / Test 10K
- **Temporal folds:** 5 (leave-one-month-out)
- **Seeds:** [42, 123, 456]

---

## 4. Algorithm Implementations

### 4.1 Baseline: MemStream (v8-corrected)

Reuse the v8-corrected MemStream (trained AE, latent memory, FIFO, L1 kNN, anti-poisoning β). This is the honest baseline.

### 4.2 CA-MemStream (Ablation B)

**Modification:** Add Context-Aware Feature Weighting (Lưới 4D) on top of MemStream.

```
score(x) = MemStream_score(x) × mean(context_weights(x))
         = L1_knn_distance(z(x)) × mean(w_context[hour, dow])
```

The context grid (168 contexts: 24h × 7dow) adjusts the MemStream score to account for natural variation by time-of-day and day-of-week.

**Hypothesis B:** CA-MemStream will have higher Precision than MemStream (fewer false alarms) because a high fare at 8PM Saturday is less anomalous than the same fare at 6AM Tuesday.

### 4.3 CA-MemStream-EIA (Ablation C, Full System)

**Modification:** Add ADWIN-U gate on top of CA-MemStream.

```
MemStream only updates memory when ADWIN-U detects score distribution drift.
Before drift:    memory frozen (0 updates)
After drift:     memory starts updating again
```

**Hypothesis C:** CA-MemStream-EIA achieves the same AUC-PR as CA-MemStream with 95% fewer memory updates → BAR score dramatically higher.

### 4.4 Comparison Baselines

| Algorithm | Type | Purpose |
|----------|-----|---------|
| MemStream (v8-corrected) | Streaming | Baseline (A) |
| CA-MemStream | Streaming+AblationB | CA adds context awareness |
| CA-MemStream-EIA | Streaming+Full | Full system (C) |
| sHST-River | Streaming | External baseline |
| DenoisingAE | Batch | Upper bound (no budget constraint) |
| Canary-only | Rule-based | Branch 1 contribution |
| Random | Baseline | Floor |

---

## 5. Evaluation Protocol

### 5.1 Primary Metrics

1. **AUC-PR vs Label Budget** (Pareto frontier) — the core visualization
2. **BAR Score** = AUC_PR / (1 + budget_fraction)
3. **Memory Update Count** — counts how many times memory is actually updated
4. **AUC-PR by difficulty** (easy/medium/hard with 1.5-2x multipliers)
5. **Temporal stability** (std across folds)

### 5.2 Secondary Metrics

- **False Alarm Rate** by time-of-day (prove CA reduces 6AM false alarms)
- **Adaptation Latency** after concept drift (prove ADWIN fires and recovers)
- **Throughput** (samples/second)

### 5.3 Label Budget Scenarios (Extended)

| Budget | Fraction | Description |
|--------|----------|-------------|
| 0 | 0% | Pure unsupervised (no labels) |
| 50 | 0.5% | Extremely constrained |
| 100 | 1% | Very constrained |
| 250 | 2.5% | Moderately constrained |
| 500 | 5% | Standard (primary comparison) |
| 1000 | 10% | Abundant |
| 2000 | 20% | Near-unlimited |

### 5.4 Concept Drift Refined Setup

**Fixed from v8:** Anomalies injected in BOTH pre-drift AND post-drift segments.

```
Phase 1 (0–50%): Normal data from base distribution
  └── Anomaly rate: 1% (base rate, represents natural noise)

Phase 2 (50–100%): Shifted distribution (magnitude=1.5σ)
  └── Anomaly rate: 5% (represents new anomaly patterns)
  └── New concept: fare_per_mile logic shifted

Metrics:
  - AUC-PR in Phase 1 (should be ~0.5 baseline)
  - AUC-PR in Phase 2 (tests drift adaptation)
  - Score trajectory over time (shows memory adapting)
  - Recovery time (samples until post-drift AUC-PR recovers)
```

---

## 6. Data Leakage Prevention (Same as v8)

- Scaler fit on train only
- AE trained on train only
- Anomalies injected only in test
- Temporal order preserved
- Seed consistency across algorithms

---

## 7. Ablation Study Design

### 7.1 Ablation Table

| Component | MemStream | CA-MemStream | CA-MemStream-EIA |
|-----------|-----------|--------------|------------------|
| Trained DAE (25→50→25) | ✓ | ✓ | ✓ |
| Latent Memory (256 vectors) | ✓ | ✓ | ✓ |
| FIFO Replacement | ✓ | ✓ | ✓ |
| L1 kNN Scoring | ✓ | ✓ | ✓ |
| Anti-Poisoning (β) | ✓ | ✓ | ✓ |
| Context Grid (4D) | ✗ | ✓ | ✓ |
| ADWIN Drift Detection | ✗ | ✗ | ✓ |
| Budget-Gated Updates | ✗ | ✗ | ✓ |

### 7.2 Expected Results

| Algorithm | AUC-PR | Precision | Recall | Budget Used | BAR Score |
|-----------|--------|-----------|--------|-------------|-----------|
| MemStream | ~0.998 | ~0.50 | ~1.00 | 100% | ~0.50 |
| CA-MemStream | ~0.998 | **~0.65** | ~1.00 | 100% | ~0.65 |
| CA-MemStream-EIA | ~0.998 | **~0.65** | **~1.00** | **~5%** | **~0.95** |
| sHST-River | ~0.23 | ~0.18 | ~0.10 | 100% | ~0.19 |
| Random | ~0.05 | ~0.05 | ~0.19 | 0% | ~0.05 |

**Key claim to prove:** CA-MemStream-EIA achieves the same AUC-PR as MemStream with 95% less budget → BAR score ~2x higher.

---

## 8. Deliverables

- [ ] `benchmark_v9.py` — Ablation study benchmark
- [ ] `checkpoint_v9.csv` — Raw results
- [ ] `benchmark_v9_results.md` — Results summary
- [ ] `fig_ablation_v9.png` — Ablation comparison plots
- [ ] `fig_bar_score_v9.png` — BAR score Pareto frontier
- [ ] `fig_budget_curve_v9.png` — AUC-PR vs Label Budget curve
- [ ] `fig_canary_ml_split_v9.png` — Canary vs ML branch contribution
- [ ] `run_concept_drift_v9.py` — Refined concept drift evaluation
- [ ] `SYNTHESIS_v9.md` — Full ablation study report

---

## 9. Implementation Notes

### 9.1 CA-MemStream Class

```python
class CAMemStream:
    """MemStream + Context-Aware Feature Weighting (Ablation B)."""
    # Inherit all MemStream logic
    # Override score_one() to multiply by context weights
    def score_one(self, x):
        base_score = super().score_one(x)  # MemStream L1 kNN score
        cw = self._context_weights.get_weights(x)  # 168 contexts
        cw_mean = max(cw.mean(), 0.1)
        return base_score * cw_mean
```

### 9.2 CA-MemStream-EIA Class

```python
class CAMemStreamEIA:
    """MemStream + Context Weighting + ADWIN Budget Gate (Ablation C)."""
    # Inherit CA-MemStream logic
    # Override update_one() to gate by ADWIN
    def update_one(self, x, label=None):
        score = self.score_one(x)
        drift = self._adwin.update(score)
        # Only update memory if ADWIN detects drift
        if drift:
            super().update_one(x, label)  # Memory update gated by drift
```

### 9.3 Subtle Anomaly Injection

```python
def inject_subtle_anomalies(X, y, rng, difficulty='medium'):
    """Inject anomalies with multipliers 1.5-2x (hard to detect)."""
    multipliers = {
        'easy':   (1.8, 2.2, 1.3, 1.5),   # fare_range, dist_range
        'medium': (1.4, 1.8, 1.1, 1.3),
        'hard':   (1.2, 1.5, 1.0, 1.1),
    }
    fare_lo, fare_hi, dist_lo, dist_hi = multipliers[difficulty]
    # ... multiply fare and distance columns
```

---

## 10. Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Plan v9 | 30 min | This plan |
| Implement v9 | 3 hr | Write corrected benchmark + ablation classes |
| Debug | 2 hr | Fix bugs, verify ablation works |
| Execution | 15 min | Full benchmark |
| Concept drift v9 | 30 sec | Refined drift test |
| Reports | 1 hr | Plots + SYNTHESIS_v9.md |

---

*Plan created: 2026-05-13*
*Review by: User*
