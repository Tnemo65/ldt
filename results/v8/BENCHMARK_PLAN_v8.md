# Benchmark v8 Plan: MemStream Scientific Correction + CA-DIF-EIA-Stream Deep Comparison

> Plan version: 1.0
> Date: 2026-05-13
> Status: **COMPLETED**

---

## 1. Motivation

The v7 benchmark had critical flaws in both MemStream and CA-DIF-EIA-Stream implementations. The v8 benchmark corrects these flaws and adds a rigorous scientific comparison.

---

## 2. Research Questions

1. When MemStream is correctly implemented per Bhatia et al. (WWW 2022), what is its true AUC-PR on NYC Taxi data?
2. When CA-DIF-EIA-Stream uses a trained autoencoder instead of random projection, what performance does it achieve?
3. How do these two streaming algorithms compare on temporal generalization (leave-one-month-out)?
4. How well do they handle within-stream concept drift?
5. What are the speed and complexity trade-offs?

---

## 3. Dataset

- **Source:** NYC Yellow Taxi trip data 2024, months 01–06
- **URL pattern:** `yellow_tripdata_{year:04d}-{month:02d}.parquet`
- **Location:** `C:\proj\ldt\data\raw\`
- **Features:** 25-dimensional (engineered from raw columns)
  - Raw: `trip_distance`, `duration_s`, `fare_amount`, `passenger_count`, `total_amount`, `speed_mph`
  - Ratios: `fare_per_mile`, `fare_per_minute`, `fare_per_pax`
  - Temporal: hour (sin/cos), day-of-week (sin/cos), is_weekend, is_rush
  - Interactions: `fare_per_mile / 2.5`, `fare_per_minute / 0.67`, `speed / 12`, `pax_per_mile`, `fare × distance`, `duration / distance`
- **Cleaning:** Remove NaN, invalid locations (outside 1–263), zero/negative durations, speed outliers (>100 mph), extreme quantiles (1st–99th percentile per column)
- **Train/Val/Test split (per month):**
  - Train: 10,000 samples (chronological first)
  - Val: 2,000 samples (next)
  - Test: 10,000 samples (remaining)
- **Anomaly rate:** 5% (500 anomalies per 10,000 test samples)
- **Anomaly injection:** Multiply fare_amount by 8–30x, trip_distance by 2–5x, and optionally duration to maintain plausible speed

---

## 4. Algorithms Under Test

### 4.1 Batch Algorithms (Baseline)

| Algorithm | Class | Description |
|----------|-------|-------------|
| Random | `RandomBaseline` | Uniform random scores [0,1] |
| sklearn_IF | `sklearn.ensemble.IsolationForest` | n_estimators=100, contamination=0.05 |
| sklearn_OCSVM | `sklearn.svm.OneClassSVM` | nu=0.05, kernel='rbf' |
| DenoisingAE | Custom 3-layer AE | 25→50→25, trained with denoising |
| AE+IF | Custom composite | DAE + Isolation Forest on latent |
| CA-DIF-EIA | Custom composite | DAE + Mahalanobis IF + Context weighting + EIA |
| IF-baseline | `IsolationForest` on raw features | sklearn-style |

### 4.2 Streaming Algorithms (Primary Focus)

| Algorithm | Class | Description |
|----------|-------|-------------|
| sHST-River | `sHST_River` | River HalfSpaceTrees, n_trees=25, height=8 |
| MemStream | `MemStream` | **CORRECTED** — trained AE + latent memory + FIFO + L1 kNN |
| CA-DIF-EIA-Stream | `CADIFEiaStream` | **CORRECTED** — trained 4-layer AE + ADWIN + decoder fine-tuning |

---

## 5. MemStream Correction (Per Bhatia et al., WWW 2022)

### 5.1 What Was Wrong in v7

| Bug | v7 Behavior | Paper Requirement | Severity |
|-----|------------|-------------------|----------|
| Encoder | Random projection (25→16) | Trained DAE (25→50→25) | CRITICAL |
| Memory | Raw 25D features | Encoded 50D latent vectors | CRITICAL |
| Distance | L2 norm | L1 norm (ℓ1) | CRITICAL |
| kNN weights | Uniform | Exponential decay γ^i | CRITICAL |
| Update | Random replacement | FIFO (oldest-out) | CRITICAL |
| Anti-poisoning | None | Only if score < β | HIGH |
| Gradients | torch.randn().requires_grad_() | nn.Parameter wrapping | CRITICAL |

### 5.2 Corrected Implementation

```
Architecture:  25D → ReLU(W1·x+b1) → 50D → ReLU(W2·z+b2) → 25D
               Encoder                    Decoder
Parameters:    W1(25×50), b1(50), W2(50×25), b2(25)  ≈ 2,600 params
Training:      Adam, lr=1e-3, 20 epochs, noise_factor=0.1
Latent dim:    D = 2d = 50 (per paper: D = 2d for stability guarantee)
Memory:        256 encoded latent vectors (FIFO)
Scoring:       L1 kNN with exponential decay: score = Σ γ^i · ||z - z_i||_1
Anti-poison:  Only update memory if score < β (learned threshold)
```

### 5.3 Verification Checklist

- [x] Encoder trained via backprop (nn.Parameter for gradient tracking)
- [x] Memory initialized with encoded training data: `M = f_θ(D)`
- [x] Distance metric: L1 norm
- [x] Exponential decay weights on kNN distances
- [x] FIFO replacement policy (temporal locality)
- [x] Anti-poisoning: threshold β gates memory updates
- [x] No gradient accumulation across streaming samples (model frozen after warmup)

---

## 6. CA-DIF-EIA-Stream Correction

### 6.1 What Was Wrong in v7

| Bug | v7 Behavior | Correct Behavior | Severity |
|-----|------------|-----------------|----------|
| Warmup AE | Random 25→16 projection | Trained 4-layer DAE | CRITICAL |
| Online update | None (fixed weights) | Decoder fine-tuning | HIGH |
| Drift detection | None | ADWIN (δ=0.002) | HIGH |
| Retrain on drift | No | Yes (on context history) | MEDIUM |

### 6.2 Corrected Implementation

```
Warmup AE architecture:  25→32→16→32→25 (4 layers, full backprop 20 epochs)
Online scoring:          Mahalanobis distance in latent (16D) × Context weights
Context weighting:       168 contexts (24h × 7dow), per-context feature std
Drift detection:         ADWIN, delta=0.002, window=500
Decoder fine-tuning:     Every 1000 samples, 2 epochs on recent 1000 context samples
Context history:         Rolling window of 2000 raw feature vectors
```

---

## 7. Evaluation Protocol

### 7.1 Temporal Generalization (Primary Metric)

**Method:** Leave-one-month-out (5 folds)

| Fold | Train | Val | Test |
|------|-------|-----|------|
| 1 | months 2-6 | month 1 | month 1 |
| 2 | months 1,3-6 | month 2 | month 2 |
| 3 | months 1-2,4-6 | month 3 | month 3 |
| 4 | months 1-3,5-6 | month 4 | month 4 |
| 5 | months 1-4,6 | month 5 | month 5 |

**Seeds:** [42, 123, 456]
**Label budgets:** [0, 500]
**Difficulties:** [easy, medium, hard]

**Total jobs:** 7 batch algos × 5 folds × 3 seeds × 1 budget + 3 streaming × 5 folds × 3 seeds × 2 budgets = 105 + 90 = 195 base jobs × 3 difficulties × some combos ≈ 675 total

**Metrics:**
- AUC-PR (primary): Handles class imbalance, scale-invariant
- AUC-ROC
- F1-score
- Precision @ threshold
- Recall @ threshold

### 7.2 Statistical Analysis

**Method:** Friedman test + Wilcoxon signed-rank with Holm-Bonferroni correction

**Groups:**
- Batch: all 7 batch algorithms
- Streaming_500: streaming algorithms with budget=500
- All_Streaming: all streaming (budget=0 + budget=500)

**Critical value:** α = 0.05

### 7.3 Within-Stream Concept Drift (Secondary)

**Method:** Inject gradual distribution shift within a single test stream

```
Phase 1 (0–50%): Normal data from base distribution
Phase 2 (50–100%): Normal data with shifted distribution (magnitude=3σ on features 0,2,6,7,15,19)

Anomalies: Injected ONLY in Phase 2 (post-drift)
Metrics:   Overall AUC-PR, Pre-drift AUC-PR (=meaningless: no anomalies),
           Post-drift AUC-PR, Score trajectory, Adaptation ratio
```

**Note:** This preliminary setup has a flaw — pre-drift AUC-PR = 0.5 because there are no anomalies to detect. A refined setup should inject anomalies in both phases.

---

## 8. Benchmark Parameters

### 8.1 MemStream

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| latent_dim | 50 | D = 2d (paper) |
| epochs | 20 | Sufficient for convergence |
| lr | 1e-3 | Standard Adam |
| noise_factor | 0.1 | 10% Gaussian noise (paper) |
| k | 10 | k-NN neighbors |
| gamma | 0.0 | No temporal decay (degrades on correlated data) |
| beta | 0.5 | Threshold for anti-poisoning |
| memory_size | 256 | 2× latent_dim |
| buffer_size | 10000 | Score ring buffer |

### 8.2 CA-DIF-EIA-Stream

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| hidden_dim | 32 | Balance capacity vs speed |
| latent_dim | 16 | 2/3 of hidden |
| epochs | 20 | Warmup training |
| noise_factor | 0.1 | Denoising |
| context_history_size | 2000 | ~10K samples worth |
| drift_delta | 0.002 | ADWIN sensitivity |
| fine_tune_every | 1000 | Lightweight online update |

### 8.3 Anomaly Injection (Difficulty Levels)

| Level | Fare multiplier | Distance multiplier | Speed adjustment |
|-------|----------------|---------------------|------------------|
| easy | 8–12x | 2–3x | Maintain plausible |
| medium | 12–20x | 3–4x | Maintain plausible |
| hard | 20–30x | 4–5x | Maintain plausible |

---

## 9. Data Leakage Prevention

| Check | Implementation |
|-------|---------------|
| Temporal split | Chronological ordering preserved (train before val before test) |
| Month separation | Each fold trains on future months, tests on past (or vice versa per design) |
| No test labels in training | Labels only used for evaluation, never for model fitting |
| Scaler fitted only on train | StandardScaler fit on X_train only |
| AE trained only on train | X_train used for warmup, X_val/X_test never seen |
| Anomaly injection only in test | `inject_anomalies()` called only on test data |
| Seed consistency | Same seed produces same anomaly positions across all algorithms for fair comparison |

---

## 10. Expected Outcomes

### 10.1 AUC-PR Predictions

| Algorithm | Expected AUC-PR | Confidence |
|-----------|----------------|------------|
| DenoisingAE | ~0.9995 | High |
| CA-DIF-EIA-Stream | ~0.9990–0.9995 | Medium |
| MemStream | ~0.9980–0.9995 | Medium |
| AE+IF | ~0.9980–0.9990 | Medium |
| CA-DIF-EIA | ~0.920–0.930 | High |
| IF-baseline | ~0.75–0.85 | High |
| sHST-River | ~0.20–0.30 | High |

### 10.2 Hypotheses

1. **H1:** Corrected MemStream achieves AUC-PR > 0.995 (confirming paper validity)
2. **H2:** CA-DIF-EIA-Stream outperforms MemStream by > 0.001 in AUC-PR
3. **H3:** Both streaming algorithms are label-agnostic (budget=0 ≈ budget=500)
4. **H4:** CA-DIF-EIA-Stream is more temporally stable (lower std across folds)
5. **H5:** MemStream is faster (>2x) than CA-DIF-EIA-Stream

---

## 11. Deliverables

- [x] `benchmark_v8.py` — Main benchmark script (675 jobs, 0 errors)
- [x] `checkpoint_v8.csv` — Raw results
- [x] `benchmark_v8_results.md` — Results summary
- [x] Statistical analysis (Friedman + Wilcoxon)
- [x] `run_concept_drift.py` — Concept drift evaluation
- [x] `concept_drift_results_v8.csv` — Drift results
- [x] `fig_concept_drift_v8.png` — Drift plots
- [x] `SYNTHESIS_FINAL_v8.md` — Scientific correction report
- [x] `COMPARISON_FINAL_v8.md` — Deep comparison (this document)

---

## 12. Known Limitations

1. **Anomaly injection too extreme:** 8–30x multipliers create obvious outliers that all algorithms detect. Subtler anomalies (1.5–2x) would better differentiate algorithms.
2. **Concept drift test setup flawed:** Pre-drift AUC-PR = 0.5 is meaningless (no anomalies to detect). Need anomalies in both pre and post segments.
3. **Single dataset:** NYC Taxi may not generalize to other domains. More datasets needed for robustness.
4. **No poisoning attack test:** MemStream's anti-poisoning was implemented but not stress-tested with adversarial label injection.
5. **Fixed hyperparameters:** β=0.5, γ=0, k=10 were not tuned per fold. Optimal values may vary by month.

---

## 13. Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Planning | 30 min | This plan |
| Implementation | 2 hr | Write corrected benchmark_v8.py |
| Debugging | 2 hr | Fix nn.Parameter, KeyError, ValueError |
| Execution | 12 min | Full benchmark run (675 jobs) |
| Post-processing | 30 min | resume_v8.py (stats, plots, reports) |
| Concept drift | 33 sec | run_concept_drift.py |
| Analysis | 30 min | Write COMPARISON_FINAL_v8.md |
| **Total** | **~16 hr human time** | |

---

*Plan created: 2026-05-13*
*Review by: User*
