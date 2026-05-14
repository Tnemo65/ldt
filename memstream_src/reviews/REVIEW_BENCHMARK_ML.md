# ML Expert Review: CA-MemStream Benchmark Methodology

**Review Date:** May 12, 2026  
**Reviewer:** ML Expert (PhD-level) — Anomaly Detection, Streaming Algorithms, Evaluation Methodology  
**Document Status:** Critical Assessment for Publication Readiness

---

## Executive Summary

This review identifies **10 critical methodological issues** in the CA-MemStream benchmark framework. The most severe problems are:

1. **MemStream Fold 1 produces near-random AUC-ROC (~0.50) on "easy" difficulty** — a catastrophic failure that indicates either warmup failure, improper initialization, or a data pipeline bug
2. **Benchmark uses batch evaluation for a streaming algorithm** — fundamentally invalidates claims about streaming capability
3. **No statistical hypothesis testing** — reported differences lack p-values or confidence intervals
4. **Only 2 algorithms compared** — insufficient for a scientific benchmark paper

**Overall Verdict:** The benchmark is **NOT publication-ready**. Critical issues must be resolved before submission.

---

## A. Dataset Quality Assessment

### A.1. Preprocessed 25D Feature Set

**Status: ACCEPTABLE with reservations**

The 25D feature set is derived from NYC taxi data and appears reasonable for anomaly detection. However:

| Concern | Severity | Issue |
|---------|----------|-------|
| No feature importance analysis | MEDIUM | Cannot determine which features drive detection |
| No dimensionality reduction validation | LOW | 25D chosen arbitrarily, no ablation study |
| Temporal features not explicitly modeled | HIGH | Taxi data is inherently temporal; features should encode time dependencies |

**Assessment:** The feature set is defensible but needs documentation justifying the 25-dimension choice and showing that these features are discriminative for anomaly detection.

### A.2. Difficulty Levels (Easy/Medium/Hard)

**Status: INADEQUATE — No operational definition provided**

The benchmark uses three difficulty levels, but there is **no documented methodology** for how these were constructed:

- How was "difficulty" operationalized?
- Is it based on anomaly rate? Anomaly subtlety? Feature overlap?
- Are the difficulty levels statistically distinguishable?

**Evidence of problem:** The results show widely varying AUC-PR values within each difficulty level (e.g., easy MemStream ranges from 0.0009 to 0.449), suggesting the difficulty levels may not be homogeneous.

**Recommendation:** Document the exact criteria for each difficulty level. If based on k-NN distance or density metrics, specify parameters. If based on injection parameters, report the exact settings.

### A.3. Fold Structure (11 Folds)

**Status: UNUSUAL — Non-standard choice without justification**

11 folds is mathematically unusual (standard choices are 5, 10, or k-fold where k divides the dataset). The choice raises questions:

| Question | Impact |
|----------|--------|
| Why 11 specifically? | No documented rationale |
| Is 11 appropriate for the dataset size? | Could lead to small test sets |
| Is this stratified by anomaly rate? | Not verified from code |

**Code Evidence:**
```python
# From benchmark_ca_memstream.py - fold loading
fold_data = [(e['fold'], e) for e in metadata if e['difficulty'] == diff]
```

The fold structure is loaded from metadata but the metadata generation process is not reviewed. This is a **BLACK BOX** that needs documentation.

### A.4. Anomaly Rates per Difficulty Level

**Status: NOT REPORTED — Critical missing information**

The benchmark does not report anomaly rates per fold or difficulty level. This is essential because:

- **AUC-PR is highly sensitive to anomaly rate**: With 5% contamination, baseline precision = 0.05
- **AUC-ROC is rate-independent**: More appropriate for cross-dataset comparison
- **The extremely low AUC-PR values** (0.0009–0.45) suggest very sparse anomalies

**Critical Finding:** AUC-PR values of 0.0009 (MemStream fold 1, easy) indicate near-zero precision, meaning the algorithm is essentially predicting all negatives. This needs explanation.

---

## B. Algorithm Selection

### B.1. Baseline Algorithms

**Status: INSUFFICIENT — Only 2 algorithms compared**

The quick benchmark (`benchmark_ca_memstream.py`) compares only:
1. `sklearn_IF` — sklearn IsolationForest
2. `MemStream_25D` — CA-MemStream

This is **grossly insufficient** for a scientific benchmark. A proper comparison requires:

| Missing Algorithm | Justification | Priority |
|------------------|---------------|----------|
| One-Class SVM (RBF) | Standard anomaly detection baseline | HIGH |
| LOF (Local Outlier Factor) | Density-based detection | HIGH |
| HBOS | Histogram-based (fast, competitive) | MEDIUM |
| DeepSVDD | Deep learning baseline | HIGH |
| LSTM-AE | Temporal anomaly detection | MEDIUM |
| OCSVM with RBF kernel | Classic method | MEDIUM |

**The full benchmark** (`run_sequential_v4.py`) includes more algorithms (LOF-50, CADIFEia, METERSCD, HBOS, LSTM-AE, sHST-River, IForestASD, sHST-Mem-Ens), but the quick benchmark used for results only ran IF vs MemStream.

### B.2. CA-MemStream Positioning

**Status: UNCLEAR — Claims of streaming but evaluated as batch**

The benchmark claims CA-MemStream is a **streaming algorithm**, but evaluation is done in **batch mode**:

```python
# From benchmark_ca_memstream.py - batch scoring
scores_ms = ms.score_batch(X_test)  # NOT score_one() per record
```

This is a **fundamental mismatch**. If claiming streaming capability, the benchmark must:
1. Score records one-at-a-time (online evaluation)
2. Update memory during scoring (as specified in the streaming paradigm)
3. Measure latency per record
4. Show performance degradation under concept drift

**Verdict:** The benchmark evaluates a **frozen MemStream model** on static test sets. This is batch evaluation, not streaming evaluation.

---

## C. Metric Selection (CRITICAL)

### C.1. AUC-PR vs AUC-ROC

**Status: CONTROVERSIAL — May be the wrong primary metric**

The benchmark uses **AUC-PR as the primary metric** with AUC-ROC as secondary. This requires careful justification.

| Metric | Behavior with Imbalanced Data | Appropriateness |
|--------|------------------------------|-----------------|
| AUC-ROC | Assumes balanced classes; can be misleading with <5% positives | **RECOMMENDED** for anomaly detection |
| AUC-PR | Focuses on positive class; appropriate for imbalanced data | Use when precision matters at fixed recall |

**The Problem:** With anomaly rates of ~5%, AUC-PR baseline is ~0.05. An AUC-PR of 0.39 (MemStream easy, fold 3) sounds good but requires calibration.

**Recommendation:** 
- Use **AUC-ROC as primary metric** (more interpretable, commonly reported in anomaly detection literature)
- Report **AUC-PR as supplementary** with clear baseline reference
- Add **Precision@K and Recall@K** for operational interpretation

### C.2. Threshold Calibration Approach

**Status: FLAWED — Uses oracle (F1-optimal) threshold**

```python
# From run_sequential_v4.py
def find_threshold_f1(scores, y_true):
    thresholds = np.percentile(scores, np.arange(80, 100, 0.5))
    best_f1, best_t = 0.0, float(np.percentile(scores, 97))
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
```

**Critical Issue:** This finds the **oracle-optimal threshold** using ground truth labels. In production:
- You don't have labels to optimize thresholds
- You must use contamination-based thresholds (e.g., contamination=0.05 → threshold = 95th percentile)

**Impact:** F1, Precision, Recall values are **unrealistically optimistic** — they represent best-case performance, not deployment performance.

**Fix:** Use contamination-based threshold (known contamination rate) or percentile-based threshold without using labels.

### C.3. Top-K Metrics

**Status: NOT USED — Should be added**

For operational deployment, practitioners care about:
- **Precision@1%** — Precision when flagging top 1% as anomalies
- **Recall@5%** — Recall when allowing 5% false positive rate
- **% of true anomalies caught in top-K alerts**

These are standard in operational anomaly detection and should be added.

---

## D. Statistical Rigor

### D.1. Number of Folds and Seeds

**Status: INSUFFICIENT — No statistical power analysis**

| Current Setup | Issue |
|---------------|-------|
| 11 folds | Non-standard; no power analysis |
| 5 seeds (implied by `SEEDS = [42, 123, 456, 789, 1024]`) | Good for variance estimation |
| No p-value reported | Cannot determine significance |
| No confidence intervals | Cannot assess uncertainty |

**Critical Missing:** **Wilcoxon signed-rank test** for paired comparisons. The benchmark does not perform any statistical hypothesis testing.

### D.2. Wilcoxon Test Applicability

**Status: NOT APPLIED — Should be used but not implemented**

Wilcoxon signed-rank test is appropriate here because:
- Paired comparisons (same folds, different algorithms)
- Non-normal distributions expected (AUC values bounded [0,1])
- Non-parametric (no distributional assumptions)

**Required additions:**
```python
from scipy.stats import wilcoxon

# Example: Compare IF vs MemStream per difficulty
for diff in ['easy', 'medium', 'hard']:
    if_auc = results[results.algorithm=='sklearn_IF'][results.difficulty==diff]['auc_roc'].values
    ms_auc = results[results.algorithm=='MemStream_25D'][results.difficulty==diff]['auc_roc'].values
    stat, p = wilcoxon(if_auc, ms_auc)
```

### D.3. Nemenyi Critical Difference Test

**Status: NOT APPLIED — Should be used for multiple comparisons**

When comparing **more than 2 algorithms** (in the full benchmark with 9+ algorithms), pairwise Wilcoxon tests inflate Type I error. The **Nemenyi post-hoc test** is the appropriate correction.

The full benchmark (`run_sequential_v4.py`) runs 11 algorithms. Without multiple-comparison correction, any "significant" result is likely a false positive.

### D.4. Effect Size Reporting

**Status: NOT REPORTED — Critical for interpretation**

The results show differences like:
- Easy: MemStream AUC-PR = 0.39 vs IF AUC-PR = 0.09 (delta = +0.30)
- Hard: MemStream AUC-PR = 0.01 vs IF AUC-PR = 0.007 (delta = +0.003)

**Both deltas are reported equally**, but they have vastly different practical significance. 

**Required:** Effect size measures:
- **Cliff's Delta** (non-parametric effect size)
- **Vargha-Delaney A** (probability of superiority)
- **Cohen's d** (if parametric assumptions hold)

A delta of 0.003 in AUC-PR is **negligible** even if statistically significant with n=11.

---

## E. MemStream-Specific Issues

### E.1. Warmup Epochs (10 epochs)

**Status: LIKELY INSUFFICIENT — No convergence analysis**

```python
# From benchmark_ca_memstream.py
cfg.warmup_epochs = 10
ms.warmup(X_warmup, epochs=10, batch_size=256, verbose=False)
```

**Evidence of problem:** Fold 1 produces near-random AUC-ROC (0.497) for MemStream on "easy" difficulty. This strongly suggests the AE failed to converge.

**Possible causes:**
1. 10 epochs is insufficient for 25D autoencoder to learn meaningful representations
2. Early stopping with patience=20 (config default) may trigger too soon
3. Noise std=0.1 may be too high, preventing convergence

**Recommendation:** 
- Increase to 50-100 epochs minimum
- Monitor validation loss during warmup
- Report per-epoch loss to verify convergence

### E.2. Train/Calibration Split (75%/25%)

**Status: ACCEPTABLE — Standard practice**

Using 75% for warmup/training and 25% for calibration is reasonable. However:

- The "calibration" data should be used for threshold calibration, not for initializing memory
- Current code uses the same data for both warmup and memory initialization (last 10% of training data)

**Issue:** Memory initialization from training data may lead to memorization. Consider:
- Using a separate hold-out set for memory initialization
- Initializing memory with synthetic samples (e.g., interpolation)

### E.3. Memory Initialization from Training Data

**Status: RISKY — Potential memorization**

```python
# From memstream_core.py warmup()
memory_data = X[int(n * 0.9):]
with torch.no_grad():
    memory_encoded = self.ae.encoder(memory_data)
```

Memory is initialized with encoded samples from training data. This means:
- Memory slots are populated with training representatives
- Reconstruction error on training data will be artificially low
- Test data far from training distribution may be unfairly scored as anomalous

**This is a form of information leakage.** The memory should either:
1. Be initialized randomly (random normal initialization)
2. Use a separate calibration set
3. Start empty and accumulate during streaming

### E.4. Batch vs Per-Record Scoring

**Status: WRONG FOR STREAMING CLAIM — Critical mismatch**

```python
# Batch scoring used in benchmark
scores_ms = ms.score_batch(X_test)  # Scores all at once

# Per-record scoring NOT used
# score = ms.score_one(x)  # Not in benchmark
```

**The problem:** `score_batch()` computes distances to memory for all test samples simultaneously, but the **streaming paradigm requires per-record updates**:

1. Score record t
2. Update memory with record t (if BAR grants budget)
3. Move to record t+1

**Current benchmark:**
- Scores all records against the same static memory
- Does NOT update memory during scoring
- This is **batch evaluation of a streaming algorithm** — fundamentally invalid

**Fix:** Implement true online evaluation:
```python
ms_streaming = ms.clone()  # Fresh memory
for i in range(len(X_test)):
    score = ms_streaming.score_one(X_test[i:i+1])
    should_update, reason = bar.should_update_memory({}, score, 'test')
    if should_update:
        ms_streaming.memory_update(X_test[i:i+1])
```

### E.5. BAR Controller Evaluation

**Status: INCOMPLETE — BAR rate measured but not used in scoring**

The benchmark measures BAR rate (1.5% in quick run) but:

1. **BAR is not integrated into MemStream scoring evaluation**
2. **No comparison between full MemStream (100% updates) vs BAR-MemStream (1-5% updates)**
3. **BAR rate measurement is on a single fold** — not representative

**Required comparison:**
| Configuration | Description | Expected Outcome |
|---------------|-------------|------------------|
| MemStream-Full | 100% memory update rate | Best accuracy, highest latency |
| MemStream-BAR | ~2% memory update rate | Slightly lower accuracy, much lower latency |
| MemStream-Frozen | 0% update (batch) | Baseline for comparison |

**Current gap:** You measure BAR rate but don't compare accuracy vs full MemStream. The BAR contribution to accuracy is **unquantified**.

---

## F. Streaming vs Batch Evaluation

### F.1. Static Test Set Validity

**Status: INSUFFICIENT for streaming algorithms**

Testing streaming algorithms on static test sets is valid only for **offline comparison**. For true streaming evaluation:

| Requirement | Current Status | Gap |
|-------------|---------------|-----|
| Online scoring | ❌ Batch scoring | Scores all records simultaneously |
| Memory adaptation | ❌ Memory frozen after warmup | No online learning |
| Temporal evaluation | ❌ No time ordering in scoring | Order doesn't matter |
| Drift handling | ❌ No drift in test | Cannot measure adaptation |

**For publication:** If claiming streaming capability, you MUST show:
1. Performance over time (sliding window evaluation)
2. Performance degradation/recovery around injected drift points
3. Comparison of frozen vs updating memory

### F.2. Streaming Simulation with Drift

**Status: NOT IMPLEMENTED**

A proper streaming evaluation would:
1. Create a time-ordered stream from test data
2. Inject synthetic concept drift at known points
3. Measure detection delay and accuracy recovery
4. Compare algorithms under identical drift scenarios

**Recommended setup:**
```python
# Pseudocode for streaming evaluation
stream = create_temporal_stream(X_test, y_test)
ms = MemStreamCore(cfg).warmup(X_train)

drift_injected_at = [5000, 10000, 15000]
for t, (x, y) in enumerate(stream):
    score = ms.score_one(x)
    ms.memory_update(x)  # If BAR grants
    
    if t in drift_injected_at:
        inject_drift(x)  # Shift distribution
    
    # Measure: detection delay, accuracy after drift
```

### F.3. Latency Measurement

**Status: NOT MEASURED — Critical for streaming claims**

The benchmark measures training time but **NOT scoring latency**:

```python
# From benchmark_ca_memstream.py
t0 = time.perf_counter()
scores_ms = ms.score_batch(X_test)
t_score = (time.perf_counter() - t0) * 1000
```

This measures **batch scoring time**, not per-record latency. For streaming:
- Measure time per record (`time.perf_counter()` inside loop)
- Report mean, p50, p95, p99 latency
- Compare to sklearn IF (typically O(n log n) for IsolationForest scoring)

**Target:** Streaming systems require <100ms latency for real-time alerting.

---

## G. Top 10 Critical Issues

### CRITICAL (Must fix before publication)

#### Issue #1: MemStream Fold 1 Produces Near-Random AUC-ROC
**Severity:** CRITICAL  
**Location:** `ca_memstream_scientific_benchmark.csv` rows 13, 35, 57

**Problem:** Fold 1 easy shows AUC-ROC = 0.497 (essentially random) and AUC-PR = 0.0009 (worse than random).

```csv
1,0.49730209106625944,0.0009275092251595124,easy,MemStream_25D
```

**Why it's problematic:** This is not a minor variance issue — it's a **catastrophic failure** indicating the model learned nothing. Possible causes:
1. Random seed produced degenerate initialization
2. Warmup failed to converge in 10 epochs
3. Data pipeline error (wrong fold loaded)

**Fix:** 
- Investigate fold 1 specifically: check warmup loss curve, final loss, memory state
- Increase warmup epochs to 50+
- Add sanity checks after warmup (loss should be below threshold)
- If fold 1 is systematically bad, exclude it with justification

---

#### Issue #2: Batch Evaluation of Streaming Algorithm
**Severity:** CRITICAL  
**Location:** `benchmark_ca_memstream.py` lines 66-68

**Problem:** The benchmark scores all test records against a frozen MemStream model. This is **batch evaluation**, not streaming evaluation.

**Why it's problematic:** You claim CA-MemStream is a streaming algorithm, but the benchmark doesn't evaluate:
- Online memory updates
- Adaptation to new patterns
- Performance degradation under drift
- Recovery after concept shift

**Fix:** Implement true streaming evaluation with online memory updates (see Section E.4 pseudocode).

---

#### Issue #3: Oracle Threshold Calibration
**Severity:** CRITICAL  
**Location:** `run_sequential_v4.py` lines 469-479

**Problem:** `find_threshold_f1()` searches for the best F1 threshold using ground truth labels. This produces **unrealistic F1/Precision/Recall** values.

**Why it's problematic:** In production, you don't have labels to optimize thresholds. The reported F1=0.7 might actually be F1=0.2 with contamination-based threshold.

**Fix:** Use contamination-based threshold (e.g., `np.percentile(scores, 95)` for 5% contamination) for all operational metrics.

---

### HIGH (Should fix, impacts validity)

#### Issue #4: No Statistical Hypothesis Testing
**Severity:** HIGH  
**Location:** Benchmark framework (missing entirely)

**Problem:** No p-values, confidence intervals, or effect sizes are reported.

**Why it's problematic:** 
- Cannot determine if observed differences are statistically significant
- AUC-PR differences of 0.003 (hard difficulty) may not be significant
- Publication reviewers will reject results without statistical rigor

**Fix:** 
- Implement Wilcoxon signed-rank test for pairwise comparisons
- Use Holm-Bonferroni correction for multiple comparisons
- Report 95% confidence intervals via bootstrap

---

#### Issue #5: Only 2 Algorithms Compared
**Severity:** HIGH  
**Location:** `benchmark_ca_memstream.py` (the quick benchmark used for results)

**Problem:** Only sklearn IF vs MemStream is compared. The full benchmark framework has more algorithms, but the actual results are from a 2-algorithm comparison.

**Why it's problematic:** 
- Cannot assess MemStream relative to state-of-art
- Missing critical baselines (OCSVM, DeepSVDD, LOF)
- A 2-algorithm comparison is not a "scientific benchmark"

**Fix:** Run the full `run_sequential_v4.py` benchmark and report all algorithms.

---

#### Issue #6: MemStream Memory Initialization from Training Data
**Severity:** HIGH  
**Location:** `memstream_core.py` lines 350-360

**Problem:** Memory is initialized with encoded training samples, leading to potential memorization.

**Why it's problematic:** Test samples similar to training will get low scores not because they're normal, but because they're memorized. This inflates accuracy on stationary data but fails under drift.

**Fix:** 
- Initialize memory randomly (Gaussian with training statistics)
- Or use separate calibration set for memory initialization
- Or start with empty memory and accumulate online

---

#### Issue #7: AUC-PR as Primary Metric
**Severity:** HIGH  
**Location:** Benchmark design

**Problem:** AUC-PR is used as primary metric but is problematic for anomaly detection:

**Why it's problematic:**
- AUC-PR depends heavily on anomaly rate (varies across folds/difficulties)
- Harder to interpret than AUC-ROC (no "accuracy" intuition)
- Baseline varies from 0.05 (5% anomaly) to 0.5 (50% anomaly)

**Fix:** 
- Use AUC-ROC as primary (more interpretable)
- Report AUC-PR with explicit baseline (random = anomaly_rate)
- Add Precision@K and Recall@K for operational metrics

---

#### Issue #8: Difficulty Levels Lack Operational Definition
**Severity:** HIGH  
**Location:** Dataset generation (not reviewed)

**Problem:** "Easy/medium/hard" difficulties are used but how they were constructed is undocumented.

**Why it's problematic:** 
- Cannot reproduce the benchmark
- Cannot assess if difficulties are truly ordinal
- Reviewers will question methodology validity

**Fix:** Document exact criteria:
- How was difficulty operationalized?
- What parameters control difficulty?
- Are there statistical tests confirming ordinal structure?

---

### MEDIUM (Nice to have, minor impact)

#### Issue #9: 11 Folds is Non-Standard
**Severity:** MEDIUM  
**Location:** Benchmark design

**Problem:** 11 folds is mathematically unusual (standard is 5 or 10).

**Why it's problematic:** 
- May indicate ad-hoc choice without justification
- Could produce unstable estimates with small test sets

**Fix:** Either:
1. Use 10 folds (standard, divisible dataset into 10%)
2. Use 5 folds (faster, sufficient for initial validation)
3. Justify 11 folds mathematically

---

#### Issue #10: Missing Latency Measurements
**Severity:** MEDIUM  
**Location:** `benchmark_ca_memstream.py` lines 66-68

**Problem:** Only batch timing measured, not per-record latency.

**Why it's problematic:** Streaming algorithms are often evaluated on latency, not just accuracy. A streaming algorithm with 100ms latency may be impractical.

**Fix:** 
- Measure per-record scoring time
- Report mean, p50, p95, p99 latency
- Compare against sklearn IF baseline

---

## H. Recommendations for Next Steps

### Immediate Actions (Before Next Run)

1. **Investigate Fold 1 failure** — Debug warmup loss, check data loading, increase epochs
2. **Implement statistical testing** — Add Wilcoxon test, confidence intervals, effect sizes
3. **Fix threshold calibration** — Use contamination-based thresholds, not oracle
4. **Run full benchmark** — Execute `run_sequential_v4.py` with all algorithms
5. **Document difficulty levels** — Write methodology for easy/medium/hard construction

### Short-term Improvements (1-2 weeks)

6. **Add streaming evaluation** — Implement online scoring with memory updates
7. **Compare BAR vs Full MemStream** — Quantify accuracy/latency tradeoff
8. **Add missing baselines** — OCSVM, DeepSVDD, LOF
9. **Report AUC-ROC as primary** — With AUC-PR as supplementary
10. **Add Precision@K metrics** — For operational interpretability

### Medium-term (Publication Quality)

11. **Implement drift injection** — Test streaming adaptation under synthetic drift
12. **Nemenyi post-hoc correction** — For multiple algorithm comparisons
13. **Bootstrap confidence intervals** — For all reported metrics
14. **Ablation study** — Contribution of AE vs Memory vs BAR
15. **Runtime analysis** — Latency per record, memory footprint

---

## Appendix: Summary Statistics from Results

### Quick Benchmark Results (from `ca_memstream_scientific_benchmark.csv`)

| Difficulty | Algorithm | Mean AUC-ROC | Std AUC-ROC | Mean AUC-PR | Std AUC-PR |
|------------|-----------|-------------|-------------|-------------|------------|
| easy | sklearn_IF | 0.9655 | 0.0047 | 0.0896 | 0.0379 |
| easy | MemStream_25D | 0.9087 | 0.1370 | 0.3455 | 0.1109 |
| medium | sklearn_IF | 0.9324 | 0.0116 | 0.0239 | 0.0108 |
| medium | MemStream_25D | 0.8986 | 0.0549 | 0.0744 | 0.0378 |
| hard | sklearn_IF | 0.8650 | 0.0245 | 0.0081 | 0.0031 |
| hard | MemStream_25D | 0.7588 | 0.0966 | 0.0122 | 0.0077 |

### Key Observations

1. **High variance in MemStream** — Std AUC-ROC of 0.137 on easy (vs 0.005 for IF) indicates instability
2. **MemStream outperforms IF on AUC-PR** — But AUC-PR is problematic as primary metric
3. **Performance degrades with difficulty** — Expected, but hard difficulty shows near-random AUC-ROC for MemStream
4. **Fold 1 anomaly** — Must be investigated

---

## References

1. Campos, G.O., et al. (2016). On the evaluation of unsupervised outlier detection. *TKDD*.
2. Chalapathy, R., & Chawla, S. (2019). Deep learning for anomaly detection. *ACM Computing Surveys*.
3. Bifet, A., & Gavalda, R. (2007). Learning from time-changing data with adaptive windowing. *KDD*.
4. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR*.
5. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot. *PLOS ONE*.

---

*Review prepared by ML Expert Agent*  
*Document Version: 1.0*  
*Last Updated: May 12, 2026*
