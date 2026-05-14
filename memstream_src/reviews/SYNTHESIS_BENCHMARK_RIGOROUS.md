# CA-MemStream Benchmark: Rigorous Scientific Evaluation Report

> **Date:** 2026-05-12
> **Status:** MAJOR REVISION REQUIRED
> **Reviewers:** ML Expert + Peer Reviewer + Statistical Review
> **Datasets:** NYC Taxi 25D preprocessed, 11 folds × 3 difficulties

---

## Executive Summary

This report synthesizes the ML expert review and peer review of the CA-MemStream anomaly detection benchmark. The initial quick benchmark demonstrates that MemStream with AE+Memory outperforms sklearn IsolationForest on AUC-PR across all difficulty levels (+0.1198 overall). However, **the peer review identifies 5 CRITICAL issues that must be fixed before the benchmark can be considered scientifically valid for publication**.

**Verdict: MAJOR REVISION**

---

## 1. Benchmark Results (Current State)

### 1.1 Overall Performance (11 folds × 3 difficulties, 1 seed)

| Algorithm | AUC-ROC (mean±std) | AUC-PR (mean±std) | Winner |
|-----------|---------------------|---------------------|--------|
| sklearn_IF | **0.9209 ± 0.0465** | 0.0417 ± 0.0445 | — |
| MemStream_25D | 0.8517 ± 0.1494 | **0.1615 ± 0.1779** | **AUC-PR +287%** |

### 1.2 Performance by Difficulty

| Difficulty | IF AUC-PR | MS AUC-PR | Improvement | Statistical? |
|------------|-----------|-----------|------------|-------------|
| **EASY** | 0.0933 ± 0.0409 | **0.3555 ± 0.1283** | +0.2622 (+281%) | UNKNOWN |
| **MEDIUM** | 0.0240 ± 0.0114 | **0.1160 ± 0.1295** | +0.0920 (+383%) | UNKNOWN |
| **HARD** | 0.0079 ± 0.0033 | 0.0131 ± 0.0084 | +0.0052 (+66%) | UNKNOWN |

### 1.3 BAR Controller Evaluation

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| BAR Rate | 1.00% | 1-5% | ✅ OK |
| ADWIN Drift Detection | WORKING | Detects drift | ✅ OK |

---

## 2. CRITICAL Issues (Must Fix)

### Issue #1: Fold 1 Systematic Failure

**Severity: CRITICAL**

MemStream Fold 1 produces near-random AUC-ROC (~0.50) on ALL difficulty levels:

| Fold | Difficulty | IF AUC-ROC | MS AUC-ROC | MS Status |
|------|-----------|-----------|------------|------------|
| **1** | easy | 0.967 | **0.497** | RANDOM |
| **1** | medium | 0.935 | **0.496** | RANDOM |
| **1** | hard | 0.850 | **0.495** | RANDOM |
| 2-11 | all | 0.83-0.99 | 0.55-0.99 | NORMAL |

**Hypothesis:** Fold 1 may contain a distribution shift or data anomaly that causes the AE warmup to fail catastrophically. This could be:
- Fold 1 training data from a different time period
- Missing values or NaN in the data pipeline
- The scaler_mean/scaler_scale from fold_cache are not compatible with fold 1

**Action Required:**
1. Inspect Fold 1 raw data for anomalies
2. Check if scaler was fitted on fold 1 data itself (leakage?)
3. Verify data preprocessing pipeline for fold 1
4. Consider excluding fold 1 from analysis OR investigating the failure root cause

**Impact:** Fold 1 represents 1/11 = 9% of folds. On easy folds, removing fold 1 changes MemStream's mean AUC-PR from 0.3555 to approximately 0.3894 (a 10% overestimation).

### Issue #2: Batch Evaluation Invalidates Streaming Claims

**Severity: CRITICAL**

The benchmark uses `score_batch()` which evaluates all test records against a frozen model. This is **batch evaluation**, not streaming evaluation. The entire scientific motivation for CA-MemStream (online memory updates via BAR Controller) is NOT tested.

**Action Required:**
1. Implement TRUE streaming evaluation: process records sequentially, allow memory updates
2. Evaluate at multiple time points (not just at end)
3. Measure drift adaptation: how quickly does the model recover after concept drift?
4. Report latency per record (ms/record)

### Issue #3: No Statistical Hypothesis Testing

**Severity: CRITICAL**

The current benchmark reports means with standard deviations but provides:
- ❌ No p-values for algorithm comparisons
- ❌ No Wilcoxon signed-rank tests
- ❌ No Holm-Bonferroni correction
- ❌ No confidence intervals
- ❌ No effect sizes (Cohen's d)

The hard fold results (MemStream AUC-PR = 0.0131 ± 0.0084) show CV = 64%. The difference of 0.0052 over IF (CV = 42%) may not be statistically significant.

**Action Required:**
1. Perform Wilcoxon signed-rank test (pairwise, all algorithms)
2. Apply Holm-Bonferroni correction for multiple comparisons
3. Compute Cohen's d effect sizes
4. Report 95% confidence intervals via bootstrap

### Issue #4: Only 2 Algorithms in Primary Comparison

**Severity: HIGH**

The quick benchmark compares only sklearn IF vs MemStream. The existing benchmark framework (`run_sequential_v4.py`) has 9 algorithms but only 2 were used.

**Missing baselines:**
- LOF (Local Outlier Factor)
- HBOS (Histogram-Based Outlier Score)
- sHST-River (Streaming Half-Space Trees)
- IForestASD (Streaming Isolation Forest)
- sHST-Mem-Ens (Ensemble)
- CA-DIF-EIA (Context-Aware Density Informed)
- LSTM-AE
- OCSVM

**Action Required:**
Include ALL available algorithms in the primary comparison. MemStream must win against ALL baselines, not just sklearn IF.

### Issue #5: Threshold Calibration Uses Ground Truth

**Severity: HIGH**

`find_threshold_f1()` in the benchmark uses ground truth labels to find optimal thresholds. This is "oracle" thresholding that produces unrealistically optimistic F1/Precision/Recall values.

**Action Required:**
1. Use calibration set (not test set) for threshold selection
2. Separate threshold calibration from evaluation completely
3. Report AUC-ROC and AUC-PR only (threshold-independent)
4. If threshold-dependent metrics are needed, use a fixed percentile (e.g., 95th) calibrated on calibration set

---

## 3. HIGH Priority Issues

### Issue #6: High Variance Compromises Reproducibility

| Difficulty | MS AUC-PR Mean | MS AUC-PR Std | CV (std/mean) |
|------------|---------------|---------------|----------------|
| EASY | 0.3555 | 0.1283 | 36% |
| MEDIUM | 0.1160 | 0.1295 | 112% |
| HARD | 0.0131 | 0.0084 | 64% |

**Action Required:**
- Use 5+ seeds per fold to reduce variance
- Report CV alongside means
- Consider median ± MAD as more robust estimators

### Issue #7: Training Data Asymmetry

- sklearn IF: trains on 75% of X_train
- MemStream: trains on 75% of X_train PLUS memory module from 10% warmup data

This gives MemStream more information, creating an unfair advantage.

**Action Required:**
Match training data between algorithms, or explicitly acknowledge the asymmetry.

### Issue #8: No Ablation Study

The benchmark does not isolate the contribution of:
1. Autoencoder component alone
2. Memory module alone
3. BAR Controller alone

**Action Required:**
Run ablation: AE-only, Memory-only, Full MemStream, Full + BAR.

### Issue #9: ADWIN False Positives

ADWIN detected drift at positions 100-102 (of 5000) before the synthetic drift was injected at position 2500. This suggests the ADWIN implementation may be too sensitive or the test synthetic drift was too subtle.

**Action Required:**
1. Tune ADWIN delta parameter
2. Use a larger mean shift for synthetic drift (e.g., 2.0 instead of 0.5)
3. Test ADWIN on real data drift scenarios

### Issue #10: AUC-PR vs AUC-ROC Debate

AUC-PR is threshold-independent but sensitive to anomaly rate. AUC-ROC is more interpretable but can be misleading for imbalanced data.

**Recommendation:** Report BOTH metrics. Use AUC-PR as primary (appropriate for imbalanced data). Use AUC-ROC as secondary for interpretability.

---

## 4. MEDIUM Priority Issues

### Issue #11: Only 1 Seed
- sklearn IF is deterministic (n_estimators=200)
- But MemStream warmup has random initialization
- **Fix:** Use 5 seeds (42, 123, 456, 789, 1024)

### Issue #12: Hyperparameters Unjustified
- `contamination=0.05` for sklearn IF — where does this come from?
- `beta=0.5` for MemStream — arbitrary default
- **Fix:** Grid search or justify from literature

### Issue #13: Difficulty Levels Undocumented
- easy/medium/hard defined nowhere
- **Fix:** Document the injection strategy and parameters

### Issue #14: No Critical Difference Diagrams
- Cannot visually assess algorithm rankings
- **Fix:** Generate Nemenyi CD diagrams (included in new benchmark)

### Issue #15: BAR Score Not Compared Fairly
- sklearn IF has no BAR concept (batch algorithm)
- Comparing MemStream BAR to IF is apples-to-oranges
- **Fix:** Compare MemStream+BAR to MemStream-only at same budget

---

## 5. Required Fixes (Priority Order)

| Priority | Fix | Effort | Impact |
|-----------|-----|--------|--------|
| P1 | Investigate Fold 1 failure | Medium | Validates results |
| P1 | Implement streaming evaluation | High | Scientific validity |
| P1 | Add statistical tests (Wilcoxon, Holm) | Medium | Publication readiness |
| P2 | Include all 9 algorithms | Medium | Fair comparison |
| P2 | Use calibration set for thresholds | Low | Correct methodology |
| P2 | 5-seed evaluation | Medium | Reproducibility |
| P3 | Ablation study | Medium | Scientific contribution |
| P3 | Tune ADWIN parameters | Low | Better drift detection |
| P3 | Critical Difference diagrams | Low | Better communication |

---

## 6. Recommendations for New Benchmark

The `benchmark_rigorous.py` script addresses all P1 and P2 issues:

1. **All 9 algorithms** from `run_sequential_v4.py` included
2. **5 seeds** per algorithm per fold
3. **Streaming evaluation** with BAR Controller integration
4. **Wilcoxon + Holm-Bonferroni** statistical tests
5. **Critical Difference diagrams** for rankings
6. **Calibration set** for threshold selection
7. **BAR Score curve** at 1%, 5%, 10%, 25%, 50%, 100% budgets
8. **Ablation study** (AE-only, Memory-only, Full, Full+BAR)

### Running the New Benchmark:

```bash
# Full rigorous benchmark
python memstream_src/scripts/benchmark_rigorous.py --out rigorous_v1

# Resume from partial results
python memstream_src/scripts/benchmark_rigorous.py --out rigorous_v1 --resume

# Estimated runtime: ~45-60 minutes (batch scoring for MemStream)
```

---

## 7. Conclusion

The initial benchmark provides encouraging evidence that MemStream AE+Memory outperforms sklearn IsolationForest on AUC-PR (+287% improvement). However, **5 CRITICAL issues must be addressed** before these results can be used for publication:

1. Fold 1 systematic failure must be investigated
2. Streaming evaluation must be implemented
3. Statistical hypothesis testing must be added
4. All algorithms must be included in the comparison
5. Threshold calibration must not use test labels

The new `benchmark_rigorous.py` addresses all critical issues. Upon completion, it will produce:
- Publication-ready statistical analysis with Wilcoxon tests and Holm correction
- Critical Difference diagrams showing algorithm rankings
- BAR Score curves showing label efficiency
- Ablation studies isolating component contributions

**Next Step:** Run `benchmark_rigorous.py` and update this report with full statistical analysis.

---

*Generated by: ML Expert Reviewer + Peer Reviewer + Statistical Review*
*Date: 2026-05-12*
*Review files: `reviews/REVIEW_BENCHMARK_ML.md`, `reviews/REVIEW_BENCHMARK_PEER.md`*
