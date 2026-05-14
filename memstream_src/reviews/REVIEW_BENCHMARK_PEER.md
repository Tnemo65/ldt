# Peer Review: CA-MemStream Scientific Benchmark

**Paper under review:** CA-MemStream Scientific Benchmark (v5)
**Reviewer:** Peer Reviewer (ICML/NeurIPS/VLDB style)
**Date:** 2026-05-12
**Submission venue:** [To be determined]

---

## Summary Statement

This submission presents a benchmark evaluating CA-MemStream (Context-Aware MemStream) against sklearn's IsolationForest on a traffic anomaly detection task. The benchmark uses AUC-ROC and AUC-PR as primary metrics across three difficulty levels (easy/medium/hard) with 11 folds per level. The claim is that CA-MemStream outperforms the baseline, particularly on the AUC-PR metric which is more appropriate for imbalanced anomaly detection.

**Overall assessment: Major Revision Required.** The benchmark has several critical methodological flaws that undermine its scientific validity. Most severely, only two algorithms are compared (MemStream vs sklearn IF), the statistical analysis is entirely absent (no p-values, no confidence intervals, no multiple comparison correction), and the IsolationForest is trained on a subset of the data while MemStream uses all of it—creating an unfair comparison. The variance in hard folds is extreme (AUC-PR ranges from 0.006 to 0.027 across folds), yet no statistical test quantifies whether the reported improvements are meaningful. These issues are individually sufficient to warrant rejection at any top venue.

---

## Major Comments

### M1: Only Two Algorithms Compared — Benchmark Scope is Insufficient

**Severity: CRITICAL**

A benchmark claiming to evaluate a new streaming anomaly detection system against "baselines" must include more than one baseline. The current benchmark compares only:

1. **MemStream_25D** (CA-MemStream)
2. **sklearn_IsolationForest** (with default parameters, contamination=0.05)

This is far below the standard set by comparable benchmarks in the literature (e.g., Chandola et al. 2009 survey benchmarks, the ODDS repository benchmarks, RiverML benchmarks). A credible benchmark should include:

- **Batch baselines:** One-Class SVM, LOF, Local Outlier Factor
- **Streaming baselines:** RS-Hash, Half-Space Trees, sHST-River (which appears in `benchmark_results_v5.csv` but not in the scientific benchmark!)
- **Publication baselines:** The original MemStream (WWW 2022), CA-DIF-EIA, LSTM-AE

The fact that `benchmark_results_v5.csv` contains results for 9 algorithms (MemStream, sklearn_IF, sklearn_OCSVM, sklearn_LOF, LSTM-AE, CA-DIF-EIA, sHST-River, IForestASD, CA-DIF-EIA streaming) but the "scientific" benchmark only reports 2 is deeply suspicious. This selective reporting raises concerns about which results were omitted and why.

**Required fix:** Either (a) expand the comparison to include all algorithms in `benchmark_results_v5.csv`, or (b) provide a rigorous justification for why only 2 algorithms are included, and show that the omitted baselines were pre-specified in an analysis plan.

---

### M2: Statistical Analysis is Entirely Absent

**Severity: CRITICAL**

The benchmark reports mean and standard deviation of AUC-PR and AUC-ROC across 11 folds, but performs **no statistical inference whatsoever**. No p-values, no confidence intervals, no correction for multiple comparisons, no effect sizes beyond raw differences.

Consider the hard difficulty results. MemStream_25D AUC-PR ranges from 0.006 to 0.027 across folds — a **4.5x range** in observed performance. sklearn_IF ranges from 0.005 to 0.013 — a **2.6x range**. With this much fold-to-fold variance, a simple mean comparison is unreliable. A paired t-test or Wilcoxon signed-rank test (with proper justification for the test choice) is essential.

The ML review (`REVIEW_ML_v4.md`) mentions multi-seed evaluation with 10 seeds and paired t-tests, but this is **completely absent** from the actual benchmark script (`benchmark_ca_memstream.py`). The script uses a single seed (42), tests only 2 algorithms, and reports no statistical tests.

**Required fixes:**
1. Report 95% confidence intervals (bootstrap CI is preferred over mean±std for n=11)
2. Conduct a paired statistical test (Wilcoxon signed-rank recommended given small n=11)
3. Apply Bonferroni or Holm-Bonferroni correction for multiple comparisons (3 difficulties × 2 metrics = 6 comparisons)
4. Report effect sizes (Cohen's d or Vargha-Delaney A) alongside p-values

---

### M3: Training Data Asymmetry — MemStream Has More Data

**Severity: CRITICAL**

```python
# Line 50: IsolationForest uses only 75% for training
X_warmup = X_train[:int(n_train * 0.75)]

# Lines 58-64: MemStream uses the same 75% for warmup, but...
ms = MemStreamCore(cfg=cfg, device='cpu')
ms.warmup(X_warmup, epochs=10, batch_size=256, verbose=False)
```

Critically, **IsolationForest trains on 75% of X_train** while **MemStream trains on 75% of X_train** (same data) but then also uses memory to store representations of normal data. However, the real asymmetry is:

1. sklearn IF: trains on 75% of X_train, scores on X_test
2. MemStream: warmup on 75% of X_train, **plus** the memory module encodes normal patterns from the warmup data

More concerning: IsolationForest is given only the warmup portion, but MemStream also initializes its memory from the same warmup data. This means MemStream has an informational advantage — it stores explicit representations of normal patterns.

But the reverse concern is equally valid: IsolationForest should be trained on the **full** X_train (or at least comparable training data) and then scored. The 75% split benefits neither algorithm fairly.

**Required fix:** Both algorithms should use identical training data. If sklearn IF trains on 75%, MemStream warmup should also be on 75%. If sklearn IF trains on 100%, MemStream warmup should also be on 100%. The comparison must be fair.

---

### M4: sklearn IsolationForest is Not Properly Tuned

**Severity: CRITICAL**

```python
# Line 53: sklearn IF uses hardcoded hyperparameters
if_clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
```

sklearn's IsolationForest is configured with only `n_estimators=200` and `contamination=0.05`. This is essentially using default settings. Meanwhile, MemStream has multiple hyperparameters:

- `warmup_epochs=10` (chosen arbitrarily)
- `warmup_batch_size=256` (standard choice)
- `beta=0.5` (arbitrary, see M6)

If the claim is that CA-MemStream is better, the comparison must show that sklearn IF was given a fair chance. At minimum, a hyperparameter sweep over `contamination` (e.g., [0.01, 0.02, 0.05, 0.1]) and `n_estimators` (e.g., [100, 200, 500]) should be performed, with the best configuration selected per fold.

This is especially important because AUC-PR is sensitive to the contamination rate — a higher contamination rate will increase the base rate of anomalies, which directly affects precision and thus AUC-PR.

**Required fix:** Either (a) tune sklearn IF hyperparameters using the same calibration data as MemStream, or (b) use multiple contamination rates and report the best result, or (c) explicitly state this as a limitation.

---

### M5: No Beta Calibration — Arbitrary Threshold

**Severity: CRITICAL**

```python
# Line 63: beta is set to 0.5 without justification
ms.set_beta(0.5)
```

Beta (the threshold parameter in MemStream's scoring function) is set to 0.5. The ML review mentions that beta should be calibrated on held-out calibration data using ECE/MCE metrics, but:

1. The benchmark script does not perform any calibration
2. The beta=0.5 value is not justified
3. Different beta values could dramatically change AUC-PR results

For IsolationForest, `contamination=0.05` implicitly sets a threshold. For MemStream, `beta=0.5` is arbitrary. This asymmetry means the threshold comparison is unfair.

**Required fix:** Either calibrate beta on held-out data (matching the methodology in the v5 plan), or sweep beta across a range and report the best result, or document this as a limitation.

---

### M6: Extreme Variance in Hard Folds — Results are Unstable

**Severity: HIGH**

Looking at the hard difficulty results for MemStream_25D:

| Fold | AUC-PR |
|------|--------|
| 1 | 0.00094 |
| 2 | 0.0101 |
| 3 | 0.0127 |
| 4 | 0.0105 |
| 5 | 0.0113 |
| 6 | 0.0273 |
| 7 | 0.0208 |
| 8 | 0.0087 |
| 9 | 0.0267 |
| 10 | 0.0086 |
| 11 | 0.0060 |

**Coefficient of variation: ~58%** — This is extremely high. Fold 1 has AUC-PR of 0.00094 (essentially random) while Fold 6 has 0.0273 (28x better). This suggests:

1. The "hard" difficulty level is not well-controlled across folds
2. Some folds have test distributions very different from the training distribution
3. Results are not reproducible across folds

For comparison, sklearn_IF on hard folds has CV of ~27%, which is also high but more reasonable.

**Required fix:** Investigate why fold 1 has AUC-PR of 0.00094 for MemStream but 0.006 for IF (a 6x difference in the wrong direction). Report results with and without outlier folds. Consider whether the hard difficulty splits are meaningful.

---

### M7: Missing from Results — Selective Reporting of Metrics

**Severity: HIGH**

The `ca_memstream_scientific_benchmark.csv` only contains AUC-ROC and AUC-PR. But `benchmark_results_v5.csv` includes F1, Precision, Recall, FPR, TP, FP, TN, FN, and optimal_threshold. These additional metrics are critical for anomaly detection:

- **FPR (False Positive Rate)** is essential for operational cost analysis
- **F1 Score** balances precision and recall
- **Precision and Recall** are needed for operational decision-making

The scientific benchmark omits these, which is concerning. AUC-PR is a good summary metric, but FPR at a fixed threshold is what operations teams care about. A method with high AUC-PR but high FPR at operational thresholds may be useless in practice.

**Required fix:** Report FPR, Precision, Recall, and F1 at a fixed threshold (e.g., threshold achieving 80% recall) in addition to AUC metrics.

---

## Minor Comments

### M8: Single Seed — No Robustness Check

**Severity: HIGH**

The benchmark script uses only `random_state=42` for IsolationForest and `seed=42` for MemStream. The ML review mentions 10-seed evaluation, but this is completely absent from the actual benchmark. Results with a single seed are not robust — different seeds could yield different rankings.

**Fix:** Evaluate across at least 5 seeds (as the ML review suggests) and report mean ± CI across seeds, not just across folds.

---

### M9: Benchmark is Batch, Not Streaming

**Severity: MEDIUM**

```python
# Line 67: Batch scoring
scores_ms = ms.score_batch(X_test)
```

The benchmark evaluates MemStream using batch scoring (`score_batch`), not streaming scoring (`score_one`). The entire motivation for CA-MemStream is online anomaly detection — but the benchmark evaluates it in batch mode. This is a fundamental disconnect.

The BAR Controller and ADWIN drift detection are tested separately on synthetic data, not on the actual benchmark data. The benchmark does not evaluate:
- Whether memory updates during streaming improve performance
- Whether drift detection works on real data
- Whether the BAR Controller maintains appropriate memory update rates

**Fix:** Either (a) evaluate in streaming mode with memory updates, or (b) rename this as an "offline evaluation" and add a separate streaming benchmark.

---

### M10: Difficulty Classification Method Unclear

**Severity: MEDIUM**

The benchmark classifies anomalies as "easy", "medium", or "hard" but the methodology for this classification is not documented. Looking at the results, the easy/medium/hard distinction is meaningful (performance degrades), but the exact criteria matter:

- Is difficulty based on anomaly magnitude (how far from normal)?
- Is it based on anomaly density (how many anomalies)?
- Is it based on distributional shift from training?
- Is it based on feature-space distance?

Without this documentation, the benchmark is not reproducible.

**Fix:** Document the difficulty classification criteria in the benchmark methodology section.

---

### M11: Scaler Statistics — Potential Cross-Contamination

**Severity: MEDIUM**

The fold cache contains `scaler_mean.npy` and `scaler_scale.npy` files. It is unclear whether these scalers were computed:
- Per-fold (from that fold's training data only — correct)
- Across all data (leakage — incorrect)
- From a global dataset (serious leakage — incorrect)

This must be explicitly documented and verified.

**Fix:** Verify that scaler statistics are computed only from each fold's training data, and document this in the methodology.

---

### M12: ADWIN Drift Detection Test Uses Synthetic Data

**Severity: MEDIUM**

```python
# Lines 158-161: Synthetic drift test
for i in range(N_DRIFT):
    score = 0.5 + np.random.normal(0, 0.05) if i < 2500 else 1.0 + np.random.normal(0, 0.1)
```

The ADWIN drift detection is tested on manually injected distributional shift (mean shift from 0.5 to 1.0), not on the real benchmark data. This validates that ADWIN *can* detect drift in synthetic data, but does not validate that it *does* detect drift in the actual traffic data.

**Fix:** Report how many ADWIN drift detections occurred during the real benchmark evaluation, and correlate these with any performance changes.

---

### M13: No Code Availability Statement

**Severity: MEDIUM**

Reproducibility requires more than just "the code is in the repository." For a benchmark paper, the following must be provided:

1. A standalone benchmark script that can be run end-to-end
2. A Docker container with all dependencies
3. The exact version of all dependencies
4. The random seed protocol
5. A README with step-by-step reproduction instructions

**Fix:** Add a reproducibility appendix with exact commands to reproduce all results.

---

### M14: 11 Folds is an Unusual Number

**Severity: LOW**

Standard practice is 5-fold or 10-fold cross-validation. 11 folds is unusual and not explained. If these correspond to 11 months of data (Jan-Nov), this should be documented. If they are random splits, 10 folds would be standard.

**Fix:** Document why 11 folds were used, or switch to 10-fold CV.

---

### M15: BAR Rate Target Range is Inconsistent

**Severity: LOW**

```python
# Line 145: Target is 0.5% to 10%
target_ok = 0.005 <= bar_rate <= 0.10

# Line 206: Target is 1-5%
print(f"3. BAR Controller: rate={bar_rate*100:.2f}% (target 1-5%)")
```

The target BAR rate is stated as both "0.5-10%" and "1-5%" in the same script. This inconsistency suggests the target was not carefully determined.

**Fix:** Determine the correct target range and use it consistently.

---

## Questions for Authors

1. **Why are only 2 of 9 algorithms from `benchmark_results_v5.csv` included in the scientific benchmark?** What happened to the other 7 baselines (OCSVM, LOF, LSTM-AE, CA-DIF-EIA, sHST-River, IForestASD, CA-DIF-EIA streaming)?

2. **How was the beta=0.5 threshold chosen?** Was this calibrated on held-out data, or is it arbitrary?

3. **Why does IsolationForest train on only 75% of X_train?** Should it not train on the same data as MemStream's warmup phase?

4. **What is the exact methodology for the "easy/medium/hard" difficulty classification?** Please provide the algorithm or criteria.

5. **Were the scaler statistics computed per-fold or globally?** If per-fold, please confirm that no information leaks across folds.

6. **Why is the benchmark evaluated in batch mode when the paper claims streaming performance advantages?** What is the expected streaming behavior, and why is it not measured?

7. **With CV~58% on hard folds, how confident are you that the results are stable?** Have you tested with different random seeds?

8. **Why is contamination=0.05 hardcoded for IsolationForest?** Was this chosen to match the expected anomaly rate in the test data?

---

## Specific Recommendations

### Immediate (Required for Resubmission)

1. **Add all 9 algorithms** from `benchmark_results_v5.csv` to the scientific benchmark, or provide written justification for omission
2. **Add statistical tests:** Wilcoxon signed-rank test for paired comparisons, Holm-Bonferroni correction for 6 comparisons
3. **Report 95% CIs** using bootstrap (n=1000) rather than asymptotic CIs
4. **Fix training data asymmetry:** Both IF and MemStream should use identical training data
5. **Tune or justify IsolationForest hyperparameters** — at minimum sweep contamination rate
6. **Calibrate beta** on held-out calibration data, or sweep and report best

### Short-term (Major Revisions)

7. **Evaluate in streaming mode** — batch evaluation does not demonstrate the streaming advantage
8. **Add multi-seed robustness** — evaluate across 5-10 seeds
9. **Document difficulty classification** methodology
10. **Report operational metrics** — FPR, Precision, Recall at fixed thresholds
11. **Add reproducibility documentation** — exact commands, Docker setup, seed protocol

### Long-term (Strengthen the Paper)

12. **Compare against published baselines** — MemStream (WWW 2022), RS-Hash, Half-Space Trees
13. **Evaluate on public benchmark datasets** — SHDB, Arrhythmia, KDD-Cup, U2R (standard anomaly detection benchmarks)
14. **Add ablation study** — compare 25D MemStream vs 40D CA-MemStream
15. **Test drift detection on real data** — report ADWIN events during benchmark evaluation

---

## Overall Verdict

**Verdict: MAJOR REVISION**

The benchmark has critical methodological flaws that prevent scientific evaluation of the claims:

1. Only 2 algorithms compared (should be 9+)
2. No statistical inference whatsoever
3. Training data asymmetry favoring MemStream
4. Unjustified IsolationForest hyperparameters
5. Arbitrary beta threshold

These are not minor issues that can be fixed in camera-ready — they require re-running the experiments with proper controls. However, the foundation (11-fold cross-validation, multiple difficulty levels, AUC-PR metric choice) is sound. With fixes to the 5 CRITICAL issues, this could be a competitive submission.

**Confidence in this review:** High. The issues identified are unambiguous violations of standard benchmark methodology. The selective reporting of 2 out of 9 algorithms, the absence of statistical tests despite high variance, and the training data asymmetry are not matters of interpretation — they are objective flaws.

---

*Peer Review — CA-MemStream Scientific Benchmark*
*Review completed: 2026-05-12*
