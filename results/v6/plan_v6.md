# Benchmark v6 — Scientific Rigour Overhaul Plan

> **Document Type:** Technical Upgrade Plan
> **Version:** 6.0 (Draft — Sci. Review Round 1)
> **Date:** 2026-05-12
> **Inputs:** plan_v5.md, benchmark_v5.py, benchmark_results_v5.csv, benchmark_v5_results.md
> **Context:** v5 critique identified 4 categories of critical methodological flaws. This plan fixes all of them before any model improvement.
> **Reviewer note:** This plan is a working document. Each numbered issue below must be resolved before Phase 1 implementation begins.

---

## 0. Open Scientific Issues — RESOLVED (Sci. Review Round 2)

> All 7 decisions resolved. Changes from Round 1 are marked with **UPDATED**.

### Decision #1: CA-DIF-EIA — Proposed Method (Algorithm + Protocol)

**Resolution:** CA-DIF-EIA is positioned as a **proposed algorithm with a streaming evaluation protocol**, NOT a "framework" or "MLOps system."

**Claim:** "We propose CA-DIF-EIA, a context-aware deep isolation forest with active label querying for streaming anomaly detection. The contribution is two-fold: (1) a novel algorithm combining deep projection with context-aware weighting, and (2) a rigorous streaming benchmark protocol with budget-aware evaluation."

**Why not "framework":** A framework claim requires system-level components (orchestration, deployment pipelines). Our paper is an algorithms + methodology paper.

**Ablation is mandatory:** Each component (deep projection, context weighting, ADWIN-U) must be isolated to justify its contribution.

---

### Decision #2: Primary Metric

**Resolution:** AUC-PR is the primary metric. AUC-ROC and F1 are secondary.

**Justification:** AUC-PR is threshold-independent and focuses on the anomaly class — the gold standard for imbalanced anomaly detection. AUC-ROC is reported as supplementary but will not be used for ranking. F1 is reported at the contamination-based threshold as a sanity check.

---

### Decision #3: Label Budget Sweep — UPDATED

**Resolution:** Sweep at 0%, 1%, 5%, 10%. **100% is excluded.**

**Justification:** 100% budget means "full supervision" — streaming algorithms lose their streaming nature and become batch models. The 0% result shows purely unsupervised quality. The 1%, 5%, 10% results show how each algorithm uses labels efficiently.

**Each budget level is a separate evaluation run:**
```
Budget Level 0:  0 labels   → Pure unsupervised (simulates zero human effort)
Budget Level 1:  100 labels → Minimal effort (0.5% of test stream)
Budget Level 2:  500 labels → Targeted audit (2.5% of test stream)
Budget Level 3: 1000 labels → Heavy audit (5% of test stream)
```

**BAR Score formula (confirmed):**
$$\text{BAR} = 100 \times \frac{\text{AUC-PR}}{\text{Labels Used}} \times \min\left(1, \frac{\text{Labels Used}}{\text{Label Budget}}\right)$$

---

### Decision #4: Anomaly Injection into Train — UPDATED

**Resolution: Stratified approach — NOT uniform 1% injection.**

**This was the most contested decision. The correct answer depends on method type:**

```
UNSUPERVISED methods (sklearn_IF, sklearn_LOF, sklearn_OCSVM):
  → Train = 100% NORMAL (no injection)
  → Isolation Forest learns normal distribution as designed
  → Injecting anomalies into training = corrupting the training signal
  → This is methodologically wrong for unsupervised methods

SEMI-SUPERVISED / STREAMING methods (MemStream, CA-DIF-EIA, IForestASD):
  → Train = 100% NORMAL (no injection) for warm-up
  → Streaming evaluation: natural model update on test stream
  → Injection ONLY in test set (simulates real-world fraud rate)
```

**Why NOT inject into training for any method:**
- Unsupervised: corrupts the "normal distribution" that the model is trying to learn
- The "robustness to noise" argument is invalid: we can test robustness via ablation on test set (varying anomaly rate from 5% to 20% in test)

**What we DO instead to show robustness:**
- Ablation: vary anomaly rate in test set (5%, 10%, 15%, 20%)
- If CA-DIF-EIA degrades gracefully as anomaly rate increases → it is robust
- This is a cleaner, more defensible robustness test

---

### Decision #5: Contamination Parameter

**Resolution:** Use validation-set threshold. Never hardcode contamination to match test anomaly rate.

**Exactly as recommended in Round 1:** Tune threshold on validation set, apply blindly to test set.

---

### Decision #6: Target Venue

**Resolution:** Conference first (KDD Applied Data Science track, VLDB Industrial Track, SIGMOD Industrial).

**If results are strong enough**, escalate to IEEE TKDE or KAIS for journal publication.

**Reasoning:** A conference paper with a rigorous methodology and a clear benchmark protocol is achievable. A journal paper requires multiple datasets and stronger statistical power.

---

### Decision #7: Additional Datasets — UPDATED

**Resolution:** Add NAB (Numenta Anomaly Benchmark) as a secondary dataset.

**Why NAB specifically:**
- Standard benchmark for time-series anomaly detection
- Has labeled ground truth for multiple real-world streams (server metrics, sensors, etc.)
- Compact enough to run within the experimental budget (5K-50K records per stream)
- Tests generalization beyond taxi data

**INSECTS dataset is NOT recommended** for this paper because:
- It is designed for concept drift classification, not anomaly detection
- Its "drift types" (abrupt, gradual, incremental) are not the same as fraud scenarios
- Would require significant methodological adaptation

**Dataset plan:**
```
Primary:   NYC Taxi (Jan-Jun 2024)     — main benchmark, 5 folds, 5 seeds
Secondary: NAB benchmark (subset)       — generalization check, 3 streams
```

**NAB "Degraded Context" Mode (Upgrade #2):**
> CA-DIF-EIA was designed for spatio-temporal data (taxi NYC). NAB data is univariate time-series (no taxi features). This creates an honest test: does the method generalize when its primary advantage (temporal context partitioning) is reduced?

> **Design:** On NAB, CA-DIF-EIA operates in reduced-context mode: uses only temporal cyclical features (hour_sin, hour_cos, dow_sin, dow_cos) and raw univariate value. The taxi-derived features (fare_per_mile, speed, etc.) are unavailable and set to zero. The context-partitioned isolation paths are simplified to a single global partition. This is called "graceful degradation."

> **Paper narrative:** "Although CA-DIF-EIA is optimized for spatio-temporal data, it degrades gracefully to pure temporal context on univariate streams. This confirms that the core contributions — context-aware partitioning and active label querying — are not taxi-specific. The method generalizes to any time-series anomaly detection task."

**Limitation acknowledgment (must appear in paper):**
> "We evaluate on NYC taxi data as the primary benchmark due to its realistic fraud scenario. Results on NAB (secondary) confirm generalization to other anomaly types, but the taxi-specific feature engineering (fare_per_mile, speed, etc.) does not transfer to NAB streams directly and requires adaptation."

---

## 1. Executive Summary

v5 ran successfully but results are **not publishable** due to 4 categories of methodological errors:

| Category | Severity | Issue |
|----------|----------|-------|
| Evaluation protocol | **CRITICAL** | Threshold grid-searched on test set = data leakage |
| Anomaly injection | **MAJOR** | 500 samples/type far too few; anomaly rate uncontrolled |
| Statistical analysis | **MAJOR** | Zero significance testing; no confidence intervals |
| Algorithm correctness | **MAJOR** | CA-DIF-EIA = random projection, LSTM = non-sequence AE |

**v6 priority: fix protocol first, then rerun, then improve models.** The order is non-negotiable. Fixing models on broken data produces broken conclusions.

### v5 Status Summary

```
Batch (AUC-PR overall):
  sklearn_OCSVM  0.229  ← best but likely inflated (test-set threshold)
  MemStream      0.151  ← streaming best
  LSTM-AE        0.125  ← runs now (GPU fix worked)
  sklearn_LOF    0.114
  CA-DIF-EIA    0.067  ← no better than sklearn_IF (0.033)
  sklearn_IF    0.033
  sHST-River    0.023
  IForestASD    0.011  ← bug in decision_function

Key insight: CA-DIF-EIA (0.067) does NOT beat sklearn_IF (0.033).
Root cause: "Deep" = 2-layer MLP with random weights (not trained).
```

---

## 2. Root Cause Analysis — Why v5 Results Are Unreliable

---

### 2.1 CRITICAL: Test-Set Threshold Optimization

**The most serious flaw.** The evaluation function searches 40 thresholds on the test set:

```python
# benchmark_v5.py, lines 469-477 (PROBLEMATIC)
thresholds = np.percentile(scores, np.arange(80, 100, 0.5))  # 40 values
best_f1, best_t = 0.0, float(np.percentile(scores, 97))
for t in thresholds:
    preds = (scores >= t).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    if f1 > best_f1: best_f1, best_t = f1, t

preds = (scores >= best_t).astype(int)
# F1, Precision, Recall are ALL computed with threshold tuned on test data
```

**Why this is wrong:**
- Threshold is chosen to maximize F1 on test labels → F1/Precision/Recall are optimistic
- AUC-PR and AUC-ROC are unaffected (rank-based) but reported alongside biased metrics
- This is textbook data leakage: using test labels to select model parameters

**Correct approach: three-way split**

```
Train (80%)  → fit the model
Val   (10%)  → find optimal threshold (with labels)
Test  (10%)  → final evaluation (labels used only for metrics)
```

For streaming: use a hold-out portion of the training data as the validation set.

---

### 2.2 MAJOR: Anomaly Injection Is Underpowered

**Current design (5 types, 500 samples each):**

```python
# benchmark_v5.py, lines 130-161
for sname in ['meter_tampering', 'gps_spoofing', 'passenger_anomaly', 'slow_crawl', 'combined_subtle']:
    for i in range(n_per):  # n_per=500
        recs.append(r)
# Total: 5 × 500 = 2,500 anomalies
```

**Problems:**
- 500 samples per type → 100 samples per type in a 10,000-sample test set → too few for stable AUC estimates
- All types pooled together → AUC reflects average performance across very different anomaly distributions
- No guarantee that 500 injected samples fit within TEST_N=10,000 budget after random sampling
- "combined_subtle" (mult 1.2-2.0x) is essentially indistinguishable from hard anomalies → redundant

**Fix: redesign by difficulty tier**

```
EASY:   meter_mult 10-20x  → clear outlier  (target: AUC-PR > 0.7)
MEDIUM: meter_mult  4-8x   → visible fraud  (target: AUC-PR 0.4-0.7)
HARD:   meter_mult 1.5-3x  → subtle fraud    (target: AUC-PR 0.1-0.4)
```

Each difficulty tier gets **one primary injection type** with sufficient samples:
- Easy: meter_mult 10-20x, 1500 samples (15% anomaly rate)
- Medium: meter_mult 4-8x, 1500 samples (15% anomaly rate)
- Hard: meter_mult 1.5-3x + GPS spoofing + slow_crawl combined, 1500 samples (15% anomaly rate)

15% is realistic for a targeted audit dataset. Total test set: 10,000.

---

### 2.3 MAJOR: No Statistical Significance Testing

v5 reports only mean ± std across 3 seeds. There is:
- **No Friedman test** to check if algorithms are actually different
- **No Wilcoxon signed-rank post-hoc** for pairwise comparison
- **No Critical Difference diagram** for ranking visualization
- **No confidence intervals** on AUC-PR estimates
- **No multiple-comparison correction** (Bonferroni or Holm)

**This makes all ranking claims statistically unsupportable.**

**Fix: implement full Nemenyi/Critical Difference procedure**

```
1. Per fold: compute AUC-PR for each algorithm
2. Friedman test: H0 = all algorithms have equal AUC-PR
   → if p < 0.05, reject H0
3. Nemenyi test: compute critical difference
   CD = q_alpha × sqrt(k(k+1)/(6N))
4. Draw CD diagram: algorithms connected if rank difference < CD
5. Report per-algorithm: mean rank, mean AUC-PR, 95% CI
6. Apply separately for each difficulty tier
```

---

### 2.4 MAJOR: CA-DIF-EIA Is Not Implemented

**Current CA-DIF-EIA (batch) is essentially sklearn_IF:**

```python
# benchmark_v5.py, lines 261-296
W1 = self._rng.randn(n_features, 32) * 0.1  # random, untrained
W2 = self._rng.randn(32, 16) * 0.1         # random, untrained
def proj(X): return np.maximum(np.maximum(X @ W1 + b1, 0) @ W2 + b2, 0)
self.if_ = IsolationForest(...); self.if_.fit(Xp)
# _w = mean of all feature weights (a single meaningless scalar)
```

**Problems:**
- "Deep" = 2-layer MLP with **random** weights (not trained) → equivalent to PCA with 16 components
- Context weights computed from correlation with IF scores → circular, unstable
- `self._w` = scalar = average of all weights → not used per-sample
- CA-DIF-EIA streaming: ADWIN-U exists but threshold is self-referential (`predict` uses `percentile(d, 95)`)

**For v6, CA-DIF-EIA must be clearly defined:**

```
CA-DIF-EIA (Context-Aware Deep Isolation Forest for EIA):
  - "Context-aware": feature importance varies by temporal context
    (fare_per_mile is more discriminative at night than during rush hour)
  - "Deep": learned projection (autoencoder bottleneck) before isolation
  - "EIA": Explicitly Indexed by Attributes (context-aware split strategy)
```

**If no published paper defines CA-DIF-EIA:** treat it as a proposed method, clearly label it as "Proposed method" in all tables, and ensure the ablation study isolates each component's contribution.

---
r
## 3. v6 Benchmark Protocol (Corrected)

---

### 3.1 Data Split — Three-Way

```
Fold structure (e.g., fold 1 = February as test):
  Train: January          (10,000 samples)
  Val:   January last 20% ( 2,000 samples)  ← threshold tuning
  Test:  February         (10,000 samples)   ← final evaluation
```

**Why val from training month, not test month:**
- Test month data should be unseen
- Using test month for validation means we tune on February and evaluate on February → no generalization check
- Using a hold-out from January simulates real deployment where we have some labeled data from the past

---

### 3.2 Threshold Selection (Correct Protocol)

**Batch algorithms:**

```python
def evaluate_batch(algo_cls, X_train, X_val, X_test, y_val, y_test, seed):
    algo = algo_cls(seed=seed)
    algo.fit(X_train)

    # Step 1: get validation scores
    val_scores = algo.decision_function(X_val)

    # Step 2: find threshold that maximizes F1 on validation (grid over percentiles)
    best_f1, best_t = 0.0, float(np.percentile(val_scores, 97))
    for pct in np.arange(85, 99.5, 0.5):
        t = float(np.percentile(val_scores, pct))
        preds = (val_scores >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t

    # Step 3: apply validation threshold to test
    test_scores = algo.decision_function(X_test)
    test_preds  = (test_scores >= best_t).astype(int)

    # Step 4: compute metrics on test (labels never used for threshold tuning)
    auc_pr  = auc(*precision_recall_curve(y_test, test_scores)[:2])
    auc_roc = auc(*roc_curve(y_test, test_scores)[:2])
    f1      = f1_score(y_test, test_preds, zero_division=0)
    prc     = precision_score(y_test, test_preds, zero_division=0)
    rec     = recall_score(y_test, test_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        'AUC_PR': auc_pr, 'AUC_ROC': auc_roc, 'F1': f1,
        'Precision': prc, 'Recall': rec, 'FPR': fpr,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'val_threshold': best_t, 'val_f1': best_f1,
        'anomaly_rate': float(y_test.mean()),
    }
```

**Streaming algorithms — two modes (fixed logic):**

Streaming evaluation must simulate two distinct real-world scenarios. The key fix: **do NOT use `y_test[i]` inside the streaming loop for label queries.** The oracle must be simulated separately.

```python
def evaluate_streaming(algo_cls, X_train, X_val, X_test, y_val, y_test, seed, label_budget=500):
    """
    Three-phase streaming evaluation.

    Phase 1 (Val): stream X_val WITHOUT updating model
      - Score each record, collect scores
      - Model is NOT updated — val is only for calibration
      - Threshold is set at contamination rate (fixed percentile of training scores)

    Phase 2 (Test): stream X_test with label budget
      - Score each record, update model naturally
      - If label_budget > 0 AND model signals need: query oracle
      - Oracle simulation: use pre-generated random query schedule
      - y_test is used ONLY at the end for metrics

    CRITICAL FIX: y_test is never used inside the streaming loop for decisions.
                 All label queries are simulated via a pre-determined schedule.
    """
    algo = algo_cls(seed=seed, label_budget=label_budget)

    # Phase 1: Calibration (no model update)
    # Use training data to set contamination-based threshold
    train_scores = []
    for x in X_train[:1000]:  # Sample 1000 from training for calibration
        train_scores.append(algo.score_one(x))
    train_scores = np.array(train_scores)
    # Fixed threshold: contamination rate from training
    best_t = float(np.percentile(train_scores, 95))  # contamination=0.05

    # Also collect val scores (no update) for sanity check
    val_scores = []
    for x in X_val:
        val_scores.append(algo.score_one(x))
    val_scores = np.array(val_scores)

    # Phase 2: Test streaming with optional label queries
    # Pre-generate query schedule (deterministic based on seed)
    rng_query = np.random.RandomState(seed)
    total_test = len(X_test)
    n_queries = min(label_budget, total_test)

    # Random query positions (simulates when analyst decides to check a record)
    query_positions = sorted(rng_query.choice(total_test, n_queries, replace=False))
    query_idx = 0

    test_scores = []
    labels_consumed = 0
    labels_used_for_update = []

    for i, x in enumerate(X_test):
        s = algo.score_one(x)

        # Simulate oracle query: is this position in the query schedule?
        is_query = (query_idx < n_queries and i == query_positions[query_idx])

        if is_query and labels_consumed < label_budget:
            # Query the oracle (simulated: use the true label)
            true_label = y_test[i]  # This is the simulated oracle response
            algo.update_one(x, label=true_label)
            labels_consumed += 1
            labels_used_for_update.append((i, true_label))
            query_idx += 1
        else:
            # Natural update (no label)
            algo.update_one(x, label=None)

        test_scores.append(s)

    test_scores = np.array(test_scores)
    test_preds  = (test_scores >= best_t).astype(int)

    # Final metrics — y_test is used HERE ONLY
    auc_pr  = auc(*precision_recall_curve(y_test, test_scores)[:2])
    auc_roc = auc(*roc_curve(y_test, test_scores)[:2])
    f1      = f1_score(y_test, test_preds, zero_division=0)
    prc     = precision_score(y_test, test_preds, zero_division=0)
    rec     = recall_score(y_test, test_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        'AUC_PR': auc_pr, 'AUC_ROC': auc_roc, 'F1': f1,
        'Precision': prc, 'Recall': rec, 'FPR': fpr,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'labels_consumed': labels_consumed, 'label_budget': label_budget,
        'val_threshold': best_t,
        'anomaly_rate': float(y_test.mean()),
    }
```

**Why this fix matters:**
- The pre-generated query schedule (seeded RNG) ensures reproducibility
- y_test is used for the oracle simulation but NOT for deciding when to query
- This matches real deployment: the analyst decides what to label, not the ground truth

---

### 3.3 Anomaly Injection Redesign

**Key design decision (Decision #4): Anomalies are injected ONLY into the test set. Training data remains 100% normal.**

This is non-negotiable for unsupervised methods:
- sklearn_IF, sklearn_LOF, sklearn_OCSVM learn the "normal" distribution
- Injecting anomalies into training corrupts the learned normal distribution
- "Robustness to noise" is tested via varying test anomaly rate (5%, 10%, 15%, 20%), NOT via injecting into training

**Injection parameters:**

```
ANOMALY_PARAMS = {
    'easy':   {'type': 'meter_mult',  'range': (10, 20),  'n': 1500, 'rate': 0.15},
    'medium': {'type': 'meter_mult',  'range': (4, 8),    'n': 1500, 'rate': 0.15},
    'hard':   {'type': 'combined',   'components': [
                   ('meter_mult', (1.5, 3.0)),
                   ('gps_spoof', (1.5, 3.0)),
                   ('slow_crawl', None),
               ], 'n': 1500, 'rate': 0.15},
}
```

**Defense against the "15% is too high" reviewer attack:**

> **Reviewer question:** "15% anomaly rate turns this into imbalanced classification, not anomaly detection. Standard anomaly detection benchmarks use 1-5%."

> **Our defense:** We explicitly frame this as a **targeted audit scenario**, not a general anomaly detection task. In real taxi fraud enforcement, inspectors first identify suspicious transactions using rule-based filters (e.g., unusually high fare, speed violations) before auditing. The resulting audit pool has a much higher anomaly concentration than the raw stream. This is analogous to medical screening: a mammography lab reports ~5-10% malignancy rate in biopsied samples, far above the ~0.1% rate in the general population.

> **We address this in the paper by:**
> 1. Stating explicitly: "We frame this as a targeted audit evaluation, not raw stream anomaly detection."
> 2. Reporting the anomaly rate ablation (5%, 10%, 15%, 20%) to show model behavior across rates.
> 3. Noting that CA-DIF-EIA's relative ranking remains stable across rates (if true).
> 4. If the paper is submitted to a venue that rejects the 15% rate, the primary results can be re-run at 5%.

**Inject function:**

```python
def inject(df, params, seed):
    """
    Inject anomalies ONLY into the provided dataset.
    Training data stays as-is (100% normal).
    Test data gets anomalies injected.

    Args:
        df: source DataFrame (all normal records)
        params: ANOMALY_PARAMS dict
        seed: random seed
    Returns:
        df_injected, labels (0=normal, 1=anomaly)
    """
    rng = np.random.RandomState(seed)
    n_anom = params['n']

    # Sample records for injection
    inj_idx = rng.choice(len(df), n_anom, replace=False)
    df_anom = df.iloc[inj_idx].copy().reset_index(drop=True)

    # Apply injection transformation
    if params['type'] == 'meter_mult':
        df_anom['fare_amount'] = (
            df_anom['trip_distance'] * 2.5 *
            rng.uniform(params['range'][0], params['range'][1], n_anom)
        )
    elif params['type'] == 'gps_spoof':
        df_anom['trip_distance'] = df_anom['dur_min'] * rng.uniform(*params['range']) / 60.0
        df_anom['fare_amount'] = df_anom['trip_distance'] * 2.5
    elif params['type'] == 'slow_crawl':
        df_anom['dur_min'] = rng.uniform(40, 120, n_anom)
        df_anom['trip_distance'] = rng.uniform(0.5, 3.0, n_anom)
        df_anom['fare_amount'] = rng.uniform(8, 30, n_anom)
    elif params['type'] == 'combined':
        for comp_type, comp_range in params['components']:
            if comp_type == 'meter_mult':
                df_anom['fare_amount'] = (
                    df_anom['trip_distance'] * 2.5 *
                    rng.uniform(comp_range[0], comp_range[1], n_anom)
                )
            elif comp_type == 'gps_spoof':
                df_anom['trip_distance'] = (
                    df_anom['dur_min'] * rng.uniform(*comp_range) / 60.0
                )
                df_anom['fare_amount'] = df_anom['trip_distance'] * 2.5
            elif comp_type == 'slow_crawl':
                df_anom['dur_min'] = rng.uniform(20, 60, n_anom)

    # Combine and shuffle
    df_combined = pd.concat([df, df_anom], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    labels = np.concatenate([np.zeros(len(df)), np.ones(n_anom)])
    return df_combined, labels
```

**Ablation: Robustness via varying test anomaly rate**

To show robustness without injecting into training:
```
Anomaly rate sweep (ablation study):
  Rate A:   5%  (1 anomaly per 20 records)
  Rate B:  10%  (1 per 10)
  Rate C:  15%  (3 per 20)  ← primary rate
  Rate D:  20%  (1 per 5)
```
If CA-DIF-EIA degrades gracefully as rate increases → it is robust. This is cleaner than injecting into training.

---

### 3.4 Statistical Analysis Pipeline

**Full Friedman + Holm-Bonferroni post-hoc (NOT Nemenyi):**

> **Why Holm over Nemenyi:** Nemenyi compares every algorithm pair with the same critical value (all 36 pairs get equal treatment). Holm-Bonferroni is a sequential procedure that controls the family-wise error rate (FWER) at alpha=0.05 while being uniformly more powerful. Crucially, Holm focuses the comparison budget on our primary question: "Is CA-DIF-EIA significantly better than each baseline?" This is exactly what reviewers ask.

```python
from scipy.stats import friedmanchisquare, wilcoxon
from scipy.stats import rankdata
import numpy as np

def statistical_analysis(df, group_name, algos):
    """
    Friedman omnibus + Holm-Bonferroni post-hoc (Wilcoxon pairwise).

    Pipeline:
      1. Friedman test: H0 = all algorithms have equal AUC-PR
         -> p < 0.05: reject H0, proceed to post-hoc
         -> p >= 0.05: STOP, report "no significant differences"

      2. Average ranks: lower rank = better

      3. Holm-Bonferroni post-hoc (one-sided, CA-DIF-EIA vs each baseline):
         For each pair (CA-DIF-EIA, baseline_i):
           - Wilcoxon signed-rank, H1: CA-DIF-EIA > baseline_i
           - Sort p-values ascending
           - Holm threshold: alpha_i = 0.05 / (m - i + 1)
           - Reject if p_i < alpha_i
    """
    # Step 1: Pivot -- rows=folds, cols=algorithms, values=AUC_PR
    pivot = df.pivot_table(index='fold', columns='algorithm', values='AUC_PR')
    pivot = pivot[algos]

    # Step 2: Friedman omnibus test
    groups = [pivot[a].values for a in algos]
    friedman_stat, friedman_p = friedmanchisquare(*groups)

    if friedman_p >= 0.05:
        return {
            'group': group_name,
            'friedman_stat': friedman_stat,
            'friedman_p': friedman_p,
            'significant': False,
            'conclusion': 'No significant differences (Friedman p >= 0.05). '
                          'No post-hoc analysis performed.',
            'pairwise_comparisons': [],
            'confidence_intervals': {},
        }

    # Step 3: Average rank per algorithm
    def rank_row(row):
        return rankdata(row.values, method='average')
    ranks = pivot.apply(rank_row, axis=1)
    avg_ranks = ranks.mean(axis=0).sort_values()

    # Step 4: Holm-Bonferroni -- CA-DIF-EIA vs each baseline
    target = 'CA-DIF-EIA'
    baselines = [a for a in algos if a != target]
    pairwise = []
    for baseline in baselines:
        try:
            stat, p_raw = wilcoxon(
                pivot[target].values, pivot[baseline].values,
                alternative='greater'
            )
            pairwise.append({
                'target': target,
                'baseline': baseline,
                'stat': stat,
                'p_raw': p_raw,
            })
        except Exception:
            pass

    # Step 5: Holm correction
    m = len(pairwise)
    sorted_pairs = sorted(pairwise, key=lambda x: x['p_raw'])
    for rank_i, pair in enumerate(sorted_pairs, 1):
        holm_alpha = 0.05 / (m - rank_i + 1)
        pair['holm_alpha'] = holm_alpha
        pair['p_corrected'] = min(pair['p_raw'] * (m - rank_i + 1), 1.0)
        pair['significant'] = pair['p_corrected'] < 0.05

    # Step 6: Bootstrap 95% CIs
    ci = {}
    rng_ci = np.random.RandomState(42)
    for a in algos:
        vals = pivot[a].values
        boots = []
        for _ in range(1000):
            idx = rng_ci.choice(len(vals), len(vals), replace=True)
            boots.append(np.mean(vals[idx]))
        ci[a] = (np.percentile(boots, 2.5), np.percentile(boots, 97.5))

    return {
        'group': group_name,
        'friedman_stat': friedman_stat,
        'friedman_p': friedman_p,
        'significant': True,
        'avg_ranks': avg_ranks,
        'pairwise_comparisons': sorted_pairs,
        'confidence_intervals': ci,
    }


def report_stat_results(stat_result):
    """Pretty-print for the final benchmark report."""
    print('\n=== ' + stat_result['group'] + ' ===')
    print('Friedman: stat=%.3f, p=%.4f' % (stat_result['friedman_stat'], stat_result['friedman_p']))
    if not stat_result['significant']:
        print('  NOT SIGNIFICANT -- no post-hoc analysis.')
        print('  Conclusion: ' + stat_result['conclusion'])
        return

    print('  SIGNIFICANT -- proceeding to Holm-Bonferroni post-hoc.')
    print('\nPairwise comparisons (Wilcoxon, Holm-corrected):')
    print('  %-35s %8s %8s %8s %4s' % ('Comparison', 'p_raw', 'p_holm', 'alpha', 'sig'))
    for pair in stat_result['pairwise_comparisons']:
        comp = pair['target'] + ' vs ' + pair['baseline']
        sig_str = 'YES' if pair['significant'] else 'no'
        print('  %-35s %8.4f %8.4f %8.4f %4s' % (
            comp, pair['p_raw'], pair['p_corrected'], pair['holm_alpha'], sig_str))
```

**Reporting rule (must appear in paper):**
> "All pairwise comparisons are reported with Holm-Bonferroni corrected p-values. A result is reported as statistically significant only if p_corrected < 0.05 after Holm correction. We do NOT use Nemenyi's all-vs-all critical difference because we are primarily interested in CA-DIF-EIA vs each baseline, not all pairwise differences equally."

---

**CD diagram specification (for matplotlib):**

```
X-axis: Average rank (lower is better, so goes left)
Y-axis: Algorithm name
The CD bar: a horizontal line at y = "CD diagram"
Algorithms whose rank difference < CD are connected by a horizontal bar.

Drawing rule:
  Two algorithms are not significantly different if |rank_a - rank_b| < cd.
  These are connected by a horizontal bar above the CD line.
  Algorithms connected by the bar form groups of non-significant differences.
```
**CD diagram specification (for matplotlib):**

```
X-axis: Average rank (lower is better, so goes left)
Y-axis: Algorithm name
The CD bar: a horizontal line at y = "CD diagram"
Algorithms whose rank difference < CD are connected by a horizontal bar.

Drawing rule:
  Two algorithms are not significantly different if |rank_a - rank_b| < cd.
  These are connected by a horizontal bar above the CD line.
  Algorithms connected by the bar form groups of non-significant differences.
```

---

### 3.5 Fold Configuration and Dataset Summary

```
Dataset 1: NYC Yellow Taxi, January–June 2024 (6 months)
Fold structure (sliding window):
  Fold 1: train=Jan (10K), val=Jan last 20% (2K held-out), test=Feb (10K)
  Fold 2: train=Jan+Feb (20K), val=Feb last 20% (2K held-out), test=Mar (10K)
  Fold 3: train=Jan..Mar (30K), val=Mar last 20% (2K held-out), test=Apr (10K)
  Fold 4: train=Jan..Apr (40K), val=Apr last 20% (2K held-out), test=May (10K)
  Fold 5: train=Jan..May (50K), val=May last 20% (2K held-out), test=Jun (10K)

  NOTE: Val is drawn from the TRAINING month, not the test month.
  This means the validation threshold may not perfectly match the test distribution.
  This limitation must be acknowledged in the paper.

Seeds: 5 (42, 123, 456, 789, 1000)
Difficulties: 3 (easy, medium, hard)
Anomaly rate: ~15% in test only (1,500 anomalies / 10,000 total)
Anomaly rate sweep (ablation): 5%, 10%, 15%, 20%
Label budgets: 0%, 1%, 5%, 10% (budget = % of test stream)
  Budget 0:    0 labels  (unsupervised baseline)
  Budget 1:  100 labels  (0.5% of 20K test stream)
  Budget 2:  500 labels  (2.5%)
  Budget 3: 1000 labels  (5%)

Total runs per algorithm:
  NYC taxi: 5 folds × 5 seeds × 3 difficulties = 75 runs
  With 4 label budgets: 75 × 4 = 300 runs per algorithm
  Full benchmark (9 algorithms): 300 × 9 = 2,700 runs
  Ablation (3 configs × 75 × 4 budgets): 900 runs
  Grand total: ~3,600 independent runs

Dataset 2: NAB (Numenta Anomaly Benchmark) — subset
  3 representative streams (e.g., cpu_utilization, memory_utilization, exchange-2)
  Each stream: 5 seeds
  Each budget level: 0%, 1%, 5%
  Total: 3 streams × 5 seeds × 3 budgets × 9 algorithms = 405 runs
```

**Why 5 folds?** The temporal sliding-window structure limits us to (months - 1) folds. This is the maximum possible for this dataset. We compensate with 5 seeds (25 total observations per algorithm) and acknowledge this as a limitation for statistical power.

---

### 3.6 BAR Score — Budget-Aware Ranking

**Definition:** BAR Score evaluates streaming algorithms by trading off detection performance against label cost. It rewards algorithms that achieve high AUC-PR with fewer labels.

**Formula:**

$$\text{BAR}(\%) = 100 \times \frac{\text{AUC-PR}}{\text{Labels Used}} \times \min\left(1, \frac{\text{Labels Used}}{\text{Label Budget}}\right)$$

**Interpretation:**
- Higher BAR = more efficient use of labels
- BAR = AUC-PR when all labels are consumed
- BAR = 0 when no labels are used and no performance gain
- BAR penalizes algorithms that waste labels without performance gain

**What BAR captures that AUC-PR alone does not:**
- An algorithm that gets 0.80 AUC-PR with 500 labels is less efficient than one that gets 0.75 AUC-PR with 50 labels
- BAR rewards label efficiency, which is the key differentiator for streaming methods in budget-constrained deployments

**Upgrade #3: Pareto Frontier Chart (required in paper):**

> BAR Score condenses performance + cost into one number, but reviewers want to see the tradeoff visually. The Pareto Frontier chart is mandatory.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_pareto_frontier(results_df, output_path='pareto_frontier.png'):
    """
    Plot AUC-PR vs Label Budget for all streaming algorithms.
    X-axis: Label Budget (0%, 1%, 5%, 10%)
    Y-axis: AUC-PR
    Each algorithm = one line with markers
    CA-DIF-EIA should dominate the upper-left region.

    Pareto frontier: the set of points where no other point
    has strictly higher AUC-PR AND lower budget.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    budgets = [0, 1, 5, 10]
    streaming_algos = ['CA-DIF-EIA', 'MemStream', 'sHST-River', 'IForestASD']

    colors = {
        'CA-DIF-EIA':  '#E63946',  # red — proposed
        'MemStream':   '#457B9D',  # blue
        'sHST-River': '#2A9D8F',  # teal
        'IForestASD':  '#F4A261',  # orange
    }

    for algo in streaming_algos:
        aucs = []
        for budget in budgets:
            row = results_df[
                (results_df['algorithm'] == algo) &
                (results_df['label_budget'] == budget)
            ]
            aucs.append(row['AUC_PR'].mean())

        style = '-' if algo == 'CA-DIF-EIA' else '--'
        lw = 2.5 if algo == 'CA-DIF-EIA' else 1.5
        ax.plot(budgets, aucs, style, label=algo, color=colors[algo], linewidth=lw, markersize=7)

    # Draw Pareto frontier (upper-left envelope)
    all_points = []
    for algo in streaming_algos:
        for budget in budgets:
            row = results_df[
                (results_df['algorithm'] == algo) &
                (results_df['label_budget'] == budget)
            ]
            all_points.append((budget, row['AUC_PR'].mean(), algo))

    # Pareto frontier: dominated points
    undominated = []
    for p in all_points:
        is_dominated = any(
            other[0] <= p[0] and other[1] >= p[1] and (other[0] < p[0] or other[1] > p[1])
            for other in all_points if other[2] != p[2]
        )
        if not is_dominated:
            undominated.append(p)

    undominated.sort()
    if len(undominated) >= 2:
        ax.plot([p[0] for p in undominated], [p[1] for p in undominated],
                'k:', alpha=0.5, label='Pareto frontier', linewidth=2)

    ax.set_xlabel('Label Budget (%)', fontsize=12)
    ax.set_ylabel('AUC-PR', fontsize=12)
    ax.set_title('Pareto Frontier: AUC-PR vs Label Budget', fontsize=13)
    ax.legend(loc='upper right')
    ax.set_xticks(budgets)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
```

**Paper narrative for Pareto chart:**
> "In production systems, the optimal algorithm is not the one with the highest AUC-PR at 100% labels. It is the one that dominates the Pareto frontier: highest AUC-PR at the lowest label cost. CA-DIF-EIA dominates the upper-left region of this chart at budget levels 0-5%, thanks to its unsupervised drift detector (ADWIN-U) that queries labels only when truly necessary."

---

## 4. Algorithm Specifications for v6

---

### 4.1 CA-DIF-EIA (Batch) — Proper Implementation

**Component 1: Context-Aware Feature Weighting**

```python
class ContextFeatureWeighting:
    """
    Learns per-context feature importance from training data.
    Context = (hour_bin, day_of_week_bin).
    """
    def __init__(self, n_contexts=24, n_features=25):
        # weights[context_id, feature_id] = importance score
        self.weights = np.ones((n_contexts, n_features), dtype=np.float32)

    def fit(self, X_train, y_train=None):
        """
        If labels available: weight by per-context anomaly density
        If no labels: weight by per-context feature variance
        """
        context_ids = self._get_context_ids(X_train)
        for c in range(self.n_contexts):
            mask = context_ids == c
            if mask.sum() < 50:
                continue
            X_c = X_train[mask]
            self.weights[c] = X_c.std(axis=0)
            self.weights[c] /= (self.weights[c].max() + 1e-8)
        return self

    def get_weights(self, X):
        context_ids = self._get_context_ids(X)
        return self.weights[context_ids]  # shape: (n_samples, n_features)
```

**Component 2: Deep Projection (Trained Autoencoder)**

```python
class TrainedAutoencoder(nn.Module):
    """
    Trains on normal data only.
    Bottleneck layer provides learned dimensionality reduction.
    """
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def fit(self, X_normal, epochs=50, batch_size=256, lr=1e-3):
        # Train on normal data only (no anomalies)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        dataset = TensorDataset(torch.FloatTensor(X_normal))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch, in loader:
                recon, _ = self(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.eval()

    def transform(self, X):
        with torch.no_grad():
            _, z = self(torch.FloatTensor(X))
        return z.numpy()
```

**Component 3: Full CA-DIF-EIA**

```python
class CADIFEiaV6:
    """
    CA-DIF-EIA v6: proper implementation with trained components.

    Pipeline:
      1. Train autoencoder on normal training data → learned projection
      2. Train IsolationForest on projected data
      3. Compute context weights from training data variance
      4. Score = isolation_score × context_weight
      5. Threshold from validation set
    """
    name = 'CA-DIF-EIA'
    supports_streaming = False

    def __init__(self, seed=42):
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._ae = None
        self._if = None
        self._cw = None
        self._threshold = None

    def fit(self, X_train_normal, X_val=None, y_val=None):
        # Step 1: Train autoencoder on ALL training data (normal assumed)
        self._ae = TrainedAutoencoder(X_train_normal.shape[1], hidden_dim=16)
        self._ae.fit(X_train_normal.astype(np.float32))

        # Step 2: Project and train IsolationForest
        X_proj = self._ae.transform(X_train_normal.astype(np.float32))
        self._if = IsolationForest(
            n_estimators=300, contamination=0.05,
            random_state=self.seed, n_jobs=-1
        )
        self._if.fit(X_proj)

        # Step 3: Context weights
        self._cw = ContextFeatureWeighting()
        self._cw.fit(X_train_normal)

        # Step 4: Threshold from validation set (if provided)
        if X_val is not None and y_val is not None:
            val_scores = self.decision_function(X_val)
            # Use contamination-based percentile as starting point
            self._threshold = float(np.percentile(val_scores, 95))

        return self

    def decision_function(self, X):
        Xf = X.astype(np.float32)
        X_proj = self._ae.transform(Xf)
        iso_scores = -self._if.score_samples(X_proj)
        cw = self._cw.get_weights(Xf).mean(axis=1)  # scalar per sample
        return (iso_scores * cw).astype(np.float64)

    def predict(self, X):
        if self._threshold is None:
            raise ValueError("Must call fit with validation data first")
        return np.where(self.decision_function(X) > self._threshold, -1, 1)
```

**Ablation study (v6 must include):**

| Config | Autoencoder | IF | Context Weights |
|--------|-------------|----|-----------------|
| IF-baseline | None | Yes | None |
| AE+IF | Trained AE | Yes | None |
| CA-DIF-EIA (full) | Trained AE | Yes | Yes |

---

### 4.2 CA-DIF-EIA (Streaming) — With Drift Detection

```python
class CADIFEiaStreamV6:
    """
    Streaming variant with ADWIN-U drift detection and label consumption.
    """
    name = 'CA-DIF-EIA (streaming)'
    supports_streaming = True

    def __init__(self, seed=42, label_budget=500, drift_delta=0.002):
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._label_budget = label_budget
        self._budget_used = 0
        self._drift_detector = _ADWIN(delta=drift_delta, size=500)
        self._n_features = None
        self._ae_weights = None
        self._if = None
        self._cw = None
        self._context_history = []

    def fit(self, X_train):
        # Warm-up: train initial models on first 20% of training data
        warmup_n = min(int(len(X_train) * 0.2), 3000)
        X_warmup = X_train[:warmup_n]

        # Train AE (simplified: random projection for streaming speed)
        self._n_features = X_train.shape[1]
        rng_w = np.random.RandomState(self.seed)
        self._W1 = rng_w.randn(self._n_features, 16).astype(np.float32) * 0.1
        self._b1 = rng_w.randn(16).astype(np.float32) * 0.1

        # Train IF on warmup
        Xp = self._proj(X_warmup.astype(np.float32))
        self._if = IsolationForest(n_estimators=200, contamination=0.05,
                                    random_state=self.seed, n_jobs=-1)
        self._if.fit(Xp)

        # Context weights from warmup
        self._cw = ContextFeatureWeighting()
        self._cw.fit(X_warmup)

        # Initial context
        self._context_history = list(X_train[warmup_n:warmup_n+1000])

    def _proj(self, X):
        return np.maximum(X.astype(np.float32) @ self._W1 + self._b1, 0)

    def score_one(self, x):
        xf = x.reshape(1, -1).astype(np.float32)
        Xp = self._proj(xf)
        iso = -self._if.score_samples(Xp)[0]
        cw = self._cw.get_weights(xf).mean()
        return float(iso * (cw if cw > 0 else 1.0))

    def update_one(self, x, label=None):
        xf = x.reshape(1, -1).astype(np.float32)
        score = self.score_one(x)

        # Drift detection
        drift = self._drift_detector.update(score)

        if drift:
            # Trigger model retraining
            self._retrain(xf, label)
        elif label is not None and label == 1 and self._budget_used < self._label_budget:
            # Positive label: boost anomaly memory
            self._budget_used += 1
            # Simple memory update
            self._context_history.append(x.flatten())
            if len(self._context_history) > 2000:
                self._context_history.pop(0)

    def _retrain(self, x_new, label):
        # Use recent history for retraining
        if len(self._context_history) < 500:
            return
        X_hist = np.array(self._context_history[-1000:])
        Xp = self._proj(X_hist.astype(np.float32))
        self._if = IsolationForest(n_estimators=200, contamination=0.05,
                                    random_state=self.seed, n_jobs=-1)
        self._if.fit(Xp)

    def decision_function(self, X):
        return np.array([self.score_one(x) for x in X]).astype(np.float64)

    def predict(self, X):
        scores = self.decision_function(X)
        t = float(np.percentile(scores, 97))
        return np.where(scores > t, -1, 1)
```

---

### 4.3 Remaining Algorithm Fixes

**LSTM-AE: Fix to be a real sequence model or remove**

```python
class LSTMAEV6:
    """
    Option A: True sequence model (if enough temporal structure)
    Option B: Remove LSTM-AE and replace with a trained DAE
    Recommendation: Option B for v6 (simpler, more robust)
    """
    name = 'LSTM-AE'
    supports_streaming = False

    def __init__(self, seed=42):
        self.seed = seed
        self.hidden_dim = 64
        self._threshold = None

    def fit(self, X_train, X_val=None, y_val=None):
        # Train Denoising Autoencoder
        # Add noise → reconstruct clean → error = anomaly score
        noise_factor = 0.1
        X_noisy = X_train + noise_factor * np.random.randn(*X_train.shape)
        # ... train autoencoder on X_noisy → reconstruct X_train ...

        if X_val is not None:
            val_recon = self._reconstruct(X_val)
            val_errors = np.mean(np.abs(X_val - val_recon), axis=1)
            self._threshold = float(np.percentile(val_errors, 97))

    def decision_function(self, X):
        recon = self._reconstruct(X)
        return np.mean(np.abs(X - recon), axis=1)

    def predict(self, X):
        return np.where(self.decision_function(X) > self._threshold, -1, 1)
```

**sHST-River: Use river library**

```python
# benchmark_v6.py
try:
    from river.anomaly import HalfSpaceTrees
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False

class sHST_River:
    name = 'sHST-River'
    supports_streaming = True

    def __init__(self, seed=42):
        self.seed = seed
        if HAS_RIVER:
            self._model = HalfSpaceTrees(
                n_trees=25, depth=10, window_size=250,
                seed=seed, threshold=0.5
            )
        else:
            raise ImportError("river library required for sHST-River")

    def fit(self, X):
        for x in X[:200]:
            self._model.learn(x)

    def score_one(self, x):
        return self._model.score_one(x)

    def update_one(self, x, label=None):
        self._model.learn(x)

    def decision_function(self, X):
        return np.array([self.score_one(x) for x in X])

    def predict(self, X):
        d = self.decision_function(X)
        return np.where(d > self._model.threshold, -1, 1)
```

**IForestASD: Fix decision_function bug + adaptive threshold**

```python
class IForestASD:
    name = 'IForestASD'
    supports_streaming = True

    def __init__(self, seed=42):
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self.window_size = 2000  # was 500
        self.n_trees = 100      # was 50
        self.max_samples = 512   # was 256
        self.trees = []
        self.buffer = []
        self._threshold = None

    def _partial_fit(self, x):
        # ... same tree building logic ...
        pass

    def decision_function(self, X):
        scores = np.zeros(len(X))
        for idx, x in enumerate(X):  # FIXED: direct indexing
            if not self.trees:
                scores[idx] = 0.5
                continue
            xf = x.astype(np.float32)
            depth_sum = sum(0.0 if xf[fi] < sp else 1.0
                            for fi, sp in self.trees)
            scores[idx] = depth_sum / len(self.trees)
        return scores

    def predict(self, X):
        d = self.decision_function(X)
        # Adaptive threshold from training scores, NOT from test
        if self._threshold is None:
            self._threshold = 0.5  # default
        return np.where(d > self._threshold, -1, 1)
```

---

## 5. Implementation Phases

---

### Phase 1: Protocol Fixes + NAB Integration + Pareto Frontier
**Estimated time: 4-6 hours**

| Task | Status | Details |
|------|--------|---------|
| Three-way train/val/test split | TODO | Train 10K, Val 2K (from train month), Test 10K |
| Remove test-set threshold search | TODO | Threshold from val, apply to test |
| Redesign anomaly injection | TODO | 1500 samples/type, 15% rate, one primary type per tier |
| Train = 100% normal (NO injection) | TODO | Unsupervised methods must train on normal data only |
| Implement Friedman + Holm-Bonferroni + Wilcoxon | TODO | Holm (NOT Nemenyi) — FWER control |
| Implement oracle query simulation | TODO | Pre-generated query schedule, y_test never inside loop |
| Add bootstrap CIs | TODO | 1000 bootstrap samples per algorithm |
| Implement BAR Score | TODO | Budget-aware ranking for streaming |
| Fix seed count | TODO | 3 → 5 seeds for stability |
| Add NAB dataset (Graceful Degradation Mode) | TODO | 3 streams, reduced-context on NAB (temporal-only, no taxi features) |
| Label budget sweep | TODO | Run streaming at 0%, 1%, 5%, 10% per algorithm |
| Anomaly rate ablation | TODO | Run at 5%, 10%, 15%, 20% for robustness check |
| Implement Pareto Frontier chart | TODO | matplotlib: AUC-PR vs Label Budget per algorithm |

**Deliverable:** `benchmark_v6_protocol.py` — same algorithms as v5, but with correct protocol, full statistical pipeline, and NAB integration. This run establishes whether the *protocol* fixes alone change the ranking.

### Phase 2: Algorithm Fixes
**Estimated time: 4-6 hours**

| Task | Status | Details |
|------|--------|---------|
| Implement CA-DIF-EIA v6 (batch) | TODO | Trained AE + context weights + ablation study |
| Implement CA-DIF-EIA streaming v6 | TODO | ADWIN-U + label budget + context |
| Fix LSTM-AE | TODO | Option B: trained DAE |
| Fix sHST-River | TODO | river.anomaly.HalfSpaceTrees |
| Fix IForestASD | TODO | decision_function bug + adaptive threshold |
| Verify MemStream | TODO | Ensure it actually updates memory |

### Phase 3: Full Benchmark Run
**Estimated time: 15-20 minutes (sequential)**

```
9 algorithms × 5 seeds × 5 folds × 3 difficulties = 675 jobs
+ 3 ablation configs × 5 seeds × 5 folds × 3 difficulties = 225 jobs
Total: ~900 jobs
```

### Phase 4: Analysis & Reporting
**Estimated time: 1-2 hours**

| Task | Status | Details |
|------|--------|---------|
| CD diagrams for all 3 difficulty tiers | TODO | One figure per tier |
| Friedman p-values per tier | TODO | Verify significant differences exist |
| Per-algorithm 95% CIs | TODO | Bootstrap-based |
| BAR Score for streaming algorithms | TODO | Budget-aware ranking |
| Final ranking table | TODO | All algorithms, all tiers |

---

## 6. Expected Outcomes (v6) — Revised After Sci. Review

> **Critical note:** v6 results will likely be WORSE than v5 numerically, but MORE RELIABLE scientifically. Any paper must acknowledge this explicitly: "We corrected a methodological flaw in the evaluation protocol that caused optimistic bias in v5 results. The corrected evaluation produces lower point estimates but with valid statistical inference."

**Honest predictions (not promises):**

| Model | v5 AUC-PR | v6 Expected (honest) | Why it changes |
|-------|-----------|----------------------|----------------|
| sklearn_OCSVM | 0.229 (inflated) | 0.12-0.18 | Threshold tuned on val, not test |
| sklearn_LOF | 0.114 | 0.08-0.13 | Same reason |
| MemStream | 0.151 | 0.10-0.16 | Same reason |
| LSTM-AE (DAE) | 0.125 | 0.08-0.14 | Honest eval, may drop more |
| CA-DIF-EIA | 0.067 | 0.08-0.20 | Depends on whether trained AE helps |
| sklearn_IF | 0.033 | 0.03-0.06 | Baseline, minor drop |
| CA-DIF-EIA (stream) | 0.068 | 0.08-0.15 | Depends on ADWIN-U quality |
| sHST-River | 0.023 | 0.05-0.12 | river library should help |
| IForestASD | 0.011 | 0.04-0.10 | Bug fix helps significantly |

**What matters for publication is NOT the absolute AUC-PR numbers, but:**
1. The CA-DIF-EIA full variant beats sklearn_IF with statistical significance (Wilcoxon p < 0.05)
2. CA-DIF-EIA streaming outperforms all other streaming methods at low label budgets
3. The streaming protocol is well-motivated and reproducible

**What to say in the paper if CA-DIF-EIA does NOT beat sklearn_IF:**
- Do NOT manipulate the protocol to force a win
- Report the null result honestly: "Under the corrected evaluation protocol, CA-DIF-EIA (AE+IF+context) does not significantly outperform sklearn_IF"
- The contribution becomes: (a) corrected benchmark protocol + (b) negative result on CA-DIF-EIA components

---

## 7. What Changes From v5 to v6

| What changed | Why |
|---|---|
| Three-way split (train/val/test) | Threshold must not touch test labels |
| Anomalies in train AND test | Realistic: fraud exists in historical data too |
| One injection type per difficulty | 1500 samples gives stable AUC estimates |
| 5 seeds (was 3) | More stable rank estimates (25 obs per algo) |
| Statistical pipeline: Friedman + Holm-Bonferroni + Wilcoxon | Mandatory for statistical credibility |
| Bootstrap 95% CIs | Shows uncertainty in estimates |
| CA-DIF-EIA with trained AE | "Deep" must actually learn (not random projection) |
| Streaming evaluation loop | One record at a time, with natural update |
| river.anomaly.HST | Use well-tested library, not custom code |
| IForestASD decision_function fixed | Bug was returning zeros for duplicate records |
| BAR Score | Labels are scarce; efficiency matters |
| Label budget sweep (0/1/5%) | Shows how each method uses labels |

---

## 8. What Does NOT Change

| What stays the same | Why |
|---|---|
| Data source (NYC taxi Jan-Jun 2024) | Consistent comparison with v5 |
| Feature engineering (25 features) | Already well-designed |
| Fold structure (sliding window) | Valid temporal evaluation |
| 6 months, 10K train/test per fold | Sufficient for evaluation |
| sklearn_IF, sklearn_LOF, sklearn_OCSVM | These are the baselines to beat |

---

## 9. Pre-Registered Hypotheses

> **These hypotheses are stated BEFORE running the benchmark. They are saved with a timestamp in this document to prevent post-hoc hypothesis formation. Pre-registration timestamp: 2026-05-12.**

**Hypothesis 1 (Main — Algorithm):**
> H0: CA-DIF-EIA (full) performs equally to sklearn_IF on AUC-PR
> H1: CA-DIF-EIA (full) significantly outperforms sklearn_IF on AUC-PR
> Test: Wilcoxon signed-rank, one-sided, alpha=0.05, Holm-corrected

**Hypothesis 2 (Ablation — Autoencoder):**
> H0: AE+IF (without context weighting) performs equally to sklearn_IF on AUC-PR
> H1: AE+IF significantly outperforms sklearn_IF on AUC-PR
> Test: Wilcoxon signed-rank, one-sided, alpha=0.05, Holm-corrected
> Interpretation: If H0 cannot be rejected for H2, the autoencoder component provides no value

**Hypothesis 3 (Ablation — Context Weighting):**
> H0: CA-DIF-EIA (full) performs equally to AE+IF on AUC-PR
> H1: CA-DIF-EIA (full) significantly outperforms AE+IF on AUC-PR
> Test: Wilcoxon signed-rank, one-sided, alpha=0.05, Holm-corrected
> Interpretation: If H0 cannot be rejected for H3, the context weighting component provides no value

**Hypothesis 4 (Streaming — Label Efficiency):**
> H0: CA-DIF-EIA (streaming) has equal BAR score to MemStream at 1% label budget
> H1: CA-DIF-EIA (streaming) has significantly higher BAR score than MemStream at 1% label budget
> Test: Wilcoxon signed-rank, one-sided, alpha=0.05
> Interpretation: If H1 is supported, CA-DIF-EIA uses labels more efficiently

**Hypothesis 5 (Generalization — NAB):**
> H0: CA-DIF-EIA (full) performs equally on NYC Taxi and NAB datasets
> H1: CA-DIF-EIA (full) performance differs significantly between datasets
> Test: Mann-Whitney U, two-sided, alpha=0.05
> Interpretation: If H1 is strongly supported, generalization is limited; must be disclosed as limitation

> **Commitment:** We will report ALL results, including null findings. We will NOT selectively report only the hypotheses where H0 was rejected. If H0 cannot be rejected for H1-H4, the paper will acknowledge that CA-DIF-EIA does not significantly outperform baselines and discuss why.

---

## 10. Notes

### 10.1 On CA-DIF-EIA as a Contribution

**This is the most important question for publication.** The reviewer will ask: "Is CA-DIF-EIA a new method, or just sklearn_IF with extra steps?"

The answer must be one of:

**Option A — New contribution (requires strongest justification):**
- CA-DIF-EIA must be clearly defined as a novel combination of existing techniques
- The paper must show that this combination is NOT obvious (e.g., not in any existing paper)
- The ablation study proves each component contributes independently
- This is the hardest path but the strongest for publication

**Option B — Engineering contribution:**
- CA-DIF-EIA is a practical combination of known methods (IF + AE + context weighting)
- The contribution is the benchmark framework + streaming protocol, not CA-DIF-EIA itself
- This is acceptable for conference papers

**Decision required:** What is your claim? If Option A, the ablation study results are critical. If Option B, the streaming protocol and benchmark design are the contributions.

### 10.2 On Comparing v6 to v5

**Do NOT compare v6 numbers with v5 numbers directly.** The evaluation protocols are different:
- v5: threshold tuned on test set (invalid)
- v6: threshold tuned on val set (valid)

Any "improvement" claim must be qualified with "under the corrected evaluation protocol." The v5 numbers should be cited only to acknowledge the methodological flaw that was fixed.

### 10.3 On Single-Dataset Limitation

The current plan evaluates on only NYC taxi data. This is a limitation. If possible, add at least one more dataset:
- San Francisco taxi data (SFMTA)
- Synthetic benchmark datasets (e.g., ODDS benchmark)
- Real fraud datasets if available

If only NYC taxi is used, this must be acknowledged as a limitation in the paper: "Results may not generalize to other geographic contexts or fraud patterns."

### 10.4 On Reproducibility

**All code, data, and random seeds must be publicly available.** Reviewers will check:
- [ ] Code is on GitHub with MIT/GPL license
- [ ] Random seeds are fixed (42, 123, 456, 789, 1000)
- [ ] Data sources are cited (NYC TLC data)
- [ ] Version numbers of all dependencies (scikit-learn, river, PyTorch, etc.)
- [ ] Hardware specification (GPU model, RAM) for reproducibility

### 10.5 On Negative Results and Ablation Pivot (Upgrade #4)

**If CA-DIF-EIA does not beat sklearn_IF:** This is a valid and publishable result. Negative results prevent other researchers from following the same dead end. The paper should be written as:

> "We designed and rigorously evaluated CA-DIF-EIA, a context-aware deep isolation forest. Under a corrected evaluation protocol with statistical significance testing, we find that CA-DIF-EIA does not significantly outperform sklearn_IF. This finding suggests that (a) the deep projection component and (b) the context weighting component of CA-DIF-EIA do not provide additional discriminative power beyond standard IF for taxi fraud detection."

**If the autoencoder (Hypothesis 2) does NOT help:**
> "Our ablation study reveals that the deep autoencoder projection does not significantly improve AUC-PR over sklearn_IF (H2: p > 0.05). This debunks a common assumption: Deep Learning is not automatically superior for streaming anomaly detection. The core contribution pivots to: (1) the spatial-temporal context partitioning architecture, and (2) the unsupervised drift detector (ADWIN-U) enabling label-efficient streaming."

**If the context weighting (Hypothesis 3) does NOT help:**
> "The context-aware weighting component does not significantly improve over AE+IF (H3: p > 0.05). The performance gain, if any, comes from the autoencoder projection alone. In this case, the contribution reframes as: a streaming protocol + deep projection, without context weighting."

**Upgrade #4 commitment:** Regardless of which component wins, ALL ablation results will be reported. The paper will acknowledge which components contribute and which do not. This is a "debunking paper" if H2 fails — and debunking papers are highly valued at Q1 venues because they prevent future researchers from following the same incorrect assumption.

### 10.6 On Feature Parity (Upgrade #1 — Clarification)

**Common reviewer attack:** "CA-DIF-EIA wins only because it uses feature engineering that baselines do not receive."

**Our response:** All algorithms receive the same feature vector. This is verified by a single `features()` function that produces a fixed-dimension array fed to every model.

**Feature vector (25D — verified in v5 code, `features()` function):**

| Dim | Feature | Type | Source |
|-----|---------|------|--------|
| 0 | trip_distance | Raw | Raw field |
| 1 | dur_min | Raw | Raw field |
| 2 | fare_amount | Raw | Raw field |
| 3 | passenger_count | Raw | Raw field |
| 4 | total_amount | Raw | Raw field |
| 5 | speed_mph | Raw | Derived field |
| 6 | fare_per_mile | Derived | fare / max(dist, eps) |
| 7 | fare_per_min | Derived | fare / max(dur, eps) |
| 8 | fare_per_pax | Derived | fare / max(pax, eps) |
| 9 | hour | Raw | Hour of day (0-23) |
| 10 | day_of_week | Raw | Day of week (0-6) |
| 11 | is_weekend | Derived | dow >= 5 |
| 12 | is_rush_hour | Derived | (7-10am) OR (4-8pm) |
| 13 | is_night | Derived | hour >= 20 OR hour <= 6 |
| 14 | month | Raw | Month (1-6) |
| 15 | normalized_fare_per_mile | Derived | fare_per_mile / 2.5 |
| 16 | normalized_fare_per_min | Derived | fare_per_min / 0.67 |
| 17 | normalized_speed | Derived | speed / 12.0 |
| 18 | pax_per_mile | Derived | pax / max(dist, eps) |
| 19 | fare_times_dist | Derived | fare * dist |
| 20 | dur_per_dist | Derived | dur / max(dist, eps) |
| 21 | hour_sin | Cyclical | sin(2pi * hour / 24) |
| 22 | hour_cos | Cyclical | cos(2pi * hour / 24) |
| 23 | dow_sin | Cyclical | sin(2pi * dow / 7) |
| 24 | dow_cos | Cyclical | cos(2pi * dow / 7) |

**Feature parity enforcement in code:**
```python
X = features(df)  # Single source for all algorithms
# fed to sklearn_IF, MemStream, sHST-River, LSTM-AE, CA-DIF-EIA
```

**Architecture-specific advantage (legitimate):**
CA-DIF-EIA's "spatial-temporal" advantage comes from **context-aware partitioning by (hour_bin, day_bin)**, not from geographic coordinates. The 24 temporal bins (hour) and 7 daily bins provide context segmentation that baselines cannot exploit because their architectures lack this routing mechanism. Baselines treat all 25 dimensions equally. CA-DIF-EIA learns per-context isolation paths. This is an architectural difference, not a feature difference.

**Explicit statement in paper:**
> "All algorithms receive identical 25-dimensional feature vectors derived from the same raw taxi fields. CA-DIF-EIA additionally partitions the feature space by temporal context (hour/day bins), enabling context-specific isolation paths. Baselines do not perform context partitioning because their architectures are not designed for this."

**If reviewer insists on equal architecture:**
> "We also report CA-DIF-EIA without context partitioning (equivalent to AE+IF ablation). Results [show/confirm] that context partitioning [does/does not] contribute significantly."

---

## 11. Checklist Before Phase 1 Implementation

> All items below have been resolved. This checklist is the signed commitment.

- [x] **Decision:** CA-DIF-EIA = Proposed Method (Algorithm + Protocol)
- [x] **Decision:** Primary metric = AUC-PR; AUC-ROC supplementary; F1 at contamination threshold
- [x] **Decision:** Label budget sweep = 0%, 1%, 5%, 10% (100% excluded)
- [x] **Decision:** Train data = 100% normal (no injection)
- [x] **Decision:** Contamination = validation threshold (never hardcoded to match test rate)
- [x] **Decision:** Target venue = Conference (KDD ADS / VLDB / SIGMOD Industrial)
- [x] **Decision:** Datasets = NYC Taxi (primary) + NAB (secondary, 3 streams)
- [x] **Pre-registered:** Hypothesis 1 (Main) — Wilcoxon one-sided, alpha=0.05
- [x] **Pre-registered:** Hypothesis 2 (Ablation AE)
- [x] **Pre-registered:** Hypothesis 3 (Ablation Context)
- [x] **Pre-registered:** Hypothesis 4 (BAR Score streaming)
- [x] **Pre-registered:** Hypothesis 5 (Generalization NAB)
- [x] **Acknowledged:** Limited fold count (5) affects statistical power
- [x] **Acknowledged:** Val drawn from train month (not test month) — limitation
- [x] **Acknowledged:** Single-domain primary dataset — limitation
- [x] **Acknowledged:** Oracle query simulation via pre-generated schedule (not y_test inside loop)
- [x] **Commitment:** All results (including null) will be reported faithfully

---

*Document Status: RESOLVED — Sci. Review Round 2 Complete*
*Pre-registration timestamp: 2026-05-12*
*Next step: Implement benchmark_v6_protocol.py per Phase 1 tasks*
