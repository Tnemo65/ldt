# Architecturally Significant Models for Anomaly Detection

## Recommended 5-Model Architecture Study

For research/production, focus on these **5 architecturally distinct approaches**:

---

## 1. Statistical: Z-Score (0.9421 AUC) ⭐ BASELINE

**Architecture:**
```python
z_score = |x - μ| / σ
```

**Why Significant:**
- ✅ **Interpretable**: "3 std deviations from mean"
- ✅ **Fast**: O(1) per point after computing μ, σ
- ✅ **Baseline**: All other models must beat this
- ✅ **Context-aware**: Uses temporal context (hour, day-of-week)

**Key Insight:**
Simple statistical rules with temporal context beat complex ML on structured time series.

**Use Case:**
- Production monitoring (latency, error rates)
- Real-time alerting
- Interpretable explanations to stakeholders

---

## 2. Classical ML: HBOS (0.9102 AUC) ⭐ BEST ML

**Architecture:**
```
For each feature f:
  1. Build histogram (n_bins)
  2. Compute density ρ(x) = count(bin) / total
  3. Score = -log(ρ(x))

Combine: total_score = Σ score_f (assumes feature independence)
```

**Why Significant:**
- ✅ **Parameter-free**: Only n_bins to tune
- ✅ **Handles non-Gaussian**: Works on any distribution
- ✅ **Fast**: O(d) where d=features
- ✅ **Feature independence assumption**: Works when features are uncorrelated

**Key Insight:**
Histogram-based density estimation scales well and doesn't assume Gaussian distribution.

**Use Case:**
- High-dimensional data (100+ features)
- Mixed distributions (some Gaussian, some not)
- Fast batch scoring

---

## 3. Deep Learning: VAE (0.8968 AUC) ⭐ BEST DL

**Architecture:**
```
Encoder:
  Input(17) → Dense(32, relu) → Dense(16, relu) → z_mean(8), z_log_var(8)

Sampling (reparameterization trick):
  z = z_mean + exp(z_log_var/2) × ε, where ε ~ N(0,1)

Decoder:
  z(8) → Dense(16, relu) → Dense(32, relu) → Output(17, linear)

Loss:
  reconstruction_loss + KL_divergence
  MSE(x, x_reconstructed) + KL(q(z|x) || p(z))

Anomaly Score:
  reconstruction_error = MSE(x, x_reconstructed)
```

**Why Significant:**
- ✅ **Probabilistic**: Models uncertainty in latent space
- ✅ **Non-linear**: Captures complex patterns
- ✅ **Generative**: Can sample from learned distribution
- ✅ **Beats Autoencoder**: KL regularization helps (0.8968 vs 0.8070)

**Key Insight:**
Probabilistic modeling (VAE) > Deterministic reconstruction (Autoencoder) because it:
- Prevents overfitting via KL regularization
- Learns smooth latent space
- Captures data distribution uncertainty

**Use Case:**
- Complex non-linear patterns
- Need generative model
- Research/benchmarking deep learning approaches

---

## 4. Tree-Based: IsolationForest (0.8591 AUC) ⭐ PRODUCTION

**Architecture:**
```
Training:
  For each tree t in [1..n_estimators]:
    1. Sample subset of data
    2. Build tree by random splits:
       - Pick random feature
       - Pick random split value
       - Recurse until max_depth or isolated
    
Scoring:
  For point x:
    avg_path_length = average depth to isolate x across all trees
    anomaly_score = 2^(-avg_path_length / c(n))
    
Key: Anomalies are easier to isolate (shorter path)
```

**Why Significant:**
- ✅ **Scalable**: O(n log n) training, O(log n) prediction
- ✅ **No assumptions**: Works on any distribution
- ✅ **Production-ready**: sklearn implementation, battle-tested
- ✅ **Fast**: 240k windows/s scoring

**Key Insight:**
Anomalies are "few and different" → easier to isolate via random partitioning.

**Use Case:**
- Production systems (fraud detection, security)
- Large-scale data (millions of points)
- Need fast, scalable solution

---

## 5. Robust Covariance: MCD (0.8999 AUC) ⭐ LOW FP

**Architecture:**
```
Training:
  1. Find subset S ⊂ data that minimizes covariance determinant
     min |Cov(S)| subject to |S| ≥ h (support fraction)
  
  2. Compute robust estimates:
     μ_robust = mean(S)
     Σ_robust = cov(S)

Scoring:
  Mahalanobis distance from robust center:
  MD(x) = √[(x - μ_robust)ᵀ Σ_robust⁻¹ (x - μ_robust)]
```

**Why Significant:**
- ✅ **Robust**: Resistant to outliers in training
- ✅ **0% FP rate**: Very conservative (good for critical systems)
- ✅ **Captures correlations**: Uses covariance matrix
- ✅ **Mathematically principled**: Based on robust statistics

**Key Insight:**
Robust covariance estimation handles contaminated training data well.

**Trade-off:**
- ⚠️ Assumes Gaussian distribution (our data is non-Gaussian, but still works!)
- Lower recall for some anomaly types

**Use Case:**
- Critical systems where false positives are expensive
- Data with correlated features
- Training data may contain outliers

---

## Comparison Matrix

| Model | Paradigm | Complexity | Speed | Interpretability | Best For |
|-------|----------|------------|-------|------------------|----------|
| **Z-Score** | Statistical | Low | ⚡⚡⚡ | ⭐⭐⭐ | Baseline, interpretable |
| **HBOS** | Density | Low | ⚡⚡ | ⭐⭐ | High-dim, non-Gaussian |
| **VAE** | Deep Learning | High | ⚡ | ⭐ | Complex patterns, research |
| **IForest** | Tree | Medium | ⚡⚡⚡ | ⭐⭐ | Production, scalable |
| **MCD** | Robust Stats | Medium | ⚡⚡⚡ | ⭐⭐ | Low FP, correlated features |

---

## Architecture Study Design

### Experiment 1: Baseline vs ML vs DL
Compare the 3 main paradigms:
- **Statistical**: Z-Score (0.9421)
- **Classical ML**: HBOS (0.9102)
- **Deep Learning**: VAE (0.8968)

**Question**: Does complexity help? (Answer: No, simple wins!)

---

### Experiment 2: Density vs Distance vs Trees
Compare detection approaches:
- **Density-based**: HBOS (histogram density)
- **Distance-based**: MCD (Mahalanobis distance)
- **Isolation-based**: IForest (path length)

**Question**: Which approach works best for temporal data?

---

### Experiment 3: Deterministic vs Probabilistic
Compare reconstruction models:
- **Deterministic**: Autoencoder (0.8070)
- **Probabilistic**: VAE (0.8968)

**Question**: Does probabilistic modeling help? (Answer: Yes, +8.9% AUC!)

---

### Experiment 4: Feature Independence Assumption
Compare models with/without correlation:
- **Assumes independence**: HBOS (0.9102)
- **Uses correlations**: MCD (0.8999)

**Question**: Does modeling correlations help? (Answer: Not much on this data)

---

### Experiment 5: Robustness to Contamination
Train with 1%, 5%, 10% contamination:
- **Robust**: MCD (0% FP), IForest
- **Non-robust**: Z-Score, HBOS

**Question**: Which models degrade gracefully?

---

## Deployment Architecture Recommendation

### Tiered Detection System

```
┌─────────────────────────────────────┐
│  Tier 1: Fast Screening (Z-Score)  │
│  - Filter 98% normal data           │
│  - Ultra fast (3M windows/s)        │
│  - Threshold: z > 3.0                │
└──────────┬──────────────────────────┘
           │ Suspicious (2%)
           ▼
┌─────────────────────────────────────┐
│  Tier 2: ML Verification (HBOS)    │
│  - Refine anomaly scores            │
│  - 11k windows/s                    │
│  - Threshold: score > p95           │
└──────────┬──────────────────────────┘
           │ Anomalies (0.5%)
           ▼
┌─────────────────────────────────────┐
│  Tier 3: DL Analysis (VAE)         │
│  - Deep pattern analysis            │
│  - 34k windows/s                    │
│  - Final confirmation               │
└──────────┬──────────────────────────┘
           │ Confirmed (0.2%)
           ▼
┌─────────────────────────────────────┐
│  Alert + Root Cause Analysis        │
└─────────────────────────────────────┘
```

**Benefits:**
- ✅ Fast: Most data filtered by Tier 1
- ✅ Accurate: Deep analysis on suspicious data
- ✅ Explainable: Z-Score provides interpretable threshold

---

## Code Artifacts to Preserve

### Training
```python
# 1. Statistical baseline
zscore_model = train_zscore(df_train)  # Context stats

# 2. Classical ML
hbos_model = train_hbos(df_train, n_bins=10)

# 3. Deep learning
vae_model, scaler = train_vae(df_train, latent_dim=16, epochs=50)

# 4. Tree-based
iforest_model = train_isolation_forest(df_train, n_estimators=100)

# 5. Robust stats
mcd_model = train_mcd(df_train, support_fraction=None)
```

### Evaluation
```python
# Per-type performance
for model in [zscore, hbos, vae, iforest, mcd]:
    for anom_type in ['point_spike', 'collective', ...]:
        auc = evaluate_per_type(model, df_test, anom_type)
```

---

## Research Questions Answered

### Q1: Do complex models beat simple baselines?
**Answer**: No. Z-Score (0.9421) > HBOS (0.9102) > VAE (0.8968)

**Why**: Temporal context (hour, day-of-week) is the key signal. Simple statistical rules with context beat complex models without explicit temporal modeling.

### Q2: Is deep learning worth it for time series anomalies?
**Answer**: Depends on data complexity.
- For **structured patterns** (our case): No. VAE (0.8968) < Z-Score (0.9421)
- For **unstructured patterns** (images, audio): Yes.

### Q3: What's the best production model?
**Answer**: **Tiered system**:
1. Z-Score for fast filtering (0.9421 AUC, ultra fast)
2. IsolationForest for batch scoring (0.8591 AUC, 240k/s)
3. VAE for complex pattern research (0.8968 AUC, GPU-ready)

### Q4: Which architecture generalizes best?
**Answer**: **IsolationForest** (0.8591)
- No distribution assumptions
- Robust to feature scale
- Works on any anomaly type

### Q5: What's the minimum viable architecture?
**Answer**: **Z-Score with temporal context**
- 5 lines of code
- 0.9421 AUC (best overall!)
- Interpretable
- Production-ready

---

## Conclusion

**For architecture significance, use these 5 models**:
1. **Z-Score** - Statistical baseline (must beat this!)
2. **HBOS** - Classical ML (density-based)
3. **VAE** - Deep learning (probabilistic)
4. **IsolationForest** - Tree-based (production)
5. **MCD** - Robust statistics (low FP)

**Key Architectural Insight**:
Simple temporal-aware statistical methods > Complex ML without temporal modeling

**For Production**:
Start with Z-Score, add IsolationForest for robustness.
