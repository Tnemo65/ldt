# Table 1: Anomaly Detection Models

**Purpose**: Detect individual unusual data points that deviate from normal patterns

**Our Task**: ✅ Correctly used for NYC taxi anomaly detection

---

## Results Summary (20 Models)

| Rank | Model | AUC | FP Rate | Category | Speed | Best For |
|------|-------|-----|---------|----------|-------|----------|
| 1 | **Z-Score** | 0.9421 | 0.51% | Statistical | ⚡ Ultra Fast | Point spikes/drops, trend shifts |
| 2 | **IQR** | 0.9253 | 0.51% | Statistical | ⚡ Ultra Fast | Collective, contextual, point spikes |
| 3 | **HBOS** | 0.9102 | 1.55% | PyOD | Fast (11k/s) | Variance anomalies |
| 4 | **MCD** | 0.8999 | 0% | PyOD | ⚡ Ultra Fast (2M/s) | Global outliers (Gaussian data) |
| 5 | **VAE (dim=16)** | 0.8968 | 0.72% | Deep Learning | Moderate (34k/s) | Complex non-linear patterns |
| 6 | **VAE (dim=8)** | 0.8965 | 0.72% | Deep Learning | Moderate (32k/s) | Complex patterns |
| 7 | **PCA (n=3)** | 0.8924 | 0.72% | PyOD | ⚡ Ultra Fast (3M/s) | Linear high-dim data |
| 8 | **OneClassSVM** | 0.8856 | 0.90% | sklearn | Fast (66k/s) | Non-linear boundaries |
| 9 | **MAD** | 0.8820 | 0.75% | Statistical | ⚡ Ultra Fast | Robust to outliers |
| 10 | **IForest (n=100)** | 0.8591 | 1.81% | sklearn | ⚡ Very Fast (240k/s) | Production systems |
| 11 | **IForest (n=200)** | 0.8462 | 1.85% | sklearn | Fast (146k/s) | Scalable detection |
| 12 | **COPOD** | 0.8186 | 1.66% | PyOD | Fast (47k/s) | Non-Gaussian data |
| 13 | **Autoencoder (16)** | 0.8070 | 1.66% | Deep Learning | Fast (43k/s) | Pattern reconstruction |
| 14 | **Autoencoder (8)** | 0.8043 | 1.74% | Deep Learning | Moderate (30k/s) | Aggressive compression |
| 15 | **ECOD** | 0.7807 | 1.83% | PyOD | Fast (165k/s) | Distribution-agnostic |
| 16 | **LOF (k=20)** | 0.5721 ❌ | 1.98% | sklearn | Fast (55k/s) | ⚠️ Failed on this data |
| 17 | **LOF (k=50)** | 0.5669 ❌ | 1.97% | sklearn | Fast (78k/s) | ⚠️ Failed on this data |
| 18 | **ABOD (fast)** | 0.5381 ❌ | 0% | PyOD | Slow (7k/s) | ⚠️ Failed on this data |
| 19 | **KNN (k=5)** | 0.5195 ❌ | 0% | PyOD | Slow (7k/s) | ⚠️ Failed on this data |
| 20 | **KNN (k=20)** | 0.5156 ❌ | 0% | PyOD | Slow (6k/s) | ⚠️ Failed on this data |

---

## Category Breakdown

### 🏆 Statistical Models (3/3 excellent)
Best overall! Simple rules capture temporal patterns well.

| Model | AUC | How It Works | When to Use |
|-------|-----|--------------|-------------|
| **Z-Score** ⭐ | 0.9421 | Measures std deviations from mean | Fast, interpretable, production |
| **IQR** ⭐ | 0.9253 | Outliers beyond Q1-1.5×IQR to Q3+1.5×IQR | Robust to training outliers |
| **MAD** | 0.8820 | Median absolute deviation (robust) | Non-Gaussian distributions |

### 🤖 PyOD Models (3/8 excellent, 5/8 failed)

**Excellent:**
| Model | AUC | Method | Best For |
|-------|-----|--------|----------|
| **HBOS** ⭐ | 0.9102 | Histogram-based, assumes feature independence | High-dim data, fast scoring |
| **MCD** | 0.8999 | Robust Mahalanobis distance | Gaussian data with correlations |
| **PCA** | 0.8924 | Reconstruction error | Linear structure, dim reduction |

**Moderate:**
| Model | AUC | Method | Best For |
|-------|-----|--------|----------|
| COPOD | 0.8186 | Copula tail probabilities | Non-Gaussian, parameter-free |
| ECOD | 0.7807 | Empirical CDF tail probabilities | Any distribution, fast |

**Failed:**
| Model | AUC | Why It Failed |
|-------|-----|---------------|
| KNN (k=5, k=20) ❌ | 0.52 | Distance-based fails on high-dim temporal data |
| ABOD ❌ | 0.54 | Angle variance doesn't capture temporal patterns |

### 🧠 Deep Learning (2/4 excellent)

| Model | AUC | Architecture | Best For |
|-------|-----|--------------|----------|
| **VAE (dim=16)** ⭐ | 0.8968 | Probabilistic encoder-decoder + KL | Complex patterns, uncertainty |
| **VAE (dim=8)** ⭐ | 0.8965 | Probabilistic encoder-decoder | Aggressive compression |
| Autoencoder (16) | 0.8070 | Deterministic encoder-decoder | Pattern reconstruction |
| Autoencoder (8) | 0.8043 | Aggressive compression | Fast inference |

**Insight**: VAE beats Autoencoder because probabilistic modeling captures data uncertainty better.

### 🌲 sklearn ML (2/5 excellent, 3/5 failed)

**Excellent:**
| Model | AUC | Method | Best For |
|-------|-----|--------|----------|
| **OneClassSVM** | 0.8856 | RBF kernel boundary around normal | Non-linear patterns |
| **IsolationForest** | 0.8591 | Random partitioning isolation | Production, scalable |

**Failed:**
| Model | AUC | Why It Failed |
|-------|-----|---------------|
| LOF (k=20, k=50) ❌ | 0.57 | Local density fails on temporal data |

---

## Per-Type Performance Champions

| Anomaly Type | Best Model | AUC | 2nd Best | AUC |
|--------------|------------|-----|----------|-----|
| **Point Spike** | IQR | 0.9976 | Z-Score | 0.9913 |
| **Collective** | IQR | 0.9800 | Z-Score | 0.9784 |
| **Contextual** | IQR | 0.9834 | Z-Score | 0.9711 |
| **Point Drop** | Z-Score | 0.9197 | HBOS | 0.8878 |
| **Trend Shift** | Z-Score | 0.8896 | MAD | 0.8234 |
| **Variance** | HBOS | 0.9085 | Z-Score | 0.9020 |

---

## Deployment Recommendations

### 🚀 Production Systems (Need Fast + Accurate)
1. **Z-Score** (0.9421, Ultra Fast) - Best overall
2. **IsolationForest** (0.8591, 240k windows/s) - Scalable
3. **PCA** (0.8924, 3M windows/s) - Fastest ML model

### 🎯 Maximum Accuracy (Speed less critical)
1. **Z-Score** (0.9421)
2. **IQR** (0.9253)
3. **HBOS** (0.9102)

### 🧪 Research / Experimentation
1. **VAE** (0.8968) - Deep learning baseline
2. **OneClassSVM** (0.8856) - Non-linear kernel methods
3. **MCD** (0.8999) - Robust covariance

### ⚖️ Low False Positives Required
1. **MCD** (0% FP rate, 0.8999 AUC)
2. **Z-Score** (0.51% FP rate, 0.9421 AUC)
3. **IQR** (0.51% FP rate, 0.9253 AUC)

---

## Key Insights

### ✅ What Worked:
1. **Statistical models dominate** - Z-Score and IQR beat all ML models
2. **HBOS is the best ML model** (0.9102) - histogram approach works well
3. **VAE beats Autoencoder** - probabilistic modeling helps
4. **Simple beats complex** - Z-Score (0.9421) > VAE (0.8968)

### ❌ What Failed:
1. **Distance-based methods** (KNN, LOF, ABOD) - don't work on temporal data
2. **More trees ≠ better** - IForest n=100 (0.8591) > n=200 (0.8462)
3. **Larger k ≠ better** - KNN k=5 (0.52) ≈ k=20 (0.52)

### 🔍 Why Statistical Models Won:
NYC taxi data has **strong temporal patterns** (hourly/daily/weekly cycles):
- Z-Score/IQR leverage **context statistics** (same hour, same day-of-week)
- They capture "100 trips at 3am on Monday is normal, but 500 is anomalous"
- ML models see 18 features but miss the **temporal context** that simple rules encode

---

## Model Selection Guide

```
High Accuracy + Speed Needed?
├─ Yes → Z-Score (0.9421, Ultra Fast)
│
Low False Positives Critical?
├─ Yes → MCD (0% FP, 0.8999 AUC)
│
Need ML for Complex Patterns?
├─ Yes → HBOS (0.9102, 11k/s)
│
Production Deployment?
├─ Yes → IsolationForest (0.8591, 240k/s)
│
Research / Deep Learning?
└─ Yes → VAE dim=16 (0.8968, 34k/s)
```
