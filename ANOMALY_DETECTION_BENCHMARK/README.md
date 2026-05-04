# Anomaly Detection Benchmark - Complete Package

**NYC Taxi Trip Count Anomaly Detection - 22 Models Evaluated**

This package contains all code, data, models, results, and documentation for the anomaly detection benchmark.

---

## 📦 Package Contents

```
ANOMALY_DETECTION_BENCHMARK/
├── README.md                          # This file
├── QUICKSTART.md                      # Quick start guide
├── code/                              # Python scripts
│   ├── analyze_data_for_params.py     # Data-driven parameter selection
│   ├── exp_train_all_models.py        # Train all 20 models
│   ├── exp_score_all.py               # Score models on test data
│   ├── exp_evaluate_comprehensive.py  # Comprehensive evaluation
│   ├── exp_ml_autoencoder.py          # Deep learning models (AE, VAE)
│   ├── exp_ml_models_pyod.py          # PyOD models
│   └── exp_offline_train.py           # sklearn models (IForest, LOF, OCSVM)
├── data/                              # Training and test data
│   ├── train_2024.parquet             # 17,589 samples (clean training data)
│   └── test_2025.parquet              # 17,529 samples (391 anomalies, 2.23%)
├── models/                            # ALL 23 trained models (85MB)
│   ├── PyOD models: hbos_bins10.pkl, knn_k5.pkl, knn_k20.pkl,
│   │   mcd_auto.pkl, pca_n3.pkl, copod.pkl, abod_fast.pkl, ecod.pkl
│   ├── Deep Learning: autoencoder_dim8/16.keras, vae_dim8/16.keras
│   │   (with scalers)
│   ├── sklearn: iforest_n100/200.pkl, lof_k20/50.pkl, ocsvm_nu002.pkl
│   ├── RRCF: rrcf_t100/200_s256.pkl (trained but not scored)
│   └── ZSCORE_NOTE.txt                # Z-Score uses context stats (best: 0.9421)
├── results/                           # ALL predictions and benchmarks (9MB)
│   ├── comprehensive_comparison.csv   # All 22 models, all metrics
│   ├── per_type_pivot.csv             # Per-type AUC matrix
│   └── 22 prediction files:           # Full predictions for each model
│       abod_fast, autoencoder_dim8/16, copod, cusum, ecod, ewma,
│       hbos, iforest_n100/200, iqr, knn_k5/20, lof_k20/50, mad,
│       mcd_auto, ocsvm, pca_n3, vae_dim8/16, zscore
└── documentation/                     # Analysis and recommendations
    ├── TABLE_1_ANOMALY_DETECTION.md   # 20 anomaly detection models
    ├── TABLE_2_CONCEPT_DRIFT.md       # 2 concept drift models (EWMA, CUSUM)
    ├── MODEL_PURPOSES.md              # What each model is designed for
    ├── MODEL_SUMMARY_TABLE.md         # Quick reference
    └── ARCHITECTURE_RECOMMENDATIONS.md # Top 5 architecturally significant models
```

---

## 🎯 Benchmark Results Summary

### Top 5 Models

| Rank | Model | AUC | FP Rate | Category | Use For |
|------|-------|-----|---------|----------|---------|
| 1 | **Z-Score** | 0.9421 | 0.51% | Statistical | Production baseline |
| 2 | **IQR** | 0.9253 | 0.51% | Statistical | Robust detection |
| 3 | **HBOS** | 0.9102 | 1.55% | PyOD (ML) | Best ML model |
| 4 | **MCD** | 0.8999 | 0% | PyOD (ML) | Low false positives |
| 5 | **VAE** | 0.8968 | 0.72% | Deep Learning | Complex patterns |

### Key Findings

✅ **Statistical methods beat ML**: Z-Score (0.9421) > HBOS (0.9102) > VAE (0.8968)

✅ **Temporal context is key**: Simple rules with hour/day-of-week context work best

✅ **VAE > Autoencoder**: Probabilistic modeling helps (0.8968 vs 0.8070)

✅ **IsolationForest is production-ready**: 0.8591 AUC, 240k windows/s

❌ **Distance-based methods failed**: KNN (0.52), LOF (0.57), ABOD (0.54)

❌ **Concept drift models misused**: EWMA (0.80), CUSUM (0.47) - not for point anomalies

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn pyod tensorflow rrcf dill joblib
```

### 2. Run Data Analysis

```bash
python code/analyze_data_for_params.py \
    --train-file data/train_2024.parquet
```

**Output**: Data characteristics, parameter recommendations

### 3. Train Models (20 models)

```bash
python code/exp_train_all_models.py \
    --train-file data/train_2024.parquet \
    --output-dir trained_models/
```

**Output**: 20 trained models (~110 seconds total)

### 4. Score Models on Test Data

```bash
python code/exp_score_all.py \
    --test-file data/test_2025.parquet \
    --models-dir trained_models/ \
    --output-dir evaluation/
```

**Output**: 22 prediction files (20 ML + 2 statistical)

### 5. Evaluate Performance

```bash
python code/exp_evaluate_comprehensive.py \
    --predictions-dir evaluation/predictions/ \
    --output-dir evaluation/
```

**Output**:
- `comprehensive_comparison.csv` - All metrics
- `per_type_pivot.csv` - Per-type AUC

---

## 📊 Dataset Details

### Training Data: `train_2024.parquet`
- **Samples**: 17,589
- **Period**: 2024 (clean data, no injected anomalies)
- **Features**: 17
  - **Context stats** (7): ctx_mean, ctx_std, ctx_median, ctx_q25, ctx_q75, ctx_dev, ctx_abs_dev
  - **Lag features** (3): lag_48, lag_144, lag_336
  - **Rolling stats** (3): roll_mean_48, roll_std_48, roll_mean_336
  - **Temporal** (4): hour_sin, hour_cos, dow_sin, dow_cos
- **Metadata**: window_start, trip_count, hour, dow

### Test Data: `test_2025.parquet`
- **Samples**: 17,529
- **Anomalies**: 391 (2.23%)
- **Anomaly Types** (6):
  1. **Point Spike** (high trip count) - 80 samples
  2. **Point Drop** (low trip count) - 80 samples
  3. **Contextual** (unusual for context) - 65 samples
  4. **Collective** (group pattern change) - 60 samples
  5. **Trend Shift** (sustained change) - 56 samples
  6. **Variance** (high volatility) - 50 samples

---

## 🏆 Model Categories

### 1. Anomaly Detection (20 models) ✅

Detect individual unusual data points:

**Statistical (3)**: Z-Score, IQR, MAD
- ✅ Best performers: Z-Score (0.9421), IQR (0.9253)

**PyOD (8)**: HBOS, KNN×2, MCD, PCA, COPOD, ABOD, ECOD
- ✅ Best: HBOS (0.9102), MCD (0.8999)
- ❌ Failed: KNN (0.52), ABOD (0.54)

**Deep Learning (4)**: Autoencoder×2, VAE×2
- ✅ Best: VAE dim=16 (0.8968)

**sklearn ML (5)**: IsolationForest×2, LOF×2, OneClassSVM
- ✅ Best: IForest (0.8591), OneClassSVM (0.8856)
- ❌ Failed: LOF (0.57)

### 2. Concept Drift Detection (2 models) ❌

Detect distribution changes (NOT for point anomalies):

**Window-based (2)**: EWMA, CUSUM
- ⚠️ EWMA: 0.7954 AUC (moderate - can catch some anomalies)
- ❌ CUSUM: 0.4727 AUC (terrible - designed for drift, not spikes)

**These were incorrectly used for anomaly detection!**

---

## 🔬 Architecturally Significant Models

For research/production, focus on **5 representative models**:

### 1. Z-Score (Statistical Baseline)
- **AUC**: 0.9421 ⭐ BEST OVERALL
- **Speed**: Ultra fast
- **Why**: Temporal context + simple rules
- **File**: Computed from context stats (no saved model)

### 2. HBOS (Classical ML)
- **AUC**: 0.9102 ⭐ BEST ML
- **Speed**: Fast (11k/s)
- **Why**: Histogram-based, handles non-Gaussian
- **File**: `models/hbos_bins10.pkl` (280KB)

### 3. VAE (Deep Learning)
- **AUC**: 0.8968 ⭐ BEST DL
- **Speed**: Moderate (34k/s)
- **Why**: Probabilistic beats deterministic
- **Files**: `models/vae_dim16.keras` (116KB) + scaler

### 4. IsolationForest (Production)
- **AUC**: 0.8591
- **Speed**: Very fast (240k/s)
- **Why**: Scalable, no assumptions
- **File**: `models/iforest_n100.pkl` (1.5MB)

### 5. MCD (Robust Statistics)
- **AUC**: 0.8999
- **Speed**: Ultra fast
- **Why**: 0% FP rate, low false alarms
- **File**: `models/mcd_auto.pkl` (456KB)

---

## 📈 Per-Type Performance

### Best Models by Anomaly Type

| Anomaly Type | Best Model | AUC | 2nd Best | AUC |
|--------------|------------|-----|----------|-----|
| **Point Spike** | IQR | 0.9976 | Z-Score | 0.9913 |
| **Collective** | IQR | 0.9800 | Z-Score | 0.9784 |
| **Contextual** | IQR | 0.9834 | Z-Score | 0.9711 |
| **Point Drop** | Z-Score | 0.9197 | HBOS | 0.8878 |
| **Trend Shift** | Z-Score | 0.8896 | MAD | 0.8234 |
| **Variance** | HBOS | 0.9085 | Z-Score | 0.9020 |

---

## 💡 Deployment Recommendations

### For Production Systems
```
1. Z-Score (0.9421) - Fast, interpretable baseline
2. IsolationForest (0.8591) - Scalable, robust
3. HBOS (0.9102) - Best ML accuracy
```

### For Low False Positives
```
1. MCD (0.8999, 0% FP) - Very conservative
2. Z-Score (0.9421, 0.51% FP)
3. IQR (0.9253, 0.51% FP)
```

### For Research/Benchmarking
```
1. VAE (0.8968) - Deep learning baseline
2. OneClassSVM (0.8856) - Kernel methods
3. COPOD (0.8186) - Parameter-free
```

### Tiered System (Best of All)
```
Tier 1: Z-Score (fast filter) → 98% normal
Tier 2: HBOS (ML verification) → 0.5% anomalies
Tier 3: VAE (deep analysis) → 0.2% confirmed
```

---

## 📝 Key Scripts

### `analyze_data_for_params.py`
Analyzes training data to select data-driven parameters:
- Distribution analysis (normality, skewness)
- PCA variance (n_components selection)
- Temporal correlation (lag features validation)
- Density analysis (KNN k selection)

**Key Findings**:
- Data is NON-Gaussian (p < 0.001)
- PCA: 3 components explain 95% variance
- Strong daily correlation (0.88) → RRCF tree_size=256
- High feature correlation → MCD may help

### `exp_train_all_models.py`
Trains all 20 models with data-driven parameters:
- PyOD: HBOS, KNN, MCD, PCA, COPOD, ABOD, ECOD
- Deep Learning: Autoencoder, VAE (dim=8, 16)
- sklearn: IsolationForest, LOF, OneClassSVM
- RRCF: 100/200 trees

**Training time**: ~110 seconds (1.8 minutes)

### `exp_score_all.py`
Scores all 22 models on test data:
- Loads trained models
- Computes anomaly scores
- Saves predictions per model
- Reports throughput (windows/second)

**Note**: RRCF skipped (too slow - 3 hours for 17k samples)

### `exp_evaluate_comprehensive.py`
Comprehensive evaluation:
- Overall AUC (ROC-AUC)
- False positive rate (98th percentile)
- Per-type AUC (6 anomaly types)
- Ranking and comparison

**Outputs**: `comprehensive_comparison.csv`, `per_type_pivot.csv`

---

## 🔍 Data Characteristics

### Non-Gaussian Distribution
- Shapiro-Wilk test: p = 0.000000 (reject normality)
- Skewness: -0.13 (slightly left-skewed)
- Kurtosis: 0.44 (heavy-tailed)

**Implication**: MCD may underperform (assumes Gaussian), COPOD/ECOD should excel

### Strong Temporal Patterns
- Lag-48 correlation: 0.88 (strong daily pattern)
- Lag-336 correlation: 0.71 (weekly pattern)
- Hourly seasonality: Clear morning/evening peaks

**Implication**: Temporal context is the key signal

### Feature Correlation
- 17 highly correlated pairs (|r| > 0.8)
- Context features are intercorrelated

**Implication**: Feature independence assumption (HBOS) may be violated but still works well

---

## 📚 Documentation

### `TABLE_1_ANOMALY_DETECTION.md`
- 20 anomaly detection models
- Performance rankings
- Category breakdown (Statistical, PyOD, DL, sklearn)
- Per-type champions
- Deployment guide

### `TABLE_2_CONCEPT_DRIFT.md`
- 2 concept drift models (EWMA, CUSUM)
- Why they failed on anomaly detection
- Correct usage examples
- Anomaly vs drift comparison

### `MODEL_PURPOSES.md`
- Detailed explanation of each model
- Scientific basis and references
- Use cases and best practices
- When to use each model

### `MODEL_SUMMARY_TABLE.md`
- Quick reference table
- All 22 models at a glance
- Detection type classification

### `ARCHITECTURE_RECOMMENDATIONS.md`
- Top 5 architecturally significant models
- Why each architecture matters
- Deployment architecture
- Research questions answered

---

## ⚠️ Common Mistakes

### ❌ Using EWMA/CUSUM for Anomaly Detection
**Wrong**:
```python
# Point anomaly detection with CUSUM
cusum_score = compute_cusum(trip_count)
auc = roc_auc_score(is_anomaly, cusum_score)  # 0.47 AUC!
```

**Right**:
```python
# Concept drift detection on daily averages
daily_avg = df.groupby('date')['trip_count'].mean()
cusum = compute_cusum(daily_avg, target=100, drift=0.5)
```

### ❌ Ignoring Temporal Context
**Wrong**:
```python
# Global threshold
threshold = df['trip_count'].mean() + 3*df['trip_count'].std()
```

**Right**:
```python
# Context-aware threshold (hour, day-of-week)
for (dow, hour), group in df.groupby(['dow', 'hour']):
    threshold = group['trip_count'].mean() + 3*group['trip_count'].std()
```

### ❌ Using contamination=0.022 (test anomaly rate)
**Wrong**:
```python
model = IsolationForest(contamination=0.022)  # Test rate!
```

**Right**:
```python
model = IsolationForest(contamination=0.01)  # Training outliers
```

---

## 🎓 Research Questions Answered

### Q1: Do complex models beat simple baselines?
**Answer**: No. Z-Score (0.9421) > HBOS (0.9102) > VAE (0.8968)

### Q2: Is deep learning worth it for time series?
**Answer**: Depends on data. For structured temporal patterns (our case): No.

### Q3: What's the best production model?
**Answer**: Z-Score for baseline + IsolationForest for robustness

### Q4: Which models generalize best?
**Answer**: IsolationForest (no distribution assumptions)

### Q5: What's the minimum viable architecture?
**Answer**: Z-Score with temporal context (5 lines, 0.9421 AUC)

---

## 📦 Package Size

```
Total: 94MB (complete package with ALL models and predictions)
- Code: ~100KB (7 Python scripts)
- Data: ~20MB (symlinks to train/test parquet files)
- Models: ~85MB (ALL 23 trained models)
- Results: ~9MB (22 prediction files + 2 CSV benchmarks)
- Documentation: ~50KB (5 markdown files)
```

**This is the COMPLETE package** - all code, data, models, and results included.

---

## 🔗 Related Work

### Papers
- Goldstein & Dengel (2012): HBOS
- Kingma & Welling (2014): VAE
- Liu et al. (2008): IsolationForest
- Rousseeuw & Driessen (1999): MCD

### Frameworks
- PyOD: https://github.com/yzhao062/pyod
- scikit-learn: https://scikit-learn.org
- TensorFlow: https://tensorflow.org

---

## 📧 Contact

For questions or issues, refer to the documentation files in `documentation/`.

---

## 🏁 Quick Commands

```bash
# Analyze data
python code/analyze_data_for_params.py --train-file data/train_2024.parquet

# Train all models
python code/exp_train_all_models.py \
    --train-file data/train_2024.parquet \
    --output-dir models_output/

# Score models
python code/exp_score_all.py \
    --test-file data/test_2025.parquet \
    --models-dir models_output/ \
    --output-dir results_output/

# Evaluate
python code/exp_evaluate_comprehensive.py \
    --predictions-dir results_output/predictions/ \
    --output-dir results_output/

# View results
cat results_output/comprehensive_comparison.csv
```

---

**Last Updated**: 2026-05-04
**Package Version**: 1.0
**Total Models**: 22 (20 anomaly + 2 drift)
