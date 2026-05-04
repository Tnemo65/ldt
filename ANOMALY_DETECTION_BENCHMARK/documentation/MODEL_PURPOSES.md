# Model Purposes: Anomaly Detection vs Concept Drift

## Summary

**Anomaly Detection** (20 models): Detect individual unusual data points that deviate from normal patterns
**Concept Drift Detection** (2 models): Detect changes in the underlying data distribution over time

---

## 1. Statistical Models (3) - ANOMALY DETECTION

### Z-Score
- **Purpose**: General anomaly detection
- **Method**: Measures how many standard deviations a point is from the mean
- **Best for**: Point anomalies (spikes, drops)
- **Use case**: Real-time monitoring, fast computation needed
- **Score**: 0.9421 AUC ⭐ BEST OVERALL

### IQR (Interquartile Range)
- **Purpose**: General anomaly detection
- **Method**: Detects outliers beyond Q1-1.5×IQR to Q3+1.5×IQR
- **Best for**: Collective, contextual, point_spike anomalies
- **Use case**: Robust to outliers in training data
- **Score**: 0.9253 AUC ⭐ 2nd BEST

### MAD (Median Absolute Deviation)
- **Purpose**: General anomaly detection
- **Method**: Robust alternative to standard deviation using median
- **Best for**: When data has extreme outliers
- **Use case**: Non-Gaussian distributions
- **Score**: 0.8820 AUC

---

## 2. PyOD Models (8) - ANOMALY DETECTION

### HBOS (Histogram-Based Outlier Score)
- **Purpose**: General anomaly detection
- **Method**: Builds histograms for each feature, scores based on bin density
- **Assumptions**: Features are independent
- **Best for**: High-dimensional data, variance anomalies
- **Use case**: Fast training/scoring needed
- **Score**: 0.9102 AUC ⭐ BEST ML MODEL

### KNN (k-Nearest Neighbors) - k=5, k=20
- **Purpose**: General anomaly detection
- **Method**: Distance to k-th nearest neighbor (large distance = outlier)
- **Best for**: Local density-based anomalies
- **Use case**: Small datasets, no assumptions about distribution
- **Score**: 0.520, 0.516 AUC ❌ POOR - Failed on this dataset

### MCD (Minimum Covariance Determinant)
- **Purpose**: General anomaly detection
- **Method**: Robust Mahalanobis distance from center of clean data
- **Assumptions**: Data follows multivariate Gaussian distribution
- **Best for**: Correlated features, global outliers
- **Use case**: When data is approximately Gaussian
- **Score**: 0.8999 AUC ⚠️ Warning: Our data is NON-Gaussian (p=0.000000)

### PCA (Principal Component Analysis)
- **Purpose**: General anomaly detection
- **Method**: Reconstruction error after dimensionality reduction
- **Best for**: High-dimensional data with linear structure
- **Use case**: Data lies on lower-dimensional manifold
- **Score**: 0.8924 AUC

### COPOD (Copula-Based Outlier Detection)
- **Purpose**: General anomaly detection
- **Method**: Tail probability using empirical copula (parameter-free)
- **Best for**: Non-Gaussian data, complex feature dependencies
- **Use case**: No hyperparameter tuning, any distribution
- **Score**: 0.8186 AUC

### ABOD (Angle-Based Outlier Detection)
- **Purpose**: General anomaly detection
- **Method**: Variance of angles between point and neighbor pairs
- **Best for**: High-dimensional data (distance-based methods fail)
- **Use case**: Detecting outliers in sparse regions
- **Score**: 0.5381 AUC ❌ POOR

### ECOD (Empirical Cumulative Distribution)
- **Purpose**: General anomaly detection
- **Method**: Tail probability from empirical CDF (parameter-free)
- **Best for**: Distribution-agnostic, fast scoring
- **Use case**: Any data distribution, interpretable scores
- **Score**: 0.7807 AUC

---

## 3. Deep Learning Models (4) - ANOMALY DETECTION

### Autoencoder - dim=8, dim=16
- **Purpose**: General anomaly detection
- **Method**: Reconstruction error (normal data reconstructs well, anomalies don't)
- **Best for**: Complex non-linear patterns
- **Use case**: High-dimensional data, pattern learning
- **Score**: 0.8043, 0.8070 AUC

### VAE (Variational Autoencoder) - dim=8, dim=16
- **Purpose**: General anomaly detection
- **Method**: Probabilistic reconstruction + KL divergence
- **Best for**: Capturing data distribution uncertainty
- **Use case**: Complex patterns with probabilistic modeling
- **Score**: 0.8965, 0.8968 AUC ⭐ BEST DEEP LEARNING

---

## 4. sklearn ML Models (5) - ANOMALY DETECTION

### IsolationForest - n=100, n=200
- **Purpose**: General anomaly detection
- **Method**: Isolates anomalies by random partitioning (fewer splits needed)
- **Best for**: Any distribution, fast, scalable
- **Use case**: Production systems, high-dimensional data
- **Score**: 0.8591, 0.8462 AUC

### LOF (Local Outlier Factor) - k=20, k=50
- **Purpose**: General anomaly detection
- **Method**: Local density deviation (compares density to neighbors)
- **Best for**: Clusters with varying densities
- **Use case**: Local anomalies in non-uniform data
- **Score**: 0.5721, 0.5669 AUC ❌ POOR

### OneClassSVM - nu=0.02
- **Purpose**: General anomaly detection
- **Method**: Learns decision boundary around normal data
- **Best for**: Non-linear decision boundaries
- **Use case**: When kernel trick is needed
- **Score**: 0.8856 AUC

---

## 5. RRCF (Robust Random Cut Forest) (2) - BOTH ANOMALY & DRIFT

### RRCF - 100 trees, 200 trees
- **Purpose**: Streaming anomaly detection + concept drift detection
- **Method**: CoDisp (Collusive Displacement) in random cut trees
- **Best for**: Time series, streaming data
- **Use case**: Online learning, detecting both anomalies and distribution shifts
- **Score**: SKIPPED (too slow - 3 hours for 17k samples)
- **Note**: Can detect BOTH point anomalies AND concept drift

---

## 6. Window-Based Models (2) - CONCEPT DRIFT DETECTION ⚠️

### EWMA (Exponentially Weighted Moving Average) - α=0.3
- **Purpose**: ⚠️ CONCEPT DRIFT / CHANGE POINT DETECTION
- **Method**: Detects sustained shifts from moving average
- **Best for**: Detecting gradual distribution changes
- **Use case**: Process control, detecting trends
- **Score**: 0.7954 AUC (moderate - not designed for point anomalies)
- **❌ WRONG USE**: We used it for anomaly detection, but it's for drift!

### CUSUM (Cumulative Sum)
- **Purpose**: ⚠️ CONCEPT DRIFT / CHANGE POINT DETECTION
- **Method**: Accumulates deviations to detect shifts in mean
- **Best for**: Detecting mean shifts in sequential data
- **Use case**: Quality control, change point detection
- **Score**: 0.4727 AUC ❌ TERRIBLE - Not designed for point anomalies!
- **❌ WRONG USE**: We used it for anomaly detection, but it's for drift!

---

## Key Insights

### ✅ Models that SHOULD be used for Anomaly Detection (20):
1. **Statistical** (3): Z-Score, IQR, MAD
2. **PyOD** (8): HBOS, KNN, MCD, PCA, COPOD, ABOD, ECOD
3. **Deep Learning** (4): Autoencoder, VAE
4. **sklearn ML** (5): IsolationForest, LOF, OneClassSVM
5. **RRCF** (2): Can do both anomaly + drift

### ❌ Models that should NOT be used for Anomaly Detection (2):
1. **EWMA**: Designed for concept drift detection
2. **CUSUM**: Designed for concept drift detection

### Why EWMA/CUSUM performed poorly:
- **CUSUM**: 0.4727 AUC (worse than random!) because it accumulates deviations over time
  - It's designed to detect **persistent shifts** (e.g., mean changes from 100 to 110)
  - It's NOT designed to detect **individual spikes** (e.g., one window with 200 trips)
- **EWMA**: 0.7954 AUC (moderate) because it tracks **trends**
  - It can detect some anomalies if they persist, but misses point anomalies
  - It's better for detecting **gradual drift** than **sudden outliers**

---

## Recommendations

### For Point Anomaly Detection (our use case):
1. **Best Overall**: Z-Score (0.9421) or IQR (0.9253) - Fast, simple, interpretable
2. **Best ML**: HBOS (0.9102) - Fast training/scoring
3. **Best Deep Learning**: VAE (0.8968) - Captures complex patterns
4. **Production Ready**: IsolationForest (0.8591) - Scalable, robust

### For Concept Drift Detection (NOT our current task):
1. **CUSUM**: Detects mean shifts
2. **EWMA**: Detects gradual trends
3. **RRCF**: Detects both anomalies and drift in streaming data
4. **Statistical Process Control (SPC)**: X-bar charts, CUSUM, EWMA

### For BOTH Anomaly + Drift:
1. **RRCF**: Designed for streaming data with both point anomalies and distribution shifts
2. **Hybrid Approach**: Run anomaly detector (Z-Score) + drift detector (CUSUM) in parallel

---

## Conclusion

We trained **20 anomaly detection models** + **2 concept drift models** by mistake.

The concept drift models (EWMA, CUSUM) performed poorly because they're not designed for detecting individual outliers - they're designed for detecting sustained changes in the data distribution.

For our NYC taxi anomaly detection task, we should focus on the **20 anomaly detection models** and ignore EWMA/CUSUM results.

If we want to detect concept drift (e.g., "traffic patterns changed after COVID"), we should use EWMA/CUSUM on a different evaluation metric (e.g., tracking changes in daily average trip counts over weeks/months).
