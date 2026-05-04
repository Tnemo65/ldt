# Model Summary Table

## Quick Reference: Anomaly Detection vs Concept Drift

| Model | Primary Purpose | AUC | Category | Use For Anomaly? | Use For Drift? |
|-------|----------------|-----|----------|------------------|----------------|
| **Z-Score** | Anomaly Detection | 0.9421 ⭐ | Statistical | ✅ YES | ❌ NO |
| **IQR** | Anomaly Detection | 0.9253 ⭐ | Statistical | ✅ YES | ❌ NO |
| **HBOS** | Anomaly Detection | 0.9102 ⭐ | PyOD | ✅ YES | ❌ NO |
| **MCD** | Anomaly Detection | 0.8999 | PyOD | ✅ YES | ❌ NO |
| **VAE (dim=16)** | Anomaly Detection | 0.8968 | Deep Learning | ✅ YES | ❌ NO |
| **VAE (dim=8)** | Anomaly Detection | 0.8965 | Deep Learning | ✅ YES | ❌ NO |
| **PCA** | Anomaly Detection | 0.8924 | PyOD | ✅ YES | ❌ NO |
| **OneClassSVM** | Anomaly Detection | 0.8856 | sklearn | ✅ YES | ❌ NO |
| **MAD** | Anomaly Detection | 0.8820 | Statistical | ✅ YES | ❌ NO |
| **IsolationForest (n=100)** | Anomaly Detection | 0.8591 | sklearn | ✅ YES | ❌ NO |
| **IsolationForest (n=200)** | Anomaly Detection | 0.8462 | sklearn | ✅ YES | ❌ NO |
| **COPOD** | Anomaly Detection | 0.8186 | PyOD | ✅ YES | ❌ NO |
| **Autoencoder (dim=16)** | Anomaly Detection | 0.8070 | Deep Learning | ✅ YES | ❌ NO |
| **Autoencoder (dim=8)** | Anomaly Detection | 0.8043 | Deep Learning | ✅ YES | ❌ NO |
| **EWMA** | **Concept Drift** | 0.7954 ⚠️ | Window-based | ⚠️ LIMITED | ✅ YES |
| **ECOD** | Anomaly Detection | 0.7807 | PyOD | ✅ YES | ❌ NO |
| **LOF (k=20)** | Anomaly Detection | 0.5721 ❌ | sklearn | ❌ POOR | ❌ NO |
| **LOF (k=50)** | Anomaly Detection | 0.5669 ❌ | sklearn | ❌ POOR | ❌ NO |
| **ABOD** | Anomaly Detection | 0.5381 ❌ | PyOD | ❌ POOR | ❌ NO |
| **KNN (k=5)** | Anomaly Detection | 0.5195 ❌ | PyOD | ❌ POOR | ❌ NO |
| **KNN (k=20)** | Anomaly Detection | 0.5156 ❌ | PyOD | ❌ POOR | ❌ NO |
| **CUSUM** | **Concept Drift** | 0.4727 ❌ | Window-based | ❌ NO | ✅ YES |
| **RRCF** | Both | SKIPPED | Streaming | ✅ YES | ✅ YES |

---

## Model Categories by Detection Type

### 1. ANOMALY DETECTION ONLY (20 models)

These models detect **individual unusual data points** that deviate from normal patterns.

#### Top Performers (AUC > 0.85):
- ✅ Z-Score (0.9421) - Statistical
- ✅ IQR (0.9253) - Statistical  
- ✅ HBOS (0.9102) - PyOD
- ✅ MCD (0.8999) - PyOD
- ✅ VAE (0.8965-0.8968) - Deep Learning
- ✅ PCA (0.8924) - PyOD
- ✅ OneClassSVM (0.8856) - sklearn
- ✅ MAD (0.8820) - Statistical
- ✅ IsolationForest (0.8462-0.8591) - sklearn

#### Moderate Performers (AUC 0.70-0.85):
- ⚠️ COPOD (0.8186) - PyOD
- ⚠️ Autoencoder (0.8043-0.8070) - Deep Learning
- ⚠️ ECOD (0.7807) - PyOD

#### Poor Performers (AUC < 0.60):
- ❌ LOF (0.5669-0.5721) - sklearn
- ❌ ABOD (0.5381) - PyOD
- ❌ KNN (0.5156-0.5195) - PyOD

---

### 2. CONCEPT DRIFT DETECTION (2 models)

These models detect **changes in the underlying data distribution** over time.

⚠️ **IMPORTANT**: These were incorrectly used for anomaly detection in our evaluation!

#### EWMA (Exponentially Weighted Moving Average)
- **Purpose**: Detect gradual distribution shifts
- **How it works**: Tracks exponentially weighted average, alerts when current value deviates significantly
- **Best for**: Detecting trends (e.g., "traffic is gradually increasing")
- **NOT good for**: Point anomalies (e.g., "one spike at 3am")
- **Score**: 0.7954 AUC (moderate - because it can catch some persistent anomalies)

#### CUSUM (Cumulative Sum)
- **Purpose**: Detect sustained shifts in mean
- **How it works**: Accumulates deviations from target, alerts when cumulative sum exceeds threshold
- **Best for**: Mean shifts (e.g., "average trips changed from 100/hr to 110/hr")
- **NOT good for**: Individual spikes (e.g., "200 trips in one window")
- **Score**: 0.4727 AUC (terrible - worse than random guessing!)

---

### 3. BOTH ANOMALY + DRIFT (1 model)

#### RRCF (Robust Random Cut Forest)
- **Purpose**: Streaming anomaly detection AND concept drift detection
- **How it works**: Maintains forest of random cut trees, computes CoDisp (collusive displacement)
- **Best for**: Real-time streaming data with both point anomalies and distribution changes
- **Use cases**:
  - Point anomalies: Sudden spike in traffic
  - Concept drift: Gradual change in traffic patterns
- **Score**: SKIPPED (too slow - ~3 hours for 17k samples)
- **Note**: Could be optimized with sampling or batch scoring

---

## Detection Type Examples

### Anomaly Detection (What we're doing):
```
Normal: [100, 102, 98, 101, 99, 103, 97, 100, ...]
Anomaly: [100, 102, 98, 101, 500, 103, 97, 100, ...]  ← Detect this spike
```

### Concept Drift Detection (What EWMA/CUSUM are for):
```
Before: [100, 102, 98, 101, 99, 103, 97, 100, ...]  (mean ≈ 100)
After:  [150, 152, 148, 151, 149, 153, 147, 150, ...]  (mean ≈ 150)
         ↑ Distribution shifted - detect this change
```

---

## Recommendations by Use Case

### For Point Anomaly Detection (our current task):
1. **Best**: Z-Score (0.9421) - Fast, interpretable
2. **Best ML**: HBOS (0.9102) - Fast training/scoring  
3. **Best DL**: VAE (0.8968) - Complex patterns
4. **Production**: IsolationForest (0.8591) - Scalable

### For Concept Drift Detection (future work):
1. **Gradual drift**: EWMA
2. **Mean shift**: CUSUM
3. **Streaming**: RRCF
4. **Advanced**: Kolmogorov-Smirnov test, Page-Hinkley test

### For Real-time Monitoring (both):
1. **Hybrid**: Z-Score (anomalies) + EWMA (drift) in parallel
2. **Streaming**: RRCF (if performance can be optimized)

---

## Key Takeaways

1. ✅ **We correctly benchmarked 20 anomaly detection models**
2. ❌ **We incorrectly included 2 concept drift models (EWMA, CUSUM)**
3. 📊 **Statistical models (Z-Score, IQR) beat all ML models on our data**
4. 🚀 **HBOS is the best ML model (0.9102 AUC)**
5. 🧠 **VAE is the best deep learning model (0.8968 AUC)**
6. ⚠️ **CUSUM's poor score (0.47) is EXPECTED - it's not designed for point anomalies**
7. 🔄 **For drift detection, use EWMA/CUSUM on aggregated metrics, not raw scores**
