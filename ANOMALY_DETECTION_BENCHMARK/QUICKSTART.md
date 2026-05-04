# Quick Start Guide

## 1-Minute Setup

```bash
cd ANOMALY_DETECTION_BENCHMARK

# Install dependencies
pip install pandas numpy scikit-learn pyod tensorflow

# View results (already computed)
cat results/comprehensive_comparison.csv
```

## Top 3 Models to Use

### 1. Z-Score (Best Overall)
```python
import pandas as pd
import numpy as np

# Load training data
df_train = pd.read_parquet('data/train_2024.parquet')

# Compute context statistics (by hour and day-of-week)
context_stats = {}
for (dow, hour), group in df_train.groupby(['dow', 'hour']):
    context_stats[(dow, hour)] = {
        'mean': group['trip_count'].mean(),
        'std': group['trip_count'].std()
    }

# Score test data
df_test = pd.read_parquet('data/test_2025.parquet')
scores = []
for idx, row in df_test.iterrows():
    stats = context_stats[(row['dow'], row['hour'])]
    z = abs(row['trip_count'] - stats['mean']) / stats['std']
    scores.append(z)

# Anomalies: z > 3.0
anomalies = np.array(scores) > 3.0
```

### 2. IsolationForest (Production)
```python
import joblib

# Load pretrained model
model = joblib.load('models/iforest_n100.pkl')

# Score test data
X_test = df_test[feature_cols].values
scores = -model.score_samples(X_test)

# Anomalies: top 2%
threshold = np.percentile(scores, 98)
anomalies = scores > threshold
```

### 3. HBOS (Best ML)
```python
import joblib

# Load pretrained model
model = joblib.load('models/hbos_bins10.pkl')

# Score test data
scores = model.predict_proba(X_test)[:, 1]

# Anomalies: top 2%
threshold = np.percentile(scores, 98)
anomalies = scores > threshold
```

## Feature Columns

```python
feature_cols = [
    'ctx_mean', 'ctx_std', 'ctx_median', 'ctx_q25', 'ctx_q75',
    'ctx_dev', 'ctx_abs_dev',
    'lag_48', 'lag_144', 'lag_336',
    'roll_mean_48', 'roll_std_48', 'roll_mean_336',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
]
```

## Training from Scratch

```bash
# Train all 20 models (110 seconds)
python code/exp_train_all_models.py \
    --train-file data/train_2024.parquet \
    --output-dir my_models/

# Score all models
python code/exp_score_all.py \
    --test-file data/test_2025.parquet \
    --models-dir my_models/ \
    --output-dir my_results/

# Evaluate
python code/exp_evaluate_comprehensive.py \
    --predictions-dir my_results/predictions/ \
    --output-dir my_results/
```

## Results Summary

| Model | AUC | Use For |
|-------|-----|---------|
| Z-Score | 0.9421 | Fast baseline |
| IQR | 0.9253 | Robust detection |
| HBOS | 0.9102 | Best ML |
| IsolationForest | 0.8591 | Production |
| VAE | 0.8968 | Deep learning |

See `documentation/` for detailed analysis.
