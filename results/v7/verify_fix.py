"""Verify the fixed inject_anomalies."""
import sys, warnings
sys.path.insert(0, r'C:\proj\ldt\results\v7')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import benchmark_v7 as bm

DATA_DIR = Path(r'C:\proj\ldt\data\raw')
df = bm.clean(bm.load_month(2024, 1))
test_df = df.iloc[:2000].reset_index(drop=True)

# Test extreme_fare
params = {'type': 'extreme_fare', 'fare_range': (150, 500), 'n': 500}
test_inj, y = bm.inject_anomalies(test_df, params, 42)
X = bm.features(test_inj).astype(np.float32)
anom = y == 1
normal = y == 0
print('=== extreme_fare ===')
print(f'  Fare - Normal: {X[normal, 2].mean():.1f}, Anom: {X[anom, 2].mean():.1f}')
print(f'  Fare/mi - Normal: {X[normal, 6].mean():.1f}, Anom: {X[anom, 6].mean():.1f}')

# Test partition
params = {'type': 'partition', 'n': 500,
           'components': [
               ('extreme_fare', (150, 500), 1),
               ('zero_dist', None, 1),
               ('slow_crawl', None, 1),
           ]}
test_inj2, y2 = bm.inject_anomalies(test_df, params, 42)
X2 = bm.features(test_inj2).astype(np.float32)
anom2 = y2 == 1
print(f'\n=== partition (hard) ===')
print(f'  Fare - Normal: {X2[~anom2, 2].mean():.1f}, Anom: {X2[anom2, 2].mean():.1f}')

# Quick sklearn_IF test
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, precision_recall_curve

train_X = X[:10000] if len(X) > 10000 else X[:len(X)//2]
scaler = StandardScaler()
scaler.fit(train_X)
test_X_scaled = scaler.transform(X)

clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
clf.fit(train_X)
scores = -clf.score_samples(test_X_scaled)
auc_val = auc(*precision_recall_curve(y, scores)[:2])
print(f'\n=== sklearn_IF AUC-PR with fixed injection ===')
print(f'  AUC-PR: {auc_val:.4f}')
print(f'  (Should be much higher than ~0.05 now)')
