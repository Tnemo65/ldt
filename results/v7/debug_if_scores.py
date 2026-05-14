"""Debug why sklearn_IF gets ~random AUC-PR."""
import sys, warnings
sys.path.insert(0, r'C:\proj\ldt\results\v7')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, precision_recall_curve

import benchmark_v7 as bm

DATA_DIR = Path(r'C:\proj\ldt\data\raw')
df = bm.clean(bm.load_month(2024, 1))
X = bm.features(df).astype(np.float32)

# Train on first 10K
train_X = X[:10000]
val_X = X[10000:12000]

# Test with injection
rng = np.random.RandomState(42)
test_df = df.iloc[:15000].reset_index(drop=True)
params = {'type': 'extreme_fare', 'fare_range': (150, 500), 'n': 750}
test_inj, y = bm.inject_anomalies(test_df, params, 42)
test_X = bm.features(test_inj).astype(np.float32)
print(f'Test: {test_X.shape}, anom={y.sum()}')

# Scale
scaler = StandardScaler()
scaler.fit(train_X)
X_tr = scaler.transform(train_X).astype(np.float32)
X_te = scaler.transform(test_X).astype(np.float32)

# Check feature 6 (fare_per_mile)
orig_mask = y == 0
anom_mask = y == 1
print(f'\nFeature 6 (fare_per_mile) after scaling:')
print(f'  Normal mean: {X_te[orig_mask, 6].mean():.4f}, std: {X_te[orig_mask, 6].std():.4f}')
print(f'  Anomaly mean: {X_te[anom_mask, 6].mean():.4f}, std: {X_te[anom_mask, 6].std():.4f}')
print(f'  Separation: {abs(X_te[orig_mask, 6].mean() - X_te[anom_mask, 6].mean()):.4f}')

# Check IF scores
clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
clf.fit(X_tr)
raw_scores = -clf.score_samples(X_tr)
print(f'\nIsolationForest score range on train:')
print(f'  mean: {raw_scores.mean():.4f}, std: {raw_scores.std():.4f}')
print(f'  min: {raw_scores.min():.4f}, max: {raw_scores.max():.4f}')

test_scores = -clf.score_samples(X_te)
print(f'\nIsolationForest score range on test:')
print(f'  Normal mean: {test_scores[orig_mask].mean():.4f}, std: {test_scores[orig_mask].std():.4f}')
print(f'  Anomaly mean: {test_scores[anom_mask].mean():.4f}, std: {test_scores[anom_mask].std():.4f}')

auc_val = auc(*precision_recall_curve(y, test_scores)[:2])
print(f'\nAUC-PR: {auc_val:.4f}')

# Check: is IF confused because of extreme values?
# IF uses random splits. If fare_per_mile is 250, IF will always split on that
# value, making it "normal" for the forest.
# But the real question: does the IF learn what's normal vs anomalous?

# Better test: check AUC of raw feature (no scaling)
raw_fare = test_X[:, 6]  # fare_per_mile BEFORE scaling
print(f'\nRaw feature 6 AUC-PR (no scaling): {auc(*precision_recall_curve(y, raw_fare)[:2]):.4f}')

# Also check raw fare_amount
raw_fare2 = test_X[:, 2]  # fare_amount
print(f'Raw feature 2 (fare) AUC-PR (no scaling): {auc(*precision_recall_curve(y, raw_fare2)[:2]):.4f}')

# Check: what happens with sklearn_IF on UNSCALED data?
clf2 = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
clf2.fit(train_X.astype(np.float64))
test_scores2 = -clf2.score_samples(test_X.astype(np.float64))
auc_val2 = auc(*precision_recall_curve(y, test_scores2)[:2])
print(f'\nsklearn_IF on UNSCALED data AUC-PR: {auc_val2:.4f}')

# Check AUC of fare_per_mile on raw test data
raw_X = bm.features(test_inj).astype(np.float32)
print(f'\nRaw fare_per_mile AUC-PR: {auc(*precision_recall_curve(y, raw_X[:,6])[:2]):.4f}')
print(f'Raw fare_amount AUC-PR: {auc(*precision_recall_curve(y, raw_X[:,2])[:2]):.4f}')
print(f'Raw fare_per_mile stats: normal mean={raw_X[y==0,6].mean():.1f}, anom mean={raw_X[y==1,6].mean():.1f}')
