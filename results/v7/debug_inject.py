"""Deep debug: why are injected anomalies NOT outliers?"""
import sys, warnings
sys.path.insert(0, r'C:\proj\ldt\results\v7')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve
import benchmark_v7 as bm

DATA_DIR = Path(r'C:\proj\ldt\data\raw')
df = bm.clean(bm.load_month(2024, 1))
X = bm.features(df).astype(np.float32)

# Setup
train_X = X[:10000]
rng = np.random.RandomState(42)
test_df = df.iloc[:20000].reset_index(drop=True)
params = {'type': 'extreme_fare', 'fare_range': (150, 500), 'n': 1000}
test_inj, y = bm.inject_anomalies(test_df, params, 42)
test_X = bm.features(test_inj).astype(np.float32)

scaler = StandardScaler()
scaler.fit(train_X)
X_tr = scaler.transform(train_X).astype(np.float32)
X_te = scaler.transform(test_X).astype(np.float32)

orig = y == 0
anom = y == 1

print('=== Feature 6 (fare_per_mile) ===')
print(f'  Normal: mean={X_te[orig,6].mean():.2f}, median={np.median(X_te[orig,6]):.2f}, std={X_te[orig,6].std():.2f}')
print(f'  Anom:  mean={X_te[anom,6].mean():.2f}, median={np.median(X_te[anom,6]):.2f}, std={X_te[anom,6].std():.2f}')

print('\n=== Raw features in test_inj ===')
print(f'  Normal fare: mean={test_inj.loc[orig,"fare_amount"].mean():.1f}, median={test_inj.loc[orig,"fare_amount"].median():.1f}')
print(f'  Anom  fare: mean={test_inj.loc[anom,"fare_amount"].mean():.1f}, median={test_inj.loc[anom,"fare_amount"].median():.1f}')
print(f'  Normal dur_min: mean={test_inj.loc[orig,"dur_min"].mean():.1f}, median={test_inj.loc[orig,"dur_min"].median():.1f}')
print(f'  Anom  dur_min: mean={test_inj.loc[anom,"dur_min"].mean():.1f}, median={test_inj.loc[anom,"dur_min"].median():.1f}')
print(f'  Normal dist:  mean={test_inj.loc[orig,"trip_distance"].mean():.1f}, median={test_inj.loc[orig,"trip_distance"].median():.1f}')
print(f'  Anom  dist:  mean={test_inj.loc[anom,"trip_distance"].mean():.1f}, median={test_inj.loc[anom,"trip_distance"].median():.1f}')

# WHY: dur_min also inflated in inject_anomalies
print('\n=== KEY INSIGHT: fare_per_mile is DUR_MIN-normalized ===')
print('  fare = rng(150,500), dur = rng(1,10), dist = rng(0.1,2)')
print('  fare_per_mile = fare / max(dist, 0.01) = rng(150,500) / rng(0.1,2)')
print('  = rng(75, 5000) -- but anomaly stats show mean=4.86')
print('  WHY? Because features() computes fare_per_mile from RAW fields,')
print('  not from the injected values!')

# Verify: raw feature value from inject
raw_fare = test_inj['fare_amount'].values
raw_dist = test_inj['trip_distance'].values
raw_dur = test_inj['dur_min'].values
computed_fare_mi = raw_fare / np.maximum(raw_dist, 0.01)
print(f'\n  Computed fare_per_mile directly:')
print(f'    Normal: mean={computed_fare_mi[orig].mean():.2f}')
print(f'    Anom:   mean={computed_fare_mi[anom].mean():.2f}')

# Check what's in the features function - does it recompute dur_min?
# dur_min comes from duration_s = (dropoff - pickup).dt.total_seconds()
# But in inject, dur_min is SET directly. So features() recomputes dur_min from dropoff-pickup
# NOT from the injected dur_min column!

print('\n=== DEEP ISSUE: features() recomputes dur_min from dropoff-pickup ===')
print('  inject_anomalies() sets dur_min column directly')
print('  but features() recomputes dur_min = duration_s / 60 from dropoff-pickup')
print('  So the injected dur_min is IGNORED!')

# Verify: what's the actual duration_s in injected records?
print(f'\n  Original dropoff-pickup: mean={test_inj.loc[orig,"dur_min"].mean():.1f} min')
print(f'  Injected dur_min (set directly): mean={test_inj.loc[anom,"dur_min"].mean():.1f} min')
print(f'  BUT features() recomputes from dropoff-pickup!')

# The real issue: inject sets dur_min but features() IGNORES it
# features() uses: duration_s = (dropoff - pickup).dt.total_seconds()
# This is the ORIGINAL duration, not the injected one!

# So what does inject_anomalies actually inject?
print('\n=== What does inject actually change? ===')
print(f'  fare_amount: set to rng(150,500) -- CHANGED')
print(f'  trip_distance: set to rng(0.1,2) -- CHANGED')
print(f'  dur_min: set to rng(1,10) -- IGNORED by features()')
print(f'  duration_s: NOT changed -- RECOMPUTED by features()')

# Check: is duration_s changed by inject?
print(f'\n  duration_s in normal records: mean={test_inj.loc[orig,"dur_min"].mean():.1f} min')
print(f'  dur_min set for anomalies:    mean={test_inj.loc[anom,"dur_min"].mean():.1f} min')
print(f'  But features() recomputes from dropoff-pickup...')

# The injection problem: features() recomputes derived fields
# So only fare_amount and trip_distance actually change
# Let's see the effect:
print('\n=== Actual injected values (from inject_anomalies) ===')
# Look at raw fields after injection
print(f'  fare_amount: normal mean={test_inj.loc[orig,"fare_amount"].mean():.1f}, anom mean={test_inj.loc[anom,"fare_amount"].mean():.1f}')
print(f'  trip_distance: normal mean={test_inj.loc[orig,"trip_distance"].mean():.1f}, anom mean={test_inj.loc[anom,"trip_distance"].mean():.1f}')
print(f'  dur_min: normal mean={test_inj.loc[orig,"dur_min"].mean():.1f}, anom mean={test_inj.loc[anom,"dur_min"].mean():.1f}')

# The FEATURES that get computed from these:
# fare_per_mile = fare_amount / max(trip_distance, eps)
# With fare=250, dist=1.0: fare_per_mile = 250 -- MASSIVE outlier!
# But after scaling: (250 - train_mean) / train_std
print('\n=== fare_per_mile after scaling ===')
print(f'  Normal: mean={X_te[orig,6].mean():.2f}, std={X_te[orig,6].std():.2f}')
print(f'  Anom:   mean={X_te[anom,6].mean():.2f}, std={X_te[anom,6].std():.2f}')
# Check: what's the SCALED value?
train_mean = X_tr[:, 6].mean()
train_std = X_tr[:, 6].std()
# Anomaly fare_per_mile before scaling
anom_raw_fare = test_inj.loc[anom, "fare_amount"].values
anom_raw_dist = test_inj.loc[anom, "trip_distance"].values
anom_raw_fare_mi = anom_raw_fare / np.maximum(anom_raw_dist, 0.01)
print(f'\n  Anom raw fare_per_mile: mean={anom_raw_fare_mi.mean():.1f}, median={np.median(anom_raw_fare_mi):.1f}')
print(f'  Scaled: (mean - {train_mean:.2f}) / {train_std:.2f} = {(anom_raw_fare_mi.mean() - train_mean) / train_std:.2f}')

# ISOLATION FOREST ISSUE: IF uses random splits
# With 200 trees and 25 features, each getting random split points
# A value of 250 for fare_per_mile has a probability of ~1/200 of being
# in the "isolated" region. Same for any value.
# IF cannot LEARN that fare_per_mile > 100 is anomalous!
# It can only use RANDOM splits, not optimal splits.

print('\n=== ROOT CAUSE SUMMARY ===')
print('1. Anomalies have extreme raw fare ($150-500)')
print('2. BUT duration also inflated, so fare_per_mile varies wildly (75-5000)')
print('3. Even extreme fare_per_mile (250): IsolationForest random splits')
print('   cannot reliably isolate it -- only 1/200 chance per tree')
print('4. With 200 trees: ~63% of trees will NOT isolate fare_per_mile=250')
print('5. AUC-PR ends up ~0.05 (random)')

print('\n=== SOLUTION OPTIONS ===')
print('A. Increase n_estimators to 1000+ (costly)')
print('B. Use supervised model with injected labels')
print('C. Fix injection: only inject ONE field (e.g., only fare_amount)')
print('   so fare_per_mile = 250/7 = 35, vs normal 17 -> 2x outlier')
print('D. Use Mahalanobis distance instead of IsolationForest')
print('E. Use One-Class SVM with RBF kernel on scaled features')
