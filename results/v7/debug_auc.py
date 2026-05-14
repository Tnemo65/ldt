"""Debug why all AUC-PR ~ anomaly_rate (i.e., random)."""
import sys, warnings
sys.path.insert(0, r'C:\proj\ldt\results\v7')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve

import benchmark_v7 as bm

# Load data
DATA_DIR = Path(r'C:\proj\ldt\data\raw')
df = bm.clean(bm.load_month(2024, 1))
X = bm.features(df).astype(np.float32)
print(f'Clean records: {len(df)}')

# Inject easy anomalies (150-500 fare)
rng = np.random.RandomState(42)
test_df = df.iloc[:15000].reset_index(drop=True)
params = {'type': 'extreme_fare', 'fare_range': (150, 500), 'n': 750}
test_inj, y = bm.inject_anomalies(test_df, params, 42)
X_inj = bm.features(test_inj).astype(np.float32)
print(f'Injected: {len(test_inj)}, anomalies: {y.sum()}')

# Check feature distributions
orig = X_inj[y==0]
anom = X_inj[y==1]
print(f'\nFeature comparison (mean +/- std):')
feature_names = ['dist','dur','fare','pax','total','speed',
                'fare/mi','fare/min','fare/pax','hour','dow']
for i in range(min(12, X_inj.shape[1])):
    om = orig[:, i].mean()
    os = orig[:, i].std()
    am = anom[:, i].mean()
    as_ = anom[:, i].std()
    sep = abs(om - am) / max(os, as_, 0.01)
    print(f'  [{i:2d}] {om:10.2f} +/- {os:7.2f} vs {am:10.2f} +/- {as_:7.2f}  sep={sep:.2f}')

# Check AUC of individual features
print(f'\nIndividual feature AUC-PR (discriminative power):')
for i in range(X_inj.shape[1]):
    try:
        auc_val = auc(*precision_recall_curve(y, X_inj[:, i])[:2])
    except:
        auc_val = 0.5
    if abs(auc_val - 0.5) > 0.05:
        print(f'  [{i:2d}] feature {i}: AUC_PR={auc_val:.4f} ***')
    else:
        print(f'  [{i:2d}] feature {i}: AUC_PR={auc_val:.4f}')

# Check: do injected anomalies have unique features?
print(f'\nFare amount raw check:')
print(f'  Original: mean={test_df["fare_amount"].mean():.2f}, median={test_df["fare_amount"].median():.2f}, max={test_df["fare_amount"].max():.2f}')
print(f'  After injection: mean={test_inj["fare_amount"].mean():.2f}, median={test_inj["fare_amount"].median():.2f}, max={test_inj["fare_amount"].max():.2f}')
print(f'  Anomalies: mean={test_inj.iloc[y==1]["fare_amount"].mean():.2f}, min={test_inj.iloc[y==1]["fare_amount"].min():.2f}, max={test_inj.iloc[y==1]["fare_amount"].max():.2f}')
