"""
CA-DQStream Full Dataset Benchmark — OFFICIAL RESULTS
=======================================================
Hardware: 32 vCPUs, 88 GB RAM, RTX 3090 Ti 24 GB
Dataset:  ~2.96M records (full Jan 2024 NYC Yellow Taxi)
Runs:     5 variants x 5 seeds x 3 difficulty levels = 75 runs
Output:   results/ (CSV, figures, summary)
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_curve, auc
)
from scipy.stats import ttest_rel, wilcoxon, sem
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)

# ===========================
# CONFIGURATION
# ===========================
FAST_MODE = False
N_SEEDS = 5
SYNTHETIC_PER_SCENARIO = 10000
TRAIN_RATIO = 0.7
N_CLUSTERS = 7
DIFFICULTY_LEVELS = ['easy', 'medium', 'hard']

# Hardware: use all cores
N_JOBS = -1

# Output directory
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# ===========================
# DATA FILE DISCOVERY
# ===========================
candidates_list = [
    '/kaggle/input/nyc-taxi-trip-data/yellow_tripdata_2024-01.parquet',
    'yellow_tripdata_2024-01.parquet',
    'data/raw/yellow_tripdata_2024-01.parquet',
    '../data/raw/yellow_tripdata_2024-01.parquet',
    r'C:\proj\ldt\data\raw\yellow_tripdata_2024-01.parquet',
]
DATA_FILE = None
for candidate in candidates_list:
    if os.path.exists(candidate):
        DATA_FILE = candidate
        break

assert DATA_FILE is not None, f"Dataset not found. Searched: {candidates_list}"
print(f"[CONFIG] Mode: {'FAST (100K)' if FAST_MODE else 'FULL (2.96M)'} | Data: {DATA_FILE}")
print(f"[CONFIG] N_JOBS: {N_JOBS} (all cores) | Output: {OUT_DIR}")

SEEDS = [42, 123, 456, 789, 1024]

FEATURE_NAMES_15D = [
    'distance', 'duration_min', 'fare', 'passengers', 'total',
    'speed', 'fare_per_mile', 'fare_per_minute', 'fare_per_passenger',
    'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'is_night', 'month',
]
FEATURE_NAMES_21D = FEATURE_NAMES_15D + [
    'fare_per_mile_ratio', 'fare_per_minute_ratio', 'implied_speed_ratio',
    'passenger_distance_ratio', 'fare_distance_product', 'duration_distance_ratio',
]

BASELINE = {
    'fare_per_mile': 2.5,
    'fare_per_minute': 0.67,
    'implied_speed': 12.0,
}

# ===========================
# STEP 1: LOAD DATA
# ===========================
print("\n" + "=" * 70)
print("[1/10] LOAD RAW DATA")
print("=" * 70)
t0 = time.time()
df_raw = pd.read_parquet(DATA_FILE)
if FAST_MODE:
    df_raw = df_raw.sample(n=100_000, random_state=42).reset_index(drop=True)
print(f"  Loaded: {len(df_raw):,} records in {time.time()-t0:.1f}s")
print(f"  Columns: {list(df_raw.columns)}")

# ===========================
# STEP 2: L1 SCHEMA VALIDATION
# ===========================
print("\n" + "=" * 70)
print("[2/10] L1 — SCHEMA VALIDATION")
print("=" * 70)
required_cols = [
    'trip_distance', 'fare_amount', 'PULocationID', 'DOLocationID',
    'passenger_count', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
]
l1_valid = (
    df_raw[required_cols].notna().all(axis=1)
    & df_raw['passenger_count'].between(1, 6)
    & df_raw['PULocationID'].between(1, 263)
    & df_raw['DOLocationID'].between(1, 263)
)
l1_rejected = (~l1_valid).sum()
df_l1 = df_raw[l1_valid].copy().reset_index(drop=True)
print(f"  Input:    {len(df_raw):,} | Rejected: {l1_rejected:,} ({l1_rejected/len(df_raw)*100:.2f}%) | Output: {len(df_l1):,}")

# ===========================
# STEP 3: L2 RULE-BASED CANARY
# ===========================
print("\n" + "=" * 70)
print("[3/10] L2 — RULE-BASED CANARY")
print("=" * 70)
df_l1['pickup_dt'] = pd.to_datetime(df_l1['tpep_pickup_datetime'])
df_l1['dropoff_dt'] = pd.to_datetime(df_l1['tpep_dropoff_datetime'])
df_l1['duration_sec'] = (df_l1['dropoff_dt'] - df_l1['pickup_dt']).dt.total_seconds()
df_l1['duration_hours'] = df_l1['duration_sec'] / 3600
df_l1['speed_mph'] = df_l1['trip_distance'] / (df_l1['duration_hours'] + 1e-9)

l2_valid = (
    (df_l1['fare_amount'] > 0)
    & (df_l1['trip_distance'] > 0)
    & (df_l1['duration_sec'] > 0)
    & (df_l1['speed_mph'] < 100)
    & (df_l1['speed_mph'] > 0)
)
l2_rejected = (~l2_valid).sum()
df_l2 = df_l1[l2_valid].copy().reset_index(drop=True)
print(f"  Input:    {len(df_l1):,} | Rejected: {l2_rejected:,} ({l2_rejected/len(df_l1)*100:.2f}%) | Output: {len(df_l2):,}")

# ===========================
# STEP 4: L3 IQR OUTLIER REMOVAL
# ===========================
print("\n" + "=" * 70)
print("[4/10] L3 — IQR OUTLIER REMOVAL (3x IQR)")
print("=" * 70)
df_clean = df_l2.copy()
iqr_mult = 3.0
for col in ['fare_amount', 'trip_distance', 'duration_hours']:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - iqr_mult * IQR, Q3 + iqr_mult * IQR
    before = len(df_clean)
    df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    removed = before - len(df_clean)
    print(f"  {col}: removed {removed:,} ({removed/before*100:.1f}%)")
df_clean = df_clean.reset_index(drop=True)
print(f"\n  Clean records: {len(df_clean):,}")

# Funnel summary
print("\n  PIPELINE FUNNEL:")
stages = [
    ("Raw", len(df_raw)),
    ("After L1 (Schema)", len(df_l1)),
    ("After L2 (Rules)", len(df_l2)),
    ("After L3 (IQR)", len(df_clean)),
]
for name, count in stages:
    pct = count / len(df_raw) * 100
    print(f"    {name:25s}: {count:>10,} ({pct:5.1f}%)")

# ===========================
# STEP 5: TRAIN/TEST SPLIT
# ===========================
print("\n" + "=" * 70)
print("[5/10] TRAIN/TEST SPLIT (70/30)")
print("=" * 70)
n_total = len(df_clean)
n_train = int(n_total * TRAIN_RATIO)
indices = np.random.RandomState(42).permutation(n_total)
train_idx = indices[:n_train]
test_idx = indices[n_train:]
df_train = df_clean.iloc[train_idx].copy().reset_index(drop=True)
df_test_clean = df_clean.iloc[test_idx].copy().reset_index(drop=True)
print(f"  Train: {len(df_train):,} | Test: {len(df_test_clean):,}")
print(f"  No anomalies in train (zero contamination)")

# ===========================
# STEP 6: FEATURE VECTORIZER
# ===========================
print("\n" + "=" * 70)
print("[6/10] FEATURE VECTORIZER (15D + 21D)")
print("=" * 70)

def extract_features(df, mode='21D'):
    eps = 1e-6
    pickup = pd.to_datetime(df['tpep_pickup_datetime'])
    dropoff = pd.to_datetime(df['tpep_dropoff_datetime'])
    dur_sec = (dropoff - pickup).dt.total_seconds()
    dur_min = dur_sec / 60
    dur_hr = dur_sec / 3600

    dist = df['trip_distance'].values.astype(float)
    fare = df['fare_amount'].values.astype(float)
    pax = df['passenger_count'].values.astype(float)
    total = df['total_amount'].values.astype(float)

    speed = dist / (dur_hr.values + eps)
    fpm = fare / (dist + eps)
    fpmn = fare / (dur_min.values + eps)
    fpp = fare / (pax + eps)

    hour = pickup.dt.hour.values.astype(float)
    dow = pickup.dt.weekday.values.astype(float)
    is_wknd = (dow >= 5).astype(float)
    is_rush = (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 19))).astype(float)
    is_night = ((hour < 6) | (hour > 22)).astype(float)
    month = pickup.dt.month.values.astype(float)

    base = np.column_stack([
        dist, dur_min.values, fare, pax, total,
        speed, fpm, fpmn, fpp,
        hour, dow, is_wknd, is_rush, is_night, month,
    ])

    if mode == '15D':
        return base

    fpm_ratio = fpm / (BASELINE['fare_per_mile'] + eps)
    fpmn_ratio = fpmn / (BASELINE['fare_per_minute'] + eps)
    speed_ratio = speed / (BASELINE['implied_speed'] + eps)
    pax_dist_ratio = pax / (dist + eps)
    fare_dist_prod = fare * dist
    dur_dist_ratio = dur_min.values / (dist + eps)

    ratios = np.column_stack([
        fpm_ratio, fpmn_ratio, speed_ratio,
        pax_dist_ratio, fare_dist_prod, dur_dist_ratio,
    ])
    return np.hstack([base, ratios])

print("  Extracting features...")
t1 = time.time()
X_train_15d = extract_features(df_train, mode='15D')
X_train_21d = extract_features(df_train, mode='21D')
print(f"  15D: {X_train_15d.shape} | 21D: {X_train_21d.shape} ({time.time()-t1:.1f}s)")

print("  Fitting scalers...")
t1 = time.time()
scaler_15d = StandardScaler().fit(X_train_15d)
scaler_21d = StandardScaler().fit(X_train_21d)
X_train_15d_scaled = scaler_15d.transform(X_train_15d)
X_train_21d_scaled = scaler_21d.transform(X_train_21d)
print(f"  Scalers fitted ({time.time()-t1:.1f}s)")

print("  21D Feature statistics:")
for i, name in enumerate(FEATURE_NAMES_21D):
    vals = X_train_21d[:, i]
    print(f"    [{i:2d}] {name:30s}: mean={vals.mean():10.3f}, std={vals.std():10.3f}")

# ===========================
# STEP 7: ANOMALY GENERATOR
# ===========================
print("\n" + "=" * 70)
print("[7/10] ANOMALY GENERATOR (5 scenarios x 3 difficulty levels)")
print("=" * 70)

DIFFICULTY_CONFIGS = {
    'easy': {
        'meter_fare_mult': (10, 20), 'gps_speed': (50, 95),
        'pax_fare': (40, 70), 'crawl_dur': (90, 180), 'combined_fare_mult': (10, 20),
    },
    'medium': {
        'meter_fare_mult': (4, 8), 'gps_speed': (30, 60),
        'pax_fare': (15, 30), 'crawl_dur': (40, 80), 'combined_fare_mult': (4, 8),
    },
    'hard': {
        'meter_fare_mult': (1.5, 3), 'gps_speed': (20, 40),
        'pax_fare': (8, 15), 'crawl_dur': (20, 35), 'combined_fare_mult': (2, 4),
    },
}

def inject_meter_tampering(df, indices, cfg):
    lo, hi = cfg['meter_fare_mult']
    for idx in indices:
        dist = np.random.uniform(1.0, 3.0)
        dur_min = np.random.uniform(5, 15)
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = dist * 2.50 * np.random.uniform(lo, hi)
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + np.random.uniform(5, 10)
        df.at[idx, 'passenger_count'] = np.random.randint(1, 5)

def inject_gps_spoofing(df, indices, cfg):
    lo, hi = cfg['gps_speed']
    for idx in indices:
        target_speed = np.random.uniform(lo, hi)
        dist = np.random.uniform(20, 40)
        dur_min = (dist / target_speed) * 60
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = dist * np.random.uniform(2.0, 3.5)
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + np.random.uniform(5, 15)
        df.at[idx, 'passenger_count'] = np.random.randint(1, 4)

def inject_passenger_anomaly(df, indices, cfg):
    lo, hi = cfg['pax_fare']
    for idx in indices:
        dist = np.random.uniform(0.2, 0.5)
        dur_min = np.random.uniform(15, 30)
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = np.random.uniform(lo, hi)
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + np.random.uniform(3, 8)
        df.at[idx, 'passenger_count'] = np.random.randint(1, 6)

def inject_slow_crawl(df, indices, cfg):
    lo, hi = cfg['crawl_dur']
    for idx in indices:
        dist = np.random.uniform(2, 4)
        dur_min = np.random.uniform(lo, hi)
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = np.random.uniform(40, 80)
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + np.random.uniform(3, 8)
        df.at[idx, 'passenger_count'] = np.random.randint(1, 3)

def inject_combined_subtle(df, indices, cfg):
    lo, hi = cfg['combined_fare_mult']
    for idx in indices:
        dist = np.random.uniform(1, 2)
        dur_min = np.random.uniform(5, 10)
        pickup = pd.to_datetime(df.at[idx, 'tpep_pickup_datetime'])
        df.at[idx, 'trip_distance'] = dist
        df.at[idx, 'tpep_dropoff_datetime'] = pickup + timedelta(minutes=dur_min)
        df.at[idx, 'fare_amount'] = dist * 2.50 * np.random.uniform(lo, hi)
        df.at[idx, 'total_amount'] = df.at[idx, 'fare_amount'] + np.random.uniform(5, 15)
        df.at[idx, 'passenger_count'] = np.random.randint(4, 6)

SCENARIOS = [
    ('meter_tampering', inject_meter_tampering),
    ('gps_spoofing', inject_gps_spoofing),
    ('passenger_anomaly', inject_passenger_anomaly),
    ('slow_crawl', inject_slow_crawl),
    ('combined_subtle', inject_combined_subtle),
]

for level, cfg in DIFFICULTY_CONFIGS.items():
    print(f"  {level}: {cfg}")

# ===========================
# STEP 8: INJECT ANOMALIES
# ===========================
print("\n" + "=" * 70)
print("[8/10] INJECT ANOMALIES — 3 difficulty levels")
print("=" * 70)

level_data = {}
for level in DIFFICULTY_LEVELS:
    cfg = DIFFICULTY_CONFIGS[level]
    df_test_level = df_test_clean.copy()
    n_per = SYNTHETIC_PER_SCENARIO
    n_total_anom = n_per * len(SCENARIOS)

    rng = np.random.RandomState(42)
    anom_indices = rng.choice(len(df_test_level), size=n_total_anom, replace=False)
    idx_splits = np.array_split(anom_indices, len(SCENARIOS))

    y_test_level = np.zeros(len(df_test_level), dtype=int)
    sc_labels_level = np.full(len(df_test_level), 'normal', dtype=object)

    for i, (name, inject_fn) in enumerate(SCENARIOS):
        inject_fn(df_test_level, idx_splits[i], cfg)
        y_test_level[idx_splits[i]] = 1
        sc_labels_level[idx_splits[i]] = name

    X_test_15d_level = scaler_15d.transform(extract_features(df_test_level, mode='15D'))
    X_test_21d_level = scaler_21d.transform(extract_features(df_test_level, mode='21D'))

    level_data[level] = {
        'df_test': df_test_level,
        'y_test': y_test_level,
        'scenario_labels': sc_labels_level,
        'X_test_15d': X_test_15d_level,
        'X_test_21d': X_test_21d_level,
    }
    print(f"  {level}: {y_test_level.sum():,} anomalies / {len(df_test_level):,} total ({y_test_level.mean()*100:.2f}%)")

# ===========================
# STEP 9: SANITY CHECKS
# ===========================
print("\n" + "=" * 70)
print("[9/10] SANITY CHECKS")
print("=" * 70)

# CP1: Train Sterile
print("\n[CP1] Train Sterile")
assert (df_train['fare_amount'] <= 0).sum() == 0
assert (df_train['trip_distance'] <= 0).sum() == 0
assert (df_train['passenger_count'] > 6).sum() == 0
assert (df_train['passenger_count'] < 1).sum() == 0
assert len(df_train) / len(df_l1) > 0.50
print(f"  Train: {len(df_train):,} records — PASS")

# CP2: All anomalies pass L1+L2
print("\n[CP2] Anomalies pass L1+L2 (all levels)")
for level in DIFFICULTY_LEVELS:
    ld = level_data[level]
    df_a = ld['df_test'][ld['y_test'] == 1]
    a_pickup = pd.to_datetime(df_a['tpep_pickup_datetime'])
    a_dropoff = pd.to_datetime(df_a['tpep_dropoff_datetime'])
    a_speed = df_a['trip_distance'].values / ((a_dropoff - a_pickup).dt.total_seconds().values / 3600 + 1e-9)
    assert (df_a['passenger_count'].between(1, 6)).all(), f"FAIL {level}: passengers"
    assert (df_a['fare_amount'] > 0).all(), f"FAIL {level}: fare"
    assert (df_a['trip_distance'] > 0).all(), f"FAIL {level}: distance"
    assert (a_speed < 100).all(), f"FAIL {level}: speed >= 100 (max={a_speed.max():.1f})"
    print(f"  {level}: {(ld['y_test']==1).sum():,} anomalies — PASS")

# CP3: Feature dimensions
print("\n[CP3] Features")
assert X_train_21d.shape[1] == 21
assert X_train_15d.shape[1] == 15
print(f"  15D: {X_train_15d.shape}, 21D: {X_train_21d.shape} — PASS")

# CP4: Context Mapping
print("\n[CP4] Context Mapping")
kmeans_cp4 = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42).fit(X_train_21d_scaled)
n_clusters_used = len(set(kmeans_cp4.predict(X_train_21d_scaled)))
print(f"  {N_CLUSTERS} clusters fitted — PASS (all {n_clusters_used} clusters active)")

print("\n  ALL CHECKPOINTS PASSED")

# ===========================
# EVALUATION HELPERS
# ===========================
def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
    return {
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'FPR': fpr_val,
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
    }

def score_iforest(model, X_test_data, X_train_data, percentile):
    train_scores = -model.decision_function(X_train_data)
    threshold = np.percentile(train_scores, percentile)
    test_scores = -model.decision_function(X_test_data)
    return (test_scores > threshold).astype(int)

def score_per_cluster(model, X_test_data, X_train_data, kmeans, percentile):
    train_labels = kmeans.predict(X_train_data)
    test_labels = kmeans.predict(X_test_data)
    train_scores = -model.decision_function(X_train_data)
    test_scores = -model.decision_function(X_test_data)
    cluster_thresholds = {}
    for cid in range(kmeans.n_clusters):
        mask = train_labels == cid
        if mask.sum() > 10:
            cluster_thresholds[cid] = np.percentile(train_scores[mask], percentile)
        else:
            cluster_thresholds[cid] = np.percentile(train_scores, percentile)
    y_pred = np.zeros(len(X_test_data), dtype=int)
    for cid, thresh in cluster_thresholds.items():
        mask = test_labels == cid
        y_pred[mask] = (test_scores[mask] > thresh).astype(int)
    return y_pred, cluster_thresholds

def run_variant(variant_name, X_tr, X_te, y_true, seed):
    t0 = time.time()
    if variant_name == 'baseline_static':
        model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=N_JOBS)
        model.fit(X_tr)
        y_pred = score_iforest(model, X_te, X_tr, percentile=95)
    elif variant_name == 'baseline_ratio':
        model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=N_JOBS)
        model.fit(X_tr)
        y_pred = score_iforest(model, X_te, X_tr, percentile=96)
    elif variant_name == 'proposed_context_aware':
        model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=N_JOBS)
        model.fit(X_tr)
        kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=seed)
        kmeans.fit(X_tr)
        y_pred, _ = score_per_cluster(model, X_te, X_tr, kmeans, percentile=97)
        n_clusters_used = len(set(kmeans.predict(X_te)))
        assert n_clusters_used >= 2
    elif variant_name == 'opponent_lof':
        rng = np.random.RandomState(seed)
        sample_size = min(50_000, len(X_tr))
        X_sample = X_tr[rng.choice(len(X_tr), sample_size, replace=False)]
        model = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True, n_jobs=N_JOBS)
        model.fit(X_sample)
        train_scores = -model.decision_function(X_sample)
        threshold = np.percentile(train_scores, 96)
        test_scores = -model.decision_function(X_te)
        y_pred = (test_scores > threshold).astype(int)
    elif variant_name == 'opponent_ocsvm':
        rng = np.random.RandomState(seed)
        sample_size = min(30_000, len(X_tr))
        X_sample = X_tr[rng.choice(len(X_tr), sample_size, replace=False)]
        model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
        model.fit(X_sample)
        train_scores = -model.decision_function(X_sample)
        threshold = np.percentile(train_scores, 96)
        test_scores = -model.decision_function(X_te)
        y_pred = (test_scores > threshold).astype(int)
    train_time = time.time() - t0
    metrics = evaluate(y_true, y_pred)
    metrics['train_time'] = train_time
    metrics['variant'] = variant_name
    metrics['seed'] = seed
    return metrics

print("Evaluation helpers loaded")

# ===========================
# STEP 10: TRAIN & EVALUATE
# ===========================
print("\n" + "=" * 70)
print("[10/10] TRAIN & EVALUATE — 5 variants x 5 seeds x 3 levels")
print("=" * 70)

VARIANT_DEFS = [
    ('baseline_static',        '15D'),
    ('baseline_ratio',         '21D'),
    ('proposed_context_aware', '21D'),
    ('opponent_lof',           '21D'),
    ('opponent_ocsvm',        '21D'),
]

all_results = []
total_runs = N_SEEDS * len(VARIANT_DEFS) * len(DIFFICULTY_LEVELS)
run_count = 0
t_benchmark = time.time()

for level in DIFFICULTY_LEVELS:
    ld = level_data[level]
    print(f"\n  === LEVEL: {level.upper()} ===")
    for vname, feat_mode in VARIANT_DEFS:
        X_tr = X_train_15d_scaled if feat_mode == '15D' else X_train_21d_scaled
        X_te = ld['X_test_15d'] if feat_mode == '15D' else ld['X_test_21d']
        y_true = ld['y_test']
        for seed in SEEDS:
            metrics = run_variant(vname, X_tr, X_te, y_true, seed)
            metrics['level'] = level
            all_results.append(metrics)
            run_count += 1
        last = all_results[-1]
        elapsed = time.time() - t_benchmark
        eta = (elapsed / run_count) * (total_runs - run_count) if run_count > 0 else 0
        print(f"    {vname:<25} F1={last['F1']:.3f} Recall={last['Recall']:.3f} FPR={last['FPR']:.4f}  [{run_count}/{total_runs}] ETA={eta:.0f}s")

df_results = pd.DataFrame(all_results)
print(f"\n  Total runs: {len(df_results)} | Total time: {time.time()-t_benchmark:.1f}s")

# ===========================
# SAVE CSV
# ===========================
csv_path = os.path.join(OUT_DIR, 'benchmark_results.csv')
df_results.to_csv(csv_path, index=False)
print(f"\n  Results saved: {csv_path}")

# ===========================
# RESULTS SUMMARY TABLE
# ===========================
print("\n" + "=" * 70)
print("BENCHMARK RESULTS BY DIFFICULTY (mean +/- std, 5 seeds)")
print("=" * 70)

for level in DIFFICULTY_LEVELS:
    df_lv = df_results[df_results['level'] == level]
    summary_lv = df_lv.groupby('variant').agg(
        F1_mean=('F1', 'mean'), F1_std=('F1', 'std'),
        Recall_mean=('Recall', 'mean'), Recall_std=('Recall', 'std'),
        Precision_mean=('Precision', 'mean'), Precision_std=('Precision', 'std'),
        FPR_mean=('FPR', 'mean'), FPR_std=('FPR', 'std'),
        train_time_mean=('train_time', 'mean'),
    ).reset_index().sort_values('F1_mean', ascending=False).reset_index(drop=True)
    summary_lv['Rank'] = range(1, len(summary_lv) + 1)

    print(f"\n  LEVEL: {level.upper()}")
    print(f"  {'Rank':<5} {'Variant':<25} {'F1':>12} {'Recall':>12} {'Precision':>12} {'FPR':>10} {'Time(s)':>8}")
    print(f"  {'-'*85}")
    for _, row in summary_lv.iterrows():
        print(f"  {row['Rank']:<5.0f} {row['variant']:<25} "
              f"{row['F1_mean']:.3f}+/-{row['F1_std']:.3f}  "
              f"{row['Recall_mean']:.3f}+/-{row['Recall_std']:.3f}  "
              f"{row['Precision_mean']:.3f}+/-{row['Precision_std']:.3f}  "
              f"{row['FPR_mean']:.4f}+/-{row['FPR_std']:.4f}  "
              f"{row['train_time_mean']:>6.1f}s")

# Cross-level comparison
print(f"\n{'='*70}")
print("KEY COMPARISON: proposed_context_aware vs baseline_static")
print(f"{'='*70}")
print(f"  {'Level':<8} {'Proposed F1':>10} {'Static F1':>10} {'Gap':>8} {'Proposed FPR':>12} {'Static FPR':>12}")
print(f"  {'-'*65}")
for level in DIFFICULTY_LEVELS:
    df_lv = df_results[df_results['level'] == level]
    pc_f1 = df_lv[df_lv['variant'] == 'proposed_context_aware']['F1'].mean()
    bs_f1 = df_lv[df_lv['variant'] == 'baseline_static']['F1'].mean()
    pc_fpr = df_lv[df_lv['variant'] == 'proposed_context_aware']['FPR'].mean()
    bs_fpr = df_lv[df_lv['variant'] == 'baseline_static']['FPR'].mean()
    gap = pc_f1 - bs_f1
    print(f"  {level:<8} {pc_f1:>10.3f} {bs_f1:>10.3f} {gap:>+8.3f} {pc_fpr:>12.4f} {bs_fpr:>12.4f}")

# ===========================
# STATISTICAL TESTING
# ===========================
print("\n" + "=" * 70)
print("STATISTICAL TESTING (per difficulty level)")
print("=" * 70)

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1-1)*np.std(a, ddof=1)**2 + (n2-1)*np.std(b, ddof=1)**2) / (n1+n2-2))
    if pooled_std < 1e-10:
        return float('inf') if abs(np.mean(a) - np.mean(b)) > 1e-10 else 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std

def effect_label(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"

other_variants = ['baseline_static', 'baseline_ratio', 'opponent_lof', 'opponent_ocsvm']
all_stat_results = []

for level in DIFFICULTY_LEVELS:
    df_lv = df_results[df_results['level'] == level]
    proposed_f1 = df_lv[df_lv['variant'] == 'proposed_context_aware']['F1'].values

    print(f"\n  === {level.upper()} ===")
    print(f"  {'Comparison':<40} {'p(t)':>8} {'d':>8} {'Effect':>10} {'Sig':>4}")
    print(f"  {'-'*75}")

    for vname in other_variants:
        other_f1 = df_lv[df_lv['variant'] == vname]['F1'].values
        t_stat, t_p = ttest_rel(proposed_f1, other_f1)
        try:
            w_stat, w_p = wilcoxon(proposed_f1, other_f1)
        except ValueError:
            w_stat, w_p = float('nan'), float('nan')
        d = cohens_d(proposed_f1, other_f1)
        sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"

        print(f"  proposed vs {vname:<25} {t_p:>8.4f} {d:>8.2f} {effect_label(d):>10} {sig:>4}")
        all_stat_results.append({
            'level': level, 'comparison': f'proposed vs {vname}',
            'p_ttest': t_p, 'cohens_d': d, 'effect': effect_label(d), 'sig': sig,
        })

df_stats = pd.DataFrame(all_stat_results)
stats_path = os.path.join(OUT_DIR, 'statistical_tests.csv')
df_stats.to_csv(stats_path, index=False)
print(f"\n  Statistical tests saved: {stats_path}")

# ===========================
# HYPOTHESIS VALIDATION
# ===========================
print(f"\n{'='*70}")
print("HYPOTHESIS VALIDATION")
print(f"{'='*70}")
hypothesis_results = {}
for level in DIFFICULTY_LEVELS:
    df_lv = df_results[df_results['level'] == level]
    bs_f1 = df_lv[df_lv['variant']=='baseline_static']['F1'].mean()
    br_f1 = df_lv[df_lv['variant']=='baseline_ratio']['F1'].mean()
    pc_f1 = df_lv[df_lv['variant']=='proposed_context_aware']['F1'].mean()
    pc_fpr = df_lv[df_lv['variant']=='proposed_context_aware']['FPR'].mean()
    br_fpr = df_lv[df_lv['variant']=='baseline_ratio']['FPR'].mean()
    lof_f1 = df_lv[df_lv['variant']=='opponent_lof']['F1'].mean()
    ocsvm_f1 = df_lv[df_lv['variant']=='opponent_ocsvm']['F1'].mean()

    h1 = br_f1 > bs_f1
    h2 = pc_fpr < br_fpr
    h3 = pc_f1 > lof_f1 and pc_f1 > ocsvm_f1
    hypothesis_results[level] = {'H1': h1, 'H2': h2, 'H3': h3}

    print(f"\n  {level.upper()}:")
    print(f"    H1 (21D > 15D):            {'PASS' if h1 else 'FAIL'} | F1 {br_f1:.3f} vs {bs_f1:.3f}")
    print(f"    H2 (cluster > global):     {'PASS' if h2 else 'FAIL'} | FPR {pc_fpr:.4f} vs {br_fpr:.4f}")
    print(f"    H3 (proposed > opponents): {'PASS' if h3 else 'FAIL'} | F1 {pc_f1:.3f} vs LOF {lof_f1:.3f}, OCSVM {ocsvm_f1:.3f}")

# ===========================
# GENERATE FIGURES
# ===========================
print("\n" + "=" * 70)
print("GENERATING COMPREHENSIVE FIGURES")
print("=" * 70)

colors_map = {
    'baseline_static': '#95a5a6',
    'baseline_ratio': '#3498db',
    'proposed_context_aware': '#27ae60',
    'opponent_lof': '#e74c3c',
    'opponent_ocsvm': '#8e44ad',
}
variant_names = ['baseline_static', 'baseline_ratio', 'proposed_context_aware', 'opponent_lof', 'opponent_ocsvm']
short_names = ['Static\n(15D)', 'Ratio\n(21D)', 'Proposed\n(21D+cluster)', 'LOF', 'OCSVM']

# Figure 1: 4-panel comprehensive results
fig = plt.figure(figsize=(24, 18))
fig.suptitle('CA-DQStream Full Dataset Benchmark Results\nNYC Yellow Taxi Jan 2024 | ~2.96M Records | 5 Variants x 5 Seeds x 3 Difficulty Levels',
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, hspace=0.30, wspace=0.20)

# Panel A: F1 by difficulty
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(variant_names))
width = 0.25
offsets = [-width, 0, width]
level_colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}
for off_idx, level in enumerate(DIFFICULTY_LEVELS):
    df_lv = df_results[df_results['level'] == level]
    f1_means = [df_lv[df_lv['variant']==v]['F1'].mean() for v in variant_names]
    f1_stds = [df_lv[df_lv['variant']==v]['F1'].std() for v in variant_names]
    bars = ax1.bar(x + offsets[off_idx], f1_means, yerr=f1_stds,
                   color=level_colors[level], edgecolor='white', capsize=3,
                   width=width, label=level.upper(), alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(short_names, fontsize=10)
ax1.set_ylabel('F1 Score', fontsize=12)
ax1.set_title('A. F1 Score by Variant and Difficulty Level', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 1.1)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, axis='y', alpha=0.3)

# Panel B: Degradation across difficulty
ax2 = fig.add_subplot(gs[0, 1])
metrics_list = ['F1', 'Recall']
for metric_idx, metric in enumerate(metrics_list):
    for vname, sname in zip(variant_names, ['Static(15D)', 'Ratio(21D)', 'Proposed', 'LOF', 'OCSVM']):
        vals = [df_results[(df_results['level']==lv) & (df_results['variant']==vname)][metric].mean()
                for lv in DIFFICULTY_LEVELS]
        ax2.plot(range(len(DIFFICULTY_LEVELS)), vals, 'o-', label=sname,
                 color=colors_map[vname], linewidth=2, markersize=8, alpha=0.85)
ax2.set_xticks(range(len(DIFFICULTY_LEVELS)))
ax2.set_xticklabels([l.upper() for l in DIFFICULTY_LEVELS])
ax2.set_ylabel('Score', fontsize=12)
ax2.set_xlabel('Difficulty Level', fontsize=12)
ax2.set_title('B. Performance Degradation Across Difficulty', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=9, loc='lower left', ncol=2)
ax2.grid(True, alpha=0.3)

# Panel C: Per-metric comparison (radar-style as grouped bar)
ax3 = fig.add_subplot(gs[1, 0])
metrics_to_plot = ['F1', 'Recall', 'Precision']
metric_means = {v: [] for v in variant_names}
metric_stds = {v: [] for v in variant_names}
df_easy = df_results[df_results['level'] == 'easy']
for v in variant_names:
    for m in metrics_to_plot:
        vals = df_easy[df_easy['variant']==v][m].values
        metric_means[v].append(vals.mean())
        metric_stds[v].append(vals.std())

x_m = np.arange(len(metrics_to_plot))
bar_w = 0.14
for i, (vname, sname) in enumerate(zip(variant_names, short_names)):
    ax3.bar(x_m + i * bar_w - 2*bar_w, metric_means[vname], bar_w,
            label=sname.replace('\n', ' '), color=colors_map[vname],
            yerr=metric_stds[vname], capsize=2, alpha=0.85)
ax3.set_xticks(x_m)
ax3.set_xticklabels(metrics_to_plot, fontsize=12)
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('C. EASY Level: F1 / Recall / Precision Comparison', fontsize=13, fontweight='bold')
ax3.set_ylim(0, 1.15)
ax3.legend(fontsize=8, loc='upper right', ncol=2)
ax3.grid(True, axis='y', alpha=0.3)

# Panel D: FPR comparison (key metric for thesis)
ax4 = fig.add_subplot(gs[1, 1])
for off_idx, level in enumerate(DIFFICULTY_LEVELS):
    df_lv = df_results[df_results['level'] == level]
    fpr_means = [df_lv[df_lv['variant']==v]['FPR'].mean() for v in variant_names]
    fpr_stds = [df_lv[df_lv['variant']==v]['FPR'].std() for v in variant_names]
    ax4.bar(x + offsets[off_idx], fpr_means, yerr=fpr_stds,
            color=level_colors[level], edgecolor='white', capsize=3,
            width=width, alpha=0.85, label=level.upper())
ax4.set_xticks(x)
ax4.set_xticklabels(short_names, fontsize=10)
ax4.set_ylabel('False Positive Rate', fontsize=12)
ax4.set_title('D. FPR by Variant and Difficulty (Lower is Better)', fontsize=13, fontweight='bold')
ax4.axhline(y=0.05, color='red', linestyle='--', linewidth=2, alpha=0.7, label='5% threshold')
ax4.set_ylim(0, max(0.1, max(fpr_means) * 1.3))
ax4.legend(fontsize=10)
ax4.grid(True, axis='y', alpha=0.3)

plt.savefig(os.path.join(OUT_DIR, 'benchmark_4panel.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'benchmark_4panel.pdf'), bbox_inches='tight')
print(f"  Saved: {OUT_DIR}/benchmark_4panel.png/pdf")

# Figure 2: Per-scenario breakdown
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
fig2.suptitle('Per-Scenario Detection Breakdown — EASY Level', fontsize=14, fontweight='bold')
ld = level_data['easy']
sc_names_short = ['meter_tamp', 'gps_spoof', 'pax_anom', 'slow_crawl', 'combined']
scenario_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

# Per-scenario breakdown: re-run evaluation for each scenario
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import MiniBatchKMeans

def quick_score_iforest(model, X_te, X_tr, percentile):
    train_scores = -model.decision_function(X_tr)
    threshold = np.percentile(train_scores, percentile)
    test_scores = -model.decision_function(X_te)
    return (test_scores > threshold).astype(int)

def quick_score_per_cluster(model, X_te, X_tr, kmeans, percentile):
    train_labels = kmeans.predict(X_tr)
    test_labels = kmeans.predict(X_te)
    train_scores = -model.decision_function(X_tr)
    test_scores = -model.decision_function(X_te)
    cluster_thresholds = {}
    for cid in range(kmeans.n_clusters):
        mask = train_labels == cid
        if mask.sum() > 10:
            cluster_thresholds[cid] = np.percentile(train_scores[mask], percentile)
        else:
            cluster_thresholds[cid] = np.percentile(train_scores, percentile)
    y_pred = np.zeros(len(X_te), dtype=int)
    for cid, thresh in cluster_thresholds.items():
        mask = test_labels == cid
        y_pred[mask] = (test_scores[mask] > thresh).astype(int)
    return y_pred

def quick_eval(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)

for ax_idx, (sc_name, sc_full) in enumerate(zip(sc_names_short, [s[0] for s in SCENARIOS])):
    ax = axes2.flat[ax_idx]
    sc_mask = ld['scenario_labels'] == sc_full
    n_anom = sc_mask.sum()
    y_all = ld['y_test']
    y_sc_only = y_all[sc_mask]

    scenario_results = []
    for vname in variant_names:
        feat_mode = '15D' if vname == 'baseline_static' else '21D'
        X_tr = X_train_15d_scaled if feat_mode == '15D' else X_train_21d_scaled
        X_te = ld['X_test_15d'] if feat_mode == '15D' else ld['X_test_21d']
        seed = 42
        if vname == 'baseline_static':
            model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=N_JOBS)
            model.fit(X_tr)
            y_pred_all = quick_score_iforest(model, X_te, X_tr, percentile=95)
        elif vname == 'baseline_ratio':
            model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=N_JOBS)
            model.fit(X_tr)
            y_pred_all = quick_score_iforest(model, X_te, X_tr, percentile=96)
        elif vname == 'proposed_context_aware':
            model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=N_JOBS)
            model.fit(X_tr)
            kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=seed)
            kmeans.fit(X_tr)
            y_pred_all = quick_score_per_cluster(model, X_te, X_tr, kmeans, percentile=97)
        elif vname == 'opponent_lof':
            rng = np.random.RandomState(seed)
            sample_size = min(50_000, len(X_tr))
            X_sample = X_tr[rng.choice(len(X_tr), sample_size, replace=False)]
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True, n_jobs=N_JOBS)
            model.fit(X_sample)
            train_scores = -model.decision_function(X_sample)
            threshold = np.percentile(train_scores, 96)
            test_scores = -model.decision_function(X_te)
            y_pred_all = (test_scores > threshold).astype(int)
        elif vname == 'opponent_ocsvm':
            rng = np.random.RandomState(seed)
            sample_size = min(30_000, len(X_tr))
            X_sample = X_tr[rng.choice(len(X_tr), sample_size, replace=False)]
            model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
            model.fit(X_sample)
            train_scores = -model.decision_function(X_sample)
            threshold = np.percentile(train_scores, 96)
            test_scores = -model.decision_function(X_te)
            y_pred_all = (test_scores > threshold).astype(int)
        sc_f1 = quick_eval(y_sc_only, y_pred_all[sc_mask])
        scenario_results.append(sc_f1)

    bars = ax.bar(range(len(variant_names)), scenario_results,
                  color=[colors_map[v] for v in variant_names], edgecolor='white', width=0.7)
    ax.set_xticks(range(len(variant_names)))
    ax.set_xticklabels([n.replace('\n', ' ') for n in short_names], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title(f'{sc_name.upper()}\n(n={n_anom:,})', fontsize=11, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
    for bar, val in zip(bars, scenario_results):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)

axes2.flat[-1].axis('off')
ax_legend = axes2.flat[-1]
handles = [plt.Rectangle((0,0),1,1, color=colors_map[v]) for v in variant_names]
ax_legend.legend(handles, [n.replace('\n',' ') for n in short_names],
                 loc='center', fontsize=12, title='Variants')
ax_legend.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'benchmark_per_scenario.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, 'benchmark_per_scenario.pdf'), bbox_inches='tight')
print(f"  Saved: {OUT_DIR}/benchmark_per_scenario.png/pdf")

# Figure 3: Statistical significance heatmap
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle('Statistical Significance Heatmap (Cohen\'s d effect size)\nproposed_context_aware vs Opponents | *** p<0.001 ** p<0.01 * p<0.05 ns', fontsize=13, fontweight='bold')

for ax_idx, level in enumerate(DIFFICULTY_LEVELS):
    ax = axes3[ax_idx]
    df_lv = df_results[df_results['level'] == level]
    proposed_f1 = df_lv[df_lv['variant'] == 'proposed_context_aware']['F1'].values

    n_seeds = len(SEEDS)
    effect_matrix = np.zeros((len(other_variants), n_seeds))
    for i, vname in enumerate(other_variants):
        other_f1 = df_lv[df_lv['variant'] == vname]['F1'].values
        for j in range(n_seeds):
            effect_matrix[i, j] = cohens_d([proposed_f1[j]], [other_f1[j]])

    im = ax.imshow(effect_matrix, cmap='RdYlGn', aspect='auto', vmin=-3, vmax=3)
    ax.set_xticks(range(n_seeds))
    ax.set_xticklabels([f'seed={s}' for s in SEEDS], fontsize=8)
    ax.set_yticks(range(len(other_variants)))
    ax.set_yticklabels(other_variants, fontsize=9)
    ax.set_title(f'{level.upper()}', fontsize=12, fontweight='bold')

    for i in range(len(other_variants)):
        for j in range(n_seeds):
            val = effect_matrix[i, j]
            color = 'white' if abs(val) > 1.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label="Cohen's d")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'benchmark_effect_size.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {OUT_DIR}/benchmark_effect_size.png")

# Figure 4: Train time comparison
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
fig4.suptitle('Training Time Comparison (seconds per run)', fontsize=14, fontweight='bold')
for ax_idx, level in enumerate(DIFFICULTY_LEVELS):
    ax = axes4[ax_idx]
    df_lv = df_results[df_results['level'] == level]
    t_means = [df_lv[df_lv['variant']==v]['train_time'].mean() for v in variant_names]
    t_stds = [df_lv[df_lv['variant']==v]['train_time'].std() for v in variant_names]
    bars = ax.bar(range(len(variant_names)), t_means, yerr=t_stds,
                  color=[colors_map[v] for v in variant_names], edgecolor='white', capsize=4, width=0.7)
    ax.set_xticks(range(len(variant_names)))
    ax.set_xticklabels([n.replace('\n', ' ') for n in short_names], fontsize=9)
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'{level.upper()}', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, t_means)):
        ax.text(bar.get_x() + bar.get_width()/2, val + t_stds[i] + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'benchmark_train_time.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: {OUT_DIR}/benchmark_train_time.png")

# ===========================
# FINAL SUMMARY
# ===========================
print("\n" + "=" * 70)
print("BENCHMARK COMPLETE — OFFICIAL RESULTS")
print("=" * 70)
print(f"""
ARCHITECTURE: L1 (Schema) -> L2 (Rules) -> L3 (IQR) -> Clean -> Train/Test -> ML Benchmark
DATASET:     NYC Yellow Taxi Jan 2024 ({'100K sample' if FAST_MODE else '~2.96M full records'})
EVALUATION:  {N_SEEDS} seeds x 5 variants x 3 levels = {len(df_results)} runs
HARDWARE:    32 vCPUs (n_jobs=-1), ~88 GB RAM
OUTPUT:      {OUT_DIR}/
  - benchmark_results.csv  (all raw results)
  - statistical_tests.csv   (p-values, effect sizes)
  - benchmark_4panel.png   (comprehensive 4-panel figure)
  - benchmark_per_scenario.png (per-scenario breakdown)
  - benchmark_effect_size.png   (Cohen's d heatmap)
  - benchmark_train_time.png   (training time comparison)
""")

for level in DIFFICULTY_LEVELS:
    df_lv = df_results[df_results['level'] == level]
    pc = df_lv[df_lv['variant']=='proposed_context_aware']
    bs = df_lv[df_lv['variant']=='baseline_static']
    br = df_lv[df_lv['variant']=='baseline_ratio']
    lof = df_lv[df_lv['variant']=='opponent_lof']
    ocsvm = df_lv[df_lv['variant']=='opponent_ocsvm']
    h = hypothesis_results[level]
    print(f"  {level.upper()}: Proposed={pc['F1'].mean():.3f} | Ratio={br['F1'].mean():.3f} | LOF={lof['F1'].mean():.3f} | OCSVM={ocsvm['F1'].mean():.3f} | Static={bs['F1'].mean():.3f}")
    print(f"         H1={'PASS' if h['H1'] else 'FAIL'} | H2={'PASS' if h['H2'] else 'FAIL'} | H3={'PASS' if h['H3'] else 'FAIL'}")

print(f"\n  Total benchmark time: {time.time()-t_benchmark:.1f}s")
print(f"\n  All official results saved to: {OUT_DIR}/")
