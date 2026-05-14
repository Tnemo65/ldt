"""
DataAgent — loads, cleans, and caches all 12 months of NYC taxi data.
Outputs per-fold train/test feature matrices saved as .npz files.
Runs in the main process (I/O-bound). Produces work for all other agents.
"""
from __future__ import annotations

import time
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
OUT_DIR  = Path(__file__).parent.parent / 'results' / 'v4_janjun' / 'features'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS        = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]
DIFFICULTIES = ['easy', 'medium', 'hard']
ANOMALY_PARAMS = {
    'easy':   {'meter_mult': (10, 20), 'speed': (50, 95),   'pax_fare': (40, 70),  'crawl_dur': (90, 180)},
    'medium': {'meter_mult': (4, 8),   'speed': (30, 60),  'pax_fare': (15, 30),  'crawl_dur': (40, 80)},
    'hard':   {'meter_mult': (1.5, 3), 'speed': (20, 40),  'pax_fare': (8, 15),   'crawl_dur': (20, 35)},
}
TRAIN_SUBSAMPLE = 80_000  # per fold
TEST_SUBSAMPLE  = 100_000  # per test month


def load_month(year: int, month: int) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f'yellow_tripdata_{year:04d}-{month:02d}.parquet')


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=['PULocationID', 'DOLocationID', 'fare_amount',
                            'trip_distance', 'passenger_count'])
    for col in ['PULocationID', 'DOLocationID', 'fare_amount',
                'trip_distance', 'passenger_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    df = df[(df['PULocationID'] >= 1) & (df['PULocationID'] <= 263)]
    df = df[(df['DOLocationID'] >= 1) & (df['DOLocationID'] <= 263)]
    df['fare_amount']    = df['fare_amount'].abs()
    df['trip_distance']  = df['trip_distance'].abs()

    pickup  = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    dropoff  = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df['duration_s'] = (dropoff - pickup).dt.total_seconds()
    df = df[(df['duration_s'] > 0) & (df['duration_s'] < 86400)]
    df['speed_mph'] = df['trip_distance'] / (df['duration_s'] / 3600)
    df = df[(df['speed_mph'] > 0) & (df['speed_mph'] < 100)]

    for col in ['fare_amount', 'trip_distance', 'duration_s']:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df = df[(df[col] >= lo) & (df[col] <= hi)]

    df['dur_min']  = df['duration_s'] / 60.0
    df['total_amt'] = df.get('total_amount', df['fare_amount'])
    return df.reset_index(drop=True)


def extract_features(df: pd.DataFrame, ablation: str = 'treatment') -> np.ndarray:
    """
    ablation='treatment': full 25D
    ablation='control':   raw 15D
    """
    n  = len(df)
    nf = 25 if ablation == 'treatment' else 15
    X  = np.zeros((n, nf), dtype=np.float32)

    pickup = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    hour   = pickup.dt.hour.fillna(12).astype(int).values
    dow    = pickup.dt.dayofweek.fillna(0).astype(int).values

    dist = df['trip_distance'].fillna(0).values
    dur  = df['dur_min'].fillna(1).values
    fare = df['fare_amount'].fillna(0).values
    pax  = df['passenger_count'].fillna(1).values
    spd  = df['speed_mph'].fillna(0).values

    X[:, 0] = dist
    X[:, 1] = dur
    X[:, 2] = fare
    X[:, 3] = pax
    X[:, 4] = df['total_amt'].fillna(0).values
    X[:, 5] = spd
    X[:, 6] = fare / np.maximum(dist, 0.01)
    X[:, 7] = fare / np.maximum(dur,  0.01)
    X[:, 8] = fare / np.maximum(pax,  0.01)

    X[:, 9]  = hour.astype(np.float32)
    X[:, 10] = dow.astype(np.float32)
    X[:, 11] = (dow >= 5).astype(np.float32)
    X[:, 12] = (((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 20))).astype(np.float32)
    X[:, 13] = ((hour >= 20) | (hour <= 6)).astype(np.float32)
    X[:, 14] = pickup.dt.month.fillna(1).astype(np.float32)

    if ablation == 'treatment':
        X[:, 15] = X[:, 6] / 2.5
        X[:, 16] = X[:, 7] / 0.67
        X[:, 17] = spd / 12.0
        X[:, 18] = pax / np.maximum(dist, 0.01)
        X[:, 19] = fare * dist
        X[:, 20] = dur / np.maximum(dist, 0.01)
        X[:, 21] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        X[:, 22] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
        X[:, 23] = np.sin(2 * np.pi * dow  / 7).astype(np.float32)
        X[:, 24] = np.cos(2 * np.pi * dow  / 7).astype(np.float32)

    return np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)


def inject_anomalies(df: pd.DataFrame, n_per: int, diff: str,
                     seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Inject 5 fraud scenarios into df, return df+labels."""
    rng    = np.random.RandomState(seed)
    df     = df.copy().reset_index(drop=True)
    labels = np.zeros(len(df), dtype=int)
    p      = ANOMALY_PARAMS[diff]

    for sname in ['meter_tampering', 'gps_spoofing', 'passenger_anomaly',
                  'slow_crawl', 'combined_subtle']:
        recs = []
        for i in range(n_per):
            r = df.iloc[i % len(df)].copy()
            if sname == 'meter_tampering':
                r['fare_amount'] = r['trip_distance'] * 2.5 * rng.uniform(*p['meter_mult'])
            elif sname == 'gps_spoofing':
                sp = rng.uniform(*p['speed'])
                r['trip_distance'] = r['dur_min'] * sp / 60.0
                r['fare_amount']   = r['trip_distance'] * 2.5
            elif sname == 'passenger_anomaly':
                r['trip_distance'] = rng.uniform(0.1, 1.5)
                r['fare_amount']   = rng.uniform(*p['pax_fare'])
            elif sname == 'slow_crawl':
                r['dur_min']       = rng.uniform(*p['crawl_dur'])
                r['fare_amount']   = rng.uniform(*p['pax_fare'])
                r['trip_distance'] = rng.uniform(0.5, 3.0)
                if r['dur_min'] > 0:
                    r['speed_mph'] = r['trip_distance'] / (r['dur_min'] / 60)
            elif sname == 'combined_subtle':
                mult = rng.uniform(1.2, 2.0)
                r['fare_amount']   = r['fare_amount'] * mult
                r['trip_distance'] = r['trip_distance'] * rng.uniform(0.8, 1.2)
                r['dur_min']       = r['dur_min'] * rng.uniform(0.9, 1.1)
            recs.append(r)

        anom   = pd.DataFrame(recs)
        df     = pd.concat([df, anom], ignore_index=True)
        labels = np.append(labels, np.ones(len(recs), dtype=int))

    return df, labels


class DataAgent:
    """
    Loads all 12 months, builds per-fold train/test matrices,
    injects anomalies for each difficulty level, saves .npz bundles.
    Other agents load these bundles directly (zero re-processing).
    """

    def __init__(self, base_dir: Path | None = None, out_dir: Path | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.out_dir  = Path(out_dir)  if out_dir  else OUT_DIR
        self.cache_dir = self.out_dir / 'fold_data'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        t0 = time.perf_counter()
        print('\n[DataAgent] Loading and preprocessing Jan-Jun 2024...')

        monthly_dfs = []
        for m in range(1, 7):  # Jan-Jun 2024
            df = clean(load_month(2024, m))
            monthly_dfs.append(df)
            print(f'  Month {m:02d}: {len(df):,} records after cleaning')

        print('\n[DataAgent] Building per-fold bundles...')
        bundles = []

        for fold_idx in range(5):  # folds 0-4 (train Jan..May, test Jun)
            train_dfs  = monthly_dfs[:fold_idx + 1]  # accumulate Jan..month-1

            # Subsample test BEFORE feature extraction to bound memory
            test_df = monthly_dfs[fold_idx + 1].copy()
            if len(test_df) > TEST_SUBSAMPLE:
                rng_test = np.random.RandomState(42 + fold_idx)
                test_df = test_df.sample(n=TEST_SUBSAMPLE, random_state=rng_test)

            X_test_raw  = extract_features(test_df, 'treatment')

            # Always have at least 1 month of training data (fold 0 = Jan)
            if fold_idx == 0:
                # First fold: train on Jan, test on Feb
                X_train_raw = extract_features(train_dfs[0], 'treatment')
            else:
                X_train_raw = extract_features(pd.concat(train_dfs, ignore_index=True), 'treatment')

            # Subsample train to keep memory bounded
            if len(X_train_raw) > TRAIN_SUBSAMPLE:
                rng = np.random.RandomState(42 + fold_idx)
                idx = rng.choice(len(X_train_raw), TRAIN_SUBSAMPLE, replace=False)
                X_train_sub = X_train_raw[idx]
            else:
                X_train_sub = X_train_raw

            # Fit scaler on train
            scaler = StandardScaler()
            scaler.fit(X_train_sub)
            X_train_scaled = scaler.transform(X_train_sub).astype(np.float32)
            X_test_scaled  = scaler.transform(X_test_raw).astype(np.float32)

            # ── Anomaly injection per difficulty ───────────────────
            # Control: extract 15D features from the same subsampled training data
            train_ctrl_raw = extract_features(
                pd.concat(train_dfs, ignore_index=True), 'control')
            if len(train_ctrl_raw) > TRAIN_SUBSAMPLE:
                rng_ctrl = np.random.RandomState(42 + fold_idx)
                train_ctrl_sub = train_ctrl_raw[rng_ctrl.choice(len(train_ctrl_raw), TRAIN_SUBSAMPLE, replace=False)]
            else:
                train_ctrl_sub = train_ctrl_raw
            scaler_ctrl = StandardScaler()
            scaler_ctrl.fit(train_ctrl_sub)

            for diff_idx, diff in enumerate(DIFFICULTIES):
                seed_idx = (fold_idx * 3 + diff_idx) % len(SEEDS)
                df_inj, y_labels = inject_anomalies(
                    test_df, 1000, diff, SEEDS[seed_idx])

                X_test_inj_raw = extract_features(df_inj, 'treatment')
                X_test_inj = scaler.transform(X_test_inj_raw).astype(np.float32)
                y_labels = np.array(y_labels, dtype=np.int32)

                test_month = fold_idx + 2  # calendar month (2=Feb..12=Dec)
                bundle = {
                    'fold': fold_idx,
                    'month': test_month,
                    'difficulty': diff,
                    'seed_idx': seed_idx,
                    'X_train': X_train_scaled,
                    'X_test': X_test_scaled,       # clean test (for ablation)
                    'X_test_inj': X_test_inj,      # injected test
                    'y_test': y_labels,
                    'n_train': len(X_train_scaled),
                    'n_test': len(X_test_inj),
                    'n_anomalies': int(y_labels.sum()),
                }

                # Control features (15D) — same test data, control scaler
                X_test_ctrl_raw = extract_features(df_inj, 'control')
                bundle['X_train_ctrl'] = scaler_ctrl.transform(train_ctrl_sub).astype(np.float32)
                bundle['X_test_inj_ctrl'] = scaler_ctrl.transform(X_test_ctrl_raw).astype(np.float32)

                path = self.cache_dir / f'fold{fold_idx:02d}_{diff}.npz'
                np.savez_compressed(
                    path,
                    X_train=bundle['X_train'],
                    X_test=bundle['X_test'],
                    X_test_inj=bundle['X_test_inj'],
                    y_test=bundle['y_test'],
                    X_train_ctrl=bundle['X_train_ctrl'],
                    X_test_inj_ctrl=bundle['X_test_inj_ctrl'],
                    fold=np.array(bundle['fold']),
                    month=np.array(bundle['month']),
                    difficulty=bundle['difficulty'],
                    seed_idx=np.array(bundle['seed_idx']),
                    n_train=np.array(bundle['n_train']),
                    n_test=np.array(bundle['n_test']),
                    n_anomalies=np.array(bundle['n_anomalies']),
                )
                bundles.append(path)

            print(f'  Fold {fold_idx+1:02d} (train Jan..{(fold_idx+1):02d}, '
                  f'test {(fold_idx+2):02d}): '
                  f'{len(X_train_scaled):,} train, '
                  f'{len(X_test_inj):,} test '
                  f'({y_labels.sum():,} injected anomalies)')

        elapsed = time.perf_counter() - t0
        print(f'\n[DataAgent] Done. {len(bundles)} bundles saved in {elapsed:.1f}s')
        print(f'  Cache dir: {self.cache_dir}')
        return {'bundles': len(bundles), 'elapsed_s': elapsed}
