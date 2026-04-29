"""
Step 4: ADWIN Drift Detection on Quality Meta-Streams
====================================================
Contribution 3: Self-monitoring the monitor.

Uses KNN anomaly scores from clean_benchmark.py (best model).
Simulates streaming: 4 windows per month (weekly chunks).

ADWIN (Adaptive Windowing) - Bifet & Gavalda (2007):
  Maintains variable-size window, detects when two sub-windows
  have significantly different means (concept drift).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import math
from dataclasses import dataclass, field

import pipeline_utils as pu

DATA_DIR = Path('d:/final/data/raw')
OUTPUT_DIR = Path('d:/final/output')


# ─────────────────────────────────────────────────────────────────────────────
# ADWIN Implementation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ADWIN:
    """
    ADWIN: Adaptive Windowing for drift detection.

    Detects concept drift by maintaining a sliding window and checking
    whether two sub-windows have significantly different means.

    Cut criterion: |mean_W0 - mean_W1| > sqrt((1/2m) * ln(4m/delta))
    where m = harmonic mean of sub-window sizes.
    """
    delta: float = 0.002
    min_window: int = 30

    _window: list = field(default_factory=list)
    _total: float = field(default=0.0)
    drift_count: int = 0
    drift_at: list = field(default_factory=list)

    def add(self, value: float) -> bool:
        if math.isnan(value) or math.isinf(value):
            return False

        self._window.append(value)
        self._total += value
        n = len(self._window)

        if n < self.min_window:
            return False

        drift = self._check_cut()
        if drift:
            self.drift_count += 1
            self.drift_at.append(n)
        return drift

    def _check_cut(self) -> bool:
        """Check if any cut point shows drift. If yes, cut."""
        n = len(self._window)
        win = np.array(self._window)

        for cut in range(self.min_window, n - self.min_window + 1):
            w0, w1 = win[:cut], win[cut:]
            n0, n1 = len(w0), len(w1)
            if n0 < 2 or n1 < 2:
                continue

            m0, m1 = w0.mean(), w1.mean()
            m = (n0 * n1) / (n0 + n1)
            eps = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 * n / self.delta))

            if abs(m0 - m1) > eps:
                self._window = self._window[cut:]
                self._total = sum(self._window)
                return True
        return False

    @property
    def mean(self) -> float:
        return self._total / len(self._window) if self._window else 0.0

    def size(self) -> int:
        return len(self._window)

    def reset(self):
        self._window = []
        self._total = 0.0
        self.drift_count = 0
        self.drift_at = []


# ─────────────────────────────────────────────────────────────────────────────
# Stream simulator: chunk files into windows
# ─────────────────────────────────────────────────────────────────────────────

def add_window_col(df: pd.DataFrame, n_windows: int = 4, label: str = '') -> pd.DataFrame:
    """Add window_col by splitting into equal chunks."""
    df = df.sort_values('tpep_pickup_datetime').reset_index(drop=True)
    chunk_size = len(df) // n_windows
    labels = [f'{label}_W{i+1}' for i in range(n_windows)]
    df['window'] = ''
    for i in range(n_windows):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_windows - 1 else len(df)
        df.iloc[start:end, df.columns.get_loc('window')] = labels[i]
    return df


def compute_window_metrics(df: pd.DataFrame) -> dict:
    """Compute meta-stream metrics for one window."""
    n = len(df)
    df = df.copy()
    df['is_neg'] = (df['fare_amount'] < 0).astype(int)
    df['is_zero_dist'] = (df['trip_distance'] <= 0).astype(int)
    df['is_null_pax'] = df['passenger_count'].isna().astype(int)
    df['is_null_rc'] = df['RatecodeID'].isna().astype(int)

    # Use rule-based is_anomaly (from Step 4b)
    any_viol = (
        df['is_anomaly'].astype(bool) if 'is_anomaly' in df.columns else
        df['is_neg'] | df['is_zero_dist'] | df['is_null_pax'] | df['is_null_rc']
    )

    return {
        'n_records': n,
        'null_rate_pax': df['is_null_pax'].mean(),
        'null_rate_rc': df['is_null_rc'].mean(),
        'violation_rate': any_viol.astype(int).mean(),
        'neg_rate': df['is_neg'].mean(),
        'zero_dist_rate': df['is_zero_dist'].mean(),
        'fare_mean': df['fare_amount'].mean(),
        'fare_std': df['fare_amount'].std(),
        'fare_p50': df['fare_amount'].median(),
        'fare_p5': df['fare_amount'].quantile(0.05),
        'fare_p95': df['fare_amount'].quantile(0.95),
        'dist_mean': df['trip_distance'].mean(),
        'dist_std': df['trip_distance'].std(),
        'total_mean': df['total_amount'].mean(),
        'total_std': df['total_amount'].std(),
        'anomaly_rate': df['is_anomaly'].mean() if 'is_anomaly' in df.columns else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT MONITOR
# ─────────────────────────────────────────────────────────────────────────────

class MetaStreamMonitor:
    """ADWIN-based monitor for quality meta-streams."""

    def __init__(self, delta: float = 0.002, min_window: int = 3):
        self.delta = delta
        self.min_window = min_window
        self.adwins: dict[str, ADWIN] = {}
        self.baseline: dict[str, dict] = {}
        self.test_stream: list[dict] = []
        self.drift_events: list[dict] = []

    def fit_baseline(self, window_metrics: list[dict], metric_keys: list[str]):
        """Build baseline from training windows."""
        for mk in metric_keys:
            adwin = ADWIN(delta=self.delta, min_window=self.min_window)
            for wm in window_metrics:
                v = wm.get(mk)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    adwin.add(float(v))
            self.adwins[mk] = adwin
            vals = [wm[mk] for wm in window_metrics if mk in wm
                   and wm[mk] is not None and not np.isnan(wm[mk])]
            if vals:
                self.baseline[mk] = {
                    'mean': np.mean(vals), 'std': np.std(vals),
                    'min': np.min(vals), 'max': np.max(vals),
                }

    def score_stream(self, window_metrics: list[dict], metric_keys: list[str]) -> list[dict]:
        """Score test windows, detect drift via ADWIN."""
        events = []
        for wm in window_metrics:
            self.test_stream.append(wm)
            for mk in metric_keys:
                v = wm.get(mk)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                if mk not in self.adwins:
                    self.adwins[mk] = ADWIN(delta=self.delta, min_window=self.min_window)

                drift = self.adwins[mk].add(float(v))
                if drift:
                    bl_mean = self.baseline.get(mk, {}).get('mean', 0) or 0
                    mag = (float(v) - bl_mean) / (abs(bl_mean) + 1e-9)
                    events.append({
                        'window': wm['window'],
                        'metric': mk,
                        'value': float(v),
                        'baseline_mean': bl_mean,
                        'adwin_mean_after_cut': self.adwins[mk].mean,
                        'drift_magnitude': mag,
                        'direction': 'increase' if mag > 0 else 'decrease',
                        'severity': (
                            'HIGH' if abs(mag) > 0.30 else
                            'MEDIUM' if abs(mag) > 0.15 else 'LOW'
                        ),
                        'adwin_size': self.adwins[mk].size(),
                    })
        self.drift_events = events
        return events


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_step4():
    print("=" * 70)
    print("STEP 4: ADWIN Drift Detection on Quality Meta-Streams")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[4a] Loading data")
    print("-" * 70)

    df_jan = pd.read_parquet(DATA_DIR / 'yellow_tripdata_2024-01.parquet')
    df_jul = pd.read_parquet(DATA_DIR / 'yellow_tripdata_2024-07.parquet')

    with open(OUTPUT_DIR / 'baseline_stats.json') as f:
        baseline = json.load(f)

    print(f"Jan 2024: {len(df_jan):,} records")
    print(f"Jul 2024: {len(df_jul):,} records")

    # ── Step 4b: Context + rule-based anomaly labels ────────────────────────────
    print("\n[4b] Computing rule-based anomaly labels (ground truth)")
    print("-" * 70)

    t0 = time.time()

    for df in [df_jan, df_jul]:
        n = len(df)
        hours = df['tpep_pickup_datetime'].dt.hour
        ratecodes = df['RatecodeID']
        time_bin = pd.cut(hours, bins=[-1, 5, 11, 17, 24],
            labels=['night','morning','afternoon','evening']).astype(str)
        ratecode_bin = np.where(ratecodes == 1, 'standard', 'special')
        df['context'] = time_bin + '_' + ratecode_bin

        df['negative_fare'] = (df['fare_amount'] < 0).astype(int)
        df['zero_distance'] = (df['trip_distance'] <= 0).astype(int)
        df['null_passenger'] = df['passenger_count'].isna().astype(int)
        df['null_ratecode'] = df['RatecodeID'].isna().astype(int)

        ctx_arr = df['context'].values
        fare_dev = np.zeros(n, dtype=float)
        dist_dev = np.zeros(n, dtype=float)
        total_dev = np.zeros(n, dtype=float)

        for ctx, b in baseline.items():
            mask = (ctx_arr == ctx)
            if mask.sum() == 0:
                continue
            for col, arr in [('fare_amount', fare_dev), ('trip_distance', dist_dev),
                              ('total_amount', total_dev)]:
                if col not in b:
                    continue
                s = b[col]
                if pd.isna(s.get('std')) or s['std'] < 0.01:
                    continue
                vals = df.loc[mask, col].fillna(s['mean']).values
                z = np.abs(vals - s['mean']) / (s['std'] + 1e-9)
                arr[mask] = np.clip(z, 0, 10)

        dropoff = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
        pickup  = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
        dur_h = (dropoff - pickup).dt.total_seconds().fillna(0) / 3600
        dur_h = dur_h.clip(lower=0, upper=8)
        speed = np.where(dur_h.values > 0, df['trip_distance'].values / dur_h.values, 0)

        df['is_anomaly'] = (
            (df['negative_fare'] > 0) | (df['zero_distance'] > 0) |
            (df['null_passenger'] > 0) | (df['null_ratecode'] > 0) |
            (fare_dev > 3) | (dist_dev > 3) | (total_dev > 3) | (speed > 100)
        ).astype(int)

    print(f"  Jan 2024: anomaly_rate={df_jan['is_anomaly'].mean()*100:.2f}%")
    print(f"  Jul 2024: anomaly_rate={df_jul['is_anomaly'].mean()*100:.2f}%")

    # ── Step 4c: Create windows (streaming simulation) ─────────────────────────
    print("\n[4c] Creating temporal windows (streaming simulation)")
    print("-" * 70)

    N_WIN = 4

    # Sort and add window label
    # Sample Jan for faster processing (use all for actual production)
    SAMPLE = 500_000
    df_jan_sample = df_jan.sample(n=min(SAMPLE, len(df_jan)), random_state=42).copy()

    df_jan_sample = df_jan_sample.sort_values('tpep_pickup_datetime').reset_index(drop=True)
    jan_n = len(df_jan_sample)
    jan_chunk = jan_n // N_WIN

    jan_windows = []
    for i in range(N_WIN):
        start = i * jan_chunk
        end = (i + 1) * jan_chunk if i < N_WIN - 1 else jan_n
        w = df_jan_sample.iloc[start:end].copy()
        w['window'] = f'Jan_W{i+1}'
        jan_windows.append(w)

    df_jul_s = df_jul.sort_values('tpep_pickup_datetime').reset_index(drop=True)
    jul_n = len(df_jul_s)
    jul_chunk = jul_n // N_WIN

    jul_windows = []
    for i in range(N_WIN):
        start = i * jul_chunk
        end = (i + 1) * jul_chunk if i < N_WIN - 1 else jul_n
        w = df_jul_s.iloc[start:end].copy()
        w['window'] = f'Jul_W{i+1}'
        jul_windows.append(w)

    del df_jan_sample
    import gc; gc.collect()

    print(f"Jan: {N_WIN} windows (~{jan_chunk:,} records each)")
    print(f"Jul: {N_WIN} windows (~{jul_chunk:,} records each)")

    # ── Step 4e: Compute quality meta-stream ───────────────────────────────────
    print("\n[4e] Computing quality meta-stream per window")
    print("-" * 70)

    all_windows = jan_windows + jul_windows

    window_metrics = []
    for w in all_windows:
        m = compute_window_metrics(w)
        m['window'] = w['window'].iloc[0]
        m['period'] = 'train' if 'Jan' in m['window'] else 'test'
        window_metrics.append(m)

    meta_df = pd.DataFrame(window_metrics)

    print("\nQuality Meta-Stream:")
    show_cols = ['window', 'period', 'n_records', 'null_rate_pax',
                  'violation_rate', 'anomaly_rate', 'fare_mean']
    print(meta_df[show_cols].to_string(index=False))

    # ── Step 4f: ADWIN drift detection ─────────────────────────────────────────
    print("\n[4f] ADWIN Drift Detection")
    print("-" * 70)

    metric_keys = [
        'null_rate_pax', 'null_rate_rc', 'violation_rate',
        'anomaly_rate', 'fare_mean', 'dist_mean', 'total_mean',
    ]

    train_metrics = [m for m in window_metrics if m['period'] == 'train']
    test_metrics = [m for m in window_metrics if m['period'] == 'test']

    # Test different delta values
    print("\n  Sensitivity analysis:")
    print(f"  {'delta':<10} {'Label':<10} {'Drift Events':>14} {'Total Mag':>12}")
    print(f"  {'-'*50}")

    results = {}
    for delta_val, label in [(0.01, 'relaxed'), (0.002, 'default'), (0.0001, 'strict')]:
        monitor = MetaStreamMonitor(delta=delta_val, min_window=2)
        monitor.fit_baseline(train_metrics, metric_keys)
        events = monitor.score_stream(test_metrics, metric_keys)

        total_mag = sum(abs(e['drift_magnitude']) for e in events)
        results[delta_val] = {'monitor': monitor, 'events': events}
        print(f"  {delta_val:<10.4f} {label:<10} {len(events):>14} {total_mag:>11.1%}")

    # Default monitor
    monitor = results[0.002]['monitor']
    events = results[0.002]['events']

    print(f"\n  Drift events at delta=0.002 (default):")
    print(f"  {'Window':<12} {'Metric':<25} {'Value':>8} {'Baseline':>8} {'Mag':>8} {'Dir':<9} {'Severity'}")
    print(f"  {'-'*85}")

    if not events:
        print(f"  {'(no drift detected)':<12}")
        print("\n  NOTE: With only 4 test windows, ADWIN needs min_window=2.")
        print("  Running 8-window fine-grained analysis...")

        # Fine-grained: 8 windows per month
        N_WIN_FINE = 8
        jan_fine = []
        jan_n_full = len(df_jan)
        jf_chunk = jan_n_full // N_WIN_FINE
        df_jan_sorted = df_jan.sort_values('tpep_pickup_datetime').reset_index(drop=True)
        for i in range(N_WIN_FINE):
            s, e = i * jf_chunk, (i + 1) * jf_chunk if i < N_WIN_FINE - 1 else jan_n_full
            w = df_jan_sorted.iloc[s:e].copy()
            w['window'] = f'Jan_W{i+1}'
            jan_fine.append(w)

        jul_fine = []
        jul_n_full = len(df_jul)
        jlf_chunk = jul_n_full // N_WIN_FINE
        df_jul_s = df_jul.sort_values('tpep_pickup_datetime').reset_index(drop=True)
        for i in range(N_WIN_FINE):
            s, e = i * jlf_chunk, (i + 1) * jlf_chunk if i < N_WIN_FINE - 1 else jul_n_full
            w = df_jul_s.iloc[s:e].copy()
            w['window'] = f'Jul_W{i+1}'
            jul_fine.append(w)

        fine_metrics = []
        for w in jan_fine + jul_fine:
            m = compute_window_metrics(w)
            m['window'] = w['window'].iloc[0]
            m['period'] = 'train' if 'Jan' in m['window'] else 'test'
            fine_metrics.append(m)

        fine_train = [m for m in fine_metrics if m['period'] == 'train']
        fine_test = [m for m in fine_metrics if m['period'] == 'test']

        monitor2 = MetaStreamMonitor(delta=0.002, min_window=4)
        monitor2.fit_baseline(fine_train, metric_keys)
        events2 = monitor2.score_stream(fine_test, metric_keys)

        print(f"\n  Fine-grained results (8 windows):")
        print(f"  {'Window':<12} {'Metric':<25} {'Value':>8} {'Baseline':>8} {'Mag':>8} {'Dir':<9} {'Severity'}")
        print(f"  {'-'*85}")

        if not events2:
            print(f"  {'(no drift in 8-window either)':<12}")
            print("\n  INTERPRETATION:")
            print("  Jan->Jul comparison shows STABLE quality meta-streams.")
            print("  This is expected for monthly data from the same dataset.")
            print("  In production: drift would appear at daily/weekly granularity.")
        else:
            for e in events2:
                print(f"  {e['window']:<12} {e['metric']:<25} {e['value']:>8.4f} "
                      f"{e['baseline_mean']:>8.4f} {e['drift_magnitude']:>+7.1%} "
                      f"{e['direction']:<9} {e['severity']}")

    else:
        for e in events:
            print(f"  {e['window']:<12} {e['metric']:<25} {e['value']:>8.4f} "
                  f"{e['baseline_mean']:>8.4f} {e['drift_magnitude']:>+7.1%} "
                  f"{e['direction']:<9} {e['severity']}")

    # ── Step 4g: Context-level drift ───────────────────────────────────────────
    print("\n[4g] Context-level drift analysis")
    print("-" * 70)

    # Per-context violation rates across windows
    ctx_metrics = {}
    for w in all_windows:
        win_name = w['window'].iloc[0]
        period = 'train' if 'Jan' in win_name else 'test'
        for ctx in sorted(w['context'].unique()):
            ctx_g = w[w['context'] == ctx]
            viol_rate = ctx_g['is_anomaly'].mean() if 'is_anomaly' in ctx_g.columns else 0.0
            neg_rate = ctx_g['negative_fare'].mean() if 'negative_fare' in ctx_g.columns else 0.0
            key = f"{ctx}"
            if key not in ctx_metrics:
                ctx_metrics[key] = {'train': [], 'test': []}
            if period == 'train':
                ctx_metrics[key]['train'].append({'window': win_name, 'viol_rate': viol_rate, 'neg_rate': neg_rate})
            else:
                ctx_metrics[key]['test'].append({'window': win_name, 'viol_rate': viol_rate, 'neg_rate': neg_rate})

    print(f"\n  Context-level violation rate comparison (train vs test):")
    print(f"  {'Context':<22} {'Train Viol':>10} {'Test Viol':>10} {'Change':>10} {'Dir'}")
    print(f"  {'-'*65}")

    ctx_drift_summary = []
    for ctx in sorted(ctx_metrics.keys()):
        train_vals = [x['viol_rate'] for x in ctx_metrics[ctx]['train']]
        test_vals = [x['viol_rate'] for x in ctx_metrics[ctx]['test']]
        if train_vals and test_vals:
            tr_m = np.mean(train_vals)
            te_m = np.mean(test_vals)
            chg = (te_m - tr_m) / (tr_m + 1e-9)
            ctx_drift_summary.append({'ctx': ctx, 'train': tr_m, 'test': te_m, 'change': chg})
            print(f"  {ctx:<22} {tr_m:>9.2%} {te_m:>9.2%} {chg:>+9.1%} "
                  f"{'increase' if chg > 0 else 'decrease'}")

    # ── Step 4h: ADWIN window size evolution ───────────────────────────────────
    print("\n[4h] ADWIN window evolution (drift detection progress)")
    print("-" * 70)

    print(f"\n  ADWIN state after processing Jul windows (delta=0.002):")
    print(f"  {'Metric':<25} {'Window Size':>12} {'ADWIN Mean':>12} {'Drift Count':>12}")
    print(f"  {'-'*65}")
    for mk, adwin in sorted(monitor.adwins.items()):
        print(f"  {mk:<25} {adwin.size():>12} {adwin.mean:>12.4f} {adwin.drift_count:>12}")

    # ── Step 4i: Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4 SUMMARY")
    print("=" * 70)

    n_drift_default = len(results[0.002]['events'])
    n_drift_relaxed = len(results[0.01]['events'])
    n_drift_strict = len(results[0.0001]['events'])

    # Compare Jan vs Jul meta-stream values
    train_df = meta_df[meta_df['period'] == 'train']
    test_df = meta_df[meta_df['period'] == 'test']

    print(f"""
ADWIN Drift Detection (Jan baseline -> Jul test):

  Configuration:
    ADWIN delta = 0.002 (default)
    min_window = 2
    Train windows: Jan (4 windows)
    Test windows: Jul (4 windows)

  Drift events:
    Relaxed (delta=0.01):   {n_drift_relaxed}
    Default (delta=0.002):  {n_drift_default}
    Strict (delta=0.0001):  {n_drift_strict}

  Meta-stream comparison (train vs test mean):
""")

    for col in ['null_rate_pax', 'null_rate_rc', 'violation_rate',
                 'anomaly_rate', 'fare_mean']:
        if col in meta_df.columns:
            tr = train_df[col].mean()
            te = test_df[col].mean()
            chg = (te - tr) / (abs(tr) + 1e-9) if tr != 0 else 0
            arrow = "^" if chg > 0 else "v"
            print(f"    {col:<25} Jan={tr:.4f}  Jul={te:.4f}  {arrow}{abs(chg):.1%}")

    print(f"""
  Key Insight:
    The ADWIN monitors QUALITY META-STREAMS (rates, scores).
    A drift in violation_rate or null_rate signals that the
    BASELINE has become stale - not individual anomalies.

  Self-Adaptation Actions:
    If drift detected -> Retrain baseline on post-drift windows
    If drift HIGH severity -> Retrain KNN/Lof model
    If context collapse -> Fallback to parent context groups

  Production Note:
    With monthly data, drift detection works across months.
    In real streaming: ADWIN would run on daily/hourly windows.
    delta parameter tuned to balance false positives vs missed drifts.
""")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        'config': {
            'adwin_delta': 0.002,
            'min_window': 2,
            'train_period': 'Jan 2024',
            'test_period': 'Jul 2024',
            'n_windows': N_WIN,
        },
        'drift_events_default': events if not events else [
            {k: str(v) if isinstance(v, (np.floating, np.integer)) else v
             for k, v in e.items()}
            for e in events
        ],
        'drift_events_relaxed': results[0.01]['events'],
        'drift_events_strict': results[0.0001]['events'],
        'baseline_stats': monitor.baseline,
        'train_vs_test': {
            col: {
                'train_mean': float(train_df[col].mean()),
                'test_mean': float(test_df[col].mean()),
                'change_pct': float((test_df[col].mean() - train_df[col].mean()) /
                                    (abs(train_df[col].mean()) + 1e-9))
            } for col in metric_keys if col in meta_df.columns
        },
    }

    with open(OUTPUT_DIR / 'step4_drift_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    meta_df.to_parquet(OUTPUT_DIR / 'step4_window_metrics.parquet', index=False)

    print(f"\nSaved: {OUTPUT_DIR / 'step4_drift_results.json'}")
    print(f"Saved: {OUTPUT_DIR / 'step4_window_metrics.parquet'}")
    print("\nDONE: Step 4 complete")


if __name__ == '__main__':
    run_step4()
