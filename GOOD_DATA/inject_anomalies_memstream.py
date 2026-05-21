"""
datasetv2/inject_anomalies_memstream.py
======================================
Inject 7 MemStream-detectable anomaly types into clean TLC taxi data.

Anomaly types (all pass hard rules, all detectable by MemStream 34D features):
  Type 1: Short Expensive  - fare x 5-15
  Type 2: Tip Anomaly      - tip = fare x 10-20
  Type 3: Slow Trip        - duration x 2-3
  Type 4: Combo            - distance x 0.05-0.3, duration x 2-5
  Type 5: Speed Anomaly    - duration / 2-4
  Type 6: Short Expensive2 - fare x 3-6
  Type 7: Slow Trip2       - duration x 1.5-2.5

Total: 8% (evenly distributed ~1.14% per type)

Hard rules enforced:
  fare_amount > 0 and <= 500
  trip_distance > 0 and <= 500
  passenger_count in [1, 6]
  PULocationID / DOLocationID in [1, 263]
  duration_s > 0 and < 86400
  speed_mph > 0 and <= 80
  RatecodeID in [1,2,3,4,5,6,99]

Slices:
  - train:  Jan-Aug 2024   (CLEAN, 0%)
  - valid:  Nov 1-15 2024  (POLLUTED, 8%)
  - test:   Nov 16-30 2024 (POLLUTED, 8%)
  - prod:   Jan-Feb 2025   (POLLUTED, 8%)

Usage:
    .venv/Scripts/python.exe datasetv2/inject_anomalies_memstream.py
"""

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLEAN_DIR = Path("data/clean")
OUT_DIR = Path("C:/proj/new/datasetv2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Anomaly type definitions
# ---------------------------------------------------------------------------
RATE_PER_TYPE = 1.0 / 7.0  # ~1.14% per type

ANOMALY_TYPES = {
    1: {
        "name": "short_expensive",
        "desc": "fare x 5-15",
        "rate": RATE_PER_TYPE,
        "key_feature": "fare_per_min",
        "key_signal": "fare x 5-15, distance unchanged",
    },
    2: {
        "name": "tip_anomaly",
        "desc": "tip = fare x 10-20",
        "rate": RATE_PER_TYPE,
        "key_feature": "total_amount",
        "key_signal": "tip x 10-20, total_amount explodes",
    },
    3: {
        "name": "slow_trip",
        "desc": "duration x 2-3",
        "rate": RATE_PER_TYPE,
        "key_feature": "fare_per_min",
        "key_signal": "duration x 2-3, fare unchanged",
    },
    4: {
        "name": "combo_short_long",
        "desc": "distance x 0.05-0.3, duration x 2-5",
        "rate": RATE_PER_TYPE,
        "key_feature": "speed_mph",
        "key_signal": "very short distance + long duration",
    },
    5: {
        "name": "speed_anomaly",
        "desc": "duration / 2-4",
        "rate": RATE_PER_TYPE,
        "key_feature": "fare_per_min",
        "key_signal": "duration / 2-4, faster trip",
    },
    6: {
        "name": "short_expensive_v2",
        "desc": "fare x 3-6",
        "rate": RATE_PER_TYPE,
        "key_feature": "fare_per_min",
        "key_signal": "fare x 3-6, distance unchanged",
    },
    7: {
        "name": "slow_trip_v2",
        "desc": "duration x 1.5-2.5",
        "rate": RATE_PER_TYPE,
        "key_feature": "fare_per_min",
        "key_signal": "duration x 1.5-2.5, fare unchanged",
    },
}

TOTAL_RATE = 0.08  # 8% total

SLICES = {
    "train": {
        "clean_file": "clean_2024.parquet",
        "output_file": "train_clean.parquet",
        "start_date": "2024-01-01",
        "end_date": "2024-09-01",
        "inject": False,
    },
    "valid": {
        "clean_file": "clean_2024.parquet",
        "output_file": "valid_polluted.parquet",
        "start_date": "2024-11-01",
        "end_date": "2024-11-16",
        "inject": True,
    },
    "test": {
        "clean_file": "clean_2024.parquet",
        "output_file": "test_polluted.parquet",
        "start_date": "2024-11-16",
        "end_date": "2024-12-01",
        "inject": True,
    },
    "prod": {
        "clean_file": "clean_2025.parquet",
        "output_file": "prod_polluted.parquet",
        "start_date": "2025-01-01",
        "end_date": "2025-03-01",
        "inject": True,
    },
}

# ---------------------------------------------------------------------------
# Polluter functions
# ---------------------------------------------------------------------------

def polluter_short_expensive(df, indices):
    if len(indices) == 0:
        return indices
    scale = np.random.uniform(5, 15, size=len(indices))
    new_fare = df.loc[df.index[indices], "fare_amount"].values * scale
    valid = (new_fare > 0) & (new_fare <= 500)
    confirmed = indices[valid]
    if len(confirmed) == 0:
        return confirmed
    df.loc[df.index[confirmed], "fare_amount"] = new_fare[valid].astype(np.float64)
    df.loc[df.index[confirmed], "total_amount"] = (
        df.loc[df.index[confirmed], "fare_amount"].values +
        df.loc[df.index[confirmed], "tip_amount"].values +
        df.loc[df.index[confirmed], "extra"].values +
        df.loc[df.index[confirmed], "mta_tax"].values +
        df.loc[df.index[confirmed], "tolls_amount"].values +
        df.loc[df.index[confirmed], "improvement_surcharge"].values +
        df.loc[df.index[confirmed], "congestion_surcharge"].values
    ).astype(np.float64)
    return confirmed


def polluter_tip_anomaly(df, indices):
    if len(indices) == 0:
        return indices
    scale = np.random.uniform(10, 20, size=len(indices))
    fare_vals = df.loc[df.index[indices], "fare_amount"].values.astype(np.float64)
    new_tip = fare_vals * scale
    df.loc[df.index[indices], "tip_amount"] = new_tip.astype(np.float64)
    df.loc[df.index[indices], "total_amount"] = (
        new_tip + fare_vals +
        df.loc[df.index[indices], "extra"].values +
        df.loc[df.index[indices], "mta_tax"].values +
        df.loc[df.index[indices], "tolls_amount"].values +
        df.loc[df.index[indices], "improvement_surcharge"].values +
        df.loc[df.index[indices], "congestion_surcharge"].values
    ).astype(np.float64)
    return indices


def polluter_slow_trip(df, indices):
    if len(indices) == 0:
        return indices
    scale = np.random.uniform(2, 3, size=len(indices))
    new_dur = df.loc[df.index[indices], "duration_s"].values * scale
    valid = (new_dur > 0) & (new_dur < 86400)
    confirmed = indices[valid]
    if len(confirmed) == 0:
        return confirmed
    df.loc[df.index[confirmed], "duration_s"] = new_dur[valid].astype(np.float64)
    return confirmed


def polluter_combo(df, indices):
    if len(indices) == 0:
        return indices
    dist_scale = np.random.uniform(0.05, 0.3, size=len(indices))
    dur_scale = np.random.uniform(2, 5, size=len(indices))
    new_dist = df.loc[df.index[indices], "trip_distance"].values * dist_scale
    new_dur = df.loc[df.index[indices], "duration_s"].values * dur_scale
    valid = (
        (new_dist > 0) & (new_dist <= 500) &
        (new_dur > 0) & (new_dur < 86400) &
        (new_dist / np.maximum(new_dur / 3600, 0.001) <= 80)
    )
    confirmed = indices[valid]
    if len(confirmed) == 0:
        return confirmed
    df.loc[df.index[confirmed], "trip_distance"] = new_dist[valid].astype(np.float64)
    df.loc[df.index[confirmed], "duration_s"] = new_dur[valid].astype(np.float64)
    return confirmed


def polluter_speed_anomaly(df, indices):
    if len(indices) == 0:
        return indices
    divisor = np.random.uniform(2, 4, size=len(indices))
    new_dur = df.loc[df.index[indices], "duration_s"].values / divisor
    dist_vals = df.loc[df.index[indices], "trip_distance"].values
    new_speed = dist_vals / np.maximum(new_dur / 3600, 0.001)
    valid = (
        (new_dur > 0) & (new_dur < 86400) &
        (new_speed > 0) & (new_speed <= 80)
    )
    confirmed = indices[valid]
    if len(confirmed) == 0:
        return confirmed
    df.loc[df.index[confirmed], "duration_s"] = new_dur[valid].astype(np.float64)
    return confirmed


def polluter_short_expensive_v2(df, indices):
    if len(indices) == 0:
        return indices
    scale = np.random.uniform(3, 6, size=len(indices))
    new_fare = df.loc[df.index[indices], "fare_amount"].values * scale
    valid = (new_fare > 0) & (new_fare <= 500)
    confirmed = indices[valid]
    if len(confirmed) == 0:
        return confirmed
    df.loc[df.index[confirmed], "fare_amount"] = new_fare[valid].astype(np.float64)
    df.loc[df.index[confirmed], "total_amount"] = (
        df.loc[df.index[confirmed], "fare_amount"].values +
        df.loc[df.index[confirmed], "tip_amount"].values +
        df.loc[df.index[confirmed], "extra"].values +
        df.loc[df.index[confirmed], "mta_tax"].values +
        df.loc[df.index[confirmed], "tolls_amount"].values +
        df.loc[df.index[confirmed], "improvement_surcharge"].values +
        df.loc[df.index[confirmed], "congestion_surcharge"].values
    ).astype(np.float64)
    return confirmed


def polluter_slow_trip_v2(df, indices):
    if len(indices) == 0:
        return indices
    scale = np.random.uniform(1.5, 2.5, size=len(indices))
    new_dur = df.loc[df.index[indices], "duration_s"].values * scale
    valid = (new_dur > 0) & (new_dur < 86400)
    confirmed = indices[valid]
    if len(confirmed) == 0:
        return confirmed
    df.loc[df.index[confirmed], "duration_s"] = new_dur[valid].astype(np.float64)
    return confirmed


POLLUTER_MAP = {
    1: polluter_short_expensive,
    2: polluter_tip_anomaly,
    3: polluter_slow_trip,
    4: polluter_combo,
    5: polluter_speed_anomaly,
    6: polluter_short_expensive_v2,
    7: polluter_slow_trip_v2,
}


# ---------------------------------------------------------------------------
# Rule validation
# ---------------------------------------------------------------------------

def validate_hard_rules(df, indices):
    if len(indices) == 0:
        return {}
    dur = df.loc[df.index[indices], "duration_s"].values
    fare = df.loc[df.index[indices], "fare_amount"].values
    dist = df.loc[df.index[indices], "trip_distance"].values
    pax = df.loc[df.index[indices], "passenger_count"].values
    pu = df.loc[df.index[indices], "PULocationID"].values
    do = df.loc[df.index[indices], "DOLocationID"].values
    rc = df.loc[df.index[indices], "RatecodeID"].values
    speed = dist / np.maximum(dur / 3600, 0.001)
    return {
        "fare_gt_0": bool(np.all(fare > 0)),
        "fare_lte_500": bool(np.all(fare <= 500)),
        "dist_gt_0": bool(np.all(dist > 0)),
        "dist_lte_500": bool(np.all(dist <= 500)),
        "pax_1_6": bool(np.all((pax >= 1) & (pax <= 6))),
        "pu_1_263": bool(np.all((pu >= 1) & (pu <= 263))),
        "do_1_263": bool(np.all((do >= 1) & (do <= 263))),
        "dur_gt_0": bool(np.all(dur > 0)),
        "dur_lt_86400": bool(np.all(dur < 86400)),
        "speed_gt_0": bool(np.all(speed > 0)),
        "speed_lte_80": bool(np.all(speed <= 80)),
        "ratecode_valid": bool(np.all(np.isin(rc, [1, 2, 3, 4, 5, 6, 99]))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_slice(name, cfg):
    t0 = time.time()

    clean_path = CLEAN_DIR / cfg["clean_file"]
    df = pd.read_parquet(clean_path)
    n_raw = len(df)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

    start = pd.Timestamp(cfg["start_date"])
    end = pd.Timestamp(cfg["end_date"])
    mask_slice = (df["tpep_pickup_datetime"] >= start) & (df["tpep_pickup_datetime"] < end)
    df = df[mask_slice].reset_index(drop=True)
    n_slice = len(df)

    print(f"\n{'='*60}")
    print(f"[{name.upper()}] Slice ({cfg['start_date']} to {cfg['end_date']}): {n_slice:,} records")
    print(f"{'='*60}")

    result = {
        "slice_name": name,
        "clean_source": cfg["clean_file"],
        "n_raw_total": int(n_raw),
        "n_slice": int(n_slice),
        "injected": cfg["inject"],
        "total_rate": TOTAL_RATE,
        "elapsed_s": 0.0,
    }

    if not cfg["inject"]:
        out_path = OUT_DIR / cfg["output_file"]
        df.to_parquet(out_path, index=False)
        sz = out_path.stat().st_size / 1e6
        print(f"  Saved (CLEAN): {out_path} ({n_slice:,} rows, {sz:.1f} MB)")
        result["elapsed_s"] = round(time.time() - t0, 2)
        return result

    # --- Injection ---
    # Step 1: select 8% of indices as the total anomaly pool
    n_target = int(round(TOTAL_RATE * n_slice))
    anomaly_pool = np.sort(np.random.choice(n_slice, size=n_target, replace=False))

    # Step 2: pre-compute eligible indices for each type (rule-specific filtering)
    pool_set = set(anomaly_pool.tolist())

    dur_vals = df["duration_s"].values
    dist_vals = df["trip_distance"].values
    fare_vals = df["fare_amount"].values

    # Compute derived for rule checks
    speed_vals = dist_vals / np.maximum(dur_vals / 3600, 0.001)

    # Type 1: short_expensive - eligible if fare * 15 <= 500
    type1_eligible = np.array([i for i in pool_set if fare_vals[i] * 15 <= 500], dtype=np.int64)

    # Type 2: tip_anomaly - all eligible (no rule constraints beyond base)
    type2_eligible = np.array(list(pool_set), dtype=np.int64)

    # Type 3: slow_trip - eligible if duration * 3 < 86400
    type3_eligible = np.array([i for i in pool_set if dur_vals[i] * 3 < 86400], dtype=np.int64)

    # Type 4: combo - eligible if trip_distance * 0.3 <= 500 AND duration * 5 < 86400 AND speed <= 80
    type4_eligible = np.array([
        i for i in pool_set
        if dist_vals[i] * 0.3 <= 500
        and dur_vals[i] * 5 < 86400
        and dist_vals[i] * 0.3 / np.maximum(dur_vals[i] * 5 / 3600, 0.001) <= 80
    ], dtype=np.int64)

    # Type 5: speed_anomaly - eligible if original speed * 4 <= 80 (so we can slow down)
    # Speed increases by divisor (2-4), so need original_speed * 4 <= 80
    type5_eligible = np.array([
        i for i in pool_set
        if speed_vals[i] > 0 and speed_vals[i] * 4 <= 80
    ], dtype=np.int64)

    # Type 6: short_expensive_v2 - eligible if fare * 6 <= 500
    type6_eligible = np.array([i for i in pool_set if fare_vals[i] * 6 <= 500], dtype=np.int64)

    # Type 7: slow_trip_v2 - eligible if duration * 2.5 < 86400
    type7_eligible = np.array([i for i in pool_set if dur_vals[i] * 2.5 < 86400], dtype=np.int64)

    TYPE_ELIGIBLE = {
        1: type1_eligible,
        2: type2_eligible,
        3: type3_eligible,
        4: type4_eligible,
        5: type5_eligible,
        6: type6_eligible,
        7: type7_eligible,
    }

    # Step 3: sample and apply per type
    per_type = {}
    used = set()  # track used indices to prevent overlap
    all_confirmed = []

    print(f"  Target total: {n_target:,} / {n_slice:,} = {TOTAL_RATE*100:.2f}%")
    print(f"  7 types, pre-filtered eligible pools\n")

    for type_id in range(1, 8):
        type_cfg = ANOMALY_TYPES[type_id]
        n_type_target = n_target // 7

        eligible = TYPE_ELIGIBLE[type_id]
        # Remove already-used indices
        eligible = np.array([i for i in eligible if i not in used], dtype=np.int64)

        n_eligible = len(eligible)
        if n_eligible == 0:
            print(f"  Type {type_id}: {type_cfg['name']:<22} - NO eligible rows left!")
            continue

        # Sample: take min(target, eligible_count)
        n_sample = min(n_type_target, n_eligible)
        chosen = np.sort(np.random.choice(eligible, size=n_sample, replace=False))

        confirmed = POLLUTER_MAP[type_id](df, chosen)
        n_confirmed = len(confirmed)
        ratio = n_confirmed / n_slice * 100

        # Mark as used
        used.update(confirmed.tolist())
        all_confirmed.extend(confirmed.tolist())

        rules = validate_hard_rules(df, confirmed)
        rules_ok = all(rules.values()) if rules else True

        pct_eligible = n_eligible / n_target * 100 if n_target > 0 else 0
        print(f"  Type {type_id}: {type_cfg['name']:<22} "
              f"eligible={n_eligible:,} ({pct_eligible:.1f}%) "
              f"confirmed={n_confirmed:,} ({ratio:.2f}%) "
              f"[{'PASS' if rules_ok else 'FAIL'}]")

        if not rules_ok:
            failed = [k for k, v in rules.items() if not v]
            print(f"    FAILED rules: {failed}")

        per_type[str(type_id)] = {
            "name": type_cfg["name"],
            "desc": type_cfg["desc"],
            "target_n": int(n_type_target),
            "eligible_n": int(n_eligible),
            "confirmed_n": int(n_confirmed),
            "ratio_pct": float(ratio),
            "key_feature": type_cfg["key_feature"],
            "key_signal": type_cfg["key_signal"],
            "indices": confirmed.tolist(),
            "rules_pass": rules_ok,
        }

    # Ground truth mask
    ground_truth_mask = np.zeros(n_slice, dtype=bool)
    all_confirmed = sorted(set(all_confirmed))
    ground_truth_mask[all_confirmed] = True

    actual_count = int(ground_truth_mask.sum())
    actual_ratio = actual_count / n_slice

    print(f"\n  === Summary ===")
    print(f"  Total confirmed: {actual_count:,} / {n_slice:,} = {actual_ratio*100:.4f}%")
    print(f"  Target:          {TOTAL_RATE*100:.2f}%")

    # Save
    out_path = OUT_DIR / cfg["output_file"]
    df.to_parquet(out_path, index=False)
    sz = out_path.stat().st_size / 1e6
    print(f"\n  Saved: {out_path} ({len(df):,} rows, {sz:.1f} MB)")

    gt_dir = OUT_DIR / name
    gt_dir.mkdir(parents=True, exist_ok=True)

    gt_mask_path = gt_dir / "ground_truth_mask.npy"
    np.save(gt_mask_path, ground_truth_mask)
    print(f"  Saved: {gt_mask_path}")

    gt_type_path = gt_dir / "ground_truth_per_type.json"
    with open(gt_type_path, "w") as f:
        json.dump(per_type, f, indent=2)
    print(f"  Saved: {gt_type_path}")

    injection_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": 42,
        "slice": name,
        "clean_source": cfg["clean_file"],
        "slice_start": cfg["start_date"],
        "slice_end": cfg["end_date"],
        "n_raw_total": int(n_raw),
        "n_slice": int(n_slice),
        "total_rate": TOTAL_RATE,
        "total_target": int(n_target),
        "total_confirmed": actual_count,
        "actual_ratio_pct": float(actual_ratio * 100),
        "type_counts": {str(k): v["confirmed_n"] for k, v in per_type.items()},
        "elapsed_s": round(time.time() - t0, 2),
    }
    log_path = gt_dir / "injection_log.json"
    with open(log_path, "w") as f:
        json.dump(injection_log, f, indent=2)
    print(f"  Saved: {log_path}")

    result["elapsed_s"] = round(time.time() - t0, 2)
    result["total_confirmed"] = actual_count
    result["actual_ratio_pct"] = float(actual_ratio * 100)
    return result


def main():
    print(f"{'='*60}")
    print(f"INJECTION - 7 MemStream-detectable Anomaly Types @ 8%")
    print(f"Output: {OUT_DIR}")
    print(f"{'='*60}")

    total_t0 = time.time()
    all_logs = {}

    for name, cfg in SLICES.items():
        log = process_slice(name, cfg)
        all_logs[name] = log

        if log["injected"]:
            target = TOTAL_RATE * 100
            actual = log["actual_ratio_pct"]
            status = "PASS" if abs(actual - target) <= 0.5 else "FAIL"
        else:
            status = "CLEAN"
        print(f"\n[{status}] {name.upper()} complete in {log['elapsed_s']:.1f}s")

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"ALL DONE in {total_elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\nOutput directory: {OUT_DIR}")
    for f in sorted(OUT_DIR.rglob("*")):
        if f.is_file():
            sz = f.stat().st_size / 1e6
            print(f"  {f.relative_to(OUT_DIR)}  ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
