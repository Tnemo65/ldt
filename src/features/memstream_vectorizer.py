"""
MemStream 34D FeatureVectorizer for NYC Taxi Anomaly Detection.

Canonical feature extraction from NYC Yellow Taxi trip records.
Implements the verified 34D feature space from benchmark v10.

Feature Categories (34 dimensions total):
    - Raw trip (6D): trip_distance, dur_min, fare_amount, passenger_count,
                     total_amount, speed_mph
    - Derived ratios (3D): fare_per_mile, fare_per_min, fare_per_pax
    - Temporal raw (3D): pickup_hour, pickup_dow, is_weekend
    - Spatial Grid (4D): PU_Grid_X/Y, DO_Grid_X/Y (micro-resolution)
    - Normalized ratios (4D): fare_per_mile_norm, fare_per_min_norm,
                              speed_norm, pax_per_mile
    - Cyclic temporal + distance (5D): hour_sin/cos, dow_sin/cos, dist_squared
    - RatecodeID one-hot (5D): ratecode_1 through ratecode_5
    - is_night (1D): hour >= 20 OR hour <= 6
    - Log transforms (2D): log_fare, log_distance
    - Inter-borough rough (1D): |PU_grid_y - DO_grid_y|

NOTE: PCA is PROHIBITED per MemStream Proposition 2.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "MemStreamVectorizer",
    "MemStreamFeatureNames",
    "MemStreamNumFeatures",
    "MemStreamNormalizationConstants",
]

LOGGER = logging.getLogger(__name__)

# =============================================================================
# Feature Configuration
# =============================================================================

# NYC Taxi Zone Grid (16x17 = 272 cells)
_GRID_COLS: int = 16
_GRID_ROWS: int = 17

# Normalization denominators (from benchmark v10 calibration)
_FARE_PER_MILE_DENOM: float = 2.5
_FARE_PER_MIN_DENOM: float = 0.67
_SPEED_DENOM: float = 12.0
_EPS: float = 1e-8

# Feature names in exact order (indices 0-33)
FEATURE_NAMES: List[str] = [
    # Indices 0-5: Raw trip features
    "trip_distance",       # 0
    "dur_min",             # 1
    "fare_amount",         # 2
    "passenger_count",     # 3
    "total_amount",        # 4
    "speed_mph",           # 5
    # Indices 6-8: Derived ratios
    "fare_per_mile",       # 6
    "fare_per_min",        # 7
    "fare_per_pax",        # 8
    # Indices 9-11: Temporal raw
    "pickup_hour",         # 9
    "pickup_dow",          # 10
    "is_weekend",          # 11
    # Indices 12-15: Spatial Grid (micro-resolution)
    "PU_Grid_X",          # 12
    "PU_Grid_Y",          # 13
    "DO_Grid_X",          # 14
    "DO_Grid_Y",          # 15
    # Indices 16-19: Normalized ratios
    "fare_per_mile_norm",  # 16
    "fare_per_min_norm",   # 17
    "speed_norm",          # 18
    "pax_per_mile",       # 19
    # Indices 20-24: Cyclic temporal + distance squared
    "hour_sin",            # 20
    "hour_cos",            # 21
    "dow_sin",             # 22
    "dow_cos",             # 23
    "dist_squared",        # 24
    # Indices 25-29: RatecodeID one-hot (codes 1-5)
    "ratecode_1",          # 25 (Standard rate)
    "ratecode_2",          # 26 (JFK)
    "ratecode_3",          # 27 (Newark)
    "ratecode_4",          # 28 (Nassau/Westchester)
    "ratecode_5",          # 29 (Negotiated fare)
    # Index 30: is_night
    "is_night",            # 30
    # Indices 31-32: Log transforms
    "log_fare",            # 31
    "log_distance",        # 32
    # Index 33: Inter-borough rough
    "inter_borough_rough",  # 33
]

NUM_FEATURES: int = len(FEATURE_NAMES)  # 34


class MemStreamVectorizer:
    """
    Canonical 34D FeatureVectorizer for NYC Taxi anomaly detection.

    Extracts and normalizes 34 features from NYC Yellow Taxi trip records.
    Supports both single-record and batch DataFrame transformations.

    Example:
        >>> vectorizer = MemStreamVectorizer()
        >>> features = vectorizer.transform(record)
        >>> batch_features = vectorizer.transform_df(df)
    """

    # Class-level constants for external access
    FEATURE_NAMES: List[str] = FEATURE_NAMES
    NUM_FEATURES: int = NUM_FEATURES
    FARE_PER_MILE_DENOM: float = _FARE_PER_MILE_DENOM
    FARE_PER_MIN_DENOM: float = _FARE_PER_MIN_DENOM
    SPEED_DENOM: float = _SPEED_DENOM
    EPS: float = _EPS

    def __init__(self) -> None:
        """Initialize the vectorizer with no fitted state."""
        self._fitted: bool = False
        self._stats_mean: Optional[np.ndarray] = None
        self._stats_std: Optional[np.ndarray] = None

    @property
    def num_features(self) -> int:
        """Return the number of features (34)."""
        return self.NUM_FEATURES

    @property
    def feature_names(self) -> List[str]:
        """Return the list of feature names."""
        return self.FEATURE_NAMES.copy()

    @property
    def is_fitted(self) -> bool:
        """Return True if fit() has been called."""
        return self._fitted

    def fit(self, records: List[Dict]) -> "MemStreamVectorizer":
        """
        Compute normalization statistics from records.

        Args:
            records: List of NYC taxi record dictionaries

        Returns:
            Self for method chaining
        """
        features = self.transform_batch(records)
        self._stats_mean = np.mean(features, axis=0)
        self._stats_std = np.std(features, axis=0)
        self._stats_std = np.clip(self._stats_std, min=self.EPS)
        self._fitted = True
        return self

    def transform(self, record: Dict) -> Optional[np.ndarray]:
        """
        Transform a single NYC taxi record into a 34D feature vector.

        Args:
            record: Dictionary with NYC taxi fields

        Returns:
            34D numpy array of features, or None if extraction fails
        """
        try:
            features = self._extract_features(record)
            return self._normalize_features(features)
        except (ValueError, TypeError, KeyError) as e:
            LOGGER.warning("Failed to transform record: %s", e)
            return None

    def transform_batch(self, records: List[Dict]) -> np.ndarray:
        """
        Transform a batch of records into feature matrix.

        Args:
            records: List of NYC taxi record dictionaries

        Returns:
            numpy array of shape (len(records), 34)
        """
        features_list = []
        for record in records:
            feat = self.transform(record)
            if feat is not None:
                features_list.append(feat)
            else:
                features_list.append(np.zeros(self.NUM_FEATURES, dtype=np.float32))
        return np.array(features_list, dtype=np.float32)

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized batch transform from DataFrame — no Python loops.

        Optimized for large batches using pandas/numpy vectorized operations.

        Args:
            df: DataFrame with NYC taxi columns

        Returns:
            numpy array of shape (len(df), 34)
        """
        n = len(df)

        # --- Temporal ---
        pickup_dt = df["tpep_pickup_datetime"]
        if hasattr(pickup_dt, "dt"):
            hours = pickup_dt.dt.hour.fillna(12).astype(int).values
            dows = pickup_dt.dt.dayofweek.fillna(0).astype(int).values
        else:
            hours = np.full(n, 12, dtype=int)
            dows = np.full(n, 0, dtype=int)

        # --- Raw fields (vectorized) ---
        trip_distance = (
            pd.to_numeric(df["trip_distance"], errors="coerce")
            .fillna(0)
            .values.astype(np.float32)
        )
        fare_amount = (
            pd.to_numeric(df["fare_amount"], errors="coerce")
            .fillna(0)
            .values.astype(np.float32)
        )
        total_amount = (
            pd.to_numeric(df["total_amount"], errors="coerce")
            .fillna(0)
            .values.astype(np.float32)
        )
        passenger_count = (
            pd.to_numeric(df["passenger_count"], errors="coerce")
            .fillna(1)
            .values.astype(np.float32)
        )
        ratecode_id = (
            pd.to_numeric(df["RatecodeID"], errors="coerce")
            .fillna(1)
            .values.astype(np.float32)
        )
        pu_loc = pd.to_numeric(df["PULocationID"], errors="coerce").fillna(1).values.astype(int)
        do_loc = pd.to_numeric(df["DOLocationID"], errors="coerce").fillna(1).values.astype(int)

        # --- Duration ---
        dropoff_dt = df.get("tpep_dropoff_datetime", None)
        if dropoff_dt is not None and hasattr(dropoff_dt, "dt"):
            dur = (dropoff_dt - pickup_dt).dt.total_seconds().fillna(900).values
        else:
            # Fallback: estimate from distance (assume 15 mph average)
            dur = trip_distance / np.where(trip_distance > 0, trip_distance / 15.0, 1.0)
        dur = np.clip(dur, 1, 86400).astype(np.float32)

        # --- Speed ---
        speed = np.where(
            trip_distance > 0,
            trip_distance / (dur / 3600.0 + 1e-8),
            0.0,
        ).astype(np.float32)

        # --- Derived ratios ---
        fare_per_mile = np.where(
            trip_distance > 0.1,
            fare_amount / (trip_distance + 1e-8),
            0.0,
        ).astype(np.float32)
        fare_per_min = np.where(
            dur > 1,
            fare_amount / (dur / 60.0 + 1e-8),
            0.0,
        ).astype(np.float32)
        fare_per_pax = np.where(
            passenger_count > 0,
            fare_amount / (passenger_count + 1e-8),
            0.0,
        ).astype(np.float32)

        # --- Grid coordinates ---
        pu_gx = ((pu_loc - 1) % _GRID_COLS).astype(np.float32)
        pu_gy = ((pu_loc - 1) // _GRID_COLS).astype(np.float32)
        do_gx = ((do_loc - 1) % _GRID_COLS).astype(np.float32)
        do_gy = ((do_loc - 1) // _GRID_COLS).astype(np.float32)

        # --- Normalized ratios ---
        fare_per_mile_norm = np.clip(
            fare_per_mile / self.FARE_PER_MILE_DENOM, 0, 20
        ).astype(np.float32)
        fare_per_min_norm = np.clip(
            fare_per_min / self.FARE_PER_MIN_DENOM, 0, 20
        ).astype(np.float32)
        speed_norm = np.clip(speed / self.SPEED_DENOM, 0, 20).astype(np.float32)
        pax_per_mile = np.where(
            trip_distance > 0.1,
            passenger_count / (trip_distance + 1e-8),
            0.0,
        ).astype(np.float32)

        # --- Cyclic temporal ---
        hour_sin = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
        hour_cos = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)
        dow_sin = np.sin(2 * np.pi * dows / 7.0).astype(np.float32)
        dow_cos = np.cos(2 * np.pi * dows / 7.0).astype(np.float32)
        dist_sq = (trip_distance**2).astype(np.float32)

        # --- Ratecode one-hot (5 columns) ---
        rc1 = (ratecode_id == 1.0).astype(np.float32)
        rc2 = (ratecode_id == 2.0).astype(np.float32)
        rc3 = (ratecode_id == 3.0).astype(np.float32)
        rc4 = (ratecode_id == 4.0).astype(np.float32)
        rc5 = (ratecode_id == 5.0).astype(np.float32)

        # --- is_night ---
        is_night = ((hours >= 20) | (hours <= 6)).astype(np.float32)

        # --- Log transforms ---
        log_fare = np.log1p(np.clip(fare_amount, 0, 1e6)).astype(np.float32)
        log_dist = np.log1p(np.clip(trip_distance, 0, 1e6)).astype(np.float32)

        # --- Inter-borough ---
        inter_borough = np.abs(pu_gy - do_gy).astype(np.float32)

        # --- is_weekend ---
        is_weekend = (dows >= 5).astype(np.float32)

        # --- Assemble [N, 34] ---
        features = np.stack(
            [
                trip_distance,
                dur,
                fare_amount,
                passenger_count,
                total_amount,
                speed,
                fare_per_mile,
                fare_per_min,
                fare_per_pax,
                hours.astype(np.float32),
                dows.astype(np.float32),
                is_weekend,
                pu_gx,
                pu_gy,
                do_gx,
                do_gy,
                fare_per_mile_norm,
                fare_per_min_norm,
                speed_norm,
                pax_per_mile,
                hour_sin,
                hour_cos,
                dow_sin,
                dow_cos,
                dist_sq,
                rc1,
                rc2,
                rc3,
                rc4,
                rc5,
                is_night,
                log_fare,
                log_dist,
                inter_borough,
            ],
            axis=1,
        ).astype(np.float32)

        return np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=0.0)

    def _extract_features(self, record: Dict) -> np.ndarray:
        """
        Extract raw 34D features from a single record.

        Args:
            record: Dictionary with NYC taxi fields

        Returns:
            numpy array of shape (34,)
        """
        features = np.zeros(self.NUM_FEATURES, dtype=np.float32)
        eps = self.EPS

        # --- Parse datetime ---
        pickup_dt = self._parse_datetime(record.get("tpep_pickup_datetime", ""))
        if pickup_dt is not None:
            pickup_hour = pickup_dt.hour
            pickup_dow = pickup_dt.weekday()
        else:
            pickup_hour = 12
            pickup_dow = 0

        # --- Raw fields ---
        trip_distance = self._safe_float(record.get("trip_distance", 0))
        fare_amount = self._safe_float(record.get("fare_amount", 0))
        passenger_count = self._safe_float(record.get("passenger_count", 1))
        total_amount = self._safe_float(record.get("total_amount", 0))
        ratecode_id = self._safe_float(record.get("RatecodeID", 1))

        # --- Duration ---
        if pickup_dt is not None and record.get("tpep_dropoff_datetime"):
            dropoff_dt = self._parse_datetime(record.get("tpep_dropoff_datetime"))
            if dropoff_dt is not None:
                duration = (dropoff_dt - pickup_dt).total_seconds() / 60.0
            else:
                duration = 15.0
        else:
            duration = 15.0
        duration = float(np.clip(duration, 0, 360))

        # --- Speed ---
        if duration > 0:
            speed = trip_distance / (duration / 60.0)
        else:
            speed = 12.0
        speed = float(np.clip(speed, 0, 60))

        # --- Spatial grid ---
        pu_loc = self._safe_int(record.get("PULocationID", 0))
        do_loc = self._safe_int(record.get("DOLocationID", 0))
        pux, puy = self._zone_to_grid(pu_loc)
        dox, doy = self._zone_to_grid(do_loc)

        # --- Ratios ---
        fare_per_mile = fare_amount / max(trip_distance, eps)
        fare_per_min = fare_amount / max(duration, eps)
        fare_per_pax = fare_amount / max(passenger_count, eps)
        pax_per_mile = passenger_count / max(trip_distance, eps)

        # --- Inter-borough ---
        inter_borough = abs(puy - doy)

        # --- Fill feature array ---

        # Indices 0-5: Raw trip
        features[0] = trip_distance
        features[1] = duration
        features[2] = fare_amount
        features[3] = passenger_count
        features[4] = total_amount
        features[5] = speed

        # Indices 6-8: Derived ratios
        features[6] = fare_per_mile
        features[7] = fare_per_min
        features[8] = fare_per_pax

        # Indices 9-11: Temporal raw
        features[9] = float(pickup_hour)
        features[10] = float(pickup_dow)
        features[11] = 1.0 if pickup_dow >= 5 else 0.0

        # Indices 12-15: Spatial Grid
        features[12] = float(pux)
        features[13] = float(puy)
        features[14] = float(dox)
        features[15] = float(doy)

        # Indices 16-19: Normalized ratios
        features[16] = fare_per_mile / self.FARE_PER_MILE_DENOM
        features[17] = fare_per_min / self.FARE_PER_MIN_DENOM
        features[18] = speed / self.SPEED_DENOM
        features[19] = pax_per_mile

        # Indices 20-24: Cyclic temporal + distance squared
        features[20] = np.sin(2 * np.pi * pickup_hour / 24.0)
        features[21] = np.cos(2 * np.pi * pickup_hour / 24.0)
        features[22] = np.sin(2 * np.pi * pickup_dow / 7.0)
        features[23] = np.cos(2 * np.pi * pickup_dow / 7.0)
        features[24] = trip_distance * trip_distance

        # Indices 25-29: RatecodeID one-hot (codes 1-5)
        for i, rc in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            features[25 + i] = 1.0 if ratecode_id == rc else 0.0

        # Index 30: is_night
        features[30] = 1.0 if (pickup_hour >= 20 or pickup_hour <= 6) else 0.0

        # Indices 31-32: Log transforms
        features[31] = float(np.log1p(fare_amount))
        features[32] = float(np.log1p(trip_distance))

        # Index 33: Inter-borough rough
        features[33] = inter_borough

        return features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features (replace NaN/inf with 0).

        Args:
            features: Raw feature array

        Returns:
            Normalized feature array
        """
        return np.nan_to_num(
            features, nan=0.0, posinf=100.0, neginf=0.0
        ).astype(np.float32)

    def _zone_to_grid(self, zone_id: int) -> tuple:
        """
        Map TLC zone [1,265] to grid coordinates on a 16x17 grid (272 cells).

        Args:
            zone_id: TLC zone ID

        Returns:
            (gx, gy) tuple where gx in [0,15], gy in [0,16]
        """
        z = int(zone_id) if zone_id > 0 else 0
        gx = (z - 1) % _GRID_COLS
        gy = (z - 1) // _GRID_COLS
        return gx, gy

    def _parse_datetime(self, dt_str: Union[str, datetime]) -> Optional[datetime]:
        """
        Parse datetime string or pandas Timestamp from NYC taxi data.

        Args:
            dt_str: Datetime string or datetime object

        Returns:
            datetime object or None if parsing fails
        """
        if not dt_str:
            return None

        # Handle pandas Timestamp / datetime objects
        if hasattr(dt_str, "to_pydatetime"):
            dt_str = dt_str.to_pydatetime()
        if hasattr(dt_str, "hour"):
            return dt_str
        if not isinstance(dt_str, str):
            dt_str = str(dt_str)

        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S UTC",
            "%Y/%m/%d %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %I:%M:%S %p",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(dt_str.strip(), fmt)
            except ValueError:
                continue
        return None

    def _safe_float(self, value: Union[str, int, float, None]) -> float:
        """Safely convert value to float."""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _safe_int(self, value: Union[str, int, float, None]) -> int:
        """Safely convert value to int."""
        if value is None:
            return 0
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0


# Aliases for backward compatibility
MemStreamFeatureNames = FEATURE_NAMES
MemStreamNumFeatures = NUM_FEATURES
MemStreamNormalizationConstants = {
    "FARE_PER_MILE_DENOM": _FARE_PER_MILE_DENOM,
    "FARE_PER_MIN_DENOM": _FARE_PER_MIN_DENOM,
    "SPEED_DENOM": _SPEED_DENOM,
    "EPS": _EPS,
}
