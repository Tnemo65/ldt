"""
Feature Extraction for NYC Taxi Data.

Canonical 34D FeatureVectorizer for Context-Aware MemStream anomaly detection.
Extracts normalized features from NYC Yellow Taxi trip records.

Reference: benchmark_v10.py verified features
"""

from typing import Dict, List, Optional, Union
from datetime import datetime

import numpy as np


# =============================================================================
# Feature Configuration
# =============================================================================

# NYC Taxi Zone definitions (TLC zones 1-265)
MANHATTAN_ZONES = set(range(1, 44))        # Zones 1-43
BROOKLYN_ZONES = set(range(104, 128))       # Zones 104-127
BRONX_ZONES = set(range(44, 104))           # Zones 44-103
STATEN_ISLAND_ZONES = set(range(162, 182)) # Zones 162-181
EWR_ZONES = set(range(182, 197))           # Newark Airport 182-196
NALP_ZONES = set(range(230, 235))           # Nassau/Westchester 230-234
UNKNOWN_ZONES = set(range(235, 266)) | set(range(197, 217))  # 235-265, 197-216

# JFK flat fare constant (verified from 2024 TLC data)
JFK_FLAT_FARE = 70.0

# Grid mapping for 16x17 grid (272 cells)
GRID_COLS = 16
GRID_ROWS = 17


# =============================================================================
# 34D Feature Names
# =============================================================================

FEATURE_NAMES = [
    # Indices 0-5: Raw trip features
    'trip_distance',      # 0
    'dur_min',            # 1
    'fare_amount',        # 2
    'passenger_count',    # 3
    'total_amount',       # 4
    'speed_mph',          # 5
    # Indices 6-8: Derived ratios
    'fare_per_mile',      # 6
    'fare_per_min',       # 7
    'fare_per_pax',       # 8
    # Indices 9-11: Temporal raw
    'pickup_hour',        # 9
    'pickup_dow',        # 10
    'is_weekend',        # 11
    # Indices 12-15: Spatial Grid (micro-resolution)
    'PU_Grid_X',         # 12
    'PU_Grid_Y',         # 13
    'DO_Grid_X',         # 14
    'DO_Grid_Y',         # 15
    # Indices 16-19: Normalized ratios
    'fare_per_mile_norm', # 16
    'fare_per_min_norm',  # 17
    'speed_norm',         # 18
    'pax_per_mile',      # 19
    # Indices 20-24: Cyclic temporal + distance squared
    'hour_sin',           # 20
    'hour_cos',           # 21
    'dow_sin',            # 22
    'dow_cos',            # 23
    'dist_squared',       # 24
    # Indices 25-29: RatecodeID one-hot (codes 1-5)
    'ratecode_1',         # 25 (Standard rate)
    'ratecode_2',         # 26 (JFK)
    'ratecode_3',         # 27 (Newark)
    'ratecode_4',         # 28 (Nassau/Westchester)
    'ratecode_5',         # 29 (Negotiated fare)
    # Index 30: is_night
    'is_night',           # 30
    # Indices 31-32: Log transforms
    'log_fare',           # 31
    'log_distance',      # 32
    # Index 33: Inter-borough rough
    'inter_borough_rough', # 33
]

NUM_FEATURES = len(FEATURE_NAMES)  # 34


# =============================================================================
# Zone to Grid Mapping
# =============================================================================

def zone_to_grid(zone_id: Union[int, float]) -> tuple:
    """
    Map TLC zone [1,265] to grid coordinates on a 16x17 grid (272 cells).
    
    The NYC taxi zone system is mapped to a rectangular grid:
    - 16 columns (x-axis)
    - 17 rows (y-axis)
    
    Returns (gx, gy) tuple where gx in [0,15], gy in [0,16].
    Zone 0 or invalid returns (0, 0).
    """
    z = int(zone_id) if not isinstance(zone_id, float) or not np.isnan(zone_id) else 0
    if z <= 0:
        return 0, 0
    gx = (z - 1) % GRID_COLS
    gy = (z - 1) // GRID_COLS
    return gx, gy


# =============================================================================
# Borough Encoding
# =============================================================================

BOROUGH_ENCODING = {
    'manhattan': 0,
    'brooklyn': 1,
    'bronx': 2,
    'staten_island': 3,
    'ewr': 4,
    'nalp': 5,
    'unknown': 6,
}


def get_borough_from_zone(zone_id: Union[int, float]) -> str:
    """Map zone ID to borough name."""
    z = int(zone_id) if not isinstance(zone_id, float) or not np.isnan(zone_id) else 0
    if z in MANHATTAN_ZONES:
        return 'manhattan'
    elif z in BROOKLYN_ZONES:
        return 'brooklyn'
    elif z in BRONX_ZONES:
        return 'bronx'
    elif z in STATEN_ISLAND_ZONES:
        return 'staten_island'
    elif z in EWR_ZONES:
        return 'ewr'
    elif z in NALP_ZONES:
        return 'nalp'
    else:
        return 'unknown'


# =============================================================================
# FeatureVectorizer (34D)
# =============================================================================

class FeatureVectorizer:
    """
    Canonical 34D FeatureVectorizer for NYC Taxi anomaly detection.
    
    Extracts and normalizes 34 features from NYC Yellow Taxi trip records.
    
    Feature categories:
    - Raw trip (6D): distance, duration, fare, passengers, total, speed
    - Derived ratios (3D): fare_per_mile, fare_per_min, fare_per_pax
    - Temporal raw (3D): hour, day-of-week, is_weekend
    - Spatial Grid (4D): PU/DO grid X/Y (micro-resolution)
    - Normalized ratios (4D): normalized versions of ratio features
    - Cyclic temporal + distance (5D): hour sin/cos, dow sin/cos, dist_sq
    - RatecodeID one-hot (5D): codes 1-5
    - is_night (1D): hour >= 20 OR hour <= 6
    - Log transforms (2D): log(fare+1), log(distance+1)
    - Inter-borough rough (1D): |PU_grid_y - DO_grid_y|
    
    Total: 34 dimensions
    """
    
    FEATURE_NAMES = FEATURE_NAMES
    DIM = NUM_FEATURES
    
    # Normalization denominators (from benchmark v10 calibration)
    FARE_PER_MILE_DENOM = 2.5
    FARE_PER_MIN_DENOM = 0.67
    SPEED_DENOM = 12.0
    EPS = 1e-8
    
    def __init__(self) -> None:
        self._stats_mean: Optional[np.ndarray] = None
        self._stats_std: Optional[np.ndarray] = None
        self._fitted: bool = False
    
    @property
    def num_features(self) -> int:
        return self.DIM
    
    @property
    def feature_names(self) -> List[str]:
        return self.FEATURE_NAMES.copy()
    
    def fit(self, records: List[Dict]) -> 'FeatureVectorizer':
        """Compute normalization statistics from records."""
        features = self.transform_batch(records)
        self._stats_mean = np.mean(features, axis=0)
        self._stats_std = np.std(features, axis=0)
        self._stats_std = np.clip(self._stats_std, min=self.EPS)
        self._fitted = True
        return self
    
    def transform(self, record: Dict) -> Optional[np.ndarray]:
        """Transform a single NYC taxi record into a 34D feature vector."""
        try:
            features = self._extract_features(record)
            return self._normalize_features(features)
        except (ValueError, TypeError, KeyError):
            return None
    
    def transform_batch(self, records: List[Dict]) -> np.ndarray:
        """Transform a batch of NYC taxi records into feature matrix."""
        features_list = []
        for record in records:
            feat = self.transform(record)
            if feat is not None:
                features_list.append(feat)
            else:
                features_list.append(np.zeros(self.DIM, dtype=np.float32))
        return np.array(features_list, dtype=np.float32)
    
    def _extract_features(self, record: Dict) -> np.ndarray:
        """Extract raw 34D features from a record."""
        features = np.zeros(self.DIM, dtype=np.float32)
        eps = self.EPS
        
        # --- Parse record fields ---
        
        pickup_dt = self._parse_datetime(record.get('tpep_pickup_datetime', ''))
        if pickup_dt is not None:
            pickup_hour = pickup_dt.hour
            pickup_dow = pickup_dt.weekday()
        else:
            pickup_hour = 12
            pickup_dow = 0
        
        trip_distance = self._safe_float(record.get('trip_distance', 0))
        fare_amount = self._safe_float(record.get('fare_amount', 0))
        passenger_count = self._safe_float(record.get('passenger_count', 1))
        total_amount = self._safe_float(record.get('total_amount', 0))
        ratecode_id = self._safe_float(record.get('RatecodeID', 1))
        
        # --- Compute derived trip features ---
        
        if pickup_dt is not None and record.get('tpep_dropoff_datetime'):
            dropoff_dt = self._parse_datetime(record.get('tpep_dropoff_datetime'))
            if dropoff_dt is not None:
                duration = (dropoff_dt - pickup_dt).total_seconds() / 60.0
            else:
                duration = 15.0
        else:
            duration = 15.0
        
        duration = np.clip(duration, 0, 360)
        
        if duration > 0:
            speed = trip_distance / (duration / 60.0)
        else:
            speed = 12.0
        speed = np.clip(speed, 0, 60)
        
        # --- Zone-based spatial features ---
        
        pu_loc = self._safe_int(record.get('PULocationID', 0))
        do_loc = self._safe_int(record.get('DOLocationID', 0))
        pux, puy = zone_to_grid(pu_loc)
        dox, doy = zone_to_grid(do_loc)
        
        # --- Ratios ---
        
        fare_per_mile = fare_amount / max(trip_distance, eps)
        fare_per_min = fare_amount / max(duration, eps)
        fare_per_pax = fare_amount / max(passenger_count, eps)
        
        # --- Inter-borough rough ---
        
        inter_borough = abs(puy - doy)
        
        # --- Indices 0-5: Raw trip ---
        features[0] = trip_distance
        features[1] = duration
        features[2] = fare_amount
        features[3] = passenger_count
        features[4] = total_amount
        features[5] = speed
        
        # --- Indices 6-8: Derived ratios ---
        features[6] = fare_per_mile
        features[7] = fare_per_min
        features[8] = fare_per_pax
        
        # --- Indices 9-11: Temporal raw ---
        features[9] = float(pickup_hour)
        features[10] = float(pickup_dow)
        features[11] = 1.0 if pickup_dow >= 5 else 0.0
        
        # --- Indices 12-15: Spatial Grid ---
        features[12] = float(pux)
        features[13] = float(puy)
        features[14] = float(dox)
        features[15] = float(doy)
        
        # --- Indices 16-19: Normalized ratios ---
        features[16] = fare_per_mile / self.FARE_PER_MILE_DENOM
        features[17] = fare_per_min / self.FARE_PER_MIN_DENOM
        features[18] = speed / self.SPEED_DENOM
        features[19] = passenger_count / max(trip_distance, eps)
        
        # --- Indices 20-24: Cyclic temporal + distance squared ---
        features[20] = np.sin(2 * np.pi * pickup_hour / 24.0)
        features[21] = np.cos(2 * np.pi * pickup_hour / 24.0)
        features[22] = np.sin(2 * np.pi * pickup_dow / 7.0)
        features[23] = np.cos(2 * np.pi * pickup_dow / 7.0)
        features[24] = trip_distance * trip_distance
        
        # --- Indices 25-29: RatecodeID one-hot (codes 1-5) ---
        ratecode_val = ratecode_id
        for i, rc in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            features[25 + i] = 1.0 if ratecode_val == rc else 0.0
        
        # --- Index 30: is_night ---
        features[30] = 1.0 if (pickup_hour >= 20 or pickup_hour <= 6) else 0.0
        
        # --- Indices 31-32: Log transforms ---
        features[31] = np.log1p(fare_amount)
        features[32] = np.log1p(trip_distance)
        
        # --- Index 33: Inter-borough rough ---
        features[33] = inter_borough
        
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features (replace NaN/inf with 0)."""
        normalized = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=0.0)
        return normalized.astype(np.float32)
    
    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse datetime string from NYC taxi data."""
        if not dt_str:
            return None
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S UTC',
            '%Y/%m/%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y %I:%M:%S %p',
        ]
        for fmt in formats:
            try:
                return datetime.strptime(dt_str.strip(), fmt)
            except ValueError:
                continue
        return None
    
    def _safe_float(self, value: Union[str, int, float, None]) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _safe_int(self, value: Union[str, int, float, None]) -> int:
        if value is None:
            return 0
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0


# =============================================================================
# Feature Statistics
# =============================================================================

class FeatureStatistics:
    """Compute and store feature statistics for normalization."""
    
    def __init__(self) -> None:
        self._sum: Optional[np.ndarray] = None
        self._sum_sq: Optional[np.ndarray] = None
        self._count: int = 0
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
    
    def update(self, features: np.ndarray) -> None:
        """Update statistics with new batch of features."""
        features = np.array(features, dtype=np.float32)
        if self._sum is None:
            self._sum = np.zeros(features.shape[1], dtype=np.float64)
            self._sum_sq = np.zeros(features.shape[1], dtype=np.float64)
            self._min = features.min(axis=0).copy()
            self._max = features.max(axis=0).copy()
        else:
            self._sum += features.sum(axis=0)
            self._sum_sq += (features ** 2).sum(axis=0)
            self._min = np.minimum(self._min, features.min(axis=0))
            self._max = np.maximum(self._max, features.max(axis=0))
        self._count += len(features)
    
    @property
    def mean(self) -> np.ndarray:
        if self._sum is None:
            return np.zeros(NUM_FEATURES, dtype=np.float32)
        return (self._sum / self._count).astype(np.float32)
    
    @property
    def std(self) -> np.ndarray:
        if self._sum is None or self._count < 2:
            return np.ones(NUM_FEATURES, dtype=np.float32)
        variance = (self._sum_sq / self._count) - (self._sum / self._count) ** 2
        std = np.sqrt(np.maximum(variance, 0))
        return std.astype(np.float32)
    
    @property
    def min(self) -> np.ndarray:
        if self._min is None:
            return np.zeros(NUM_FEATURES, dtype=np.float32)
        return self._min.astype(np.float32)
    
    @property
    def max(self) -> np.ndarray:
        if self._max is None:
            return np.ones(NUM_FEATURES, dtype=np.float32)
        return self._max.astype(np.float32)
    
    def zscore_normalize(self, features: np.ndarray) -> np.ndarray:
        features = np.array(features, dtype=np.float32)
        std = self.std + 1e-8
        return ((features - self.mean) / std).astype(np.float32)
    
    def minmax_normalize(self, features: np.ndarray) -> np.ndarray:
        features = np.array(features, dtype=np.float32)
        range_vals = self.max - self.min + 1e-8
        return ((features - self.min) / range_vals).astype(np.float32)
