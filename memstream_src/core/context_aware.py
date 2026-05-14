"""
Context-Aware Components for CA-MemStream-EIA v10.

Key changes from v10 benchmark:
1. 10 ADWIN instances (1 per neighborhood) instead of 36
2. 80 context-beta thresholds (10 neighborhoods x 8 context cells)
3. ContextBeta class for neighborhood-aware thresholding
4. 34D base features from FeatureVectorizer

Reference: benchmark_v10.py verified configuration
"""

from collections import deque
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np

LOGGER = logging.getLogger('cadqstream-eia')


# =============================================================================
# ADWIN-U (Adaptive Windowing for Drift Detection)
# Scientific: Bifet & Gavalda (2007) - KDD
# =============================================================================

class ADWIN:
    """
    ADWIN-U: Adaptive Windowing for Drift Detection.
    
    Detects concept drift by monitoring the mean of a data stream.
    Uses an exponential histogram to maintain time-sorted windows.
    """
    
    def __init__(self, delta: float = 0.002, max_window: int = 500) -> None:
        """
        Args:
            delta: Confidence parameter (smaller = more conservative)
            max_window: Maximum window size
        """
        self.delta = delta
        self.max_window = max_window
        self._window: deque = deque(maxlen=max_window)
        self._total: float = 0.0
        self._n: int = 0
    
    def update(self, value: float) -> bool:
        """
        Add value and check for drift.
        
        Returns:
            True if drift detected, False otherwise
        """
        drift_detected = False
        
        # Add to window
        self._window.append(value)
        self._total += value
        self._n += 1
        
        # ADWIN drift detection: check if any split point has significantly
        # different means (confidence based on delta)
        if self._n > 100:  # Minimum window size
            mean = self._total / self._n
            drift_detected = self._detect_drift(mean)
        
        return drift_detected
    
    def _detect_drift(self, overall_mean: float) -> bool:
        """Detect drift using ADWIN's variance-based test."""
        n = len(self._window)
        if n < 50:
            return False
        
        for split in range(n // 4, 3 * n // 4):
            left = list(self._window)[:split]
            right = list(self._window)[split:]
            
            n1, n2 = len(left), len(right)
            if n1 < 20 or n2 < 20:
                continue
            
            mean1 = sum(left) / n1
            mean2 = sum(right) / n2
            
            # ADWIN's drift detection threshold
            m = 1.0 / (1.0 / n1 + 1.0 / n2)
            epsilon_cut = (2.0 / m) * (self.delta ** 0.5)
            
            if abs(mean1 - mean2) > epsilon_cut:
                return True
        
        return False
    
    def reset(self) -> None:
        """Reset ADWIN state."""
        self._window.clear()
        self._total = 0.0
        self._n = 0
    
    def get_stats(self) -> Dict:
        """Get ADWIN statistics."""
        return {
            'n': self._n,
            'mean': self._total / self._n if self._n > 0 else 0.0,
        }


# =============================================================================
# Neighborhood Definitions (10 neighborhoods from zone_mapping.py)
# =============================================================================

NEIGHBORHOODS: List[str] = [
    'manhattan',       # 0
    'brooklyn',        # 1
    'queens_lower',    # 2
    'queens_upper',    # 3
    'bronx',           # 4
    'staten_island',   # 5
    'ewr',             # 6
    'jfk',             # 7
    'nalp',            # 8
    'unknown',         # 9
]

NUM_NEIGHBORHOODS: int = len(NEIGHBORHOODS)  # 10

NEIGHBORHOOD_TO_ID: Dict[str, int] = {nb: i for i, nb in enumerate(NEIGHBORHOODS)}
ID_TO_NEIGHBORHOOD: Dict[int, str] = {i: nb for i, nb in enumerate(NEIGHBORHOODS)}


# =============================================================================
# Context Cell Definitions (8 cells)
# Context cell ID = (is_special << 2) | (is_night << 1) | is_weekend
# =============================================================================

CONTEXT_CELLS: List[str] = [
    'std_day_weekday',    # 0: Standard, Day, Weekday
    'std_night_weekday',  # 1: Standard, Night, Weekday
    'std_day_weekend',    # 2: Standard, Day, Weekend
    'std_night_weekend',  # 3: Standard, Night, Weekend
    'sp_day_weekday',     # 4: Special, Day, Weekday
    'sp_night_weekday',   # 5: Special, Night, Weekday
    'sp_day_weekend',     # 6: Special, Day, Weekend
    'sp_night_weekend',   # 7: Special, Night, Weekend
]

NUM_CONTEXT_CELLS: int = len(CONTEXT_CELLS)  # 8


# =============================================================================
# Context Beta Thresholds (80 total: 10 neighborhoods x 8 context cells)
# =============================================================================

class ContextBeta:
    """
    Context-beta threshold management for neighborhood-aware scoring.
    
    Computes and stores 80 thresholds:
    - 10 neighborhoods (manhattan, brooklyn, queens_lower, queens_upper, bronx,
      staten_island, ewr, jfk, nalp, unknown)
    - 8 context cells (Standard/Special x Day/Night x Weekday/Weekend)
    
    All thresholds are computed from warmup only (zero leakage).
    """
    
    def __init__(self, default_beta: float = 0.5):
        """
        Args:
            default_beta: Default threshold when no context-beta is available
        """
        self.default_beta = default_beta
        
        # 80 thresholds: [num_neighborhoods][num_context_cells]
        self._thresholds: np.ndarray = np.full(
            (NUM_NEIGHBORHOODS, NUM_CONTEXT_CELLS),
            default_beta,
            dtype=np.float32
        )
        
        # Tracking for warmup computation
        self._fitted: bool = False
        self._scores_per_context: Dict[Tuple[int, int], List[float]] = {}
    
    def record_score(
        self,
        score: float,
        neighborhood_id: int,
        context_cell: int
    ) -> None:
        """Record a score for warmup threshold computation."""
        key = (neighborhood_id, context_cell)
        if key not in self._scores_per_context:
            self._scores_per_context[key] = []
        self._scores_per_context[key].append(score)
    
    def fit_from_scores(self, percentile: float = 95.0) -> 'ContextBeta':
        """
        Compute context-beta thresholds from recorded warmup scores.
        
        Args:
            percentile: Percentile for threshold (e.g., 95 = 5% FPR target)
        
        Returns:
            self for chaining
        """
        for key, scores in self._scores_per_context.items():
            nb_id, ctx_id = key
            if len(scores) > 10:  # Minimum samples for threshold
                threshold = np.percentile(scores, percentile)
                self._thresholds[nb_id, ctx_id] = threshold
        
        self._fitted = True
        LOGGER.info(
            "[ContextBeta] Fitted %d context-beta thresholds",
            len(self._scores_per_context)
        )
        return self
    
    def get_beta(self, neighborhood_id: int, context_cell: int) -> float:
        """
        Get threshold for neighborhood and context.
        
        Args:
            neighborhood_id: Neighborhood index (0-9)
            context_cell: Context cell index (0-7)
        
        Returns:
            Threshold value
        """
        if 0 <= neighborhood_id < NUM_NEIGHBORHOODS and 0 <= context_cell < NUM_CONTEXT_CELLS:
            return float(self._thresholds[neighborhood_id, context_cell])
        return self.default_beta
    
    def get_beta_by_name(self, neighborhood: str, context_cell: str) -> float:
        """Get threshold by name."""
        nb_id = NEIGHBORHOOD_TO_ID.get(neighborhood, NUM_NEIGHBORHOODS - 1)
        ctx_id = CONTEXT_CELLS.index(context_cell) if context_cell in CONTEXT_CELLS else 0
        return self.get_beta(nb_id, ctx_id)
    
    def to_dict(self) -> Dict:
        """Export thresholds as dict for serialization."""
        thresholds_dict = {}
        for nb_id in range(NUM_NEIGHBORHOODS):
            nb_name = ID_TO_NEIGHBORHOOD[nb_id]
            thresholds_dict[nb_name] = {}
            for ctx_id in range(NUM_CONTEXT_CELLS):
                ctx_name = CONTEXT_CELLS[ctx_id]
                thresholds_dict[nb_name][ctx_name] = float(self._thresholds[nb_id, ctx_id])
        return thresholds_dict
    
    @classmethod
    def from_dict(cls, data: Dict, default_beta: float = 0.5) -> 'ContextBeta':
        """Import thresholds from dict."""
        cb = cls(default_beta=default_beta)
        for nb_name, ctx_dict in data.items():
            nb_id = NEIGHBORHOOD_TO_ID.get(nb_name, -1)
            if nb_id < 0:
                continue
            for ctx_name, threshold in ctx_dict.items():
                ctx_id = CONTEXT_CELLS.index(ctx_name) if ctx_name in CONTEXT_CELLS else -1
                if ctx_id >= 0:
                    cb._thresholds[nb_id, ctx_id] = float(threshold)
        cb._fitted = True
        return cb


# =============================================================================
# 10 ADWIN Instances (1 per neighborhood)
# =============================================================================

class NeighborhoodADWINManager:
    """
    Manages 10 ADWIN instances for drift detection (1 per neighborhood).
    
    Each neighborhood has independent drift detection to catch localized
    concept drift (e.g., surge pricing changes in Manhattan only).
    """
    
    def __init__(self, delta: float = 0.002, max_window: int = 500):
        """
        Args:
            delta: ADWIN confidence parameter
            max_window: Maximum window size per ADWIN
        """
        self.delta = delta
        self.max_window = max_window
        
        # 10 ADWIN instances, one per neighborhood
        self._adwins: Dict[int, ADWIN] = {
            nb_id: ADWIN(delta=delta, max_window=max_window)
            for nb_id in range(NUM_NEIGHBORHOODS)
        }
    
    def update(self, score: float, neighborhood_id: int) -> bool:
        """
        Update ADWIN for neighborhood and check for drift.
        
        Args:
            score: Anomaly score
            neighborhood_id: Neighborhood index (0-9)
        
        Returns:
            True if drift detected, False otherwise
        """
        if 0 <= neighborhood_id < NUM_NEIGHBORHOODS:
            return self._adwins[neighborhood_id].update(score)
        return False
    
    def reset(self, neighborhood_id: Optional[int] = None) -> None:
        """
        Reset ADWIN state.
        
        Args:
            neighborhood_id: Specific neighborhood to reset, or None for all
        """
        if neighborhood_id is not None:
            if 0 <= neighborhood_id < NUM_NEIGHBORHOODS:
                self._adwins[neighborhood_id].reset()
        else:
            for adwin in self._adwins.values():
                adwin.reset()
    
    def get_stats(self, neighborhood_id: int) -> Dict:
        """Get ADWIN stats for neighborhood."""
        if 0 <= neighborhood_id < NUM_NEIGHBORHOODS:
            return self._adwins[neighborhood_id].get_stats()
        return {}


# =============================================================================
# Context Cell Computation
# =============================================================================

def compute_context_cell(
    ratecode_id: float,
    hour: int,
    day_of_week: int
) -> int:
    """
    Compute context cell ID from record attributes.
    
    Context cell ID = (is_special << 2) | (is_night << 1) | is_weekend
    
    Args:
        ratecode_id: RatecodeID (1 = Standard, >1 = Special)
        hour: Hour of day (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
    
    Returns:
        Context cell ID (0-7)
    """
    is_special = 1 if ratecode_id > 1.0 else 0
    is_night = 1 if (hour >= 20 or hour < 6) else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    
    return (is_special << 2) | (is_night << 1) | is_weekend


def get_context_cell_name(cell_id: int) -> str:
    """Get human-readable context cell name."""
    if 0 <= cell_id < NUM_CONTEXT_CELLS:
        return CONTEXT_CELLS[cell_id]
    return 'unknown'


# =============================================================================
# Context Extraction from Record
# =============================================================================

def extract_context_from_record(record: Dict) -> Tuple[int, int, int, int]:
    """
    Extract all context information from a taxi record.
    
    Returns:
        Tuple of (neighborhood_id, context_cell, hour, day_of_week)
    """
    from datetime import datetime
    
    # Parse datetime
    dt_str = record.get('tpep_pickup_datetime', '')
    hour = 12
    day_of_week = 0
    
    if dt_str:
        try:
            # Try multiple formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S UTC',
                '%Y/%m/%d %H:%M:%S',
                '%m/%d/%Y %H:%M:%S',
                '%m/%d/%Y %I:%M:%S %p',
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(dt_str.strip(), fmt)
                    hour = dt.hour
                    day_of_week = dt.weekday()
                    break
                except ValueError:
                    continue
        except Exception:
            pass
    
    # Get ratecode
    ratecode_id = float(record.get('RatecodeID', 1))
    
    # Get zone and compute neighborhood
    zone_id = int(float(record.get('PULocationID', 1)))
    neighborhood_id = _zone_to_neighborhood_id(zone_id)
    
    # Compute context cell
    context_cell = compute_context_cell(ratecode_id, hour, day_of_week)
    
    return neighborhood_id, context_cell, hour, day_of_week


def _zone_to_neighborhood_id(zone_id: int) -> int:
    """Convert zone ID to neighborhood index (0-9)."""
    from .zone_mapping import location_to_neighborhood
    return location_to_neighborhood(zone_id)


# =============================================================================
# EIA Integration
# =============================================================================

class EIAIntegration:
    """
    EIA (External Information Coupling) integration for CA-MemStream.
    
    Coordinates:
    1. NeighborhoodADWINManager: 10 ADWIN instances for drift detection
    2. ContextBeta: 80 context-beta thresholds for scoring
    3. BAR Controller: Budget allocation rate management
    """
    
    def __init__(
        self,
        delta: float = 0.002,
        max_window: int = 500,
        default_beta: float = 0.5,
        min_budget_fraction: float = 0.01,
        enable_adwin: bool = True
    ):
        """
        Args:
            delta: ADWIN confidence parameter
            max_window: ADWIN max window size
            default_beta: Default threshold
            min_budget_fraction: Minimum budget for memory updates
            enable_adwin: Enable ADWIN drift detection
        """
        # 10 ADWIN instances (1 per neighborhood)
        self.adwin_manager = NeighborhoodADWINManager(
            delta=delta,
            max_window=max_window
        )
        
        # 80 context-beta thresholds
        self.context_beta = ContextBeta(default_beta=default_beta)
        
        # BAR Controller state
        self.enable_adwin = enable_adwin
        self.min_budget_fraction = min_budget_fraction
        self._total_records: int = 0
        self._memory_updates: int = 0
        self._drift_events: int = 0
        self._budget_granted: bool = False
    
    @property
    def bar_rate(self) -> float:
        """Current BAR rate."""
        if self._total_records == 0:
            return 0.0
        return self._memory_updates / self._total_records
    
    def score_and_update(
        self,
        score: float,
        neighborhood_id: int,
        context_cell: int,
        grant_budget: bool = False
    ) -> Tuple[bool, str, float]:
        """
        Score record and determine if memory should be updated.
        
        Args:
            score: Anomaly score
            neighborhood_id: Neighborhood index (0-9)
            context_cell: Context cell index (0-7)
            grant_budget: Whether IEC granted budget
        
        Returns:
            Tuple of (is_anomaly, update_reason, threshold)
        """
        self._total_records += 1
        
        # Get threshold
        threshold = self.context_beta.get_beta(neighborhood_id, context_cell)
        is_anomaly = score > threshold
        
        # Determine if memory should be updated
        should_update, reason = self._should_update_memory(score, neighborhood_id)
        
        if should_update:
            self._memory_updates += 1
        
        return is_anomaly, reason, threshold
    
    def _should_update_memory(
        self,
        score: float,
        neighborhood_id: int
    ) -> Tuple[bool, str]:
        """
        Determine if memory should be updated for this record.
        
        Returns:
            Tuple of (should_update, reason)
        """
        # Rule 1: ADWIN drift detection
        if self.enable_adwin:
            drift_detected = self.adwin_manager.update(score, neighborhood_id)
            if drift_detected:
                self._drift_events += 1
                LOGGER.info(
                    "[EIA] Drift detected for neighborhood %d",
                    neighborhood_id
                )
                return True, "drift_detected"
        
        # Rule 2: Explicit budget grant
        if self._budget_granted:
            self._budget_granted = False
            return True, "budget_granted"
        
        # Rule 3: Minimum budget guarantee
        if self.bar_rate < self.min_budget_fraction:
            return True, "minimum_budget"
        
        # Default: No update
        return False, "no_budget"
    
    def grant_budget(self, reason: str = "manual") -> None:
        """Grant budget for memory update."""
        self._budget_granted = True
        LOGGER.info("[EIA] Budget granted: %s", reason)
    
    def get_stats(self) -> Dict:
        """Get EIA statistics."""
        return {
            'total_records': self._total_records,
            'memory_updates': self._memory_updates,
            'drift_events': self._drift_events,
            'bar_rate': self.bar_rate,
            'bar_rate_pct': self.bar_rate * 100,
        }
    
    def reset(self) -> None:
        """Reset all state."""
        self.adwin_manager.reset()
        self._total_records = 0
        self._memory_updates = 0
        self._drift_events = 0
        self._budget_granted = False


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# For code that imports from this module
def get_4d_context(record: Dict, neighborhood_mapping=None) -> Dict:
    """
    Legacy 4D context extraction (backward compatibility).
    
    Note: This is kept for compatibility but is deprecated.
    Use extract_context_from_record() instead.
    """
    nb_id, ctx_id, hour, dow = extract_context_from_record(record)
    nb_name = ID_TO_NEIGHBORHOOD.get(nb_id, 'unknown')
    ctx_name = CONTEXT_CELLS[ctx_id]
    
    # Map context to 4D format
    hour_bucket = 'night' if (hour >= 20 or hour < 6) else (
        'morning_rush' if 6 <= hour < 10 else (
            'evening_rush' if 17 <= hour < 21 else 'midday'
        )
    )
    day_type = 'weekend' if dow >= 5 else 'weekday'
    trip_distance = float(record.get('trip_distance', 0))
    trip_type = 'short' if trip_distance < 2 else ('medium' if trip_distance < 10 else 'long')
    
    return {
        'neighborhood': nb_name,
        'hour_bucket': hour_bucket,
        'day_type': day_type,
        'trip_type': trip_type,
        'neighborhood_id': nb_id,
        'context_cell': ctx_id,
    }


def get_neighborhood_from_zone(zone_id: int) -> str:
    """Get neighborhood name from zone ID."""
    nb_id = _zone_to_neighborhood_id(zone_id)
    return ID_TO_NEIGHBORHOOD.get(nb_id, 'unknown')


__all__ = [
    # ADWIN
    'ADWIN',
    # Neighborhoods
    'NEIGHBORHOODS',
    'NUM_NEIGHBORHOODS',
    'NEIGHBORHOOD_TO_ID',
    'ID_TO_NEIGHBORHOOD',
    # Context cells
    'CONTEXT_CELLS',
    'NUM_CONTEXT_CELLS',
    # Context beta
    'ContextBeta',
    # ADWIN manager
    'NeighborhoodADWINManager',
    # Context computation
    'compute_context_cell',
    'get_context_cell_name',
    'extract_context_from_record',
    # EIA integration
    'EIAIntegration',
    # Backward compatibility
    'get_4d_context',
    'get_neighborhood_from_zone',
]
