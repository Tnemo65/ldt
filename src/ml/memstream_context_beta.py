"""
ContextBeta: Neighborhood-Aware Threshold Matrix for MemStream.

Implements 80 beta values (10 neighborhoods x 8 context cells) for
context-aware anomaly scoring. Each cell stores a threshold computed
from warmup scores.

Context Cell Definition:
    Cell ID = (is_special << 2) | (is_night << 1) | is_weekend

    is_special = ratecode > 1
    is_night = hour >= 20 or hour < 6  (UNIFIED: >= 20)
    is_weekend = dow >= 5

8 Context Cells:
    0: std_day_weekday    (Standard, Day, Weekday)
    1: std_night_weekday  (Standard, Night, Weekday)
    2: std_day_weekend    (Standard, Day, Weekend)
    3: std_night_weekend  (Standard, Night, Weekend)
    4: sp_day_weekday     (Special, Day, Weekday)
    5: sp_night_weekday   (Special, Night, Weekday)
    6: sp_day_weekend     (Special, Day, Weekend)
    7: sp_night_weekend   (Special, Night, Weekend)

Cell Minimum: 11 samples (code uses len(scores) > 10)
Weighted fallback to overall beta if < 11 samples.

Reference: Phase 1C migration plan
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Neighborhood Definitions (10 neighborhoods)
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
# ContextBeta Implementation
# =============================================================================

class ContextBeta:
    """
    Context-beta threshold matrix: 10 neighborhoods x 8 context cells = 80 values.

    Each cell stores a threshold computed from warmup scores using the
    specified percentile. During scoring, the threshold is used to
    normalize the raw anomaly score: score / beta.

    A cell is considered valid (non-empty) only if it has collected
    at least 11 samples (code: len(scores) > 10). Cells with fewer
    samples fall back to the overall beta.

    For quick_retrain baseline, maintains a rolling buffer of recent
    scores for recomputing thresholds after drift events.
    """

    CELL_MIN_SAMPLES: int = 11  # len(scores) > 10

    def __init__(
        self,
        default_beta: float = 0.5,
        percentile: float = 95.0,
    ):
        """
        Args:
            default_beta: Default threshold when no cell data is available
            percentile: Percentile for threshold computation (95 = 5% FPR target)
        """
        self.default_beta = default_beta
        self.percentile = percentile

        # 80 thresholds: [num_neighborhoods][num_context_cells]
        self._thresholds: np.ndarray = np.full(
            (NUM_NEIGHBORHOODS, NUM_CONTEXT_CELLS),
            default_beta,
            dtype=np.float32
        )

        # Score tracking per cell: {(nb_id, ctx_id): [scores]}
        self._scores_per_cell: Dict[Tuple[int, int], List[float]] = {}

        # Rolling buffer for quick_retrain baseline
        self._recent_scores: deque = deque(maxlen=5000)

        # Overall beta (computed from all warmup scores)
        self._overall_beta: float = default_beta

        # Whether thresholds have been fitted from warmup data
        self._fitted: bool = False

        # Per-cell counts for debugging
        self._cell_counts: Dict[Tuple[int, int], int] = {}

    def record(self, neighborhood_id: int, cell_id: int, score: float) -> None:
        """
        Record a score for threshold computation.

        This accumulates warmup scores per (neighborhood, context_cell).

        Args:
            neighborhood_id: Neighborhood index (0-9)
            cell_id: Context cell index (0-7)
            score: Anomaly score to record
        """
        key = (int(neighborhood_id), int(cell_id))
        if key not in self._scores_per_cell:
            self._scores_per_cell[key] = []
        self._scores_per_cell[key].append(float(score))
        self._recent_scores.append(float(score))

        # Track counts
        self._cell_counts[key] = len(self._scores_per_cell[key])

    def fit(self, overall_beta: float = None) -> 'ContextBeta':
        """
        Compute thresholds from recorded scores.

        For each (neighborhood, cell) with >= 11 samples, compute
        the percentile threshold. Cells with fewer samples fall back
        to the overall beta.

        Args:
            overall_beta: Override overall beta (default: compute from all scores)

        Returns:
            self for chaining
        """
        if overall_beta is not None:
            self._overall_beta = float(overall_beta)
        elif len(self._recent_scores) > 0:
            self._overall_beta = float(
                np.percentile(list(self._recent_scores), self.percentile)
            )

        fitted_count = 0
        underpopulated_cells = []

        for nb_id in range(NUM_NEIGHBORHOODS):
            for ctx_id in range(NUM_CONTEXT_CELLS):
                key = (nb_id, ctx_id)
                scores = self._scores_per_cell.get(key, [])

                if len(scores) > self.CELL_MIN_SAMPLES - 1:
                    threshold = float(np.percentile(scores, self.percentile))
                    self._thresholds[nb_id, ctx_id] = threshold
                    fitted_count += 1
                else:
                    self._thresholds[nb_id, ctx_id] = self._overall_beta
                    if len(scores) > 0:
                        underpopulated_cells.append(
                            (nb_id, ctx_id, len(scores))
                        )

        self._fitted = True

        return self

    def fit_from_scores(
        self,
        scores: List[float],
        neighborhood_ids: List[int],
        context_ids: List[int],
        overall_beta: float = None,
    ) -> 'ContextBeta':
        """
        Fit thresholds from arrays of scores.

        Args:
            scores: List of anomaly scores
            neighborhood_ids: List of neighborhood indices (0-9)
            context_ids: List of context cell indices (0-7)
            overall_beta: Override overall beta

        Returns:
            self for chaining
        """
        for score, nb_id, ctx_id in zip(scores, neighborhood_ids, context_ids):
            self.record(nb_id, ctx_id, score)

        return self.fit(overall_beta=overall_beta)

    def get_beta(
        self,
        neighborhood_id: int,
        hour: int,
        dow: int,
        ratecode: float,
    ) -> float:
        """
        Get threshold for a record's neighborhood and temporal context.

        This is the primary lookup method used at scoring time.
        Computes context cell ID from hour/dow/ratecode.

        Args:
            neighborhood_id: Neighborhood index (0-9)
            hour: Hour of day (0-23)
            dow: Day of week (0=Mon, 6=Sun)
            ratecode: Rate code (1=Standard, >1=Special)

        Returns:
            Beta threshold for this context
        """
        cell_id = self.compute_cell_id(hour, dow, ratecode)
        return self.get_beta_by_cell(neighborhood_id, cell_id)

    def get_beta_by_cell(
        self,
        neighborhood_id: int,
        cell_id: int,
    ) -> float:
        """
        Get threshold by explicit cell ID.

        Args:
            neighborhood_id: Neighborhood index (0-9)
            cell_id: Context cell index (0-7)

        Returns:
            Beta threshold for this (neighborhood, cell)
        """
        nb = int(neighborhood_id)
        ctx = int(cell_id)

        if 0 <= nb < NUM_NEIGHBORHOODS and 0 <= ctx < NUM_CONTEXT_CELLS:
            return float(self._thresholds[nb, ctx])
        return self._overall_beta

    @staticmethod
    def compute_cell_id(
        hour: int,
        dow: int,
        ratecode: float,
    ) -> int:
        """
        Compute context cell ID from temporal and fare attributes.

        UNIFIES night definition to hour >= 20.
        Previous bug: context_aware.py used hour >= 18.

        Cell ID = (is_special << 2) | (is_night << 1) | is_weekend

        Args:
            hour: Hour of day (0-23)
            dow: Day of week (0=Mon, 6=Sun)
            ratecode: Rate code (1=Standard, >1=Special)

        Returns:
            Context cell ID (0-7)
        """
        is_special = 1 if float(ratecode) > 1.0 else 0
        is_night = 1 if (hour >= 20 or hour < 6) else 0  # UNIFIED: hour >= 20
        is_weekend = 1 if int(dow) >= 5 else 0

        return (is_special << 2) | (is_night << 1) | is_weekend

    def get_cell_name(self, cell_id: int) -> str:
        """Get human-readable context cell name."""
        if 0 <= cell_id < NUM_CONTEXT_CELLS:
            return CONTEXT_CELLS[cell_id]
        return 'unknown'

    def get_neighborhood_name(self, neighborhood_id: int) -> str:
        """Get neighborhood name from index."""
        if 0 <= neighborhood_id < NUM_NEIGHBORHOODS:
            return NEIGHBORHOODS[neighborhood_id]
        return 'unknown'

    @property
    def overall_beta(self) -> float:
        """Get the overall beta threshold."""
        return self._overall_beta

    @property
    def is_fitted(self) -> bool:
        """Whether thresholds have been fitted."""
        return self._fitted

    def get_cell_count(self, neighborhood_id: int, cell_id: int) -> int:
        """Get sample count for a specific cell."""
        return self._cell_counts.get((int(neighborhood_id), int(cell_id)), 0)

    def get_state_dict(self) -> dict:
        """
        Export state for checkpointing.

        Returns:
            Dict with thresholds and metadata
        """
        thresholds_dict = {}
        for nb_id in range(NUM_NEIGHBORHOODS):
            nb_name = NEIGHBORHOODS[nb_id]
            thresholds_dict[nb_name] = {}
            for ctx_id in range(NUM_CONTEXT_CELLS):
                ctx_name = CONTEXT_CELLS[ctx_id]
                thresholds_dict[nb_name][ctx_name] = float(
                    self._thresholds[nb_id, ctx_id]
                )

        return {
            'thresholds': thresholds_dict,
            'thresholds_array': self._thresholds.copy(),
            'default_beta': self.default_beta,
            'percentile': self.percentile,
            'overall_beta': self._overall_beta,
            'fitted': self._fitted,
            'recent_scores': list(self._recent_scores),
            'cell_counts': {f"{k[0]}_{k[1]}": v for k, v in self._cell_counts.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restore state from checkpoint.

        Args:
            state: Dict from get_state_dict()
        """
        if 'thresholds_array' in state:
            self._thresholds = state['thresholds_array'].astype(np.float32)
        elif 'thresholds' in state:
            for nb_name, ctx_dict in state['thresholds'].items():
                nb_id = NEIGHBORHOOD_TO_ID.get(nb_name, -1)
                if nb_id < 0:
                    continue
                for ctx_name, threshold in ctx_dict.items():
                    ctx_id = CONTEXT_CELLS.index(ctx_name) if ctx_name in CONTEXT_CELLS else -1
                    if ctx_id >= 0:
                        self._thresholds[nb_id, ctx_id] = float(threshold)

        self.default_beta = state.get('default_beta', 0.5)
        self.percentile = state.get('percentile', 95.0)
        self._overall_beta = state.get('overall_beta', self.default_beta)
        self._fitted = state.get('fitted', True)

        if 'recent_scores' in state:
            self._recent_scores = deque(
                state['recent_scores'],
                maxlen=5000
            )

        if 'cell_counts' in state:
            self._cell_counts = {
                tuple(map(int, k.split('_'))): v
                for k, v in state['cell_counts'].items()
            }

    def quick_retrain(self, recent_scores: List[float]) -> float:
        """
        Recompute overall beta from recent scores.

        Used after drift events to update the fallback threshold
        from the rolling recent_scores buffer.

        Args:
            recent_scores: New scores to incorporate

        Returns:
            New overall beta value
        """
        all_scores = list(self._recent_scores) + list(recent_scores)
        if len(all_scores) < self.CELL_MIN_SAMPLES:
            return self._overall_beta

        self._overall_beta = float(np.percentile(all_scores, self.percentile))

        for nb_id in range(NUM_NEIGHBORHOODS):
            for ctx_id in range(NUM_CONTEXT_CELLS):
                key = (nb_id, ctx_id)
                if key in self._scores_per_cell:
                    scores = self._scores_per_cell[key]
                    if len(scores) > self.CELL_MIN_SAMPLES - 1:
                        self._thresholds[nb_id, ctx_id] = float(
                            np.percentile(scores, self.percentile)
                        )
                    else:
                        self._thresholds[nb_id, ctx_id] = self._overall_beta

        return self._overall_beta

    def summary(self) -> str:
        """Get a human-readable summary of the threshold matrix."""
        lines = [
            f"ContextBeta Summary (fitted={self._fitted})",
            f"  Overall beta: {self._overall_beta:.4f}",
            f"  Cell minimum: {self.CELL_MIN_SAMPLES} samples",
            "",
        ]

        for nb_id, nb_name in enumerate(NEIGHBORHOODS):
            lines.append(f"  {nb_name}:")
            for ctx_id in range(NUM_CONTEXT_CELLS):
                count = self.get_cell_count(nb_id, ctx_id)
                beta = self._thresholds[nb_id, ctx_id]
                ctx_name = CONTEXT_CELLS[ctx_id]
                lines.append(
                    f"    [{ctx_id}] {ctx_name}: beta={beta:.4f} "
                    f"(n={count})"
                )

        return "\n".join(lines)


def get_context_id(
    hour: int,
    dow: int,
    ratecode: float,
) -> int:
    """
    Compute context cell ID (standalone utility).

    UNIFIES night definition to hour >= 20.

    Args:
        hour: Hour of day (0-23)
        dow: Day of week (0=Mon, 6=Sun)
        ratecode: Rate code (1=Standard, >1=Special)

    Returns:
        Context cell ID (0-7)
    """
    return ContextBeta.compute_cell_id(hour, dow, ratecode)


__all__ = [
    'ContextBeta',
    'NEIGHBORHOODS',
    'NUM_NEIGHBORHOODS',
    'NEIGHBORHOOD_TO_ID',
    'ID_TO_NEIGHBORHOOD',
    'CONTEXT_CELLS',
    'NUM_CONTEXT_CELLS',
    'get_context_id',
]
