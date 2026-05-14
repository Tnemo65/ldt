"""
Centralized Hyperparameters for CA-DQStream + MemStream.

This module re-exports MemStreamConfig and provides shared constants
used across the core modules.
"""

from memstream_src.core.memstream_core import MemStreamConfig


# =============================================================================
# Default Hyperparameters
# =============================================================================

# Architecture (v10 benchmark: in_dim=34, hidden_dim=68)
DEFAULT_IN_DIM: int = 34
DEFAULT_HIDDEN_DIM: int = 68
DEFAULT_OUT_DIM: int = 34

# Memory (v10 benchmark: 256)
DEFAULT_MEMORY_LEN: int = 256
DEFAULT_MEMORY_INIT_FRACTION: float = 0.1

# Training (warmup) - v10: 20 epochs
DEFAULT_WARMUP_LR: float = 1e-3
DEFAULT_WARMUP_EPOCHS: int = 20
DEFAULT_WARMUP_BATCH_SIZE: int = 256
DEFAULT_WARMUP_NOISE_STD: float = 0.1
DEFAULT_WARMUP_GRADIENT_CLIP: float = 1.0
DEFAULT_WARMUP_EARLY_STOP_PATIENCE: int = 20

# Scoring
DEFAULT_BETA: float = 0.5
DEFAULT_LATENCY_WARNING_MS: float = 50.0

# Determinism
DEFAULT_SEED: int = 42

# Context-Aware (40D)
CONTEXT_RAW_DIM: int = 25
CONTEXT_NBR_DIM: int = 6
CONTEXT_HOUR_DIM: int = 4
CONTEXT_DAY_DIM: int = 2
CONTEXT_TRIP_DIM: int = 3
CONTEXT_TOTAL_DIM: int = 40  # 25 + 6 + 4 + 2 + 3


# =============================================================================
# BAR Controller Defaults
# =============================================================================

BAR_MIN_BUDGET_FRACTION: float = 0.01  # 1%
BAR_MAX_BUDGET_FRACTION: float = 0.05   # 5%
BAR_ADWIN_DELTA: float = 0.002
BAR_WINDOW_SIZE: int = 10000


# =============================================================================
# Feature Extraction
# =============================================================================

NUM_FEATURES: int = 25
FEATURE_NAMES: list = [
    'pickup_hour_sin',
    'pickup_hour_cos',
    'dropoff_hour_sin',
    'dropoff_hour_cos',
    'pickup_day_of_week_sin',
    'pickup_day_of_week_cos',
    'dropoff_day_of_week_sin',
    'dropoff_day_of_week_cos',
    'trip_distance',
    'fare_amount',
    'extra',
    'mta_tax',
    'tip_amount',
    'tolls_amount',
    'improvement_surcharge',
    'total_amount',
    'passenger_count',
    'trip_duration_minutes',
    'speed_mph',
    'fare_per_mile',
    'is_airport_trip',
    'is_rush_hour',
    'is_weekend',
    'pickup_borough_encoded',
    'dropoff_borough_encoded',
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_default_config() -> MemStreamConfig:
    """Create a MemStreamConfig with default hyperparameters."""
    return MemStreamConfig()


def get_context_aware_config(in_dim: int = CONTEXT_TOTAL_DIM) -> MemStreamConfig:
    """Create a MemStreamConfig for context-aware mode (40D input)."""
    cfg = MemStreamConfig()
    cfg.in_dim = in_dim
    cfg.out_dim = in_dim
    return cfg


__all__ = [
    'MemStreamConfig',
    # Architecture
    'DEFAULT_IN_DIM',
    'DEFAULT_HIDDEN_DIM',
    'DEFAULT_OUT_DIM',
    # Memory
    'DEFAULT_MEMORY_LEN',
    'DEFAULT_MEMORY_INIT_FRACTION',
    # Training
    'DEFAULT_WARMUP_LR',
    'DEFAULT_WARMUP_EPOCHS',
    'DEFAULT_WARMUP_BATCH_SIZE',
    'DEFAULT_WARMUP_NOISE_STD',
    'DEFAULT_WARMUP_GRADIENT_CLIP',
    'DEFAULT_WARMUP_EARLY_STOP_PATIENCE',
    # Scoring
    'DEFAULT_BETA',
    'DEFAULT_LATENCY_WARNING_MS',
    # Determinism
    'DEFAULT_SEED',
    # Context-Aware
    'CONTEXT_RAW_DIM',
    'CONTEXT_NBR_DIM',
    'CONTEXT_HOUR_DIM',
    'CONTEXT_DAY_DIM',
    'CONTEXT_TRIP_DIM',
    'CONTEXT_TOTAL_DIM',
    # BAR Controller
    'BAR_MIN_BUDGET_FRACTION',
    'BAR_MAX_BUDGET_FRACTION',
    'BAR_ADWIN_DELTA',
    'BAR_WINDOW_SIZE',
    # Feature Extraction
    'NUM_FEATURES',
    'FEATURE_NAMES',
    # Helpers
    'get_default_config',
    'get_context_aware_config',
]
