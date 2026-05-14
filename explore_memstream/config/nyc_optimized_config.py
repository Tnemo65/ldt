"""
NYC Taxi-Optimized Hyperparameters for MemStream.

Derived from:
- MemStream paper (Wang et al., 2022)
- CA-DQStream v10 benchmark results
- NYC taxi data characteristics (weekly periodicity, rush-hour patterns)

Reference: memstream_src/core/config.py for defaults.
"""

from memstream_src.core.config import (
    DEFAULT_IN_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_OUT_DIM,
    DEFAULT_MEMORY_LEN,
    DEFAULT_WARMUP_LR,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WARMUP_BATCH_SIZE,
    DEFAULT_WARMUP_NOISE_STD,
    DEFAULT_WARMUP_GRADIENT_CLIP,
    DEFAULT_WARMUP_EARLY_STOP_PATIENCE,
    DEFAULT_BETA,
    DEFAULT_SEED,
    BAR_MIN_BUDGET_FRACTION,
    BAR_MAX_BUDGET_FRACTION,
    BAR_ADWIN_DELTA,
    BAR_WINDOW_SIZE,
)

# NYC taxi-specific overrides
NYC_IN_DIM = 34
NYC_HIDDEN_DIM = 68
NYC_MEMORY_LEN = 1024  # Cover ~1 day of NYC taxi diversity
NYC_WARMUP_EPOCHS = 100  # Early stopping typically triggers at 20-100

NYC_ABLATION_BEST = {
    'memory_len': 1024,
    'hidden_dim': 68,
    'k': 10,
    'gamma': 0.5,  # Self-recovery from memory poisoning
    'default_beta': 0.5,
    'warmup_epochs': 100,
    'warmup_lr': 1e-3,
    'warmup_batch_size': 256,
    'warmup_noise_std': 0.1,
}

NYC_BAR_CONFIG = {
    'target_bar_rate': 0.02,  # 2% label budget
    'adwin_delta': 0.002,
    'min_budget_fraction': 0.01,  # 1% minimum guarantee
    'max_budget_fraction': 0.05,  # 5% maximum
    'bar_window_size': 10000,
}

NYC_CONTEXT_BETA = {
    'n_neighborhoods': 10,
    'n_cells': 8,  # (special/standard) x (day/night) x (weekday/weekend)
    'percentile': 95,  # Threshold from warmup score distribution
}


def get_nyc_config():
    """Return MemStreamConfig with NYC-optimized parameters."""
    from memstream_src.core.memstream_core import MemStreamConfig
    cfg = MemStreamConfig()
    for key, val in NYC_ABLATION_BEST.items():
        setattr(cfg, key, val)
    return cfg
