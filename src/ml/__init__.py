"""
ML Module - MemStream and related components.

Exports:
- MemStreamCore: Full MemStream implementation (AE + Memory)
- MemStreamConfig: Configuration for MemStream
- BARController: Budget Allocation Rate controller
- ADWIN_U: Higher-order statistics drift detection (replaces SimpleADWIN)
- set_determinism: Set random seeds for reproducibility
"""

from src.ml.memstream_core import (
    MemStreamCore,
    MemStreamConfig,
    MemStreamAE,
    SimpleADWIN,
    BARController,
    set_determinism,
)
from src.ml.memstream_context_beta import ContextBeta
from src.ml.adwin_u import ADWIN_U

__all__ = [
    'MemStreamCore',
    'MemStreamConfig',
    'MemStreamAE',
    'SimpleADWIN',  # DEPRECATED - use ADWIN_U
    'BARController',
    'set_determinism',
    'ContextBeta',
    'ADWIN_U',
]
