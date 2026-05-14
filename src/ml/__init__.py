"""
ML Module - MemStream and related components.

Exports:
- MemStreamCore: Full MemStream implementation (AE + Memory)
- MemStreamConfig: Configuration for MemStream
- BARController: Budget Allocation Rate controller
- ADWIN: Drift detection
- set_determinism: Set random seeds for reproducibility
"""

from src.ml.memstream_core import (
    MemStreamCore,
    MemStreamConfig,
    MemStreamAE,
    MemoryModule,
    ADWIN,
    BARController,
    set_determinism,
)

__all__ = [
    'MemStreamCore',
    'MemStreamConfig',
    'MemStreamAE',
    'MemoryModule',
    'ADWIN',
    'BARController',
    'set_determinism',
]
