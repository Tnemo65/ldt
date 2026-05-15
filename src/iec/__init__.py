"""
IEC (Intelligent Evolution Controller) package for MemStream.

Phase 2D Migration: IEC Operators + Security Hardening

Components:
- circuit_breaker.py: Circuit breaker for IEC operations
- drift_aggregator.py: Severity assessment + strategy prediction
- adwin_multi_instance.py: Multi-instance ADWIN for drift detection
- verification_feedback.py: Tracks adaptation effectiveness
- iec_controller.py: Main IEC controller with HMAC security fixes

HMAC Security:
- All beta writes to MinIO require HMAC signature
- IEC_SIGNING_KEY environment variable is mandatory
- HMAC verification failures trigger Prometheus alerts
"""

from .circuit_breaker import CircuitBreaker, CircuitState
from .drift_aggregator import DriftAggregator, SeverityLevel, EvolutionStrategy
from .adwin_multi_instance import MultiInstanceADWIN
from .verification_feedback import VerificationFeedbackLoop
from .iec_controller import (
    IECConfig,
    IECController,
    create_iec_controller,
    verify_beta_hmac,
    IEC_SIGNING_KEY_ENV,
)

__all__ = [
    # Circuit breaker
    'CircuitBreaker',
    'CircuitState',
    # Drift aggregation
    'DriftAggregator',
    'SeverityLevel',
    'EvolutionStrategy',
    # ADWIN
    'MultiInstanceADWIN',
    # Verification
    'VerificationFeedbackLoop',
    # Main controller
    'IECConfig',
    'IECController',
    'create_iec_controller',
    'verify_beta_hmac',
    'IEC_SIGNING_KEY_ENV',
]
