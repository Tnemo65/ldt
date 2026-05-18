"""
IEC (Intelligent Evolution Controller) package for MemStream.

Phase 3 (Sequential Pipeline): Simplified to 2 strategies.

Components:
- drift_aggregator.py: Severity assessment + 2-strategy prediction
- adwin_multi_instance.py: Multi-instance ADWIN for drift detection
- verification_feedback.py: Tracks adaptation effectiveness
- iec_controller.py: Main IEC controller (do_nothing / quick_retrain)

Strategies:
- do_nothing: no drift or minor drift
- quick_retrain: severe drift — retrain MemStream AE + reset ADWIN thresholds

HMAC Security:
- Retrain signals written to MinIO require HMAC signature
- IEC_SIGNING_KEY environment variable is mandatory
"""

from .drift_aggregator import DriftAggregator, SeverityLevel, EvolutionStrategy
from .adwin_multi_instance import MultiInstanceADWIN
from .verification_feedback import VerificationFeedbackLoop
from .iec_controller import (
    IECConfig,
    IECController,
    create_iec_controller,
    verify_retrain_signal,
    IEC_SIGNING_KEY_ENV,
)

__all__ = [
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
    'verify_retrain_signal',
    'IEC_SIGNING_KEY_ENV',
]
