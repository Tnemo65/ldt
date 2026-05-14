"""CA-DQStream + MemStream Core Module.

This module contains the core MemStream implementation with:
- MemStreamAE: Autoencoder neural network
- MemoryModule: FIFO circular buffer for normal patterns
- MemStreamCore: Main anomaly detection class
- ADWIN: Adaptive drift detection
- BARController: Budget Allocation Rate controller
- FeatureVectorizer: 25D feature extraction for NYC taxi data
- ContextAwareFeatureVectorizer: 40D context-aware feature extraction

Based on: Zhang et al., WWW 2022
"""

from memstream_src.core.memstream_core import (
    MemStreamCore,
    MemStreamAE,
    MemoryModule,
    MemStreamConfig,
    SecurityError,
    set_determinism,
    BARController,
    get_context_key,
)

from memstream_src.core.feature_extractor import (
    FeatureVectorizer,
    FEATURE_NAMES,
    NUM_FEATURES,
)

from memstream_src.core.context_aware import (
    ADWIN,
    NEIGHBORHOODS,
    NEIGHBORHOOD_TO_ID,
    ID_TO_NEIGHBORHOOD,
    ContextBeta,
    NeighborhoodADWINManager,
    EIAIntegration,
    get_4d_context,
    get_neighborhood_from_zone,
)

from memstream_src.core.zone_mapping import (
    NEIGHBORHOODS,
    ZONE_TO_NEIGHBORHOOD,
    HOUR_BUCKETS,
    DAY_TYPES,
    TRIP_TYPES,
    get_neighborhood_from_zone,
)

from memstream_src.core.adwin_multi_instance import (
    ADWIN as ADWINMultiInstance,
    MultiInstanceADWIN,
)

from memstream_src.core.drift_aggregator import (
    SeverityLevel,
    EvolutionStrategy,
    DriftAggregator,
)

from memstream_src.core.iec_controller import (
    IECConfig,
    IECController,
    create_iec_controller,
    VerificationFeedbackLoop,
)

from memstream_src.core.iec import (
    IEC_CONFIG,
    IECController as IECControllerUnified,
    create_iec_controller as create_iec_unified,
)

__all__ = [
    # MemStream Core
    'MemStreamCore',
    'MemStreamAE',
    'MemoryModule',
    'MemStreamConfig',
    'SecurityError',
    'set_determinism',
    'BARController',
    'get_context_key',
    # Feature Extraction
    'FeatureVectorizer',
    'FEATURE_NAMES',
    'NUM_FEATURES',
    # Context-Aware
    'ADWIN',
    'NEIGHBORHOODS',
    'NEIGHBORHOOD_TO_ID',
    'ID_TO_NEIGHBORHOOD',
    'ContextBeta',
    'NeighborhoodADWINManager',
    'EIAIntegration',
    'get_4d_context',
    'get_neighborhood_from_zone',
    # Zone Mapping
    'ZONE_TO_NEIGHBORHOOD',
    'HOUR_BUCKETS',
    'DAY_TYPES',
    'TRIP_TYPES',
    # Multi-Instance ADWIN (IEC)
    'ADWINMultiInstance',
    'MultiInstanceADWIN',
    # Drift Aggregation
    'SeverityLevel',
    'EvolutionStrategy',
    'DriftAggregator',
    # IEC Controller
    'IECConfig',
    'IECController',
    'create_iec_controller',
    'VerificationFeedbackLoop',
    # IEC Unified (with config)
    'IEC_CONFIG',
    'IECControllerUnified',
    'create_iec_unified',
]
