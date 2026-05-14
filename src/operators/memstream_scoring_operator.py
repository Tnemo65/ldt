"""
MemStream Scoring Operator for Layer 2 Complex Branch.

This operator replaces MockIFScoringOperator with Full MemStream:
- Denoising Autoencoder (30D -> 60D -> 30D)
- Memory Module (FIFO queue)
- ADWIN Drift Detection per neighborhood
- BAR Controller (Budget Allocation Rate)

Pipeline Flow:
  valid_stream -> MemStreamScoringOperator -> complex_stream

Usage:
  complex_stream = valid_stream.map(MemStreamScoringOperator())
"""

"""
ADWIN SCOPE SEPARATION - CRITICAL ARCHITECTURE DECISION
=====================================================

This file uses a LOCAL ADWIN (inside MemStreamCore) for BAR Controller decisions.
This is DIFFERENT from the GLOBAL ADWIN-U used in IECOperator (Layer 4).

LOCAL ADWIN (MemStreamCore):
  - Scope: MICRO (per neighborhood, per record)
  - Metric: anomaly_score from scoring
  - Purpose: BAR budget decision - should memory be updated?
  - Question: "Should I remember this record?"
  - Output: should_update_memory (bool)

GLOBAL ADWIN-U (IECOperator):
  - Scope: MACRO (system-wide, per 1-minute window)
  - Metrics: 6 meta-metrics from MetaAggregator
  - Purpose: METER strategy decision
  - Question: "Should I change system strategy?"
  - Output: drift events for METER

These two ADWIN systems operate INDEPENDENTLY:
  - LOCAL ADWIN does NOT affect GLOBAL ADWIN-U
  - GLOBAL ADWIN-U does NOT touch MemStream memory
"""

from pyflink.datastream import MapFunction
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.common.typeinfo import BasicTypeInfo
import logging
import pickle
import io
import os
from typing import Optional, Dict

import numpy as np

LOGGER = logging.getLogger('memstream-scoring')


# =============================================================================
# Configuration
# =============================================================================

# Model checkpoint path
MODEL_CHECKPOINT_DIR = os.getenv('MEMSTREAM_CHECKPOINT_DIR', '/models/memstream')
MEMORY_CHECKPOINT_INTERVAL = int(os.getenv('MEMSTREAM_CHECKPOINT_INTERVAL', '1000'))

# Default MemStream config for production
DEFAULT_CONFIG = {
    'in_dim': 30,
    'hidden_dim': 60,
    'out_dim': 30,
    'memory_len': 50000,  # 50K samples for production
    'warmup_epochs': 500,
    'warmup_batch_size': 256,
    'warmup_noise_std': 0.1,
    'default_beta': 0.5,
    'k_neighbors': 10,
    'seed': 42,
}

# Neighborhood zone mappings
MANHATTAN_ZONES = set(range(1, 51))
BROOKLYN_ZONES = set(range(51, 101))
QUEENS_ZONES = set(range(101, 151))
BRONX_ZONES = set(range(151, 201))
AIRPORT_ZONES = {132, 138}
STATEN_ISLAND_ZONES = set(range(201, 264))


# =============================================================================
# Helper Functions
# =============================================================================

def get_neighborhood(zone_id: int) -> str:
    """Get neighborhood from zone ID."""
    if zone_id in MANHATTAN_ZONES:
        return 'manhattan'
    elif zone_id in BROOKLYN_ZONES:
        return 'brooklyn'
    elif zone_id in QUEENS_ZONES:
        return 'queens'
    elif zone_id in BRONX_ZONES:
        return 'bronx'
    elif zone_id in AIRPORT_ZONES:
        return 'airport'
    elif zone_id in STATEN_ISLAND_ZONES:
        return 'staten_island'
    else:
        return 'unknown'


def get_context_key(record: Dict) -> str:
    """Create context key for tracking."""
    zone_id = int(float(record.get('PULocationID', 1)))
    neighborhood = get_neighborhood(zone_id)
    return neighborhood


# =============================================================================
# MemStreamScoringOperator
# =============================================================================

class MemStreamScoringOperator(MapFunction):
    """
    Full MemStream Scoring Operator for Layer 2 Complex Branch.

    This operator:
    1. Extracts 30D features from NYC taxi records
    2. Scores records using MemStream (AE + Memory)
    3. Tracks drift per neighborhood using ADWIN
    4. Controls memory updates using BAR Controller

    Outputs enriched record with:
    - anomaly_score: float
    - is_anomaly: bool
    - threshold: float
    - context_key: str
    - neighborhood: str
    - drift_detected: bool
    - bar_rate: float
    - memory_updates: int
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        warmup_data: Optional[np.ndarray] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize MemStream Scoring Operator.

        Args:
            config: MemStream configuration dict
            warmup_data: Pre-computed feature matrix for warmup (optional)
            checkpoint_dir: Directory for memory checkpoints
        """
        self.config = config or DEFAULT_CONFIG.copy()
        self.warmup_data = warmup_data
        self.checkpoint_dir = checkpoint_dir or MODEL_CHECKPOINT_DIR

        # These will be initialized in open()
        self._ms_core = None
        self._feature_vectorizer = None
        self._bar_controller = None
        self._cfg = None  # Store MemStreamConfig reference
        self._checkpoint_counter = 0

        # Statistics
        self._total_scored = 0
        self._total_anomalies = 0
        self._total_memory_updates = 0

    def open(self, runtime_context):
        """Initialize MemStream core and load/checkpoint."""
        # Import here to avoid circular imports
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.ml.memstream_core import (
            MemStreamCore, MemStreamConfig, BARController, set_determinism
        )
        from src.features.vectorizer_25d import FeatureVectorizer25D

        LOGGER.info("[MemStreamScoring] Initializing...")

        # Set determinism
        set_determinism(self.config['seed'])

        # Initialize feature vectorizer
        self._feature_vectorizer = FeatureVectorizer25D()
        LOGGER.info("[MemStreamScoring] Feature vectorizer initialized (25D)")

        # Create MemStream config
        cfg = MemStreamConfig()
        cfg.in_dim = self.config['in_dim']
        cfg.hidden_dim = self.config['hidden_dim']
        cfg.out_dim = self.config['out_dim']
        cfg.memory_len = self.config['memory_len']
        cfg.warmup_epochs = self.config['warmup_epochs']
        cfg.warmup_batch_size = self.config['warmup_batch_size']
        cfg.warmup_noise_std = self.config['warmup_noise_std']
        cfg.default_beta = self.config['default_beta']
        cfg.k_neighbors = self.config['k_neighbors']
        cfg.seed = self.config['seed']

        # Store config reference for checkpointing
        self._cfg = cfg

        # Initialize MemStream core
        self._ms_core = MemStreamCore(cfg=cfg, device='cpu')
        LOGGER.info("[MemStreamScoring] MemStreamCore initialized")

        # Initialize BAR controller
        # LOCAL ADWIN: Monitors anomaly_score for BAR budget decision
        # See module docstring for scope separation details
        self._bar_controller = BARController(
            memory_len=cfg.memory_len,
            min_budget_fraction=0.01,
            max_budget_fraction=0.05,
            adwin_delta=0.002,
        )

        # Try to load from checkpoint (REQUIRED - warmup must be done offline)
        if self._try_load_checkpoint():
            LOGGER.info("[MemStreamScoring] Loaded pre-trained checkpoint")
            LOGGER.info("[MemStreamScoring] Warmup done OFFLINE by scripts/warmup_memstream.py")
        else:
            # FATAL: No checkpoint found - warmup was not done offline
            LOGGER.error("[MemStreamScoring] FATAL: No checkpoint found!")
            LOGGER.error("[MemStreamScoring] Run scripts/warmup_memstream.py BEFORE starting Flink job")
            raise RuntimeError(
                "MemStream checkpoint missing - run offline warmup script first:\n"
                "  python scripts/warmup_memstream.py --data <baseline_data> --output /models/memstream"
            )

        # Create checkpoint directory if needed
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        except PermissionError:
            LOGGER.warning(
                "[MemStreamScoring] Cannot create checkpoint dir %s (permission denied). "
                "Checkpointing disabled.", self.checkpoint_dir
            )
            self.checkpoint_dir = None

        LOGGER.info(f"[MemStreamScoring] Ready. Memory size: {cfg.memory_len}")

    def _try_load_checkpoint(self) -> bool:
        """Try to load model checkpoint (AE weights, scaler, and memory state)."""
        import torch

        weights_path = os.path.join(self.checkpoint_dir, 'memstream_weights.pt')
        memory_path = os.path.join(self.checkpoint_dir, 'memstream_memory.pt')

        # All checkpoint files must exist
        if not os.path.exists(memory_path):
            LOGGER.warning(f"[MemStreamScoring] Memory checkpoint not found: {memory_path}")
            return False
        if not os.path.exists(weights_path):
            LOGGER.warning(f"[MemStreamScoring] Weights checkpoint not found: {weights_path}")
            return False

        try:
            # Load AE weights and normalization stats
            weights_state = torch.load(weights_path, map_location='cpu', weights_only=True)
            self._ms_core.ae.load_state_dict(weights_state['ae_state_dict'])
            self._ms_core.mean = weights_state['mean'].to(self._ms_core.device)
            self._ms_core.std = weights_state['std'].to(self._ms_core.device)
            LOGGER.info("[MemStreamScoring] Loaded AE weights and scaler")

            # Load memory state
            mem_state = torch.load(memory_path, map_location='cpu', weights_only=True)
            self._ms_core.load_state_dict(mem_state)

            # Load bar controller stats if available
            bar_path = os.path.join(self.checkpoint_dir, 'bar_controller.pt')
            if os.path.exists(bar_path):
                bar_state = torch.load(bar_path, map_location='cpu', weights_only=True)
                self._bar_controller._total_records = bar_state.get('total_records', 0)
                self._bar_controller._memory_updates = bar_state.get('memory_updates', 0)
                self._bar_controller._drift_events = bar_state.get('drift_events', 0)

            return True
        except Exception as e:
            LOGGER.warning(f"[MemStreamScoring] Failed to load checkpoint: {e}")
            return False

    def _save_checkpoint(self):
        """Save model checkpoint (AE weights, scaler, and memory state)."""
        if self.checkpoint_dir is None:
            return
        try:
            import torch

            # Save AE weights and normalization stats
            weights_path = os.path.join(self.checkpoint_dir, 'memstream_weights.pt')
            torch.save({
                'ae_state_dict': self._ms_core.ae.state_dict(),
                'mean': self._ms_core.mean.cpu(),
                'std': self._ms_core.std.cpu(),
                'cfg_in_dim': self._cfg.in_dim,
                'cfg_hidden_dim': self._cfg.hidden_dim,
            }, weights_path)

            # Save scaler
            scaler_path = os.path.join(self.checkpoint_dir, 'memstream_scaler.pkl')
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump({
                    'mean': self._ms_core.mean.cpu(),
                    'std': self._ms_core.std.cpu(),
                }, f)

            # Save memory state
            memory_path = os.path.join(self.checkpoint_dir, 'memstream_memory.pt')
            state = self._ms_core.get_state_dict()
            torch.save(state, memory_path)

            # Save bar controller stats
            bar_path = os.path.join(self.checkpoint_dir, 'bar_controller.pt')
            torch.save({
                'total_records': self._bar_controller._total_records,
                'memory_updates': self._bar_controller._memory_updates,
                'drift_events': self._bar_controller._drift_events,
            }, bar_path)

            self._checkpoint_counter = 0
            LOGGER.info("[MemStreamScoring] Checkpoint saved")
        except Exception as e:
            LOGGER.error(f"[MemStreamScoring] Failed to save checkpoint: {e}")

    def map(self, value):
        """
        Score a single record using MemStream.

        Args:
            value: Record dict from Layer 1

        Returns:
            Enriched record with anomaly_score, is_anomaly, etc.
        """
        if value is None:
            return None

        try:
            # Extract features
            features = self._feature_vectorizer.transform(value)

            if features is None:
                # Invalid record, return with error flag
                return {
                    **value,
                    'anomaly_score': -1.0,
                    'threshold': self.config['default_beta'],
                    'is_anomaly': False,
                    'context_key': 'error',
                    'neighborhood': 'unknown',
                    'drift_detected': False,
                    'bar_rate': 0.0,
                    'bar_update_reason': 'none',
                    'ml_model': 'memstream_v1',
                    'scoring_error': 'feature_extraction_failed',
                }

            # Get neighborhood for per-neighborhood tracking
            neighborhood = get_context_key(value)

            # Score with MemStream
            score = self._ms_core.score_one(features)
            threshold = self.config['default_beta']
            is_anomaly = score > threshold

            # Update statistics
            self._total_scored += 1
            if is_anomaly:
                self._total_anomalies += 1

            # BAR controller: decide if memory should be updated
            should_update, reason = self._bar_controller.should_update_memory(
                neighborhood, score
            )

            if should_update:
                self._ms_core.memory_update(features)
                self._total_memory_updates += 1

                # Check for drift
                if reason == "drift_detected":
                    LOGGER.info(f"[MemStreamScoring] Drift detected in {neighborhood}")

            # Periodic checkpoint
            self._checkpoint_counter += 1
            if self._checkpoint_counter >= MEMORY_CHECKPOINT_INTERVAL:
                self._save_checkpoint()

            # Build result
            result = {
                **value,
                'anomaly_score': float(score),
                'threshold': float(threshold),
                'is_anomaly': bool(is_anomaly),
                'context_key': neighborhood,
                'neighborhood': neighborhood,
                'drift_detected': reason == "drift_detected",
                'bar_rate': float(self._bar_controller.bar_rate),
                'bar_rate_pct': float(self._bar_controller.bar_rate_pct),
                'bar_update_reason': reason,
                'memory_updates': self._total_memory_updates,
                'ml_model': 'memstream_v1',
            }

            return result

        except Exception as e:
            LOGGER.error(f"[MemStreamScoring] Error scoring record: {e}")
            return {
                **value,
                'anomaly_score': -1.0,
                'threshold': self.config['default_beta'],
                'is_anomaly': False,
                'context_key': 'error',
                'neighborhood': 'unknown',
                'drift_detected': False,
                'bar_rate': 0.0,
                'bar_update_reason': 'none',
                'ml_model': 'memstream_v1',
                'scoring_error': str(e),
            }

    def close(self):
        """Save checkpoint on close."""
        if self._ms_core is not None:
            self._save_checkpoint()
            LOGGER.info(f"[MemStreamScoring] Closed. Total scored: {self._total_scored}, "
                       f"Anomalies: {self._total_anomalies}, Memory updates: {self._total_memory_updates}")
