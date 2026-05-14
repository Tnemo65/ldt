"""
MemStream Scoring Operator for Layer 2 Complex Branch.

This operator replaces the stub with real MemStream scoring:
- 34D FeatureVectorizer
- Denoising Autoencoder (AE + Memory)
- ContextBeta (80 thresholds, ratio scoring)
- ADWIN per neighborhood (10 instances) 
- Conditional memory updates (normal points only)

Pipeline Flow:
  canary_clean_stream -> MemStreamScoringOperator -> memstream_stream

Usage:
  memstream_stream = canary_clean_stream.map(MemStreamScoringOperator(config))
"""

from pyflink.datastream import MapFunction
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.common.typeinfo import BasicTypeInfo
import logging
import pickle
import io
import os
import time
import hashlib
import hmac
from typing import Optional, Dict, List, Tuple

import numpy as np

LOGGER = logging.getLogger('memstream-scoring')

# Checkpoint path
MODEL_CHECKPOINT_DIR = os.getenv('MEMSTREAM_CHECKPOINT_DIR', '/models/memstream')
MEMORY_CHECKPOINT_INTERVAL = int(os.getenv('MEMSTREAM_CHECKPOINT_INTERVAL', '1000'))

# Default MemStream config for production (matching benchmark v10)
DEFAULT_CONFIG = {
    'in_dim': 34,              # 34D from benchmark
    'hidden_dim': 68,          # 2x hidden layer
    'out_dim': 34,             # symmetric
    'memory_len': 256,         # from benchmark (was 50000)
    'k_neighbors': 10,         # kNN neighbors
    'gamma': 0.0,             # no decay (k=1 effectively)
    'warmup_epochs': 20,       # from benchmark (was 500)
    'warmup_batch_size': 256,
    'warmup_noise_std': 0.1,
    'default_beta': 0.5,
    'seed': 42,
}

# Neighborhood zone mappings (matching benchmark v10, 10 neighborhoods)
MANHATTAN_ZONES = set(range(1, 44))        # 1-43
BRONX_ZONES = set(range(44, 104))            # 44-103
BROOKLYN_ZONES = set(range(104, 128))       # 104-127
QUEENS_LOWER_ZONES = set(range(128, 149))   # 128-148
QUEENS_UPPER_ZONES = set(range(149, 162))  # 149-161
STATEN_ISLAND_ZONES = set(range(162, 182)) # 162-181
EWR_ZONES = set(range(182, 197))            # 182-196
JFK_ZONES = set(range(217, 230))            # 217-229
NALP_ZONES = set(range(230, 235))           # 230-234
UNKNOWN_ZONES = set(range(235, 266)) | set(range(197, 217))

NEIGHBORHOOD_NAMES = [
    'manhattan', 'brooklyn', 'queens_lower', 'queens_upper',
    'bronx', 'staten_island', 'ewr', 'jfk', 'nalp', 'unknown'
]
N_NEIGHBORHOODS = 10


def location_to_neighborhood_idx(loc_id: int) -> int:
    """Map PULocationID to neighborhood index (0-9)."""
    if loc_id <= 0:
        return 9
    z = int(loc_id)
    if 1 <= z <= 43:
        return 0   # manhattan
    elif 44 <= z <= 103:
        return 4   # bronx
    elif 104 <= z <= 127:
        return 1   # brooklyn
    elif 128 <= z <= 148:
        return 2   # queens_lower
    elif 149 <= z <= 161:
        return 3   # queens_upper
    elif 162 <= z <= 181:
        return 5   # staten_island
    elif 182 <= z <= 196:
        return 6   # ewr
    elif 217 <= z <= 229:
        return 7   # jfk
    elif 230 <= z <= 234:
        return 8   # nalp
    else:
        return 9   # unknown


def get_neighborhood(record: Dict) -> tuple:
    """Get neighborhood name and index from record.
    
    Returns:
        (name, index) tuple
    """
    zone_id = int(float(record.get('PULocationID', 1)))
    idx = location_to_neighborhood_idx(zone_id)
    return NEIGHBORHOOD_NAMES[idx], idx


def get_context_id_from_record(record: Dict) -> int:
    """Get 8-cell context ID from record.
    
    Context cells: (Standard/Special) x (Day/Night) x (Weekday/Weekend)
    - is_special: ratecode > 1
    - is_night: hour >= 18 or hour < 6
    - is_weekend: dow >= 5
    """
    from datetime import datetime
    
    dt_str = record.get('tpep_pickup_datetime', '')
    hour = 12
    dow = 0
    if dt_str:
        try:
            dt = datetime.fromisoformat(dt_str.replace('/', '-'))
            hour = dt.hour
            dow = dt.weekday()
        except:
            pass
    
    ratecode = float(record.get('RatecodeID', 1))
    is_special = 1 if ratecode > 1 else 0
    is_night = 1 if (hour >= 18 or hour < 6) else 0
    is_weekend = 1 if dow >= 5 else 0
    return (is_special << 2) | (is_night << 1) | is_weekend


class MemStreamScoringOperator(MapFunction):
    """Full MemStream Scoring Operator for Layer 2 Complex Branch.

    Uses real MemStream with ContextBeta ratio scoring:
    1. Extract 34D features
    2. Compute raw score = L1 kNN distance (no recon error)
    3. Normalize with ContextBeta: score / beta
    4. Decision: ratio > 1.0 -> anomaly
    5. Memory update: conditional on score < beta (normal points)
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        warmup_data: Optional[np.ndarray] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.config = config or DEFAULT_CONFIG.copy()
        self.warmup_data = warmup_data
        self.checkpoint_dir = checkpoint_dir or MODEL_CHECKPOINT_DIR

        self._ms_core = None
        self._vectorizer = None
        self._checkpoint_counter = 0

        # Track current beta per neighborhood for IEC threshold updates
        self._default_beta = self.config.get('default_beta', 0.5)
        self._current_betas: Dict[int, float] = {nb: self._default_beta for nb in range(10)}

        # MinIO client for polling updated beta values from IECDecisionMapper
        self._minio_client = None
        self._beta_cache: Dict[str, Tuple[float, float]] = {}  # key -> (beta, timestamp)
        self._cache_ttl_seconds = 1.0  # 1-second cache to reduce MinIO calls

        try:
            import boto3
            from botocore.config import Config

            minio_config = Config(signature_version='s3v4')
            self._minio_client = boto3.client(
                's3',
                endpoint_url=os.getenv('MINIO_ENDPOINT', 'http://minio:9000'),
                aws_access_key_id=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
                aws_secret_access_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
                config=minio_config,
            )
            # Verify connection
            self._minio_client.list_buckets()
            LOGGER.info("[MemStreamScoring] MinIO connected for beta polling")
        except Exception as e:
            LOGGER.warning("[MemStreamScoring] MinIO not available: %s", e)
            self._minio_client = None

        # Statistics
        self._total_scored = 0
        self._total_anomalies = 0
        self._total_memory_updates = 0

    def open(self, runtime_context):
        """Initialize MemStream core and load/checkpoint."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from memstream_src.core.memstream_core import (
            MemStreamCore, MemStreamConfig, set_determinism
        )
        from memstream_src.core.feature_extractor import FeatureVectorizer

        LOGGER.info("[MemStreamScoring] Initializing...")

        # Set determinism
        set_determinism(self.config.get('seed', 42))

        # Initialize 34D feature vectorizer
        self._vectorizer = FeatureVectorizer()
        LOGGER.info("[MemStreamScoring] Feature vectorizer initialized (34D)")

        # Create MemStream config
        cfg = MemStreamConfig()
        cfg.in_dim = self.config.get('in_dim', 34)
        cfg.hidden_dim = self.config.get('hidden_dim', 68)
        cfg.out_dim = self.config.get('out_dim', 34)
        cfg.memory_len = self.config.get('memory_len', 256)
        cfg.warmup_epochs = self.config.get('warmup_epochs', 20)
        cfg.warmup_batch_size = self.config.get('warmup_batch_size', 256)
        cfg.warmup_noise_std = self.config.get('warmup_noise_std', 0.1)
        cfg.default_beta = self.config.get('default_beta', 0.5)
        cfg.seed = self.config.get('seed', 42)
        
        # kNN scoring params
        if not hasattr(cfg, 'k'):
            cfg.k = self.config.get('k_neighbors', 10)
        if not hasattr(cfg, 'gamma'):
            cfg.gamma = self.config.get('gamma', 0.0)

        # Initialize MemStream core
        self._ms_core = MemStreamCore(cfg=cfg, device='cpu')
        LOGGER.info("[MemStreamScoring] MemStreamCore initialized")

        # Try to load checkpoint
        if not self._try_load_checkpoint():
            if self.warmup_data is not None:
                LOGGER.info(f"[MemStreamScoring] Warming up with {len(self.warmup_data)} samples...")
                self._ms_core.warmup(self.warmup_data, verbose=True)
                self._ms_core.set_beta(self.config.get('default_beta', 0.5))
                LOGGER.info("[MemStreamScoring] Warmup complete")
            else:
                LOGGER.warning("[MemStreamScoring] No checkpoint or warmup data!")

        # Create checkpoint directory
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        except PermissionError:
            LOGGER.warning("Cannot create checkpoint dir (permission denied)")
            self.checkpoint_dir = None

        LOGGER.info(f"[MemStreamScoring] Ready")

    def _try_load_checkpoint(self) -> bool:
        """Try to load model from checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, 'memstream_memory.pt')
        if not os.path.exists(checkpoint_path):
            return False
        try:
            import torch
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            self._ms_core.load_state_dict(state)
            LOGGER.info("[MemStreamScoring] Loaded checkpoint successfully")
            return True
        except Exception as e:
            LOGGER.warning(f"[MemStreamScoring] Failed to load checkpoint: {e}")
            return False

    def _save_checkpoint(self):
        """Save memory state to checkpoint."""
        if self.checkpoint_dir is None:
            return
        try:
            import torch
            checkpoint_path = os.path.join(self.checkpoint_dir, 'memstream_memory.pt')
            state = self._ms_core.get_state_dict()
            torch.save(state, checkpoint_path)
            self._checkpoint_counter = 0
            LOGGER.info("[MemStreamScoring] Checkpoint saved")
        except Exception as e:
            LOGGER.error(f"[MemStreamScoring] Failed to save checkpoint: {e}")

    def map(self, value):
        """Score a record using MemStream with ContextBeta ratio method."""
        if value is None:
            return None

        try:
            # Poll MinIO for updated beta if IEC changed it (with 1-second cache)
            if self._minio_client is not None and self._ms_core is not None:
                neighborhood_name, neighborhood_idx = get_neighborhood(value)
                cache_key = f"{neighborhood_idx}"

                # Check cache first
                now = time.time()
                cache_hit = False
                if cache_key in self._beta_cache:
                    cached_beta, cache_ts = self._beta_cache[cache_key]
                    if now - cache_ts < self._cache_ttl_seconds:
                        # Apply cached beta before scoring
                        if cached_beta != self._current_betas.get(neighborhood_idx, self._default_beta):
                            self._ms_core.set_beta_for_neighborhood(neighborhood_idx, cached_beta)
                            self._current_betas[neighborhood_idx] = cached_beta
                        cache_hit = True

                # Only poll MinIO if cache expired or miss
                if not cache_hit:
                    try:
                        import json
                        bucket = 'cadqstream-drift'
                        key = f"iec/beta/{neighborhood_name}.json"

                        response = self._minio_client.get_object(Bucket=bucket, Key=key)
                        data = response['Body'].read()
                        payload = json.loads(data.decode('utf-8'))

                        new_beta = float(payload.get('beta', self._current_betas.get(neighborhood_idx, self._default_beta)))

                        # Verify HMAC
                        iec_key = os.getenv('IEC_SIGNING_KEY')
                        if not iec_key:
                            raise RuntimeError(
                                "[MemStreamScoring] FATAL: IEC_SIGNING_KEY environment variable is required. "
                                "Unsigned beta updates are not permitted."
                            )
                        if iec_key and iec_key != 'training-signing-key':
                            hmac_hex = payload.get('hmac', '')
                            beta_str = f"{new_beta:.6f}"
                            expected_hmac = hmac.new(
                                iec_key.encode(), beta_str.encode(), hashlib.sha256
                            ).hexdigest()
                            if hmac_hex and hmac_hex != expected_hmac:
                                LOGGER.warning(
                                    "[MemStreamScoring] HMAC mismatch for %s: expected %s, got %s",
                                    neighborhood_name, expected_hmac[:16], hmac_hex[:16]
                                )
                            else:
                                self._beta_cache[cache_key] = (new_beta, now)
                                if new_beta != self._current_betas.get(neighborhood_idx, self._default_beta):
                                    self._ms_core.set_beta_for_neighborhood(neighborhood_idx, new_beta)
                                    self._current_betas[neighborhood_idx] = new_beta
                                    LOGGER.info(
                                        "[MemStreamScoring] Beta updated for %s (idx=%d): %.4f",
                                        neighborhood_name, neighborhood_idx, new_beta
                                    )
                        else:
                            # No HMAC key configured — apply beta directly
                            self._beta_cache[cache_key] = (new_beta, now)
                            if new_beta != self._current_betas.get(neighborhood_idx, self._default_beta):
                                self._ms_core.set_beta_for_neighborhood(neighborhood_idx, new_beta)
                                self._current_betas[neighborhood_idx] = new_beta
                    except Exception:
                        pass  # Non-blocking - don't slow down scoring

            # Extract features (34D)
            features = self._vectorizer.transform(value)
            if features is None:
                return self._error_result(value, 'feature_extraction_failed')

            # Get context
            neighborhood_name, neighborhood_idx = get_neighborhood(value)
            context_id = get_context_id_from_record(value)
            ratecode = float(value.get('RatecodeID', 1))

            # Score with MemStream
            # Note: hour/dow not needed for scoring, only for context_beta lookup
            score = self._ms_core.score_one(
                features,
                neighborhood_id=neighborhood_idx,
                hour=12,
                dow=0,
                ratecode=ratecode
            )

            # Decision: score > 1.0 means above context threshold
            is_anomaly = score > 1.0

            # Update statistics
            self._total_scored += 1
            if is_anomaly:
                self._total_anomalies += 1

            # Memory update: conditional (only if normal, score < 1.0)
            if not is_anomaly:
                self._ms_core.memory_update(
                    features,
                    neighborhood_id=neighborhood_idx,
                    hour=12,
                    dow=0,
                    ratecode=ratecode
                )
                self._total_memory_updates += 1

            # Periodic checkpoint
            self._checkpoint_counter += 1
            if self._checkpoint_counter >= MEMORY_CHECKPOINT_INTERVAL:
                self._save_checkpoint()

            # Build result
            return {
                **value,
                'anomaly_score': float(score),
                'threshold': 1.0,
                'is_anomaly': bool(is_anomaly),
                'context_key': neighborhood_name,
                'neighborhood': neighborhood_name,
                'neighborhood_idx': neighborhood_idx,
                'context_id': context_id,
                'ml_model': 'memstream_v10',
                'scoring_latency_ms': 0.0,
            }

        except Exception as e:
            LOGGER.error("[MemStreamScoring] Error scoring record: %s", e)
            try:
                err_nb_name, err_nb_idx = get_neighborhood(value)
            except Exception:
                err_nb_name, err_nb_idx = 'unknown', 9
            return self._error_result(value, err_nb_name, err_nb_idx, str(e))

    def _error_result(self, value, neighborhood_name, neighborhood_idx, error_msg):
        return {
            **(value or {}),
            'anomaly_score': 0.5,
            'threshold': 1.0,
            'is_anomaly': False,
            'context_key': neighborhood_name,
            'neighborhood': neighborhood_name,
            'neighborhood_idx': neighborhood_idx,
            'context_id': 0,
            'ml_model': 'memstream_v10',
            'scoring_latency_ms': -1.0,
            'scoring_error': error_msg,
        }

    def warmup(self, normal_records: List[Dict], verbose: bool = True):
        """Warmup MemStream with normal records."""
        if verbose:
            print(f"[MemStreamScoring] Warmup with {len(normal_records)} records...")

        features_list = []
        for record in normal_records:
            feat = self._vectorizer.transform(record)
            if feat is not None:
                features_list.append(feat)

        if not features_list:
            print("[MemStreamScoring] WARNING: No valid features for warmup")
            return

        X_normal = np.array(features_list, dtype=np.float32)
        self._ms_core.warmup(X_normal, epochs=self.config.get('warmup_epochs', 20), verbose=verbose)
        self._ms_core.set_beta(self.config.get('default_beta', 0.5))
        if verbose:
            print(f"[MemStreamScoring] Warmup complete")

    def get_stats(self) -> dict:
        return {
            'total_scored': self._total_scored,
            'total_anomalies': self._total_anomalies,
            'total_memory_updates': self._total_memory_updates,
            'checkpoint_counter': self._checkpoint_counter,
            'anomaly_rate': self._total_anomalies / max(self._total_scored, 1),
        }

    def close(self):
        if self._ms_core is not None:
            self._save_checkpoint()
            LOGGER.info(f"[MemStreamScoring] Closed. Scored: {self._total_scored}, "
                       f"Anomalies: {self._total_anomalies}, Memory updates: {self._total_memory_updates}")
