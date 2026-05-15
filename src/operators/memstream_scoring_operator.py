"""
MemStream Scoring Operator for Layer 2 Complex Branch (Phase 2C).

Replaces IsolationForest with real MemStream scoring:
- 34D FeatureVectorizer
- Denoising Autoencoder (AE + Memory)
- ContextBeta (80 thresholds: 10 neighborhoods x 8 context cells)
- ADWIN per neighborhood (10 instances)
- Conditional memory updates (normal points only)

Pipeline Flow:
  taxi-nyc-raw -> Layer1 validation -> dq-stream-processed
  dq-stream-processed -> CanaryRulesValidator -> dq-stream-processed-clean
  dq-stream-processed-clean -> MemStreamScoringOperator -> anomaly scores

Usage:
  memstream_stream = clean_stream.map(MemStreamScoringOperator(config))

Architecture:
- MapFunction (NOT BroadcastProcessFunction) with runtime_context.get_state()
- Memory state persisted via Flink ValueState for checkpoint recovery
- Beta thresholds polled from MinIO (IECDecisionMapper writes them)
- HMAC verification on all model/beta loading

Kafka Topics (from 01-create-topics.sh):
  Input:         taxi-nyc-raw
  Processed:     dq-stream-processed
  Anomalies:     dq-stream-anomalies
  Canary clean:  dq-stream-processed-clean
  Violations:    dq-hard-rule-violations
  IEC:           dq-meta-stream
  IF model:      if-model-updates (1 partition → increase to 4)
  MemStream:     memstream-model-updates:4:1  (NEW)

Flink: env.set_parallelism(1) for initial migration.
"""

from pyflink.datastream import MapFunction
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.common.typeinfo import Types
import hashlib
import hmac
import io
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger('memstream-scoring')

# =============================================================================
# Environment & Configuration
# =============================================================================

# MinIO / S3 configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', os.getenv('S3_ENDPOINT', 'http://minio:9000'))
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin'))
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin'))
MINIO_BUCKET = os.getenv('MEMSTREAM_MINIO_BUCKET', 'cadqstream-drift')

# Model signing key (REQUIRED - no bypass)
MODEL_SIGNING_KEY = os.getenv('MEMSTREAM_MODEL_SIGNING_KEY')
if not MODEL_SIGNING_KEY:
    raise RuntimeError(
        "[MemStream] FATAL: MEMSTREAM_MODEL_SIGNING_KEY environment variable is required. "
        "Model files will not be loaded without HMAC verification. "
        "Set to a 256-bit secret (32+ characters)."
    )
if len(MODEL_SIGNING_KEY) < 32:
    raise RuntimeError(
        f"[MemStream] FATAL: MEMSTREAM_MODEL_SIGNING_KEY must be at least 32 characters "
        f"(256-bit). Got {len(MODEL_SIGNING_KEY)} characters."
    )

# IEC signing key for beta updates (REQUIRED)
IEC_SIGNING_KEY = os.getenv('IEC_SIGNING_KEY')
if not IEC_SIGNING_KEY:
    raise RuntimeError(
        "[MemStream] FATAL: IEC_SIGNING_KEY environment variable is required. "
        "Beta updates will not be accepted without HMAC signing."
    )
if len(IEC_SIGNING_KEY) < 32:
    raise RuntimeError(
        f"[MemStream] FATAL: IEC_SIGNING_KEY must be at least 32 characters. "
        f"Got {len(IEC_SIGNING_KEY)} characters."
    )

# Checkpoint / memory persistence
MEMORY_CHECKPOINT_INTERVAL = int(os.getenv('MEMSTREAM_CHECKPOINT_INTERVAL', '1000'))
CHECKPOINT_BUCKET = os.getenv('MEMSTREAM_CHECKPOINT_BUCKET', 'cadqstream-drift')
CHECKPOINT_PREFIX = os.getenv('MEMSTREAM_CHECKPOINT_PREFIX', 'checkpoints/memstream/')

# Beta polling
BETA_CACHE_TTL_SECONDS = 1.0  # 1.1s max staleness acceptable
BETA_MAX_STALENESS_SECONDS = 1.1  # Alert threshold for metrics

# Default MemStream config (production, v10 benchmark)
DEFAULT_CONFIG = {
    'in_dim': 34,
    'hidden_dim': 68,
    'out_dim': 34,
    'memory_len': 100000,    # Production scale
    'k_neighbors': 10,
    'gamma': 0.0,
    'warmup_epochs': 500,    # Production warmup
    'warmup_batch_size': 256,
    'warmup_noise_std': 0.1,
    'default_beta': 0.5,
    'seed': 42,
}

# =============================================================================
# Neighborhood Definitions (10 neighborhoods matching benchmark v10)
# =============================================================================

MANHATTAN_ZONES = set(range(1, 44))
BRONX_ZONES = set(range(44, 104))
BROOKLYN_ZONES = set(range(104, 128))
QUEENS_LOWER_ZONES = set(range(128, 149))
QUEENS_UPPER_ZONES = set(range(149, 162))
STATEN_ISLAND_ZONES = set(range(162, 182))
EWR_ZONES = set(range(182, 197))
JFK_ZONES = set(range(217, 230))
NALP_ZONES = set(range(230, 235))
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
        return 0
    elif 44 <= z <= 103:
        return 4
    elif 104 <= z <= 127:
        return 1
    elif 128 <= z <= 148:
        return 2
    elif 149 <= z <= 161:
        return 3
    elif 162 <= z <= 181:
        return 5
    elif 182 <= z <= 196:
        return 6
    elif 217 <= z <= 229:
        return 7
    elif 230 <= z <= 234:
        return 8
    else:
        return 9


def get_neighborhood(record: Dict) -> Tuple[str, int]:
    """Get neighborhood name and index from record."""
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
    dt_str = record.get('tpep_pickup_datetime', '')
    hour = 12
    dow = 0
    if dt_str:
        try:
            dt = datetime.fromisoformat(dt_str.replace('/', '-'))
            hour = dt.hour
            dow = dt.weekday()
        except Exception:
            pass

    ratecode = float(record.get('RatecodeID', 1))
    is_special = 1 if ratecode > 1 else 0
    is_night = 1 if (hour >= 18 or hour < 6) else 0
    is_weekend = 1 if dow >= 5 else 0
    return (is_special << 2) | (is_night << 1) | is_weekend


def extract_ratecode_from_record(record: Dict) -> float:
    """Extract ratecode from record, preferring one-hot over raw field.

    The ratecodeID one-hot encoding (indices 25-29) takes priority over
    the raw RatecodeID field, resolving JFK flat fare ambiguity.
    """
    # Check one-hot encoding first (indices 25-29 in feature vector)
    ratecode_fields = ['ratecode_1', 'ratecode_2', 'ratecode_3', 'ratecode_4', 'ratecode_5']
    for i, field in enumerate(ratecode_fields, start=1):
        val = record.get(field)
        if val is not None:
            try:
                if float(val) == 1.0:
                    return float(i)
            except (ValueError, TypeError):
                pass

    # Fall back to raw RatecodeID
    try:
        return float(record.get('RatecodeID', 1))
    except (ValueError, TypeError):
        return 1.0


# =============================================================================
# MinIO Client (inline, avoids extra import dependency)
# =============================================================================

def _get_minio_client():
    """Create a boto3 S3 client configured for MinIO."""
    import boto3
    from botocore.config import Config

    cfg = Config(
        signature_version='s3v4',
        retries={'max_attempts': 3, 'mode': 'standard'},
        connect_timeout=5.0,
        read_timeout=30.0,
        s3={'addressing_style': 'path'},
    )
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=cfg,
    )


def _load_from_minio(bucket: str, key: str) -> bytes:
    """Download an object from MinIO."""
    client = _get_minio_client()
    response = client.get_object(Bucket=bucket, Key=key)
    return response['Body'].read()


# =============================================================================
# MemStreamScoringOperator
# =============================================================================

class MemStreamScoringOperator(MapFunction):
    """Full MemStream Scoring Operator for Layer 2 Complex Branch.

    Uses real MemStream with ContextBeta ratio scoring:
    1. Extract 34D features from record
    2. Extract ACTUAL hour/dow from record (NOT hardcoded)
    3. Compute raw score = L1 kNN distance
    4. Normalize with ContextBeta: score / beta
    5. Decision: ratio > 1.0 -> anomaly
    6. Memory update: conditional on score < beta (normal points)

    Architecture:
    - MapFunction (NOT BroadcastProcessFunction)
    - Memory state via Flink ValueState for checkpoint recovery
    - Beta thresholds from MinIO with HMAC verification
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        checkpoint_bucket: str = CHECKPOINT_BUCKET,
        checkpoint_prefix: str = CHECKPOINT_PREFIX,
    ):
        self.config = config or DEFAULT_CONFIG.copy()
        self.checkpoint_bucket = checkpoint_bucket
        self.checkpoint_prefix = checkpoint_prefix

        # Import path setup
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        # Validate config BEFORE any initialization
        self._validate_config()

        self._ms_core = None
        self._vectorizer = None
        self._checkpoint_counter = 0
        self._default_beta = self.config.get('default_beta', 0.5)
        self._current_betas: Dict[int, float] = {
            nb: self._default_beta for nb in range(N_NEIGHBORHOODS)
        }
        self._beta_cache: Dict[str, Tuple[float, float]] = {}
        self._beta_last_poll: Dict[str, float] = {}

        # MinIO client (lazy)
        self._minio_client = None

        # Statistics
        self._total_scored = 0
        self._total_anomalies = 0
        self._total_memory_updates = 0
        self._beta_staleness_violations = 0

        # Flink state descriptor for memory checkpoint
        self._memory_state = None

    def _validate_config(self):
        """CRITICAL assertions on config (HARD-BLOCK)."""
        cfg = self.config
        assert cfg.get('memory_len', 0) >= 50000, (
            f"[MemStream] HARD-BLOCK: memory_len must be >= 50000, "
            f"got {cfg.get('memory_len')}"
        )
        assert cfg.get('warmup_epochs', 0) >= 500, (
            f"[MemStream] HARD-BLOCK: warmup_epochs must be >= 500, "
            f"got {cfg.get('warmup_epochs')}"
        )

    def _get_minio_client(self):
        """Lazily initialize MinIO client."""
        if self._minio_client is None:
            self._minio_client = _get_minio_client()
        return self._minio_client

    def open(self, runtime_context):
        """Initialize MemStream core, load checkpoint, setup Flink state."""
        from src.ml.memstream_core import MemStreamCore, MemStreamConfig, set_determinism
        from src.features.vectorizer import FeatureVectorizer

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
        cfg.memory_len = self.config.get('memory_len', 100000)
        cfg.warmup_epochs = self.config.get('warmup_epochs', 500)
        cfg.warmup_batch_size = self.config.get('warmup_batch_size', 256)
        cfg.warmup_noise_std = self.config.get('warmup_noise_std', 0.1)
        cfg.default_beta = self.config.get('default_beta', 0.5)
        cfg.seed = self.config.get('seed', 42)
        cfg.k = self.config.get('k_neighbors', 10)
        cfg.gamma = self.config.get('gamma', 0.0)

        # Initialize MemStream core
        self._ms_core = MemStreamCore(cfg=cfg, device='cpu')
        LOGGER.info("[MemStreamScoring] MemStreamCore initialized")

        # Setup Flink ValueState for memory persistence
        state_desc = ValueStateDescriptor(
            "memstream_memory",
            Types.PICKLED_BYTE_ARRAY()
        )
        self._memory_state = runtime_context.get_state(state_desc)
        LOGGER.info("[MemStreamScoring] Flink ValueState initialized")

        # Try to load checkpoint from MinIO
        if not self._try_load_checkpoint():
            LOGGER.warning(
                "[MemStreamScoring] No checkpoint in MinIO. "
                "Will require warmup data or accept degraded performance."
            )

        LOGGER.info("[MemStreamScoring] Ready")

    def _verify_hmac(self, data: bytes, hmac_hex: str, key: str) -> bool:
        """Verify HMAC of data using the signing key.

        Args:
            data: Raw bytes to verify
            hmac_hex: Expected HMAC hex string
            key: Key name for logging ('model' or 'beta')

        Returns:
            True if HMAC matches, False otherwise.
        """
        if not hmac_hex:
            LOGGER.error("[MemStream] HMAC missing for %s — rejecting", key)
            return False

        expected = hmac.new(
            MODEL_SIGNING_KEY.encode(),
            data,
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(hmac_hex, expected):
            LOGGER.error(
                "[MemStream] %s HMAC mismatch — expected %.8s..., got %.8s...",
                key, expected, hmac_hex
            )
            return False
        return True

    def _try_load_checkpoint(self) -> bool:
        """Try to load model + memory from MinIO with HMAC verification."""
        import torch

        # Try latest checkpoint
        keys_to_try = [
            f"{self.checkpoint_prefix}latest/memstream_memory.pt",
            f"{self.checkpoint_prefix}memstream_memory.pt",
        ]
        hmac_keys = [
            f"{self.checkpoint_prefix}latest/memstream_memory.pt.hmac",
            f"{self.checkpoint_prefix}memstream_memory.pt.hmac",
        ]

        for ckpt_key, hmac_key in zip(keys_to_try, hmac_keys):
            try:
                # Load checkpoint data
                data = _load_from_minio(self.checkpoint_bucket, ckpt_key)

                # Load HMAC
                hmac_hex = ''
                try:
                    hmac_data = _load_from_minio(self.checkpoint_bucket, hmac_key)
                    hmac_hex = hmac_data.decode('utf-8').strip()
                except Exception:
                    LOGGER.warning("[MemStream] HMAC file not found: %s", hmac_key)

                # Verify HMAC
                if not self._verify_hmac(data, hmac_hex, 'checkpoint'):
                    LOGGER.error("[MemStream] Checkpoint HMAC verification failed: %s", ckpt_key)
                    continue

                # Deserialize
                buf = io.BytesIO(data)
                state = torch.load(buf, map_location='cpu', weights_only=False)
                self._ms_core.load_state_dict(state)

                # Also load beta overrides if present
                beta_key = f"{self.checkpoint_prefix}beta_overrides.json"
                try:
                    beta_data = _load_from_minio(self.checkpoint_bucket, beta_key)
                    beta_hmac_hex = ''
                    try:
                        beta_hmac_data = _load_from_minio(
                            self.checkpoint_bucket, beta_key + '.hmac'
                        )
                        beta_hmac_hex = beta_hmac_data.decode('utf-8').strip()
                    except Exception:
                        pass

                    if self._verify_hmac(beta_data, beta_hmac_hex, 'beta_overrides'):
                        beta_overrides = json.loads(beta_data.decode('utf-8'))
                        for nb_name, beta_val in beta_overrides.items():
                            nb_idx = NEIGHBORHOOD_NAMES.index(nb_name) if nb_name in NEIGHBORHOOD_NAMES else -1
                            if nb_idx >= 0:
                                self._current_betas[nb_idx] = beta_val
                                if self._ms_core is not None:
                                    self._ms_core.set_beta_for_neighborhood(nb_idx, beta_val)
                        LOGGER.info(
                            "[MemStreamScoring] Loaded %d beta overrides from MinIO",
                            len(beta_overrides)
                        )
                except Exception:
                    pass

                LOGGER.info("[MemStreamScoring] Loaded checkpoint from MinIO: %s", ckpt_key)
                return True

            except Exception as e:
                LOGGER.warning(
                    "[MemStreamScoring] Failed to load checkpoint %s: %s",
                    ckpt_key, e
                )
                continue

        return False

    def _save_checkpoint(self):
        """Save memory state to MinIO with HMAC verification."""
        if self._ms_core is None:
            return
        if not self.checkpoint_bucket:
            return

        try:
            import torch

            # Serialize state
            state = self._ms_core.get_state_dict()
            buf = io.BytesIO()
            torch.save(state, buf, pickle_module=pickle)
            data = buf.getvalue()

            # Compute HMAC
            hmac_hex = hmac.new(
                MODEL_SIGNING_KEY.encode(),
                data,
                hashlib.sha256
            ).hexdigest()

            # Upload state + HMAC
            key = f"{self.checkpoint_prefix}memstream_memory.pt"
            hmac_key = f"{key}.hmac"

            client = self._get_minio_client()
            client.put_object(
                Bucket=self.checkpoint_bucket,
                Key=key,
                Body=data,
                ContentType='application/octet-stream',
            )
            client.put_object(
                Bucket=self.checkpoint_bucket,
                Key=hmac_key,
                Body=hmac_hex.encode('utf-8'),
                ContentType='text/plain',
            )

            # Save beta overrides
            beta_overrides = {
                NEIGHBORHOOD_NAMES[nb_idx]: beta
                for nb_idx, beta in self._current_betas.items()
                if beta != self._default_beta
            }
            if beta_overrides:
                beta_data = json.dumps(beta_overrides).encode('utf-8')
                beta_hmac = hmac.new(
                    MODEL_SIGNING_KEY.encode(),
                    beta_data,
                    hashlib.sha256
                ).hexdigest()
                beta_key = f"{self.checkpoint_prefix}beta_overrides.json"
                client.put_object(
                    Bucket=self.checkpoint_bucket,
                    Key=beta_key,
                    Body=beta_data,
                    ContentType='application/json',
                )
                client.put_object(
                    Bucket=self.checkpoint_bucket,
                    Key=f"{beta_key}.hmac",
                    Body=beta_hmac.encode('utf-8'),
                    ContentType='text/plain',
                )

            # Update Flink ValueState
            if self._memory_state is not None:
                self._memory_state.update(data)

            LOGGER.info("[MemStreamScoring] Checkpoint saved to MinIO")
            self._checkpoint_counter = 0

        except Exception as e:
            LOGGER.error("[MemStreamScoring] Failed to save checkpoint: %s", e)

    def map(self, value):
        """Score a record using MemStream with ContextBeta ratio method.

        CRITICAL: Extracts ACTUAL hour/dow from record — NOT hardcoded.
        """
        if value is None:
            return None

        try:
            # ── Extract actual hour/dow from record (NOT hardcoded) ──────────
            dt_str = value.get('tpep_pickup_datetime', '')
            hour = 12
            dow = 0
            if dt_str:
                try:
                    dt = datetime.fromisoformat(dt_str.replace('/', '-'))
                    hour = dt.hour
                    dow = dt.weekday()
                except Exception:
                    pass

            # ── Poll MinIO for updated beta (with cache) ────────────────────
            if self._ms_core is not None:
                neighborhood_name, neighborhood_idx = get_neighborhood(value)
                cache_key = f"{neighborhood_idx}"

                now = time.time()
                cache_hit = False
                if cache_key in self._beta_cache:
                    cached_beta, cache_ts = self._beta_cache[cache_key]
                    if now - cache_ts < BETA_CACHE_TTL_SECONDS:
                        if cached_beta != self._current_betas.get(
                            neighborhood_idx, self._default_beta
                        ):
                            self._ms_core.set_beta_for_neighborhood(
                                neighborhood_idx, cached_beta
                            )
                            self._current_betas[neighborhood_idx] = cached_beta
                        cache_hit = True

                # Track staleness
                if cache_key in self._beta_last_poll:
                    staleness = now - self._beta_last_poll[cache_key]
                    if staleness > BETA_MAX_STALENESS_SECONDS:
                        self._beta_staleness_violations += 1

                if not cache_hit:
                    self._beta_last_poll[cache_key] = now
                    self._poll_beta_from_minio(neighborhood_name, neighborhood_idx)

            # ── Extract features (34D) ──────────────────────────────────────
            features = self._vectorizer.transform(value)
            if features is None:
                return self._error_result(value, 'feature_extraction_failed')

            # ── Extract temporal + ratecode from record ─────────────────────
            neighborhood_name, neighborhood_idx = get_neighborhood(value)
            context_id = get_context_id_from_record(value)
            ratecode = extract_ratecode_from_record(value)

            # ── Score with MemStream (using ACTUAL hour/dow) ───────────────
            score = self._ms_core.score_one(
                features,
                neighborhood_id=neighborhood_idx,
                hour=hour,      # ACTUAL extracted hour
                dow=dow,        # ACTUAL extracted dow
                ratecode=ratecode
            )

            # Decision: ratio > 1.0 means above context threshold
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
                    hour=hour,    # ACTUAL extracted hour
                    dow=dow,      # ACTUAL extracted dow
                    ratecode=ratecode
                )
                self._total_memory_updates += 1

            # Periodic checkpoint
            self._checkpoint_counter += 1
            if self._checkpoint_counter >= MEMORY_CHECKPOINT_INTERVAL:
                self._save_checkpoint()

            # ── Build result ─────────────────────────────────────────────────
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
                'beta_staleness_seconds': self._beta_staleness_violations,
                'pickup_hour': hour,     # Include actual hour for downstream
                'pickup_dow': dow,       # Include actual dow for downstream
            }

        except Exception as e:
            LOGGER.error("[MemStreamScoring] Error scoring record: %s", e)
            try:
                err_nb_name, err_nb_idx = get_neighborhood(value)
            except Exception:
                err_nb_name, err_nb_idx = 'unknown', 9
            return self._error_result(value, err_nb_name, err_nb_idx, str(e))

    def _poll_beta_from_minio(
        self,
        neighborhood_name: str,
        neighborhood_idx: int
    ):
        """Poll MinIO for updated beta value with HMAC verification."""
        try:
            bucket = 'cadqstream-drift'
            key = f"iec/beta/{neighborhood_name}.json"

            data = _load_from_minio(bucket, key)
            payload = json.loads(data.decode('utf-8'))
            new_beta = float(payload.get('beta', self._default_beta))

            # Verify HMAC
            hmac_hex = payload.get('hmac', '')
            beta_str = f"{new_beta:.6f}"
            expected_hmac = hmac.new(
                IEC_SIGNING_KEY.encode(),
                beta_str.encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac_hex:
                LOGGER.warning(
                    "[MemStreamScoring] Beta update for %s has no HMAC — rejecting",
                    neighborhood_name
                )
                return

            if not hmac.compare_digest(hmac_hex, expected_hmac):
                LOGGER.warning(
                    "[MemStreamScoring] Beta HMAC mismatch for %s — rejecting",
                    neighborhood_name
                )
                return

            # Apply beta
            now = time.time()
            self._beta_cache[f"{neighborhood_idx}"] = (new_beta, now)
            if new_beta != self._current_betas.get(neighborhood_idx, self._default_beta):
                self._ms_core.set_beta_for_neighborhood(neighborhood_idx, new_beta)
                self._current_betas[neighborhood_idx] = new_beta
                LOGGER.info(
                    "[MemStreamScoring] Beta updated for %s (idx=%d): %.4f",
                    neighborhood_name, neighborhood_idx, new_beta
                )

        except Exception as e:
            # Non-blocking — don't slow down scoring
            pass

    def _error_result(
        self,
        value,
        neighborhood_name: str,
        neighborhood_idx: int,
        error_msg: str
    ):
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

    def get_stats(self) -> dict:
        return {
            'total_scored': self._total_scored,
            'total_anomalies': self._total_anomalies,
            'total_memory_updates': self._total_memory_updates,
            'checkpoint_counter': self._checkpoint_counter,
            'anomaly_rate': self._total_anomalies / max(self._total_scored, 1),
            'beta_staleness_violations': self._beta_staleness_violations,
            'current_betas': {
                NEIGHBORHOOD_NAMES[nb]: f"{beta:.4f}"
                for nb, beta in self._current_betas.items()
            },
        }

    def close(self):
        if self._ms_core is not None:
            self._save_checkpoint()
            LOGGER.info(
                "[MemStreamScoring] Closed. Scored: %s, Anomalies: %s, "
                "Memory updates: %s, Beta staleness violations: %s",
                self._total_scored, self._total_anomalies,
                self._total_memory_updates, self._beta_staleness_violations
            )
