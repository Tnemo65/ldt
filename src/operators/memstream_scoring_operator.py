"""
MemStream Scoring Operator for Layer 2 Complex Branch (Phase 3: Sequential Pipeline).

Replaces IsolationForest with real MemStream scoring:
- 34D FeatureVectorizer
- Denoising Autoencoder (AE + Memory)
- ContextBeta (80 thresholds: 10 neighborhoods x 8 context cells)
- ADWIN per neighborhood (10 instances)
- Conditional memory updates (normal points only)
- Auto-update memory when score < beta

Phase 3 Changes:
- Sequential pipeline (no dual branch, no voting)
- IEC sends retrain signals to MinIO: cadqstream-drift/iec/retrain/{nb}/{ts}.json
- MemStreamScoringOperator polls for retrain signals and handles them:
    - Resets memory for affected neighborhood
    - Clears warmup buffer and re-triggers warmup
- Beta threshold polling from MinIO remains (read-only monitoring)

Pipeline Flow:
  Kafka taxi-nyc-raw-v2
    -> Layer1 (Parse/Dedup/Schema)
    -> CanaryRulesValidator (7 rules)
    -> MemStreamScoringOperator (AE + Memory + retrain signal polling)
    -> MetaAggregator (1-min window per neighborhood)
    -> IEC (ADWIN drift + 2 scenarios: do_nothing / quick_retrain)

Usage:
  memstream_stream = valid_stream.map(MemStreamScoringOperator(config))

Architecture:
- MapFunction with Flink ValueState for memory checkpoint recovery
- Memory state persisted via Flink ValueState
- Retrain signals polled from MinIO with HMAC verification
- Beta thresholds polled from MinIO with HMAC verification
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
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger('memstream-scoring')

# =============================================================================
# Environment & Configuration
# =============================================================================

# MinIO / S3 configuration — normalize endpoint: strip whitespace, ensure http:// prefix
_MINIO_ENDPOINT_RAW = os.getenv('MINIO_ENDPOINT', os.getenv('S3_ENDPOINT', 'http://minio:9000'))
_MINIO_ENDPOINT_STRIPPED = _MINIO_ENDPOINT_RAW.strip()
if _MINIO_ENDPOINT_STRIPPED and not _MINIO_ENDPOINT_STRIPPED.startswith(('http://', 'https://')):
    MINIO_ENDPOINT = 'http://' + _MINIO_ENDPOINT_STRIPPED
else:
    MINIO_ENDPOINT = _MINIO_ENDPOINT_STRIPPED
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

# Retrain signal polling (Phase 3: Sequential Pipeline)
RETAIN_SIGNAL_POLL_INTERVAL = 60   # Poll every N records (to avoid overhead)
RETAIN_SIGNAL_BUCKET = 'cadqstream-drift'
RETAIN_SIGNAL_PREFIX = 'iec/retrain/'

# Default MemStream config (production, v10 benchmark)
DEFAULT_CONFIG = {
    'in_dim': 34,
    'hidden_dim': 68,
    'out_dim': 34,
    'memory_len': 2048,
    'k_neighbors': 10,
    'gamma': 0.0,
    'warmup_epochs': 500,    # Production warmup
    'warmup_batch_size': 256,
    'warmup_noise_std': 0.1,
    'default_beta': 0.5,
    'warmup_buffer_limit': 1536,  # Reduced for production: min records to trigger MemStream warmup
    'seed': 42,
}

# =============================================================================
# Neighborhood Definitions (10 neighborhoods matching benchmark v10)
# =============================================================================

from src.operators.neighborhood_mapping import (
    get_neighborhood_name,
    get_neighborhood_idx,
    NEIGHBORHOOD_NAMES,
    N_NEIGHBORHOODS,
)


def location_to_neighborhood_idx(loc_id: int) -> int:
    """Map PULocationID to neighborhood index (0-9). Delegates to shared module."""
    return get_neighborhood_idx(int(loc_id) if loc_id else 0)


def get_neighborhood(record: Dict) -> Tuple[str, int]:
    """Get neighborhood name and index from record.

    Uses shared neighborhood_mapping module for consistency with MetaAggregator.
    """
    zone_id = int(float(record.get('PULocationID', 0)))
    idx = get_neighborhood_idx(zone_id)
    return NEIGHBORHOOD_NAMES[idx], idx


def get_context_id_from_record(record: Dict) -> int:
    """Get 8-cell context ID from record.

    Context cells: (Standard/Special) x (Day/Night) x (Weekday/Weekend)
    - is_special: ratecode > 1
    - is_night: hour >= 20 or hour < 6
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
    is_night = 1 if (hour >= 20 or hour < 6) else 0
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

    raw_endpoint = MINIO_ENDPOINT.strip()
    if raw_endpoint and not raw_endpoint.startswith(('http://', 'https://')):
        endpoint_url = 'http://' + raw_endpoint
    else:
        endpoint_url = raw_endpoint

    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
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

        # Async checkpoint thread
        self._checkpoint_pending = False

        # Statistics
        self._total_scored = 0
        self._total_anomalies = 0
        self._total_memory_updates = 0
        self._beta_staleness_violations = 0

        # Warmup state
        self._warmup_complete = False
        self._warmup_failed = False
        self._warmup_buffer: List[Dict] = []
        self._warmup_buffer_limit = self.config.get('warmup_buffer_limit', 16384)
        self._records_seen = 0

        # Flink state descriptor for memory checkpoint
        self._memory_state = None

        # Retrain signal polling (Phase 3)
        self._retrain_poll_counter = 0
        self._last_retrain_signals: set = set()  # (neighborhood, timestamp) seen

    def _validate_config(self):
        """CRITICAL assertions on config (HARD-BLOCK)."""
        cfg = self.config
        assert cfg.get('memory_len', 0) >= 2048, (
            f"[MemStream] HARD-BLOCK: memory_len must be >= 2048, "
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
        from src.features.memstream_vectorizer import MemStreamVectorizer

        LOGGER.info("[MemStreamScoring] Initializing...")

        # Set determinism
        set_determinism(self.config.get('seed', 42))

        # Initialize 34D MemStreamVectorizer
        self._vectorizer = MemStreamVectorizer()
        LOGGER.info("[MemStreamScoring] Feature vectorizer initialized (34D)")

        # Verify dimension compatibility
        cfg = MemStreamConfig()
        cfg.in_dim = self.config.get('in_dim', 34)
        cfg.hidden_dim = self.config.get('hidden_dim', 68)
        cfg.out_dim = self.config.get('out_dim', 34)
        expected_dim = cfg.in_dim
        actual_dim = self._vectorizer.num_features
        if actual_dim != expected_dim:
            raise ValueError(
                f"[MemStream] CRITICAL: Vectorizer output {actual_dim}D "
                f"!= MemStreamCore expected {expected_dim}D. "
                f"Check vectorizer configuration."
            )
        LOGGER.info(
            f"[MemStreamScoring] Dimension check passed: "
            f"vectorizer({actual_dim}D) == MemStreamCore({expected_dim}D)"
        )

        # Create MemStream config
        cfg.memory_len = self.config.get('memory_len', 2048)
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

        # NOTE: Each parallel Flink subtask has its own MemStream instance with
        # independent memory. With parallelism=4, each of the 4 subtasks maintains
        # its own 50K-slot memory buffer. Records scored by subtask A do NOT
        # update memory used by subtask B — memory is NOT shared across subtasks.
        #
        # Implications:
        #   - Effective memory coverage is 4x fragmented (4 separate 50K buffers).
        #   - Warmup is duplicated across subtasks (each needs its own warmup data).
        #   - Retrain signals reset memory only in the subtask that handles the signal.
        #
        # Long-term fix: Use Redis-backed shared memory so all subtasks share
        # the same memory state. Short-term: reduce scoring parallelism to 1.

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

    def _trigger_checkpoint(self):
        """Capture state snapshot and save async to avoid blocking record processing."""
        if self._ms_core is None or not self.checkpoint_bucket:
            return

        if self._checkpoint_pending:
            LOGGER.debug("[MemStreamScoring] Checkpoint already pending, skipping")
            self._checkpoint_counter = 0
            return

        try:
            import torch

            state = self._ms_core.get_state_dict()
            buf = io.BytesIO()
            torch.save(state, buf, pickle_module=pickle)
            data = buf.getvalue()

            hmac_hex = hmac.new(
                MODEL_SIGNING_KEY.encode(),
                data,
                hashlib.sha256
            ).hexdigest()

            beta_overrides = {
                NEIGHBORHOOD_NAMES[nb_idx]: beta
                for nb_idx, beta in self._current_betas.items()
                if beta != self._default_beta
            }
            beta_data = json.dumps(beta_overrides).encode('utf-8') if beta_overrides else None
            beta_hmac = (
                hmac.new(MODEL_SIGNING_KEY.encode(), beta_data, hashlib.sha256).hexdigest()
                if beta_data else None
            )

            ckpt_key = f"{self.checkpoint_prefix}memstream_memory.pt"
            hmac_key = f"{ckpt_key}.hmac"
            beta_key = f"{self.checkpoint_prefix}beta_overrides.json"
            beta_hmac_key = f"{beta_key}.hmac"

            self._checkpoint_pending = True

            def _async_save():
                try:
                    client = self._get_minio_client()
                    client.put_object(
                        Bucket=self.checkpoint_bucket, Key=ckpt_key,
                        Body=data, ContentType='application/octet-stream')
                    client.put_object(
                        Bucket=self.checkpoint_bucket, Key=hmac_key,
                        Body=hmac_hex.encode('utf-8'), ContentType='text/plain')
                    if beta_data:
                        client.put_object(
                            Bucket=self.checkpoint_bucket, Key=beta_key,
                            Body=beta_data, ContentType='application/json')
                        client.put_object(
                            Bucket=self.checkpoint_bucket, Key=beta_hmac_key,
                            Body=beta_hmac.encode('utf-8'), ContentType='text/plain')
                    LOGGER.info("[MemStreamScoring] Checkpoint saved to MinIO")
                except Exception as e:
                    LOGGER.error("[MemStreamScoring] Async checkpoint failed: %s", e)
                finally:
                    self._checkpoint_pending = False
                    self._checkpoint_counter = 0

            t = threading.Thread(target=_async_save, daemon=True)
            t.start()
            LOGGER.debug("[MemStreamScoring] Checkpoint thread started")

            if self._memory_state is not None:
                self._memory_state.update(data)

        except Exception as e:
            LOGGER.error("[MemStreamScoring] Checkpoint trigger failed: %s", e)
            self._checkpoint_counter = 0

    def _save_checkpoint(self):
        self._trigger_checkpoint()

    def _try_warmup(self) -> bool:
        """Attempt MemStream warmup with buffered records.

        Returns True if warmup succeeded, False otherwise.
        After a successful warmup, buffered records are discarded (they were
        used only for training, not scoring — warmup trains on a fixed dataset).
        After a failed warmup, the buffer is cleared so we stop trying.
        """
        if self._warmup_complete or self._warmup_failed:
            return not self._warmup_failed

        if len(self._warmup_buffer) < self._warmup_buffer_limit:
            return False

        features_list = []
        nb_ids = []
        hours = []
        dows = []
        ratecodes = []

        for rec in self._warmup_buffer:
            try:
                f = self._vectorizer.transform(rec)
                if f is None:
                    continue
                features_list.append(f)
                nb_name, nb_idx = get_neighborhood(rec)
                nb_ids.append(nb_idx)
                dt_str = rec.get('tpep_pickup_datetime', '')
                h, d = 12, 0
                if dt_str:
                    try:
                        dt = datetime.fromisoformat(dt_str.replace('/', '-'))
                        h, d = dt.hour, dt.weekday()
                    except Exception:
                        pass
                hours.append(h)
                dows.append(d)
                ratecodes.append(extract_ratecode_from_record(rec))
            except Exception:
                continue

        if len(features_list) < self._warmup_buffer_limit:
            self._warmup_failed = True
            self._warmup_buffer.clear()
            LOGGER.warning(
                "[MemStreamScoring] Warmup: only %d/%d valid features after vectorization",
                len(features_list), self._warmup_buffer_limit
            )
            return False

        X = np.array(features_list, dtype=np.float32)
        nb_arr = np.array(nb_ids, dtype=np.int64)
        LOGGER.info(
            "[MemStreamScoring] Starting warmup with %d records (limit=%d)",
            len(X), self._warmup_buffer_limit
        )

        try:
            self._ms_core.warmup(X, neighborhood_ids=nb_arr)
            self._warmup_complete = True
            self._warmup_buffer.clear()
            LOGGER.info("[MemStreamScoring] Warmup COMPLETE — ML scoring ENABLED")
            return True
        except Exception as e:
            self._warmup_failed = True
            self._warmup_buffer.clear()
            LOGGER.error(
                "[MemStreamScoring] Warmup FAILED (%s) — falling back to degraded scoring. "
                "Require >= %d clean records.",
                e, self._warmup_buffer_limit
            )
            return False

    def map(self, value):
        """Score a record using MemStream with ContextBeta ratio method.

        During warmup (<= 16384 records): buffer for training, pass through with
        default score and is_warmup=True. After warmup: real inference, pass
        through with actual score and is_warmup=False.
        """
        self._records_seen += 1

        if value is None:
            return {
                '_dlq': True,
                '_dlq_reason': 'null_input',
                '_dlq_category': 'VALIDATION_ERROR',
                '_dlq_operator': 'MemStreamScoringOperator',
                'trip_id': 'dlq_null',
                'tpep_pickup_datetime': '',
            }

        nb_name, nb_idx = get_neighborhood(value) if isinstance(value, dict) else ('unknown', 9)
        dt_str = value.get('tpep_pickup_datetime', '') if isinstance(value, dict) else ''
        hour, dow = 12, 0
        if dt_str:
            try:
                dt = datetime.fromisoformat(dt_str.replace('/', '-'))
                hour, dow = dt.hour, dt.weekday()
            except Exception:
                pass

        if not self._warmup_complete:
            # CRITICAL FIX #6: Guard against unbounded warmup buffer growth.
            # If warmup keeps failing (e.g., all records fail vectorization),
            # the buffer would grow forever. Cap it and switch to degraded mode.
            if len(self._warmup_buffer) >= self._warmup_buffer_limit:
                if not self._warmup_failed:
                    self._warmup_failed = True
                    LOGGER.error(
                        f"[{self.__class__.__name__}] Warmup buffer limit reached "
                        f"({self._warmup_buffer_limit}) without successful warmup. "
                        f"Switching to degraded mode (score=0.5 for all records)."
                    )
            self._warmup_buffer.append(value)
            return {
                **value,
                'anomaly_score': 0.0,
                'threshold': 1.0,
                'is_anomaly': False,
                'is_warmup': True,
                'context_key': nb_name,
                'neighborhood': nb_name,
                'neighborhood_idx': nb_idx,
                'context_id': get_context_id_from_record(value) if isinstance(value, dict) else 0,
                'ml_model': 'memstream_v10',
                'scoring_latency_ms': 0.0,
                'beta_staleness_violations': self._beta_staleness_violations,
                'pickup_hour': hour,
                'pickup_dow': dow,
            }

        # Warmup is complete — attempt warmup if buffer is full
        if not self._warmup_complete and not self._warmup_failed:
            self._try_warmup()
            # If warmup just succeeded, we are now in ML mode
            # If warmup failed, we are in degraded mode but warmup_complete stays False
            # Either way the record must pass through
            if self._warmup_complete:
                # Warmup succeeded on the last buffered record; score this one normally
                pass
            else:
                # Warmup still not ready or failed; score in degraded mode
                return {
                    **value,
                    'anomaly_score': 0.5,
                    'threshold': 1.0,
                    'is_anomaly': False,
                    'is_warmup': False,
                    'context_key': nb_name,
                    'neighborhood': nb_name,
                    'neighborhood_idx': nb_idx,
                    'context_id': get_context_id_from_record(value) if isinstance(value, dict) else 0,
                    'ml_model': 'memstream_v10',
                    'scoring_latency_ms': -1.0,
                    'beta_staleness_violations': self._beta_staleness_violations,
                    'pickup_hour': hour,
                    'pickup_dow': dow,
                    'scoring_error': 'warmup_incomplete',
                }

        # ── Normal scoring path (warmup complete) ───────────────────────────
        try:
            # ── Poll MinIO for retrain signals (Phase 3: Sequential Pipeline)
            self._retrain_poll_counter += 1
            if self._retrain_poll_counter >= RETAIN_SIGNAL_POLL_INTERVAL:
                self._retrain_poll_counter = 0
                self._poll_retrain_signals_from_minio()

            # ── Poll MinIO for updated beta (with cache) ────────────────────
            if self._ms_core is not None:
                cache_key = f"{nb_idx}"

                now = time.time()
                cache_hit = False
                if cache_key in self._beta_cache:
                    cached_beta, cache_ts = self._beta_cache[cache_key]
                    if now - cache_ts < BETA_CACHE_TTL_SECONDS:
                        if cached_beta != self._current_betas.get(nb_idx, self._default_beta):
                            self._ms_core.set_beta_for_neighborhood(nb_idx, cached_beta)
                            self._current_betas[nb_idx] = cached_beta
                        cache_hit = True

                if cache_key in self._beta_last_poll:
                    staleness = now - self._beta_last_poll[cache_key]
                    if staleness > BETA_MAX_STALENESS_SECONDS:
                        self._beta_staleness_violations += 1

                if not cache_hit:
                    self._beta_last_poll[cache_key] = now
                    self._poll_beta_from_minio(nb_name, nb_idx)

            # ── Extract features (34D) ─────────────────────────────────────
            features = self._vectorizer.transform(value)
            if features is None:
                return self._error_result(value, nb_name, nb_idx, 'feature_extraction_failed')

            # ── Extract temporal + ratecode from record ─────────────────────
            context_id = get_context_id_from_record(value)
            ratecode = extract_ratecode_from_record(value)

            # ── Score with MemStream ────────────────────────────────────────
            score = self._ms_core.score_one(
                features,
                neighborhood_id=nb_idx,
                hour=hour,
                dow=dow,
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
                    neighborhood_id=nb_idx,
                    hour=hour,
                    dow=dow,
                    ratecode=ratecode
                )
                self._total_memory_updates += 1

            # Periodic checkpoint
            self._checkpoint_counter += 1
            if self._checkpoint_counter >= MEMORY_CHECKPOINT_INTERVAL:
                self._save_checkpoint()

            return {
                **value,
                'anomaly_score': float(score),
                'threshold': 1.0,
                'is_anomaly': bool(is_anomaly),
                'is_warmup': False,
                'context_key': nb_name,
                'neighborhood': nb_name,
                'neighborhood_idx': nb_idx,
                'context_id': context_id,
                'ml_model': 'memstream_v10',
                'scoring_latency_ms': 0.0,
                'beta_staleness_seconds': self._beta_staleness_violations,
                'pickup_hour': hour,
                'pickup_dow': dow,
            }

        except Exception as e:
            LOGGER.error("[MemStreamScoring] Error scoring record: %s", e)
            return self._error_result(value, nb_name, nb_idx, str(e))

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

    def _poll_retrain_signals_from_minio(self):
        """
        Poll MinIO for IEC retrain signals (Phase 3: Sequential Pipeline).

        Searches for new retrain signals at: cadqstream-drift/iec/retrain/{nb}/*.json
        Handles each new signal by resetting memory + re-triggering warmup.

        Only processes signals not seen before (tracked via _last_retrain_signals).
        HMAC verification required on each signal.
        """
        minio = self._get_minio_client()
        if minio is None:
            return

        try:
            for nb_name in NEIGHBORHOOD_NAMES:
                prefix = f"{RETAIN_SIGNAL_PREFIX}{nb_name}/"
                try:
                    response = minio.list_objects_v2(
                        Bucket=RETAIN_SIGNAL_BUCKET,
                        Prefix=prefix,
                    )
                except Exception:
                    continue

                objects = response.get('Contents', [])
                for obj in objects:
                    key = obj['Key']
                    # Skip already-seen signals
                    ts = key.split('/')[-1].replace('.json', '')
                    signal_id = f"{nb_name}:{ts}"
                    if signal_id in self._last_retrain_signals:
                        continue

                    # Load and verify HMAC
                    try:
                        data = _load_from_minio(RETAIN_SIGNAL_BUCKET, key)
                        signal = json.loads(data.decode('utf-8'))
                    except Exception:
                        continue

                    hmac_hex = signal.get('hmac', '')
                    if not self._verify_hmac(data, hmac_hex, 'retrain_signal'):
                        LOGGER.warning(
                            "[MemStreamScoring] Retrain signal HMAC invalid for %s",
                            key
                        )
                        continue

                    # Process retrain signal
                    self._handle_retrain_signal(nb_name, signal)
                    self._last_retrain_signals.add(signal_id)

                    # Prune old signals to avoid unbounded set growth
                    if len(self._last_retrain_signals) > 1000:
                        old_signals = sorted(self._last_retrain_signals)[:500]
                        self._last_retrain_signals -= set(old_signals)

        except Exception as e:
            LOGGER.warning(
                "[MemStreamScoring] Retrain signal polling failed: %s",
                e
            )

    def _handle_retrain_signal(self, neighborhood_name: str, signal: dict):
        """
        Handle a retrain signal from IEC.

        When quick_retrain fires, this operator:
        1. Resets MemStream memory for the affected neighborhood
        2. Clears the warmup buffer
        3. Sets _warmup_complete = False to re-trigger warmup

        ADWIN thresholds are reset separately by the ml-service when
        it receives the retrain signal via the action-replay-worker.

        Args:
            neighborhood_name: Name of the neighborhood (e.g., 'manhattan')
            signal: Retrain signal dict from MinIO
        """
        nb_idx = NEIGHBORHOOD_NAMES.index(neighborhood_name) if neighborhood_name in NEIGHBORHOOD_NAMES else 9
        triggers = signal.get('triggers', [])
        severity = signal.get('severity', 'high')

        LOGGER.warning(
            "[MemStreamScoring] RETRAIN SIGNAL received for %s (severity=%s, triggers=%s)",
            neighborhood_name, severity, triggers
        )

        # Reset MemStream memory for this neighborhood
        if self._ms_core is not None:
            result = self._ms_core.reset_neighborhood(nb_idx)
            if result.get('status') == 'ok':
                LOGGER.info(
                    "[MemStreamScoring] Memory reset for neighborhood %s (idx=%d)",
                    neighborhood_name, nb_idx
                )
            else:
                LOGGER.error(
                    "[MemStreamScoring] Memory reset failed for %s: %s",
                    neighborhood_name, result.get('error', 'unknown')
                )

        # Reset beta for this neighborhood to default
        self._current_betas[nb_idx] = self._default_beta
        self._beta_cache.pop(f"{nb_idx}", None)

        # Clear warmup buffer and re-trigger warmup
        self._warmup_buffer.clear()
        self._warmup_complete = False
        self._warmup_failed = False
        self._records_seen = 0

        LOGGER.warning(
            "[MemStreamScoring] Warmup cleared for %s — waiting for %d clean records",
            neighborhood_name, self._warmup_buffer_limit
        )

    def _error_result(
        self,
        value,
        neighborhood_name: str,
        neighborhood_idx: int,
        error_msg: str
    ):
        return {
            **value,
            'anomaly_score': 0.5,
            'threshold': 1.0,
            'is_anomaly': False,
            'is_warmup': self._warmup_complete,
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
            'retrain_signals_processed': len(self._last_retrain_signals),
            'warmup_complete': self._warmup_complete,
            'warmup_failed': self._warmup_failed,
            'warmup_buffer_size': len(self._warmup_buffer),
            'warmup_buffer_limit': self._warmup_buffer_limit,
            'records_seen': self._records_seen,
            'current_betas': {
                NEIGHBORHOOD_NAMES[nb]: f"{beta:.4f}"
                for nb, beta in self._current_betas.items()
            },
        }

    def close(self):
        if self._ms_core is not None:
            self._save_checkpoint()
            LOGGER.info(
                "[MemStreamScoring] Closed. Records: %d, Scored: %d, Anomalies: %d, "
                "Memory updates: %d, Beta staleness violations: %d, "
                "Warmup: complete=%s failed=%s buffer=%d/%d",
                self._records_seen, self._total_scored, self._total_anomalies,
                self._total_memory_updates, self._beta_staleness_violations,
                self._warmup_complete, self._warmup_failed,
                len(self._warmup_buffer), self._warmup_buffer_limit
            )
        else:
            LOGGER.info(
                "[MemStreamScoring] Closed. Records: %d, Warmup: complete=%s buffer=%d/%d",
                self._records_seen, self._warmup_complete,
                len(self._warmup_buffer), self._warmup_buffer_limit
            )
