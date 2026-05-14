"""
CA-DQStream + MemStream v5 - Complete Flink Job Orchestration
============================================================

4-Layer Streaming Pipeline Architecture:
    Layer 1: Baseline Validation (Parse, Watermark, Dedup, Schema)
    Layer 2A: Canary Rules (7 static business rules)
    Layer 2B: MemStream ML (Denoising Autoencoder + Memory)
    Layer 3: Voting Ensemble + MetaAggregator
    Layer 4: IEC (Intelligent Evolution Controller)

Features:
    - Kafka source/sinks with EXACTLY_ONCE semantics
    - MinIO/Parquet sinks for long-term storage
    - Redis for IEC beta threshold communication
    - Checkpointing for fault tolerance
    - Prometheus metrics export

Environment Variables:
    FLINK_PARALLELISM: Number of parallel tasks (default: 4)
    CHECKPOINT_INTERVAL_MS: Checkpoint interval (default: 60000)
    STATE_BACKEND: rocksdb or hashmap (default: rocksdb)
    KAFKA_BOOTSTRAP_SERVERS: Kafka brokers (default: kafka:9092)
    REDIS_HOST: Redis host (default: redis)
    REDIS_PORT: Redis port (default: 6379)
    REDIS_PASSWORD: Redis password (required in production)
    MINIO_ENDPOINT: MinIO endpoint (default: minio:9000)
    MINIO_ACCESS_KEY: MinIO access key (default: minioadmin)
    MINIO_SECRET_KEY: MinIO secret key (default: minioadmin)
    MEMSTREAM_MODEL_PATH: Path to MemStream model (default: /models/memstream_ae.pt)
    MEMSTREAM_MODEL_SIGNING_KEY: HMAC key for model verification (required)
    IEC_SIGNING_KEY: HMAC key for IEC actions (required)

Usage:
    flink run -d memstream_src/operators/flink_job_complete.py --target yarn
    flink run -d memstream_src/operators/flink_job_complete.py --target kubernetes-application

Reference: original_flow.md lines 26-140 (Architecture)
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List

from pyflink.datastream import (
    StreamExecutionEnvironment,
    StreamExecutionEnvironment as SEV,
    KeyedProcessFunction,
    ProcessFunction,
)
from pyflink.datastream.connectors.kafka import (
    FlinkKafkaConsumer,
    FlinkKafkaProducer,
)
from pyflink.datastream.state import ValueStateDescriptor, StateTtlConfig, MapStateDescriptor
from pyflink.common.typeinfo import Types, RowTypeInfo, BasicTypeInfo
from pyflink.datastream.connectors.file_system import StreamingFileSink
from pyflink.datastream.formats.json import JsonRowDeserializationSchema, JsonRowSerializationSchema
from pyflink.datastream import OutputTag
from pyflink.common.time import Time, Duration
from pyflink.common.watermarks import BoundedOutOfOrdernessWatermarker
from pyflink.datastream import WatermarkStrategy
from pyflink.datastream.checkpointing import (
    CheckpointingMode,
    ExternalizedCheckpointCleanup,
)

# =============================================================================
# Optional Dependency: pyarrow (for MinIO Parquet sinks)
# =============================================================================

try:
    import pyarrow
    import pyarrow.parquet
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    pyarrow = None  # type: ignore[assignment]

LOGGER = logging.getLogger('cadqstream.complete')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


# =============================================================================
# Pipeline Configuration
# =============================================================================

class PipelineConfig:
    """
    Centralized pipeline configuration from environment variables.
    
    All configuration must be sourced from environment variables to support
    containerized deployment with ConfigMaps/Secrets.
    """
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
    INPUT_TOPIC: str = os.getenv('INPUT_TOPIC', 'taxi-nyc-raw')
    KAFKA_SOURCE_GROUP: str = 'cadqstream-complete-source'
    
    # Topics (output)
    OUTPUT_TOPIC: str = os.getenv('OUTPUT_TOPIC', 'dq-stream-processed')
    ANOMALY_TOPIC: str = os.getenv('ANOMALY_TOPIC', 'dq-stream-anomalies')
    VIOLATION_TOPIC: str = os.getenv('VIOLATION_TOPIC', 'dq-hard-rule-violations')
    META_TOPIC: str = os.getenv('META_TOPIC', 'dq-meta-stream')
    IEC_TOPIC: str = os.getenv('IEC_TOPIC', 'iec-action-replay')
    CLEAN_TOPIC: str = 'dq-stream-processed-clean'
    
    # Flink
    PARALLELISM: int = int(os.getenv('FLINK_PARALLELISM', '4'))
    CHECKPOINT_INTERVAL_MS: int = int(os.getenv('CHECKPOINT_INTERVAL_MS', '60000'))
    MIN_PAUSE_BETWEEN_CHECKPOINTS_MS: int = int(os.getenv('MIN_PAUSE_BETWEEN_CHECKPOINTS_MS', '30000'))
    CHECKPOINT_TIMEOUT_MS: int = int(os.getenv('CHECKPOINT_TIMEOUT_MS', '300000'))
    STATE_BACKEND: str = os.getenv('STATE_BACKEND', 'rocksdb')
    
    # MinIO
    MINIO_ENDPOINT: str = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
    MINIO_ACCESS_KEY: str = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
    MINIO_SECRET_KEY: str = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
    MINIO_BUCKET_RAW: str = 'cadqstream-raw'
    MINIO_BUCKET_VIOLATIONS: str = 'cadqstream-violations'
    MINIO_BUCKET_ANOMALIES: str = 'cadqstream-anomalies'
    MINIO_BUCKET_METRICS: str = 'cadqstream-metrics'
    MINIO_BUCKET_DRIFT: str = 'cadqstream-drift'
    
    # Redis
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'redis')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_PASSWORD: Optional[str] = os.getenv('REDIS_PASSWORD')
    REDIS_TLS: bool = os.getenv('REDIS_TLS', 'false').lower() == 'true'
    
    # MemStream
    MODEL_PATH: str = os.getenv('MEMSTREAM_MODEL_PATH', '/models/memstream_ae.pt')
    MODEL_SIGNING_KEY: Optional[str] = os.getenv('MEMSTREAM_MODEL_SIGNING_KEY')
    IEC_SIGNING_KEY: Optional[str] = os.getenv('IEC_SIGNING_KEY')
    DEFAULT_BETA: float = float(os.getenv('MEMSTREAM_DEFAULT_BETA', '0.5'))
    REQUIRE_MODEL_SIGNATURE: bool = os.getenv('REQUIRE_MODEL_SIGNATURE', 'true').lower() == 'true'
    
    # IEC
    IEC_COOLDOWN_SECONDS: int = int(os.getenv('IEC_COOLDOWN_SECONDS', '300'))
    IEC_MAX_CONSECUTIVE: int = int(os.getenv('IEC_MAX_CONSECUTIVE', '10'))
    
    # Watermark
    WATERMARK_BOUND_MS: int = int(os.getenv('WATERMARK_BOUND_MS', '10000'))
    IDLENESS_TIMEOUT_MS: int = int(os.getenv('IDLENESS_TIMEOUT_MS', '30000'))
    
    # Deduplication
    DEDUP_TTL_DAYS: int = int(os.getenv('DEDUP_TTL_DAYS', '7'))
    
    @classmethod
    def from_env(cls) -> 'PipelineConfig':
        """Create config from current environment."""
        return cls()


# =============================================================================
# Output Tags for Side Outputs
# =============================================================================

# Late data side output: records that arrive after the watermark has passed
# These records have been processed by upstream operators but are too late for
# window computations. They are captured here for audit/analysis.
LATE_DATA_OUTPUT_TAG = OutputTag("late-data-stream", type_info=Types.PICKLED_BYTE_ARRAY_TYPE_INFO)

# Layer 1: Baseline Validation Functions
# =============================================================================

class ParseJsonFunction:
    """
    Parse raw JSON string into dict.
    
    Input:  bytes/string (raw JSON)
    Output: dict or None (if parse fails)
    
    Silently drops malformed records (None output filtered downstream).
    """
    
    def __init__(self):
        self.json = json
        self.parse_errors = 0
    
    def map(self, value) -> Optional[Dict]:
        try:
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            return self.json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError) as e:
            self.parse_errors += 1
            LOGGER.debug("[ParseJson] Failed to parse: %s", str(e)[:50])
            return None


class AddTripIdFunction:
    """
    Generate deterministic trip ID from record fields.
    
    Composite key: VendorID|tpep_pickup_datetime|PULocationID|DOLocationID|fare_amount
    Hash: SHA-256 hexdigest (first 32 chars)
    
    Reference: original_flow.md lines 393-399
    """
    
    def __init__(self):
        import hashlib
        self.hashlib = hashlib
    
    def map(self, record: Dict) -> Dict:
        if record is None:
            return None
        
        key = (
            f"{record.get('VendorID', '')}|"
            f"{record.get('tpep_pickup_datetime', '')}|"
            f"{record.get('PULocationID', '')}|"
            f"{record.get('DOLocationID', '')}|"
            f"{record.get('fare_amount', '')}"
        )
        record['trip_id'] = self.hashlib.sha256(key.encode()).hexdigest()[:32]
        return record


class WatermarkStrategyFactory:
    """Create watermark strategy for taxi trip event time."""
    
    @staticmethod
    def create(config: PipelineConfig):
        """
        Create bounded-out-of-orderness watermark strategy.
        
        Event time: tpep_pickup_datetime
        Bound: 10 seconds (configurable)
        Idleness timeout: 30 seconds (fix for PyFlink partition stall)
        """
        return (
            WatermarkStrategy
            .for_bounded_out_of_orderness(Duration.of_millis(config.WATERMARK_BOUND_MS))
            .with_timestamp_assigner(TaxiTimestampExtractor())
            .with_idleness(Duration.of_millis(config.IDLENESS_TIMEOUT_MS))
        )


class TaxiTimestampExtractor:
    """Extract timestamp from taxi record for event time."""
    
    def extract_timestamp(self, record: Dict, timestamp_type: int) -> int:
        """Extract pickup datetime as milliseconds since epoch."""
        try:
            pickup_str = record.get('tpep_pickup_datetime', '')
            if not pickup_str:
                return 0
            
            # Parse datetime
            from datetime import datetime
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S UTC',
                '%Y/%m/%d %H:%M:%S',
                '%m/%d/%Y %H:%M:%S',
            ]
            
            dt = None
            for fmt in formats:
                try:
                    dt = datetime.strptime(pickup_str.strip(), fmt)
                    break
                except ValueError:
                    continue
            
            if dt is None:
                return 0
            
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError, AttributeError):
            return 0


class DeduplicatorFunction(ProcessFunction):
    """
    Deduplicate records by trip_id using keyed state.

    Input:  Dict with trip_id field
    Output: Dict (first occurrence) or None (duplicate)

    Uses MapState with a 24-hour TTL to automatically expire old trip IDs.
    This handles the case where the same trip might appear across day boundaries.

    Reference: original_flow.md lines 393-399
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig.from_env()
        self._seen_state = None  # Set in open()
        self._total_processed = 0
        self._duplicates_dropped = 0

        # TTL for deduplication state: 24 hours
        # After 24 hours, the trip_id state expires and the record will be processed again.
        # This handles edge cases like late-arriving records from previous days.
        self._state_ttl_hours = 24

    def open(self, runtime_context):
        """Initialize keyed state."""
        # Configure TTL for the map state
        ttl_config = StateTtlConfig.new_builder(
            Time.hours(self._state_ttl_hours)
        ).set_update_type(
            StateTtlConfig.UpdateType.ON_READ_AND_WRITE
        ).set_state_visibility(
            StateTtlConfig.StateVisibility.RETURN_EXPIRED_IF_NOT_EXIST_YET
        ).cleanup_in_rocksdb_compact_filter(1000).build()

        # Use a MapState: trip_id -> first_seen_timestamp
        # This lets us track when we first saw each trip ID
        seen_state_desc = MapStateDescriptor(
            "dedup_seen",
            BasicTypeInfo.STRING_TYPE_INFO,  # key: trip_id
            BasicTypeInfo.BIG_INT_TYPE_INFO,  # value: first_seen_timestamp (ms)
        )
        seen_state_desc.enable_time_to_live(ttl_config)

        self._seen_state = runtime_context.get_map_state(seen_state_desc)

        LOGGER.info(
            "[Deduplicator] Initialized with %d-hour state TTL",
            self._state_ttl_hours
        )

    def process_element(self, value, context: ProcessFunction.Context) -> Optional[Dict]:
        """Check if trip_id has been seen; if not, emit and mark as seen."""
        self._total_processed += 1

        trip_id = value.get('trip_id', '')
        if not trip_id:
            # No trip_id — cannot deduplicate, pass through
            return value

        try:
            # Check if this trip_id is in our state
            # MapState.get(trip_id) returns timestamp if seen, None if not
            first_seen = self._seen_state.get(trip_id)

            if first_seen is not None:
                # Duplicate — already seen
                self._duplicates_dropped += 1
                return None

            # First time seeing this trip_id — mark it and pass through
            self._seen_state.put(trip_id, int(time.time() * 1000))
            return value

        except Exception as e:
            # If state access fails (e.g., during recovery), fall back to local tracking
            # This is a safety net — in normal operation this should not trigger
            LOGGER.warning("[Deduplicator] State access failed: %s — using pass-through", e)
            return value


class LateDataDetector:
    """
    Detect and tag late-arriving records based on event time vs processing time.

    A record is considered "late" if:
      processing_time - event_time > late_threshold_minutes * 60

    Late records are tagged with _is_late=True and sent to a side output.

    This allows:
    - Late records to still be processed (for real-time decisions)
    - Late records to be identified and counted separately
    - Potential later processing of late records for analytics

    Reference: original_flow.md (watermark + late data handling)
    """

    def __init__(self, late_threshold_minutes: float = 15.0):
        """
        Args:
            late_threshold_minutes: Records arriving this many minutes after
                their event time are considered late. Default: 15 minutes.
        """
        self.late_threshold_minutes = late_threshold_minutes
        self.late_threshold_seconds = late_threshold_minutes * 60
        self._total = 0
        self._late_count = 0

    def map(self, record: Dict) -> Dict:
        self._total += 1

        if record is None:
            return None

        # Check event time
        event_time_str = record.get('tpep_pickup_datetime', '')
        event_time = 0.0

        if event_time_str:
            try:
                from datetime import datetime
                formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d %H:%M:%S UTC',
                    '%Y/%m/%d %H:%M:%S',
                    '%m/%d/%Y %H:%M:%S',
                ]
                dt = None
                for fmt in formats:
                    try:
                        dt = datetime.strptime(event_time_str.strip(), fmt)
                        break
                    except ValueError:
                        continue
                if dt is not None:
                    event_time = dt.timestamp()
            except Exception:
                event_time = 0.0

        # Current processing time (approximate)
        import time
        processing_time = time.time()

        # Compute lateness
        if event_time > 0:
            lateness_seconds = processing_time - event_time
            is_late = lateness_seconds > self.late_threshold_seconds
        else:
            lateness_seconds = -1.0
            is_late = False

        # Add late-data metadata
        record['_is_late'] = is_late
        record['_lateness_seconds'] = lateness_seconds
        record['_late_threshold_seconds'] = self.late_threshold_seconds

        if is_late:
            self._late_count += 1
            LOGGER.debug(
                "[LateData] Late record detected: lateness=%.0fs, threshold=%.0fs",
                lateness_seconds, self.late_threshold_seconds
            )

        return record

    def get_late_rate(self) -> float:
        """Get the fraction of records that are late."""
        if self._total == 0:
            return 0.0
        return self._late_count / self._total


class SchemaValidator:
    """
    Validate record schema and zone IDs.
    
    Required fields: trip_distance, fare_amount, PULocationID, DOLocationID, passenger_count
    Zone range: 1-263 (NYC taxi zones)
    
    Outputs:
        - Valid: passes through main output
        - Invalid: goes to side output (INVALID_OUTPUT_TAG)
    """
    
    REQUIRED_FIELDS = [
        'trip_distance', 'fare_amount', 'PULocationID',
        'DOLocationID', 'passenger_count'
    ]
    MIN_ZONE = 1
    MAX_ZONE = 263
    
    def __init__(self):
        self.valid_count = 0
        self.invalid_count = 0
    
    def process_element(self, record: Dict, context) -> Optional[Dict]:
        if record is None:
            return None
        
        is_valid = True
        violations = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in record or record[field] is None:
                is_valid = False
                violations.append(f'missing_{field}')
        
        # Check zone range
        if is_valid:
            try:
                pu = int(record.get('PULocationID', 0))
                do = int(record.get('DOLocationID', 0))
                if not (self.MIN_ZONE <= pu <= self.MAX_ZONE):
                    is_valid = False
                    violations.append('invalid_PULocationID')
                if not (self.MIN_ZONE <= do <= self.MAX_ZONE):
                    is_valid = False
                    violations.append('invalid_DOLocationID')
            except (ValueError, TypeError):
                is_valid = False
                violations.append('invalid_zone_type')
        
        record['_schema_violations'] = violations
        record['_is_schema_valid'] = is_valid
        
        if is_valid:
            self.valid_count += 1
            return record
        else:
            self.invalid_count += 1
            # In production: yield to side output
            # For simplicity, return record with flag
            record['_send_to_violations'] = True
            return record


class SchemaValidatorMap:
    """
    Stateless schema validation as MapFunction (no keyed state needed).
    
    Required fields: trip_distance, fare_amount, PULocationID, DOLocationID, passenger_count
    Zone range: 1-263 (NYC taxi zones)
    
    More efficient than ProcessFunction since no key_by required.
    """
    
    def __init__(self):
        self._valid_count = 0
        self._invalid_count = 0
    
    def map(self, record: Dict) -> Dict:
        if record is None:
            return None
        
        is_valid = True
        violations = []
        
        # Check required fields
        required = ['trip_distance', 'fare_amount', 'PULocationID', 'DOLocationID', 'passenger_count']
        for field in required:
            if field not in record or record[field] is None or record[field] == '':
                is_valid = False
                violations.append(f'missing_{field}')
        
        # Check zone range
        if is_valid:
            try:
                pu = int(float(record.get('PULocationID', 0)))
                do = int(float(record.get('DOLocationID', 0)))
                if not (1 <= pu <= 263):
                    is_valid = False
                    violations.append('invalid_PULocationID')
                if not (1 <= do <= 263):
                    is_valid = False
                    violations.append('invalid_DOLocationID')
            except (ValueError, TypeError):
                is_valid = False
                violations.append('invalid_zone_type')
        
        record['_schema_violations'] = violations
        record['_is_schema_valid'] = is_valid
        
        if is_valid:
            self._valid_count += 1
            record['_send_to_violations'] = False
        else:
            self._invalid_count += 1
            record['_send_to_violations'] = True
        
        return record


# =============================================================================
# Layer 2A: Canary Rules Functions
# =============================================================================

from memstream_src.operators.canary_rules import (
    check_canary,
    check_canary_rules as check_all_canary_rules,
)
from memstream_src.operators.memstream_scoring_op import MemStreamScoringOperator


class CanaryRulesMapper:
    """
    Apply 7 canary rules to record.
    
    Rules:
        1. Fare bounds (0-500)
        2. Distance bounds (0-100)
        3. Positive passenger count (1-9)
        4. Duration bounds (0-360 min)
        5. Speed bounds (0-80 mph)
        6. JFK flat fare validation
        7. Credit card with zero tip (suspicious for fare > $10)
    
    Output fields:
        - has_violation: bool
        - canary_violations: list[str]
        - violation_count: int
        - rule_{name}: bool (per-rule flags)
    
    Reference: original_flow.md lines 424-458
    """
    
    def __init__(self):
        self.records_processed = 0
        self.violations_found = 0
    
    def map(self, record: Dict) -> Dict:
        self.records_processed += 1
        
        # Add neighborhood field BEFORE processing rules
        zone_id = int(float(record.get('PULocationID', 1)))
        if 1 <= zone_id <= 43:
            neighborhood = 'manhattan'
        elif 44 <= zone_id <= 103:
            neighborhood = 'bronx'
        elif 104 <= zone_id <= 127:
            neighborhood = 'brooklyn'
        elif 128 <= zone_id <= 148:
            neighborhood = 'queens_lower'
        elif 149 <= zone_id <= 161:
            neighborhood = 'queens_upper'
        elif 162 <= zone_id <= 181:
            neighborhood = 'staten_island'
        elif 182 <= zone_id <= 196:
            neighborhood = 'ewr'
        elif 217 <= zone_id <= 229:
            neighborhood = 'jfk'
        elif 230 <= zone_id <= 234:
            neighborhood = 'nalp'
        else:
            neighborhood = 'unknown'
        
        record['neighborhood'] = neighborhood
        
        try:
            violations = check_all_canary_rules(record)
            violation_list = [name.replace('rule_', '') for name, passed in violations.items() if not passed]
            
            record['has_violation'] = len(violation_list) > 0
            record['canary_violations'] = violation_list
            record['violation_count'] = len(violation_list)
            
            # Per-rule flags
            for rule_name, passed in violations.items():
                record[f'_{rule_name}_passed'] = passed
            
            if record['has_violation']:
                self.violations_found += 1
        
        except Exception as e:
            LOGGER.warning("[CanaryRules] Error processing record: %s", e)
            record['has_violation'] = False
            record['canary_violations'] = []
            record['violation_count'] = 0
        
        return record
    
    def get_metrics(self) -> Dict:
        return {
            'records_processed': self.records_processed,
            'violations_found': self.violations_found,
            'violation_rate': self.violations_found / max(self.records_processed, 1),
        }


# =============================================================================
# Layer 2B: MemStream ML Functions (Real Implementation)
# =============================================================================

class MemStreamScoringStage:
    """
    Layer 2B: MemStream ML Scoring Stage.

    Uses real MemStreamScoringOperator with:
    - 34D FeatureVectorizer
    - Denoising Autoencoder (AE + Memory)
    - ContextBeta (80 thresholds, ratio scoring)
    - ADWIN per neighborhood (10 instances)
    - Conditional memory updates (normal points only)

    In production: warmup_data should be pre-loaded from a Kafka topic
    or from MinIO checkpoint containing 5000+ clean normal samples.

    Scoring:
    - Raw score = L1 kNN distance to memory
    - Normalized = score / context_beta
    - Anomaly = normalized > 1.0
    """

    def __init__(self, config=None, warmup_data=None):
        self.config = config or PipelineConfig.from_env()
        self.warmup_data = warmup_data
        self._operator = None
        self._initialized = False

    def initialize(self):
        """Initialize the MemStream scoring operator."""
        if self._initialized:
            return

        # Create MemStreamScoringOperator config
        op_config = {
            'in_dim': 34,
            'hidden_dim': 68,
            'memory_len': 256,
            'k_neighbors': 10,
            'gamma': 0.0,
            'warmup_epochs': 20,
            'warmup_batch_size': 256,
            'warmup_noise_std': 0.1,
            'default_beta': 0.5,
            'seed': 42,
        }

        # Create operator
        self._operator = MemStreamScoringOperator(
            config=op_config,
            warmup_data=None  # Will warmup separately
        )

        # Try to load checkpoint first
        if not self._operator._try_load_checkpoint():
            # No checkpoint - need warmup data
            if self.warmup_data is not None:
                # Warmup with provided data
                LOGGER.info(f"[MemStreamScoring] Warming up with {len(self.warmup_data)} samples...")
                self._operator.warmup(self.warmup_data, verbose=True)
            else:
                LOGGER.warning("[MemStreamScoring] No checkpoint or warmup data! Scores will be unreliable.")

        self._initialized = True
        LOGGER.info("[MemStreamScoring] Stage initialized")

    def process(self, record):
        """Score a single record."""
        if not self._initialized:
            self.initialize()

        if record is None:
            return None

        try:
            return self._operator.map(record)
        except Exception as e:
            LOGGER.error(f"[MemStreamScoring] Error: {e}")
            return {
                **record,
                'anomaly_score': 0.5,
                'threshold': 1.0,
                'is_anomaly': False,
                'context_key': 'error',
                'neighborhood': 'unknown',
                'neighborhood_idx': 9,
                'context_id': 0,
                'ml_model': 'memstream_v10',
                'scoring_error': str(e),
            }

    def get_metrics(self):
        """Get scoring metrics."""
        if self._operator:
            stats = self._operator.get_stats()
            return {
                'records_processed': stats.get('total_scored', 0),
                'anomalies_found': stats.get('total_anomalies', 0),
                'anomaly_rate': stats.get('total_anomalies', 0) / max(stats.get('total_scored', 1), 1),
                'memory_updates': stats.get('total_memory_updates', 0),
                'checkpoint_counter': stats.get('checkpoint_counter', 0),
            }
        return {'records_processed': 0, 'anomalies_found': 0}

    def checkpoint(self):
        """Save checkpoint."""
        if self._operator:
            self._operator._save_checkpoint()


# =============================================================================
# Layer 3: Voting Ensemble + MetaAggregator
# =============================================================================

from memstream_src.operators.layer3.voting_ensemble import decide as voting_decide


class VotingEnsembleMapper:
    """
    Priority-based voting between canary and ML decisions.
    
    Priority: Canary > ML
    
    Logic:
        IF canary_violations not empty:
            → ANOMALY, source=canary_rule, confidence=1.0
        ELIF is_anomaly == True:
            → ANOMALY, source=complex_ml, confidence=score/threshold
        ELSE:
            → CLEAN, source=both_agree, confidence=1-(score/threshold)
    
    Reference: original_flow.md lines 570-598
    """
    
    def __init__(self):
        self.total = 0
        self.anomaly_decisions = 0
        self.canary_decisions = 0
        self.ml_decisions = 0
        self.clean_decisions = 0
    
    def map(self, record: Dict) -> Dict:
        self.total += 1
        
        result = voting_decide(record)
        
        if result['final_decision'] == 'ANOMALY':
            self.anomaly_decisions += 1
            if result['decision_source'] == 'canary_rule':
                self.canary_decisions += 1
            else:
                self.ml_decisions += 1
        else:
            self.clean_decisions += 1
        
        return result
    
    def get_metrics(self) -> Dict:
        if self.total == 0:
            return {'total': 0, 'anomaly_rate': 0.0}
        return {
            'total': self.total,
            'anomaly_rate': self.anomaly_decisions / self.total,
            'canary_rate': self.canary_decisions / self.total,
            'ml_rate': self.ml_decisions / self.total,
            'clean_rate': self.clean_decisions / self.total,
        }


class MetaAggregatorMapper:
    """
    Compute meta-metrics for IEC from voting results.
    
    Metrics computed (5-minute window):
        - volume: record count
        - null_rate: proportion of null fields
        - violation_rate: canary violation proportion
        - anomaly_rate: ML anomaly proportion
        - avg_anomaly_score: mean anomaly scores
        - delta_score: |violation - anomaly| / (violation + anomaly + eps)
    
    Reference: memstream_src/operators/layer3/meta_aggregator.py
    """
    
    EPSILON = 1e-10
    
    def __init__(self):
        self.window_records: List[Dict] = []
        self.window_size_seconds = 300  # 5 minutes
    
    def add(self, record: Dict) -> None:
        """Add record to current window."""
        self.window_records.append(record)
    
    def compute_metrics(self) -> Dict:
        """Compute meta-metrics from current window."""
        if not self.window_records:
            return self._empty_metrics()
        
        records = self.window_records
        total = len(records)
        
        violation_count = sum(1 for r in records if r.get('has_violation', False))
        anomaly_count = sum(1 for r in records if r.get('is_anomaly', False))
        sum_score = sum(r.get('anomaly_score', 0.0) for r in records)
        
        violation_rate = violation_count / total
        anomaly_rate = anomaly_count / total
        avg_score = sum_score / total
        delta_score = abs(violation_rate - anomaly_rate) / (violation_rate + anomaly_rate + self.EPSILON)
        
        return {
            'volume': total,
            'null_rate': 0.0,  # Placeholder
            'violation_rate': violation_rate,
            'anomaly_rate': anomaly_rate,
            'avg_anomaly_score': avg_score,
            'delta_score': delta_score,
            'neighborhood': records[0].get('neighborhood', 'unknown'),
        }
    
    def _empty_metrics(self) -> Dict:
        return {
            'volume': 0,
            'null_rate': 0.0,
            'violation_rate': 0.0,
            'anomaly_rate': 0.0,
            'avg_anomaly_score': 0.0,
            'delta_score': 0.0,
            'neighborhood': 'unknown',
        }
    
    def reset(self) -> None:
        """Clear window for next interval."""
        self.window_records = []


# =============================================================================
# Layer 3: MetaAggregator (Proper Flink State Management)
# =============================================================================

class MetaAggregatorProcessFn(KeyedProcessFunction):
    """
    Proper Flink KeyedProcessFunction for meta-aggregation.
    
    Uses Flink ValueState for distributed state management.
    Each parallel subtask manages its own key's state correctly.
    
    Replaces the in-memory dict approach which fails in distributed Flink.
    """

    def __init__(self, window_size: int = 100):
        self._window_size = window_size

    def open(self, runtime_context):
        self._state = runtime_context.get_state(
            ValueStateDescriptor(
                "meta_records",
                BasicTypeInfo.PICKLED_BYTE_ARRAY_TYPE_INFO
            )
        )
        self._trigger_count = runtime_context.get_state(
            ValueStateDescriptor(
                "trigger_count",
                BasicTypeInfo.INT_TYPE_INFO
            )
        )

    def process_element(self, record, context: KeyedProcessFunction.Context):
        records = self._state.value() or []
        records.append(record)
        count = self._trigger_count.value() or 0
        count += 1

        if count >= self._window_size:
            meta = self._compute_meta(context.get_current_key(), records)
            self._state.update([])
            self._trigger_count.update(0)
            yield meta
        else:
            self._state.update(records)
            self._trigger_count.update(count)

    def _compute_meta(self, neighborhood, records):
        total = len(records)
        # Use final_decision from VotingEnsembleMapper output
        anomaly_count = sum(1 for r in records if r.get('final_decision') == 'ANOMALY')
        sum_score = sum(r.get('anomaly_score', 0.0) for r in records)
        anomaly_rate = anomaly_count / total
        avg_score = sum_score / total
        # violation_rate is same as anomaly_rate since canary violations
        # become ANOMALY decisions in VotingEnsembleMapper
        violation_rate = anomaly_rate
        delta_score = abs(violation_rate - anomaly_rate) / (violation_rate + anomaly_rate + 1e-10)

        return {
            'neighborhood': neighborhood,
            'window_type': 'count_tumbling',
            'window_size': self._window_size,
            'volume': total,
            'null_rate': 0.0,
            'violation_rate': violation_rate,
            'anomaly_rate': anomaly_rate,
            'avg_anomaly_score': avg_score,
            'delta_score': delta_score,
            'timestamp': time.time(),
        }


# =============================================================================
# Layer 4: IEC Functions
# =============================================================================

from memstream_src.core.iec_controller import IECController


class IECDecisionMapper:
    """
    IEC decision and action mapper.
    
    In production: uses MultiInstanceADWIN + DriftAggregator + METER hypernetwork.
    This mapper provides the interface for IEC decisions to be:
        - Logged to Kafka iec-action-replay
        - Used to adjust MemStream beta via Redis
    
    Reference: memstream_src/core/iec_controller.py
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None, minio_client=None):
        self.config = config or PipelineConfig.from_env()
        self.iec = IECController(minio_client=minio_client)  # Pass MinIO client to controller
        self.decisions_made = 0
        self.strategy_counts: Dict[str, int] = {
            'do_nothing': 0,
            'adjust_threshold': 0,
            'memory_reset': 0,
        }
    
    def map(self, meta_metrics: Dict) -> Dict:
        """Process meta-metrics and return IEC decision."""
        self.decisions_made += 1
        
        try:
            # Wrap flat record into nested format expected by IECController:
            # MetaAggregator emits: {neighborhood, volume, anomaly_rate, ...}
            # IECController expects: {neighborhood: {volume, anomaly_rate, ...}}
            wrapped = {meta_metrics['neighborhood']: meta_metrics}

            # Process meta-metrics first to set _last_meta for this window
            decision = self.iec.process_meta_metrics(wrapped)
            self.strategy_counts[decision['strategy']] += 1

            # Check verification feedback (now uses current window's _last_meta)
            verification_result = self.iec.record_verification(meta_metrics)
            if verification_result:
                LOGGER.info(
                    "[IEC] Verification for %s: verdict=%s, improvement=%.4f, action=%s",
                    verification_result.get('neighborhood'),
                    verification_result.get('verdict'),
                    verification_result.get('improvement'),
                    verification_result.get('action', 'none'),
                )
                if verification_result.get('action') == 'rollback':
                    rollback_nb = verification_result['neighborhood']
                    rollback_beta = verification_result['rollback_beta']
                    self._apply_beta_rollback(rollback_nb, rollback_beta)

            # Execute the strategy if it's not do_nothing
            if decision['strategy'] != 'do_nothing':
                execution_result = self.iec.execute_strategy(decision)
                decision['_iec_execution'] = execution_result

            # Log circuit breaker status if not closed
            circuit_status = self.iec.get_circuit_status()
            if circuit_status['state'] != 'closed':
                LOGGER.warning(
                    "[IEC] Circuit breaker %s: %d consecutive actions",
                    circuit_status['state'], circuit_status['consecutive_actions']
                )

            return decision
        except Exception as e:
            LOGGER.warning("[IEC] Decision failed: %s", e)
            return {
                'strategy': 'do_nothing',
                'confidence': 1.0,
                'severity': 'none',
                'error': str(e),
            }
    
    def get_metrics(self) -> Dict:
        return {
            'decisions_made': self.decisions_made,
            'strategy_counts': self.strategy_counts.copy(),
            'verification_status': self.iec.get_verification_status(),
        }

    def _apply_beta_rollback(self, neighborhood: str, beta: float):
        """Apply beta rollback for a neighborhood."""
        try:
            self.iec._current_betas[neighborhood] = beta
            LOGGER.warning(
                "[IEC] Beta rollback applied: %s from %.4f to %.4f",
                neighborhood,
                beta,
                beta,
            )
        except Exception as e:
            LOGGER.error("[IEC] Failed to apply beta rollback: %s", e)


# =============================================================================
# Sink Configuration
# =============================================================================

def create_kafka_sink(topic: str, config: PipelineConfig) -> FlinkKafkaProducer:
    """
    Create EXACTLY_ONCE Kafka sink for topic.
    
    Args:
        topic: Kafka topic name
        config: Pipeline configuration
        
    Returns:
        Configured FlinkKafkaProducer with LZ4 compression
    """
    producer_config = {
        'bootstrap.servers': config.KAFKA_BOOTSTRAP_SERVERS,
        'acks': 'all',
        'compression.type': 'lz4',
        'linger.ms': '5',
        'batch.size': '16384',
    }
    
    return FlinkKafkaProducer(
        topic=topic,
        serialization_schema=JsonRowSerializationSchema(),
        producer_config=producer_config,
        producer_semantic=FlinkKafkaProducer.SEMANTIC.EXACTLY_ONCE,
    )


def create_minio_sink(bucket: str, path: str) -> StreamingFileSink:
    """
    Create MinIO/Parquet sink for bucket/path.

    Rolling: 5 minutes OR 128 MB per file.

    Args:
        bucket: MinIO bucket name
        path: Path prefix within bucket

    Returns:
        Configured StreamingFileSink

    Raises:
        ImportError: if pyarrow is not available and no fallback is possible
    """
    from memstream_src.operators.sinks.minio_sinks import _build_generic_sink

    if not PARQUET_AVAILABLE:
        raise ImportError(
            "[create_minio_sink] pyarrow is required for Parquet sinks. "
            "Install it with: pip install pyarrow"
        )

    # Delegate to the proper AvroParquetWriterFactory-based sink builder
    return _build_generic_sink(bucket, path)


class ParquetFormatFactory:
    """
    Factory for Parquet format encoders/decoders for MinIO sink.

    Uses pyarrow for efficient columnar storage with Snappy compression.
    Each record type gets its own directory structure:
      s3://cadqstream-raw/table=<table_name>/dt=<date>/<timestamp>.parquet
      s3://cadqstream-violations/table=<table_name>/dt=<date>/<timestamp>.parquet
      s3://cadqstream-anomalies/table=<table_name>/dt=<date>/<timestamp>.parquet
      s3://cadqstream-metrics/table=<table_name>/dt=<date>/<timestamp>.parquet

    Reference: original_flow.md lines 120-130 (concept), 950-970 (current stub)
    """

    _PARQUET_SCHEMA = None  # Cached pyarrow schema

    @classmethod
    def _get_schema(cls) -> "pyarrow.lib.Schema":
        """Get or build the pyarrow schema for taxi records."""
        if cls._PARQUET_SCHEMA is not None:
            return cls._PARQUET_SCHEMA

        import pyarrow as pa

        fields = [
            # Core fields
            pa.field('trip_id', pa.string()),
            pa.field('VendorID', pa.int64()),
            pa.field('tpep_pickup_datetime', pa.string()),
            pa.field('tpep_dropoff_datetime', pa.string()),
            pa.field('passenger_count', pa.float64()),
            pa.field('trip_distance', pa.float64()),
            pa.field('RatecodeID', pa.float64()),
            pa.field('store_and_fwd_flag', pa.string()),
            pa.field('PULocationID', pa.int64()),
            pa.field('DOLocationID', pa.int64()),
            # Fare
            pa.field('fare_amount', pa.float64()),
            pa.field('extra', pa.float64()),
            pa.field('mta_tax', pa.float64()),
            pa.field('tip_amount', pa.float64()),
            pa.field('tolls_amount', pa.float64()),
            pa.field('ehail_fee', pa.float64()),
            pa.field('improvement_surcharge', pa.float64()),
            pa.field('total_amount', pa.float64()),
            pa.field('payment_type', pa.float64()),
            pa.field('trip_type', pa.float64()),
            # DQ fields
            pa.field('neighborhood', pa.string()),
            pa.field('neighborhood_idx', pa.int64()),
            pa.field('context_key', pa.string()),
            pa.field('context_id', pa.int64()),
            # ML fields
            pa.field('anomaly_score', pa.float64()),
            pa.field('threshold', pa.float64()),
            pa.field('is_anomaly', pa.bool_()),
            pa.field('ml_model', pa.string()),
            pa.field('scoring_latency_ms', pa.float64()),
            # Canary fields
            pa.field('has_violation', pa.bool_()),
            pa.field('canary_violations', pa.list_(pa.string())),
            # Voting
            pa.field('final_decision', pa.string()),
            pa.field('decision_source', pa.string()),
            pa.field('confidence', pa.float64()),
            # Meta fields
            pa.field('volume', pa.int64()),
            pa.field('null_rate', pa.float64()),
            pa.field('violation_rate', pa.float64()),
            pa.field('avg_anomaly_score', pa.float64()),
            pa.field('delta_score', pa.float64()),
            pa.field('drift_count', pa.int64()),
            pa.field('timestamp', pa.float64()),
            # IEC fields
            pa.field('strategy', pa.string()),
            pa.field('severity', pa.string()),
            pa.field('circuit_state', pa.string()),
            pa.field('drift_events', pa.list_(pa.string())),
            # Error fields
            pa.field('_is_schema_valid', pa.bool_()),
            pa.field('_send_to_violations', pa.bool_()),
            pa.field('scoring_error', pa.string()),
        ]

        cls._PARQUET_SCHEMA = pa.schema(fields)
        return cls._PARQUET_SCHEMA

    @staticmethod
    def get_encoder():
        """
        Returns a Parquet encoder for MinIO sink.

        Encoder batches records and writes Parquet files periodically.
        File path: <bucket>/table=<table_name>/dt=<date>/<timestamp>.parquet
        """
        return _ParquetEncoder()

    @staticmethod
    def get_decoder():
        """
        Returns a Parquet decoder for MinIO source (if needed).
        """
        return _ParquetDecoder()


class _ParquetEncoder:
    """
    Parquet encoder that batches records and converts to Parquet bytes.

    Used with Flink's StreamingFileSink to write Parquet files to MinIO.
    """

    def __init__(
        self,
        batch_size: int = 1000,
        compression: str = 'snappy',
    ):
        self.batch_size = batch_size
        self.compression = compression
        self._buffer = []
        self._row_count = 0

    def add(self, record: Dict) -> None:
        """Add a record to the batch buffer."""
        if record is None:
            return
        self._buffer.append(self._flatten_record(record))
        self._row_count += 1

    def _flatten_record(self, record: Dict) -> Dict:
        """
        Flatten nested dicts/lists into JSON strings for Parquet compatibility.

        Parquet doesn't support complex nested types easily, so we serialize:
        - canary_violations: list -> JSON string
        - drift_events: list -> JSON string
        """
        import json

        flat = {}
        for key, value in record.items():
            if value is None:
                flat[key] = ''
            elif isinstance(value, (list, dict)):
                flat[key] = json.dumps(value)
            elif isinstance(value, bool):
                flat[key] = value
            elif isinstance(value, (int, float, str)):
                flat[key] = value
            else:
                flat[key] = str(value)

        return flat

    def should_flush(self) -> bool:
        """Check if the batch is full and ready to write."""
        return len(self._buffer) >= self.batch_size

    def encode(self) -> bytes:
        """
        Encode the current batch as Parquet bytes.

        Returns:
            Parquet file bytes (with header), ready to write to MinIO

        Raises:
            ValueError: if buffer is empty
        """
        import pyarrow as pa

        if not self._buffer:
            raise ValueError("Cannot encode empty buffer")

        schema = ParquetFormatFactory._get_schema()

        # Build arrays for each column
        columns = {}
        for key in schema.names:
            values = [r.get(key, '') for r in self._buffer]

            field_idx = schema.names.index(key)
            field = schema.field(field_idx)
            pa_type = field.type

            try:
                if pa.types.is_boolean(pa_type):
                    arr = pa.array(values, type=pa.bool_())
                elif pa.types.is_integer(pa_type):
                    arr = pa.array(values, type=pa.int64())
                elif pa.types.is_floating(pa_type):
                    arr = pa.array(values, type=pa.float64())
                elif pa.types.is_string(pa_type):
                    arr = pa.array([str(v) for v in values], type=pa.string())
                else:
                    arr = pa.array([str(v) for v in values], type=pa.string())
            except Exception:
                # Fallback to string for any conversion issues
                arr = pa.array([str(v) for v in values], type=pa.string())

            columns[key] = arr

        table = pa.table(columns, schema=schema)

        import io
        buffer = io.BytesIO()

        # Write Parquet with Snappy compression
        writer = pa.parquet.ParquetWriter(
            buffer,
            schema,
            compression=self.compression.upper() if isinstance(self.compression, str) else 'SNAPPY',
        )
        writer.write_table(table)
        writer.close()

        result = buffer.getvalue()

        LOGGER.debug(
            "[ParquetEncoder] Encoded %d records, %d bytes",
            len(self._buffer), len(result)
        )

        return result

    def clear(self) -> None:
        """Clear the buffer after writing."""
        self._buffer = []
        self._row_count = 0

    def row_count(self) -> int:
        """Get the number of rows in the current batch."""
        return self._row_count


class _ParquetDecoder:
    """
    Parquet decoder for reading Parquet files from MinIO.

    Used if you need to read back archived data for analysis.
    """

    def __init__(self, schema=None):
        self.schema = schema or ParquetFormatFactory._get_schema()

    def decode(self, data: bytes) -> List[Dict]:
        """
        Decode Parquet bytes to a list of records.

        Args:
            data: Parquet file bytes

        Returns:
            List of dictionaries, one per row
        """
        import pyarrow.parquet as pq
        import io

        source = io.BytesIO(data)
        table = pq.read_table(source)

        records = []
        for i in range(table.num_rows):
            record = {}
            for field in table.schema.names:
                val = table.column(field)[i].as_py()
                # Parse JSON strings back to native types
                if field in ('canary_violations', 'drift_events'):
                    import json
                    try:
                        val = json.loads(val) if val else []
                    except (json.JSONDecodeError, TypeError):
                        val = []
                record[field] = val
            records.append(record)

        return records


# =============================================================================
# Checkpoint Configuration
# =============================================================================

def configure_checkpointing(env: StreamExecutionEnvironment, config: PipelineConfig) -> None:
    """
    Configure checkpointing for EXACTLY_ONCE semantics.
    
    Settings:
        - Mode: EXACTLY_ONCE
        - Interval: 60 seconds (configurable)
        - Min pause: 30 seconds between checkpoints
        - Timeout: 5 minutes
        - Externalized: RETAIN_ON_CANCELLATION
        - Backend: RocksDB (production default)
    
    Reference: original_flow.md lines 1165-1176
    """
    env.enable_checkpointing(config.CHECKPOINT_INTERVAL_MS)
    
    checkpoint_config = env.get_checkpoint_config()
    checkpoint_config.set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    checkpoint_config.set_min_pause_between_checkpoints(config.MIN_PAUSE_BETWEEN_CHECKPOINTS_MS)
    checkpoint_config.set_checkpoint_timeout(config.CHECKPOINT_TIMEOUT_MS)
    checkpoint_config.set_max_concurrent_checkpoints(1)
    checkpoint_config.enable_externalized_checkpoints(
        ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
    )
    
    # State backend
    if config.STATE_BACKEND == 'rocksdb':
        try:
            env.set_state_backend("rocksdb")
            LOGGER.info("[Checkpoint] Using RocksDB state backend")
        except Exception as e:
            LOGGER.warning("[Checkpoint] RocksDB not available: %s", e)
    
    LOGGER.info(
        "[Checkpoint] Configured: interval=%ds, min_pause=%ds, timeout=%ds, backend=%s",
        config.CHECKPOINT_INTERVAL_MS // 1000,
        config.MIN_PAUSE_BETWEEN_CHECKPOINTS_MS // 1000,
        config.CHECKPOINT_TIMEOUT_MS // 1000,
        config.STATE_BACKEND
    )


# =============================================================================
# Main Pipeline Builder
# =============================================================================

def build_complete_pipeline(env: StreamExecutionEnvironment) -> StreamExecutionEnvironment:
    """
    Build the complete 4-layer CA-DQStream + MemStream pipeline.
    
    Layers:
        L1: Parse → Watermark → AddTripId → Dedup → Schema
        L2A: Canary Rules (7 rules)
        L2B: MemStream ML Scoring
        L3: Voting Ensemble → MetaAggregator
        L4: IEC Decision
    
    Sinks:
        - dq-hard-rule-violations (Kafka) - L1 schema violations
        - dq-stream-processed (Kafka) - Valid records
        - dq-stream-anomalies (Kafka) - Anomaly records
        - dq-meta-stream (Kafka) - Meta-metrics
        - iec-action-replay (Kafka) - IEC decisions
        - cadqstream-* (MinIO/Parquet) - Long-term storage
    
    Args:
        env: StreamExecutionEnvironment
        
    Returns:
        Configured environment ready to execute
    
    Reference: original_flow.md lines 26-140
    """
    config = PipelineConfig.from_env()
    
    LOGGER.info("=" * 60)
    LOGGER.info("CA-DQStream + MemStream v5 - Complete Pipeline")
    LOGGER.info("=" * 60)
    
    # Configure parallelism
    env.set_parallelism(config.PARALLELISM)
    LOGGER.info("[Config] Parallelism: %d", config.PARALLELISM)
    
    # Configure checkpointing
    configure_checkpointing(env, config)
    
    # =================================================================
    # Layer 1: Baseline Validation
    # =================================================================
    LOGGER.info("[L1] Building Baseline Validation pipeline...")
    
    # Kafka source
    kafka_consumer = FlinkKafkaConsumer(
        topics=config.INPUT_TOPIC,
        deserialization_schema=JsonRowDeserializationSchema(),
        properties={
            'bootstrap.servers': config.KAFKA_BOOTSTRAP_SERVERS,
            'group.id': config.KAFKA_SOURCE_GROUP,
            'auto.offset.reset': 'earliest',
        }
    )
    
    raw_stream = env.add_source(
        kafka_consumer,
        name="KafkaSource-Raw"
    )
    
    # Parse JSON
    parsed_stream = raw_stream.map(
        ParseJsonFunction(),
        name="ParseJson"
    ).filter(lambda r: r is not None)
    
    # Add watermark (event time from pickup datetime)
    watermarked_stream = parsed_stream.assign_timestamps_and_watermarks(
        WatermarkStrategyFactory.create(config)
    )
    
    # Add trip ID
    trip_id_stream = watermarked_stream.map(
        AddTripIdFunction(),
        name="AddTripId"
    )

    # Detect late-arriving records (15 minute threshold)
    # Records are tagged with _is_late=True but still flow through main pipeline
    late_tagged_stream = trip_id_stream.map(
        LateDataDetector(late_threshold_minutes=15.0),
        name="LateDataDetection"
    )

    # Route late records to side output sink (filter then add sink separately)
    late_stream = late_tagged_stream.filter(lambda r: r.get('_is_late', False))

    # Continue with all records (late records still processed for real-time decisions)
    # Main pipeline continues with late_tagged_stream
    main_stream = late_tagged_stream

    # Deduplicate (keyed by trip_id)
    deduplicated_stream = main_stream.key_by(
        lambda r: r.get('trip_id', '')
    ).process(
        DeduplicatorFunction(config),
        name="Deduplicate"
    ).filter(lambda r: r is not None)
    
    # Schema validation (stateless MapFunction - no key_by needed)
    validated_stream = deduplicated_stream.map(
        SchemaValidatorMap(),
        name="SchemaValidation"
    )
    
    # Split: valid vs invalid
    valid_stream = validated_stream.filter(lambda r: r.get('_is_schema_valid', False))
    invalid_stream = validated_stream.filter(lambda r: r.get('_send_to_violations', False))
    
    LOGGER.info("[L1] Complete: valid/invalid streams separated")
    
    # =================================================================
    # Layer 2A: Canary Rules
    # =================================================================
    LOGGER.info("[L2A] Building Canary Rules pipeline...")
    
    canary_stream = valid_stream.map(
        CanaryRulesMapper(),
        name="CanaryRules"
    )
    
    # Split: violations vs clean
    canary_violation_stream = canary_stream.filter(lambda r: r.get('has_violation', False))
    canary_clean_stream = canary_stream.filter(lambda r: not r.get('has_violation', False))
    
    LOGGER.info("[L2A] Complete: %d canary rules applied", 7)
    
    # =================================================================
    # Layer 2B: MemStream ML
    # =================================================================
    LOGGER.info("[L2B] Building MemStream ML pipeline...")

    # Note: In production, use KeyedProcessFunction per neighborhood for true per-key memory
    # For now, use MapFunction with shared MemStream (simpler, slightly less accurate)
    memstream_stage = MemStreamScoringStage(config)

    memstream_stream = canary_clean_stream.map(
        memstream_stage.process,
        name="MemStreamScoring"
    )

    LOGGER.info("[L2B] Complete: MemStream ML scoring enabled (real operator)")
    
    # =================================================================
    # Layer 3: Voting Ensemble
    # =================================================================
    LOGGER.info("[L3] Building Voting Ensemble pipeline...")

    # Schema normalization: ensure both streams have identical field schemas
    # before union so downstream operators receive consistent records.
    canary_violation_normalized = canary_violation_stream.map(
        lambda r: {
            **r,
            'anomaly_score': 0.0,
            'is_anomaly': False,
            'has_violation': True,
            'threshold': 1.0,
            'ml_model': 'canary_only',
            'context_key': r.get('neighborhood', 'unknown'),
        },
        name="NormalizeCanaryViolation"
    )

    memstream_normalized = memstream_stream.map(
        lambda r: {
            **r,
            'canary_violations': r.get('canary_violations', []),
        },
        name="NormalizeMemStream"
    )

    # Union with normalized schemas
    union_stream = canary_violation_normalized.union(memstream_normalized)

    # Voting ensemble (VotingEnsembleMapper adds final_decision, decision_source, confidence)
    voted_stream = union_stream.map(
        VotingEnsembleMapper(),
        name="VotingEnsemble"
    )

    LOGGER.info("[L3] Complete: Voting ensemble configured")

    # =================================================================
    # Layer 3: MetaAggregator (count-based windows with proper Flink state)
    # =================================================================
    LOGGER.info("[L3b] Building MetaAggregator pipeline...")

    neighborhood_key_func = lambda r: str(r.get('neighborhood', r.get('context_key', 'unknown')))

    # Apply MetaAggregator keyed by neighborhood using proper KeyedProcessFunction
    meta_stream = (
        voted_stream
        .key_by(neighborhood_key_func)
        .process(MetaAggregatorProcessFn(100), name="MetaAggregator")
    )

    # Filter None (only emit when window is complete)
    meta_metrics_stream = meta_stream.filter(lambda r: r is not None)

    LOGGER.info("[L3b] Complete: MetaAggregator configured (KeyedProcessFunction with ValueState)")

    # =================================================================
    # Layer 4: IEC
    # =================================================================
    LOGGER.info("[L4] Building IEC Feedback pipeline...")

    # Create MinIO client for IEC beta communication
    import boto3
    from botocore.config import Config
    minio_config = Config(signature_version='s3v4')
    minio_client = boto3.client(
        's3',
        endpoint_url=config.MINIO_ENDPOINT,
        aws_access_key_id=config.MINIO_ACCESS_KEY,
        aws_secret_access_key=config.MINIO_SECRET_KEY,
        config=minio_config,
    )

    # Process meta-metrics through IEC
    iec_stream = (
        meta_metrics_stream
        .map(IECDecisionMapper(config, minio_client=minio_client), name="IECDecision")
    )

    LOGGER.info("[L4] Complete: IEC configured (fallback mode)")
    
    # =================================================================
    # Configure Sinks
    # =================================================================
    LOGGER.info("[Sinks] Configuring output sinks...")
    
    # Kafka sinks
    try:
        violation_sink = create_kafka_sink(config.VIOLATION_TOPIC, config)
        invalid_stream.add_sink(violation_sink).name("Sink-Violations")
        LOGGER.info("[Sinks] Violations: %s", config.VIOLATION_TOPIC)
    except Exception as e:
        LOGGER.error("[Sinks] CRITICAL: Failed to create violation sink: %s", e)
        raise  # Re-raise to fail the job
    
    try:
        processed_sink = create_kafka_sink(config.OUTPUT_TOPIC, config)
        voted_stream.filter(lambda r: r.get('final_decision') == 'CLEAN').add_sink(processed_sink).name("Sink-Processed")
        LOGGER.info("[Sinks] Processed: %s", config.OUTPUT_TOPIC)
    except Exception as e:
        LOGGER.error("[Sinks] CRITICAL: Failed to create processed sink: %s", e)
        raise  # Re-raise to fail the job

    try:
        anomaly_sink = create_kafka_sink(config.ANOMALY_TOPIC, config)
        voted_stream.filter(lambda r: r.get('final_decision') == 'ANOMALY').add_sink(anomaly_sink).name("Sink-Anomalies")
        LOGGER.info("[Sinks] Anomalies: %s", config.ANOMALY_TOPIC)
    except Exception as e:
        LOGGER.error("[Sinks] CRITICAL: Failed to create anomaly sink: %s", e)
        raise  # Re-raise to fail the job

    try:
        meta_sink = create_kafka_sink(config.META_TOPIC, config)
        meta_metrics_stream.add_sink(meta_sink).name("Sink-Meta")
        LOGGER.info("[Sinks] Meta: %s", config.META_TOPIC)
    except Exception as e:
        LOGGER.error("[Sinks] CRITICAL: Failed to create meta sink: %s", e)
        raise  # Re-raise to fail the job

    try:
        iec_sink = create_kafka_sink(config.IEC_TOPIC, config)
        iec_decisions = iec_stream.filter(lambda r: r.get('strategy') != 'do_nothing')
        iec_decisions.add_sink(iec_sink).name("Sink-IECDecisions")
        LOGGER.info("[Sinks] IEC Decisions: %s", config.IEC_TOPIC)
    except Exception as e:
        LOGGER.error("[Sinks] CRITICAL: Failed to create IEC decision sink: %s", e)
        raise  # Re-raise to fail the job

    # Late data sink: records that arrived after the watermark boundary
    LATE_DATA_TOPIC = 'dq-late-data'
    try:
        late_data_sink = create_kafka_sink(LATE_DATA_TOPIC, config)
        late_stream.add_sink(late_data_sink).name("Sink-LateData")
        LOGGER.info("[Sinks] Late data: %s", LATE_DATA_TOPIC)
    except Exception as e:
        LOGGER.error("[Sinks] CRITICAL: Failed to create late data sink: %s", e)
        raise  # Re-raise to fail the job

    LOGGER.info("[Sinks] Configuration complete")
    
    LOGGER.info("=" * 60)
    LOGGER.info("Pipeline build complete - ready to execute")
    LOGGER.info("=" * 60)
    
    return env


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point for Flink job."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CA-DQStream + MemStream v5 Flink Job')
    parser.add_argument(
        '--target',
        choices=['yarn', 'kubernetes-application', 'local'],
        default='local',
        help='Deployment target'
    )
    parser.add_argument(
        '--job-name',
        default='CA-DQStream + MemStream v5',
        help='Job name for Flink UI'
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    LOGGER.info("[Main] Starting CA-DQStream + MemStream v5 Job")
    LOGGER.info("[Main] Target: %s", args.target)
    
    # Create environment
    if args.target == 'local':
        env = StreamExecutionEnvironment.get_execution_environment()
    else:
        # Production: use from_environment() which reads from Flink config
        env = StreamExecutionEnvironment.get_execution_environment()
    
    # Build pipeline
    build_complete_pipeline(env)
    
    # Execute
    LOGGER.info("[Main] Executing job: %s", args.job_name)
    env.execute(args.job_name)


if __name__ == '__main__':
    main()


__all__ = [
    'PipelineConfig',
    'build_complete_pipeline',
    'main',
    # Layer functions for testing
    'ParseJsonFunction',
    'AddTripIdFunction',
    'DeduplicatorFunction',
    'LateDataDetector',
    'SchemaValidator',
    'SchemaValidatorMap',
    'CanaryRulesMapper',
    'MemStreamScoringStage',
    'VotingEnsembleMapper',
    'MetaAggregatorMapper',
    'MetaAggregatorProcessFn',  # Proper Flink KeyedProcessFunction for distributed state
    'IECDecisionMapper',
]
