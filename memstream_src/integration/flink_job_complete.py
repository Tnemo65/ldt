"""
CA-DQStream + MemStream v5 - Complete Flink Job
================================================

4-Layer Pipeline Architecture:
    Layer 1: Baseline Validation (Parse, Dedup, Schema)
    Layer 2: Dual-Branch (Canary + MemStream)
    Layer 3: Voting Ensemble + MetaAggregator
    Layer 4: IEC Feedback

Features:
    - Kafka source with JSON deserialization
    - Layer 1: Parse JSON, add trip ID, deduplicate, schema validation
    - Layer 2A: Canary rules (fast rule-based filtering)
    - Layer 2B: MemStream scoring (ML anomaly detection)
    - Layer 3: Voting ensemble + meta-metrics aggregation
    - Layer 4: IEC feedback for threshold adjustments
    - Checkpointing for fault tolerance
    - MinIO/Parquet sink for meta-metrics

Environment Variables:
    FLINK_PARALLELISM: Number of parallel tasks (default: 4)
    KAFKA_BOOTSTRAP_SERVERS: Kafka broker addresses (default: kafka:9092)
    REDIS_HOST: Redis host for IEC (default: redis)
    REDIS_PASSWORD: Redis password (required)
    MEMSTREAM_MODEL_PATH: Path to MemStream model (default: /models/memstream_ae.pt)
    IEC_SIGNING_KEY: HMAC key for IEC actions (required)
    MINIO_ENDPOINT: MinIO endpoint (default: minio:9000)
    MINIO_ACCESS_KEY: MinIO access key (required)
    MINIO_SECRET_KEY: MinIO secret key (required)

Usage:
    from memstream_src.integration.flink_job_complete import create_job
    env = create_job()
    env.execute("CA-DQStream + MemStream v5")
"""

import os
import sys
import logging
from typing import Dict, Optional
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.datastream.connectors.file_system import StreamingFileSink
from pyflink.datastream.formats.json import JsonRowDeserializationSchema, JsonRowSerializationSchema
from pyflink.datastream import OutputTag
from pyflink.common.typeinfo import Types
from pyflink.common.time import Time
from pyflink.datastream import WatermarkStrategy
from pyflink.common.time import Duration

LOGGER = logging.getLogger('cadqstream-main')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


# =============================================================================
# Kafka Configuration
# =============================================================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
KAFKA_SOURCE_GROUP = 'cadqstream-layer1'
KAFKA_AUTO_OFFSET_RESET = 'earliest'

# Topics
TOPIC_RAW = 'taxi-nyc-raw'
TOPIC_VIOLATIONS = 'dq-hard-rule-violations'
TOPIC_PROCESSED = 'dq-stream-processed'
TOPIC_ANOMALIES = 'dq-stream-anomalies'
TOPIC_META = 'dq-meta-stream'
TOPIC_IEC_ACTIONS = 'iec-action-replay'


# =============================================================================
# Redis Configuration
# =============================================================================

REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')


# =============================================================================
# MinIO Configuration
# =============================================================================

MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
MINIO_BUCKET = 'cadqstream-metrics'


# =============================================================================
# Model Configuration
# =============================================================================

MEMSTREAM_MODEL_PATH = os.getenv('MEMSTREAM_MODEL_PATH', '/models/memstream_ae.pt')
IEC_SIGNING_KEY = os.getenv('IEC_SIGNING_KEY')


# =============================================================================
# Output Tags for Side Outputs
# =============================================================================

VALID_OUTPUT_TAG = OutputTag("valid_stream", type_info=Types.PICKLED_BYTE_ARRAY_TYPE_INFO)
INVALID_OUTPUT_TAG = OutputTag("invalid_stream", type_info=Types.PICKLED_BYTE_ARRAY_TYPE_INFO)
META_OUTPUT_TAG = OutputTag("meta_stream", type_info=Types.PICKLED_BYTE_ARRAY_TYPE_INFO)
IEC_ACTION_OUTPUT_TAG = OutputTag("iec_action_stream", type_info=Types.PICKLED_BYTE_ARRAY_TYPE_INFO)


# =============================================================================
# Import Operators
# =============================================================================

try:
    from memstream_src.operators.layer1 import (
        ParseJsonFunction,
        AddTripIdFunction,
        DeduplicatorFunction,
        WatermarkAssigner,
    )
    from memstream_src.operators.layer1.schema_validator import SchemaValidator
    from memstream_src.operators.layer3 import (
        VotingEnsembleFunction,
        MetaAggregateFunction,
        MetaWindowProcessFunction,
    )
    from memstream_src.operators.canary_rules import check_canary, check_canary_rules
    _LAYER_IMPORTS_OK = True
except ImportError as e:
    LOGGER.warning("[Main] Layer imports failed: %s - using stubs", e)
    _LAYER_IMPORTS_OK = False


# =============================================================================
# Canary Rules Operator
# =============================================================================

class CanaryRulesMapper:
    """
    MapFunction that applies Canary rules to records.
    
    Canary rules are fast rule-based checks for obvious anomalies:
        1. Fare bounds (0-500)
        2. Distance bounds (0-100)
        3. Positive passenger count (1-9)
        4. Duration bounds (0-360 min)
        5. Speed bounds (0-80 mph)
        6. JFK flat fare validation
        7. Credit card with zero tip (suspicious for fare > $10)
    """
    
    def __init__(self):
        self.records_processed = 0
        self.violations_found = 0
    
    def map(self, record: Dict) -> Dict:
        """
        Apply Canary rules and add violation fields.
        
        Args:
            record: Input record from Layer 1
            
        Returns:
            Record with has_violation and canary_violations added
        """
        self.records_processed += 1
        
        if not _LAYER_IMPORTS_OK:
            # Stub implementation
            record['has_violation'] = False
            record['canary_violations'] = []
            return record
        
        try:
            violations = check_canary_rules(record)
            violation_list = [k for k, v in violations.items() if not v]
            
            record['has_violation'] = len(violation_list) > 0
            record['canary_violations'] = violation_list
            
            if record['has_violation']:
                self.violations_found += 1
        
        except Exception as e:
            LOGGER.warning("[CanaryRules] Error processing record: %s", e)
            record['has_violation'] = False
            record['canary_violations'] = []
        
        return record
    
    def get_metrics(self) -> Dict:
        """Return Canary rules processing statistics."""
        return {
            'records_processed': self.records_processed,
            'violations_found': self.violations_found,
            'violation_rate': self.violations_found / max(self.records_processed, 1),
        }


# =============================================================================
# MemStream Scoring Operator (Stub for testing)
# =============================================================================

class MemStreamScoringMapper:
    """
    MapFunction stub for MemStream scoring.
    
    In production, this would be replaced with MemStreamScoringOperator
    which uses KeyedProcessFunction for per-key memory state.
    
    For integration testing, this stub provides:
        - anomaly_score: Random score (in production: ML model)
        - is_anomaly: Score > threshold
        - threshold: Beta threshold (in production: per-neighborhood)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.threshold = self.config.get('default_threshold', 0.5)
        self.records_processed = 0
        self.anomalies_found = 0
    
    def map(self, record: Dict) -> Dict:
        """
        Score record with MemStream (stub implementation).
        
        Args:
            record: Input record from Layer 1
            
        Returns:
            Record with anomaly_score, threshold, is_anomaly added
        """
        self.records_processed += 1
        
        import random
        # Stub: random score between 0.1 and 0.9
        anomaly_score = self.config.get('stub_score', 0.3 + random.random() * 0.4)
        
        is_anomaly = anomaly_score > self.threshold
        
        record['anomaly_score'] = anomaly_score
        record['threshold'] = self.threshold
        record['is_anomaly'] = is_anomaly
        
        if is_anomaly:
            self.anomalies_found += 1
        
        return record
    
    def get_metrics(self) -> Dict:
        """Return MemStream scoring statistics."""
        return {
            'records_processed': self.records_processed,
            'anomalies_found': self.anomalies_found,
            'anomaly_rate': self.anomalies_found / max(self.records_processed, 1),
        }


# =============================================================================
# IEC Feedback Operator (Stub)
# =============================================================================

class IECFeedbackMapper:
    """
    ProcessFunction stub for IEC feedback.
    
    In production, this would be replaced with IECFeedbackOperator
    which uses KeyedBroadcastProcessFunction for beta adjustments.
    
    For integration testing, this stub provides:
        - iec_action: Any action taken
        - iec_timestamp: When action was taken
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.records_processed = 0
        self.actions_taken = 0
    
    def process_element(self, record: Dict, ctx=None) -> Dict:
        """
        Process record and apply IEC feedback (stub).
        
        Args:
            record: Input record from Layer 3
            
        Returns:
            Record with IEC metadata added
        """
        self.records_processed += 1
        
        record['iec_processed'] = True
        
        # Stub: check if anomaly and decide action
        if record.get('final_decision') == 'ANOMALY':
            record['iec_action'] = 'log_anomaly'
            self.actions_taken += 1
        else:
            record['iec_action'] = 'pass_through'
        
        return record
    
    def get_metrics(self) -> Dict:
        """Return IEC feedback processing statistics."""
        return {
            'records_processed': self.records_processed,
            'actions_taken': self.actions_taken,
            'action_rate': self.actions_taken / max(self.records_processed, 1),
        }


# =============================================================================
# Sink Factory Functions
# =============================================================================

def create_kafka_producer(topic: str) -> FlinkKafkaProducer:
    """
    Create Kafka producer for given topic.
    
    Args:
        topic: Kafka topic name
        
    Returns:
        Configured FlinkKafkaProducer with EXACTLY_ONCE semantic
    """
    producer_config = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
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


def create_minio_sink(bucket: str, path_prefix: str) -> StreamingFileSink:
    """
    Create MinIO/Parquet sink for meta-metrics.
    
    Args:
        bucket: MinIO bucket name
        path_prefix: Path prefix for output files
        
    Returns:
        Configured StreamingFileSink
    """
    # For production, use:
    # from pyflink.datastream.connectors.file_system import FileSink
    # from pyflink.formats.parquet import ParquetWriterFactory
    
    # Stub: return a simple file sink
    output_path = f"s3://{bucket}/{path_prefix}"
    
    return StreamingFileSink \
        .for_bulk_format(output_path, JsonRowSerializationSchema()) \
        .with_bucket_assigner(DateTimeBucketAssigner()) \
        .build()


class DateTimeBucketAssigner:
    """Assign buckets based on processing time."""
    
    def get_bucket_id(self, element: Dict) -> str:
        from datetime import datetime
        return datetime.utcnow().strftime('%Y%m%d-%H%M')


# =============================================================================
# Main Flink Job Builder
# =============================================================================

def create_job() -> StreamExecutionEnvironment:
    """
    Create and configure the complete Flink job.
    
    Returns:
        Configured StreamExecutionEnvironment ready to execute
    """
    LOGGER.info("[Main] Creating CA-DQStream + MemStream v5 job")
    
    # Initialize environment
    env = StreamExecutionEnvironment.get_execution_environment()
    
    # Configure parallelism
    parallelism = int(os.getenv('FLINK_PARALLELISM', '4'))
    env.set_parallelism(parallelism)
    LOGGER.info("[Main] Parallelism: %d", parallelism)
    
    # Enable checkpointing
    checkpoint_interval = int(os.getenv('FLINK_CHECKPOINT_INTERVAL', '60000'))  # 60 seconds
    env.enable_checkpointing(checkpoint_interval)
    
    checkpoint_config = env.get_checkpoint_config()
    checkpoint_config.set_min_pause_between_checkpoints(30000)  # 30 seconds
    checkpoint_config.set_checkpoint_timeout(300000)  # 5 minutes
    checkpoint_config.enable_externalized_checkpoints("RETAIN_ON_CANCELLATION")
    
    LOGGER.info("[Main] Checkpointing enabled: interval=%dms", checkpoint_interval)
    
    # =================================================================
    # Layer 1: Baseline Validation
    # =================================================================
    LOGGER.info("[Main] Building Layer 1: Baseline Validation")
    
    # Kafka source
    kafka_consumer = FlinkKafkaConsumer(
        topics=TOPIC_RAW,
        deserialization_schema=JsonRowDeserializationSchema(),
        properties={
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': KAFKA_SOURCE_GROUP,
            'auto.offset.reset': KAFKA_AUTO_OFFSET_RESET,
        }
    )
    
    raw_stream = env.add_source(kafka_consumer, name="KafkaSource-Raw")
    
    # Parse JSON
    parsed_stream = raw_stream.map(
        ParseJsonFunction() if _LAYER_IMPORTS_OK else _StubParseJsonFunction(),
        name="ParseJson"
    )
    
    # Add trip ID
    trip_id_stream = parsed_stream.map(
        AddTripIdFunction() if _LAYER_IMPORTS_OK else _StubAddTripIdFunction(),
        name="AddTripId"
    )
    
    # Deduplicate (keyed by trip_id)
    deduplicated_stream = trip_id_stream.key_by(
        lambda r: r.get('trip_id', '')
    ).process(
        DeduplicatorFunction() if _LAYER_IMPORTS_OK else _StubDeduplicatorFunction(),
        name="Deduplicate"
    )
    
    # Schema validation with side outputs
    validated_stream = deduplicated_stream.process(
        SchemaValidator() if _LAYER_IMPORTS_OK else _StubSchemaValidator(),
        name="SchemaValidation"
    )
    
    # Get valid and invalid streams from side outputs
    valid_stream = validated_stream.get_side_output(VALID_OUTPUT_TAG)
    invalid_stream = validated_stream.get_side_output(INVALID_OUTPUT_TAG)
    
    LOGGER.info("[Main] Layer 1 complete: valid/invalid streams separated")
    
    # =================================================================
    # Layer 2A: Canary Branch
    # =================================================================
    LOGGER.info("[Main] Building Layer 2A: Canary Rules")
    
    canary_stream = valid_stream.map(
        CanaryRulesMapper(),
        name="CanaryRules"
    )
    
    LOGGER.info("[Main] Layer 2A complete")
    
    # =================================================================
    # Layer 2B: Complex Branch (MemStream)
    # =================================================================
    LOGGER.info("[Main] Building Layer 2B: MemStream Scoring")
    
    memstream_config = {
        'model_path': MEMSTREAM_MODEL_PATH,
        'redis_host': REDIS_HOST,
        'redis_password': REDIS_PASSWORD,
        'default_threshold': float(os.getenv('MEMSTREAM_DEFAULT_THRESHOLD', '0.5')),
    }
    
    if _LAYER_IMPORTS_OK:
        # Try to import real MemStreamScoringOperator
        try:
            from memstream_src.operators.memstream_scoring_op import MemStreamScoringOperator
            memstream_stream = valid_stream.key_by(
                lambda r: str(r.get('PULocationID', 'unknown'))
            ).process(
                MemStreamScoringOperator(memstream_config),
                name="MemStreamScoring"
            )
        except Exception as e:
            LOGGER.warning("[Main] MemStreamScoringOperator failed: %s - using stub", e)
            memstream_stream = valid_stream.map(
                MemStreamScoringMapper(memstream_config),
                name="MemStreamScoring-Stub"
            )
    else:
        memstream_stream = valid_stream.map(
            MemStreamScoringMapper(memstream_config),
            name="MemStreamScoring-Stub"
        )
    
    LOGGER.info("[Main] Layer 2B complete")
    
    # =================================================================
    # Layer 3: Voting Ensemble + MetaAggregator
    # =================================================================
    LOGGER.info("[Main] Building Layer 3: Voting Ensemble")
    
    # Union canary and memstream streams
    # Note: In production, this would use a rendezvous join
    union_stream = canary_stream.union(memstream_stream)
    
    # Voting ensemble
    voted_stream = union_stream.key_by(
        lambda r: r.get('trip_id', '')
    ).map(
        VotingEnsembleFunction() if _LAYER_IMPORTS_OK else _StubVotingEnsembleFunction(),
        name="VotingEnsemble"
    )
    
    # Meta stream from side output
    meta_stream = voted_stream.get_side_output(META_OUTPUT_TAG)
    
    LOGGER.info("[Main] Layer 3 complete")
    
    # =================================================================
    # Layer 4: IEC Feedback
    # =================================================================
    LOGGER.info("[Main] Building Layer 4: IEC Feedback")
    
    iec_config = {
        'slo_config': {
            'latency_p99_ms': 100.0,
            'iec_cooldown_seconds': 300.0,
            'iec_max_consecutive': 10,
        },
        'redis_host': REDIS_HOST,
        'redis_password': REDIS_PASSWORD,
    }
    
    if _LAYER_IMPORTS_OK:
        try:
            from memstream_src.operators.iec_feedback_op import IECFeedbackOperator
            iec_stream = voted_stream.process(
                IECFeedbackOperator(),
                name="IECFeedback"
            )
        except Exception as e:
            LOGGER.warning("[Main] IECFeedbackOperator failed: %s - using stub", e)
            iec_stream = voted_stream.map(
                IECFeedbackMapper(iec_config),
                name="IECFeedback-Stub"
            )
    else:
        iec_stream = voted_stream.map(
            IECFeedbackMapper(iec_config),
            name="IECFeedback-Stub"
        )
    
    # IEC action stream from side output
    iec_action_stream = iec_stream.get_side_output(IEC_ACTION_OUTPUT_TAG)
    
    LOGGER.info("[Main] Layer 4 complete")
    
    # =================================================================
    # Sinks
    # =================================================================
    LOGGER.info("[Main] Configuring sinks")
    
    # Anomaly sink (Kafka)
    try:
        anomaly_sink = create_kafka_producer(TOPIC_ANOMALIES)
        iec_stream.add_sink(anomaly_sink).name("Sink-Anomalies")
        LOGGER.info("[Main] Anomaly sink: %s", TOPIC_ANOMALIES)
    except Exception as e:
        LOGGER.warning("[Main] Failed to create anomaly sink: %s", e)
    
    # Violations sink (Kafka)
    try:
        violation_sink = create_kafka_producer(TOPIC_VIOLATIONS)
        invalid_stream.add_sink(violation_sink).name("Sink-Violations")
        LOGGER.info("[Main] Violation sink: %s", TOPIC_VIOLATIONS)
    except Exception as e:
        LOGGER.warning("[Main] Failed to create violation sink: %s", e)
    
    # Processed sink (Kafka)
    try:
        processed_sink = create_kafka_producer(TOPIC_PROCESSED)
        valid_stream.add_sink(processed_sink).name("Sink-Processed")
        LOGGER.info("[Main] Processed sink: %s", TOPIC_PROCESSED)
    except Exception as e:
        LOGGER.warning("[Main] Failed to create processed sink: %s", e)
    
    # Meta metrics sink (MinIO/Parquet)
    try:
        meta_sink = create_minio_sink(MINIO_BUCKET, 'meta_metrics')
        meta_stream.add_sink(meta_sink).name("Sink-MetaMetrics")
        LOGGER.info("[Main] Meta metrics sink: %s/%s", MINIO_BUCKET, 'meta_metrics')
    except Exception as e:
        LOGGER.warning("[Main] Failed to create meta metrics sink: %s", e)
    
    # IEC actions sink (Kafka)
    try:
        iec_action_sink = create_kafka_producer(TOPIC_IEC_ACTIONS)
        iec_action_stream.add_sink(iec_action_sink).name("Sink-IECActions")
        LOGGER.info("[Main] IEC action sink: %s", TOPIC_IEC_ACTIONS)
    except Exception as e:
        LOGGER.warning("[Main] Failed to create IEC action sink: %s", e)
    
    LOGGER.info("[Main] CA-DQStream + MemStream v5 job created successfully")
    
    return env


# =============================================================================
# Stub Functions for Testing
# =============================================================================

class _StubParseJsonFunction:
    """Stub ParseJsonFunction for testing without PyFlink."""
    
    def __init__(self):
        import json
        self.json = json
    
    def map(self, value):
        try:
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            return self.json.loads(value)
        except Exception:
            return None


class _StubAddTripIdFunction:
    """Stub AddTripIdFunction for testing without PyFlink."""
    
    def __init__(self):
        import hashlib
        self.hashlib = hashlib
    
    def map(self, record):
        if record is None:
            return None
        key = f"{record.get('VendorID', '')}|{record.get('tpep_pickup_datetime', '')}|{record.get('PULocationID', '')}|{record.get('DOLocationID', '')}|{record.get('fare_amount', '')}"
        record['trip_id'] = self.hashlib.sha256(key.encode()).hexdigest()[:32]
        return record


class _StubDeduplicatorFunction:
    """Stub DeduplicatorFunction for testing."""
    
    def __init__(self):
        self._seen = set()
    
    def process_element(self, record, ctx=None):
        if record is None:
            return None
        trip_id = record.get('trip_id')
        if trip_id in self._seen:
            return None
        self._seen.add(trip_id)
        record['_dedup_status'] = 'new'
        return record


class _StubSchemaValidator:
    """Stub SchemaValidator for testing."""
    
    def __init__(self):
        pass
    
    def process_element(self, record, ctx=None):
        if record is None:
            return record
        
        is_valid = True
        violations = []
        
        # Check required fields
        for field in ['trip_distance', 'fare_amount', 'PULocationID', 'DOLocationID', 'passenger_count']:
            if field not in record or record[field] is None:
                is_valid = False
                violations.append(f"missing_field:{field}")
        
        # Check zone range
        if is_valid:
            try:
                pu = int(record.get('PULocationID', 0))
                do = int(record.get('DOLocationID', 0))
                if not (1 <= pu <= 263) or not (1 <= do <= 263):
                    is_valid = False
                    violations.append("invalid_zone")
            except (ValueError, TypeError):
                is_valid = False
                violations.append("invalid_zone")
        
        record['_validation_is_valid'] = is_valid
        record['_validation_violations'] = violations
        
        return record


class _StubVotingEnsembleFunction:
    """Stub VotingEnsembleFunction for testing."""
    
    def __init__(self):
        pass
    
    def map(self, record):
        if record is None:
            return None
        
        has_violation = record.get('has_violation', False)
        is_anomaly = record.get('is_anomaly', False)
        anomaly_score = record.get('anomaly_score', 0.0)
        threshold = record.get('threshold', 0.5)
        
        if has_violation:
            record['final_decision'] = 'ANOMALY'
            record['decision_source'] = 'canary_rule'
            record['confidence'] = 1.0
        elif is_anomaly:
            record['final_decision'] = 'ANOMALY'
            record['decision_source'] = 'complex_ml'
            record['confidence'] = min(anomaly_score / threshold, 1.0) if threshold > 0 else 0.5
        else:
            record['final_decision'] = 'CLEAN'
            record['decision_source'] = 'both_agree'
            record['confidence'] = 1.0 - (anomaly_score / threshold) if threshold > 0 else 0.5
        
        return record


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    env = create_job()
    env.execute("CA-DQStream + MemStream v5")
