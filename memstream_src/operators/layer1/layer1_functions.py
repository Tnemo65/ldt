"""
Layer 1 Combined Functions - Baseline Validation.

This module re-exports all Layer 1 operators for convenience.
Also provides factory functions for creating the complete Layer 1 pipeline.

Usage:
    from .layer1.layer1_functions import create_layer1_pipeline
    
    # Or import individual functions
    from .layer1 import ParseJsonFunction, AddTripIdFunction, etc.
"""

from typing import Dict, Optional, Tuple
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream import OutputTag
from pyflink.common.typeinfo import Types

# Re-export all Layer 1 functions
from .parse_json import ParseJsonFunction
from .watermark_assigner import WatermarkAssigner, create_watermark_strategy
from .add_trip_id import AddTripIdFunction, hash_bytes
from .deduplicator import DeduplicatorFunction, create_ttl_config_7_days
from .schema_validator import SchemaValidator, validate_record

# For type hints
try:
    from pyflink.datastream import ProcessFunction, KeyedProcessFunction
except ImportError:
    ProcessFunction = None
    KeyedProcessFunction = None


# =============================================================================
# Pipeline Factory Functions
# =============================================================================

def create_layer1_pipeline(
    env: StreamExecutionEnvironment,
    kafka_bootstrap_servers: str = "localhost:9092",
    kafka_topic: str = "nyc-taxi-trips",
    kafka_group_id: str = "layer1-consumer",
    consumer_mode: str = "EARLIEST",
) -> Tuple:
    """
    Create complete Layer 1 pipeline.
    
    Pipeline Flow:
        Kafka Source
            → ParseJsonFunction
            → WatermarkAssigner (event time)
            → AddTripIdFunction
            → DeduplicatorFunction (keyed by trip_id)
            → SchemaValidator (split valid/violation streams)
    
    Args:
        env: Flink StreamExecutionEnvironment
        kafka_bootstrap_servers: Kafka bootstrap servers
        kafka_topic: Input Kafka topic
        kafka_group_id: Consumer group ID
        consumer_mode: EARLIEST or LATEST
        
    Returns:
        Tuple of (valid_stream, violation_stream, stats_stream)
    """
    from pyflink.datastream.connectors.kafka import KafkaSource
    from pyflink.datastream.connectors.kafka import KafkaOffsetsInitializer
    from pyflink.datastream import WatermarkStrategy
    
    # Create Kafka source
    kafka_offsets = (
        KafkaOffsetsInitializer.earliest_offsets()
        if consumer_mode == "EARLIEST"
        else KafkaOffsetsInitializer.latest_offsets()
    )
    
    kafka_source = (
        KafkaSource.builder()
        .set_bootstrap_servers(kafka_bootstrap_servers)
        .set_topics(kafka_topic)
        .set_group_id(kafka_group_id)
        .set_starting_offsets(kafka_offsets)
        .build()
    )
    
    raw_stream = env.from_source(
        source=kafka_source,
        watermark_strategy=create_watermark_strategy(),
        source_name="Kafka_NYC_Taxi_Source"
    )
    
    # Layer 1: Parse JSON
    parsed_stream = raw_stream.map(ParseJsonFunction()).filter(
        lambda r: r is not None
    )
    
    # Layer 1: Add trip ID
    tripid_stream = parsed_stream.map(AddTripIdFunction())
    
    # Layer 1: Deduplicate (keyed by trip_id)
    deduplicated_stream = tripid_stream.key_by(
        lambda r: r.get('trip_id', 'unknown')
    ).process(DeduplicatorFunction()).filter(
        lambda r: r is not None
    )
    
    # Layer 1: Schema validation (split to valid/violation streams)
    valid_tag = OutputTag("valid_stream")()
    violation_tag = OutputTag("violation_stream")()
    
    validated_stream = deduplicated_stream.process(SchemaValidator())
    
    valid_stream = validated_stream.get_side_output(valid_tag)
    violation_stream = validated_stream.get_side_output(violation_tag)
    
    # Stats stream (for monitoring)
    stats_stream = validated_stream.map(
        lambda r: {
            'timestamp': r.get('_validation_timestamp', 0),
            'is_valid': r.get('_validation_is_valid', False),
            'trip_id': r.get('trip_id', 'unknown'),
        }
    )
    
    return valid_stream, violation_stream, stats_stream


def create_simple_layer1_stream(
    raw_stream,
) -> Tuple:
    """
    Create Layer 1 pipeline from existing raw stream.
    
    Use this when you already have a DataStream with raw bytes.
    
    Args:
        raw_stream: DataStream[bytes] with raw Kafka messages
        
    Returns:
        Tuple of (valid_stream, violation_stream)
    """
    from pyflink.datastream import WatermarkStrategy
    from pyflink.common.time import Duration
    
    # Parse JSON
    parsed_stream = (
        raw_stream
        .assign_timestamps_and_watermarks(
            create_watermark_strategy()
        )
        .map(ParseJsonFunction())
        .filter(lambda r: r is not None)
    )
    
    # Add trip ID
    tripid_stream = parsed_stream.map(AddTripIdFunction())
    
    # Deduplicate
    deduplicated_stream = (
        tripid_stream
        .key_by(lambda r: r.get('trip_id', 'unknown'))
        .process(DeduplicatorFunction())
        .filter(lambda r: r is not None)
    )
    
    # Validate schema
    valid_tag = OutputTag("valid_stream")()
    violation_tag = OutputTag("violation_stream")()
    
    validated_stream = deduplicated_stream.process(SchemaValidator())
    
    valid_stream = validated_stream.get_side_output(valid_tag)
    violation_stream = validated_stream.get_side_output(violation_tag)
    
    return valid_stream, violation_stream


# =============================================================================
# Metrics Helpers
# =============================================================================

class Layer1Metrics:
    """
    Aggregated metrics for all Layer 1 operators.
    
    Usage:
        metrics = Layer1Metrics()
        metrics.record_parse(result)
        metrics.record_dedup(is_duplicate)
        metrics.record_validation(is_valid)
        stats = metrics.get_summary()
    """
    
    def __init__(self):
        self.parse_total = 0
        self.parse_errors = 0
        
        self.dedup_total = 0
        self.dedup_duplicates = 0
        
        self.validation_total = 0
        self.validation_valid = 0
        self.validation_invalid = 0
    
    def record_parse(self, success: bool):
        """Record JSON parsing result."""
        self.parse_total += 1
        if not success:
            self.parse_errors += 1
    
    def record_dedup(self, is_duplicate: bool):
        """Record deduplication result."""
        self.dedup_total += 1
        if is_duplicate:
            self.dedup_duplicates += 1
    
    def record_validation(self, is_valid: bool):
        """Record validation result."""
        self.validation_total += 1
        if is_valid:
            self.validation_valid += 1
        else:
            self.validation_invalid += 1
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'parse': {
                'total': self.parse_total,
                'errors': self.parse_errors,
                'error_rate': self.parse_errors / max(self.parse_total, 1),
            },
            'dedup': {
                'total': self.dedup_total,
                'duplicates': self.dedup_duplicates,
                'duplicate_rate': self.dedup_duplicates / max(self.dedup_total, 1),
            },
            'validation': {
                'total': self.validation_total,
                'valid': self.validation_valid,
                'invalid': self.validation_invalid,
                'valid_rate': self.validation_valid / max(self.validation_total, 1),
                'invalid_rate': self.validation_invalid / max(self.validation_total, 1),
            },
        }


__all__ = [
    # Individual functions
    'ParseJsonFunction',
    'WatermarkAssigner',
    'create_watermark_strategy',
    'AddTripIdFunction',
    'hash_bytes',
    'DeduplicatorFunction',
    'create_ttl_config_7_days',
    'SchemaValidator',
    'validate_record',
    # Pipeline factories
    'create_layer1_pipeline',
    'create_simple_layer1_stream',
    # Metrics
    'Layer1Metrics',
]
