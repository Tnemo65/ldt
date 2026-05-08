"""
CA-DQStream Complete Pipeline - Laptop Version (Low Memory)
Optimized for 16GB RAM laptops (~8GB available)

Changes from full version:
- Parallelism: 4 → 1 (single thread)
- Checkpointing: 45s → 120s (reduce overhead)
- No external sinks (print only for testing)
"""

from pyflink.datastream import StreamExecutionEnvironment, CheckpointingMode
from pyflink.datastream.connectors.kafka import FlinkKafkaConsumer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.window import TumblingEventTimeWindows
from pyflink.common.time import Time
import os
import json

# Import operators (same as full version)
from src.operators.watermark_assigner import create_watermark_strategy
from src.operators.key_generator import generate_trip_id
from src.operators.deduplicator import DeduplicatorFunction
from src.operators.schema_validator import SchemaValidator
from src.operators.canary_rules import CanaryRulesValidator, ViolationFilter
from src.operators.meta_aggregator import (
    VotingEnsembleFunction,
    MetaAggregateFunction,
    MetaWindowProcessFunction,
    extract_neighborhood_key
)
from src.operators.iec_operator import IECOperator
from pyflink.datastream import MapFunction


class ParseJsonFunction(MapFunction):
    def map(self, value):
        try:
            return json.loads(value)
        except:
            return None


class AddTripIdFunction(MapFunction):
    def map(self, record):
        if record is None:
            return None
        record['trip_id'] = generate_trip_id(record)
        return record


class MockMLScoringFunction(MapFunction):
    """Mock ML scoring - không cần load model (tiết kiệm RAM)"""
    def map(self, value):
        import random
        value['anomaly_score'] = random.uniform(0.2, 0.8)
        value['threshold'] = 0.50
        value['is_anomaly'] = value['anomaly_score'] > value['threshold']
        value['context_key'] = 'manhattan_midday_weekday_medium'
        return value


class ExtractNeighborhoodFunction(MapFunction):
    def map(self, record):
        return extract_neighborhood_key(record), record


def main():
    print("="*80)
    print("CA-DQStream Laptop Version - Lightweight (1 parallelism)")
    print("="*80)

    env = StreamExecutionEnvironment.get_execution_environment()

    # ⚠️ LAPTOP CONFIG: Parallelism 1 (tiết kiệm RAM)
    env.set_parallelism(1)

    # ⚠️ LAPTOP CONFIG: Checkpointing 120s (giảm overhead)
    checkpoint_config = env.get_checkpoint_config()
    checkpoint_config.set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
    checkpoint_config.set_checkpoint_interval(120000)  # 120s (was 45s)
    checkpoint_config.set_min_pause_between_checkpoints(60000)  # 60s
    checkpoint_config.set_checkpoint_timeout(180000)  # 3min
    checkpoint_config.set_max_concurrent_checkpoints(1)

    print("\n✓ Laptop config:")
    print(f"  Parallelism: 1 (single-threaded)")
    print(f"  Checkpointing: 120s interval")
    print(f"  Memory: Optimized for ~2GB")

    # ═══════════════════════════════════════════════════════════════
    # LAYER 1: Baseline Validation
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("LAYER 1: Baseline Validation")
    print("="*80)

    properties = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'cadqstream-laptop',
        'auto.offset.reset': 'earliest',
    }

    kafka_source = FlinkKafkaConsumer(
        topics='taxi-nyc-raw',
        deserialization_schema=SimpleStringSchema(),
        properties=properties
    )

    stream = env.add_source(kafka_source)

    stream = (
        stream
        .map(ParseJsonFunction(), output_type=Types.PICKLED_BYTE_ARRAY())
        .filter(lambda x: x is not None)
        .assign_timestamps_and_watermarks(create_watermark_strategy())
    )

    stream = stream.map(AddTripIdFunction(), output_type=Types.PICKLED_BYTE_ARRAY())

    deduplicated_stream = (
        stream
        .key_by(lambda x: x['trip_id'], key_type=Types.STRING())
        .map(DeduplicatorFunction(), output_type=Types.PICKLED_BYTE_ARRAY())
        .filter(lambda x: x is not None)
    )

    validator = SchemaValidator()
    valid_stream = deduplicated_stream.filter(validator)

    print("✓ Layer 1 ready")

    # ═══════════════════════════════════════════════════════════════
    # LAYER 2: Dual-Branch (Mock ML for laptop)
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("LAYER 2: Dual-Branch (Canary + Mock ML)")
    print("="*80)

    canary_stream = valid_stream.map(
        CanaryRulesValidator(),
        output_type=Types.PICKLED_BYTE_ARRAY()
    )

    complex_stream = valid_stream.map(
        MockMLScoringFunction(),
        output_type=Types.PICKLED_BYTE_ARRAY()
    )

    print("✓ Layer 2 ready (using mock ML - no model loading)")

    # ═══════════════════════════════════════════════════════════════
    # LAYER 3: Rendezvous + MetaAggregator
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("LAYER 3: Rendezvous + MetaAggregator")
    print("="*80)

    merged_stream = canary_stream.union(complex_stream)

    voting_stream = merged_stream.map(
        VotingEnsembleFunction(),
        output_type=Types.PICKLED_BYTE_ARRAY()
    )

    meta_stream = (
        voting_stream
        .map(ExtractNeighborhoodFunction(), output_type=Types.TUPLE([Types.STRING(), Types.PICKLED_BYTE_ARRAY()]))
        .key_by(lambda x: x[0], key_type=Types.STRING())
        .window(TumblingEventTimeWindows.of(Time.minutes(1)))
        .aggregate(
            MetaAggregateFunction(),
            MetaWindowProcessFunction(),
            accumulator_type=Types.PICKLED_BYTE_ARRAY(),
            output_type=Types.PICKLED_BYTE_ARRAY()
        )
    )

    print("✓ Layer 3 ready")

    # ═══════════════════════════════════════════════════════════════
    # LAYER 4: IEC
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("LAYER 4: IEC")
    print("="*80)

    iec_stream = meta_stream.map(
        IECOperator(),
        output_type=Types.PICKLED_BYTE_ARRAY()
    )

    print("✓ Layer 4 ready")

    # ═══════════════════════════════════════════════════════════════
    # Outputs: Print only (no external sinks for laptop)
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("OUTPUTS (print only)")
    print("="*80)

    # Sample 1 in 100 records to avoid overwhelming console
    valid_stream.filter(lambda x: hash(x['trip_id']) % 100 == 0).print()

    print("\n" + "="*80)
    print("STARTING LAPTOP PIPELINE")
    print("="*80)

    env.execute("CA-DQStream Laptop - Lightweight")


if __name__ == "__main__":
    main()
