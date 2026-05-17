"""
Dead Letter Queue (DLQ) Infrastructure for CA-DQStream.

NOTE: DLQ is implemented via inline sentinel dicts with _dlq=True flag,
not Flink SideOutputs. See flink_job_complete.py DlqFilter pattern which
routes records by checking record['_dlq'] == True.
The OutputTag definitions below are kept for reference but are NOT connected
to any Flink side output in the current pipeline.

ROOT CAUSE FIX #2: Zero-Null Tolerance

Every record that fails processing must NOT be silently dropped or passed as None
downstream. The rule is: either filter it out completely (FlatMap returning nothing)
OR route it to a DLQ side output with a standard schema.

DLQ Categories:
  - PARSE_ERROR: JSON decode failed in SafeParseJsonFunction
  - SCHEMA_ERROR: Missing/invalid required fields in SchemaValidator
  - ALGORITHM_ERROR: Math/numeric error in MemStreamScoringOperator or IECOperator
  - VALIDATION_ERROR: Record type mismatch or unexpected null in serialization

Each DLQ entry includes:
  - _dlq_reason: Human-readable reason code
  - _dlq_category: One of the categories above
  - _dlq_timestamp: Event time from record (not processing time)
  - _dlq_original: Original record dict (serialized to JSON string)
  - _dlq_operator: Which operator produced the DLQ entry
  - _dlq_error: Specific error message

Usage:
  from src.operators.dlq import (
      DLQ_PARSE_ERROR, DLQ_SCHEMA_ERROR, DLQ_ALGORITHM_ERROR,
      DLQ_VALIDATION_ERROR,
      emit_to_dlq,
  )

  class MyOperator(MapFunction):
      def map(self, value):
          try:
              return self._process(value)
          except SomeError as e:
              ctx.output(DLQ_ALGORITHM_ERROR, emit_to_dlq(
                  value, ctx, 'ALGORITHM_ERROR',
                  f'failed in MyOperator: {e}'
              ))
              return self._safe_sentinel(value)
"""

from pyflink.datastream import OutputTag
from pyflink.common.typeinfo import Types
import json
import logging
from typing import Dict, Any, Optional

LOGGER = logging.getLogger('cadqstream-dlq')

# ─────────────────────────────────────────────────────────────────────────────
# OutputTag definitions — one per DLQ category
# ─────────────────────────────────────────────────────────────────────────────

DLQ_PARSE_ERROR = OutputTag(
    'dlq-parse-error',
    type_info=Types.PICKLED_BYTE_ARRAY(),
    description='Records that failed JSON parsing'
)

DLQ_SCHEMA_ERROR = OutputTag(
    'dlq-schema-error',
    type_info=Types.PICKLED_BYTE_ARRAY(),
    description='Records that failed schema validation'
)

DLQ_ALGORITHM_ERROR = OutputTag(
    'dlq-algorithm-error',
    type_info=Types.PICKLED_BYTE_ARRAY(),
    description='Records that caused algorithm/math errors'
)

DLQ_VALIDATION_ERROR = OutputTag(
    'dlq-validation-error',
    type_info=Types.PICKLED_BYTE_ARRAY(),
    description='Records that failed type/value validation'
)


def emit_to_dlq(
    record: Any,
    ctx: 'SideOutputProcessFunction.Context',
    category: str,
    reason: str,
    operator_name: str = 'unknown',
) -> Dict[str, Any]:
    """Build a standard DLQ entry from a failed record.

    ROOT CAUSE FIX #2: This function guarantees a valid dict with all required
    fields. It NEVER returns None. Downstream operators can safely access any field
    without risk of AttributeError/KeyError.

    Args:
        record: The original record (dict, str, bytes, or None)
        ctx: Flink SideOutputProcessFunction.Context
        category: One of PARSE_ERROR, SCHEMA_ERROR, ALGORITHM_ERROR, VALIDATION_ERROR
        reason: Human-readable error reason
        operator_name: Name of the operator that produced the failure

    Returns:
        A fully-populated DLQ dict (never None)
    """
    # Extract event time from record if available
    event_time = ''
    if isinstance(record, dict):
        event_time = record.get('tpep_pickup_datetime', '') or record.get('iec_timestamp', '')

    # Serialize original record (handle non-dict types)
    if isinstance(record, dict):
        original_json = json.dumps(record, default=str)
    elif isinstance(record, (str, bytes)):
        original_json = str(record)[:2000]
    else:
        original_json = repr(record)[:2000]

    dlq_entry = {
        '_dlq_reason': str(reason),
        '_dlq_category': category,
        '_dlq_timestamp': event_time,
        '_dlq_operator': operator_name,
        '_dlq_error': str(reason),
        '_dlq_original': original_json,
        # passthrough fields so DLQ records can still be analyzed
        '_dlq_dlq': True,
    }

    # Preserve any fields from the original record for context
    if isinstance(record, dict):
        for key in ('trip_id', 'VendorID', 'tpep_pickup_datetime', 'PULocationID',
                    'DOLocationID', 'fare_amount', 'trip_distance'):
            if key in record and record[key] is not None:
                dlq_entry[key] = record[key]

    return dlq_entry


def dlq_entry_to_json(entry: Dict[str, Any]) -> str:
    """Serialize a DLQ entry to JSON for MinIO/Kafka sinks.

    Args:
        entry: DLQ entry dict from emit_to_dlq

    Returns:
        JSON string representation
    """
    return json.dumps(entry, default=str)


def build_dlq_sink_record(
    record: Any,
    category: str,
    reason: str,
    operator_name: str,
) -> Dict[str, Any]:
    """Build a DLQ entry dict WITHOUT a Flink context.

    Use this in non-Flink contexts (e.g., MapFunction without SideOutput access)
    to build a DLQ record that will be routed via a separate DLQ stream.

    This is used when we want to emit to DLQ without access to SideOutputContext.

    Returns:
        Fully-populated DLQ dict (never None)
    """
    event_time = ''
    if isinstance(record, dict):
        event_time = record.get('tpep_pickup_datetime', '') or record.get('iec_timestamp', '')

    if isinstance(record, dict):
        original_json = json.dumps(record, default=str)
    elif isinstance(record, (str, bytes)):
        original_json = str(record)[:2000]
    else:
        original_json = repr(record)[:2000]

    entry = {
        '_dlq_reason': str(reason),
        '_dlq_category': category,
        '_dlq_timestamp': event_time,
        '_dlq_operator': operator_name,
        '_dlq_error': str(reason),
        '_dlq_original': original_json,
        '_dlq_dlq': True,
    }

    if isinstance(record, dict):
        for key in ('trip_id', 'VendorID', 'tpep_pickup_datetime', 'PULocationID',
                    'DOLocationID', 'fare_amount', 'trip_distance'):
            if key in record and record[key] is not None:
                entry[key] = record[key]

    return entry


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel builders — return valid dicts for broken records
# These prevent None propagation into downstream operators
# ─────────────────────────────────────────────────────────────────────────────

def safe_sentinel_for_output(record: Any, operator: str) -> Dict[str, Any]:
    """Return a valid sentinel dict for records that cannot be processed.

    Used when we need to continue the pipeline without a record but cannot emit
    None (because downstream expects non-null dicts).

    The sentinel is marked with _sentinel=True so sinks can distinguish it.

    Returns:
        Dict with sentinel marker and preserved trip_id if available.
    """
    if isinstance(record, dict):
        trip_id = record.get('trip_id', 'unknown')
        pickup = record.get('tpep_pickup_datetime', '')
    else:
        trip_id = 'unknown'
        pickup = ''

    return {
        '_sentinel': True,
        '_sentinel_source': operator,
        'trip_id': trip_id,
        'tpep_pickup_datetime': pickup,
    }


__all__ = [
    'DLQ_PARSE_ERROR',
    'DLQ_SCHEMA_ERROR',
    'DLQ_ALGORITHM_ERROR',
    'DLQ_VALIDATION_ERROR',
    'emit_to_dlq',
    'dlq_entry_to_json',
    'build_dlq_sink_record',
    'safe_sentinel_for_output',
]
