"""
Layer 1: Baseline Validation Operators.

This module contains PyFlink operators for Layer 1 of the CA-DQStream pipeline:
- ParseJsonFunction: Parse raw JSON bytes to dict
- AddTripIdFunction: Generate composite trip ID using MD5 hash
- DeduplicatorFunction: Deduplicate records using keyed state with 7-day TTL
- SchemaValidator: Validate required fields and zone ranges
- WatermarkAssigner: Assign event-time watermarks with bounded out-of-orderness

Reference: original_flow.md lines 374-422
"""

from .parse_json import ParseJsonFunction
from .add_trip_id import AddTripIdFunction
from .deduplicator import DeduplicatorFunction
from .schema_validator import SchemaValidator, ValidFilter, InvalidFilter
from .watermark_assigner import WatermarkAssigner, create_watermark_strategy

__all__ = [
    'ParseJsonFunction',
    'AddTripIdFunction',
    'DeduplicatorFunction',
    'SchemaValidator',
    'ValidFilter',
    'InvalidFilter',
    'WatermarkAssigner',
    'create_watermark_strategy',
]
