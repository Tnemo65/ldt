"""
SchemaValidator - Layer 1 Schema Validation.

Validates required fields and zone ranges for NYC taxi data.
Splits stream into valid and violation streams.

Required fields: trip_distance, fare_amount, PULocationID, DOLocationID, passenger_count
Zone range: PULocationID, DOLocationID must be in 1-263

Reference: original_flow.md lines 414-422
"""

import logging
from typing import Dict, Tuple, Optional, Iterable

from pyflink.datastream import ProcessFunction
from pyflink.common import Time

LOGGER = logging.getLogger('cadqstream.layer1')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


# Zone range validation (NYC TLC zones 1-263)
MIN_ZONE_ID = 1
MAX_ZONE_ID = 263

# Required fields
REQUIRED_FIELDS = [
    'trip_distance',
    'fare_amount',
    'PULocationID',
    'DOLocationID',
    'passenger_count',
]


class SchemaValidationResult:
    """Result of schema validation."""
    
    def __init__(self, is_valid: bool, violations: Optional[list] = None):
        self.is_valid = is_valid
        self.violations = violations or []
    
    def __bool__(self) -> bool:
        return self.is_valid


def validate_schema(record: Dict) -> SchemaValidationResult:
    """
    Validate record against schema requirements.
    
    Args:
        record: Input record
        
    Returns:
        SchemaValidationResult with is_valid and violations list
    """
    violations = []
    
    for field in REQUIRED_FIELDS:
        if field not in record:
            violations.append(f"missing_field:{field}")
            continue
        
        value = record[field]
        if value is None:
            violations.append(f"null_field:{field}")
            continue
        
        if isinstance(value, str) and value.strip() == '':
            violations.append(f"empty_field:{field}")
    
    pu_zone = record.get('PULocationID')
    do_zone = record.get('DOLocationID')
    
    if pu_zone is not None:
        try:
            pu_int = int(float(pu_zone))
            if pu_int < MIN_ZONE_ID or pu_int > MAX_ZONE_ID:
                violations.append(f"invalid_zone:PULocationID={pu_int}")
        except (ValueError, TypeError):
            violations.append(f"non_numeric:PULocationID={pu_zone}")
    
    if do_zone is not None:
        try:
            do_int = int(float(do_zone))
            if do_int < MIN_ZONE_ID or do_int > MAX_ZONE_ID:
                violations.append(f"invalid_zone:DOLocationID={do_int}")
        except (ValueError, TypeError):
            violations.append(f"non_numeric:DOLocationID={do_zone}")
    
    return SchemaValidationResult(
        is_valid=len(violations) == 0,
        violations=violations
    )


class SchemaValidator(ProcessFunction):
    """
    Validate records against schema and emit to appropriate outputs.
    
    Output[0]: valid_stream - Records passing validation
    Output[1]: violation_stream - Records failing validation
    
    Note: Uses SideOutput for violation_stream (more efficient in Flink).
    """
    
    def __init__(self):
        self._total_records = 0
        self._valid_records = 0
        self._invalid_records = 0
    
    def open(self, runtime_context):
        """Initialize counters."""
        self._total_records = 0
        self._valid_records = 0
        self._invalid_records = 0
    
    def process_element(self, record: Dict, ctx: ProcessFunction.Context) -> Iterable[Dict]:
        """
        Validate record and route to appropriate output.
        
        Args:
            record: Input record
            ctx: ProcessFunction context
            
        Yields:
            To main output (valid) or side output (violation)
        """
        self._total_records += 1
        
        result = validate_schema(record)
        
        if result.is_valid:
            self._valid_records += 1
            yield record
        else:
            self._invalid_records += 1
            
            violation_record = {
                **record,
                '_validation_errors': result.violations,
                '_validation_timestamp': ctx.timestamp(),
            }
            
            ctx.output(SideOutputTag.VIOLATIONS, violation_record)
    
    def close(self):
        """Log final statistics."""
        if self._total_records > 0:
            valid_pct = self._valid_records / self._total_records * 100
            invalid_pct = self._invalid_records / self._total_records * 100
            LOGGER.info(
                "[SchemaValidator] Stats: total=%d, valid=%d (%.1f%%), "
                "invalid=%d (%.1f%%)",
                self._total_records, self._valid_records, valid_pct,
                self._invalid_records, invalid_pct
            )


class SideOutputTag:
    """Side output tags for SchemaValidator."""
    VIOLATIONS = 'violations'


class ValidFilter(ProcessFunction):
    """
    Filter valid records (pass-through, no additional checks).
    
    Used after SchemaValidator to filter main output.
    """
    
    def process_element(self, record: Dict, ctx) -> Iterable[Dict]:
        """Pass through all records."""
        yield record


class InvalidFilter(ProcessFunction):
    """
    Filter invalid records (side output only).
    
    Used to extract violation records from side output.
    """
    
    def __init__(self):
        self._total = 0
        self._violations = 0
    
    def process_element(self, record: Dict, ctx) -> Iterable[Dict]:
        """Extract records with validation errors."""
        self._total += 1
        
        if '_validation_errors' in record and record['_validation_errors']:
            self._violations += 1
            yield record
