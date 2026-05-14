"""
DeduplicatorFunction - Keyed Record Deduplication.

Deduplicates records using keyed ValueState with 7-day TTL.
Key: trip_id (from AddTripIdFunction)
Backend: RocksDB (production default)

Reference: original_flow.md lines 401-412
"""

import logging
from typing import Dict, Optional, Iterable

from pyflink.datastream import KeyedProcessFunction
from pyflink.datastream.state import ValueStateDescriptor, StateTtlConfig
from pyflink.datastream.time_characteristic import TimeCharacteristic
from pyflink.common.typeinfo import BasicTypeInfo
from pyflink.common.time import Time
from pyflink.common import Watermark

LOGGER = logging.getLogger('cadqstream.layer1')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


# Default TTL: 7 days (as per original_flow.md)
DEFAULT_TTL_DAYS = 7


class DeduplicatorFunction(KeyedProcessFunction):
    """
    Deduplicate records using keyed state with TTL.
    
    Key: trip_id
    State: ValueState("seen", BOOLEAN)
    TTL: 7 days
    
    Logic:
        if seen.value() == True → return None (drop duplicate)
        else → seen.update(True) → pass through
    
    Note: Uses on_create_and_write update type and never_return_expired visibility.
    """
    
    def __init__(self, ttl_days: int = DEFAULT_TTL_DAYS):
        """
        Initialize deduplicator.
        
        Args:
            ttl_days: TTL for deduplication state (default: 7 days)
        """
        self.ttl_days = ttl_days
        
        state_ttl_config = StateTtlConfig.new_builder(Time.days(ttl_days)) \
            .set_update_type(StateTtlConfig.UpdateType.ON_CREATE_AND_WRITE) \
            .set_state_visibility(StateTtlConfig.StateVisibility.NEVER_RETURN_EXPIRED) \
            .cleanup_in_rocksdb_compact_filter(1000) \
            .build()
        
        self._seen_state_desc = ValueStateDescriptor(
            "dedup_seen",
            BasicTypeInfo.BOOLEAN_TYPE_INFO
        )
        self._seen_state_desc.enable_time_to_live(state_ttl_config)
        
        self._total_records = 0
        self._duplicates = 0
    
    def open(self, runtime_context):
        """Initialize state descriptors."""
        self._seen_state = runtime_context.get_state(self._seen_state_desc)
        self._total_records = 0
        self._duplicates = 0
        LOGGER.info(
            "[Deduplicator] Initialized with %d-day TTL",
            self.ttl_days
        )
    
    def process_element(self, record: Dict, context) -> Iterable[Dict]:
        """
        Check and update deduplication state.
        
        Args:
            record: Input record with trip_id
            context: KeyedProcessFunction context
            
        Yields:
            Record if not duplicate, None if duplicate
        """
        self._total_records += 1
        
        trip_id = record.get('trip_id')
        if not trip_id:
            LOGGER.warning(
                "[Deduplicator] Record without trip_id, passing through"
            )
            yield record
            return
        
        seen = self._seen_state.value()
        
        if seen is True:
            self._duplicates += 1
            LOGGER.debug(
                "[Deduplicator] Duplicate dropped: %s",
                trip_id[:16]
            )
            return
        else:
            self._seen_state.update(True)
            yield record
    
    def close(self):
        """Log final statistics."""
        if self._total_records > 0:
            dup_rate = self._duplicates / self._total_records * 100
            LOGGER.info(
                "[Deduplicator] Stats: total=%d, duplicates=%d (%.2f%%)",
                self._total_records, self._duplicates, dup_rate
            )


class DeduplicatorFactory:
    """
    Factory for creating DeduplicatorFunction instances.
    
    Provides static method to create deduplicator with default config.
    """
    
    @staticmethod
    def create(ttl_days: int = DEFAULT_TTL_DAYS) -> DeduplicatorFunction:
        """
        Create DeduplicatorFunction instance.
        
        Args:
            ttl_days: TTL for deduplication state
            
        Returns:
            Configured DeduplicatorFunction
        """
        return DeduplicatorFunction(ttl_days=ttl_days)
    
    @staticmethod
    def create_with_custom_ttl(
        ttl_hours: Optional[int] = None,
        ttl_minutes: Optional[int] = None,
        ttl_seconds: Optional[int] = None
    ) -> DeduplicatorFunction:
        """
        Create DeduplicatorFunction with custom TTL unit.
        
        Args:
            ttl_hours: TTL in hours
            ttl_minutes: TTL in minutes
            ttl_seconds: TTL in seconds
            
        Returns:
            Configured DeduplicatorFunction
        """
        if ttl_hours:
            return DeduplicatorFunction(ttl_days=ttl_hours / 24)
        elif ttl_minutes:
            return DeduplicatorFunction(ttl_days=ttl_minutes / 1440)
        elif ttl_seconds:
            return DeduplicatorFunction(ttl_days=ttl_seconds / 86400)
        else:
            return DeduplicatorFunction()
