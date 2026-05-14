"""
WatermarkAssigner - Event-Time Watermark Generator.

Assigns event-time watermarks based on tpep_pickup_datetime.
Bounded out-of-orderness: 10 seconds

Reference: original_flow.md lines 384-391
"""

import logging
from datetime import datetime
from typing import Optional, Dict

from pyflink.datastream import TimeCharacteristic
from pyflink.datastream.watermark_strategy import WatermarkStrategy, WatermarkStrategyWithTimestampGuesser
from pyflink.common.time import Time
from pyflink.common.typeinfo import Types
from pyflink.common.watermark_strategy import Duration

LOGGER = logging.getLogger('cadqstream.layer1')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


# Default bounded out-of-orderness: 10 seconds
DEFAULT_BOUNDED_OUT_OF_ORDERNESS_MS = 10_000

# Idleness timeout: 30 seconds (fix for PyFlink v1.9 bug)
IDLENESS_TIMEOUT_MS = 30_000

# Supported datetime formats
DATETIME_FORMATS = [
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M:%S UTC',
    '%Y/%m/%d %H:%M:%S',
    '%m/%d/%Y %H:%M:%S',
    '%m/%d/%Y %I:%M:%S %p',
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%dT%H:%M:%S.%f',
]


def parse_tpep_datetime(value) -> Optional[int]:
    """
    Parse tpep_pickup_datetime to Unix milliseconds.
    
    Args:
        value: Datetime string
        
    Returns:
        Unix timestamp in milliseconds, or None if parsing fails
    """
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        ts = int(value)
        if ts > 1e12:
            return ts
        else:
            return ts * 1000
    
    if not isinstance(value, str):
        value = str(value)
    
    value = value.strip()
    
    for fmt in DATETIME_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    
    return None


class WatermarkAssigner:
    """
    Watermark assigner for taxi event-time processing.
    
    Event time: tpep_pickup_datetime
    Bounded out-of-orderness: 10 seconds
    Watermark: max(timestamp) - 10s
    Late data: still processed (not dropped)
    """
    
    def __init__(
        self,
        bounded_out_of_orderness_ms: int = DEFAULT_BOUNDED_OUT_OF_ORDERNESS_MS,
        idleness_timeout_ms: int = IDLENESS_TIMEOUT_MS
    ):
        """
        Initialize watermark assigner.
        
        Args:
            bounded_out_of_orderness_ms: Max allowed lateness (default: 10 seconds)
            idleness_timeout_ms: Idleness timeout for source partitions
        """
        self.bounded_out_of_orderness_ms = bounded_out_of_orderness_ms
        self.idleness_timeout_ms = idleness_timeout_ms
    
    def assign_timestamp(self, record: Dict) -> Optional[int]:
        """
        Extract timestamp from record.
        
        Args:
            record: Input record with tpep_pickup_datetime
            
        Returns:
            Unix timestamp in milliseconds
        """
        pickup_dt = record.get('tpep_pickup_datetime')
        return parse_tpep_datetime(pickup_dt)
    
    def create_strategy(self) -> WatermarkStrategy:
        """
        Create watermark strategy with bounded out-of-orderness.
        
        Returns:
            WatermarkStrategy configured for taxi event-time
        """
        def timestamp_extractor(record: Dict) -> int:
            ts = self.assign_timestamp(record)
            if ts is None:
                return 0
            return ts
        
        bounded_out_of_orderness = Duration.of_millis(self.bounded_out_of_orderness_ms)
        
        strategy = (WatermarkStrategy
            .for_bounded_out_of_orderness(bounded_out_of_orderness)
            .with_timestamp_assigner(lambda record, timestamp: timestamp_extractor(record))
            .with_idleness_timeout(Duration.of_millis(self.idleness_timeout_ms)))
        
        return strategy


def create_watermark_strategy(
    bounded_out_of_orderness_ms: int = DEFAULT_BOUNDED_OUT_OF_ORDERNESS_MS,
    idleness_timeout_ms: int = IDLENESS_TIMEOUT_MS
) -> WatermarkStrategy:
    """
    Create watermark strategy with standard config.
    
    Args:
        bounded_out_of_orderness_ms: Max allowed lateness (default: 10 seconds)
        idleness_timeout_ms: Idleness timeout for source partitions
        
    Returns:
        WatermarkStrategy configured for taxi event-time
    """
    assigner = WatermarkAssigner(
        bounded_out_of_orderness_ms=bounded_out_of_orderness_ms,
        idleness_timeout_ms=idleness_timeout_ms
    )
    return assigner.create_strategy()


class WatermarkAssignerWithPeriodicWatermarks(WatermarkAssigner):
    """
    WatermarkAssigner with periodic watermark emission.
    
    Alternative implementation using periodic watermarks
    (useful for sources that don't emit records frequently).
    """
    
    def __init__(
        self,
        bounded_out_of_orderness_ms: int = DEFAULT_BOUNDED_OUT_OF_ORDERNESS_MS,
        watermark_interval_ms: int = 1000
    ):
        super().__init__(bounded_out_of_orderness_ms)
        self.watermark_interval_ms = watermark_interval_ms
        self._last_watermark = 0
        self._max_timestamp = 0
    
    def assign_timestamp_and_watermark(self, record: Dict) -> int:
        """
        Assign timestamp and track max for watermark.
        
        Returns:
            Extracted timestamp
        """
        ts = self.assign_timestamp(record)
        if ts is None:
            ts = 0
        
        if ts > self._max_timestamp:
            self._max_timestamp = ts
        
        return ts
    
    def get_current_watermark(self) -> int:
        """
        Calculate current watermark.
        
        Returns:
            Watermark = max_timestamp - bounded_out_of_orderness
        """
        return self._max_timestamp - self.bounded_out_of_orderness_ms
