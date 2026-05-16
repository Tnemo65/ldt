"""
Watermark assignment for event-time processing.
Spec: Lines 1491-1510 (withIdleness 30s)

FIX: Use processing time instead of event time for L4 windowing.
Reason: Data has historical event timestamps (2024) which block watermark advancement.
Solution: Return current processing time as event time so windows trigger every 1 minute.

Task 1 - WatermarkStrategy Refactoring:
  - BoundedOutOfOrderness explicitly set to 5 seconds (not 10s)
  - Idleness set to 30 seconds (prevents partition stall)
  - Processing time used for timestamp (unbounded advancement)
"""

from pyflink.common import WatermarkStrategy, Duration
from pyflink.common.watermark_strategy import TimestampAssigner


class TaxiTripTimestampAssigner(TimestampAssigner):
    """Extract pickup_datetime as event timestamp for bounded event-time processing.

    FORBIDDEN: Processing time (time.time(), datetime.now()) was previously used here.
    This caused a 2-year time delta when streaming historical 2024/2025 data,
    breaking ML algorithm decay factors (decayed to 0) and causing all
    IEC strategies to fall back to do_nothing.

    FIX: Extract the actual event time from the record's tpep_pickup_datetime field.
    If the field is missing or unparseable, fall back to the Kafka record timestamp.
    This allows bounded watermark advancement based on the event's temporal ordering
    rather than wall-clock processing time.
    """

    def extract_timestamp(self, value, record_timestamp):
        """Return event time from pickup_datetime, with Kafka record timestamp as fallback.

        Returns event time in milliseconds for downstream windowing and watermark bounding.
        """
        if not isinstance(value, dict):
            return record_timestamp if record_timestamp is not None else 0

        dt_str = value.get('tpep_pickup_datetime', '')
        if dt_str:
            from datetime import datetime
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S UTC',
                '%Y/%m/%d %H:%M:%S',
                '%m/%d/%Y %H:%M:%S',
                '%m/%d/%Y %I:%M:%S %p',
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(str(dt_str).strip(), fmt)
                    return int(dt.timestamp() * 1000)
                except (ValueError, TypeError):
                    continue

        return record_timestamp if record_timestamp is not None else 0


def create_watermark_strategy():
    """Create watermark strategy for bounded event-time processing.

    ROOT CAUSE FIX #1 (Time Paradox):
      - BoundedOutOfOrderness of 5 seconds (handle late arrivals within 5s window)
      - Idleness of 30 seconds (unblock partitions that go silent)
      - Event-time timestamp assigner (extracts tpep_pickup_datetime from record)
      - Fallback to Kafka record timestamp if pickup_datetime is missing

    ROOT CAUSE FIX #1: Previously used processing time (time.time()), which caused
    a 2-year time delta when streaming 2024/2025 historical data. Decay factors
    in MemStream/ADWIN decayed to 0, breaking all ML algorithms and causing
    IEC to fall back to do_nothing for all records.

    Returns:
        WatermarkStrategy configured for bounded, fault-tolerant event-time streaming
    """
    strategy = (
        WatermarkStrategy
        .for_bounded_out_of_orderness(Duration.of_seconds(5))
        .with_timestamp_assigner(TaxiTripTimestampAssigner())
        .with_idleness(Duration.of_seconds(30))
    )

    return strategy
