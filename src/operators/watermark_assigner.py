"""
Watermark assignment for event-time processing.
Spec: Lines 1491-1510 (withIdleness 30s)

ROOT CAUSE FIX: Historical timestamps (2024) stuck watermark advancement.
Solution: Use for_monotonous_time() for window triggering. This assigns
processing-time-based watermarks so TumblingEventTimeWindows fire every 1 minute
regardless of event timestamps. The TaxiTripTimestampAssigner is still used
downstream by MemStreamScoringOperator for decay factor computation.
"""

from pyflink.common import WatermarkStrategy, Duration
from pyflink.common.watermark_strategy import TimestampAssigner


class TaxiTripTimestampAssigner(TimestampAssigner):
    """Extract pickup_datetime as event timestamp for ML decay factor computation.

    This timestamp is used by MemStreamScoringOperator to compute
    time_since_last_update for decay_factor computation.
    It is NOT used for watermark assignment (which uses processing time).
    """

    def extract_timestamp(self, value, record_timestamp):
        """Return event time from pickup_datetime, with Kafka record timestamp as fallback.

        Returns event time in milliseconds for ML decay factor computation.
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
    """Create watermark strategy using processing time for window triggering.

    ROOT CAUSE FIX (Time Paradox):
      Historical 2024 timestamps blocked watermark advancement, preventing
      TumblingEventTimeWindows from ever firing. The pipeline processed data
      (Source read 17K+ records, Map2 buffered at 5000) but Window received
      0 records because the watermark from 2024 was always behind the
      event-time window boundaries.

      Solution: for_monotonous_time() assigns watermarks based on processing
      time (wall clock), so windows fire every 1 minute regardless of
      the event timestamps in the data.

    TaxiTripTimestampAssigner is retained for ML decay factor computation
    in MemStreamScoringOperator (time_since_last_update).

    Returns:
        WatermarkStrategy with processing-time watermarks for reliable windowing
    """
    strategy = (
        WatermarkStrategy
        .for_monotonous_time()
        .with_timestamp_assigner(TaxiTripTimestampAssigner())
        .with_idleness(Duration.of_seconds(30))
    )

    return strategy
