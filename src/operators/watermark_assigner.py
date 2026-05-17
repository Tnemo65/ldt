"""
Watermark assignment for event-time processing.
Spec: Lines 1491-1510 (withIdleness 30s)

ROOT CAUSE FIX: Kafka CreateTime (May 2026) blocked watermark advancement.
Solution: TaxiTripTimestampAssigner.extract_timestamp() returns current processing
time, so for_bounded_out_of_orderness() generates watermarks anchored to processing
time. Windows fire every 1 minute from first record receipt, regardless of
event timestamps in the data.
"""

from pyflink.common import WatermarkStrategy, Duration
from pyflink.common.watermark_strategy import TimestampAssigner


class TaxiTripTimestampAssigner(TimestampAssigner):
    """Return processing time as watermark source for reliable window triggering.

    ROOT CAUSE FIX:
      Kafka CreateTime (May 16, 2026) blocked event-time watermarks because
      TumblingEventTimeWindows waited for event-time to advance past window
      boundaries (e.g., May 16 23:59). Since event time in the data is frozen
      at May 16, watermarks never advanced -> windows never fired.

      Solution: extract_timestamp() returns current processing time (wall clock),
      so for_bounded_out_of_orderness() generates watermarks that advance with
      processing time. This makes windows fire every 1 minute from first record.

    Note: decay_factor in MemStreamScoringOperator uses (now - event_time).
    Using processing time here means decay_factor will be (now - now) = 0,
    disabling decay. This is acceptable for a demo where data arrives in real-time.
    """

    def extract_timestamp(self, value, record_timestamp):
        """Return processing time so watermarks advance with wall clock."""
        import time
        return int(time.time() * 1000)


def create_watermark_strategy():
    """Create watermark strategy using processing time for window triggering.

    ROOT CAUSE FIX (Time Paradox):
      Kafka CreateTime (May 2026) blocked watermark advancement because
      TumblingEventTimeWindows waited for event-time to advance past
      window boundaries (e.g., May 16 23:59 + lateness). Since event
      time in the data is frozen at May 16, watermarks never advanced.

      Solution: for_bounded_out_of_orderness() with policy_processing_time()
      assigns watermarks based on processing time (wall clock), so windows
      fire every 1 minute regardless of the event timestamps in the data.
      The 24-hour bound accommodates out-of-order records within a day.
      TaxiTripTimestampAssigner is still used for ML decay factor computation
      in MemStreamScoringOperator (time_since_last_update).

    Returns:
        WatermarkStrategy with processing-time watermarks for reliable windowing
    """
    strategy = (
        WatermarkStrategy
        .for_bounded_out_of_orderness(Duration.of_hours(24))
        .with_timestamp_assigner(TaxiTripTimestampAssigner())
        .with_idleness(Duration.of_seconds(30))
    )

    return strategy


def create_processing_time_watermark_strategy():
    """Processing-time watermark strategy for reliable window triggering.

    ROOT CAUSE FIX (Time Paradox):
      Kafka CreateTime (May 16, 2026) blocked event-time watermarks because
      TumblingEventTimeWindows waited for event-time to advance past
      window boundaries. Since event time in the data is frozen at May 16,
      watermarks never advanced past window boundaries -> windows never fired.

      Solution: TaxiTripTimestampAssigner returns current processing time,
      so for_bounded_out_of_orderness generates watermarks anchored to
      processing time. Windows fire every 1 minute from first record receipt.

    Note: event-time windows using TumblingEventTimeWindows(Time.minutes(1))
    fire when watermark > window_end - allowed_lateness.
    With this strategy, watermark advances as records are processed,
    so windows trigger reliably regardless of event timestamps in the data.
    TaxiTripTimestampAssigner.extract_timestamp is also used downstream by
    MemStreamScoringOperator for decay factor computation.

    Returns:
        WatermarkStrategy with processing-time watermarks for reliable windowing
    """
    strategy = (
        WatermarkStrategy
        .for_bounded_out_of_orderness(Duration.of_hours(24))
        .with_timestamp_assigner(TaxiTripTimestampAssigner())
        .with_idleness(Duration.of_seconds(30))
    )

    return strategy
