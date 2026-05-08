"""
Watermark assignment for event-time processing.
Spec: Lines 1491-1510 (withIdleness 30s)
"""

from pyflink.common import WatermarkStrategy, Duration
from pyflink.common.watermark_strategy import TimestampAssigner
from datetime import datetime

class TaxiTripTimestampAssigner(TimestampAssigner):
    """Extract pickup_datetime as event timestamp."""

    def extract_timestamp(self, value, record_timestamp):
        """Extract timestamp from record.

        Args:
            value: Record dict (already parsed from JSON)
            record_timestamp: Fallback timestamp
        """
        try:
            # value is already a dict from ParseJsonFunction
            pickup_dt = datetime.fromisoformat(value['tpep_pickup_datetime'])
            return int(pickup_dt.timestamp() * 1000)  # milliseconds
        except:
            return record_timestamp

def create_watermark_strategy():
    """Create watermark strategy with 30s idleness.

    Spec V1.9 Bug Fix: withIdleness(Duration.ofSeconds(30))
    Prevents watermark stalling when partitions have no data.
    """

    strategy = (
        WatermarkStrategy
        .for_bounded_out_of_orderness(Duration.of_seconds(10))
        .with_timestamp_assigner(TaxiTripTimestampAssigner())
        .with_idleness(Duration.of_seconds(30))  # V1.9 fix
    )

    return strategy
