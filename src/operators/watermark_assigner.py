"""
Watermark assignment for event-time processing.

FIX: Use Event Time from tpep_pickup_datetime payload, not Processing Time.
Kafka stores taxi data from Jan-Feb 2024. We extract timestamps from the
tpep_pickup_datetime field so watermarks advance through real event time
(e.g., Jan 15 2024 -> Jan 16 2024 -> Feb 01 2024) as data arrives.

This ensures TumblingEventTimeWindows fire correctly as data is consumed,
rather than waiting for Processing Time or being blocked by stale watermarks.
"""

from pyflink.common import WatermarkStrategy, Duration
from pyflink.common.watermark_strategy import TimestampAssigner
from datetime import datetime
import logging

LOGGER = logging.getLogger(__name__)

# Supported datetime formats in tpep_pickup_datetime
_DATETIME_FORMATS = [
    '%Y-%m-%d %H:%M:%S',    # 2024-01-15 14:30:00
    '%Y-%m-%dT%H:%M:%S',    # 2024-01-15T14:30:00
    '%Y/%m/%d %H:%M:%S',    # 2024/01/15 14:30:00
    '%Y/%m/%dT%H:%M:%S',    # 2024/01/15T14:30:00
    '%m/%d/%Y %H:%M:%S',    # 01/15/2024 14:30:00
    '%m/%d/%Y %I:%M:%S %p', # 01/15/2024 02:30:00 PM
]


def _parse_tpep_datetime(value):
    """Parse tpep_pickup_datetime from a record dict, return epoch-ms or -1 on failure.

    Tries multiple common datetime formats. Returns -1 if all parsing fails,
    which causes Flink to assign a default timestamp.
    """
    if not isinstance(value, dict):
        return -1

    dt_str = value.get('tpep_pickup_datetime', '')
    if not dt_str:
        return -1

    dt_str = str(dt_str).strip()

    for fmt in _DATETIME_FORMATS:
        try:
            dt = datetime.strptime(dt_str, fmt)
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError):
            continue

    # Fallback: try ISO format withreplace
    try:
        dt = datetime.fromisoformat(dt_str.replace('/', '-'))
        return int(dt.timestamp() * 1000)
    except Exception:
        pass

    LOGGER.warning("[Watermark] Could not parse tpep_pickup_datetime: %r", dt_str)
    return -1


class TaxiTripTimestampAssigner(TimestampAssigner):
    """Extract event-time timestamp from tpep_pickup_datetime field.

    This enables TumblingEventTimeWindows to fire based on real event time
    from the taxi trip data (e.g., Jan-Feb 2024), not Processing Time.
    The watermark advances as records with increasing pickup times arrive,
    allowing windows to close and fire correctly.
    """

    def extract_timestamp(self, value, record_timestamp):
        """Return epoch-milliseconds from tpep_pickup_datetime."""
        ts = _parse_tpep_datetime(value)
        if ts > 0:
            return ts
        # Fallback to Kafka record timestamp if tpep_pickup_datetime is missing/invalid
        if record_timestamp and record_timestamp > 0:
            return int(record_timestamp)
        return int(datetime.now().timestamp() * 1000)


def create_watermark_strategy():
    """Create event-time watermark strategy from tpep_pickup_datetime.

    Uses for_bounded_out_of_orderness with a 5-second tolerance, which:
    1. Extracts timestamps from tpep_pickup_datetime (event time)
    2. Allows records up to 5 seconds late (common in streaming ingestion)
    3. Advances watermark = max(event_time_seen) - 5 seconds

    This ensures TumblingEventTimeWindows fire within seconds of data arrival,
    since taxi data spans Jan-Feb 2024 and watermarks will advance through
    those event times as the Kafka consumer processes records.
    """
    return (
        WatermarkStrategy
        .for_bounded_out_of_orderness(Duration.of_seconds(5))
        .with_timestamp_assigner(TaxiTripTimestampAssigner())
    )
