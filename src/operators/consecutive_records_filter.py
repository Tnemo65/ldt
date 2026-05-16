"""
Consecutive Records Filter — KeyedProcessFunction for unbounded streams.

Task 3: Drop strictly consecutive duplicate records keyed by meter_id.
Crucial rule: crawl_time (or equivalent timestamp) is STRICTLY IGNORED when
comparing consecutive records, because two identical trips published at
different crawl times are still duplicates.

This is distinct from DeduplicatorFunction which deduplicates by trip_id
(MurmurHash3 of VendorID + pickup_datetime + PU/DO + fare). The
ConsecutiveRecordsFilter deduplicates by meter_id (VendorID + PU/DO + fare
only — no timestamp), catching duplicates that slipped past trip_id dedup
due to slightly different pickup_datetime values.
"""

from pyflink.datastream import KeyedProcessFunction
from pyflink.datastream.state import ValueStateDescriptor, StateTtlConfig
from pyflink.common.time import Time
from pyflink.common.typeinfo import Types
import json
import logging

LOGGER = logging.getLogger("consecutive-records-filter")


def _extract_meter_key(record: dict) -> str:
    """Extract the deduplication key that ignores timestamps.

    Ignores:
      - tpep_pickup_datetime, tpep_dropoff_datetime (crawl_time equivalent)
      - trip_id (derived, not a source of duplication)
      - any _ts or _at suffix fields

    Includes (meter dimensions):
      - VendorID, PULocationID, DOLocationID, RatecodeID
      - passenger_count, trip_distance, payment_type
      - fare_amount, total_amount, tip_amount, tolls_amount

    Numeric fields are parsed as floats and formatted to 2 decimal places
    to prevent "17.92" vs "17.920" mismatches that cause 17% data loss.
    """

    def _normalize_numeric(value, default="") -> str:
        """Parse value to float and format with exactly 2 decimal places."""
        if value is None or value == "":
            return default
        try:
            return f"{float(value):.2f}"
        except (ValueError, TypeError):
            return default

    return "|".join([
        str(record.get("VendorID", "")),
        str(record.get("PULocationID", "")),
        str(record.get("DOLocationID", "")),
        str(record.get("RatecodeID", "")),
        str(record.get("passenger_count", "")),
        _normalize_numeric(record.get("trip_distance")),
        str(record.get("payment_type", "")),
        _normalize_numeric(record.get("fare_amount")),
        _normalize_numeric(record.get("total_amount")),
        _normalize_numeric(record.get("tip_amount")),
        _normalize_numeric(record.get("tolls_amount")),
    ])


def _to_dict(value):
    """Normalize input to dict. Handles JSON strings, bytes, and dicts."""
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


class ConsecutiveRecordsFilter(KeyedProcessFunction):
    """Drop consecutive duplicates keyed by meter_id using ValueState.

    Keyed on meter_id so each meter tracks its own consecutive duplicate chain.
    ValueState stores the last meter_key seen for that meter.
    On each record: if meter_key matches last meter_key -> drop (return None).
    Crucially: timestamp fields are excluded from meter_key so records that differ
    only in crawl_time are still caught as consecutive duplicates.

    State TTL: 1 hour per key to prevent unbounded state growth.
    """

    def open(self, runtime_context):
        descriptor = ValueStateDescriptor(
            "meter_key",
            Types.STRING(),
        )
        ttl_config = (
            StateTtlConfig
            .new_builder(Time.minutes(60))
            .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite)
            .set_state_visibility(StateTtlConfig.StateVisibility.NeverReturnExpired)
            .build()
        )
        descriptor.enable_time_to_live(ttl_config)
        self._last_key_state = runtime_context.get_state(descriptor)
        self._dropped = 0
        self._seen = 0

    def process_element(self, value, ctx: KeyedProcessFunction.Context):
        """Evaluate whether this record is a consecutive duplicate.

        Args:
            value: Record dict or JSON string
            ctx: Flink keyed context (provides timestamp via ctx.timestamp())

        Returns:
            The record if NOT a consecutive duplicate.
            None ONLY for legitimate consecutive duplicates (dedup intent).
            Sentinel dict for None/malformed input (DLQ routing).
        """
        if value is None:
            return {
                '_dlq': True,
                '_dlq_reason': 'null_input',
                '_dlq_category': 'VALIDATION_ERROR',
                '_dlq_operator': 'ConsecutiveRecordsFilter',
                'trip_id': 'dlq_null',
                'tpep_pickup_datetime': '',
            }

        record = _to_dict(value)
        if record is None:
            return {
                '_dlq': True,
                '_dlq_reason': 'malformed_json',
                '_dlq_category': 'PARSE_ERROR',
                '_dlq_operator': 'ConsecutiveRecordsFilter',
                'trip_id': 'dlq_malformed',
                'tpep_pickup_datetime': '',
            }

        self._seen += 1
        current_key = _extract_meter_key(record)

        try:
            last_key = self._last_key_state.value()
        except Exception:
            last_key = None

        if last_key is not None and current_key == last_key:
            self._dropped += 1
            if self._dropped % 5000 == 0:
                LOGGER.info(
                    "[ConsecutiveRecordsFilter] seen=%d dropped=%d (%.2f%%) key=%s",
                    self._seen, self._dropped,
                    self._dropped / max(self._seen, 1) * 100,
                    current_key[:32],
                )
            return None

        self._last_key_state.update(current_key)
        return value
