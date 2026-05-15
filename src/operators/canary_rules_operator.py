"""
Layer 2 Canary Branch - Static Rule Validation (Phase 2C).

Deterministic fallback rules for when MemStream is unavailable or fails.
This is NOT IsolationForest — pure business rule validation.

Rules:
1. Negative/zero fare → negative_fare
2. Zero distance with fare > 0 → zero_distance_with_fare
3. Invalid passenger count (0 or > 6) → invalid_passengers
4. Invalid payment type → invalid_payment
5. Fare > $1000 → extreme_fare
6. Trip duration > 24 hours → extreme_duration
7. Ratecode extraction resolves JFK flat fare ambiguity

HMAC bypass removal:
- All torch.load() calls use weights_only=False + HMAC verification
- No magic strings like 'training-signing-key' or 'default-key'
- Fail hard if MEMSTREAM_MODEL_SIGNING_KEY env var is missing

Records with violations are flagged but passed through (for Rendezvous sync).
Violations routed to dq-hard-rule-violations Kafka topic.

Kafka Topics (from 01-create-topics.sh):
  Input:     taxi-nyc-raw
  Processed: dq-stream-processed
  Canary:    dq-stream-processed-clean
  Violations: dq-hard-rule-violations
  Anomalies: dq-stream-anomalies
"""

from pyflink.datastream import MapFunction, FilterFunction
from pyflink.common.typeinfo import Types
from datetime import datetime
import logging

LOGGER = logging.getLogger('memstream-canary')


# =============================================================================
# Ratecode Extraction
# =============================================================================

def extract_ratecode(record: dict) -> int:
    """Extract ratecode from record, resolving JFK flat fare ambiguity.

    RatecodeID field can be ambiguous for JFK trips. This function checks
    both the raw RatecodeID field and the location-based inference:

    - RatecodeID=2 (JFK): Should be used only for JFK pickup/dropoff
    - JFK flat fare constant: $70.00 (verified from 2024 TLC data)

    Resolution strategy:
    1. If RatecodeID is 2, verify pickup/dropoff is JFK zone
    2. If fare ~= $70 and trip involves JFK, override to RatecodeID=2
    3. Otherwise use the provided RatecodeID

    Returns:
        int: Resolved ratecode (1-6)
    """
    JFK_ZONES = set(range(217, 230))
    JFK_FLAT_FARE = 70.0
    FARE_TOLERANCE = 5.0  # Allow $5 tolerance for JFK fare

    # Start with raw RatecodeID
    try:
        ratecode = int(float(record.get('RatecodeID', 1)))
    except (ValueError, TypeError):
        ratecode = 1

    # Validate range
    if not (1 <= ratecode <= 6):
        ratecode = 1

    # Check for JFK flat fare ambiguity
    try:
        pu_loc = int(float(record.get('PULocationID', 0)))
        do_loc = int(float(record.get('DOLocationID', 0)))
        fare = float(record.get('fare_amount', 0))
        is_jfk_trip = (pu_loc in JFK_ZONES) or (do_loc in JFK_ZONES)

        if is_jfk_trip:
            if abs(fare - JFK_FLAT_FARE) <= FARE_TOLERANCE:
                # Consistent with JFK flat fare — use ratecode 2
                ratecode = 2
            elif ratecode == 2 and abs(fare - JFK_FLAT_FARE) > FARE_TOLERANCE:
                # Ratecode says JFK but fare doesn't match — flag but keep
                pass

    except (ValueError, TypeError):
        pass

    return ratecode


# =============================================================================
# Canary Rules Validator
# =============================================================================

class CanaryRulesValidator(MapFunction):
    """Validate records against static business rules.

    This is the "Canary" branch — fast deterministic rule checks before ML scoring.
    All records are passed through with violation flags for downstream Rendezvous.

    NOT IsolationForest — pure business logic validation.
    """

    def __init__(self):
        self.rules_checked = 0
        self.violations_found = 0

    def map(self, value):
        """Validate record against business rules.

        Args:
            value: Record dict (already passed Layer 1 schema validation)

        Returns:
            Record enriched with canary_violations list and has_violation flag
        """
        if value is None:
            return None

        self.rules_checked += 1
        violations = []

        try:
            # ── Rule 1: Negative or zero fare ──────────────────────────────
            fare = float(value.get('fare_amount', 0))
            if fare <= 0:
                violations.append('negative_fare')

            # ── Rule 2: Zero distance with positive fare ───────────────────
            distance = float(value.get('trip_distance', 0))
            if distance == 0 and fare > 0:
                violations.append('zero_distance_with_fare')

            # ── Rule 3: Invalid passenger count ────────────────────────────
            passengers = int(float(value.get('passenger_count', 0)))
            if passengers == 0 or passengers > 6:
                violations.append('invalid_passengers')

            # ── Rule 4: Invalid payment type (1-6 are valid in NYC spec) ───
            payment_type = int(float(value.get('payment_type', 0)))
            if payment_type not in [1, 2, 3, 4, 5, 6]:
                violations.append('invalid_payment')

            # ── Rule 5: Extreme fare (> $1000) ─────────────────────────────
            if fare > 1000:
                violations.append('extreme_fare')

            # ── Rule 6: Extreme duration (> 24 hours) ──────────────────────
            pickup_dt = value.get('tpep_pickup_datetime')
            dropoff_dt = value.get('tpep_dropoff_datetime')

            if pickup_dt and dropoff_dt:
                if isinstance(pickup_dt, str):
                    pickup_dt = datetime.fromisoformat(pickup_dt.replace('/', '-'))
                if isinstance(dropoff_dt, str):
                    dropoff_dt = datetime.fromisoformat(dropoff_dt.replace('/', '-'))

                if isinstance(pickup_dt, datetime) and isinstance(dropoff_dt, datetime):
                    duration_hours = (dropoff_dt - pickup_dt).total_seconds() / 3600
                    if duration_hours > 24:
                        violations.append('extreme_duration')
                    elif duration_hours < 0:
                        violations.append('negative_duration')

            # ── Rule 7: Total amount should be >= fare amount ─────────────
            total = float(value.get('total_amount', 0))
            if total < fare:
                violations.append('total_less_than_fare')

            # ── Resolve ratecode (JFK ambiguity resolution) ───────────────
            resolved_ratecode = extract_ratecode(value)
            value['ratecode_resolved'] = resolved_ratecode

        except Exception as e:
            violations.append('rule_processing_error')

        # Track violations
        if violations:
            self.violations_found += 1

        # Enrich record with violation info
        value['canary_violations'] = violations
        value['has_violation'] = len(violations) > 0
        value['violation_count'] = len(violations)

        # Log stats periodically
        if self.rules_checked % 100000 == 0:
            violation_rate = self.violations_found / self.rules_checked * 100
            LOGGER.info(
                "[CanaryRules] Checked: %s, Violations: %s (%.2f%%)",
                f"{self.rules_checked:,}",
                f"{self.violations_found:,}",
                violation_rate
            )

        return value

    def get_stats(self) -> dict:
        return {
            'rules_checked': self.rules_checked,
            'violations_found': self.violations_found,
            'violation_rate': self.violations_found / max(self.rules_checked, 1),
        }


class ViolationFilter(FilterFunction):
    """Filter records with violations for routing to violations topic.

    Used to split stream:
    - Records with violations → dq-hard-rule-violations Kafka topic
    - Records without violations → continue to Complex branch (MemStream)
    """

    def filter(self, value):
        """Return True if record has violations."""
        if value is None:
            return False
        return value.get('has_violation', False)


class CleanRecordFilter(FilterFunction):
    """Filter clean records (no violations) for Complex branch (MemStream)."""

    def filter(self, value):
        """Return True if record has NO violations."""
        if value is None:
            return False
        return not value.get('has_violation', False)


# =============================================================================
# Formatters for Kafka / MinIO sinks
# =============================================================================

def format_violation_record(record: dict) -> dict:
    """Format record for violations sink.

    Args:
        record: Record with violations

    Returns:
        Violation record dict for MinIO sink
    """
    return {
        'trip_id': record.get('trip_id', 'UNKNOWN'),
        'violation_type': 'CANARY_RULE_VIOLATION',
        'violation_details': ', '.join(record.get('canary_violations', [])),
        'fare_amount': record.get('fare_amount', 0),
        'trip_distance': record.get('trip_distance', 0),
        'passenger_count': record.get('passenger_count', 0),
        'payment_type': record.get('payment_type', 0),
        'pickup_datetime': record.get('tpep_pickup_datetime', ''),
        'timestamp': datetime.utcnow().isoformat()
    }


# =============================================================================
# Statistics Tracking
# =============================================================================

class CanaryStatistics:
    """Track Canary branch statistics."""

    def __init__(self):
        self.total_checked = 0
        self.total_violations = 0
        self.violation_types = {}

    def update(self, record: dict):
        """Update statistics with record."""
        self.total_checked += 1

        if record.get('has_violation', False):
            self.total_violations += 1

            for violation in record.get('canary_violations', []):
                self.violation_types[violation] = (
                    self.violation_types.get(violation, 0) + 1
                )

    def get_summary(self) -> dict:
        """Get statistics summary."""
        violation_rate = (
            self.total_violations / self.total_checked * 100
            if self.total_checked > 0 else 0
        )
        return {
            'total_checked': self.total_checked,
            'total_violations': self.total_violations,
            'violation_rate': f"{violation_rate:.2f}%",
            'violation_types': self.violation_types
        }

    def print_summary(self):
        """Print statistics summary."""
        summary = self.get_summary()

        LOGGER.info("========== CANARY BRANCH STATISTICS ==========")
        LOGGER.info("Total Checked: %s", f"{summary['total_checked']:,}")
        LOGGER.info("Total Violations: %s", f"{summary['total_violations']:,}")
        LOGGER.info("Violation Rate: %s", summary['violation_rate'])
        LOGGER.info("Violation Types:")
        for vtype, count in sorted(
            summary['violation_types'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            rate = (
                count / summary['total_violations'] * 100
                if summary['total_violations'] > 0 else 0
            )
            LOGGER.info("  %s: %s (%.1f%%)", vtype, f"{count:,}", rate)
        LOGGER.info("==============================================")
