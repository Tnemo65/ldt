"""
Rendezvous Operator - Synchronize Canary + Complex Branches.
Task 3.6-3.10: CoProcessFunction with MapState inbox

Architecture:
- Two input streams: Canary (rule-based) + Complex (ML-based)
- MapState inbox per branch with 5-second TTL
- Merge records from both branches by trip_id
- Handle timeouts (one branch arrives, other doesn't)

Spec: Task 3.6-3.10 (Rendezvous sync pattern)
"""

from pyflink.datastream import CoProcessFunction
from pyflink.datastream.state import MapStateDescriptor, StateTtlConfig
from pyflink.common.time import Time
from pyflink.common.typeinfo import Types
from datetime import datetime


class RendezvousOperator(CoProcessFunction):
    """Synchronize Canary + Complex branches using Rendezvous pattern.

    Uses MapState inbox to buffer records from each branch until matching
    record from other branch arrives. 5-second TTL ensures memory efficiency.

    Flow:
    1. Record arrives from Canary branch → check if Complex already has it
       - If yes: merge and emit
       - If no: store in Canary inbox, set timer
    2. Record arrives from Complex branch → check if Canary already has it
       - If yes: merge and emit
       - If no: store in Complex inbox, set timer
    3. Timer fires (5s timeout) → emit partial record with missing branch flagged

    Spec: V1.9 Rendezvous architecture
    """

    def __init__(self):
        """Initialize operator."""
        self.canary_inbox = None
        self.complex_inbox = None
        self.stats = {
            'merged': 0,
            'canary_only': 0,
            'complex_only': 0,
            'canary_timeout': 0,
            'complex_timeout': 0
        }

    def open(self, runtime_context):
        """Initialize MapState inboxes with 5-second TTL."""

        # Canary inbox: trip_id → canary_record
        canary_desc = MapStateDescriptor(
            "canary_inbox",
            Types.STRING(),
            Types.PICKLED_BYTE_ARRAY()
        )

        # 5-second TTL (OnCreateAndWrite)
        ttl_config = (
            StateTtlConfig
            .new_builder(Time.seconds(5))
            .set_update_type(StateTtlConfig.UpdateType.OnCreateAndWrite)
            .set_state_visibility(StateTtlConfig.StateVisibility.NeverReturnExpired)
            .build()
        )

        canary_desc.enable_time_to_live(ttl_config)
        self.canary_inbox = runtime_context.get_map_state(canary_desc)

        # Complex inbox: trip_id → complex_record
        complex_desc = MapStateDescriptor(
            "complex_inbox",
            Types.STRING(),
            Types.PICKLED_BYTE_ARRAY()
        )
        complex_desc.enable_time_to_live(ttl_config)
        self.complex_inbox = runtime_context.get_map_state(complex_desc)

        print("[Rendezvous] Operator initialized with 5s TTL inboxes")

    def process_element1(self, canary_record, ctx):
        """Process Canary branch record (element1).

        Args:
            canary_record: Record from Canary branch with violation flags
            ctx: Context with timer service

        Yields:
            Merged record if Complex already arrived, else buffer in inbox
        """
        if canary_record is None:
            return

        trip_id = canary_record.get('trip_id')

        if trip_id is None:
            # Can't sync without trip_id - emit as canary-only
            yield self._create_merged_record(canary_record, None, 'canary_only_no_id')
            return

        # Check if Complex branch already has this record
        if self.complex_inbox.contains(trip_id):
            # Match found! Merge and emit
            complex_record = self.complex_inbox.get(trip_id)
            self.complex_inbox.remove(trip_id)

            merged = self._create_merged_record(canary_record, complex_record, 'merged')
            self.stats['merged'] += 1

            yield merged

        else:
            # No match yet - store in Canary inbox
            self.canary_inbox.put(trip_id, canary_record)

            # Register timer (5s from now)
            # If Complex doesn't arrive within 5s, onTimer will emit partial
            timer_timestamp = ctx.timestamp() + 5000  # 5 seconds
            ctx.timer_service().register_event_time_timer(timer_timestamp)

            self.stats['canary_only'] += 1

    def process_element2(self, complex_record, ctx):
        """Process Complex branch record (element2).

        Args:
            complex_record: Record from Complex branch with anomaly scores
            ctx: Context with timer service

        Yields:
            Merged record if Canary already arrived, else buffer in inbox
        """
        if complex_record is None:
            return

        trip_id = complex_record.get('trip_id')

        if trip_id is None:
            # Can't sync without trip_id - emit as complex-only
            yield self._create_merged_record(None, complex_record, 'complex_only_no_id')
            return

        # Check if Canary branch already has this record
        if self.canary_inbox.contains(trip_id):
            # Match found! Merge and emit
            canary_record = self.canary_inbox.get(trip_id)
            self.canary_inbox.remove(trip_id)

            merged = self._create_merged_record(canary_record, complex_record, 'merged')
            self.stats['merged'] += 1

            yield merged

        else:
            # No match yet - store in Complex inbox
            self.complex_inbox.put(trip_id, complex_record)

            # Register timer (5s from now)
            timer_timestamp = ctx.timestamp() + 5000
            ctx.timer_service().register_event_time_timer(timer_timestamp)

            self.stats['complex_only'] += 1

    def on_timer(self, timestamp, ctx):
        """Handle timer expiration (5s timeout).

        When timer fires, iterate through inbox and emit partial records
        (records that arrived from one branch but not the other within 5s).
        """
        from datetime import datetime

        # Emit timeouts from canary inbox
        if self.canary_inbox:
            for trip_id in list(self.canary_inbox.keys()):
                try:
                    canary_record = self.canary_inbox.get(trip_id)
                    if canary_record:
                        partial = self._create_merged_record(canary_record, None, 'canary_timeout')
                        self.stats['canary_timeout'] += 1
                        yield partial
                        self.canary_inbox.remove(trip_id)
                except Exception:
                    pass

        # Emit timeouts from complex inbox
        if self.complex_inbox:
            for trip_id in list(self.complex_inbox.keys()):
                try:
                    complex_record = self.complex_inbox.get(trip_id)
                    if complex_record:
                        partial = self._create_merged_record(None, complex_record, 'complex_timeout')
                        self.stats['complex_timeout'] += 1
                        yield partial
                        self.complex_inbox.remove(trip_id)
                except Exception:
                    pass

    def _create_merged_record(self, canary_record, complex_record, merge_status):
        """Merge records from both branches.

        Args:
            canary_record: Record from Canary branch (can be None)
            complex_record: Record from Complex branch (can be None)
            merge_status: 'merged', 'canary_only', 'complex_only', etc.

        Returns:
            Merged record dict
        """
        # Start with base record (prefer canary for base fields)
        if canary_record:
            merged = canary_record.copy()
        elif complex_record:
            merged = complex_record.copy()
        else:
            return None

        # Add Canary fields
        if canary_record:
            merged['canary_violations'] = canary_record.get('canary_violations', [])
            merged['has_violation'] = canary_record.get('has_violation', False)
        else:
            merged['canary_violations'] = []
            merged['has_violation'] = None  # Canary not checked

        # Add Complex fields
        if complex_record:
            merged['anomaly_score'] = complex_record.get('anomaly_score', -1.0)
            merged['threshold'] = complex_record.get('threshold', 0.0)
            merged['is_anomaly'] = complex_record.get('is_anomaly', False)
            merged['context_key'] = complex_record.get('context_key', 'unknown')
        else:
            merged['anomaly_score'] = None  # Complex not scored
            merged['threshold'] = None
            merged['is_anomaly'] = None
            merged['context_key'] = None

        # Add merge metadata
        merged['merge_status'] = merge_status
        merged['merge_timestamp'] = datetime.utcnow().isoformat()

        return merged

    def close(self):
        """Print statistics on close."""
        print("\n" + "="*60)
        print("RENDEZVOUS STATISTICS")
        print("="*60)
        print(f"Merged: {self.stats['merged']:,}")
        print(f"Canary Only: {self.stats['canary_only']:,}")
        print(f"Complex Only: {self.stats['complex_only']:,}")
        print(f"Canary Timeouts: {self.stats['canary_timeout']:,}")
        print(f"Complex Timeouts: {self.stats['complex_timeout']:,}")
        print("="*60)
