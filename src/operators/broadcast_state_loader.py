"""
Broadcast State Loader for Model Updates.
Spec: Task 2.7 (V1.9 Bug Fix: clear() before put())

Loads ML model artifacts into Broadcast State for hot-swappable updates.
Consumes from if-model-updates Kafka topic (compacted).
"""

from pyflink.datastream import BroadcastProcessFunction
from pyflink.datastream.state import MapStateDescriptor
from pyflink.common.typeinfo import Types
import pickle
import json
from pathlib import Path


class BroadcastStateLoaderFunction(BroadcastProcessFunction):
    """Load model updates into Broadcast State.

    V1.9 Critical Bug Fix:
    - MUST call broadcast_state.clear() before put()
    - Prevents stale state accumulation
    - Avoids memory leaks in long-running jobs

    Kafka Topic: if-model-updates (compacted, single partition)
    Message format: {
        "model_bytes": <base64>,
        "scaler_bytes": <base64>,
        "thresholds_json": <json_string>,
        "version": "v1.0.0",
        "timestamp": "2024-01-15T10:30:00"
    }
    """

    def __init__(self):
        """Initialize loader."""
        self.updates_processed = 0

    def process_broadcast_element(self, value, ctx):
        """Process model update from Kafka.

        Args:
            value: Model update message (dict or JSON string)
            ctx: Broadcast context
        """
        try:
            # Parse message if JSON string
            if isinstance(value, str):
                update = json.loads(value)
            else:
                update = value

            # Get broadcast state
            model_state_desc = MapStateDescriptor(
                "model_state",
                Types.STRING(),
                Types.PICKLED_BYTE_ARRAY()
            )

            broadcast_state = ctx.get_broadcast_state(model_state_desc)

            # V1.9 CRITICAL FIX: Clear before put()
            print(f"[BroadcastStateLoader] Clearing old state...")
            broadcast_state.clear()

            # Load model bytes
            model_bytes = update.get('model_bytes')
            if model_bytes:
                # Decode base64 if needed
                if isinstance(model_bytes, str):
                    import base64
                    model_bytes = base64.b64decode(model_bytes)

                broadcast_state.put("current_model", model_bytes)
                print(f"[BroadcastStateLoader] Model loaded ({len(model_bytes)} bytes)")

            # Load scaler bytes
            scaler_bytes = update.get('scaler_bytes')
            if scaler_bytes:
                if isinstance(scaler_bytes, str):
                    import base64
                    scaler_bytes = base64.b64decode(scaler_bytes)

                broadcast_state.put("scaler", scaler_bytes)
                print(f"[BroadcastStateLoader] Scaler loaded ({len(scaler_bytes)} bytes)")

            # Load thresholds JSON
            thresholds_json = update.get('thresholds_json')
            if thresholds_json:
                if isinstance(thresholds_json, dict):
                    thresholds_json = json.dumps(thresholds_json)

                broadcast_state.put("thresholds", thresholds_json.encode('utf-8'))
                print(f"[BroadcastStateLoader] Thresholds loaded")

            # Optional: Load neighborhood mapping
            neighborhood_mapping = update.get('neighborhood_mapping')
            if neighborhood_mapping:
                if isinstance(neighborhood_mapping, dict):
                    neighborhood_mapping = json.dumps(neighborhood_mapping)

                broadcast_state.put("neighborhood_mapping", neighborhood_mapping.encode('utf-8'))
                print(f"[BroadcastStateLoader] Neighborhood mapping loaded")

            # Track version
            version = update.get('version', 'unknown')
            timestamp = update.get('timestamp', 'unknown')

            self.updates_processed += 1

            print(f"[BroadcastStateLoader] ✅ Model update complete")
            print(f"  Version: {version}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Total updates: {self.updates_processed}")

        except Exception as e:
            print(f"[BroadcastStateLoader] ❌ ERROR processing update: {e}")
            import traceback
            traceback.print_exc()

    def process_element(self, value, ctx, out):
        """Process data stream element (pass through).

        Args:
            value: Data stream record
            ctx: Read-only broadcast context
            out: Output collector
        """
        # Just pass through - model is loaded from broadcast side
        out.collect(value)


def load_initial_model_to_broadcast_state(
    model_path: str = 'models/iforest_model_v2.pkl',
    scaler_path: str = 'models/scaler.pkl',
    thresholds_path: str = 'models/context_thresholds_v2.json',
    output_path: str = 'models/initial_model_update.json'
):
    """Create initial model update message for Kafka.

    This function is called BEFORE starting Flink job to prepare
    the initial model update message that will be consumed from Kafka.

    Args:
        model_path: Path to trained model
        scaler_path: Path to fitted scaler
        thresholds_path: Path to context thresholds
        output_path: Where to save the update message JSON

    Returns:
        Update message dict
    """
    import base64
    from datetime import datetime

    print("="*60)
    print("Creating Initial Model Update Message")
    print("="*60)

    # Load model
    print(f"\n1. Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model_bytes = f.read()

    model_bytes_b64 = base64.b64encode(model_bytes).decode('utf-8')
    print(f"   ✓ Model: {len(model_bytes)} bytes")

    # Load scaler
    print(f"\n2. Loading scaler: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler_bytes = f.read()

    scaler_bytes_b64 = base64.b64encode(scaler_bytes).decode('utf-8')
    print(f"   ✓ Scaler: {len(scaler_bytes)} bytes")

    # Load thresholds
    print(f"\n3. Loading thresholds: {thresholds_path}")
    with open(thresholds_path) as f:
        thresholds = json.load(f)

    thresholds_json = json.dumps(thresholds)
    print(f"   ✓ Thresholds: {len(thresholds.get('thresholds', {}))} contexts")

    # Create update message
    update_message = {
        "model_bytes": model_bytes_b64,
        "scaler_bytes": scaler_bytes_b64,
        "thresholds_json": thresholds_json,
        "version": "v1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

    # Save to file
    print(f"\n4. Saving update message: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(update_message, f, indent=2)

    print(f"   ✓ Saved ({Path(output_path).stat().st_size / 1e6:.1f} MB)")

    print(f"\n{'='*60}")
    print(f"✅ Initial Model Update Ready")
    print(f"{'='*60}")
    print(f"\nNext: Produce this message to Kafka topic 'if-model-updates'")
    print(f"  kafka-console-producer --topic if-model-updates --bootstrap-server localhost:9092")
    print(f"  < {output_path}")

    return update_message


if __name__ == "__main__":
    # Create initial model update message
    load_initial_model_to_broadcast_state()
