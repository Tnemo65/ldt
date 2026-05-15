#!/usr/bin/env python3
"""Register Avro schemas with Schema Registry."""
import json
import requests

SCHEMA_REGISTRY_URL = "http://schema-registry:8081"

schemas = {
    "taxi-nyc-raw-value": {
        "type": "record", "name": "TaxiNYCRaw", "namespace": "com.cadqstream.events",
        "fields": [
            {"name": "VendorID", "type": ["null", "int"], "default": None},
            {"name": "tpep_pickup_datetime", "type": "string"},
            {"name": "tpep_dropoff_datetime", "type": "string"},
            {"name": "passenger_count", "type": ["null", "double"], "default": None},
            {"name": "trip_distance", "type": ["null", "double"], "default": None},
            {"name": "RatecodeID", "type": ["null", "double"], "default": None},
            {"name": "store_and_fwd_flag", "type": "string"},
            {"name": "PULocationID", "type": ["null", "double"], "default": None},
            {"name": "DOLocationID", "type": ["null", "double"], "default": None},
            {"name": "payment_type", "type": ["null", "double"], "default": None},
            {"name": "fare_amount", "type": ["null", "double"], "default": None},
            {"name": "extra", "type": ["null", "double"], "default": None},
            {"name": "mta_tax", "type": ["null", "double"], "default": None},
            {"name": "tip_amount", "type": ["null", "double"], "default": None},
            {"name": "tolls_amount", "type": ["null", "double"], "default": None},
            {"name": "improvement_surcharge", "type": ["null", "double"], "default": None},
            {"name": "total_amount", "type": ["null", "double"], "default": None},
            {"name": "congestion_surcharge", "type": ["null", "double"], "default": None},
            {"name": "trip_duration", "type": ["null", "double"], "default": None},
            {"name": "speed_mph", "type": ["null", "double"], "default": None},
        ]
    },
    "dq-stream-processed-value": {
        "type": "record", "name": "DQStreamProcessed", "namespace": "com.cadqstream.events",
        "fields": [
            {"name": "trip_id", "type": "string"},
            {"name": "VendorID", "type": ["null", "int"], "default": None},
            {"name": "tpep_pickup_datetime", "type": "string"},
            {"name": "tpep_dropoff_datetime", "type": "string"},
            {"name": "passenger_count", "type": "int"},
            {"name": "trip_distance", "type": "double"},
            {"name": "PULocationID", "type": "int"},
            {"name": "DOLocationID", "type": "int"},
            {"name": "payment_type", "type": "int"},
            {"name": "fare_amount", "type": "double"},
            {"name": "total_amount", "type": "double"},
            {"name": "has_violation", "type": "boolean"},
            {"name": "canary_violations", "type": {"type": "array", "items": "string"}},
            {"name": "anomaly_score", "type": "double"},
            {"name": "is_anomaly", "type": "boolean"},
            {"name": "final_decision", "type": "string"},
            {"name": "neighborhood", "type": "string"},
            {"name": "_producer_ts", "type": ["null", "string"], "default": None}
        ]
    },
    "dq-meta-stream-value": {
        "type": "record", "name": "DQMetaStream", "namespace": "com.cadqstream.events",
        "fields": [
            {"name": "neighborhood", "type": "string"},
            {"name": "neighborhood_id", "type": "string"},
            {"name": "window_start", "type": "string"},
            {"name": "window_end", "type": "string"},
            {"name": "volume", "type": "long"},
            {"name": "null_rate", "type": "double"},
            {"name": "violation_rate", "type": "double"},
            {"name": "anomaly_rate", "type": "double"},
            {"name": "avg_anomaly_score", "type": "double"},
            {"name": "delta_score", "type": "double"}
        ]
    },
    "iec-action-replay-value": {
        "type": "record", "name": "IECActionReplay", "namespace": "com.cadqstream.events",
        "fields": [
            {"name": "scenario", "type": "string"},
            {"name": "neighborhood", "type": "string"},
            {"name": "metric_name", "type": "string"},
            {"name": "drift_indicator", "type": "string"},
            {"name": "drift_magnitude", "type": "double"},
            {"name": "neighborhood_count", "type": "int"},
            {"name": "strategy", "type": "string"},
            {"name": "iec_confidence", "type": "double"},
            {"name": "action_result", "type": {"type": "record", "name": "IECActionResult", "fields": [
                {"name": "action", "type": "string"},
                {"name": "message", "type": "string"},
                {"name": "new_threshold", "type": ["null", "double"], "default": None}
            ]}},
            {"name": "drifts_detected", "type": {"type": "array", "items": "string"}},
            {"name": "iec_timestamp", "type": "string"}
        ]
    }
}

for subject, schema in schemas.items():
    body = json.dumps({"schema": json.dumps(schema)})
    resp = requests.post(
        f"{SCHEMA_REGISTRY_URL}/subjects/{subject}/versions",
        headers={"Content-Type": "application/vnd.schemaregistry.v1+json"},
        data=body
    )
    status = resp.status_code
    if status in (200, 201):
        rid = resp.json().get("id")
        print(f"OK: {subject} -> ID {rid}")
    elif status == 409:
        print(f"SKIP: {subject} (already registered)")
    else:
        print(f"ERR: {subject}: {status} {resp.text}")

subjects = requests.get(f"{SCHEMA_REGISTRY_URL}/subjects").json()
print(f"All subjects: {subjects}")
