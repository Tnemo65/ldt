"""Register Avro schema with Schema Registry."""

import requests
import json
from pathlib import Path

def register_schema(schema_path: Path, subject: str, registry_url: str):
    """Register Avro schema with Schema Registry."""

    # Load schema
    with open(schema_path) as f:
        schema = json.load(f)

    # Prepare payload
    payload = {
        "schema": json.dumps(schema)
    }

    # Register
    url = f"{registry_url}/subjects/{subject}/versions"
    response = requests.post(url, json=payload, headers={"Content-Type": "application/vnd.schemaregistry.v1+json"})

    if response.status_code == 200:
        schema_id = response.json()['id']
        print(f"✅ Registered schema: {subject} (ID: {schema_id})")
        return schema_id
    else:
        print(f"❌ Failed: {response.status_code}")
        print(response.text)
        raise Exception("Schema registration failed")

def main():
    schema_path = Path("schemas/taxi_trip_v1.avsc")
    subject = "taxi-nyc-raw-value"
    registry_url = "http://localhost:8081"

    print(f"Registering schema: {schema_path}")
    schema_id = register_schema(schema_path, subject, registry_url)

    # Verify
    response = requests.get(f"{registry_url}/subjects/{subject}/versions/latest")
    print(f"\nVerification:")
    print(f"  Subject: {subject}")
    print(f"  Version: {response.json()['version']}")
    print(f"  ID: {response.json()['id']}")

if __name__ == "__main__":
    main()
