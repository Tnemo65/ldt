"""Avro schema validation tests.
Spec: Lines 136, 176-178, 1446-1464
"""

import pytest
from pathlib import Path
from avro.schema import parse as avro_parse
from avro.io import DatumWriter, DatumReader, BinaryEncoder, BinaryDecoder
import io

@pytest.fixture
def taxi_schema():
    schema_path = Path(__file__).parent.parent.parent / 'schemas' / 'taxi_trip_v1.avsc'
    with open(schema_path) as f:
        return avro_parse(f.read())

def test_schema_file_exists():
    """Schema file must exist."""
    schema_path = Path(__file__).parent.parent.parent / 'schemas' / 'taxi_trip_v1.avsc'
    assert schema_path.exists(), "Schema file not found"

def test_valid_record_serialization(taxi_schema):
    """Valid record should serialize/deserialize."""
    record = {
        "VendorID": 1,
        "tpep_pickup_datetime": "2024-01-15T10:30:00",
        "tpep_dropoff_datetime": "2024-01-15T10:45:00",
        "passenger_count": 2,
        "trip_distance": 3.5,
        "RatecodeID": 1,
        "store_and_fwd_flag": "N",
        "PULocationID": 161,
        "DOLocationID": 230,
        "payment_type": 1,
        "fare_amount": 15.5,
        "extra": 0.0,
        "mta_tax": 0.5,
        "tip_amount": 3.0,
        "tolls_amount": 0.0,
        "improvement_surcharge": 0.3,
        "total_amount": 19.3,
        "congestion_surcharge": 2.5,
        "airport_fee": None
    }

    # Serialize
    writer = DatumWriter(taxi_schema)
    bytes_writer = io.BytesIO()
    encoder = BinaryEncoder(bytes_writer)
    writer.write(record, encoder)
    avro_bytes = bytes_writer.getvalue()

    assert len(avro_bytes) > 0

    # Deserialize
    reader = DatumReader(taxi_schema)
    bytes_reader = io.BytesIO(avro_bytes)
    decoder = BinaryDecoder(bytes_reader)
    result = reader.read(decoder)

    assert result['trip_distance'] == 3.5
    assert result['PULocationID'] == 161

def test_null_in_strict_field_fails(taxi_schema):
    """ML features must not accept NULL."""
    invalid = {
        "VendorID": 1,
        "tpep_pickup_datetime": "2024-01-15T10:30:00",
        "tpep_dropoff_datetime": "2024-01-15T10:45:00",
        "passenger_count": 2,
        "trip_distance": None,  # INVALID: strict field
        "PULocationID": 161,
        "DOLocationID": 230,
        "payment_type": 1,
        "fare_amount": 15.5,
        "total_amount": 19.3
    }

    writer = DatumWriter(taxi_schema)
    encoder = BinaryEncoder(io.BytesIO())

    with pytest.raises(Exception):
        writer.write(invalid, encoder)
