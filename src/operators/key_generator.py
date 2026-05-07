"""
Surrogate key generation with MurmurHash3.
Spec: Lines 1515-1535 (10-20x faster than MD5)
"""

import mmh3
import json

def generate_trip_id(record: dict) -> str:
    """Generate deterministic trip_id using MurmurHash3.

    Composite key: VendorID + pickup_datetime + PU/DO + fare

    Spec V1.9: MurmurHash3 (not MD5) for 10-20x speedup.
    """

    # Composite key components
    key_parts = [
        str(record.get('VendorID', '')),
        record.get('tpep_pickup_datetime', ''),
        str(record.get('PULocationID', '')),
        str(record.get('DOLocationID', '')),
        str(record.get('fare_amount', ''))
    ]

    # Concatenate
    composite = '|'.join(key_parts)

    # MurmurHash3 128-bit
    hash_bytes = mmh3.hash_bytes(composite.encode('utf-8'))

    # Convert to hex string (64 chars)
    trip_id = hash_bytes.hex()

    return trip_id
