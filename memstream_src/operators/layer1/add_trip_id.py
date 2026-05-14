"""
AddTripIdFunction - Trip ID Generator.

Generates a composite trip ID using MD5 hash of key fields.
This ID is used for deduplication in downstream operators.

Composite key format: "VendorID|tpep_pickup_datetime|PULocationID|DOLocationID|fare_amount"

Reference: original_flow.md lines 393-398
"""

import hashlib
import logging
from typing import Dict, Optional

from pyflink.datastream import MapFunction

LOGGER = logging.getLogger('cadqstream.layer1')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


class AddTripIdFunction(MapFunction):
    """
    Add composite trip_id to each record.
    
    Composite key = "VendorID|tpep_pickup_datetime|PULocationID|DOLocationID|fare_amount"
    Hash: MD5 of composite.encode() -> 32-character hex string
    
    The trip_id is used as the key for deduplication.
    """
    
    def map(self, record: Dict) -> Dict:
        """
        Add trip_id to record.
        
        Args:
            record: Input dict with taxi fields
            
        Returns:
            Record with added 'trip_id' field
        """
        trip_id = generate_trip_id(record)
        return {
            **record,
            'trip_id': trip_id,
        }


def generate_trip_id(record: Dict) -> str:
    """
    Generate composite trip ID from record.
    
    Args:
        record: Dict with taxi fields
        
    Returns:
        32-character MD5 hex string
    """
    vendor_id = record.get('VendorID', '')
    pickup_dt = record.get('tpep_pickup_datetime', '')
    pu_loc = record.get('PULocationID', '')
    do_loc = record.get('DOLocationID', '')
    fare = record.get('fare_amount', '')
    
    composite_key = f"{vendor_id}|{pickup_dt}|{pu_loc}|{do_loc}|{fare}"
    
    trip_hash = hashlib.md5(composite_key.encode('utf-8')).hexdigest()
    
    return trip_hash


def generate_trip_id_sha256(record: Dict) -> str:
    """
    Alternative: Generate composite trip ID using SHA256.
    
    Args:
        record: Dict with taxi fields
        
    Returns:
        64-character SHA256 hex string
    """
    vendor_id = record.get('VendorID', '')
    pickup_dt = record.get('tpep_pickup_datetime', '')
    pu_loc = record.get('PULocationID', '')
    do_loc = record.get('DOLocationID', '')
    fare = record.get('fare_amount', '')
    
    composite_key = f"{vendor_id}|{pickup_dt}|{pu_loc}|{do_loc}|{fare}"
    
    trip_hash = hashlib.sha256(composite_key.encode('utf-8')).hexdigest()
    
    return trip_hash
