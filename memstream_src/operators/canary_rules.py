"""
Canary Rules for CA-MemStream-EIA v10.

Layer 1 static filter: 7 rules that detect obviously anomalous records.
Returns True if record is CLEAN (passes all rules).
Returns False if any rule triggers (anomaly detected).

Reference: benchmark_v10.py check_canary() function
"""

from typing import Dict

import numpy as np


# =============================================================================
# Canary Rules Configuration
# =============================================================================

# Fare thresholds
MIN_FARE = 0.0
MAX_FARE = 500.0

# Distance thresholds
MIN_DISTANCE = 0.0
MAX_DISTANCE = 100.0

# Duration thresholds (minutes)
MIN_DURATION = 0.0
MAX_DURATION = 360.0

# Speed thresholds (mph)
MIN_SPEED = 0.0
MAX_SPEED = 80.0

# JFK flat fare (verified from 2024 TLC data)
JFK_FLAT_FARE = 70.0
JFK_FARE_TOLERANCE = 5.0  # Allow $5 variation for tolls/surcharges

# Airport zone IDs
AIRPORT_ZONES = {132, 138, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229}


# =============================================================================
# Canary Rules
# =============================================================================

def rule_1_fare_bounds(record: Dict) -> bool:
    """
    Rule 1: Fare amount must be within reasonable bounds.
    
    Returns True (pass) if fare is valid.
    """
    fare = record.get('fare_amount', 0)
    if fare < MIN_FARE or fare > MAX_FARE:
        return False
    return True


def rule_2_distance_bounds(record: Dict) -> bool:
    """
    Rule 2: Trip distance must be within reasonable bounds.
    
    Returns True (pass) if distance is valid.
    """
    dist = record.get('trip_distance', 0)
    if dist < MIN_DISTANCE or dist > MAX_DISTANCE:
        return False
    return True


def rule_3_positive_passenger_count(record: Dict) -> bool:
    """
    Rule 3: Passenger count must be positive.
    
    Returns True (pass) if passenger count > 0.
    """
    pax = record.get('passenger_count', 1)
    if pax <= 0 or pax > 9:
        return False
    return True


def rule_4_duration_bounds(record: Dict) -> bool:
    """
    Rule 4: Trip duration must be within reasonable bounds.
    
    Returns True (pass) if duration is valid.
    """
    try:
        pickup = record.get('tpep_pickup_datetime', '')
        dropoff = record.get('tpep_dropoff_datetime', '')
        if pickup and dropoff:
            from datetime import datetime
            formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S UTC', '%Y/%m/%d %H:%M:%S',
                      '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %I:%M:%S %p']
            p_dt, d_dt = None, None
            for fmt in formats:
                try:
                    p_dt = datetime.strptime(pickup.strip(), fmt)
                    break
                except ValueError:
                    continue
            for fmt in formats:
                try:
                    d_dt = datetime.strptime(dropoff.strip(), fmt)
                    break
                except ValueError:
                    continue
            if p_dt and d_dt:
                duration = (d_dt - p_dt).total_seconds() / 60
                if duration < MIN_DURATION or duration > MAX_DURATION:
                    return False
    except (ValueError, TypeError):
        pass
    return True


def rule_5_speed_bounds(record: Dict) -> bool:
    """
    Rule 5: Average speed must be within reasonable bounds.
    
    Returns True (pass) if speed is valid.
    """
    dist = record.get('trip_distance', 0)
    try:
        pickup = record.get('tpep_pickup_datetime', '')
        dropoff = record.get('tpep_dropoff_datetime', '')
        if pickup and dropoff and dist > 0:
            from datetime import datetime
            formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S UTC', '%Y/%m/%d %H:%M:%S',
                      '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %I:%M:%S %p']
            p_dt, d_dt = None, None
            for fmt in formats:
                try:
                    p_dt = datetime.strptime(pickup.strip(), fmt)
                    break
                except ValueError:
                    continue
            for fmt in formats:
                try:
                    d_dt = datetime.strptime(dropoff.strip(), fmt)
                    break
                except ValueError:
                    continue
            if p_dt and d_dt:
                duration = (d_dt - p_dt).total_seconds() / 3600  # hours
                if duration > 0:
                    speed = dist / duration
                    if speed < MIN_SPEED or speed > MAX_SPEED:
                        return False
    except (ValueError, TypeError):
        pass
    return True


def rule_6_jfk_flat_fare(record: Dict) -> bool:
    """
    Rule 6: JFK flat fare must be approximately $70.
    
    JFK airport trips have a flat fare of $70 (verified 2024 data).
    Allow small tolerance for tolls/surcharges.
    
    Returns True (pass) if JFK fare is reasonable.
    """
    pu_loc = int(record.get('PULocationID', 0))
    do_loc = int(record.get('DOLocationID', 0))
    
    # Check if this is an airport trip
    is_airport = (pu_loc in AIRPORT_ZONES or do_loc in AIRPORT_ZONES)
    
    if is_airport:
        fare = record.get('fare_amount', 0)
        # JFK flat fare with tolerance
        if fare < JFK_FLAT_FARE - JFK_FARE_TOLERANCE:
            return False
    
    return True


def rule_7_credit_no_tip(record: Dict) -> bool:
    """
    Rule 7: Credit card payments (type=1) with zero tip are suspicious.
    
    Valid signal: credit card (type=1) with zero tip is genuinely suspicious.
    Cash payments (type=2) always have tip=0 in TLC data (not suspicious).
    
    Returns True (pass) if not suspicious, False if suspicious.
    """
    payment_type = record.get('payment_type', 0)
    tip_amount = record.get('tip_amount', 0)
    
    # Credit card (type=1) with zero tip is suspicious
    if payment_type == 1 and tip_amount == 0:
        # Only flag as suspicious if fare > $10 (cheap rides don't always tip)
        fare = record.get('fare_amount', 0)
        if fare > 10:
            return False
    
    return True


# =============================================================================
# Canary Check Function
# =============================================================================

def check_canary(record: Dict) -> bool:
    """
    Check if a record passes all 7 canary rules.
    
    Layer 1 static filter for obviously anomalous records.
    
    Args:
        record: Dict with NYC taxi fields
        
    Returns:
        True if record is CLEAN (passes all rules),
        False if any rule triggers (anomaly detected)
    """
    rules = [
        rule_1_fare_bounds,
        rule_2_distance_bounds,
        rule_3_positive_passenger_count,
        rule_4_duration_bounds,
        rule_5_speed_bounds,
        rule_6_jfk_flat_fare,
        rule_7_credit_no_tip,
    ]
    
    for rule in rules:
        if not rule(record):
            return False
    
    return True


def check_canary_rules(record: Dict) -> Dict[str, bool]:
    """
    Check all canary rules and return detailed results.
    
    Args:
        record: Dict with NYC taxi fields
        
    Returns:
        Dict mapping rule names to pass/fail results
    """
    return {
        'rule_1_fare_bounds': rule_1_fare_bounds(record),
        'rule_2_distance_bounds': rule_2_distance_bounds(record),
        'rule_3_positive_passenger_count': rule_3_positive_passenger_count(record),
        'rule_4_duration_bounds': rule_4_duration_bounds(record),
        'rule_5_speed_bounds': rule_5_speed_bounds(record),
        'rule_6_jfk_flat_fare': rule_6_jfk_flat_fare(record),
        'rule_7_credit_no_tip': rule_7_credit_no_tip(record),
    }


__all__ = [
    'check_canary',
    'check_canary_rules',
    'rule_1_fare_bounds',
    'rule_2_distance_bounds',
    'rule_3_positive_passenger_count',
    'rule_4_duration_bounds',
    'rule_5_speed_bounds',
    'rule_6_jfk_flat_fare',
    'rule_7_credit_no_tip',
    'JFK_FLAT_FARE',
]
