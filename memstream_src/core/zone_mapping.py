"""
Zone Mapping for NYC Taxi Data.

Defines 10 neighborhoods, rush hour windows, and zone-to-neighborhood mappings
for the Context-Aware MemStream anomaly detection system (v10).

Reference: benchmark_v10.py neighborhood definitions
"""

from typing import Dict, Set


# =============================================================================
# NYC Taxi Zone Definitions (TLC Zones 1-265)
# =============================================================================

# Manhattan: Zones 1-43
MANHATTAN_ZONES: Set[int] = set(range(1, 44))

# Brooklyn: Zones 104-127
BROOKLYN_ZONES: Set[int] = set(range(104, 128))

# Bronx: Zones 44-103
BRONX_ZONES: Set[int] = set(range(44, 104))

# Staten Island: Zones 162-181
STATEN_ISLAND_ZONES: Set[int] = set(range(162, 182))

# EWR (Newark Airport): Zones 182-196
EWR_ZONES: Set[int] = set(range(182, 197))

# NALP (Nassau/Westchester): Zones 230-234
NALP_ZONES: Set[int] = set(range(230, 235))

# Unknown/External: Zones 235-265 and 197-216
UNKNOWN_ZONES: Set[int] = set(range(235, 266)) | set(range(197, 217))

# JFK: Zones 217-229
JFK_ZONES: Set[int] = set(range(217, 230))

# Queens Lower: Zones 128-148
QUEENS_LOWER_ZONES: Set[int] = set(range(128, 149))

# Queens Upper: Zones 149-161 (includes LaGuardia area 120+)
QUEENS_UPPER_ZONES: Set[int] = set(range(149, 162))


# =============================================================================
# Neighborhood Definitions (10 neighborhoods)
# =============================================================================

NEIGHBORHOODS: list = [
    'manhattan',       # 0: Zones 1-43
    'brooklyn',        # 1: Zones 104-127
    'queens_lower',    # 2: Zones 128-148
    'queens_upper',    # 3: Zones 149-161
    'bronx',          # 4: Zones 44-103
    'staten_island',  # 5: Zones 162-181
    'ewr',            # 6: Newark Airport Zones 182-196
    'jfk',            # 7: JFK Airport Zones 217-229
    'nalp',           # 8: Nassau/Westchester Zones 230-234
    'unknown',        # 9: Zones 235-265, 197-216
]

# Neighborhood to zones mapping
NEIGHBORHOOD_ZONES: Dict[str, Set[int]] = {
    'manhattan': MANHATTAN_ZONES,
    'brooklyn': BROOKLYN_ZONES,
    'queens_lower': QUEENS_LOWER_ZONES,
    'queens_upper': QUEENS_UPPER_ZONES,
    'bronx': BRONX_ZONES,
    'staten_island': STATEN_ISLAND_ZONES,
    'ewr': EWR_ZONES,
    'jfk': JFK_ZONES,
    'nalp': NALP_ZONES,
    'unknown': UNKNOWN_ZONES,
}

# Zone to neighborhood mapping (inverse)
ZONE_TO_NEIGHBORHOOD: Dict[int, str] = {}
for neighborhood, zones in NEIGHBORHOOD_ZONES.items():
    for zone in zones:
        ZONE_TO_NEIGHBORHOOD[zone] = neighborhood

# Neighborhood index mapping (for array-based lookups)
NEIGHBORHOOD_TO_ID: Dict[str, int] = {nb: i for i, nb in enumerate(NEIGHBORHOODS)}
ID_TO_NEIGHBORHOOD: Dict[int, str] = {i: nb for i, nb in enumerate(NEIGHBORHOODS)}


# =============================================================================
# Context Cell Definitions
# =============================================================================

# Context cell ID = (is_special << 2) | (is_night << 1) | is_weekend
# 8 context cells: (Standard/Special) x (Day/Night) x (Weekday/Weekend)
CONTEXT_CELLS: list = [
    'std_day_weekday',    # 0: Standard, Day, Weekday
    'std_night_weekday',  # 1: Standard, Night, Weekday
    'std_day_weekend',    # 2: Standard, Day, Weekend
    'std_night_weekend',  # 3: Standard, Night, Weekend
    'sp_day_weekday',     # 4: Special, Day, Weekday
    'sp_night_weekday',   # 5: Special, Night, Weekday
    'sp_day_weekend',     # 6: Special, Day, Weekend
    'sp_night_weekend',   # 7: Special, Night, Weekend
]

CONTEXT_CELL_BUCKETS: int = len(CONTEXT_CELLS)  # 8


# =============================================================================
# Rush Hour Definitions
# =============================================================================

# Morning rush hour: 6:00 AM - 10:00 AM
MORNING_RUSH_START: int = 6
MORNING_RUSH_END: int = 10

# Evening rush hour: 5:00 PM - 9:00 PM
EVENING_RUSH_START: int = 17
EVENING_RUSH_END: int = 21

# Midday: 10:00 AM - 5:00 PM
MIDDAY_START: int = 10
MIDDAY_END: int = 17

# Night: 9:00 PM - 6:00 AM (hour >= 20 OR hour <= 6)
NIGHT_START: int = 20
NIGHT_END: int = 6

# Rush hour hours
RUSH_HOURS: Set[int] = set(range(MORNING_RUSH_START, MORNING_RUSH_END)) | set(range(EVENING_RUSH_START, EVENING_RUSH_END))


# =============================================================================
# Hour Bucket Definitions
# =============================================================================

HOUR_BUCKETS: list = ['morning_rush', 'midday', 'evening_rush', 'night']

HOUR_TO_BUCKET: Dict[int, str] = {}
for h in range(24):
    if MORNING_RUSH_START <= h < MORNING_RUSH_END:
        HOUR_TO_BUCKET[h] = 'morning_rush'
    elif MIDDAY_START <= h < MIDDAY_END:
        HOUR_TO_BUCKET[h] = 'midday'
    elif EVENING_RUSH_START <= h < EVENING_RUSH_END:
        HOUR_TO_BUCKET[h] = 'evening_rush'
    else:
        HOUR_TO_BUCKET[h] = 'night'


# =============================================================================
# Day Type Definitions
# =============================================================================

DAY_TYPES: list = ['weekday', 'weekend']

# Weekend days (Python weekday(): Monday=0, Sunday=6)
WEEKEND_DAYS: Set[int] = {5, 6}


# =============================================================================
# Trip Type Definitions
# =============================================================================

TRIP_TYPES: list = ['short', 'medium', 'long']

TRIP_SHORT_THRESHOLD: float = 2.0
TRIP_MEDIUM_THRESHOLD: float = 10.0


# =============================================================================
# Helper Functions
# =============================================================================

def location_to_neighborhood(loc_id: int) -> int:
    """
    Convert location ID to neighborhood index (0-9).
    
    Args:
        loc_id: TLC zone ID (1-265)
        
    Returns:
        Neighborhood index (0-9)
    """
    neighborhood = ZONE_TO_NEIGHBORHOOD.get(int(loc_id), 'unknown')
    return NEIGHBORHOOD_TO_ID.get(neighborhood, 9)


def get_neighborhood_from_zone(zone_id: int) -> str:
    """Get neighborhood name from zone ID."""
    return ZONE_TO_NEIGHBORHOOD.get(int(zone_id), 'unknown')


def get_neighborhood_id(zone_id: int) -> int:
    """Get neighborhood ID (0-9) from zone ID."""
    return location_to_neighborhood(zone_id)


def get_context_cell(is_special: bool, is_night: bool, is_weekend: bool) -> int:
    """
    Compute context cell ID from conditions.
    
    Context cell ID = (is_special << 2) | (is_night << 1) | is_weekend
    
    Args:
        is_special: True if RatecodeID > 1
        is_night: True if hour >= 20 OR hour <= 6
        is_weekend: True if dow >= 5
        
    Returns:
        Context cell ID (0-7)
    """
    return (int(is_special) << 2) | (int(is_night) << 1) | int(is_weekend)


def get_context_cell_name(cell_id: int) -> str:
    """Get context cell name from ID."""
    return CONTEXT_CELLS[cell_id] if 0 <= cell_id < len(CONTEXT_CELLS) else 'unknown'


def get_hour_bucket(hour: int) -> str:
    """Get hour bucket from hour of day."""
    return HOUR_TO_BUCKET.get(int(hour) % 24, 'night')


def is_night_hour(hour: int) -> bool:
    """Check if hour is night time (>=20 OR <=6)."""
    h = int(hour) % 24
    return h >= NIGHT_START or h <= NIGHT_END


def is_rush_hour(hour: int) -> bool:
    """Check if hour is during rush hour."""
    return int(hour) % 24 in RUSH_HOURS


def is_weekend(day_of_week: int) -> bool:
    """Check if day of week is weekend (Python weekday format)."""
    return int(day_of_week) % 7 in WEEKEND_DAYS


def is_special_rate(ratecode_id: float) -> bool:
    """Check if ratecode indicates special fare (RatecodeID > 1)."""
    return float(ratecode_id) > 1.0


def get_trip_type(distance: float) -> str:
    """Classify trip type based on distance."""
    d = float(distance)
    if d < TRIP_SHORT_THRESHOLD:
        return 'short'
    elif d < TRIP_MEDIUM_THRESHOLD:
        return 'medium'
    else:
        return 'long'


def get_zones_for_neighborhood(neighborhood: str) -> Set[int]:
    """Get all zones belonging to a neighborhood."""
    return NEIGHBORHOOD_ZONES.get(neighborhood, set())


def get_neighborhood_centroid(neighborhood: str) -> tuple:
    """Get approximate centroid coordinates for a neighborhood (lat, lon)."""
    centroids = {
        'manhattan': (40.7831, -73.9712),
        'brooklyn': (40.6782, -73.9442),
        'queens_lower': (40.7282, -73.7949),
        'queens_upper': (40.7580, -73.8455),  # Near LaGuardia
        'bronx': (40.8448, -73.8648),
        'staten_island': (40.5795, -74.1502),
        'ewr': (40.6895, -74.1745),            # Newark Airport
        'jfk': (40.6413, -73.7781),            # JFK Airport
        'nalp': (40.7569, -73.5314),           # Nassau area
        'unknown': (40.7128, -74.0060),        # NYC default
    }
    return centroids.get(neighborhood, (40.7128, -74.0060))


__all__ = [
    # Zone definitions
    'MANHATTAN_ZONES',
    'BROOKLYN_ZONES',
    'BRONX_ZONES',
    'STATEN_ISLAND_ZONES',
    'EWR_ZONES',
    'NALP_ZONES',
    'UNKNOWN_ZONES',
    'JFK_ZONES',
    'QUEENS_LOWER_ZONES',
    'QUEENS_UPPER_ZONES',
    # Neighborhoods
    'NEIGHBORHOODS',
    'NEIGHBORHOOD_ZONES',
    'ZONE_TO_NEIGHBORHOOD',
    'NEIGHBORHOOD_TO_ID',
    'ID_TO_NEIGHBORHOOD',
    # Context cells
    'CONTEXT_CELLS',
    'CONTEXT_CELL_BUCKETS',
    # Rush hours
    'MORNING_RUSH_START',
    'MORNING_RUSH_END',
    'EVENING_RUSH_START',
    'EVENING_RUSH_END',
    'MIDDAY_START',
    'MIDDAY_END',
    'NIGHT_START',
    'NIGHT_END',
    'RUSH_HOURS',
    # Hour buckets
    'HOUR_BUCKETS',
    'HOUR_TO_BUCKET',
    # Day types
    'DAY_TYPES',
    'WEEKEND_DAYS',
    # Trip types
    'TRIP_TYPES',
    'TRIP_SHORT_THRESHOLD',
    'TRIP_MEDIUM_THRESHOLD',
    # Helper functions
    'location_to_neighborhood',
    'get_neighborhood_from_zone',
    'get_neighborhood_id',
    'get_context_cell',
    'get_context_cell_name',
    'get_hour_bucket',
    'is_night_hour',
    'is_rush_hour',
    'is_weekend',
    'is_special_rate',
    'get_trip_type',
    'get_zones_for_neighborhood',
    'get_neighborhood_centroid',
]
