"""
Shared Neighborhood Mapping for CA-DQStream.

All zone-to-neighborhood lookups MUST use this module to ensure consistency
across operators. Inconsistent mappings cause records to be classified
differently by MetaAggregator (Layer 3) and MemStreamScoringOperator (Layer 2).

Zone IDs follow TLC (Taxi & Limousine Commission) numbering for NYC.
See: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Mappings (consistent across all operators):
    manhattan:       1-43
    bronx:          44-103
    brooklyn:      104-127
    queens_lower:  128-148
    queens_upper:  149-161
    staten_island: 162-263
    ewr:              1 (Newark, DOLocationID=EWR but PULocationID=1)
    jfk:           2-6  (JFK special handling via ratecode + zone)
    nalp:           ? (Not a location / water)
    unknown:        ? (fallback)

Index order matches NEIGHBORHOOD_NAMES order:
    0=manhattan, 1=brooklyn, 2=queens_lower, 3=queens_upper,
    4=bronx, 5=staten_island, 6=ewr, 7=jfk, 8=nalp, 9=unknown
"""

from typing import Dict

# Ordered list matching the index used by MemStreamCore ContextBeta
# DO NOT change order — ContextBeta uses hardcoded index->context mapping.
NEIGHBORHOOD_NAMES = [
    'manhattan', 'brooklyn', 'queens_lower', 'queens_upper',
    'bronx', 'staten_island', 'ewr', 'jfk', 'nalp', 'unknown'
]
N_NEIGHBORHOODS = 10


def get_neighborhood_idx(zone_id: int) -> int:
    """Map PULocationID (zone_id) to neighborhood index (0-9).

    Index order matches NEIGHBORHOOD_NAMES above.
    Must be consistent with MemStreamScoringOperator.location_to_neighborhood_idx.
    """
    if zone_id <= 0:
        return 9  # unknown

    z = int(zone_id)
    if 1 <= z <= 43:
        return 0   # manhattan
    elif 44 <= z <= 103:
        return 4   # bronx
    elif 104 <= z <= 127:
        return 1   # brooklyn
    elif 128 <= z <= 148:
        return 2   # queens_lower
    elif 149 <= z <= 161:
        return 3   # queens_upper
    elif 162 <= z <= 263:
        return 5   # staten_island
    else:
        return 9   # unknown


def get_neighborhood_name(zone_id: int) -> str:
    """Map PULocationID (zone_id) to neighborhood name string."""
    return NEIGHBORHOOD_NAMES[get_neighborhood_idx(zone_id)]
