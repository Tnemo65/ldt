"""
Schema validation for NYC Taxi records.
Spec: Lines 1570-1590 (Required fields, zone validation)
"""

from pyflink.datastream import FilterFunction


class SchemaValidator(FilterFunction):
    """Validate records against Avro schema constraints.

    V1.9 Spec:
    - Required fields: trip_distance, fare_amount, PU/DOLocationID, passenger_count
    - Zone ID range: 1-263 (NYC TLC zones)
    - Rejects records with null/missing required fields
    """

    def __init__(self):
        # Required fields per Avro schema
        self.required_fields = [
            'trip_distance',
            'fare_amount',
            'PULocationID',
            'DOLocationID',
            'passenger_count'
        ]

    def filter(self, value):
        """Validate record against schema constraints.

        Args:
            value: Record dict to validate

        Returns:
            True if valid (pass through), False if invalid (filter out)
        """
        if value is None:
            return False

        # Check required fields exist and are not null
        for field in self.required_fields:
            if field not in value or value[field] is None:
                return False

        # Validate zone IDs (1-263 per NYC TLC)
        try:
            pu_zone = int(value['PULocationID'])
            do_zone = int(value['DOLocationID'])

            if not (1 <= pu_zone <= 263):
                return False
            if not (1 <= do_zone <= 263):
                return False
        except (ValueError, TypeError):
            return False

        return True  # Valid record
