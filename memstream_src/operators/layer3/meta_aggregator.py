"""
MetaAggregator - Layer 3 Meta-Metrics Calculation.

Computes meta-metrics per neighborhood using 5-minute tumbling windows.
Outputs to Kafka dq-meta-stream.

Metrics computed:
- volume: number of records
- null_rate: proportion of null fields
- violation_rate: proportion of canary violations
- anomaly_rate: proportion of ML anomalies
- avg_anomaly_score: mean anomaly scores
- delta_score: |violation_rate - anomaly_rate| / (violation_rate + anomaly_rate + epsilon)

Reference: original_flow.md lines 113-116
"""

import logging
from typing import Dict, Iterable, List, Optional
from datetime import datetime

from pyflink.datastream import ProcessFunction
from pyflink.datastream.window import TumblingEventTimeWindows
from pyflink.datastream.time_characteristic import TimeCharacteristic
from pyflink.common.time import Time
from pyflink.common.typeinfo import Types, RowTypeInfo

LOGGER = logging.getLogger('cadqstream.layer3')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


DEFAULT_WINDOW_SIZE_MINUTES = 5
EPSILON = 1e-10


class MetaAggregator:
    """
    Meta-aggregator for neighborhood-level metrics.
    
    Computes 5-minute windowed statistics per neighborhood.
    """
    
    def __init__(self, window_size_minutes: int = DEFAULT_WINDOW_SIZE_MINUTES):
        """
        Initialize meta aggregator.
        
        Args:
            window_size_minutes: Window size in minutes (default: 5)
        """
        self.window_size_minutes = window_size_minutes
    
    def calculate_delta_score(
        self,
        violation_rate: float,
        anomaly_rate: float
    ) -> float:
        """
        Calculate delta score.
        
        delta_score = |violation_rate - anomaly_rate| / (violation_rate + anomaly_rate + epsilon)
        
        This measures the disagreement between canary and ML.
        High delta_score suggests concept drift or model degradation.
        """
        denominator = violation_rate + anomaly_rate + EPSILON
        return abs(violation_rate - anomaly_rate) / denominator
    
    def calculate_null_rate(self, record: Dict) -> float:
        """
        Calculate null rate for a record.
        
        Returns proportion of fields that are null or missing.
        """
        if not record:
            return 1.0
        
        null_count = 0
        total_fields = 0
        
        key_fields = [
            'trip_distance', 'fare_amount', 'PULocationID',
            'DOLocationID', 'passenger_count', 'tpep_pickup_datetime'
        ]
        
        for field in key_fields:
            total_fields += 1
            value = record.get(field)
            if value is None or value == '':
                null_count += 1
        
        return null_count / total_fields if total_fields > 0 else 0.0
    
    def aggregate(self, records: List[Dict]) -> Dict:
        """
        Aggregate records into meta-metrics.
        
        Args:
            records: List of records in window
            
        Returns:
            Meta-metrics dict
        """
        if not records:
            return self._empty_metrics()
        
        total = len(records)
        
        violation_count = 0
        anomaly_count = 0
        sum_anomaly_score = 0.0
        sum_null_rate = 0.0
        neighborhoods = set()
        
        for record in records:
            canary_violations = record.get('canary_violations', [])
            if canary_violations and len(canary_violations) > 0:
                violation_count += 1
            
            if record.get('is_anomaly', False):
                anomaly_count += 1
            
            sum_anomaly_score += record.get('anomaly_score', 0.0)
            sum_null_rate += self.calculate_null_rate(record)
            
            neighborhood = record.get('neighborhood', record.get('context_key', 'unknown'))
            neighborhoods.add(neighborhood)
        
        violation_rate = violation_count / total
        anomaly_rate = anomaly_count / total
        avg_anomaly_score = sum_anomaly_score / total
        avg_null_rate = sum_null_rate / total
        delta_score = self.calculate_delta_score(violation_rate, anomaly_rate)
        
        return {
            'volume': total,
            'null_rate': avg_null_rate,
            'violation_rate': violation_rate,
            'anomaly_rate': anomaly_rate,
            'avg_anomaly_score': avg_anomaly_score,
            'delta_score': delta_score,
            'neighborhood': self._get_primary_neighborhood(neighborhoods),
            'window_size_minutes': self.window_size_minutes,
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure."""
        return {
            'volume': 0,
            'null_rate': 0.0,
            'violation_rate': 0.0,
            'anomaly_rate': 0.0,
            'avg_anomaly_score': 0.0,
            'delta_score': 0.0,
            'neighborhood': 'unknown',
            'window_size_minutes': self.window_size_minutes,
        }
    
    def _get_primary_neighborhood(self, neighborhoods: set) -> str:
        """Get primary neighborhood from set."""
        if not neighborhoods:
            return 'unknown'
        if len(neighborhoods) == 1:
            return list(neighborhoods)[0]
        return 'mixed'


class MetaAggregatorProcessWindowFunction(ProcessFunction):
    """
    ProcessWindowFunction for meta-aggregation.
    
    Keyed by neighborhood, computes 5-minute windowed metrics.
    """
    
    def __init__(self, window_size_minutes: int = DEFAULT_WINDOW_SIZE_MINUTES):
        """
        Initialize window function.
        
        Args:
            window_size_minutes: Window size in minutes
        """
        self.window_size_minutes = window_size_minutes
        self.aggregator = MetaAggregator(window_size_minutes)
        
        self._window_count = 0
        self._total_records = 0
    
    def open(self, runtime_context):
        """Initialize counters."""
        self._window_count = 0
        self._total_records = 0
    
    def process(self, key: str, context: ProcessFunction.Context) -> Iterable[Dict]:
        """
        Process window elements and emit meta-metrics.
        
        Args:
            key: Neighborhood key
            context: ProcessFunction context
            
        Yields:
            Meta-metrics record
        """
        elements = list(context.elements())
        
        self._window_count += 1
        self._total_records += len(elements)
        
        metrics = self.aggregator.aggregate(elements)
        
        window_start = context.window().get_start()
        window_end = context.window().get_end()
        
        meta_record = {
            'neighborhood': key,
            'window_start': window_start,
            'window_end': window_end,
            **metrics,
            'timestamp': datetime.utcnow().isoformat(),
        }
        
        LOGGER.info(
            "[MetaAggregator] Window %d: neighborhood=%s, volume=%d, "
            "anomaly_rate=%.4f, delta_score=%.4f",
            self._window_count, key, metrics['volume'],
            metrics['anomaly_rate'], metrics['delta_score']
        )
        
        yield meta_record
    
    def close(self):
        """Log final statistics."""
        LOGGER.info(
            "[MetaAggregator] Total: windows=%d, records=%d",
            self._window_count, self._total_records
        )


class MetaAggregatorFactory:
    """Factory for creating MetaAggregator instances."""
    
    @staticmethod
    def create(
        window_size_minutes: int = DEFAULT_WINDOW_SIZE_MINUTES
    ) -> MetaAggregatorProcessWindowFunction:
        """
        Create MetaAggregatorProcessWindowFunction.
        
        Args:
            window_size_minutes: Window size in minutes
            
        Returns:
            Configured MetaAggregatorProcessWindowFunction
        """
        return MetaAggregatorProcessWindowFunction(window_size_minutes)
    
    @staticmethod
    def create_with_window(
        window_size_minutes: int = DEFAULT_WINDOW_SIZE_MINUTES
    ):
        """
        Create windowed MetaAggregator.
        
        Args:
            window_size_minutes: Window size in minutes
            
        Returns:
            Tuple of (window function, window assignment)
        """
        window_func = MetaAggregatorProcessWindowFunction(window_size_minutes)
        window = TumblingEventTimeWindows.of(Time.minutes(window_size_minutes))
        return window_func, window


class NeighborhoodStats:
    """
    Per-neighborhood statistics tracker.
    
    Tracks running statistics for drift detection.
    """
    
    def __init__(self, neighborhood: str):
        self.neighborhood = neighborhood
        self.windows = []
        self.max_history = 100
    
    def add_window(self, metrics: Dict):
        """Add window metrics to history."""
        self.windows.append(metrics)
        if len(self.windows) > self.max_history:
            self.windows.pop(0)
    
    def get_trend(self, metric_name: str) -> Optional[str]:
        """
        Get trend for a metric.
        
        Returns: 'increasing', 'decreasing', or None
        """
        if len(self.windows) < 3:
            return None
        
        values = [w.get(metric_name, 0.0) for w in self.windows[-3:]]
        
        if values[2] > values[1] > values[0]:
            return 'increasing'
        elif values[2] < values[1] < values[0]:
            return 'decreasing'
        
        return None
    
    def get_anomaly_rate_ma(self, windows: int = 5) -> float:
        """Get moving average of anomaly rate."""
        if len(self.windows) == 0:
            return 0.0
        
        recent = self.windows[-windows:]
        return sum(w.get('anomaly_rate', 0.0) for w in recent) / len(recent))
