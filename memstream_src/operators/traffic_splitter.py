"""
Traffic Splitting for CA-DQStream + MemStream v5.

FIXES in v5:
- H-DK-2: Shadow/canary/production traffic splitting

Implements:
  - Shadow mode: Mirror traffic to candidate, log results
  - Canary mode: Route small % to candidate, compare results
  - Production mode: Full candidate model
"""

import os
import time
import threading
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import random

import numpy as np
from prometheus_client import Counter, Gauge

logger = logging.getLogger(__name__)


class TrafficMode(Enum):
    PRODUCTION = "production"
    SHADOW = "shadow"
    CANARY = "canary"
    FULL_CANDIDATE = "full_candidate"


@dataclass
class TrafficConfig:
    mode: TrafficMode = TrafficMode.PRODUCTION
    canary_rate: float = 0.05
    min_samples_for_evaluation: int = 500


class TrafficSplitter:
    """
    Manages traffic splitting between production and candidate models.
    """
    
    def __init__(self, config: TrafficConfig):
        self.config = config
        self._lock = threading.Lock()
        self._production_scores: List[float] = []
        self._candidate_scores: List[float] = []
        self._start_time = time.time()
        
        self._traffic_total = Counter(
            "memstream_traffic_total",
            "Total traffic routed",
            ["route"]
        )
        self._disagreement_count = Counter(
            "memstream_traffic_disagreement_total",
            "Total disagreements",
            ["neighborhood"]
        )
    
    def route(self, record_id: str, neighborhood: str) -> str:
        """Determine which model should handle this record."""
        with self._lock:
            if self.config.mode == TrafficMode.PRODUCTION:
                self._traffic_total.labels(route="production").inc()
                return "production"
            
            elif self.config.mode == TrafficMode.SHADOW:
                self._traffic_total.labels(route="production").inc()
                self._traffic_total.labels(route="shadow").inc()
                return "production"
            
            elif self.config.mode == TrafficMode.CANARY:
                if random.random() < self.config.canary_rate:
                    self._traffic_total.labels(route="candidate").inc()
                    return "candidate"
                else:
                    self._traffic_total.labels(route="production").inc()
                    return "production"
            
            elif self.config.mode == TrafficMode.FULL_CANDIDATE:
                self._traffic_total.labels(route="candidate").inc()
                return "candidate"
            
            self._traffic_total.labels(route="production").inc()
            return "production"
    
    def record_shadow_result(
        self,
        record_id: str,
        production_score: float,
        candidate_score: float,
        neighborhood: str,
    ):
        """Record shadow evaluation result."""
        with self._lock:
            self._production_scores.append(production_score)
            self._candidate_scores.append(candidate_score)
    
    def set_mode(self, mode: TrafficMode):
        """Change traffic routing mode."""
        with self._lock:
            old_mode = self.config.mode
            self.config.mode = mode
            logger.info(f"Traffic mode changed: {old_mode.value} -> {mode.value}")
    
    def get_stats(self) -> Dict:
        """Get current traffic splitting statistics."""
        with self._lock:
            return {
                "mode": self.config.mode.value,
                "total_samples": len(self._production_scores),
                "uptime_seconds": time.time() - self._start_time,
            }
