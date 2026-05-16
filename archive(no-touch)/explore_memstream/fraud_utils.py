#!/usr/bin/env python3
"""
Fraud/Anomaly injection utilities cho MemStream evaluation.

Tái sử dụng cho tất cả eval scripts.

Usage:
    from fraud_utils import inject_anomalies, FraudType
    
    df_test, labels = inject_anomalies(df, rate=0.03, seed=42)
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Tuple, Optional, List
from dataclasses import dataclass


# =============================================================================
# Fraud Types
# =============================================================================

class FraudType(Enum):
    """Các loại fraud có thể inject."""
    
    # Type 1: Short-trip meter fraud (low distance, inflated fare)
    SHORT_TRIP = "short_trip"
    
    # Type 2: Long-trip undercharge (high distance, low fare)
    LONG_TRIP = "long_trip"
    
    # Type 3: Ratecode mismatch
    RATECODE_MISMATCH = "ratecode_mismatch"
    
    # Type 4: Night trip anomaly
    NIGHT_FRAUD = "night"
    
    # Combined
    MIXED = "mixed"


@dataclass
class FraudConfig:
    """Configuration cho fraud injection."""
    
    # Injection parameters
    anomaly_rate: float = 0.03  # 3% anomaly rate
    fraud_type: FraudType = FraudType.MIXED
    
    # Type 1: Short-trip parameters
    short_trip_max_dist: float = 1.0  # miles
    short_trip_min_fare: float = 15.0  # minimum fare to set
    
    # Type 3: Ratecode parameters  
    ratecode_swap_prob: float = 0.3  # probability to swap ratecode
    
    # Random seed
    seed: int = 42
    
    # Distribution of fraud types (for MIXED)
    type1_ratio: float = 0.60  # 60% short-trip
    type3_ratio: float = 0.40  # 40% ratecode
    
    def __post_init__(self):
        if isinstance(self.fraud_type, str):
            self.fraud_type = FraudType(self.fraud_type)


# =============================================================================
# Core Injection Functions
# =============================================================================

def inject_anomalies(df: pd.DataFrame,
                     config: Optional[FraudConfig] = None,
                     n_anomalies: Optional[int] = None,
                     anomaly_rate: Optional[float] = None,
                     fraud_type: Optional[FraudType] = None,
                     seed: Optional[int] = None,
                     **kwargs) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Inject anomalies vào dataframe.
    
    Args:
        df: Input dataframe (clean data)
        config: FraudConfig object (optional, overrides other params)
        n_anomalies: Exact number of anomalies to inject
        anomaly_rate: Fraction of data to mark as anomaly
        fraud_type: Type of fraud to inject
        seed: Random seed
        **kwargs: Additional params passed to FraudConfig
        
    Returns:
        (modified_df, labels) - labels[i]=1 if injected anomaly
        
    Usage:
        # Simple usage
        df_test, labels = inject_anomalies(df, anomaly_rate=0.03, seed=42)
        
        # Custom config
        config = FraudConfig(anomaly_rate=0.05, fraud_type=FraudType.SHORT_TRIP)
        df_test, labels = inject_anomalies(df, config=config)
        
        # Fixed number
        df_test, labels = inject_anomalies(df, n_anomalies=500)
    """
    # Build config
    if config is None:
        config = FraudConfig(
            anomaly_rate=anomaly_rate or 0.03,
            fraud_type=fraud_type or FraudType.MIXED,
            seed=seed or 42,
            **kwargs
        )
    
    # Determine number of anomalies
    n_total = len(df)
    if n_anomalies is not None:
        n_anom = n_anomalies
    else:
        n_anom = int(n_total * config.anomaly_rate)
    
    # Initialize labels (all normal)
    labels = np.zeros(n_total, dtype=int)
    
    # Copy dataframe để không modify original
    df = df.copy()
    
    # Select random indices for anomalies
    rng = np.random.RandomState(config.seed)
    anom_indices = rng.choice(n_total, size=n_anom, replace=False)
    labels[anom_indices] = 1
    
    # Apply fraud based on type
    if config.fraud_type == FraudType.SHORT_TRIP:
        _inject_short_trip(df, anom_indices, config, rng)
    elif config.fraud_type == FraudType.LONG_TRIP:
        _inject_long_trip(df, anom_indices, config, rng)
    elif config.fraud_type == FraudType.RATECODE_MISMATCH:
        _inject_ratecode_mismatch(df, anom_indices, config, rng)
    elif config.fraud_type == FraudType.NIGHT_FRAUD:
        _inject_night_fraud(df, anom_indices, config, rng)
    elif config.fraud_type == FraudType.MIXED:
        # Split into type1 and type3
        n_type1 = int(n_anom * config.type1_ratio)
        type1_idx = anom_indices[:n_type1]
        type3_idx = anom_indices[n_type1:]
        _inject_short_trip(df, type1_idx, config, rng)
        _inject_ratecode_mismatch(df, type3_idx, config, rng)
    
    return df, labels


def _inject_short_trip(df: pd.DataFrame,
                       indices: np.ndarray,
                       config: FraudConfig,
                       rng: np.random.RandomState) -> None:
    """Inject short-trip fraud: low distance, inflated fare."""
    if len(indices) == 0:
        return
    
    for idx in indices:
        # Reduce distance
        df.iloc[idx, df.columns.get_loc('trip_distance')] = rng.uniform(
            0.1, config.short_trip_max_dist
        )
        
        # Increase fare
        df.iloc[idx, df.columns.get_loc('fare_amount')] = rng.uniform(
            config.short_trip_min_fare, 
            config.short_trip_min_fare * 2
        )
        
        # Also adjust total_amount
        if 'total_amount' in df.columns:
            total = df.iloc[idx]['fare_amount'] + df.iloc[idx].get('tip_amount', 0)
            df.iloc[idx, df.columns.get_loc('total_amount')] = total


def _inject_long_trip(df: pd.DataFrame,
                     indices: np.ndarray,
                     config: FraudConfig,
                     rng: np.random.RandomState) -> None:
    """Inject long-trip undercharge: high distance, low fare."""
    if len(indices) == 0:
        return
    
    for idx in indices:
        # Increase distance significantly
        df.iloc[idx, df.columns.get_loc('trip_distance')] = rng.uniform(
            15.0, 30.0
        )
        
        # Keep fare suspiciously low
        df.iloc[idx, df.columns.get_loc('fare_amount')] = rng.uniform(
            3.0, 8.0
        )


def _inject_ratecode_mismatch(df,
                             indices: np.ndarray,
                             config: FraudConfig,
                             rng: np.random.RandomState) -> None:
    """Inject ratecode mismatch fraud."""
    if len(indices) == 0:
        return
    
    for idx in indices:
        if 'RatecodeID' in df.columns:
            current_rate = df.iloc[idx]['RatecodeID']
            
            # Swap to a different ratecode
            ratecodes = [1, 2, 3, 4, 5, 99]
            ratecodes = [r for r in ratecodes if r != current_rate]
            new_rate = rng.choice(ratecodes)
            
            df.iloc[idx, df.columns.get_loc('RatecodeID')] = new_rate
            
            # Adjust fare based on new ratecode
            if new_rate == 2:  # JFK
                df.iloc[idx, df.columns.get_loc('fare_amount')] = rng.uniform(45, 55)
            elif new_rate == 3:  # Newark
                df.iloc[idx, df.columns.get_loc('fare_amount')] = rng.uniform(50, 65)
            elif new_rate == 4:  # Nassau/Westchester
                df.iloc[idx, df.columns.get_loc('fare_amount')] = rng.uniform(30, 45)


def _inject_night_fraud(df: pd.DataFrame,
                       indices: np.ndarray,
                       config: FraudConfig,
                       rng: np.random.RandomState) -> None:
    """Inject night-time fraud: suspicious patterns after midnight."""
    if len(indices) == 0:
        return
    
    for idx in indices:
        # Make it look like night trip
        df.iloc[idx, df.columns.get_loc('trip_distance')] = rng.uniform(
            2.0, 8.0
        )
        # Inflated night fare
        df.iloc[idx, df.columns.get_loc('fare_amount')] = rng.uniform(
            25.0, 50.0
        )


# =============================================================================
# Utility Functions
# =============================================================================

def get_anomaly_summary(labels: np.ndarray) -> dict:
    """
    Get summary statistics của injected anomalies.
    
    Args:
        labels: Binary labels array
        
    Returns:
        Dict với summary statistics
    """
    n_total = len(labels)
    n_anom = int(labels.sum())
    n_norm = n_total - n_anom
    
    return {
        'n_total': n_total,
        'n_normal': n_norm,
        'n_anomaly': n_anom,
        'anomaly_rate': n_anom / n_total if n_total > 0 else 0,
        'normal_rate': n_norm / n_total if n_total > 0 else 0,
    }


def split_by_fraud_type(df: pd.DataFrame,
                        labels: np.ndarray,
                        method: str = 'distance') -> Tuple[np.ndarray, np.ndarray]:
    """
    Attempt to separate anomalies by type based on features.
    
    Args:
        df: DataFrame
        labels: Binary labels
        method: 'distance' or 'ratecode'
        
    Returns:
        (type1_mask, type3_mask) - boolean arrays
    """
    anom_mask = labels == 1
    
    if method == 'distance':
        # Type 1: short trip, Type 3: longer
        dist = df['trip_distance'].fillna(0).values
        type1_mask = anom_mask & (dist < 1.5)
        type3_mask = anom_mask & (dist >= 1.5)
    elif method == 'ratecode':
        # Type 3: non-standard ratecode
        ratecode = df['RatecodeID'].fillna(1).astype(float).values
        type3_mask = anom_mask & (ratecode != 1)
        type1_mask = anom_mask & (ratecode == 1)
    else:
        type1_mask = anom_mask
        type3_mask = np.zeros_like(anom_mask, dtype=bool)
    
    return type1_mask, type3_mask


# Alias cho backwards compatibility
inject_fraud = inject_anomalies
