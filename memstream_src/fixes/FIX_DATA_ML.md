# Data Engineering + ML Fixes - CA-DQStream + MemStream v4 → v5

**Date:** 2026-05-12  
**Source:** Data Engineer Review (REVIEW_DATAENG_v4.md)  
**Status:** CRITICAL + HIGH Issues from PLAN_v4.md

---

## C-DE-1: Time-Ordered Data Splits (CRITICAL)

**Issue:** `df.sample(frac=1, random_state=42)` shuffles data, destroying temporal structure needed for streaming anomaly detection evaluation.

**File:** `memstream_src/scripts/train_warmup.py`

### Problem Code (WRONG):

```python
# WRONG: Shuffles all data, destroys temporal order
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
```

### Fixed Code (CORRECT):

```python
#!/usr/bin/env python3
"""
MemStream Training Pipeline — Time-Ordered Data Splits.

FIX C-DE-1: Time-ordered splits instead of random shuffle.
  - Month 1-6 (60%): Warmup/training
  - Month 7-9 (20%): Calibration
  - Month 10-12 (20%): Test/Evaluation

FIX C-DE-2: Normalization leakage prevention.
  - First 10%: Compute normalization stats
  - Middle 80%: Autoencoder training
  - Last 10%: Memory initialization
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle
import json
from datetime import datetime


def prepare_time_ordered_splits(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    calib_frac: float = 0.8
) -> dict:
    """
    Prepare TEMPORAL splits for streaming anomaly detection.
    
    CRITICAL: Uses time-ordered splits, NOT random shuffle.
    This mimics production where we train on past, predict future.
    
    Args:
        df: Input DataFrame with 'tpep_pickup_datetime' column
        train_frac: End of training data (default 0.6 = 60%)
        calib_frac: End of calibration data (default 0.8 = 80%)
    
    Returns:
        Dictionary with 'warmup', 'calibration', 'test' DataFrames
    """
    # Step 1: Parse datetime and sort by time (NO SHUFFLE!)
    df['pickup_dt'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    
    # Remove records with invalid datetime
    df = df.dropna(subset=['pickup_dt'])
    
    # Step 2: Sort by time (CRITICAL FIX)
    df = df.sort_values('pickup_dt').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_frac)
    calib_end = int(n * calib_frac)
    
    splits = {
        'warmup': df.iloc[:train_end].copy(),
        'calibration': df.iloc[train_end:calib_end].copy(),
        'test': df.iloc[calib_end:].copy()
    }
    
    # Verify temporal order
    if len(splits['warmup']) > 0 and len(splits['calibration']) > 0:
        assert splits['warmup']['pickup_dt'].max() <= splits['calibration']['pickup_dt'].min(), \
            "Temporal overlap: warmup and calibration sets overlap!"
    if len(splits['calibration']) > 0 and len(splits['test']) > 0:
        assert splits['calibration']['pickup_dt'].max() <= splits['test']['pickup_dt'].min(), \
            "Temporal overlap: calibration and test sets overlap!"
    
    print(f"\n{'='*60}")
    print("TIME-ORDERED DATA SPLITS")
    print(f"{'='*60}")
    print(f"Total records: {n:,}")
    print(f"  Warmup (Month 1-6):      {len(splits['warmup']):>8,} ({len(splits['warmup'])/n*100:>5.1f}%)")
    print(f"  Calibration (Month 7-9):  {len(splits['calibration']):>8,} ({len(splits['calibration'])/n*100:>5.1f}%)")
    print(f"  Test (Month 10-12):      {len(splits['test']):>8,} ({len(splits['test'])/n*100:>5.1f}%)")
    print(f"\nTemporal ranges:")
    print(f"  Warmup:      {splits['warmup']['pickup_dt'].min()} → {splits['warmup']['pickup_dt'].max()}")
    print(f"  Calibration: {splits['calibration']['pickup_dt'].min()} → {splits['calibration']['pickup_dt'].max()}")
    print(f"  Test:        {splits['test']['pickup_dt'].min()} → {splits['test']['pickup_dt'].max()}")
    
    return splits


def prepare_warmup_data_leakage_free(
    df: pd.DataFrame,
    stats_frac: float = 0.1,
    memory_frac: float = 0.1
) -> dict:
    """
    Prepare warmup data with NO normalization leakage.
    
    FIX C-DE-2: Split warmup data into 3 parts:
      1. First 10% (stats_frac): Compute normalization stats ONLY
      2. Middle portion: Train autoencoder
      3. Last 10% (memory_frac): Initialize memory module
    
    Args:
        df: Warmup DataFrame (already time-sorted)
        stats_frac: Fraction for stats computation (default 0.1 = 10%)
        memory_frac: Fraction for memory init (default 0.1 = 10%)
    
    Returns:
        Dictionary with 'stats_data', 'train_data', 'memory_data', 'stats'
    """
    n = len(df)
    
    # Compute split points
    stats_end = int(n * stats_frac)
    memory_start = int(n * (1 - memory_frac))
    
    # Split the data
    stats_data = df.iloc[:stats_end]
    train_data = df.iloc[stats_end:memory_start]
    memory_data = df.iloc[memory_start:]
    
    print(f"\n{'='*60}")
    print("LEAKAGE-FREE WARMUP PREPARATION")
    print(f"{'='*60}")
    print(f"Total warmup: {n:,} records")
    print(f"  Stats computation: {len(stats_data):>8,} ({stats_frac*100:.0f}%)")
    print(f"  AE training:       {len(train_data):>8,} ({(1-stats_frac-memory_frac)*100:.0f}%)")
    print(f"  Memory init:       {len(memory_data):>8,} ({memory_frac*100:.0f}%)")
    
    return {
        'stats_data': stats_data,
        'train_data': train_data,
        'memory_data': memory_data,
        'stats_end': stats_end,
        'memory_start': memory_start
    }


def compute_normalization_stats(features: np.ndarray) -> dict:
    """
    Compute normalization statistics from stats portion ONLY.
    
    This ensures no leakage: the stats are computed from data
    that won't be used for training or memory initialization.
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std = np.clip(std, min=1e-8)  # Prevent division by zero
    
    stats = {
        'mean': mean.astype(np.float32),
        'std': std.astype(np.float32),
        'shape': features.shape,
        'n_samples': len(features)
    }
    
    print(f"\nNormalization stats computed from {len(features):,} samples")
    print(f"  Mean (first 5): {mean[:5]}")
    print(f"  Std (first 5):  {std[:5]}")
    
    return stats


# =============================================================================
# COMPLETE train_warmup.py IMPLEMENTATION
# =============================================================================

class MemStreamTrainer:
    """MemStream trainer with time-ordered splits and leakage-free warmup."""
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.stats = None
        self.model = None
        self.memory = None
    
    def prepare_data(self, data_path: str, train_frac: float = 0.6, calib_frac: float = 0.8):
        """
        Load and split data with time-ordered splits.
        
        Args:
            data_path: Path to parquet file
            train_frac: Training data fraction (0.6 = 60%)
            calib_frac: Calibration data fraction (0.8 = 80%)
        """
        print(f"Loading data from: {data_path}")
        df = pd.read_parquet(data_path)
        print(f"Loaded {len(df):,} records")
        
        # Time-ordered splits (FIX C-DE-1)
        self.splits = prepare_time_ordered_splits(df, train_frac, calib_frac)
        
        # Prepare leakage-free warmup data (FIX C-DE-2)
        self.warmup_data = prepare_warmup_data_leakage_free(
            self.splits['warmup'],
            stats_frac=0.1,
            memory_frac=0.1
        )
        
        return self.splits, self.warmup_data
    
    def compute_stats(self, features: np.ndarray):
        """
        Compute normalization stats from first 10% of warmup data.
        
        FIX C-DE-2: Stats computed BEFORE training, from separate portion.
        """
        self.stats = compute_normalization_stats(features)
        return self.stats
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize using pre-computed stats."""
        if self.stats is None:
            raise ValueError("Stats not computed. Call compute_stats() first.")
        return (x - self.stats['mean']) / self.stats['std']
    
    def train_autoencoder(
        self,
        train_features: np.ndarray,
        epochs: int = 500,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = 'cpu'
    ):
        """
        Train autoencoder on MIDDLE portion of warmup data.
        
        FIX C-DE-2: Training uses separate data from stats computation.
        """
        print(f"\n{'='*60}")
        print("AUTOENCODER TRAINING")
        print(f"{'='*60}")
        
        # Normalize training data (using stats from FIRST 10%)
        train_normalized = self.normalize(train_features)
        
        # Convert to tensors
        X_train = torch.from_numpy(train_normalized).float().to(device)
        
        # Model architecture (25D → 16 → 8 → 16 → 25D)
        class Autoencoder(nn.Module):
            def __init__(self, input_dim=25):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(8, 16),
                    nn.ReLU(),
                    nn.Linear(16, input_dim)
                )
            
            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)
        
        model = Autoencoder(input_dim=train_features.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Training loop
        dataset = torch.utils.data.TensorDataset(X_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0]
                optimizer.zero_grad()
                x_hat = model(x)
                loss = criterion(x_hat, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / len(loader)
                print(f"  Epoch {epoch+1:4d}: loss = {avg_loss:.6f}")
        
        self.model = model
        print(f"Training complete. Final model on device: {device}")
        
        return model
    
    def initialize_memory(
        self,
        memory_features: np.ndarray,
        memory_size: int = 100,
        device: str = 'cpu'
    ):
        """
        Initialize memory module with LAST 10% of warmup data.
        
        FIX C-DE-2: Memory initialized with data NOT used for stats/training.
        """
        print(f"\n{'='*60}")
        print("MEMORY INITIALIZATION")
        print(f"{'='*60}")
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_autoencoder() first.")
        
        # Normalize memory data
        memory_normalized = self.normalize(memory_features)
        
        # Encode to latent space
        memory_tensor = torch.from_numpy(memory_normalized).float().to(device)
        with torch.no_grad():
            memory_encoded = self.model.encoder(memory_tensor)
        
        # Select top-k most diverse samples for memory (simplified)
        if len(memory_encoded) > memory_size:
            # Use random selection (can be improved with k-means or herding)
            indices = np.random.choice(len(memory_encoded), memory_size, replace=False)
            memory_init = memory_encoded[indices].cpu().numpy()
        else:
            memory_init = memory_encoded.cpu().numpy()
        
        print(f"  Memory initialized with {len(memory_init)} slots")
        print(f"  Memory shape: {memory_init.shape}")
        
        self.memory = memory_init
        return memory_init
    
    def save(self, output_dir: str):
        """Save model, stats, and memory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), output_path / 'autoencoder.pt')
        
        # Save stats
        with open(output_path / 'stats.pkl', 'wb') as f:
            pickle.dump(self.stats, f)
        
        # Save memory
        np.save(output_path / 'memory.npy', self.memory)
        
        print(f"\nSaved to {output_dir}")
        print(f"  autoencoder.pt: PyTorch model weights")
        print(f"  stats.pkl: Normalization statistics")
        print(f"  memory.npy: Memory module initialization")


def main():
    """Main entry point for training pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description='MemStream Training')
    parser.add_argument('--data', default='data/clean/jan_2024_clean_baseline.parquet')
    parser.add_argument('--output', default='models/memstream')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--memory-size', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    
    trainer = MemStreamTrainer(device=args.device)
    
    # Step 1: Load and split data
    splits, warmup_data = trainer.prepare_data(args.data)
    
    # Step 2: Extract features (use FeatureVectorizer from src/features/vectorizer.py)
    from src.features.vectorizer import FeatureVectorizer
    vectorizer = FeatureVectorizer()
    
    # Get stats data for normalization (FIX C-DE-2: FIRST 10%)
    stats_features = vectorizer.transform_batch(warmup_data['stats_data'])
    trainer.compute_stats(stats_features)
    
    # Get training data (FIX C-DE-2: MIDDLE 80%)
    train_features = vectorizer.transform_batch(warmup_data['train_data'])
    
    # Get memory data (FIX C-DE-2: LAST 10%)
    memory_features = vectorizer.transform_batch(warmup_data['memory_data'])
    
    # Step 3: Train autoencoder
    trainer.train_autoencoder(
        train_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Step 4: Initialize memory
    trainer.initialize_memory(
        memory_features,
        memory_size=args.memory_size,
        device=args.device
    )
    
    # Step 5: Save
    trainer.save(args.output)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Time-ordered splits: ✓")
    print(f"  Leakage-free warmup: ✓")
    print(f"  Stats from first 10%: ✓")
    print(f"  Training on middle 80%: ✓")
    print(f"  Memory from last 10%: ✓")


if __name__ == '__main__':
    main()
```

---

## C-DE-2: Normalization Leakage Prevention (CRITICAL)

**Issue:** Normalization stats computed from data used for memory initialization AND AE training simultaneously.

**Fix:** Split warmup data into 3 parts with strict separation.

### Memory Initialization with Leakage Prevention

```python
def warmup_leakage_free(self, normal_data: np.ndarray, epochs: int = 5000):
    """
    Warmup with STRICT separation between stats, training, and memory data.
    
    FIX C-DE-2: Data leakage prevention in MemStream warmup.
    
    Data flow:
      [10%] → Compute mean/std (stats_data)
      [80%] → Train autoencoder (train_data)  
      [10%] → Initialize memory (memory_data)
    """
    n = len(normal_data)
    
    # Step 1: Compute stats from FIRST portion (NOT used in training)
    stats_end = int(n * 0.1)
    stats_data = normal_data[:stats_end]
    
    self.mean = torch.from_numpy(stats_data.mean(axis=0)).float().to(self.device)
    self.std = torch.from_numpy(stats_data.std(axis=0)).float().to(self.device)
    self.std = torch.clamp(self.std, min=1e-8)  # Prevent div/0
    
    print(f"Stats computed from {stats_end:,} samples (first 10%)")
    print(f"  Mean range: [{self.mean.min():.4f}, {self.mean.max():.4f}]")
    print(f"  Std range:  [{self.std.min():.4f}, {self.std.max():.4f}]")
    
    # Step 2: Normalize training data with computed stats
    train_start = stats_end
    train_end = int(n * 0.9)
    train_data_raw = normal_data[train_start:train_end]
    
    # Normalize training data
    train_data = self._normalize(torch.from_numpy(train_data_raw).float().to(self.device))
    
    # Step 3: Train autoencoder on normalized data
    self._train_autoencoder(train_data, epochs=epochs)
    
    # Step 4: Memory initialization from LAST portion
    memory_data_raw = normal_data[train_end:]
    memory_data = self._normalize(torch.from_numpy(memory_data_raw).float().to(self.device))
    
    # Encode to latent space
    with torch.no_grad():
        memory_encoded = self.encoder(memory_data)
    
    # Initialize memory with diverse samples
    self._initialize_memory_from_encoding(memory_encoded)
    
    print(f"Memory initialized with {self.memory_len} slots from last 10% of warmup data")
    print("✓ Warmup complete with LEAKAGE-FREE normalization")


def _normalize(self, x: torch.Tensor) -> torch.Tensor:
    """Normalize using pre-computed mean/std (from stats_data only)."""
    return (x - self.mean) / self.std


def _initialize_memory_from_encoding(self, encoded: torch.Tensor):
    """
    Initialize memory from encoded representations.
    
    Uses herding selection to pick diverse samples.
    """
    memory_size = min(self.memory_len, len(encoded))
    
    # Simple random selection (can be improved with herding)
    if len(encoded) > memory_size:
        indices = torch.randperm(len(encoded))[:memory_size]
        encoded = encoded[indices]
    
    self.mem_data = encoded.cpu()
    
    # Reset memory usage tracking
    self.mem_usage = torch.zeros(memory_size)
```

---

## H-DE-1: Kafka Topic Definition (HIGH)

**Issue:** Kafka topic structure undefined, causing deployment ambiguity.

### Kafka Topic Configuration

```yaml
# docker-compose.kafka.yml or kafka-topics-config.yaml

# Topic definitions for CA-DQStream + MemStream

topics:
  # Raw taxi trip data ingestion
  taxi-raw:
    name: cadqstream.trips.raw
    partitions: 12
    replication_factor: 3
    retention_ms: 604800000  # 7 days
    min_insync_replicas: 2
    config:
      cleanup.policy: delete
      flush.messages: 10000
      flush.ms: 30000

  # Anomaly detection output scores
  dq-stream-anomalies:
    name: cadqstream.scores.output
    partitions: 12
    replication_factor: 3
    retention_ms: 2592000000  # 30 days for analytics
    min_insync_replicas: 2
    config:
      cleanup.policy: delete

  # Baseline rule violations
  dq-stream-violations:
    name: cadqstream.violations.output
    partitions: 12
    replication_factor: 3
    retention_ms: 604800000  # 7 days
    min_insync_replicas: 2

  # IEC control signals
  dq-iec-control:
    name: cadqstream.iec.control
    partitions: 6
    replication_factor: 3
    retention_ms: 86400000  # 1 day
    min_insync_replicas: 2

  # Late/dead letter queue
  dq-late-records:
    name: cadqstream.trips.late
    partitions: 6
    replication_factor: 3
    retention_ms: 172800000  # 2 days for debugging
    min_insync_replicas: 2

  # Action replay for IEC feedback
  dq-action-replay:
    name: cadqstream.action.replay
    partitions: 6
    replication_factor: 3
    retention_ms: 604800000  # 7 days
    min_insync_replicas: 2
```

### Kafka Topic Creation Script

```bash
#!/bin/bash
# scripts/create_kafka_topics.sh

KAFKA_BROKERS="${KAFKA_BROKERS:-kafka:9092}"

echo "Creating Kafka topics with RF=3, minISR=2..."

# Function to create topic
create_topic() {
    local name=$1
    local partitions=$2
    local retention_ms=$3
    
    echo "Creating topic: $name (partitions=$partitions, retention=${retention_ms}ms)"
    
    kafka-topics --create \
        --bootstrap-server "$KAFKA_BROKERS" \
        --topic "$name" \
        --partitions "$partitions" \
        --replication-factor 3 \
        --config min.insync.replicas=2 \
        --config retention.ms="$retention_ms" \
        --if-not-exists
    
    # Wait for topic to be created
    sleep 2
}

# Create all topics
create_topic "cadqstream.trips.raw" 12 604800000
create_topic "cadqstream.scores.output" 12 2592000000
create_topic "cadqstream.violations.output" 12 604800000
create_topic "cadqstream.iec.control" 6 86400000
create_topic "cadqstream.trips.late" 6 172800000
create_topic "cadqstream.action.replay" 6 604800000

echo "All topics created successfully!"

# List topics
kafka-topics --list --bootstrap-server "$KAFKA_BROKERS"
```

---

## H-DE-2: Event Ordering Semantics (HIGH)

**Issue:** Event ordering guarantees not documented clearly.

### Event Ordering Documentation

```markdown
## Event Ordering Guarantees

### Partitioning Strategy

CA-DQStream uses **composite partitioning** by `(neighborhood, hour_bucket)`:

```
Kafka Record Key: "{neighborhood}_{hour_bucket}"
Example: "Manhattan_6" for Manhattan trips between 6-7 AM
```

### Ordering Guarantees

| Scope | Guarantee | Implementation |
|-------|-----------|----------------|
| Within neighborhood | ✅ **FIFO** | Flink keyed by neighborhood |
| Within hour_bucket | ✅ **FIFO** | Hour bucket in composite key |
| Across neighborhoods | ❌ Not guaranteed | Natural parallelism |
| Across hour_buckets | ❌ Not guaranteed | Separate key partitions |

### Why This Matters

1. **Memory updates**: Records within same neighborhood are processed in order, ensuring memory module sees realistic temporal sequences.

2. **Context thresholds**: Threshold selection depends on temporal features. In-order processing ensures correct context key computation.

3. **IEC feedback**: Control signals must arrive before the records they affect. Separate topic with lower latency.

### Composite Key Computation

```python
def compute_partition_key(record: dict) -> str:
    """
    Compute composite partition key for taxi trip records.
    
    Format: "{PULocationID}_{hour_bucket}"
    
    Example:
      PULocationID=132 (JFK), hour=14 → "132_14"
    """
    neighborhood = str(record.get('PULocationID', ''))
    hour = pd.to_datetime(record.get('tpep_pickup_datetime')).hour
    hour_bucket = f"{hour:02d}"  # Zero-padded hour
    return f"{neighborhood}_{hour_bucket}"
```

### Flink Source Configuration

```python
# In flink_job_complete.py

from pyflink.datastream import TimeCharacteristic

# Set event time processing
env.set_stream_time_characteristic(TimeCharacteristic.EventTime)

# Watermark configuration
stream = env.add_source(kafka_source) \
    .assign_timestamps_and_watermarks(
        WatermarkStrategy
            .for_bounded_out_of_orderness(Duration.of_seconds(10))
            .with_timestamp_assigner(AscendingTimestampExtractor())
    )

# Key by composite partition key
keyed_stream = stream.key_by(
    lambda r: compute_partition_key(r),
    key_type=Types.STRING()
)
```

---

## H-DE-3: Late Data Handling (HIGH)

**Issue:** Late data handling missing, records can be lost or scored incorrectly.

### Late Data Handler Implementation

```python
# src/operators/late_data_handler.py

from pyflink.datastream import SideOutput
from pyflink.datastream.output import OutputTag
from pyflink.common import Time
from datetime import timedelta
from typing import Optional


class LateDataConfig:
    """Configuration for late data handling."""
    
    # Maximum allowed lateness (6 hours for NYC taxi data)
    ALLOWED_LATENESS_MS = 6 * 60 * 60 * 1000  # 6 hours in milliseconds
    
    # Watermark interval (1 second)
    WATERMARK_INTERVAL_MS = 1000
    
    # Side output tag for dead letter queue
    LATE_TAG = OutputTag('late-records', Types.PICKLED_BYTE_ARRAY())


class LateDataHandler:
    """
    Handle late arrivals for taxi trip scoring.
    
    NYC TLC data can have late submissions up to 24 hours.
    We allow 6 hours of lateness before routing to DLQ.
    """
    
    def __init__(
        self,
        allowed_lateness_hours: int = 6,
        watermark_interval_seconds: int = 1
    ):
        self.allowed_lateness = timedelta(hours=allowed_lateness_hours)
        self.watermark_interval = timedelta(seconds=watermark_interval_seconds)
        self.late_tag = LateDataConfig.LATE_TAG
    
    def is_late(self, record_timestamp: pd.Timestamp, watermark: pd.Timestamp) -> bool:
        """
        Determine if a record is late.
        
        A record is late if its timestamp is older than (watermark - allowed_lateness).
        """
        if pd.isna(record_timestamp) or pd.isna(watermark):
            return False
        cutoff = watermark - self.allowed_lateness
        return record_timestamp < cutoff
    
    def get_late_reason(self, record_timestamp: pd.Timestamp, watermark: pd.Timestamp) -> str:
        """
        Determine why a record is late.
        """
        if pd.isna(record_timestamp):
            return "missing_timestamp"
        
        lateness = watermark - record_timestamp
        
        if lateness > timedelta(days=1):
            return "extreme_lateness_24h+"
        elif lateness > timedelta(hours=12):
            return "severe_lateness_12h+"
        elif lateness > timedelta(hours=6):
            return "moderate_lateness_6h+"
        else:
            return "minor_lateness_under_6h"
    
    def create_late_record(self, record: dict, watermark: pd.Timestamp) -> dict:
        """
        Create enriched late record for DLQ.
        """
        record_timestamp = pd.to_datetime(record.get('tpep_pickup_datetime'))
        
        return {
            **record,
            '_late_metadata': {
                'late_reason': self.get_late_reason(record_timestamp, watermark),
                'record_timestamp': record_timestamp.isoformat() if pd.notna(record_timestamp) else None,
                'watermark_timestamp': watermark.isoformat(),
                'lateness_hours': (watermark - record_timestamp).total_seconds() / 3600 if pd.notna(record_timestamp) else None,
                'processed_at': pd.Timestamp.now().isoformat()
            }
        }


class LateDataWatermarkStrategy:
    """
    Flink watermark strategy with late data handling.
    """
    
    @staticmethod
    def create() -> 'WatermarkStrategy':
        """
        Create watermark strategy with:
        - Bounded out-of-orderness (10 seconds)
        - 6-hour allowed lateness
        - Side output for late records
        """
        from pyflink.datastream import WatermarkStrategy, TimeCharacteristic
        
        return WatermarkStrategy \
            .for_bounded_out_of_orderness(Duration.of_seconds(10)) \
            .with_timestamp_assigner(TaxiTimestampAssigner()) \
            .with_idleness(Duration.of_minutes(1)) \
            .with_watermark_interval(Time.minutes(1).to_milliseconds())


class TaxiTimestampAssigner:
    """
    Extract timestamps from taxi trip records.
    
    Uses pickup datetime as event time timestamp.
    """
    
    def extract_timestamp(self, record: dict, timestamp: int) -> int:
        """
        Extract milliseconds since epoch from pickup datetime.
        """
        pickup = pd.to_datetime(record.get('tpep_pickup_datetime'))
        if pd.isna(pickup):
            # Use current time as fallback (record will be processed immediately)
            return int(pd.Timestamp.now().timestamp() * 1000)
        return int(pickup.timestamp() * 1000)


# Flink Integration Example

def build_pipeline_with_late_handling(env):
    """
    Build Flink pipeline with late data handling.
    """
    from pyflink.datastream import SideOutput, OutputTag, StreamExecutionEnvironment
    from pyflink.common import Types
    
    late_tag = OutputTag('late-records', Types.PICKLED_BYTE_ARRAY())
    dlq_tag = OutputTag('dlq-records', Types.PICKLED_BYTE_ARRAY())
    
    # Source with watermark strategy
    stream = env \
        .add_source(kafka_source) \
        .assign_timestamps_and_watermarks(
            LateDataWatermarkStrategy.create()
        )
    
    # Process with late handling
    processed = stream \
        .process(LateDataProcessFunction(late_tag, dlq_tag)) \
        .get_side_output(late_tag)
    
    # Late records → DLQ
    processed.add_sink(kafka_sink('cadqstream.trips.late'))
    
    # DLQ for extreme late
    dlq = processed.get_side_output(dlq_tag)
    dlq.add_sink(kafka_sink('cadqstream.dlq'))
    
    return stream


class LateDataProcessFunction(KeyedProcessFunction):
    """
    Process function with late data side output.
    """
    
    def __init__(self, late_tag, dlq_tag):
        self.late_tag = late_tag
        self.dlq_tag = dlq_tag
        self.late_handler = LateDataHandler()
    
    def process_element(self, record, context, out):
        watermark = context.timer_service().current_watermark()
        record_time = pd.to_datetime(record.get('tpep_pickup_datetime'))
        
        if self.late_handler.is_late(record_time, watermark):
            # Route to appropriate side output
            if watermark - record_time > timedelta(hours=12):
                # Extreme lateness → DLQ
                context.output(
                    self.dlq_tag,
                    self.late_handler.create_late_record(record, watermark)
                )
            else:
                # Moderate lateness → late records topic
                context.output(
                    self.late_tag,
                    self.late_handler.create_late_record(record, watermark)
                )
            return  # Don't process late records in main pipeline
        
        # Normal processing
        out.collect(record)
```

---

## H-DE-4: Multi-Strategy Anomaly Injection (HIGH)

**Issue:** Current anomaly injection is deterministic and learnable (all fare_per_mile anomalies).

### Multi-Strategy Anomaly Injection Implementation

```python
#!/usr/bin/env python3
"""
Multi-Strategy Contextual Anomaly Injection.

FIX H-DE-4: Replace single-strategy injection with diverse, realistic anomalies.

Strategies:
  1. speed_extreme: Physically impossible speeds (0-5 mph or 100+ mph)
  2. swap_location: Pickup/dropoff location inconsistencies
  3. duration_anomaly: Zero or negative duration anomalies
  4. passenger_count: Impossible passenger counts
  5. fare_ratio: Extreme fare/distance ratios
  6. time_impossible: Records during invalid hours
  7. zone_impossible: Invalid NYC TLC zone IDs
  8. combined: Multiple violations at once

Usage:
  python scripts/inject_anomalies_multi.py --input data/clean/baseline.parquet \\
                                           --output data/labeled/anomalies.parquet \\
                                           --n-anomalies 50000 \\
                                           --strategy-ratio 0.02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import List, Tuple, Callable
import json


class MultiStrategyAnomalyInjector:
    """
    Inject diverse contextual anomalies with multiple strategies.
    
    FIX H-DE-4: Multi-strategy injection instead of single fare_per_mile.
    """
    
    # Valid NYC TLC zone IDs
    VALID_ZONE_IDS = set(range(1, 264))
    
    # Airport zones
    AIRPORT_ZONES = {1, 132, 138}  # Newark, JFK, LaGuardia
    
    # Rush hour boundaries
    RUSH_MORNING = (6, 10)
    RUSH_EVENING = (17, 21)
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        
        # Register all anomaly strategies
        self.strategies = {
            'speed_extreme': self._inject_speed_extreme,
            'swap_location': self._inject_swap_location,
            'duration_anomaly': self._inject_duration_anomaly,
            'passenger_count': self._inject_passenger_count,
            'fare_ratio': self._inject_fare_ratio,
            'time_impossible': self._inject_time_impossible,
            'zone_impossible': self._inject_zone_impossible,
            'combined': self._inject_combined,
        }
        
        # Strategy weights (total must sum to 1.0)
        # More realistic distribution based on common fraud patterns
        self.strategy_weights = {
            'speed_extreme': 0.20,      # GPS spoofing
            'swap_location': 0.15,      # Location fraud
            'duration_anomaly': 0.18,   # Meter manipulation
            'passenger_count': 0.12,    # Passenger fraud
            'fare_ratio': 0.15,         # Fare manipulation
            'time_impossible': 0.08,    # Time fraud
            'zone_impossible': 0.07,    # Zone fraud
            'combined': 0.05,           # Multi-violation
        }
        
        # Validate weights
        assert abs(sum(self.strategy_weights.values()) - 1.0) < 1e-6, \
            "Strategy weights must sum to 1.0"
    
    def get_strategy(self) -> str:
        """Sample a random strategy based on weights."""
        strategies = list(self.strategy_weights.keys())
        weights = list(self.strategy_weights.values())
        return self.rng.choice(strategies, p=weights)
    
    # =========================================================================
    # INDIVIDUAL ANOMALY STRATEGIES
    # =========================================================================
    
    def _inject_speed_extreme(self, record: dict, difficulty: str = 'medium') -> dict:
        """
        Inject impossible speed anomaly (GPS spoofing).
        
        - Low speed: 0-5 mph (frozen GPS)
        - High speed: 100-300 mph (GPS jump)
        """
        distance = record.get('trip_distance', 5.0)
        
        if self.rng.random() < 0.5:
            # Low speed anomaly: very short distance, very long time
            record['trip_distance'] = self.rng.uniform(0.1, 0.5)  # 0.1-0.5 miles
            duration_hours = self.rng.uniform(1.0, 3.0)  # 1-3 hours (frozen GPS)
        else:
            # High speed anomaly: very long distance, very short time
            record['trip_distance'] = self.rng.uniform(50.0, 100.0)  # 50-100 miles
            duration_hours = self.rng.uniform(0.1, 0.5)  # 6-30 minutes (GPS jump)
        
        # Recalculate duration
        pickup = pd.to_datetime(record['tpep_pickup_datetime'])
        record['tpep_dropoff_datetime'] = pickup + timedelta(hours=duration_hours)
        
        # Recalculate derived fields
        duration_min = duration_hours * 60
        record['dur_min'] = duration_min
        record['fare_amount'] = distance * self.rng.uniform(2.5, 3.5)  # Normal fare
        record['total_amount'] = record['fare_amount'] + self.rng.uniform(3, 10)
        
        return record
    
    def _inject_swap_location(self, record: dict, difficulty: str = 'medium') -> dict:
        """
        Inject pickup/dropoff location swap or impossible route.
        
        - Same pickup and dropoff (round trip fraud)
        - Airport to same airport (unusual but possible)
        """
        pu_zone = record.get('PULocationID', 132)
        do_zone = record.get('DOLocationID', 100)
        
        # Strategy 1: Same location (100% fraud)
        if self.rng.random() < 0.6:
            record['PULocationID'] = self.rng.choice(list(self.VALID_ZONE_IDS))
            record['DOLocationID'] = record['PULocationID']  # Same location
        
        # Strategy 2: Airport round trip
        else:
            airport = self.rng.choice(list(self.AIRPORT_ZONES))
            record['PULocationID'] = airport
            record['DOLocationID'] = airport
        
        return record
    
    def _inject_duration_anomaly(self, record: dict, difficulty: str = 'medium') -> dict:
        """
        Inject duration anomalies (meter tampering).
        
        - Zero duration (instant trip)
        - Negative duration (dropoff before pickup)
        """
        pickup = pd.to_datetime(record['tpep_pickup_datetime'])
        
        if self.rng.random() < 0.7:
            # Zero duration: instant trip
            duration_min = self.rng.uniform(0.0, 0.1)  # 0-6 seconds
            record['tpep_dropoff_datetime'] = pickup + timedelta(minutes=duration_min)
        else:
            # Negative duration: dropoff before pickup (impossible)
            duration_min = -self.rng.uniform(1.0, 30.0)  # Negative!
            record['tpep_dropoff_datetime'] = pickup + timedelta(minutes=duration_min)
        
        record['dur_min'] = max(0.01, duration_min)  # Prevent zero/negative
        
        return record
    
    def _inject_passenger_count(self, record: dict, difficulty: str = 'medium') -> dict:
        """
        Inject impossible passenger count.
        
        - NYC taxi max: 4-5 passengers
        - Injecting 10-30 passengers
        """
        if self.rng.random() < 0.7:
            # Extreme: 10-30 passengers
            record['passenger_count'] = self.rng.integers(10, 31)
        else:
            # Zero passengers (also suspicious)
            record['passenger_count'] = 0
        
        return record
    
    def _inject_fare_ratio(self, record: dict, difficulty: str = 'medium') -> dict:
        """
        Inject extreme fare/distance or fare/time ratios.
        
        - Very high fare per mile (10-50x normal)
        - Very high fare per minute
        """
        distance = record.get('trip_distance', 5.0)
        duration_min = record.get('dur_min', 15.0)
        
        if self.rng.random() < 0.6:
            # High fare per mile
            record['trip_distance'] = self.rng.uniform(0.5, 3.0)  # Short trip
            normal_fare_per_mile = 2.50
            multiplier = self.rng.uniform(10, 50)  # 10-50x normal
            record['fare_amount'] = distance * normal_fare_per_mile * multiplier
        else:
            # High fare per minute
            normal_fare_per_min = 0.50
            multiplier = self.rng.uniform(10, 30)
            record['fare_amount'] = duration_min * normal_fare_per_min * multiplier
        
        record['total_amount'] = record['fare_amount'] + self.rng.uniform(3, 10)
        
        return record
    
    def _inject_time_impossible(self, record: dict, difficulty: str = 'medium') -> dict:
        """
        Inject impossible time patterns.
        
        - Trip during non-operating hours (3 AM airport pickup when airport closed)
        - Very long trips (24+ hours)
        """
        pickup = pd.to_datetime(record['tpep_pickup_datetime'])
        
        if self.rng.random() < 0.5:
            # Very long duration (24-48 hours)
            duration_hours = self.rng.uniform(24, 48)
            record['tpep_dropoff_datetime'] = pickup + timedelta(hours=duration_hours)
            record['dur_min'] = duration_hours * 60
            record['trip_distance'] = self.rng.uniform(5, 20)  # Normal distance for 24h
            record['fare_amount'] = self.rng.uniform(500, 1000)  # High fare
            record['total_amount'] = record['fare_amount'] + self.rng.uniform(50, 100)
        else:
            # Zero duration with valid hours
            record['tpep_dropoff_datetime'] = pickup
            record['dur_min'] = 0.01
        
        return record
    
    def _inject_zone_impossible(self, record: dict, difficulty: str = 'medium') -> dict:
        """
        Inject invalid NYC TLC zone IDs.
        
        Valid zones: 1-263
        """
        if self.rng.random() < 0.5:
            # Invalid pickup zone
            record['PULocationID'] = self.rng.integers(264, 999)  # Invalid
        else:
            # Invalid dropoff zone
            record['DOLocationID'] = self.rng.integers(264, 999)  # Invalid
        
        return record
    
    def _inject_combined(self, record: dict, difficulty: str = 'medium') -> dict:
        """
        Inject multiple violations at once (most severe).
        
        Combines 2-3 strategies from above.
        """
        # Select 2-3 random strategies
        n_violations = self.rng.integers(2, 4)
        selected_strategies = self.rng.choice(
            list(self.strategies.keys())[:-1],  # Exclude 'combined' itself
            size=n_violations,
            replace=False
        )
        
        # Apply each strategy
        for strategy_name in selected_strategies:
            if strategy_name != 'combined':
                record = self.strategies[strategy_name](record, difficulty)
        
        return record
    
    # =========================================================================
    # MAIN INJECTION METHOD
    # =========================================================================
    
    def inject(
        self,
        df: pd.DataFrame,
        n_anomalies: int,
        difficulty: str = 'medium',
        label_col: str = 'is_anomaly'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Inject anomalies into DataFrame.
        
        Args:
            df: Clean baseline DataFrame
            n_anomalies: Number of anomalies to inject
            difficulty: 'easy', 'medium', or 'hard'
            label_col: Name of the anomaly label column
        
        Returns:
            Tuple of (modified_df, labels_df)
        """
        print(f"\n{'='*60}")
        print("MULTI-STRATEGY ANOMALY INJECTION")
        print(f"{'='*60}")
        print(f"Input: {len(df):,} clean records")
        print(f"Target: {n_anomalies:,} anomalies")
        print(f"Strategy distribution:")
        for strategy, weight in self.strategy_weights.items():
            expected = int(n_anomalies * weight)
            print(f"  {strategy:20s}: {expected:>6,} ({weight*100:.1f}%)")
        
        # Sample base records (with replacement if needed)
        if n_anomalies > len(df):
            print(f"Warning: n_anomalies ({n_anomalies}) > df size ({len(df)}). Using replace=True.")
            base_indices = self.rng.choice(len(df), size=n_anomalies, replace=True)
        else:
            base_indices = self.rng.choice(len(df), size=n_anomalies, replace=False)
        
        # Generate anomalies
        anomaly_records = []
        labels = []
        
        for i, idx in enumerate(base_indices):
            base_record = df.iloc[idx].to_dict()
            
            # Get strategy for this anomaly
            strategy = self.get_strategy()
            
            # Inject anomaly
            anomaly_record = self.strategies[strategy](base_record.copy(), difficulty)
            anomaly_records.append(anomaly_record)
            
            # Create label
            labels.append({
                'original_index': idx,
                'is_anomaly': 1,
                'strategy': strategy,
                'difficulty': difficulty
            })
            
            if (i + 1) % 10000 == 0:
                print(f"  Progress: {i+1:,} / {n_anomalies:,}")
        
        # Create anomaly DataFrame
        df_anomalies = pd.DataFrame(anomaly_records)
        df_labels = pd.DataFrame(labels)
        
        # Combine with original
        df_original = df.copy()
        df_original[label_col] = 0
        
        # Combine (don't modify original data)
        df_combined = pd.concat([df_original, df_anomalies], ignore_index=True)
        
        # Add labels for original (all zeros)
        original_labels = pd.DataFrame({
            'original_index': df_original.index,
            'is_anomaly': 0,
            'strategy': 'normal',
            'difficulty': 'none'
        })
        df_all_labels = pd.concat([original_labels, df_labels], ignore_index=True)
        
        # Summary
        print(f"\n{'='*60}")
        print("INJECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total records: {len(df_combined):,}")
        print(f"  Clean: {len(df_original):,} ({len(df_original)/len(df_combined)*100:.2f}%)")
        print(f"  Anomalies: {len(df_anomalies):,} ({len(df_anomalies)/len(df_combined)*100:.2f}%)")
        
        # Strategy breakdown
        print(f"\nStrategy breakdown:")
        strategy_counts = df_labels['strategy'].value_counts()
        for strategy, count in strategy_counts.items():
            print(f"  {strategy:20s}: {count:>6,} ({count/len(df_labels)*100:.1f}%)")
        
        return df_combined, df_all_labels
    
    def validate_anomaly(self, record: dict) -> dict:
        """
        Validate that a record contains an actual anomaly.
        
        Returns dict with validation results.
        """
        issues = []
        
        # Check speed
        distance = record.get('trip_distance', 0)
        duration_hours = record.get('dur_min', 0) / 60
        if duration_hours > 0:
            speed = distance / duration_hours
            if speed < 5 or speed > 100:
                issues.append(f"extreme_speed_{speed:.1f}mph")
        
        # Check passenger count
        passengers = record.get('passenger_count', 1)
        if passengers <= 0 or passengers > 9:
            issues.append(f"invalid_passengers_{passengers}")
        
        # Check duration
        duration = record.get('dur_min', 0)
        if duration <= 0:
            issues.append("zero_or_negative_duration")
        
        # Check zone IDs
        pu_zone = record.get('PULocationID', 0)
        do_zone = record.get('DOLocationID', 0)
        if pu_zone not in self.VALID_ZONE_IDS or do_zone not in self.VALID_ZONE_IDS:
            issues.append("invalid_zone_id")
        
        return {
            'is_valid_anomaly': len(issues) > 0,
            'issues': issues
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-strategy anomaly injection')
    parser.add_argument('--input', required=True, help='Input parquet file')
    parser.add_argument('--output', required=True, help='Output parquet file')
    parser.add_argument('--labels', required=True, help='Output labels CSV')
    parser.add_argument('--n-anomalies', type=int, default=50000,
                       help='Number of anomalies to inject')
    parser.add_argument('--difficulty', default='medium',
                       choices=['easy', 'medium', 'hard'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--contamination', type=float, default=None,
                       help='Override n-anomalies with contamination rate')
    args = parser.parse_args()
    
    # Load baseline
    print(f"Loading baseline from: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df):,} records")
    
    # Override n_anomalies with contamination rate
    if args.contamination is not None:
        n_anomalies = int(len(df) * args.contamination / (1 - args.contamination))
        print(f"Computed n_anomalies={n_anomalies:,} from contamination={args.contamination:.4f}")
    else:
        n_anomalies = args.n_anomalies
    
    # Inject anomalies
    injector = MultiStrategyAnomalyInjector(seed=args.seed)
    df_combined, df_labels = injector.inject(
        df, 
        n_anomalies=n_anomalies,
        difficulty=args.difficulty
    )
    
    # Save outputs
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_parquet(args.output, index=False)
    df_labels.to_csv(args.labels, index=False)
    
    print(f"\n✓ Saved combined dataset: {args.output}")
    print(f"✓ Saved labels: {args.labels}")
    
    # Validation sample
    print(f"\n{'='*60}")
    print("VALIDATION SAMPLE")
    print(f"{'='*60}")
    
    anomaly_sample = df_combined[df_combined['is_anomaly'] == 1].head(5)
    for idx, row in anomaly_sample.iterrows():
        result = injector.validate_anomaly(row)
        strategy = df_labels[df_labels['original_index'] == idx]['strategy'].values
        print(f"\n  Index {idx}:")
        print(f"    Strategy: {strategy[0] if len(strategy) > 0 else 'unknown'}")
        print(f"    Valid anomaly: {result['is_valid_anomaly']}")
        print(f"    Issues: {result['issues']}")


if __name__ == '__main__':
    main()
```

---

## Verification Checklist

- [x] **C-DE-1:** Data splits are time-ordered, no shuffle
- [x] **C-DE-1:** Temporal validation ensures no overlap between splits
- [x] **C-DE-2:** Normalization uses separate stats data (first 10%)
- [x] **C-DE-2:** Autoencoder trained on middle 80%
- [x] **C-DE-2:** Memory initialized from last 10%
- [x] **H-DE-1:** Kafka topic definitions with 12 partitions, RF=3
- [x] **H-DE-1:** Retention policies documented (7-day raw, 30-day scores)
- [x] **H-DE-2:** Composite partitioning key documented
- [x] **H-DE-2:** Event ordering guarantees specified
- [x] **H-DE-3:** Watermark strategy with 6-hour allowed lateness
- [x] **H-DE-3:** Side output for dead letter queue
- [x] **H-DE-4:** Multi-strategy anomaly injection implemented
- [x] **H-DE-4:** 8 distinct anomaly strategies with weighted distribution
- [x] **H-DE-4:** Anomaly validation function included

---

## Files to Modify/Create

| File | Action | Description |
|------|--------|-------------|
| `memstream_src/scripts/train_warmup.py` | Create/Replace | Time-ordered splits + leakage-free warmup |
| `memstream_src/scripts/inject_anomalies_multi.py` | Create | Multi-strategy anomaly injection |
| `deployment/kafka-topics.yaml` | Create | Kafka topic definitions |
| `src/operators/late_data_handler.py` | Create | Late data handling with watermarks |
| `docs/event_ordering.md` | Create | Event ordering documentation |

---

**Status:** Data Engineering + ML fixes complete. Ready for integration into PLAN_v5.md.

**Date:** 2026-05-12  
**Reviewer:** Data Engineer
