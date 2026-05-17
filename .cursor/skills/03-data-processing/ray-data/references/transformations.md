# Ray Data Transformations

Reference guide for data transformation patterns using Ray Data for streaming data quality and context-aware systems.

## When to Use

Use these patterns when transforming records in a data pipeline. Ray Data provides scalable, distributed transformations that can be chained into complex pipelines.

## Core Transformation Patterns

### 1. Basic Map Transformations

```python
import ray.data
import pandas as pd

# Row-level transform
def normalize_record(record: dict) -> dict:
    record["value_normalized"] = (record["value"] - record["mean"]) / record["std"]
    return record

ds = ds.map(normalize_record)

# Batch-level transform (more efficient)
def normalize_batch(batch: pd.DataFrame) -> pd.DataFrame:
    batch["value_normalized"] = (
        (batch["value"] - batch["mean"]) / batch["std"]
    ).clip(-5, 5)  # Outlier clipping
    return batch

ds = ds.map_batches(normalize_batch, batch_format="pandas")
```

### 2. Context Window Transformations

```python
def add_temporal_context(batch: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal context features for context-aware processing.
    """
    batch["timestamp"] = pd.to_datetime(batch["timestamp"])
    batch = batch.sort_values(["entity_id", "timestamp"])

    # Time-based windows
    batch["minute"] = batch["timestamp"].dt.floor("1min")
    batch["hour"] = batch["timestamp"].dt.floor("1h")
    batch["day"] = batch["timestamp"].dt.floor("1D")

    # Time since last event
    batch["time_since_last"] = batch.groupby("entity_id")["timestamp"].diff().dt.total_seconds()

    # Event count in window
    batch["event_count_5min"] = batch.groupby("entity_id")["timestamp"].transform(
        lambda x: x.between(x - pd.Timedelta("5min"), x).sum()
    )

    return batch


def add_rolling_features(batch: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling window features for anomaly detection.
    """
    batch = batch.sort_values(["entity_id", "timestamp"])

    for window in [5, 10, 30]:
        batch[f"rolling_mean_{window}"] = (
            batch.groupby("entity_id")["value"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        batch[f"rolling_std_{window}"] = (
            batch.groupby("entity_id")["value"]
            .transform(lambda x: x.rolling(window, min_periods=1).std())
        )
        batch[f"z_score_{window}"] = (
            (batch["value"] - batch[f"rolling_mean_{window}"])
            / batch[f"rolling_std_{window}"].clip(lower=1e-6)
        )

    return batch
```

### 3. Data Quality Transformations

```python
def apply_quality_transforms(batch: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Apply data quality transformation rules.
    """
    for col, rule in rules.items():
        # Type coercion
        if rule.get("coerce_type"):
            batch[col] = batch[col].astype(rule["coerce_type"])

        # Null imputation
        if "impute_strategy" in rule:
            if rule["impute_strategy"] == "mean":
                batch[col] = batch[col].fillna(batch[col].mean())
            elif rule["impute_strategy"] == "forward_fill":
                batch[col] = batch.groupby("entity_id")[col].transform(
                    lambda x: x.fillna(method="ffill")
                )

        # Outlier handling
        if "clip_percentiles" in rule:
            lower, upper = rule["clip_percentiles"]
            batch[col] = batch[col].clip(
                batch[col].quantile(lower),
                batch[col].quantile(upper)
            )

    return batch


def add_quality_flags(batch: pd.DataFrame) -> pd.DataFrame:
    """
    Add quality flag columns for each record.
    """
    batch["has_nulls"] = batch.isnull().any(axis=1)
    batch["outlier_flag"] = batch["z_score"].abs() > 3
    batch["completeness"] = 1 - batch.isnull().sum(axis=1) / len(batch.columns)

    return batch
```

### 4. Feature Engineering Transformations

```python
def engineer_features(batch: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for context-aware and streaming quality models.
    """
    # Categorical encoding
    for col in ["status", "priority", "category"]:
        if col in batch.columns:
            batch[f"{col}_encoded"] = pd.Categorical(batch[col]).codes

    # Interaction features
    if "value_a" in batch.columns and "value_b" in batch.columns:
        batch["value_ratio"] = batch["value_a"] / batch["value_b"].clip(lower=1e-6)
        batch["value_product"] = batch["value_a"] * batch["value_b"]

    # Temporal features
    if "timestamp" in batch.columns:
        batch["hour_sin"] = np.sin(2 * np.pi * batch["timestamp"].dt.hour / 24)
        batch["hour_cos"] = np.cos(2 * np.pi * batch["timestamp"].dt.hour / 24)
        batch["is_weekend"] = batch["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

    # Lag features (for time-series patterns)
    for lag in [1, 3, 7]:
        batch[f"value_lag_{lag}"] = batch.groupby("entity_id")["value"].shift(lag)

    # Delta features
    batch["value_delta"] = batch.groupby("entity_id")["value"].diff()

    return batch


def anonymize_pii(batch: pd.DataFrame, pii_columns: list) -> pd.DataFrame:
    """
    Anonymize PII columns for privacy compliance.
    """
    import hashlib

    for col in pii_columns:
        if col in batch.columns:
            # Hash-based pseudonymization
            batch[f"{col}_anon"] = batch[col].apply(
                lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
                if pd.notna(x) else None
            )

    return batch
```

### 5. Streaming Aggregation Transformations

```python
def streaming_aggregate(batch: pd.DataFrame, window_seconds: int = 60) -> pd.DataFrame:
    """
    Aggregate metrics over a sliding window for streaming data.
    """
    batch["window_start"] = (
        batch["timestamp"]
        .dt.floor(f"{window_seconds}s")
    )

    aggregated = batch.groupby(["entity_id", "window_start"]).agg(
        count=("value", "count"),
        sum_value=("value", "sum"),
        mean_value=("value", "mean"),
        std_value=("value", "std"),
        min_value=("value", "min"),
        max_value=("value", "max"),
        null_count=("value", lambda x: x.isnull().sum())
    ).reset_index()

    aggregated["completeness"] = 1 - (aggregated["null_count"] / aggregated["count"])

    return aggregated
```

### 6. State-Aware Transformations

```python
from ray.data import ActorPoolStrategy

class ContextTracker:
    """Actor for maintaining context state across batches."""

    def __init__(self):
        self.context_window = {}
        self.max_window_size = 1000

    def update(self, batch: pd.DataFrame) -> pd.DataFrame:
        for _, row in batch.iterrows():
            entity = row["entity_id"]
            if entity not in self.context_window:
                self.context_window[entity] = []
            self.context_window[entity].append(row.to_dict())

            # Trim window
            if len(self.context_window[entity]) > self.max_window_size:
                self.context_window[entity] = self.context_window[entity][-self.max_window_size:]

        return batch

    def get_context(self, entity_id: str, lookback: int = 10) -> list:
        return self.context_window.get(entity_id, [])[-lookback:]


tracker = ContextTracker()

ds.map_batches(
    lambda batch: tracker.update(batch),
    compute=ActorPoolStrategy(min_size=1, max_size=4)
)
```

## Transformation Chain Template

```python
def build_pipeline(ds: ray.data.Dataset, config: dict) -> ray.data.Dataset:
    """
    Build a complete data quality pipeline.
    """
    return (
        ds
        .map_batches(normalize_record)
        .map_batches(add_temporal_context)
        .map_batches(add_rolling_features)
        .map_batches(engineer_features)
        .map_batches(add_quality_flags)
        .filter(predicate=lambda x: x["completeness"] > 0.9)
        .repartition(config["num_partitions"])
    )
```

## Output Schema

| Field | Type | Description |
|-------|------|-------------|
| record_id | string | Unique record identifier |
| timestamp | datetime | Event timestamp |
| entity_id | string | Entity (user/device/session) ID |
| value | float | Primary value |
| quality_score | float | Overall quality score [0-1] |
| is_anomaly | bool | Anomaly detection flag |
| features_* | various | Engineered features |
