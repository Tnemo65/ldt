# Ray Data Integration

Reference guide for integrating Ray Data with streaming pipelines for context-aware and streaming data quality systems.

## When to Use

Use Ray Data when processing large-scale data pipelines that need:
- Parallel data processing across distributed workers
- Integration with streaming sources (Kafka, Kinesis, custom)
- GPU-accelerated data transforms
- Seamless scaling from prototype to production

## Integration Patterns

### 1. Kafka to Ray Data

```python
import ray
import ray.data

def kafka_to_ray_dataset(
    bootstrap_servers: list,
    topic: str,
    stream_mode: str = "monotonically_increasing",
    stop_condition: callable = None
):
    """
    Read from Kafka into a Ray Dataset.
    stream_mode: 'newest', 'oldest', 'monotonically_increasing'
    """
    ds = ray.data.read_kafka(
        brokers=bootstrap_servers,
        topic=topic,
        stream_buffer_size_blocks=64,
        ray_remote_args={"num_cpus": 2}
    )

    return ds

# Streaming mode
ds_stream = ray.data.read_kafka(
    brokers=["kafka:9092"],
    topic="data-quality-events",
    stream_mode="newest"
)

# Process streaming
for batch in ds_stream.iter_batches(batch_format="pandas"):
    process_quality_check(batch)
```

### 2. Batch Processing with Ray Data

```python
import ray.data
from ray.data.context import DataContext

# Optimize for large-scale processing
ctx = DataContext.get_current()
ctx.set_config(
    "resource_manager",
    {"object_store_memory_fraction": 0.6}
)

ds = ray.data.read_parquet("s3://data-lake/raw/")

# Transform with parallelism
ds = (
    ds
    .map_batches(process_quality_batch, concurrency=8)
    .filter(predicate=quality_filter)
    .repartition(32)
    .write_parquet("s3://data-lake/processed/")
)
```

### 3. Flink to Ray Data Bridge

```python
# For Flink pipelines sending data to Ray workers
def flink_ray_sink(ds: ray.data.Dataset, output_topic: str):
    """
    Write Ray Dataset results back to Kafka for downstream Flink consumption.
    """
    def convert_to_kafka_records(batch: pd.DataFrame) -> list:
        records = []
        for _, row in batch.iterrows():
            records.append({
                "topic": output_topic,
                "key": str(row["record_id"]).encode(),
                "value": row.to_json().encode()
            })
        return records

    ds.map_batches(convert_to_kafka_records)
```

### 4. Context-Aware Feature Pipeline

```python
import ray.data

def build_context_features(ds: ray.data.Dataset) -> ray.data.Dataset:
    """
    Build context-aware features for streaming data.
    """
    def enrich_with_context(batch: pd.DataFrame) -> pd.DataFrame:
        # Add contextual features
        batch["window_id"] = batch["timestamp"].dt.floor("5min")
        batch["hour_of_day"] = batch["timestamp"].dt.hour
        batch["day_of_week"] = batch["timestamp"].dt.dayofweek

        # Rolling statistics per entity
        batch = batch.sort_values(["entity_id", "timestamp"])
        batch["rolling_mean"] = (
            batch.groupby("entity_id")["value"]
            .transform(lambda x: x.rolling(10, min_periods=1).mean())
        )

        # Context deviation score
        batch["context_score"] = (
            (batch["value"] - batch["rolling_mean"]).abs()
            / batch["rolling_mean"].clip(lower=1e-6)
        )

        return batch

    return ds.map_batches(enrich_with_context, batch_format="pandas")
```

### 5. Streaming Quality Assessment

```python
def streaming_quality_assessment(ds: ray.data.Dataset) -> dict:
    """
    Assess data quality metrics across the dataset.
    """
    quality_stats = ds.map_batches(
        compute_batch_quality,
        batch_format="pandas",
        reduce_fn=aggregate_quality_stats
    ).take()

    return quality_stats

def compute_batch_quality(batch: pd.DataFrame) -> dict:
    return {
        "total_records": len(batch),
        "null_counts": batch.isnull().sum().to_dict(),
        "schema_drift": detect_schema_drift(batch),
        "completeness": 1 - (batch.isnull().sum().sum() / batch.size)
    }
```

## Ray Data + Spark Compatibility

```python
# For environments running both Ray and Spark
def ray_to_spark(ray_ds: ray.data.Dataset, spark_session) -> "DataFrame":
    """
    Convert Ray Dataset to Spark DataFrame for Spark-compatible environments.
    """
    # Convert Ray to Pandas, then to Spark
    pandas_df = ray_ds.take_all()
    return spark_session.createDataFrame(pandas_df)


def spark_to_ray(spark_df: "DataFrame") -> ray.data.Dataset:
    """
    Convert Spark DataFrame to Ray Dataset.
    """
    return ray.data.from_spark(spark_df)
```

## Performance Tuning

```python
# Optimize Ray Data for streaming workloads
ray.data.DataContext.get_current().set_config(
    # Tune block size for streaming
    "target_max_block_size", 64 * 1024 * 1024  # 64 MB
)

# Enable actor pool strategy for stateful ops
ds.map_batches(
    MyStatefulClass,
    compute=ray.data.ActorPoolStrategy(min_size=4, max_size=16)
)
```

## Resource Allocation

| Operation | CPU | Memory | GPU |
|-----------|-----|--------|-----|
| Read (Kafka/S3) | 2-4 per worker | 4GB per worker | No |
| Transform (CPU) | 4-8 per worker | 8GB per worker | No |
| Transform (GPU) | 2 per worker | 4GB per worker | Yes |
| Write | 2-4 per worker | 4GB per worker | No |
