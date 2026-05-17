# NeMo-Curator Deduplication

Reference guide for deduplication strategies using NeMo-Curator for streaming data quality.

## When to Use

Use these patterns when deduplicating records in a data stream to improve data quality. Deduplication is critical for streaming pipelines where records may arrive multiple times due to retries, late arrivals, or upstream reprocessing.

## Deduplication Strategies

### 1. Exact Deduplication (Primary Key Based)

```python
import pandas as pd
from nemo_curator import ExactDuplicates

# Define primary key columns
duplicates_remover = ExactDuplicates(
    columns=["record_id"],
    bytes_per_item=1000
)

# Apply deduplication
df = duplicates_remover(df)
```

### 2. Fuzzy Deduplication (Semantic)

```python
from nemo_curator.scripts.fuzzy_dedup import get_fuzzy_dedup_module

fuzzy_dedup = get_fuzzy_dedup_module(
    match_threshold=0.8,
    tokenizer="nemo_wb",
    input_is_jsonl=True
)

result_df = fuzzy_dedup(df)
```

### 3. MinHash Deduplication (Large-Scale)

```python
from nemo_curator.utils.modular_imports import import_from

# For document-level deduplication
minhash_dedup = import_from(
    "nemo_curator.modules.fuzzy_dedup",
    "FuzzyDuplicates"
)

minhash = MinHash(
    num_hashes=128,
    minhash_lsh=True,
    bands=32,
    rows=4
)
```

### 4. Streaming Window Deduplication

```python
# Deduplicate within sliding windows (important for streaming)
def streaming_dedup(stream_df, key_col="record_id", window_ms=60000):
    """Deduplicate records within a time window."""
    stream_df = stream_df.withColumn(
        "window_start",
        (F.col("timestamp_ms") / window_ms).cast("long") * window_ms
    )

    return (
        stream_df
        .dropDuplicates([key_col, "window_start"])
    )
```

## Streaming-Specific Patterns

### Late-Arrival Handling

```python
def deduplicate_with_late_arrivals(df, key_col, grace_period_ms=300000):
    """
    Handle late-arriving duplicates with a grace period.
    Records arriving within grace_period_ms are considered updates.
    """
    df = df.withColumn(
        "_event_time",
        F.col("timestamp_ms") - F.lit(grace_period_ms)
    )

    return (
        df
        .withColumn(
            "_rank",
            F.row_number().over(
                Window.partitionBy(key_col)
                      .orderBy(F.col("_event_time").desc())
            )
        )
        .filter(F.col("_rank") == 1)
        .drop("_rank", "_event_time")
    )
```

### Exactly-Once Semantics

```python
# For systems requiring exactly-once delivery
def deduplicate_exactly_once(df, key_col, event_time_col="timestamp_ms"):
    """
    Deduplicate with watermarking for exactly-once semantics.
    """
    watermark_duration = "5 minutes"

    return (
        df
        .withWatermark(event_time_col, watermark_duration)
        .dropDuplicatesWithinWatermark([key_col])
    )
```

## Quality Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Duplicate Rate | `duplicates / total_records` | < 1% |
| Recall | `true_positives / (true_positives + false_negatives)` | > 99% |
| Precision | `true_positives / (true_positives + false_positives)` | > 95% |

## Output Files

After deduplication, log the following metrics:
- Input record count
- Output record count
- Duplicates removed count
- Duplicate rate percentage
