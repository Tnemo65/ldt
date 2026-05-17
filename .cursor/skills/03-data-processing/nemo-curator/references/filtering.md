# NeMo-Curator Data Filtering

Reference guide for data filtering strategies using NeMo-Curator for streaming data quality.

## When to Use

Use these patterns when filtering records in a data stream to enforce data quality rules. Filtering is the first line of defense against bad data entering downstream systems.

## Filtering Strategies

### 1. Rule-Based Filtering

```python
import pandas as pd
import numpy as np

def apply_quality_filters(df, rules_config):
    """
    Apply configurable quality rules to a dataframe.
    rules_config: dict of {column: {rule: value}}
    """
    mask = pd.Series([True] * len(df))

    for col, rules in rules_config.items():
        if "min" in rules:
            mask &= (df[col] >= rules["min"])
        if "max" in rules:
            mask &= (df[col] <= rules["max"])
        if "not_null" in rules and rules["not_null"]:
            mask &= df[col].notna()
        if "allowed_values" in rules:
            mask &= df[col].isin(rules["allowed_values"])
        if "regex" in rules:
            mask &= df[col].str.match(rules["regex"], na=False)

    return df[mask]
```

### 2. Streaming Filtering with Spark

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def streaming_quality_filter(df, config):
    """
    Apply quality filters in a Spark streaming context.
    """
    for col, rules in config["columns"].items():
        if rules.get("required", False):
            df = df.filter(F.col(col).isNotNull())

        if "min" in rules:
            df = df.filter(F.col(col) >= rules["min"])

        if "max" in rules:
            df = df.filter(F.col(col) <= rules["max"])

        if "min_length" in rules:
            df = df.filter(F.length(F.col(col)) >= rules["min_length"])

        if "max_length" in rules:
            df = df.filter(F.length(F.col(col)) <= rules["max_length"])

    return df
```

### 3. Schema Validation Filtering

```python
from pyspark.sql.types import StructType

def schema_filter(df, expected_schema: StructType):
    """
    Filter records that don't conform to the expected schema.
    """
    actual_fields = set(f.name for f in df.schema.fields)
    expected_fields = set(f.name for f in expected_schema.fields)

    # Records missing required fields
    missing = expected_fields - actual_fields
    if missing:
        print(f"WARNING: Missing fields: {missing}")

    # Records with extra fields (optional: filter or pass)
    extra = actual_fields - expected_fields

    # Select only expected columns
    return df.select(*expected_fields)
```

### 4. Cross-Record Consistency Filtering

```python
def cross_record_filter(df, consistency_rules):
    """
    Filter based on cross-record consistency checks.

    Example: filter records where sum(detail_amounts) != header.total_amount
    """
    violations = []

    for rule_name, rule in consistency_rules.items():
        col_a = rule["column_a"]
        col_b = rule["column_b"]
        tolerance = rule.get("tolerance", 0)

        diff = (df[col_a] - df[col_b]).abs()
        mask = diff <= tolerance

        n_violations = (~mask).sum()
        violations.append({
            "rule": rule_name,
            "violations": n_violations,
            "total": len(df)
        })

        df = df[mask]

    return df, violations
```

### 5. Anomaly-Based Filtering

```python
from sklearn.ensemble import IsolationForest
import numpy as np

def anomaly_filter(df, feature_cols, contamination=0.01):
    """
    Filter records detected as anomalies using Isolation Forest.
    """
    features = df[feature_cols].values

    clf = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    predictions = clf.fit_predict(features)
    scores = clf.decision_function(features)

    # Keep only normal records (-1 = anomaly, 1 = normal)
    df["anomaly_score"] = scores
    df["is_anomaly"] = (predictions == -1)

    return df[df["is_anomaly"] == False].drop(columns=["anomaly_score", "is_anomaly"])
```

## Streaming Patterns

### Rate-Based Filtering

```python
def filter_by_rate(df, group_col, max_rate_per_second=1000):
    """
    Filter records that exceed a rate threshold per group.
    Useful for detecting burst/anomaly patterns.
    """
    WINDOW_DURATION = "1 second"

    return (
        df
        .withColumn(
            "event_time",
            F.window(F.col("timestamp"), WINDOW_DURATION)
        )
        .withColumn("count", F.count("*").over(
            Window.partitionBy(group_col, "event_time")
        ))
        .filter(F.col("count") <= max_rate_per_second)
        .drop("count", "event_time")
    )
```

### Staleness Filtering

```python
def filter_stale_records(df, event_time_col, max_staleness_seconds=3600):
    """
    Filter records that are too old (stale) for the pipeline.
    """
    current_time = datetime.now()
    staleness_cutoff = current_time - timedelta(seconds=max_staleness_seconds)

    return df.filter(
        F.col(event_time_col) >= F.lit(staleness_cutoff)
    )
```

## Quality Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Pass Rate | `passed / total` | > 95% |
| Filter Rate | `filtered / total` | < 5% |
| Rule Coverage | `rules_applied / rules_defined` | 100% |

## Logging

Always log:
- Number of input records
- Number of output records
- Number of filtered records per rule
- Filter rate percentage
- Any anomalies detected
