# Data Quality Metrics Reference

Detailed formulas and implementations for data quality metrics.

## Completeness Metrics

### Missing Value Rate
```python
def missing_value_rate(df):
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    return missing_cells / total_cells
```

### Schema Coverage
```python
def schema_coverage(actual_schema, expected_schema):
    expected_fields = set(expected_schema.keys())
    actual_fields = set(actual_schema.keys())
    return len(expected_fields & actual_fields) / len(expected_fields)
```

## Accuracy Metrics

### Type Accuracy
```python
def type_accuracy(df, schema):
    correct = 0
    total = 0
    for col, dtype in schema.items():
        if col in df.columns:
            if dtype == "int":
                correct += df[col].dropna().apply(lambda x: isinstance(x, int)).sum()
            elif dtype == "float":
                correct += df[col].dropna().apply(lambda x: isinstance(x, (int, float))).sum()
            elif dtype == "str":
                correct += df[col].dropna().apply(lambda x: isinstance(x, str)).sum()
            total += df[col].dropna().shape[0]
    return correct / total if total > 0 else 0.0
```

### Range Accuracy
```python
def range_accuracy(df, range_constraints):
    valid = 0
    total = 0
    for col, (min_val, max_val) in range_constraints.items():
        if col in df.columns:
            in_range = ((df[col] >= min_val) & (df[col] <= max_val)).sum()
            valid += in_range
            total += df[col].dropna().shape[0]
    return valid / total if total > 0 else 0.0
```

## Timeliness Metrics

### Latency Percentiles
```python
def latency_percentiles(df, event_col, process_col):
    df = df.copy()
    df["latency_ms"] = (df[process_col] - df[event_col]).dt.total_seconds() * 1000
    return {
        "p50": df["latency_ms"].quantile(0.50),
        "p95": df["latency_ms"].quantile(0.95),
        "p99": df["latency_ms"].quantile(0.99)
    }
```

## Consistency Metrics

### Duplicate Rate
```python
def duplicate_rate(df, key_columns):
    n_unique = df[key_columns].drop_duplicates().shape[0]
    n_total = df.shape[0]
    return (n_total - n_unique) / n_total
```

### Drift Detection
```python
from scipy.stats import ks_2samp

def distribution_drift(baseline_col, current_col):
    stat, p_value = ks_2samp(baseline_col, current_col)
    return {"ks_statistic": stat, "p_value": p_value, "drifted": p_value < 0.05}
```

## Statistical Reporting Standards

- Report all metrics with: point estimate + 95% CI + sample size
- For streaming: use rolling windows with CI via bootstrap
- Threshold violations: report as percentage of time below threshold
