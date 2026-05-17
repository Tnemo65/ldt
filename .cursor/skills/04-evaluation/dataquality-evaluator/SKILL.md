---
name: data-quality-evaluator
description: >
  Evaluation framework for streaming data quality assessment and data quality benchmarking.
  Use when evaluating data quality metrics, detecting anomalies in streaming pipelines,
  measuring data drift, assessing completeness/accuracy/timeliness, or comparing
  data quality across different pipelines or configurations. Applies to both Context-Aware
  and Streaming Data Quality research domains.
version: 1.0.0
tags: [Evaluation, Data-Quality, Streaming, Benchmark, Anomaly-Detection, Drift-Detection]
---

# Data Quality Evaluator

Comprehensive evaluation framework for streaming data quality assessment.

## When to Use

- Evaluating data quality metrics in streaming pipelines
- Measuring data completeness, accuracy, timeliness, consistency
- Detecting anomalies and data drift in streaming contexts
- Benchmarking data quality across pipeline configurations
- Assessing context-aware data processing quality
- Comparing baseline vs improved data quality methods

## Core Metrics Framework

### 1. Completeness Metrics

| Metric | Definition | Measurement |
|--------|-----------|-------------|
| Missing Value Rate | % of null/missing fields | count_missing / total_records |
| Schema Coverage | % of expected fields present | fields_present / fields_expected |
| Record Completeness | % of records with all required fields | complete_records / total_records |
| Temporal Completeness | % of expected time windows with data | windows_covered / windows_expected |

### 2. Accuracy Metrics

| Metric | Definition | Measurement |
|--------|-----------|-------------|
| Type Accuracy | % of values matching expected schema types | correct_types / total_values |
| Range Accuracy | % of values within valid ranges | valid_range / total_values |
| Reference Accuracy | % matching external reference/dimension tables | matches / total |
| Cross-field Consistency | % of records passing cross-field validation | valid_records / total |

### 3. Timeliness Metrics (Streaming)

| Metric | Definition | Measurement |
|--------|-----------|-------------|
| Latency (P50/P95/P99) | Time from event to processed | percentile(event_time, processed_time) |
| Freshness | Age of most recent data | current_time - latest_event_time |
| Processing Throughput | Events processed per second | total_events / total_time |
| Backlog Rate | Rate of event accumulation | backlog_delta / time_delta |

### 4. Consistency Metrics

| Metric | Definition | Measurement |
|--------|-----------|-------------|
| Duplicate Rate | % of duplicate records | duplicates / total_records |
| Contradiction Rate | % of logically contradicting records | contradictions / total_records |
| Drift Score | Distribution shift vs baseline | KL-divergence / Wasserstein distance |

### 5. Context-Aware Metrics

| Metric | Definition | Measurement |
|--------|-----------|-------------|
| Context Recall | Relevant context retrieved when needed | relevant_retrieved / relevant_total |
| Context Precision | Retrieved context is actually relevant | relevant_retrieved / retrieved_total |
| Context Latency | Time to retrieve relevant context | percentile retrieval_time |
| Context Freshness | How up-to-date is cached context | age_of_context_records |

## Evaluation Protocol

### Step 1: Define Benchmark Scope

```yaml
benchmark:
  scope:
    - completeness
    - accuracy
    - timeliness
    - consistency
    # Add context metrics for Context-Aware domain

  thresholds:
    missing_rate: < 0.01      # < 1%
    accuracy: > 0.99           # > 99%
    latency_p99: < 1000         # ms
    throughput: > 1000          # events/sec
```

### Step 2: Execute Evaluation

```python
def evaluate_data_quality(pipeline_output, benchmark_config):
    results = {}
    for metric_category in benchmark_config.scope:
        results[metric_category] = compute_metrics(
            pipeline_output,
            metric_category
        )
    return results
```

### Step 3: Statistical Validation

- Bootstrap confidence intervals (n=1000)
- Statistical significance tests (Wilcoxon for paired comparisons)
- Multiple comparison correction (Bonferroni / Benjamini-Hochberg)
- Effect sizes (Cohen's d) for metric improvements

### Step 4: Reporting

```yaml
report:
  summary:
    overall_score: 0.947
    grade: "A"  # Based on weighted metric scores
  details:
    completeness: 0.98  # grade: A
    accuracy: 0.99      # grade: A
    timeliness: 0.91     # grade: B
    consistency: 0.96   # grade: A
```

## Quality Gates

| Score | Grade | Action |
|-------|-------|--------|
| >= 0.95 | A | Excellent — proceed |
| 0.90-0.95 | B | Good — minor improvements possible |
| 0.80-0.90 | C | Acceptable — improvements recommended |
| < 0.80 | D/F | Poor — blocking issues |

## Integration

Use this skill alongside:
- `ml-training-recipes` — for experiment loop patterns
- `mlflow` or `tensorboard` — for tracking evaluation metrics
- `statistical-analysis` — for statistical validation of results
