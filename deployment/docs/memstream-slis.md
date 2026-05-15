# MemStream Operational SLIs (Phase 5B)

This document defines the Service Level Indicators (SLIs) for MemStream monitoring,
tracking key operational metrics for the streaming anomaly detection system.

## Overview

MemStream SLIs are organized into three categories:
1. **Availability** - System uptime and accessibility
2. **Performance** - Latency and throughput metrics
3. **Quality** - Detection accuracy and model health

## SLI Definitions

### Availability SLIs

| SLI Name | Metric | SLO Target | Critical Threshold | Alert |
|----------|--------|------------|-------------------|-------|
| Uptime | `up{job="memstream"}` == 1 | 99.9% | < 99.5% | MemStreamDown |
| Redis Connectivity | `memstream_redis_connected` | 100% | < 95% | MemStream_RedisConnectionFailure |
| Model HMAC Valid | `rate(memstream_hmac_verification_failures_total)` == 0 | 100% | > 0 | MemStream_HMACVerificationFailures |

### Performance SLIs

| SLI Name | Metric | SLO Target | Critical Threshold | Alert |
|----------|--------|------------|-------------------|-------|
| **Warmup Time** | `memstream_warmup_duration_seconds` | < 30 min | > 45 min | MemStream_WarmupIncomplete |
| **Checkpoint Save Latency** | `histogram_quantile(0.99, rate(memstream_checkpoint_save_duration_seconds_bucket))` | < 30s | > 60s | MemStream_CheckpointLatency |
| **Checkpoint Size** | `memstream_checkpoint_size_bytes` | < 500 MB | > 1 GB | - |
| **Model Loading Time** | `memstream_model_loading_duration_seconds` | < 60s | > 120s | - |
| **kNN Query Latency P50** | `histogram_quantile(0.50, rate(memstream_knn_query_duration_seconds_bucket))` | < 10ms | > 50ms | - |
| **kNN Query Latency P95** | `histogram_quantile(0.95, rate(memstream_knn_query_duration_seconds_bucket))` | < 50ms | > 200ms | MemStream_KNNQueryLatencyHigh |
| **kNN Query Latency P99** | `histogram_quantile(0.99, rate(memstream_knn_query_duration_seconds_bucket))` | < 100ms | > 500ms | - |
| Scoring Latency P99 | `histogram_quantile(0.99, rate(memstream_scoring_latency_seconds_bucket))` | < 100ms | > 500ms | - |

### Quality SLIs

| SLI Name | Metric | SLO Target | Critical Threshold | Alert |
|----------|--------|------------|-------------------|-------|
| **Memory Horizon** | `memstream_memory_cycle_seconds` | > 300s | < 300s for 10m | MemStream_MemoryHorizonViolation |
| **Cell Coverage** | `min(memstream_contextbeta_cell_samples)` | >= 10 per cell | < 10 for 5m | MemStream_CellCoverageLow |
| **kNN Stability** | `memstream_knn_distance_rolling / memstream_knn_distance_baseline` | < 1.5x | > 2.0x for 1h | MemStream_KNNStabilityViolation |
| **Beta Staleness** | `max(memstream_beta_staleness_seconds)` | < 3600s | > 3600s for 15m | MemStream_BetaStalenessHigh |
| **IEC Coverage** | `count(memstream_adwin_active == 1)` | 10 neighborhoods | < 10 for 30m | MemStream_IECNeighborhoodCoverageLow |
| Anomaly Rate | `memstream_anomaly_rate` | < 15% | > 15% for 4h | MemStream_AnomalyRateSpike |
| Critical Anomaly Rate | `memstream_anomaly_rate` | < 25% | > 25% for 2h | MemStream_CriticalAnomalyRate |
| **Memory Utilization** | `memstream_memory_utilization` | < 80% | > 90% for 15m | MemStream_MemoryUtilizationHigh |
| Circuit Breaker | `memstream_circuit_breaker_state` == 0 | Closed | Open for 5m | MemStream_CircuitBreakerTripped |

**Bold** = Phase 5B new SLIs

## SLI Calculation Methods

### Warmup Time

```promql
# Time from MemStream start to warmup completion
time() - memstream_start_time{status="running"}
```

**Measurement**: Record `memstream_warmup_start_timestamp` when system starts,
and `memstream_warmup_complete_timestamp` when warmup completes.
Warmup duration = `warmup_complete - warmup_start`.

**Target**: < 30 minutes for 100K memory, 500 warmup epochs.

### Checkpoint Size

```promql
# Current checkpoint size in bytes
memstream_checkpoint_size_bytes
```

**Measurement**: Size of `memstream_memory.pt` in MinIO at each save cycle.

**Target**: < 500 MB for 100K memory slots.

### Checkpoint Save Latency

```promql
# P99 checkpoint save duration
histogram_quantile(0.99,
  rate(memstream_checkpoint_save_duration_seconds_bucket[5m])
)
```

**Measurement**: Time from checkpoint save start to completion (including HMAC).

**Target**: < 30 seconds (SLO), > 60 seconds triggers warning.

### Model Loading Time

```promql
# Time to load model from MinIO
memstream_model_loading_duration_seconds
```

**Measurement**: Time from job restart to model availability.

**Target**: < 60 seconds for full model.

### kNN Query Latency

```promql
# P50 kNN query latency
histogram_quantile(0.50,
  rate(memstream_knn_query_duration_seconds_bucket[5m])
)

# P95 kNN query latency
histogram_quantile(0.95,
  rate(memstream_knn_query_duration_seconds_bucket[5m])
)

# P99 kNN query latency
histogram_quantile(0.99,
  rate(memstream_knn_query_duration_seconds_bucket[5m])
)
```

**Measurement**: Time from query submission to result return, including memory fetch.

**Target**: P50 < 10ms, P95 < 50ms, P99 < 100ms.

### Memory Utilization Per Neighborhood

```promql
# Memory utilization by neighborhood
memstream_memory_utilization{neighborhood=~"$neighborhood"}
```

**Measurement**: Ratio of used memory slots to total slots per neighborhood.

**Target**: < 80% average, alert at > 90%.

## Grafana Dashboard Queries

### Memory Horizon Over Time

```promql
memstream_memory_cycle_seconds
```

Visualization: Time series with threshold line at 300s.

### Cell Coverage Heatmap

```promql
memstream_contextbeta_cell_samples
```

Visualization: Heatmap showing sample counts per (neighborhood, cell_id).

### kNN Distance vs Baseline

```promql
memstream_knn_distance_rolling / memstream_knn_distance_baseline
```

Visualization: Ratio over time with 2.0x threshold line.

### Beta Staleness Per Neighborhood

```promql
memstream_beta_staleness_seconds{neighborhood=~"$neighborhood"}
```

Visualization: Time series per neighborhood with 3600s threshold.

### IEC Neighborhood Coverage Gauge

```promql
count(memstream_adwin_active == 1) by (neighborhood)
```

Visualization: Gauge showing active ADWIN count per neighborhood (target: 10).

## Alert Thresholds Summary

| Alert Name | Metric | Threshold | Duration | Severity |
|------------|--------|-----------|----------|----------|
| MemStream_MemoryHorizonViolation | `memstream_memory_cycle_seconds` | < 300 | 10m | warning |
| MemStream_CellCoverageLow | `min(memstream_contextbeta_cell_samples)` | < 10 | 5m | warning |
| MemStream_KNNStabilityViolation | kNN ratio | > 2.0 | 1h | warning |
| MemStream_BetaStalenessHigh | `memstream_beta_staleness_seconds` | > 3600 | 15m | warning |
| MemStream_IECNeighborhoodCoverageLow | count ADWIN | < 10 | 30m | warning |
| MemStream_CheckpointLatency | checkpoint duration | > 30s | 5m | warning |
| MemStream_KNNQueryLatencyHigh | P95 kNN | > 100ms | 10m | warning |
| MemStream_MemoryUtilizationHigh | utilization | > 90% | 15m | warning |
| MemStream_WarmupIncomplete | warmup_complete | == 0 | 30m | warning |
| MemStream_CheckpointCorruption | corruption rate | > 0 | 1m | critical |
| MemStream_HMACVerificationFailures | HMAC failures | > 0 | 1m | critical |
| MemStream_ModelHMACFailure | model HMAC failures | > 0 | 1m | critical |
| MemStream_RedisConnectionFailure | Redis failures | > 0 | 2m | warning |
| MemStream_CircuitBreakerTripped | circuit state | == 1 | 5m | warning |

## SLO Error Budget

For each SLO, error budget tracking:

| SLO | Target | 30-day Budget | 7-day Budget | Burn Rate Alert |
|-----|--------|---------------|--------------|-----------------|
| Warmup Time | 99% complete in 30m | 7.2 hours | 1.68 hours | > 10x |
| Checkpoint Latency | 99.9% under 30s | 43 minutes | 10 minutes | > 10x |
| kNN P99 Latency | 99% under 100ms | 7.2 minutes | 1.68 minutes | > 10x |
| Memory Horizon | 99% above 300s | 7.2 hours | 1.68 hours | > 10x |

## SLI Recording

MemStream exposes the following metrics for SLI tracking:

```python
# In memstream_scoring_operator.py - metrics exposed:

# Performance SLIs
'MemStream_ScoringLatencyMs'          # Histogram
'MemStream_CheckpointSaveDuration'     # Histogram
'MemStream_CheckpointSizeBytes'        # Gauge
'MemStream_ModelLoadingDuration'       # Gauge
'MemStream_WarmupDuration'             # Gauge
'MemStream_KNNQueryDuration'           # Histogram

# Quality SLIs
'MemStream_MemoryCycleSeconds'         # Gauge
'MemStream_ContextBetaCellSamples'     # Gauge (per cell)
'MemStream_KNNDistanceBaseline'        # Gauge
'MemStream_KNNDistanceRolling'         # Gauge
'MemStream_BetaStalenessSeconds'       # Gauge (per neighborhood)
'MemStream_ADWINActive'                # Gauge (per neighborhood)
'MemStream_MemoryUtilization'          # Gauge (per neighborhood)
'MemStream_AnomalyRate'                # Gauge

# Security SLIs
'MemStream_HMACVerificationFailures'   # Counter
'MemStream_ModelHMACFailures'          # Counter
'MemStream_CheckpointCorruption'        # Counter

# Availability SLIs
'MemStream_RedisConnected'             # Gauge
'MemStream_RedisConnectionFailures'     # Counter
'MemStream_CircuitBreakerState'        # Gauge
```

## Document History

- **Phase 5B**: Initial SLI definitions added
  - Added Memory Horizon SLI
  - Added Cell Coverage SLI
  - Added kNN Stability SLI
  - Added Beta Staleness SLI
  - Added IEC Neighborhood Coverage SLI
  - Added Checkpoint Size SLI
  - Added Model Loading Time SLI
  - Added kNN Query Latency SLIs (P50, P95, P99)
  - Added Memory Utilization SLI
