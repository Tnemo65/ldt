# Monitoring/Observability Review - CA-DQStream + MemStream Hybrid v4

**Reviewer:** SRE/Observability Engineer  
**Date:** 2026-05-12  
**Plan Version:** v4  
**Status:** Requires fixes before production

---

## Summary

The plan defines SLOs in `SLOConfig` and includes a Prometheus alerting rules file (`prometheus_alerts.yaml`). However, the observability implementation is **incomplete and has critical gaps**. The plan mentions "Distributed tracing → OpenTelemetry" and "SLOs → defined and instrumented" as v1→v2→v3 fixes, but the actual instrumentation code is missing from the operator implementation. The alerts defined are sound in principle but cannot fire without the underlying metrics being exposed.

**Overall Assessment:** The alerting rules in `prometheus_alerts.yaml` are a good starting skeleton, but they reference metrics that are never exposed in the operator code. Key gaps include: no Prometheus client instrumentation in the operator, no OpenTelemetry spans in the critical paths, no structured logging correlation IDs, no SLO burn-rate alerts, and no dashboard definition.

---

## SLO Analysis

### Defined SLOs

| SLO | Target | Source | Measurable? |
|-----|--------|--------|-------------|
| Latency p99 | < 100ms | `SLOConfig.latency_p99_ms = 100.0` | ⚠️ Partially — histogram defined in alerts but NOT in code |
| Availability | 99.9% | `SLOConfig.availability_target = 0.999` | ❌ No availability metric exposed |
| FPR (False Positive Rate) | 5% | `SLOConfig.fpr_target = 0.05` | ❌ No per-stream FPR metric exposed |
| Anomaly Rate Lower Bound | 1% | `SLOConfig.anomaly_rate_min = 0.01` | ⚠️ Alert exists, metric undefined |
| Anomaly Rate Upper Bound | 15% | `SLOConfig.anomaly_rate_max = 0.15` | ⚠️ Alert exists, metric undefined |
| Checkpoint Success Rate | 99% | `SLOConfig.checkpoint_success_rate = 0.99` | ❌ No metric exposed |
| IEC Actions/Hour | ≤ 5 | `SLOConfig.iec_max_actions_per_hour = 5` | ⚠️ Alert exists, metric partial |
| IEC Cooldown | 300s | `SLOConfig.iec_cooldown_seconds = 300` | ✅ Implemented in circuit breaker logic |
| IEC Max Consecutive | 3 | `SLOConfig.iec_max_consecutive = 3` | ⚠️ Alert exists, metric partial |

### SLO Budget

**Latency SLO Budget (99.9% target, p99 < 100ms):**
- Monthly window: 43,200 minutes
- Allowed error budget: 43.2 minutes/month at 99.9%
- Burn rate alert threshold: ~6x burn rate for 1-hour window

**FPR SLO Budget (5% FPR target):**
- Monthly allowed false positives: Depends on total inference volume
- At 1M records/month: 50,000 false positives allowed
- Burn rate needs per-stream tracking

**Critical Gap:** No error budget tracking metric is exposed. Cannot compute remaining budget or burn rate without per-stream total count and error count metrics.

---

## Metrics & Instrumentation

### Custom Metrics

**Defined in `SLOConfig` but NOT exposed in operator code:**

| Metric Name | Defined In | Exposed in Operator? | Type |
|-------------|------------|----------------------|------|
| `memstream_latency_seconds` | `prometheus_alerts.yaml` | ❌ NO | Histogram |
| `anomaly_rate_per_context` | `prometheus_alerts.yaml` | ❌ NO | Gauge |
| `memstream_beta_per_context` | `prometheus_alerts.yaml` | ❌ NO | Gauge |
| `cadqstream_checkpoints_failed_total` | `prometheus_alerts.yaml` | ❌ NO | Counter |
| `cadqstream_iec_consecutive_actions` | `prometheus_alerts.yaml` | ❌ NO | Gauge |
| `cadqstream_iec_actions_total` | `prometheus_alerts.yaml` | ❌ NO | Counter |
| `cadqstream_model_integrity_failures_total` | `prometheus_alerts.yaml` | ❌ NO | Counter |

**What the operator actually tracks (internal only, NOT exposed):**

```python
# From MemStreamScoringOperator.__init__:
self._records_scored = 0          # internal counter, never exposed
self._anomalies_detected = 0      # internal counter, never exposed
self._last_metric_log = time.time()  # used for periodic log only
self._beta_cache = {}             # internal cache, never exposed
```

The operator has counters but never registers them with a Prometheus registry.

### Prometheus Integration

**Format Compliance:** The `prometheus_alerts.yaml` uses correct Prometheus syntax with `histogram_quantile`, `rate()`, `stddev()` functions. Alert expressions are well-formed.

**CRITICAL Issue:** The alerts reference metrics that don't exist. For example:
- `memstream_latency_seconds` — histogram_quantile requires the metric to be registered as a histogram
- `anomaly_rate_per_context` — requires per-context labels (neighborhood)
- `stddev(memstream_beta_per_context)` — requires beta to be a gauge with labels

**What is missing:**
1. No `prometheus_client` import in the operator
2. No metric registry initialization
3. No histogram for latency
4. No counters for records/anomalies/checkpoints
5. No gauges for beta values per context

### Cardinality Concerns

The alerts reference `anomaly_rate_per_context` and `memstream_beta_per_context` with a `neighborhood` label. This is acceptable since there are only 6 neighborhoods (Manhattan, Brooklyn, Queens, Bronx, airport, unknown). However, if beta tracking expands to include `hour_bucket` or `day_type` labels, the cardinality could grow to ~42 combinations (6 neighborhoods × 5 hour buckets × 2 day types). This is still manageable.

**Potential cardinality explosion:** The 4D context key (`ctx_4d`) has ~252 combinations (3 trip types × 5 hour buckets × 2 day types × 6 neighborhoods). Exposing `anomaly_rate` per `ctx_4d` as a label would be risky. Use `neighborhood` or `hour_bucket` only.

---

## OpenTelemetry Integration

The architecture overview states: "Distributed tracing → OpenTelemetry" as a fixed item from v1→v2→v3. However, **no OpenTelemetry code exists in the plan**. The operator:

1. Does NOT create spans for scoring operations
2. Does NOT propagate trace context
3. Does NOT add span attributes (neighborhood, context_key, score, latency)
4. Does NOT record span events for anomalies

Without OpenTelemetry, distributed tracing across the Flink pipeline is not possible. Anomaly detection failures cannot be correlated with upstream/downstream components.

---

## Alerting Rules

### Critical Alerts

| Alert | Expression | Issue |
|-------|------------|-------|
| `AnomalyRateAnomalouslyLow` | `anomaly_rate_per_context < 0.01` | ✅ Correct logic, but metric doesn't exist |
| `CheckpointFailures` | `rate(cadqstream_checkpoints_failed_total[5m]) > 0.01` | ✅ Correct logic, metric not exposed |
| `IECCircuitBreakerTripped` | `cadqstream_iec_consecutive_actions >= 3` | ✅ Correct logic, metric not exposed |
| `ModelIntegrityCheckFailed` | `cadqstream_model_integrity_failures_total > 0` | ✅ Correct logic, metric not exposed |

### Warning Alerts

| Alert | Expression | Issue |
|-------|------------|-------|
| `AnomalyRateAnomalouslyHigh` | `anomaly_rate_per_context > 0.15` | ⚠️ Good threshold, metric missing |
| `LatencySLOBreach` | `histogram_quantile(0.99, memstream_latency_seconds) > 0.1` | ⚠️ Good threshold, histogram not exposed |
| `IECActionRateHigh` | `rate(cadqstream_iec_actions_total[1h]) > 5` | ✅ Correct logic, counter missing |
| `BetaDivergenceAcrossContexts` | `stddev(memstream_beta_per_context) > 0.5` | ⚠️ Good check, metric missing |

### Missing Alerts

The following important alerts are NOT defined:

1. **IEC Cooldown Violation** — When IEC tries to act but cooldown hasn't elapsed. Logs warning but no Prometheus metric.
2. **HMAC Verification Failure** — When model or beta HMAC verification fails. Only logged, not metricked.
3. **Model Version Mismatch** — When deployed model version differs from expected.
4. **Memory Utilization** — When memory is filling up or stale.
5. **SLO Burn Rate** — No multi-window burn rate alerts (1h, 6h, 3d windows).
6. **Redis Connectivity** — No metric for Redis connection failures in the IEC beta polling path.
7. **Error Rate per Neighborhood** — No alert for elevated errors in specific neighborhoods.
8. **Feature Extraction Failures** — No metric for parsing failures in `parse_datetime` or `FeatureVectorizer`.

---

## Logging Strategy

### Structured Logs

The operator uses standard Python `logging`:

```python
LOGGER = logging.getLogger('cadqstream-memstream')
```

**Issues:**
1. Not JSON-structured. Cannot be easily parsed by log aggregation systems (ELK, Loki, Splunk).
2. Metric logging is only every 60 seconds (periodic batch). No per-record logging for anomalies.
3. No log level separation — INFO logs at 60s intervals may overwhelm or miss critical events.

**Current logging points (all non-structured):**
- Model pre-flight check result (INFO)
- Periodic metrics (every 60s): records_scored, anomalies, neighborhood (INFO)
- Beta update received (INFO)
- HMAC verification failure (ERROR)
- Redis error (WARNING)
- Circuit breaker tripped (ERROR)
- Scoring errors (ERROR)

### Correlation

**CRITICAL Gap:** No correlation IDs are generated or propagated.

For a distributed streaming system, the following should be present but are missing:
- `trace_id` — OpenTelemetry trace ID
- `span_id` — OpenTelemetry span ID
- `request_id` — Unique per-record ID for tracing through voting ensemble
- `model_version` — Already in output but not in log lines
- `neighborhood` — Present in context but not consistently in all log lines

The output record includes `model_version` but this is not echoed in log lines.

---

## Dashboard Recommendations

### Essential Panels

The plan does not define a dashboard. Recommended panels for a Grafana dashboard:

**Row 1: SLO Health**
- SLO: Latency p99 (gauge, green/yellow/red)
- SLO: Error rate (gauge)
- Error budget remaining (time series)
- Burn rate indicator (single stat)

**Row 2: Throughput & Scoring**
- Records scored per second (rate)
- Anomalies detected per second (rate)
- Anomaly rate per neighborhood (heatmap)
- Scoring method breakdown (pie chart: memstream vs if)

**Row 3: Latency**
- Latency p50/p90/p99 (heatmap or quantile chart)
- Latency histogram (bucket chart)
- Slow queries (>100ms) per minute (counter)

**Row 4: IEC Health**
- IEC consecutive actions gauge
- IEC actions per hour (rate)
- Beta value per neighborhood (time series, 6 lines)
- Beta divergence (stddev chart)

**Row 5: Model & Checkpoint**
- Model integrity failures (counter)
- Checkpoint success rate (gauge)
- Checkpoint duration (histogram)
- Last successful checkpoint age (gauge)

**Row 6: Redis Health (IEC Path)**
- Redis connection errors (counter)
- HMAC verification failures (counter)
- Beta update latency (histogram)

### SLA Tracking

**Missing:** No burn rate alert definitions. Standard multi-window burn rate alerts should be:

```yaml
# Fast burn: 1-hour window, 14.4% of budget consumed
- alert: LatencySLOBurnRateFast
  expr: |
    (
      1 - (
        histogram_quantile(0.99, rate(memstream_latency_seconds_bucket[1h]))
        <= 0.1
      )
    ) > 0.001
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Fast burn rate: 10% of error budget in 1 hour"

# Slow burn: 6-hour window
- alert: LatencySLOBurnRateSlow
  expr: |
    (
      1 - (
        histogram_quantile(0.99, rate(memstream_latency_seconds_bucket[6h]))
        <= 0.1
      )
    ) > 0.001
  for: 30m
  labels:
    severity: warning
```

---

## Issues Found

### CRITICAL Issues

1. **No Prometheus Metrics Exposed** — `MemStreamScoringOperator` never registers any metrics with Prometheus. The alerts in `prometheus_alerts.yaml` reference `memstream_latency_seconds`, `anomaly_rate_per_context`, `cadqstream_checkpoints_failed_total`, etc., but none of these are registered. **Fix:** Add `prometheus_client` instrumentation to the operator.

2. **No OpenTelemetry Tracing** — The architecture overview claims "Distributed tracing → OpenTelemetry" is fixed, but no tracing code exists in the operator. Critical paths (score_one, _get_beta_from_redis, memory update) have no spans. **Fix:** Add OpenTelemetry spans with appropriate attributes.

3. **No Error Budget Tracking** — SLOConfig defines targets but no metric computes remaining error budget or burn rate. Burn-rate alerts are not defined. **Fix:** Add SLO burn-rate metric exposure.

4. **Anomaly Rate Metric Not Exposed** — The `AnomalyRateAnomalouslyLow` alert (which detects possible model compromise) cannot fire because `anomaly_rate_per_context` is never registered as a metric. **Fix:** Expose anomaly rate as a Prometheus gauge with neighborhood label.

5. **HMAC Failure Not Metriced** — Security errors (model HMAC mismatch, beta HMAC mismatch) are logged but not metricked. `ModelIntegrityCheckFailed` alert references `cadqstream_model_integrity_failures_total` which doesn't exist. **Fix:** Add counter increment on SecurityError.

### HIGH Issues

6. **No Availability Metric** — `availability_target = 0.999` in SLOConfig but no metric tracks uptime. Availability should be computed as `(total_records - error_records) / total_records`. **Fix:** Expose error count and total count counters.

7. **No FPR Tracking Metric** — `fpr_target = 0.05` is defined but not measured. FPR requires ground truth labels, which may not be available in production. If labels are available (from IEC feedback), this should be tracked. **Fix:** If IEC provides labels, expose FPR as a metric. Otherwise, document this as a limitation.

8. **Redis Beta Polling Has No Health Metric** — The `_get_beta_from_redis()` method has try/except with only WARNING logs. Redis connection failures are not exposed as Prometheus metrics. **Fix:** Add Redis error counter.

9. **No Checkpoint Metrics** — `CheckpointFailures` alert references `cadqstream_checkpoints_failed_total` but PyFlink's checkpoint state is not instrumented. **Fix:** Hook into Flink's checkpoint listener interface or expose metrics from the checkpointed state operations.

10. **Logging Not JSON-Structured** — Logs are plain text, making log aggregation and querying difficult. **Fix:** Use `python-json-logger` or `structlog` for JSON-structured output.

### MEDIUM Issues

11. **No Correlation IDs** — No `trace_id` or `request_id` in logs. Anomaly investigation requires correlating logs across voting ensemble, MemStream operator, and IEC. **Fix:** Generate UUID per record, propagate through the pipeline.

12. **No Beta Value Tracking** — `BetaDivergenceAcrossContexts` alert cannot fire because beta values per neighborhood are not exposed as metrics. **Fix:** Expose `memstream_beta_per_context` gauge with neighborhood label on each scoring call.

13. **No Model Version in Logs** — `model_version` is in the output record but not in log lines, making it harder to correlate version with behavior. **Fix:** Include `model_version` in all log lines.

14. **Metric Logging Only Every 60s** — Anomaly detection issues may go unnoticed for up to 60 seconds. For critical scoring failures, immediate logging or metrics are needed. **Fix:** Log or metric on first anomaly detection or error, not just periodic summary.

15. **No Latency Distribution Labels** — Even if latency histogram is added, there are no labels for `neighborhood`, `scoring_method`, or `context_key`. These labels would enable more granular alerting. **Fix:** Add meaningful labels to the histogram.

### LOW Issues

16. **IEC Cooldown Metric Missing** — `iec_cooldown_seconds = 300` is defined but no metric tracks cooldown state. While the circuit breaker logic exists, operators cannot see cooldown state in dashboards. **Fix:** Expose cooldown state as a gauge.

17. **No Memory Freshness Metric** — Memory slot utilization is tracked internally (`get_memory_utilization()`) but not exposed. A stale memory indicator would help diagnose drift detection failures. **Fix:** Expose memory utilization per neighborhood as a gauge.

18. **Alert Annotations Lack Runbook Links** — All alerts have `summary` annotations but no `runbook_url` or `description` with remediation steps. **Fix:** Add runbook links to alert annotations.

19. **No Canary Traffic Metrics** — The 3-phase dark launch plan mentions monitoring `shadow_disagreement_rate` and `canary_rate` but no metrics are defined for these phases. **Fix:** Add metrics for canary/shadow mode indicators.

---

## Recommendations

### 1. Add Prometheus Instrumentation (CRITICAL)

Add to `MemStreamScoringOperator`:

```python
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY

# Use a custom registry to avoid conflicts
_registry = CollectorRegistry()
SCORED_TOTAL = Counter(
    'memstream_records_scored_total',
    'Total records scored',
    ['neighborhood'],
    registry=_registry
)
ANOMALIES_DETECTED = Counter(
    'memstream_anomalies_detected_total',
    'Total anomalies detected',
    ['neighborhood'],
    registry=_registry
)
LATENCY_HISTOGRAM = Histogram(
    'memstream_latency_seconds',
    'Scoring latency in seconds',
    ['neighborhood'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0],
    registry=_registry
)
BETA_GAUGE = Gauge(
    'memstream_beta_per_context',
    'Current beta threshold per neighborhood',
    ['neighborhood'],
    registry=_registry
)
MODEL_INTEGRITY_FAILURES = Counter(
    'cadqstream_model_integrity_failures_total',
    'Model integrity check failures',
    registry=_registry
)
HMAC_VERIFICATION_FAILURES = Counter(
    'cadqstream_hmac_verification_failures_total',
    'HMAC verification failures',
    ['type'],  # type: model, beta
    registry=_registry
)
ERRORS_TOTAL = Counter(
    'memstream_scoring_errors_total',
    'Scoring errors',
    ['neighborhood', 'error_type'],
    registry=_registry
)
```

**In `process_element`:**
```python
def process_element(self, record, context):
    start_time = time.time()
    try:
        # ... existing scoring logic ...
        
        # After successful scoring:
        SCORED_TOTAL.labels(neighborhood=neighborhood).inc()
        if is_anom:
            ANOMALIES_DETECTED.labels(neighborhood=neighborhood).inc()
        BETA_GAUGE.labels(neighborhood=neighborhood).set(ms.max_thres.item())
        
        # Latency (end of try block)
    except SecurityError as e:
        MODEL_INTEGRITY_FAILURES.inc()
        HMAC_VERIFICATION_FAILURES.labels(type='model').inc()
        ERRORS_TOTAL.labels(neighborhood=neighborhood, error_type='security').inc()
    except Exception as e:
        ERRORS_TOTAL.labels(neighborhood=neighborhood, error_type='scoring').inc()
    finally:
        latency = time.time() - start_time
        LATENCY_HISTOGRAM.labels(neighborhood=neighborhood).observe(latency)
```

**In `_get_beta_from_redis`:**
```python
# On HMAC failure:
HMAC_VERIFICATION_FAILURES.labels(type='beta').inc()
```

### 2. Add OpenTelemetry Tracing (CRITICAL)

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode

# Initialize once at module level
resource = Resource.create({
    "service.name": "cadqstream-memstream",
    "service.version": MODEL_VERSION,
})
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

# In process_element:
def process_element(self, record, context):
    with tracer.start_as_current_span("memstream_score") as span:
        span.set_attribute("neighborhood", neighborhood)
        span.set_attribute("context_key", ctx_4d)
        span.set_attribute("model_version", self.model_version)
        
        try:
            score, is_anom = ms.score_one(features)
            span.set_attribute("anomaly_score", float(score))
            span.set_attribute("is_anomaly", bool(is_anom))
            span.set_attribute("threshold", float(ms.max_thres.item()))
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
```

### 3. Add JSON Structured Logging (HIGH)

```python
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
```

Replace all `LOGGER.info/warning/error` with structlog:

```python
log = structlog.get_logger('cadqstream-memstream')
log.info("scoring_complete",
    records_scored=self._records_scored,
    anomalies_detected=self._anomalies_detected,
    neighborhood=neighborhood,
    anomaly_rate=rate,
    model_version=self.model_version,
)
```

### 4. Add SLO Burn Rate Alerts (HIGH)

```yaml
groups:
  - name: cadqstream-slo-burn-rate
    interval: 30s
    rules:
      - alert: LatencySLOBurnRateFast
        expr: |
          (
            sum(rate(memstream_latency_seconds_bucket{le="0.1"}[1h]))
            /
            sum(rate(memstream_latency_seconds_count[1h]))
          ) < 0.999
        for: 1m
        labels:
          severity: critical
          slo: latency_p99
        annotations:
          summary: "Fast burn rate: 10% of latency error budget in 1 hour"
          description: "Immediate action required. Less than 6 hours of error budget remain."

      - alert: LatencySLOBurnRateSlow
        expr: |
          (
            sum(rate(memstream_latency_seconds_bucket{le="0.1"}[6h]))
            /
            sum(rate(memstream_latency_seconds_count[6h]))
          ) < 0.999
        for: 30m
        labels:
          severity: warning
          slo: latency_p99
        annotations:
          summary: "Slow burn rate: 10% of latency error budget in 6 hours"

      - alert: ErrorRateSLOBurnRate
        expr: |
          (
            sum(rate(memstream_scoring_errors_total[1h]))
            /
            sum(rate(memstream_records_scored_total[1h]))
          ) > 0.001
        for: 5m
        labels:
          severity: critical
          slo: availability
        annotations:
          summary: "Error rate > 0.1% — availability SLO at risk"
```

### 5. Add Redis Health Metrics (HIGH)

```python
REDIS_CONNECTION_FAILURES = Counter(
    'cadqstream_redis_connection_failures_total',
    'Redis connection failures',
    ['operation'],  # get_beta, set_beta
    registry=_registry
)

REDIS_LATENCY = Histogram(
    'cadqstream_redis_latency_seconds',
    'Redis operation latency',
    ['operation'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
    registry=_registry
)

def _get_beta_from_redis(self, key: str) -> Optional[float]:
    op_start = time.time()
    try:
        # ... existing logic ...
    except redis.ConnectionError as e:
        REDIS_CONNECTION_FAILURES.labels(operation='get_beta').inc()
        raise
    finally:
        REDIS_LATENCY.labels(operation='get_beta').observe(time.time() - op_start)
```

### 6. Add Runbook Links to Alerts (MEDIUM)

```yaml
- alert: AnomalyRateAnomalouslyLow
  annotations:
    summary: "Anomaly rate < 1% — possible model compromise"
    runbook_url: "https://wiki.internal/runbooks/memstream/low-anomaly-rate"
    description: |
      Possible causes:
      1. Model HMAC compromised — check cadqstream_model_integrity_failures_total
      2. Beta threshold too high — check memstream_beta_per_context
      3. Normal data drift — check if IEC triggered memory reset
      Remediation: Check model integrity, verify beta calibration, review recent IEC actions.
```

---

## Verification Checklist

- [ ] **SLOs Measurable:** ❌ Latency histogram not exposed. Availability not computed. FPR not tracked.
- [ ] **Alerts Actionable:** ⚠️ Alert expressions are correct but metrics missing. Runbook links absent.
- [ ] **Logs Queryable:** ❌ Not JSON-structured. No correlation IDs. Cannot filter by neighborhood in structured queries.
- [ ] **Dashboards Accurate:** ❌ No dashboard definition. No panels defined for key metrics.
- [ ] **Tracing Integrated:** ❌ No OpenTelemetry spans. No trace context propagation.
- [ ] **Security Events Metriced:** ❌ HMAC failures logged but not metricked. No alerting on security events.
- [ ] **Redis Path Observed:** ❌ Redis polling has no latency or error metrics.
- [ ] **IEC Circuit Breaker Visible:** ⚠️ Logic exists but state not exposed as Prometheus metric.
- [ ] **Dark Launch Monitored:** ❌ No shadow_disagreement_rate or canary_rate metrics defined.
- [ ] **Error Budget Tracked:** ❌ No burn-rate metric. No budget remaining gauge.

---

## Priority Order for Fixes

| Priority | Fix | Impact |
|----------|-----|--------|
| P0 | Add Prometheus histogram for latency | Enables LatencySLOBreach alert |
| P0 | Add anomaly rate metric | Enables AnomalyRateAnomalouslyLow alert |
| P0 | Add model integrity failure counter | Enables ModelIntegrityCheckFailed alert |
| P1 | Add OpenTelemetry tracing | Enables distributed tracing |
| P1 | Add JSON structured logging with correlation IDs | Enables log correlation |
| P1 | Add Redis health metrics | Monitors IEC beta path |
| P1 | Add SLO burn-rate alerts | Prevents SLO breaches |
| P2 | Add beta value per neighborhood gauge | Enables BetaDivergenceAcrossContexts |
| P2 | Add availability/error rate metrics | Enables availability SLO tracking |
| P2 | Add checkpoint metrics | Monitors checkpoint health |
| P3 | Define Grafana dashboard JSON | Visual SLO health |
| P3 | Add runbook links to alerts | Faster incident response |

---

*Review generated: 2026-05-12*  
*Reviewer: SRE/Observability Engineer*  
*Next review: After P0/P1 fixes implemented*
