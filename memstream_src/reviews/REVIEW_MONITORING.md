# Monitoring/SRE Engineer Review — PLAN_v3.md

**Reviewer:** Principal SRE/Monitoring Engineer (12 yrs Prometheus, Grafana, OpenTelemetry, Jaeger)
**Date:** 2026-05-12
**Files Reviewed:** `src/monitoring/metrics.py`, `cadqstream_metrics.py`, `deployment/prometheus/prometheus.yml`, `deployment/prometheus/alert-rules/cadqstream-alerts.yml`, `deployment/flink/flink-conf.yml`, docker-compose, Grafana JSON dashboards

---

## 1. Prometheus Metrics (RED Method)

### 1.1 Rate — Missing Cardinality

`cadqstream_records_input_total` is a bare counter with NO labels. Cannot distinguish L1 vs L2 vs L3 throughput. Cannot detect per-neighborhood rate anomalies.

**Fix:** `cadqstream_records_total{phase="schema"|"canary"|"ml"|"rendezvous"|"meta", neighborhood="..."}`

### 1.2 Errors — Zero Error Rate Instrumentation — CRITICAL

**NO metric** for scoring errors. `MemStreamScoringOperator` catches all exceptions and yields `anomaly_score=-1.0`. This error is **indistinguishable from a legitimate low score** in dashboards.

No metric for deserialization errors, pre-flight check failures, or HMAC verification failures.

`cadqstream_model_integrity_failures_total` is referenced in alerts but **never registered or incremented anywhere**.

### 1.3 Duration — Completely Absent — CRITICAL

`memstream_latency_seconds` is referenced in `LatencySLOBreach` alert but:
- Not registered in `_KNOWN_COUNTERS` or `_KNOWN_GAUGES`
- Not emitted by any Flink operator
- `histogram_quantile` will always return NaN

The entire latency SLO monitoring pipeline is a dead letter.

### 1.4 MemStream Operator — Zero Prometheus Metrics

The `MemStreamScoringOperator` has zero Prometheus metrics. Only Python integers logged every 60 seconds. No:
- Per-neighborhood score distribution
- Memory update rate
- Beta threshold value per context
- AE fine-tuning trigger rate

### 1.5 IEC Operator — Extremely Sparse

`cadqstream_iec_decisions_total` and `cadqstream_iec_drift_detected_total` referenced in dashboards are **never emitted**. Only print-logged every 10 windows.

---

## 2. Grafana Dashboards

### 2.1 Missing MemStream-Specific Dashboard

No dashboard exists for MemStream operations. No panel for:
- Per-neighborhood memory utilization
- Beta threshold values across contexts
- AE fine-tuning trigger rate
- Memory rebuild progress after reset

### 2.2 Metric Name Inconsistencies

Overview dashboard uses `kafka_consumergroup_lag{topic="taxi-nyc-raw"}` but the exporter exposes `kafka_consumer_group_lag` (underscores). Panels will show "No data."

### 2.3 Grafana Variables Absent

No templating for environment, job name, or neighborhood. All dashboards are single-value hardcoded views.

### 2.4 SLO Error Budget Dashboard — Absent

No panel shows error budget burn rate. No "Time to SLO Exhaustion." Standard for production-grade observability.

---

## 3. OpenTelemetry Tracing — Completely Absent

Despite PLAN_v3.md stating "Distributed tracing → OpenTelemetry":
- No Jaeger deployment
- No OTLP exporter
- No trace context propagation across Kafka messages
- No span attributes on scoring operations
- No trace IDs in log output

---

## 4. SLO/SLA Definition

### 4.1 Recording Rules Don't Exist — CRITICAL

SLOs are defined as Python dataclass values but Prometheus **recording rules** do not exist. The SLOConfig values are inert.

### 4.2 Latency SLO Measures Wrong Thing

`latency_p99_ms: 100.0` measures per-record scoring latency. The meaningful SLO is **end-to-end pipeline latency**: Kafka produce → anomaly decision. Not measured.

### 4.3 Availability SLO Undefined

`availability_target: 0.999` — defined but not measured. "Available" is ambiguous.

### 4.4 Error Budget Policy Not Defined

No multi-window burn rate alerts. No SLO error budget tracking. No "SLO breach" definition.

### 4.5 IEC Action SLO Unmeasured

`iec_max_actions_per_hour: 5` limit exists in config but no Prometheus gauge for current actions. Alert references non-existent `rate(cadqstream_iec_actions_total[1h])`.

---

## 5. Alert Fatigue

### 5.1 Stale Alert Definitions — CRITICAL

Multiple alerts reference metrics that **don't exist**:
- `cadqstream_checkpoints_failed_total` — never registered
- `cadqstream_iec_consecutive_actions` — never registered
- `cadqstream_model_integrity_failures_total` — never registered
- `cadqstream_records_input_total` — registered but NEVER incremented
- `cadqstream_iec_actions_total` — registered but never incremented
- `anomaly_rate_per_context` — never registered
- `memstream_beta_per_context` — never registered
- `memstream_latency_seconds` — never registered

These alerts will return empty result sets or fire at startup before metrics register.

### 5.2 Too Many Infrastructure Alerts

35 alerts (17 infrastructure + 18 pipeline). Should be consolidated:
- `PostgreSQLDown`, `PostgreSQLHighConnectionUsage`, `PostgreSQLLongRunningQueries` → `PostgreSQLHealthWarning{severity="info"}`

### 5.3 No Runbooks

Zero runbooks exist. PLAN_v3.md references `docs/RUNBOOK.md` but it doesn't exist.

---

## 6. Incident Response

### 6.1 Severity Matrix Absent

No documented severity levels, response times, or resolution time targets.

### 6.2 No On-Call Channel

All alerts route to "no-op" receiver (`http://127.0.0.1:9999/noop`). No Slack, no PagerDuty.

### 6.3 No Post-Incident Review

No blameless post-mortem template, no incident database.

---

## 7. Chaos Engineering — Absent

Zero chaos engineering program. No failure mode testing for:
- Kafka broker failure
- Flink TaskManager crash
- PostgreSQL unavailable
- MemStream memory corruption

---

## 8. CRITICAL Issues

### CR-1: Zero Latency Metrics

**What is wrong:** `memstream_latency_seconds` referenced in alert but never registered/emitted.

**Fix:**
```python
self._scoring_latency = Histogram(
    'memstream_scoring_latency_seconds',
    'MemStream score_one() latency',
    ['neighborhood', 'scoring_method'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
# Wrap in process_element:
with self._scoring_latency.labels(neighborhood=neighborhood, scoring_method='memstream').time():
    score, is_anom = ms.score_one(features)
```

### CR-2: All Error Paths Invisible

**What is wrong:** Exceptions produce `anomaly_score=-1.0` with no Prometheus counter.

**Fix:**
```python
self._scoring_errors = Counter(
    'memstream_scoring_errors_total',
    'Scoring errors by type',
    ['error_type', 'neighborhood']
)
# In except block:
self._scoring_errors.labels(error_type=type(e).__name__, neighborhood=neighborhood).inc()
```

### CR-3: Checkpoints on Ephemeral /tmp — CRITICAL DR Gap

**What is wrong:** `state.checkpoints.dir: file:///tmp/flink-checkpoints`. Ephemeral storage. On container restart, ALL checkpoints are LOST. MemStream per-context memory must be rebuilt from scratch.

**Fix:**
```yaml
# In flink-conf.yml:
state.backend: rocksdb
state.checkpoints.dir: s3://cadqstream-checkpoints/flink/checkpoints
state.savepoints.dir: s3://cadqstream-savepoints/flink/savepoints
# Requires: flink-s3-fs-hadoop JAR in /opt/flink/lib/
# AND: MinIO credentials in env
```

### CR-4: Broken Alerts Must Be Fixed First

Disable all alerts that reference non-existent metrics until metrics are implemented. Otherwise alerts are noise.

**Fix — Disable broken alerts:**
```yaml
# In cadqstream-alerts.yml, for each broken alert, add:
- alert: LatencySLOBreach
  enabled: false  # memstream_latency_seconds not yet instrumented
  annotations:
    note: "Enable after implementing memstream_scoring_latency_seconds histogram"
```

### CR-5: No PostgreSQL Backup

MinIO data has no backup automation.

**Fix:**
```bash
# Add backup strategy: MinIO bucket lifecycle policies via mc ilm
# Configure versioning and expiry in mc alias config
docker exec ldt-minio mc ilm import local/cadqstream-raw \
  --data '{"Rules": [{"Status": "Enabled", "Expiration": {"Days": 7}}]}'
```

---

## 9. HIGH Issues

| # | Issue | Fix |
|---|-------|-----|
| 1 | MemStream operator has zero metrics | Add `_records_scored`, `_memory_utilization`, `_beta_value`, `_scoring_latency`, `_scoring_errors` |
| 2 | IEC operator has zero metrics | Add `_drift_detected`, `_strategy_decisions`, `_strategy_confidence`, `_circuit_breaker_state` |
| 3 | OTel completely absent | Add OTLP exporter, Jaeger, per-operator spans |
| 4 | Shadow disagreement not monitored | Add `cadqstream_shadow_disagreement_total` + Grafana panel |
| 5 | Kafka metric names mismatch | Standardize to `kafka_consumer_group_lag` (underscores) |
| 6 | Grafana contact point = no-op | Replace with Slack/PagerDuty webhook |

---

## 10. MEDIUM/LOW Issues

| # | Issue | Fix |
|---|-------|-----|
| 1 | TSDB retention only 15 days | Extend to 90d + add size limit |
| 2 | Health endpoint doesn't check Flink/Kafka | Add `/health` that pings JM + lag |
| 3 | SLO Error Budget dashboard missing | Create dedicated SLO dashboard |
| 4 | Kafka RF=1, no HA | Kafka with RF=3 + 3 brokers |
| 5 | Prometheus no WAL compression | Add `--storage.tsdb.wal-compression` |

---

## 11. Priority Implementation Order

| Priority | Action | Owner | Hours |
|----------|--------|-------|-------|
| 1 | Instrument `memstream_latency_seconds` histogram | SRE | 2 |
| 2 | Instrument all error counters | SRE | 1 |
| 3 | Move checkpoints to S3 | SRE | 1 |
| 4 | Disable broken alerts, implement missing metrics | SRE | 4 |
| 5 | Full MemStream + IEC Prometheus metrics | SRE | 5 |
| 6 | OTel tracing (Jaeger + spans) | SRE | 8 |
| 7 | SLO recording rules + error budget dashboard | SRE | 6 |
| 8 | Grafana contact points (Slack/PagerDuty) | SRE | 2 |
| 9 | PostgreSQL backup | SRE | 2 |
| 10 | Runbooks for top 5 failures | SRE | 8 |

---

*Reviewed by: Principal SRE/Monitoring Engineer | 2026-05-12*
