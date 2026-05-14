# CA-DQStream + MemStream Hybrid v4 — Expert Review Synthesis

> **Date:** 2026-05-12
> **Reviews Conducted By:** 6 Expert Subagents (Flink, Data Eng, Docker/SRE, Security, ML, Monitoring)
> **Plan Version:** v4

---

## Executive Summary

### Overall Verdict: **CONDITIONALLY APPROVED**

The v4 plan represents a substantial architectural improvement over v1-v3, successfully addressing the foundational Flink API misuse (broadcast state trap), implementing Redis-based IEC communication with HMAC verification, and establishing a solid 3-phase dark launch strategy. The core MemStream algorithm implementation is correct and follows the WWW 2022 paper specifications. The team should proceed to Phase 1 fixes.

However, **18 CRITICAL issues** and **24 HIGH issues** remain across all domains. The most serious concerns are:
1. **Security:** HMAC bypass vulnerability when `IEC_SIGNING_KEY` is not set — allows unrestricted beta manipulation
2. **Deployment:** No Dockerfile or docker-compose.yml exist despite the "Docker-only" pivot
3. **Observability:** No Prometheus metrics exposed in the operator — alerts cannot fire
4. **Data Integrity:** Temporal data shuffling destroys streaming evaluation validity

The remaining issues are localized and fixable. Estimated total fix effort for all CRITICAL + HIGH issues: **~10-15 engineering days**.

### Critical Issues Count by Domain

| Domain | CRITICAL | HIGH | MEDIUM | LOW | Total |
|--------|----------|------|--------|-----|-------|
| Flink | 3 | 4 | 5 | 4 | 16 |
| Data Eng | 2 | 4 | 5 | 3 | 14 |
| Docker/SRE | 4 | 4 | 4 | 3 | 15 |
| Security | 3 | 4 | 4 | 3 | 14 |
| ML | 1 | 3 | 3 | 3 | 10 |
| Monitoring | 5 | 5 | 5 | 4 | 19 |
| **TOTAL** | **18** | **24** | **26** | **20** | **88** |

---

## Consolidated Issue Tracker

### CRITICAL Issues (Must Fix Before Production)

#### C-FL-1: `IECFeedbackOperator` circuit breaker state not in `BroadcastState`
- **Domain:** Flink
- **Source:** REVIEW_FLINK_v4.md
- **Description:** `_circuit_breaker` is a plain Python dict in `__init__`. For `KeyedBroadcastProcessFunction`, only `BroadcastState` is checkpointable. Plain Python attributes on `self` are NOT checkpointed.
- **Risk:** After Flink restart, circuit breaker resets to initial state. Attacker or misconfigured IEC can issue unlimited beta adjustments.
- **Fix:** Move circuit breaker state into `BroadcastState` using `MapStateDescriptor`.

#### C-FL-2: Redis polling on every record introduces unbounded latency
- **Domain:** Flink
- **Source:** REVIEW_FLINK_v4.md
- **Description:** `_get_beta_from_redis()` called on every single record. Redis round-trip latency (1-5ms) × 100 records/micro-batch = 500ms overhead. P99 latency SLO (100ms) will be breached.
- **Risk:** P99 latency SLO breach under production load. Cascades into checkpoint alignment delays.
- **Fix:** Use time-based poll interval (10 seconds) instead of per-record polling.

#### C-FL-3: Missing `import hashlib` in `memstream_scoring_op.py`
- **Domain:** Flink
- **Source:** REVIEW_FLINK_v4.md
- **Description:** `hashlib.sha256` and `hmac` used but not imported in the operator file.
- **Risk:** `NameError: name 'hashlib' is not defined` at runtime when beta update arrives from IEC.
- **Fix:** Add `import hashlib` and `import hmac` to imports section.

#### C-DE-1: Temporal shuffle destroys evaluation validity
- **Domain:** Data Engineering
- **Source:** REVIEW_DATAENG_v4.md
- **Description:** `df.sample(frac=1, random_state=42)` shuffles data, destroying temporal structure required for streaming anomaly detection evaluation.
- **Risk:** Concept drift evaluation is invalid. Train/calibration/test splits should use time-ordered data.
- **Fix:** Use `df.sort_values('tpep_pickup_datetime')` instead of shuffle. Time-ordered splits: Month 1-6 warmup, Month 7-9 calibration, Month 10-12 test.

#### C-DE-2: Normalization leakage in autoencoder training
- **Domain:** Data Engineering
- **Source:** REVIEW_DATAENG_v4.md
- **Description:** Normalization stats computed from data used for memory initialization AND AE training. AE is trained on data whose statistics inform its own normalization.
- **Risk:** Overfitted normalization. Model sees "clean" normalized data during warmup but different characteristics during streaming.
- **Fix:** Split warmup data: first 10% for stats, middle 80% for training, last 10% for memory init.

#### C-DK-1: No Dockerfile in plan
- **Domain:** Docker/SRE
- **Source:** REVIEW_DOCKER_v4.md
- **Description:** Plan references `memstream_src/operators/deployment/dockerfile` but provides zero Dockerfile code.
- **Risk:** Cannot build container. Python 3.8 EOL (October 2024) not addressed.
- **Fix:** Create Dockerfile using `python:3.12-slim-bookworm`, multi-stage build, non-root user, PyTorch CPU-only.

#### C-DK-2: No docker-compose.yml defined
- **Domain:** Docker/SRE
- **Source:** REVIEW_DOCKER_v4.md
- **Description:** Despite "Docker-only, no K8s" pivot, no docker-compose.yml provided.
- **Risk:** Cannot orchestrate containers. Deployment impossible.
- **Fix:** Create docker-compose.yml with MemStream operator, Redis, Prometheus exporter services.

#### C-DK-3: `MEMSTREAM_MODEL_SIGNING_KEY` not in docker-compose
- **Domain:** Docker/SRE
- **Source:** REVIEW_DOCKER_v4.md
- **Description:** HMAC signing key not included in compose file. Security regression from v3 fix.
- **Risk:** HMAC verification bypassed. Beta updates can be manipulated.
- **Fix:** Add secrets section to docker-compose.yml or document required environment variables.

#### C-DK-4: No Redis configuration for IEC communication
- **Domain:** Docker/SRE
- **Source:** REVIEW_DOCKER_v4.md
- **Description:** Redis required for v4 IEC→MemStream communication but no configuration provided.
- **Risk:** IEC→MemStream communication fails. Beta updates cannot be delivered.
- **Fix:** Add Redis service to docker-compose.yml with healthcheck, password auth, resource limits.

#### C-SEC-1: HMAC bypass when `IEC_SIGNING_KEY=None` in either operator
- **Domain:** Security
- **Source:** REVIEW_SECURITY_v4.md
- **Description:** If `IEC_SIGNING_KEY` is not set in either operator, unsigned beta values are written to Redis and accepted without verification.
- **Risk:** CRITICAL — Any process with Redis access can manipulate beta thresholds. Suppress all anomaly detection (`beta=999999`) or flood with false positives (`beta=0.0001`).
- **Fix:** Add startup validation that enforces `IEC_SIGNING_KEY` presence. Fail-fast instead of silent bypass.

#### C-SEC-2: Redis unauthenticated by default, no TLS
- **Domain:** Security
- **Source:** REVIEW_SECURITY_v4.md
- **Description:** `REDIS_PASSWORD=None` silently disables authentication. No TLS/SSL configuration.
- **Risk:** Man-in-the-middle attacks can read/modify beta values. Any process on network segment can access Redis.
- **Fix:** Require password in production. Add TLS configuration with certificate validation.

#### C-SEC-3: Duplicate HMAC verification code in `memstream_core.py`
- **Domain:** Security
- **Source:** REVIEW_SECURITY_v4.md, REVIEW_ML_v4.md
- **Description:** Lines 564-576 and 577-589 contain identical HMAC verification blocks. File opened twice per load.
- **Risk:** Maintenance hazard. Wastes I/O. Potential for divergent behavior on code updates.
- **Fix:** Remove duplicate block (lines 577-589).

#### C-ML-1: `max_thres` used before initialization
- **Domain:** ML
- **Source:** REVIEW_ML_v4.md
- **Description:** `max_thres` never initialized in `__init__`. Only set by `warmup()` or `set_beta()`. If `score_one()` called before these, crashes with `AttributeError`.
- **Risk:** Runtime crash in production if model loaded without calibration.
- **Fix:** Initialize `self.max_thres = torch.tensor(0.0, dtype=torch.float32, device=self.device)` in `__init__`.

#### C-MON-1: No Prometheus metrics exposed in operator
- **Domain:** Monitoring
- **Source:** REVIEW_MONITORING_v4.md
- **Description:** `prometheus_alerts.yaml` references `memstream_latency_seconds`, `anomaly_rate_per_context`, `cadqstream_checkpoints_failed_total`, etc. None are registered.
- **Risk:** All alerts cannot fire. No visibility into system health.
- **Fix:** Add `prometheus_client` instrumentation with Counter, Histogram, Gauge for all defined metrics.

#### C-MON-2: No OpenTelemetry tracing
- **Domain:** Monitoring
- **Source:** REVIEW_MONITORING_v4.md
- **Description:** Architecture overview claims "Distributed tracing → OpenTelemetry" but no tracing code exists.
- **Risk:** Cannot correlate anomalies across distributed pipeline. Debugging requires manual log correlation.
- **Fix:** Add OpenTelemetry spans with attributes (neighborhood, context_key, score, latency).

#### C-MON-3: No error budget tracking
- **Domain:** Monitoring
- **Source:** REVIEW_MONITORING_v4.md
- **Description:** `SLOConfig` defines targets but no metric computes remaining error budget or burn rate.
- **Risk:** No advance warning of SLO breach. Only know after breach occurs.
- **Fix:** Add SLO burn-rate metric exposure. Implement multi-window burn rate alerts.

#### C-MON-4: Anomaly rate metric not exposed
- **Domain:** Monitoring
- **Source:** REVIEW_MONITORING_v4.md
- **Description:** `AnomalyRateAnomalouslyLow` alert (detects possible model compromise) cannot fire because `anomaly_rate_per_context` never registered.
- **Risk:** Model compromise or threshold manipulation goes undetected.
- **Fix:** Expose anomaly rate as Prometheus gauge with neighborhood label.

#### C-MON-5: HMAC failure not metriced
- **Domain:** Monitoring
- **Source:** REVIEW_MONITORING_v4.md
- **Description:** Security errors (model/beta HMAC mismatch) logged but not metricked. `ModelIntegrityCheckFailed` alert references non-existent metric.
- **Risk:** Security incidents not visible in dashboards. No alerting on integrity failures.
- **Fix:** Add counter increment on `SecurityError` for both model and beta HMAC failures.

---

### HIGH Issues (Should Fix Before Production)

#### H-FL-1: `torch.load` with `weights_only=True` may fail on custom state dicts
- **Domain:** Flink
- **Description:** Internal checkpoint state uses `weights_only=True`. While currently safe, fragile if developers change `.item()` to tensor access.
- **Fix:** Use `weights_only=False` for internal checkpoints, `weights_only=True` only for external model files.

#### H-FL-2: No watermark strategy defined
- **Domain:** Flink
- **Description:** Event-time processing assumes timestamps but no watermark strategy defined.
- **Fix:** Add `WatermarkStrategy.forBoundedOutOfOrderness(Duration.ofSeconds(30))` with timestamp assigner.

#### H-FL-3: `MemStreamScoringOperator` missing `CheckpointedFunction` interface
- **Domain:** Flink
- **Description:** Model loaded in `open()` but no version compatibility validation. If base model changes, old checkpoints restore incorrectly.
- **Fix:** Add version compatibility check in `open()` validating `in_dim` and `out_dim`.

#### H-FL-4: No parallelism configuration
- **Domain:** Flink
- **Description:** `env.setParallelism()` not specified. Should match Kafka partition count.
- **Fix:** Document recommended parallelism settings (4 for 4 Kafka partitions).

#### H-DE-1: Kafka topic structure undefined
- **Domain:** Data Engineering
- **Description:** Partition count, replication factor, retention policy not specified.
- **Fix:** Define 12 partitions, replication factor 3, 7-day retention for raw topic.

#### H-DE-2: Event ordering semantics unclear
- **Domain:** Data Engineering
- **Description:** `KeyedProcessFunction` guarantees within-key ordering but cross-key ordering not documented.
- **Fix:** Document partitioning strategy and guarantees. Implement late data handling with watermarks.

#### H-DE-3: Late data handling missing
- **Domain:** Data Engineering
- **Description:** NYC TLC data can have late submissions up to 24 hours.
- **Fix:** Implement watermark strategy with 6-hour allowed lateness. Use side output for dead letter queue.

#### H-DE-4: Anomaly injection unrealistic
- **Domain:** Data Engineering
- **Description:** `inject_anomalies()` uses deterministic formula — all anomalies have `fare_amount = trip_distance * 2.5 * factor`.
- **Fix:** Multi-strategy injection: speed_extreme, swap_location, duration_anomaly, passenger_count.

#### H-DK-1: No CPU/memory limits
- **Domain:** Docker/SRE
- **Description:** No Docker resource limits defined for MemStream container.
- **Fix:** Add 2 CPU, 4GB memory limits. 0.5 CPU, 1GB reservations.

#### H-DK-2: No traffic splitting implementation
- **Domain:** Docker/SRE
- **Description:** Dark launch described conceptually but zero implementation code.
- **Fix:** Implement shadow/canary/production traffic splitting with disagreement tracking.

#### H-DK-3: No `/health` endpoint code
- **Domain:** Docker/SRE
- **Description:** Prometheus alerts defined but no endpoint for HEALTHCHECK.
- **Fix:** Add Flask health server with `/health`, `/ready`, `/metrics` endpoints.

#### H-DK-4: No requirements.txt
- **Domain:** Docker/SRE
- **Description:** Build reproducibility impossible without explicit dependencies.
- **Fix:** Create requirements.txt with torch, numpy, pandas, redis, pyflink versions.

#### H-SEC-1: Redis credentials traverse network unencrypted
- **Domain:** Security
- **Description:** No TLS configuration for Redis connections.
- **Fix:** Add `ssl=True` and `ssl_cert_reqs='required'` for production deployments.

#### H-SEC-2: HMAC key is environment variable
- **Domain:** Security
- **Description:** Env vars exposed via `/proc/*/environ` and process listing.
- **Fix:** Document migration to secrets manager (Vault, AWS Secrets Manager). Add minimum key length validation.

#### H-SEC-3: No Redis connection timeout
- **Domain:** Security
- **Description:** Vulnerable to DoS via Redis unavailability. Operator hangs indefinitely.
- **Fix:** Add `socket_timeout=5.0` and `socket_connect_timeout=3.0` to Redis client.

#### H-SEC-4: `REQUIRE_MODEL_SIGNATURE = True` hardcoded overrides env var
- **Domain:** Security
- **Description:** Line 1010 overrides env var check unconditionally. Dev testing impossible.
- **Fix:** Remove hardcoded override. Use `os.getenv('REQUIRE_MODEL_SIGNATURE', 'true').lower() == 'true'`.

#### H-ML-1: Beta calibration implementation not verified
- **Domain:** ML
- **Description:** `calibrate_beta.py` referenced but implementation details not shown.
- **Fix:** Verify calibration uses held-out data, optimizes F1, not performed on warmup data.

#### H-ML-2: Determinism flags incomplete
- **Domain:** ML
- **Description:** Only `torch.backends.cudnn.deterministic` set. Missing `benchmark=False` and `use_deterministic_algorithms`.
- **Fix:** Add all determinism flags. Document `PYTHONHASHSEED=42`.

#### H-ML-3: Missing seeds for CUDA
- **Domain:** ML
- **Description:** `torch.cuda.manual_seed_all(seed)` not called for multi-GPU training.
- **Fix:** Add CUDA seed setting alongside CPU seeds.

#### H-MON-6: No availability metric
- **Domain:** Monitoring
- **Description:** `availability_target = 0.999` defined but not computed. No uptime visibility.
- **Fix:** Expose `error_count` and `total_count` counters. Compute availability as `(total - errors) / total`.

#### H-MON-7: No FPR tracking metric
- **Domain:** Monitoring
- **Description:** `fpr_target = 0.05` defined but not measured. Requires ground truth labels.
- **Fix:** If IEC provides labels, expose FPR metric. Document limitation if not available.

#### H-MON-8: Redis beta polling has no health metric
- **Domain:** Monitoring
- **Description:** `_get_beta_from_redis()` has try/except but Redis errors not metriced.
- **Fix:** Add Redis connection failure counter and latency histogram.

#### H-MON-9: No checkpoint metrics
- **Domain:** Monitoring
- **Description:** PyFlink checkpoint state not instrumented. `cadqstream_checkpoints_failed_total` not exposed.
- **Fix:** Hook into Flink checkpoint listener interface or expose metrics from checkpoint operations.

#### H-MON-10: Logging not JSON-structured
- **Domain:** Monitoring
- **Description:** Plain text logs difficult to parse in ELK/Loki/Splunk.
- **Fix:** Use `python-json-logger` or `structlog` for JSON-structured output.

---

### MEDIUM Issues (Fix Before or After Launch)

| ID | Domain | Issue | Fix |
|----|--------|-------|-----|
| M-FL-1 | Flink | `_clone_base_model` does full deep copy on every new key | Add model cache or document limitation |
| M-FL-2 | Flink | Error handling swallows exceptions, yields fallback as "normal" | Use side output for error records |
| M-FL-3 | Flink | No `open()` timeout or retry for model loading | Add retry with exponential backoff |
| M-FL-4 | Flink | No TTL on `ValueState` for memory | Add 7-day TTL with RocksDB compaction |
| M-FL-5 | Flink | No Flink version-specific code paths documented | Document target (1.20.x LTS recommended) |
| M-DE-1 | Data Eng | Month not circularly encoded | Add month_sin/month_cos features |
| M-DE-2 | Data Eng | No duplicate trip ID detection | Add uniqueness check |
| M-DE-3 | Data Eng | Location ID range not validated | Add NYC TLC zone ID validation (1-263) |
| M-DE-4 | Data Eng | Recalibration strategy missing | Implement rolling calibration window |
| M-DE-5 | Data Eng | Checkpoint data retention undefined | Define retention policy for recovery |
| M-DK-1 | Docker | No graceful shutdown code | Add SIGTERM/SIGINT handlers |
| M-DK-2 | Docker | No log aggregation config | Configure json-file driver with rotation |
| M-DK-3 | Docker | No rollback automation | Add AUTO_ROLLBACK_ON_DISAGREEMENT env var |
| M-DK-4 | Docker | Prometheus alerts but no scrape config | Add prometheus scrape config |
| M-SEC-1 | Security | Beta updates not bound to operator identity | Add operator_id to signed payload |
| M-SEC-2 | Security | HMAC parsing uses colon delimiter (fragile) | Use JSON encoding for payload |
| M-SEC-3 | Security | No key rotation strategy documented | Document rotation procedure |
| M-SEC-4 | Security | FeatureVectorizer defaults missing fields to 0.0 | Add strict mode with validation |
| M-ML-1 | ML | Warmup noise std ablation not documented | Track ablation results in production |
| M-ML-2 | ML | `is_all_finite` missing fallback | Verify PyTorch 1.8+ or add fallback |
| M-ML-3 | ML | Memory utilization not used in decision making | Alert when cycling begins |
| M-MON-11 | Monitoring | No correlation IDs | Generate UUID per record, propagate |
| M-MON-12 | Monitoring | Beta value tracking missing | Expose beta gauge per neighborhood |
| M-MON-13 | Monitoring | No model version in logs | Include in all log lines |
| M-MON-14 | Monitoring | Metric logging only every 60s | Log/metric on first anomaly/error |
| M-MON-15 | Monitoring | No latency distribution labels | Add neighborhood/scoring_method labels |

---

### LOW Issues (Nice to Have)

| ID | Domain | Issue | Fix |
|----|--------|-------|-----|
| L-FL-1 | Flink | `REQUIRE_MODEL_SIGNATURE = True` dead code | Remove or structure properly |
| L-FL-2 | Flink | LOGGER never configured with handler | Add basic console handler fallback |
| L-FL-3 | Flink | No Flink version-specific considerations | Document 1.20.x vs 2.x differences |
| L-DE-1 | Data Eng | No data lineage tracking | Add trace IDs |
| L-DE-2 | Data Eng | Missing data freshness metrics | Track ingestion lag |
| L-DE-3 | Data Eng | No backfill strategy | Document procedure |
| L-DK-1 | Docker | No `.dockerignore` | Add to reduce image size |
| L-DK-2 | Docker | No multi-stage build | Implement for smaller image |
| L-DK-3 | Docker | No image tag strategy | Document versioning |
| L-SEC-1 | Security | No SHA256 verification of model file | Add content hash verification |
| L-SEC-2 | Security | No Prometheus alert for Redis auth failures | Add alert |
| L-SEC-3 | Security | Signing key minimum length not enforced | Add 32-char validation |
| L-ML-1 | ML | Documentation inconsistency (v3/v4 title) | Fix header consistency |
| L-ML-2 | ML | `torch.no_grad()` called incorrectly | Remove (unnecessary after `self.eval()`) |
| L-ML-3 | ML | Missing LOGGER import | Add `import logging` |
| L-MON-16 | Monitoring | IEC cooldown metric missing | Expose cooldown state gauge |
| L-MON-17 | Monitoring | No memory freshness metric | Expose utilization per neighborhood |
| L-MON-18 | Monitoring | Alert annotations lack runbook links | Add `runbook_url` annotations |
| L-MON-19 | Monitoring | No canary traffic metrics | Add shadow/canary mode indicators |

---

## Cross-Cutting Concerns

### Issues Spanning Multiple Domains

| Issue | Domains Affected | Description |
|-------|------------------|-------------|
| HMAC bypass vulnerability | Security + Docker | If key not set, beta manipulation possible |
| Duplicate HMAC code | Security + ML | Lines 564-589 in memstream_core.py |
| Redis security | Security + Docker + Monitoring | No auth, no TLS, no metrics |
| Missing metrics | Monitoring + Security | HMAC failures not metricked |
| No docker artifacts | Docker + Flink | Dockerfile + compose required |
| Data integrity | Data Eng + ML | Temporal shuffle + normalization leakage |

### Risk Assessment

**Overall Risk Posture:** MODERATE-HIGH

The architecture is sound and the v4 redesign addresses all v1-v3 foundational issues. However, the plan is not production-ready due to:

1. **Security regressions** from v3 fixes not fully implemented
2. **Missing deployment artifacts** (Dockerfile, compose) despite stated deployment strategy
3. **No observability infrastructure** despite defined SLOs and alerts
4. **Data integrity issues** that invalidate evaluation

**Top 3 risks requiring immediate attention:**
1. HMAC bypass (C-SEC-1) — allows unrestricted anomaly detection manipulation
2. No Dockerfile/compose (C-DK-1, C-DK-2) — deployment blocked
3. No Prometheus metrics (C-MON-1) — no operational visibility

---

## What's Working Well

### Flink
- Correct `KeyedProcessFunction` usage keyed by neighborhood
- Efficient checkpoint design (memory-only, ~400KB per neighborhood)
- HMAC security for Redis communication with `hmac.compare_digest`
- Frozen normalization stats prevent score drift
- Circuit breaker design (cooldown + max consecutive limits)
- Comprehensive test suite (regression, serialization, edge cases)

### Data Engineering
- 25D feature vector correctly implemented
- Circular encoding for hour and day-of-week
- Separate calibration set (20%) not used in training
- ECE/MCE metrics for calibration quality
- Proper data type enforcement (float32 throughout)

### Docker/SRE
- 3-phase dark launch documented correctly (shadow → canary → production)
- Resource configuration template provided
- Healthcheck concept identified

### Security
- `weights_only=True` prevents RCE from malicious pickle files
- HMAC verification architecture sound (when keys are set)
- Timing-safe comparison (`hmac.compare_digest`) prevents timing attacks
- Model SHA256 size check prevents memory exhaustion

### ML
- AE architecture matches MemStream paper (25→50→25 with Tanh)
- Memory mechanism correct (FIFO, update gating, gradient detachment)
- Scoring function correct (L1 distance to minimum memory slot)
- Frozen normalization prevents drift (v3 fix verified)
- Early stopping pattern correct (best state saved)
- Training/validation/calibration/test splits proper

### Monitoring
- SLO targets well-defined in `SLOConfig`
- Alert expressions in `prometheus_alerts.yaml` are well-formed
- Latency threshold (100ms p99) appropriate

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)

**Priority: Eliminate Security Vulnerabilities and Deployment Blockers**

| # | Issue | Owner | Effort |
|---|-------|-------|--------|
| 1 | Add Dockerfile to `memstream_src/operators/deployment/Dockerfile` | DevOps | 2h |
| 2 | Create docker-compose.yml with Redis + MemStream + Metrics | DevOps | 3h |
| 3 | Enforce HMAC signing key at startup (fail-fast) | Security | 1h |
| 4 | Add Redis authentication + TLS configuration | Security | 2h |
| 5 | Add `import hashlib, hmac` to operator | Flink | 5min |
| 6 | Move circuit breaker to BroadcastState | Flink | 4h |
| 7 | Implement time-bounded Redis polling (10s interval) | Flink | 3h |
| 8 | Initialize `max_thres` in `__init__` | ML | 10min |
| 9 | Fix data split (time-ordered instead of shuffle) | Data Eng | 1h |
| 10 | Fix normalization leakage (split warmup data) | ML | 2h |
| 11 | Add Prometheus instrumentation to operator | Monitoring | 4h |
| 12 | Remove duplicate HMAC verification code | ML/Security | 5min |

**Phase 1 Exit Criteria:** 
- [ ] Docker build succeeds
- [ ] HMAC bypass impossible (verified by startup check)
- [ ] Redis metrics visible in Prometheus
- [ ] Checkpoint contains only memory state
- [ ] Data splits are time-ordered

### Phase 2: High Priority (Week 2)

**Priority: Complete Observability and Operational Readiness**

| # | Issue | Owner | Effort |
|---|-------|-------|--------|
| 1 | Add OpenTelemetry tracing spans | Monitoring | 4h |
| 2 | Add JSON structured logging with correlation IDs | Monitoring | 3h |
| 3 | Add SLO burn-rate alerts | Monitoring | 3h |
| 4 | Add watermark strategy to pipeline | Flink | 2h |
| 5 | Define Kafka topic structure + retention | Data Eng | 2h |
| 6 | Implement late data handling (side output) | Data Eng | 3h |
| 7 | Add model loading retry with backoff | Flink | 2h |
| 8 | Add health endpoint server | DevOps | 2h |
| 9 | Implement dark launch traffic splitting | DevOps | 4h |
| 10 | Add beta value gauge per neighborhood | Monitoring | 2h |
| 11 | Add Redis health metrics | Monitoring | 2h |

**Phase 2 Exit Criteria:**
- [ ] All Prometheus alerts have underlying metrics
- [ ] Traces visible in Jaeger/Zipkin
- [ ] Logs queryable by trace_id, neighborhood
- [ ] Dark launch can be executed

### Phase 3: Medium Priority (Week 3)

**Priority: Polish and Hardening**

| # | Issue | Owner | Effort |
|---|-------|-------|--------|
| 1 | Add multi-strategy anomaly injection | Data Eng | 3h |
| 2 | Add circular encoding for month | Data Eng | 1h |
| 3 | Add graceful shutdown handlers | DevOps | 2h |
| 4 | Add TTL to memory state (7 days) | Flink | 2h |
| 5 | Add version compatibility check in `open()` | Flink | 1h |
| 6 | Document Flink version requirements | Flink | 1h |
| 7 | Implement rolling calibration window | ML | 4h |
| 8 | Add deterministic training flags | ML | 1h |
| 9 | Define Grafana dashboard panels | Monitoring | 4h |

### Phase 4: Post-Launch

**Priority: Future Work**

- Secrets manager migration (Vault/AWS)
- mTLS for Redis
- Operator identity binding for beta updates
- Checkpoint metrics instrumentation
- Model version tracking dashboard
- Canary comparison automation

---

## Verification Plan

### Pre-Deployment Checklist

- [ ] **Docker build verified:** `docker build -f memstream_src/operators/deployment/Dockerfile .`
- [ ] **HMAC enforcement tested:** Verify crash when `IEC_SIGNING_KEY` not set
- [ ] **Redis connectivity verified:** `docker compose run --rm memstream-operator redis-cli -h redis ping`
- [ ] **HMAC verification tested:** Verify crash on tampered model
- [ ] **Checkpoint recovery tested:** Restart job, verify per-key memory restored
- [ ] **Redis beta flow verified:** Send IEC action, verify operator picks up within 10s
- [ ] **Circuit breaker persistence tested:** Kill JobManager, verify state preserved
- [ ] **P99 latency benchmarked:** 10,000 records/sec, verify < 100ms
- [ ] **Dark launch tested:** Shadow → canary → production phases
- [ ] **Graceful shutdown tested:** `docker stop` preserves state
- [ ] **Data splits verified:** Time-ordered, no shuffle

### Post-Deployment Monitoring

**Key Metrics to Watch:**

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| memstream_latency_seconds_p99 | < 100ms | > 100ms |
| anomaly_rate_per_context | 1% - 15% | < 1% or > 15% |
| cadqstream_iec_consecutive_actions | < 3 | >= 3 |
| cadqstream_model_integrity_failures_total | 0 | > 0 |
| cadqstream_hmac_verification_failures_total | 0 | > 0 |
| memstream_beta_per_context (stddev) | < 0.5 | > 0.5 |
| cadqstream_redis_connection_failures_total | 0 | > 0 |
| memstream_anomaly_rate_current | 1% - 15% | < 1% |

**Dashboard Panels Required:**
1. SLO Health (latency p99, error rate, burn rate)
2. Throughput (records/sec, anomalies/sec)
3. Latency distribution (p50/p90/p99 heatmap)
4. IEC Health (consecutive actions, beta values, divergence)
5. Model & Checkpoint (integrity failures, checkpoint success)
6. Redis Health (connection errors, HMAC failures)

---

## Recommendation

**Move to Phase 1 Critical Fixes:** The plan has solid architectural foundations (Redis-based IEC, memory-only checkpointing, HMAC verification, 3-phase dark launch). All v1-v3 CRITICAL issues were properly addressed in v4. The remaining 18 CRITICAL issues are localized and fixable within 1 week.

**Estimated Fix Effort:** ~10-15 engineering days for all CRITICAL + HIGH issues.

### Immediate Actions Required

1. **Create Dockerfile** (2 hours) — blocking deployment
2. **Create docker-compose.yml** (3 hours) — blocking deployment
3. **Enforce HMAC signing key** (1 hour) — critical security
4. **Add Prometheus metrics** (4 hours) — operational visibility
5. **Fix data integrity issues** (3 hours) — evaluation validity

### Confidence Assessment

| Dimension | Confidence | Notes |
|-----------|------------|-------|
| Architecture | HIGH | Core design sound |
| Security | MEDIUM | HMAC bypass must be fixed |
| Observability | LOW | No metrics exposed yet |
| Deployment | LOW | No artifacts exist |
| Data Integrity | MEDIUM | Shuffle + leakage issues |
| ML Correctness | HIGH | Algorithm verified |

---

## Prior v3 Issues Status

| v3 Issue | Status in v4 |
|----------|--------------|
| Broadcast state trap (API misuse) | ✅ FIXED |
| Memory serialization errors | ✅ FIXED |
| Context mismatches | ✅ FIXED |
| Redis-based IEC communication | ✅ FIXED |
| HMAC verification | ⚠️ PARTIAL (bypass path exists) |
| Circuit breaker | ⚠️ PARTIAL (not checkpointable) |
| Dockerfile | ❌ NOT ADDRESSED (new critical gap) |
| Docker compose | ❌ NOT ADDRESSED (new critical gap) |

---

## Appendix: Issue Count Summary by Reviewer

| Reviewer | CRITICAL | HIGH | MEDIUM | LOW |
|----------|----------|------|--------|-----|
| Flink Engineer | 3 | 4 | 5 | 4 |
| Data Engineer | 2 | 4 | 5 | 3 |
| Docker/SRE | 4 | 4 | 4 | 3 |
| Security | 3 | 4 | 4 | 3 |
| ML Engineer | 1 | 3 | 3 | 3 |
| Monitoring | 5 | 5 | 5 | 4 |
| **TOTAL** | **18** | **24** | **26** | **20** |

---

*Compiled by Synthesis Agent — 2026-05-12*
*Source Reviews: REVIEW_FLINK_v4.md, REVIEW_DATAENG_v4.md, REVIEW_DOCKER_v4.md, REVIEW_SECURITY_v4.md, REVIEW_ML_v4.md, REVIEW_MONITORING_v4.md*
