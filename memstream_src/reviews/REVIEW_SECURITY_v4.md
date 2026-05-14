# Security Engineer Review - CA-DQStream + MemStream Hybrid v4

**Reviewer:** Security Engineer
**Date:** 2026-05-12
**Plan Version:** v4 (Expert review fixes applied)
**Target:** `memstream_src/PLAN_v4.md`

---

## Summary

The v4 plan addresses 7 previously critical/high issues (Redis-based IEC communication, HMAC enforcement, etc.). From a security perspective, **4 CRITICAL/HIGH issues remain unfixed**, and **8 MEDIUM/LOW hardening opportunities** should be addressed before production deployment.

**Overall Security Posture:** ⚠️ **CONDITIONAL PASS** — Security-critical gaps exist in secrets management, HMAC bypass paths, Redis transport security, and DoS resilience. Fixes are achievable but must be implemented before production.

---

## Security Analysis

### 1. Model Integrity (torch.load, HMAC, Signing)

#### ✅ weights_only=True (RCE Mitigation)
Both model loading paths correctly use `weights_only=True`:
- Primary model load: `torch.load(path, map_location=..., weights_only=True)` at line 598
- Memory-only checkpoint: `torch.load(buf, map_location='cpu', weights_only=True)` at line 1249

This prevents arbitrary code execution via malicious pickle payloads in model files.

#### ⚠️ HMAC Verification: Redundant Code + Bypass Path

**Finding 1: HMAC Verification Duplicated (Code Quality / Security)**

Lines 564-589 in `MemStreamCore.load()` contain duplicate HMAC verification blocks:

```564:    if signing_key:
565:        # ... HMAC check ...
566:        if not os.path.exists(path + '.hmac'):
567:            if require_signature:
568:                raise SecurityError(...)
569:            else:
570:                LOGGER.warning(...)
571:        else:
572:            # verify HMAC
573:    elif require_signature:
574:        raise SecurityError(...)
575:
576:    # [DUPLICATE BLOCK - lines 577-589]
577:    if signing_key:  # ← executes same check again
578:        with open(path + '.hmac') as f:
579:            expected_hmac = f.read().strip()
580:        with open(path + '.hmac') as f:  # ← file opened twice
581:            actual_hmac = hmac.new(...)
582:        if not hmac.compare_digest(...):
583:            raise SecurityError(...)
```

**Impact:** Wastes I/O (opens `.hmac` file twice per load). While not a security vulnerability per se, duplicate code paths are maintenance hazards and increase attack surface.

**Recommendation:** Remove the duplicate block (lines 577-589). The first block (lines 564-576) already handles all cases correctly.

---

**Finding 2: HMAC Bypass When Signing Key Absent in IEC Path**

In `IECFeedbackOperator.process_broadcast_element()` (lines 1318-1329):

```1316:            beta_str = str(beta)
1317:            if IEC_SIGNING_KEY:
1318:                sig = hmac.new(...)
1319:                redis_client.set(f'beta:{neighborhood}', f'{beta_str}:{sig}')
1320:            else:
1321:                # [HIGH v4] Warn if no signing key
1322:                LOGGER.warning(f"[IECFeedback] No IEC_SIGNING_KEY, beta update unsigned")
1323:                redis_client.set(f'beta:{neighborhood}', beta_str)  # ← UNSIGNED
```

And in `MemStreamScoringOperator._get_beta_from_redis()` (lines 1104-1113):

```1104:            if IEC_SIGNING_KEY:  # ← Reads operator's signing key
1105:                expected_sig = hmac.new(...)
1106:                if not hmac.compare_digest(signature, expected_sig):
1107:                    LOGGER.error(...)
1108:                    return None
1109:            # If IEC_SIGNING_KEY is None, HMAC check is SKIPPED entirely
```

**Impact:** CRITICAL — If `IEC_SIGNING_KEY` env var is not set in either operator:
1. IECFeedbackOperator writes unsigned beta values to Redis
2. MemStreamScoringOperator skips HMAC verification (because `IEC_SIGNING_KEY` is None)
3. Result: **Unauthenticated beta updates** — same vulnerability as C6 in v3 that was supposedly fixed

**Attack Scenario:** Any process with Redis access (internal compromise, SSRF, or shared-hosting vulnerability) can set arbitrary `beta:{neighborhood}` keys, directly manipulating anomaly detection thresholds. This could:
- Suppress all anomaly detection by setting `beta:neighborhood = 999999`
- Flood the system with false positives by setting `beta:neighborhood = 0.0001`

**Recommendation:**
```python
# IECFeedbackOperator — enforce signing in production
if not IEC_SIGNING_KEY:
    raise RuntimeError(
        "[IECFeedback] FATAL: IEC_SIGNING_KEY environment variable is required. "
        "Unsigned beta updates are not permitted."
    )

# MemStreamScoringOperator — enforce verification
if IEC_SIGNING_KEY is None:
    raise RuntimeError(
        "[MemStream] FATAL: IEC_SIGNING_KEY environment variable is required. "
        "Unsigned beta updates will not be accepted."
    )
```

Add a startup check (in `open()`) rather than checking per-record.

---

#### ⚠️ HMAC Signature Parsing: Colon Delimiter Collision

In `_get_beta_from_redis()` (line 1096):

```1096:            parts = raw_value.rsplit(':', 1)
```

In `IECFeedbackOperator` (line 1325):

```1325:                redis_client.set(f'beta:{neighborhood}', f'{beta_str}:{sig}')
```

**Impact:** MEDIUM — HMAC signatures use hex digits (0-9, a-f), but `beta_str` could theoretically contain a colon if:
- Beta value formatting changes (e.g., scientific notation `1e-5:abc123...` has no colon, but custom formatting could)
- Future expansion adds metadata fields

Using `rsplit(':', 1)` with limit=1 is correct (splits from right, so `value:with:colons:hexsig` → `['value:with:colons', 'hexsig']`). This is safe as long as signatures don't contain colons (they don't — hex only). However, the `beta_str` itself is also parsed as `float(beta_str)` which would reject non-numeric values. **Low risk but fragile.**

**Recommendation:** Use a separator that's guaranteed not in either field. JSON encoding is more robust:
```python
import json
# Write: redis_client.set(f'beta:{neighborhood}', json.dumps({'beta': beta, 'sig': sig}))
# Read: data = json.loads(raw_value); beta = float(data['beta']); sig = data['sig']
```

---

### 2. Authentication & Authorization

#### ⚠️ Redis Authentication: Password Optional, No TLS

Lines 997-999:

```997:REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
998:REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
999:REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)  # Optional
```

**Impact:** HIGH — In distributed deployments:
- If Redis is on a different host, credentials and data traverse the network unencrypted
- `REDIS_PASSWORD = None` silently disables authentication
- No TLS/SSL configuration provided
- Any process on the network segment can read/write Redis keys

**Recommendation:**
```python
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')  # No default — fail if missing in prod
REDIS_TLS = os.getenv('REDIS_TLS', 'false').lower() == 'true'

# redis.Redis configuration:
self._redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,  # Required in production
    ssl=REDIS_TLS,            # Enable TLS
    ssl_cert_reqs='required', # Reject invalid certs
    socket_timeout=5.0,       # Prevent indefinite hangs
    socket_connect_timeout=3.0,
)
```

Add Prometheus alert for failed Redis auth attempts:
```yaml
- alert: RedisAuthenticationFailures
  expr: rate(redis_rejected_connections_total[5m]) > 0
  annotations:
    summary: "Redis authentication failures detected"
```

---

#### ⚠️ IEC Beta Update Authorization: No Operator Identity

Beta updates are HMAC-signed but **not bound to operator identity**. The HMAC only verifies the payload integrity (beta value hasn't been tampered with), not the *source* of the update.

**Impact:** MEDIUM — If the HMAC key is compromised:
1. Attacker can send valid HMAC-signed beta updates from anywhere
2. No audit trail of which operator sent the update
3. No rate limiting per operator instance

**Recommendation:** Add operator instance ID to the signed payload:
```python
OPERATOR_INSTANCE_ID = os.getenv('IEC_OPERATOR_INSTANCE_ID', 'unknown')

payload = json.dumps({
    'operator_id': OPERATOR_INSTANCE_ID,
    'neighborhood': neighborhood,
    'beta': beta,
    'timestamp': time.time(),
})
sig = hmac.new(IEC_SIGNING_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
redis_client.set(f'beta:{neighborhood}', json.dumps({'payload': payload, 'sig': sig}))
```

---

### 3. Secrets Management

#### ⚠️ Model Signing Key: Environment Variable Pattern

Lines 1000-1001 and throughout:

```1000:    IEC_SIGNING_KEY = os.getenv('IEC_SIGNING_KEY', None)
1001:    MODELSIGNING_KEY = os.getenv('MEMSTREAM_MODEL_SIGNING_KEY', None)
```

**Impact:** MEDIUM — Environment variables for secrets are a known anti-pattern in production:

1. **Process listing exposure:** `ps aux | grep` or `/proc/*/environ` can read env vars
2. **Log leakage:** If env vars are printed in stack traces or error messages, secrets leak
3. **No key rotation mechanism:** Changing the key requires restarting all services
4. **No audit trail:** Who changed the key? When?

**Recommendation:** For production, use a secrets management solution:
- **Vault** (HashiCorp): `vault kv get -field=key secret/memstream/iec-signing-key`
- **AWS Secrets Manager / GCP Secret Manager** via SDK
- **Kubernetes Secrets** (with encryption at rest + RBAC)

For the Docker deployment, at minimum:
```bash
# Dockerfile: Read from mounted secret, not baked-in env
# docker-compose.yml
services:
  memstream:
    secrets:
      - iec_signing_key
    environment:
      - IEC_SIGNING_KEY__FILE=/run/secrets/iec_signing_key
```

Also add at startup:
```python
if IEC_SIGNING_KEY and len(IEC_SIGNING_KEY) < 32:
    raise ValueError("[SECURITY] IEC_SIGNING_KEY must be at least 32 characters (256-bit)")
```

---

#### ⚠️ No Key Rotation Strategy

**Impact:** LOW — No documented key rotation procedure exists. If a signing key is compromised, there is no defined recovery process.

**Recommendation:** Document rotation procedure:
1. Generate new key
2. Re-sign all model files with new key
3. Deploy new key to all operators (rolling restart)
4. Monitor for failures during transition window
5. Old key can be retained briefly for rollback

---

### 4. Network Security

#### ⚠️ No TLS Configuration for Redis

As noted in Section 2. Redis connections in production (distributed Flink deployment) will be unencrypted.

**Impact:** HIGH — Man-in-the-middle attacks can:
- Read beta values and infer anomaly detection thresholds
- Modify beta values if HMAC is bypassed (see Finding 2)
- Intercept Redis credentials if authentication is used without TLS

**Recommendation:** Add TLS configuration (see Section 2 for code).

---

#### ⚠️ No Network Policies in Docker-Only Deployment

The plan explicitly states "Docker-only, no K8s." For Docker deployments:
- No network isolation between containers
- Redis port 6379 defaults to `localhost` (acceptable for single-host Docker Compose)
- If Redis is external, no firewall rules are specified

**Impact:** LOW (for single-host Docker Compose) / HIGH (for multi-host).

**Recommendation:** For multi-host deployments:
```yaml
# docker-compose.yml network section
networks:
  memstream_internal:
    driver: bridge
    internal: false  # Allow external for Redis host if needed
    ipam:
      config:
        - subnet: 172.28.0.0/16

services:
  redis:
    networks:
      - memstream_internal
    ports: []  # No exposed ports if MemStream is on same network
```

---

### 5. Input Validation

#### ✅ Feature Vector Shape Validation

`score_one()` at lines 736-741 correctly validates input shape:

```736:        features = np.asarray(features, dtype=np.float32)
737:        if features.shape != (self.in_dim,):
738:            raise ValueError(
739:                f"Expected features shape ({self.in_dim},), got {features.shape}. "
740:                "This likely means the FeatureVectorizer returned wrong dimensions."
741:            )
```

#### ✅ NaN/Inf Handling

`torch.where(torch.isfinite(...), ...)` guards at lines 661-662, 747-748 prevent NaN propagation.

#### ⚠️ FeatureVectorizer: No Schema Enforcement on Input

`FeatureVectorizer.transform()` uses `.get()` with defaults:

```313:        trip_distance = safe(record.get('trip_distance'))
314:        dur_min = safe(record.get('dur_min'))
315:        # ... all fields with safe() defaults to 0.0
```

**Impact:** LOW — Missing fields silently default to 0.0. This could mask data pipeline issues.

**Recommendation:** Add optional strict mode:
```python
class FeatureVectorizer:
    def __init__(self, dtype=np.float32, strict=False):
        self.strict = strict

    def transform(self, record, strict=None):
        strict = strict if strict is not None else self.strict
        missing = [f for f in REQUIRED_FIELDS if record.get(f) is None]
        if missing and strict:
            raise ValueError(f"Missing required fields: {missing}")
```

---

#### ⚠️ Model File Size: 500MB Limit Acceptable but No Content Validation

Lines 1023-1027 check file size but not content:

```1023:    if os.path.getsize(MODEL_PATH) > 500 * 1024 * 1024:
1024:        raise ValueError(
1025:            f"[MemStreamOp] FATAL: Model file at {MODEL_PATH} is too large "
1026:            f"({os.path.getsize(MODEL_PATH)} bytes). Possible attack."
1027:        )
```

**Impact:** LOW — While `weights_only=True` prevents code execution, a maliciously crafted pickle could still:
- Contain tensors with extreme values that cause GPU OOM
- Contain specially crafted tensors that trigger numerical instability
- Consume excessive memory during load

**Recommendation:** Add SHA256 verification against a known-good hash:
```python
# Expected hash should be computed during model training and stored securely
EXPECTED_MODEL_SHA256 = os.getenv('MEMSTREAM_MODEL_SHA256', None)
if EXPECTED_MODEL_SHA256:
    actual_hash = hashlib.sha256(open(MODEL_PATH, 'rb').read()).hexdigest()
    if actual_hash != EXPECTED_MODEL_SHA256:
        raise SecurityError(f"Model SHA256 mismatch: expected {EXPECTED_MODEL_SHA256[:16]}..., got {actual_hash[:16]}...")
```

---

## Vulnerabilities Found

### CRITICAL Issues

| ID | Issue | Location | Status |
|----|-------|----------|--------|
| **C-SEC-1** | HMAC bypass: IEC beta updates skip verification when `IEC_SIGNING_KEY=None` in either operator | `memstream_scoring_op.py:1104-1108`, `1318-1329` | **UNFIXED** |
| **C-SEC-2** | Redis unauthenticated by default (`REDIS_PASSWORD=None`); no TLS in transit | `memstream_scoring_op.py:999`, `1074-1079` | **UNFIXED** |
| **C-SEC-3** | Duplicate HMAC verification code (lines 564-589 vs 577-589) creates maintenance hazard | `memstream_core.py:577-589` | **UNFIXED** (Code quality) |

### HIGH Issues

| ID | Issue | Location | Status |
|----|-------|----------|--------|
| **H-SEC-1** | Redis credentials and data traverse network unencrypted | `memstream_scoring_op.py:997-999` | **UNFIXED** |
| **H-SEC-2** | HMAC key is env var — exposed via process listing and logs | Throughout: env var pattern | **UNFIXED** |
| **H-SEC-3** | No Redis connection timeout — vulnerable to DoS via Redis unavailability | `memstream_scoring_op.py:1074-1079` | **UNFIXED** |
| **H-SEC-4** | `REQUIRE_MODEL_SIGNATURE = True` hardcoded overrides env var in production | `memstream_scoring_op.py:1010` | **UNFIXED** |

### MEDIUM Issues

| ID | Issue | Location | Status |
|----|-------|----------|--------|
| **M-SEC-1** | Beta updates not bound to operator identity — no audit trail | `iec_feedback_op.py:1316-1325` | **UNFIXED** |
| **M-SEC-2** | HMAC parsing uses colon delimiter — fragile if payload format changes | `memstream_scoring_op.py:1096` | **UNFIXED** |
| **M-SEC-3** | No key rotation strategy documented | Throughout | **UNFIXED** |
| **M-SEC-4** | FeatureVectorizer silently defaults missing fields to 0.0 | `feature_extractor.py:313-315` | **UNFIXED** |

### LOW Issues

| ID | Issue | Location | Status |
|----|-------|----------|--------|
| **L-SEC-1** | No SHA256 verification of model file content (only size check) | `memstream_scoring_op.py:1018-1027` | **UNFIXED** |
| **L-SEC-2** | No Prometheus alert for Redis auth failures | `prometheus_alerts.yaml` | **UNFIXED** |
| **L-SEC-3** | Signing key minimum length not enforced | Throughout | **UNFIXED** |

---

## Threat Model

### Attack Vectors

```
┌─────────────────────────────────────────────────────────────────┐
│                     THREAT MODEL: CA-DQStream + MemStream       │
└─────────────────────────────────────────────────────────────────┘

1. MODEL TAMPERING
   Attacker modifies model.pt → HMAC mismatch → Blocked by weights_only=True ✅
   Attacker modifies .hmac file → HMAC mismatch → Blocked ✅
   Attacker suppresses HMAC check → require_signature=True blocks ⚠️ (if set)

2. IEC BETA MANIPULATION (CRITICAL)
   ┌──────────────────────────────────────────────────────────────┐
   │  Path A: IEC_SIGNING_KEY set in BOTH operators              │
   │    IECFeedbackOp signs → Redis stores signed → MemStream    │
   │    verifies HMAC → Accepted only if valid ⚠️                 │
   │    Risk: Key compromise → invalid beta accepted            │
   │                                                              │
   │  Path B: IEC_SIGNING_KEY missing in ANY operator            │
   │    IECFeedbackOp writes UNSIGNED → MemStream skips verify   │
   │    → UNRESTRICTED beta manipulation ⚠️⚠️⚠️ (C-SEC-1)         │
   └──────────────────────────────────────────────────────────────┘

3. REDIS COMPROMISE
   - Network access to Redis → Read beta values (confidentiality)
   - No TLS → MITM modifies beta mid-flight (integrity)
   - No AUTH → Unauthorized write to any key (integrity)
   - No TIMEOUT → Redis down → MemStream hangs (availability)

4. SECRETS EXFILTRATION
   - Env vars readable via /proc/*/environ
   - Process listing shows env vars
   - Logs accidentally print secrets
   - No key rotation → compromised key remains valid indefinitely

5. DENIAL OF SERVICE
   - Redis timeout missing → operator hangs indefinitely
   - Large model file (>500MB) → memory exhaustion
   - Malicious checkpoint payload → memory/CPU exhaustion
   - Flash crash via beta=0.0 → all records flagged as anomalies
   - Suppression via beta=999999 → no anomalies detected
```

### Attack Scenario: Beta Suppression Attack

1. Attacker gains foothold on a container in the same Docker network
2. Discovers Redis is accessible at `redis:6379` (Docker DNS)
3. `IEC_SIGNING_KEY` is not set (development configuration)
4. Attacker runs: `redis-cli SET 'beta:manhattan' '999999.0'`
5. All Manhattan anomaly detection is suppressed
6. Fraudulent taxi trips go undetected for the cooldown period (300s minimum)
7. IEC circuit breaker trips after 3 consecutive actions — but attacker only needs one

**This scenario is achievable if `IEC_SIGNING_KEY` is not set in production, which is the default.**

---

## Recommendations

### Immediate (Before Production)

#### REC-1: Enforce HMAC Verification (Fix C-SEC-1)

In `memstream_scoring_op.py`, add startup validation:

```python
# After line 1001, add:
def _validate_hmac_config():
    """Fail fast if HMAC is not properly configured."""
    if not IEC_SIGNING_KEY:
        raise RuntimeError(
            "[MemStream] FATAL: IEC_SIGNING_KEY environment variable is required. "
            "Beta updates will not be accepted without HMAC signing. "
            "Set IEC_SIGNING_KEY to a 256-bit secret (32+ characters)."
        )
    if not MODEL_SIGNING_KEY:
        raise RuntimeError(
            "[MemStream] FATAL: MEMSTREAM_MODEL_SIGNING_KEY environment variable is required. "
            "Model files will not be loaded without HMAC verification. "
            "Set MEMSTREAM_MODEL_SIGNING_KEY to a 256-bit secret (32+ characters)."
        )

_validate_hmac_config()
```

In `iec_feedback_op.py`, add the same check in `__init__`:

```python
if not IEC_SIGNING_KEY:
    raise RuntimeError(
        "[IECFeedback] FATAL: IEC_SIGNING_KEY environment variable is required. "
        "Unsigned beta updates are not permitted."
    )
```

**Rationale:** Fail-fast prevents silent bypass. This is the single most important security fix.

---

#### REC-2: Redis Hardening (Fix C-SEC-2, H-SEC-1, H-SEC-3)

```python
import redis
from redis.exceptions import ConnectionError, TimeoutError

REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
REDIS_TLS = os.getenv('REDIS_TLS', 'false').lower() == 'true'
REDIS_SOCKET_TIMEOUT = 5.0
REDIS_SOCKET_CONNECT_TIMEOUT = 3.0

class RedisClient:
    """Hardened Redis client with timeouts and TLS."""

    def __init__(self):
        self._client = None
        self._connect_time = None

    def get_client(self):
        if self._client is None:
            self._client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                ssl=REDIS_TLS,
                ssl_cert_reqs='required' if REDIS_TLS else None,
                socket_timeout=REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            self._connect_time = time.time()
        return self._client

    def get(self, key):
        """Get with timeout and error handling."""
        try:
            return self.get_client().get(key)
        except (ConnectionError, TimeoutError) as e:
            LOGGER.error(f"[Redis] Connection failed: {e}")
            raise RedisUnavailableError(f"Redis unavailable: {e}")
```

Add Prometheus alert:
```yaml
- alert: RedisUnavailable
  expr: rate(memstream_redis_errors_total[5m]) > 0.1
  annotations:
    summary: "Redis unavailable — beta updates blocked"
```

---

#### REC-3: Remove Duplicate HMAC Code (Fix C-SEC-3)

In `memstream_core.py`, delete lines 577-589. The block from line 564 to 576 already handles all HMAC verification cases correctly, including the `require_signature` check in the `elif` branch.

---

### Short-term (Pre-deployment)

#### REC-4: Secrets Management Migration

Replace environment variable secrets with a secrets manager. Document the migration path in `docs/SECURITY.md`:

```python
# Prefer Vault (example)
import hvac
client = hvac.Client(url=os.getenv('VAULT_ADDR'))
client.auth.kubernetes.login(os.getenv('VAULT_K8S_ROLE'))
secret = client.secrets.kv.v2.read_secret_version(
    path='secret/memstream/iec-signing-key'
)
IEC_SIGNING_KEY = secret['data']['data']['key']
```

#### REC-5: Add Model Content Hash Verification

In the pre-flight check section (after line 1027):

```python
EXPECTED_MODEL_SHA256 = os.getenv('MEMSTREAM_MODEL_SHA256')
if EXPECTED_MODEL_SHA256:
    actual = hashlib.sha256(open(MODEL_PATH, 'rb').read()).hexdigest()
    if actual.lower() != EXPECTED_MODEL_SHA256.lower():
        raise SecurityError(
            f"[MemStreamOp] FATAL: Model SHA256 mismatch. "
            f"Expected {EXPECTED_MODEL_SHA256[:16]}..., got {actual[:16]}... "
            "Model file may have been tampered with."
        )
```

#### REC-6: Operator Identity Binding for Beta Updates

Replace the simple `value:signature` format with a structured payload:

```python
import json, time

def _write_beta_update(redis_client, neighborhood, beta):
    """Write beta update with operator identity and timestamp."""
    payload = json.dumps({
        'beta': beta,
        'operator_id': os.getenv('IEC_OPERATOR_ID', 'unknown'),
        'timestamp': time.time(),
    })
    sig = hmac.new(IEC_SIGNING_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    redis_client.set(f'beta:{neighborhood}', json.dumps({'payload': payload, 'sig': sig}))

def _read_beta_update(raw_value):
    """Read and verify beta update."""
    data = json.loads(raw_value)
    payload = data['payload']
    sig = data['sig']
    expected_sig = hmac.new(IEC_SIGNING_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        raise SecurityError("Beta update HMAC mismatch")
    parsed = json.loads(payload)
    return parsed['beta'], parsed.get('operator_id'), parsed.get('timestamp')
```

---

### Long-term (Production Hardening)

#### REC-7: Mutual TLS for Redis

Configure Redis with TLS and require client certificates:

```bash
# redis.conf
tls-port 6380
port 0
tls-cert-file /certs/redis.crt
tls-key-file /certs/redis.key
tls-ca-cert-file /certs/ca.crt
tls-auth-clients no  # Accept clients without certs (cert optional)
# For stricter mTLS:
tls-auth-clients yes  # Require client certificates
```

#### REC-8: Audit Logging for Beta Updates

Emit structured audit logs for every beta update:

```python
import structlog
logger = structlog.get_logger("audit")

logger.info(
    "beta_update",
    action="beta_adjusted",
    neighborhood=neighborhood,
    old_beta=old_beta,
    new_beta=beta,
    operator_id=operator_id,
    timestamp=timestamp,
    redis_key=f'beta:{neighborhood}',
)
```

#### REC-9: Rate Limiting on Beta Updates

Add Redis-based rate limiting:

```python
def _check_rate_limit(redis_client, neighborhood, max_per_hour=5):
    """Enforce rate limit on beta updates per neighborhood."""
    key = f'ratelimit:beta:{neighborhood}'
    count = redis_client.incr(key)
    if count == 1:
        redis_client.expire(key, 3600)  # 1-hour window
    if count > max_per_hour:
        raise RateLimitExceeded(f"Rate limit exceeded for {neighborhood}")
```

---

## Verification Checklist

- [ ] **C-SEC-1**: Startup validation enforces `IEC_SIGNING_KEY` and `MODEL_SIGNING_KEY` presence
- [ ] **C-SEC-1**: `_get_beta_from_redis()` rejects unsigned updates when key is present
- [ ] **C-SEC-2**: Redis password is required (not optional) in production
- [ ] **C-SEC-2**: Redis connections use TLS with certificate validation
- [ ] **C-SEC-3**: Duplicate HMAC verification block removed from `memstream_core.py`
- [ ] **H-SEC-1**: Network traffic between operators and Redis is encrypted
- [ ] **H-SEC-2**: Signing keys are loaded from secrets manager, not environment variables
- [ ] **H-SEC-3**: Redis client has 5-second socket timeout
- [ ] **H-SEC-4**: `REQUIRE_MODEL_SIGNATURE` is env-var controlled, not hardcoded
- [ ] **M-SEC-1**: Beta updates include operator identity in signed payload
- [ ] **M-SEC-2**: Beta value parsing uses JSON, not colon-delimited format
- [ ] **M-SEC-3**: Key rotation procedure documented
- [ ] **L-SEC-1**: Model SHA256 hash verified against known-good value
- [ ] **L-SEC-2**: Prometheus alert fires on Redis authentication failures
- [ ] **L-SEC-3**: Signing key minimum length (32 chars) validated at startup

---

## Security Review Sign-off

| Area | Status | Blocking? |
|------|--------|-----------|
| Model Integrity (torch.load, HMAC) | ⚠️ Partial | **YES** — HMAC bypass must be fixed |
| Authentication (Redis, IEC) | ❌ Incomplete | **YES** — Auth + TLS required |
| Secrets Management | ⚠️ Weak | **YES** — Fail-fast + secrets manager |
| Network Security | ❌ Missing | **YES** — TLS for Redis |
| Input Validation | ✅ Adequate | No |
| Secrets Rotation | ❌ Missing | No (short-term) |

**Verdict:** **NOT READY FOR PRODUCTION** until C-SEC-1, C-SEC-2, and H-SEC-2 are addressed. Estimated fix effort: 2-4 hours.

**Next Review:** After REC-1 through REC-3 are implemented.

---

*Reviewer: Security Engineer*
*Reviewed Plan: CA-DQStream + MemStream Hybrid v4*
*Date: 2026-05-12*
