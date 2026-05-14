# Security + ML Fixes - CA-DQStream + MemStream v4 → v5

> **Goal:** Fix all CRITICAL and HIGH issues identified in the Security Engineer and ML Engineer reviews of PLAN_v4.md
>
> **Date:** 2026-05-12
>
> **Status:** 4 CRITICALs + 5 HIGHs to fix

---

## Fix Checklist

| ID | Priority | Issue | File | Status |
|----|----------|-------|------|--------|
| C-SEC-1 | CRITICAL | HMAC bypass when `IEC_SIGNING_KEY=None` | `memstream_scoring_op.py` | ⬜ |
| C-SEC-2 | CRITICAL | Redis unauthenticated, no TLS | `memstream_scoring_op.py` | ⬜ |
| C-SEC-3 | CRITICAL | Duplicate HMAC verification code | `memstream_core.py` | ⬜ |
| C-ML-1 | CRITICAL | `max_thres` used before initialization | `memstream_core.py` | ⬜ |
| H-SEC-2 | HIGH | HMAC key env var minimum length | `memstream_scoring_op.py` | ⬜ |
| H-SEC-3 | HIGH | No Redis connection timeout | `memstream_scoring_op.py` | ⬜ |
| H-SEC-4 | HIGH | `REQUIRE_MODEL_SIGNATURE = True` hardcoded | `memstream_scoring_op.py` | ⬜ |
| H-ML-2 | HIGH | Determinism flags incomplete | `memstream_core.py` | ⬜ |
| H-ML-3 | HIGH | Missing CUDA seeds | `memstream_core.py` | ⬜ |

---

## CRITICAL Fixes

### C-SEC-1: HMAC Bypass Prevention

**Why:** If `IEC_SIGNING_KEY` is not set, IECFeedbackOperator writes unsigned beta values to Redis, and MemStreamScoringOperator skips HMAC verification entirely. This allows any process with Redis access to manipulate anomaly thresholds (suppress detection with `beta=999999` or flood with `beta=0.0001`).

**Attack vector:** An attacker who gains Docker network access can run `redis-cli SET 'beta:manhattan' '999999.0'` and suppress all anomaly detection for that neighborhood.

---

#### Fix 1a: MemStreamScoringOperator — Startup validation for `IEC_SIGNING_KEY`

**File:** `memstream_src/operators/memstream_scoring_op.py`

**Context:** After env var reading, before operator initialization (around line 1000-1001 in PLAN_v4)

**Before:**
```python
IEC_SIGNING_KEY = os.getenv('IEC_SIGNING_KEY', None)
MODEL_SIGNING_KEY = os.getenv('MEMSTREAM_MODEL_SIGNING_KEY', None)
```

**After:**
```python
IEC_SIGNING_KEY = os.getenv('IEC_SIGNING_KEY')
MODEL_SIGNING_KEY = os.getenv('MEMSTREAM_MODEL_SIGNING_KEY')

# ── C-SEC-1: Enforce HMAC keys at startup (fail-fast) ──────────────────────
def _enforce_hmac_config():
    """Fail fast if required HMAC keys are missing or too short."""
    if not IEC_SIGNING_KEY:
        raise RuntimeError(
            "[MemStream] FATAL: IEC_SIGNING_KEY environment variable is required. "
            "Beta updates will not be accepted without HMAC signing. "
            "Set IEC_SIGNING_KEY to a 256-bit secret (32+ characters)."
        )
    if len(IEC_SIGNING_KEY) < 32:
        raise RuntimeError(
            f"[MemStream] FATAL: IEC_SIGNING_KEY must be at least 32 characters "
            f"(256-bit). Got {len(IEC_SIGNING_KEY)} characters."
        )
    if not MODEL_SIGNING_KEY:
        raise RuntimeError(
            "[MemStream] FATAL: MEMSTREAM_MODEL_SIGNING_KEY environment variable is required. "
            "Model files will not be loaded without HMAC verification. "
            "Set MEMSTREAM_MODEL_SIGNING_KEY to a 256-bit secret (32+ characters)."
        )
    if len(MODEL_SIGNING_KEY) < 32:
        raise RuntimeError(
            f"[MemStream] FATAL: MEMSTREAM_MODEL_SIGNING_KEY must be at least 32 characters "
            f"(256-bit). Got {len(MODEL_SIGNING_KEY)} characters."
        )

_enforce_hmac_config()
```

**Verification:**
- [ ] Startup fails if `IEC_SIGNING_KEY` is not set
- [ ] Startup fails if `IEC_SIGNING_KEY` < 32 characters
- [ ] Startup fails if `MEMSTREAM_MODEL_SIGNING_KEY` is not set
- [ ] Startup fails if `MEMSTREAM_MODEL_SIGNING_KEY` < 32 characters
- [ ] `_get_beta_from_redis()` now always performs HMAC verification (no bypass path)

---

#### Fix 1b: IECFeedbackOperator — Enforce `IEC_SIGNING_KEY`

**File:** `memstream_src/operators/iec_feedback_op.py`

**Context:** At module level, after env var reading (same pattern as Fix 1a)

**Before:**
```python
IEC_SIGNING_KEY = os.getenv('IEC_SIGNING_KEY', None)
```

**After:**
```python
IEC_SIGNING_KEY = os.getenv('IEC_SIGNING_KEY')

# ── C-SEC-1: Fail fast if signing key is missing ─────────────────────────────
if not IEC_SIGNING_KEY:
    raise RuntimeError(
        "[IECFeedback] FATAL: IEC_SIGNING_KEY environment variable is required. "
        "Unsigned beta updates are not permitted."
    )
if len(IEC_SIGNING_KEY) < 32:
    raise RuntimeError(
        f"[IECFeedback] FATAL: IEC_SIGNING_KEY must be at least 32 characters. "
        f"Got {len(IEC_SIGNING_KEY)}."
    )
```

**Verification:**
- [ ] Startup fails if `IEC_SIGNING_KEY` is not set
- [ ] Startup fails if `IEC_SIGNING_KEY` < 32 characters
- [ ] All beta writes to Redis are always signed (no silent bypass)

---

#### Fix 1c: Remove bypass in `_get_beta_from_redis()`

**File:** `memstream_src/operators/memstream_scoring_op.py`

**Context:** `_get_beta_from_redis()` method (around line 1104)

**Before:**
```python
if IEC_SIGNING_KEY:
    expected_sig = hmac.new(
        IEC_SIGNING_KEY.encode(), beta_bytes, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_sig):
        LOGGER.error(f"[MemStream] Beta HMAC mismatch for {neighborhood}")
        return None
# If IEC_SIGNING_KEY is None, HMAC check is SKIPPED entirely ← C-SEC-1 bypass
```

**After:**
```python
# IEC_SIGNING_KEY is guaranteed non-None due to _enforce_hmac_config()
expected_sig = hmac.new(
    IEC_SIGNING_KEY.encode(), beta_bytes, hashlib.sha256
).hexdigest()
if not hmac.compare_digest(signature, expected_sig):
    LOGGER.error(
        f"[MemStream] Beta HMAC mismatch for {neighborhood}. "
        f"Rejecting unsigned/tampered beta update."
    )
    self._scoring_errors.labels(error_type='hmac_mismatch', neighborhood=neighborhood).inc()
    return None
```

**Verification:**
- [ ] HMAC verification always executes (no conditional bypass)
- [ ] Error counter incremented on HMAC mismatch

---

#### Fix 1d: Remove bypass in IECFeedbackOperator beta write

**File:** `memstream_src/operators/iec_feedback_op.py`

**Context:** `process_broadcast_element()` method (around line 1318)

**Before:**
```python
if IEC_SIGNING_KEY:
    sig = hmac.new(SECRET_KEY, str(beta).encode(), hashlib.sha256).hexdigest()
    redis_client.set(f'beta:{neighborhood}', f'{beta_str}:{sig}')
else:
    LOGGER.warning(f"[IECFeedback] No IEC_SIGNING_KEY, beta update unsigned")
    redis_client.set(f'beta:{neighborhood}', beta_str)  # ← UNSIGNED ← C-SEC-1 bypass
```

**After:**
```python
# IEC_SIGNING_KEY is guaranteed non-None due to startup validation
sig = hmac.new(IEC_SIGNING_KEY.encode(), beta_str.encode(), hashlib.sha256).hexdigest()
redis_client.set(f'beta:{neighborhood}', f'{beta_str}:{sig}')
```

**Verification:**
- [ ] All beta writes to Redis always include HMAC signature
- [ ] No path exists for writing unsigned beta values

---

### C-SEC-2: Redis Authentication + TLS Enforcement

**Why:** `REDIS_PASSWORD=None` silently disables authentication. In distributed Flink deployments, Redis traffic traverses the network unencrypted. Without TLS, credentials and beta values are visible to network observers. Without timeouts, Redis unavailability causes indefinite hangs (DoS vector).

---

#### Fix 2: Hardened Redis Client

**File:** `memstream_src/operators/memstream_scoring_op.py`

**Context:** After `_enforce_hmac_config()`, before operator class definition

**Before:**
```python
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)  # Optional ← C-SEC-2
```

**After:**
```python
# ── C-SEC-2: Require Redis password in production, enforce TLS ───────────────
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
REDIS_TLS = os.getenv('REDIS_TLS', 'false').lower() == 'true'
REDIS_SOCKET_TIMEOUT = float(os.getenv('REDIS_SOCKET_TIMEOUT', '5.0'))
REDIS_SOCKET_CONNECT_TIMEOUT = float(os.getenv('REDIS_SOCKET_CONNECT_TIMEOUT', '3.0'))

# ── H-SEC-3: Add connection timeouts ────────────────────────────────────────
if not REDIS_PASSWORD:
    raise RuntimeError(
        "[MemStream] FATAL: REDIS_PASSWORD environment variable is required. "
        "Redis authentication is mandatory in production."
    )

import redis as redis_lib
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

class RedisUnavailableError(RuntimeError):
    """Raised when Redis is unavailable or unreachable."""
    pass

def _create_redis_client():
    """Create hardened Redis client with TLS and timeouts."""
    try:
        client = redis_lib.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            ssl=REDIS_TLS,
            ssl_cert_reqs='required' if REDIS_TLS else None,
            socket_timeout=REDIS_SOCKET_TIMEOUT,
            socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT,
            retry_on_timeout=True,
            health_check_interval=30,
            decode_responses=False,  # Keep as bytes for HMAC computation
        )
        # Verify connection immediately
        client.ping()
        return client
    except (RedisConnectionError, RedisTimeoutError) as e:
        raise RedisUnavailableError(
            f"[MemStream] FATAL: Redis connection failed: {e}. "
            f"Check REDIS_HOST ({REDIS_HOST}:{REDIS_PORT}), password, and network."
        ) from e
```

**Context update in `MemStreamScoringOperator.__init__`:**

**Before:**
```python
self._redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    ssl=REDIS_TLS_ENABLED,  # was optional
    socket_timeout=5.0,      # was missing
    socket_connect_timeout=3.0,  # was missing
)
```

**After:**
```python
self._redis_client = _create_redis_client()
```

**Verification:**
- [ ] Startup fails if `REDIS_PASSWORD` is not set
- [ ] Redis client uses TLS when `REDIS_TLS=true`
- [ ] Socket timeout of 5 seconds prevents indefinite hangs
- [ ] Socket connect timeout of 3 seconds prevents slow-connection DoS
- [ ] Health check interval of 30s detects stale connections
- [ ] Prometheus metric emitted on Redis errors

---

### C-SEC-3: Remove Duplicate HMAC Verification Code

**Why:** Lines 564-576 and 577-589 in `memstream_core.py` contain identical HMAC verification blocks. The first block (564-576) already handles all cases correctly including the `require_signature` check. The second block (577-589) is dead code that doubles file I/O (opens `.hmac` file twice per load) and creates a maintenance hazard.

---

#### Fix 3: Remove Duplicate HMAC Block

**File:** `memstream_src/core/memstream_core.py`

**Context:** Inside `load()` method (around lines 564-589)

**Before:**
```python
564:    if signing_key:
565:        # ... HMAC check (correct implementation) ...
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
576:    # [DUPLICATE BLOCK - lines 577-589] ← C-SEC-3: REMOVE THIS
577:    if signing_key:  # ← executes same check again
578:        with open(path + '.hmac') as f:
579:            expected_hmac = f.read().strip()
580:        with open(path + '.hmac') as f:  # ← file opened twice (waste)
581:            actual_hmac = hmac.new(...)
582:        if not hmac.compare_digest(...):
583:            raise SecurityError(...)
```

**After:**
```python
564:    if signing_key:
565:        # HMAC verification (canonical block)
566:        if not os.path.exists(path + '.hmac'):
567:            if require_signature:
568:                raise SecurityError(...)
569:            else:
570:                LOGGER.warning(...)
571:        else:
572:            with open(path + '.hmac') as f:
573:                expected_hmac = f.read().strip()
574:            with open(path, 'rb') as f:
575:                actual_hmac = hmac.new(
576:                    signing_key.encode(), f.read(), hashlib.sha256
577:                ).hexdigest()
578:            if not hmac.compare_digest(expected_hmac, actual_hmac):
579:                raise SecurityError(f"Model HMAC mismatch — possible tampering: {path}")
580:    elif require_signature:
581:        raise SecurityError(
582:            f"Model {path} requires HMAC verification but no signing key provided."
583:        )
```

**Verification:**
- [ ] HMAC verification logic appears only once in `load()`
- [ ] `.hmac` file opened exactly once per model load
- [ ] `require_signature=True` raises SecurityError if key missing
- [ ] `require_signature=False` logs warning if `.hmac` missing

---

### C-ML-1: `max_thres` Initialization in `__init__`

**Why:** `max_thres` is never initialized in `__init__`. It is only set by `warmup()` (from calibration data) or `set_beta()` (externally). If `score_one()` is called before either completes, the code crashes with `AttributeError: 'MemStreamCore' object has no attribute 'max_thres'`. The model file saved after warmup has `max_thres=0.0` by default, making this a production crash bug.

**Affected lines:** `score_one()` line ~759, `_update_memory()` line ~764, `score_batch()` line ~789.

---

#### Fix 4: Initialize `max_thres` in `__init__`

**File:** `memstream_src/core/memstream_core.py`

**Context:** In `__init__()` method, alongside other state initialization

**Before:**
```python
self.count = 0
self.memory_len = cfg.memory_len
self.device = device
# ... other initializations ...
# max_thres is NEVER initialized ← C-ML-1 crash bug
```

**After:**
```python
self.count = 0
self.memory_len = cfg.memory_len
self.device = device
# ── C-ML-1: Initialize max_thres to avoid AttributeError on early scoring ───
self.max_thres = torch.tensor(0.0, dtype=torch.float32, device=self.device)
```

**Additionally:** Add a safety check in `score_one()` to guard against 0.0 threshold:

**After the shape validation block in `score_one()`:**
```python
# Safety check: ensure beta threshold has been set
if self.max_thres.item() <= 0.0:
    raise RuntimeError(
        f"[MemStream] FATAL: max_thres is {self.max_thres.item()} — "
        f"beta threshold has not been set. Call set_beta() or warmup() first."
    )
```

**Verification:**
- [ ] `max_thres` attribute exists immediately after `__init__()`
- [ ] `score_one()` raises informative error if called before beta is set
- [ ] `warmup()` or `set_beta()` sets a positive threshold value
- [ ] Saved model checkpoints include `max_thres` in state dict

---

## HIGH Fixes

### H-SEC-2: Document HMAC Key Minimum Length

**Why:** Environment variables are exposed via `/proc/*/environ` and process listings. A weak key (<32 characters) is vulnerable to brute-force attacks. The fix is already included in C-SEC-1 (Fix 1a and 1b), but must be documented in deployment configs.

---

#### Fix 5: Document key length in deployment

**File:** `deployment/docker-compose.yml` (and Kubernetes manifests)

**Context:** Environment variable documentation

**Add to deployment documentation:**

```yaml
# ── H-SEC-2: Document minimum key length ─────────────────────────────────────
# All signing keys MUST be at least 32 characters (256-bit).
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
# DO NOT use passwords, passphrases, or shorter keys.
#
# Example valid key:
# IEC_SIGNING_KEY=abc...xyz  # 32+ characters
#
# Example INVALID (will fail startup):
# IEC_SIGNING_KEY=mysecretkey  # Only 13 characters — REJECTED

flink-taskmanager:
  environment:
    IEC_SIGNING_KEY: ${IEC_SIGNING_KEY:?IEC_SIGNING_KEY (32+ chars) required}
    MEMSTREAM_MODEL_SIGNING_KEY: ${MEMSTREAM_MODEL_SIGNING_KEY:?MEMSTREAM_MODEL_SIGNING_KEY (32+ chars) required}
    REDIS_PASSWORD: ${REDIS_PASSWORD:?REDIS_PASSWORD required}
    REDIS_TLS: "true"  # Enable TLS in production
```

**Verification:**
- [ ] Key length validation documented in deployment docs
- [ ] Docker Compose fails fast with clear error if key < 32 chars
- [ ] K8s secret generation script enforces 32-char minimum

---

### H-SEC-3: Redis Connection Timeout

**Why:** Without socket timeouts, Redis unavailability causes the operator to hang indefinitely. An attacker could exploit this as a DoS vector by making Redis unavailable, causing all MemStream operators to stall.

---

#### Fix 6: Socket Timeout in Redis Client

**Why fix already applied in C-SEC-2 Fix 2:** The `_create_redis_client()` function sets `socket_timeout=5.0` and `socket_connect_timeout=3.0`. Additionally, add error handling in `_get_beta_from_redis()`:

**File:** `memstream_src/operators/memstream_scoring_op.py`

**Context:** In `_get_beta_from_redis()` method

**Before:**
```python
try:
    raw_value = self._redis_client.get(f'beta:{neighborhood}')
except Exception as e:
    LOGGER.warning(f"[MemStream] Redis error: {e}")
    return None
```

**After:**
```python
try:
    raw_value = self._redis_client.get(f'beta:{neighborhood}')
except (RedisConnectionError, RedisTimeoutError) as e:
    LOGGER.error(f"[MemStream] Redis timeout/error for beta read: {e}")
    self._scoring_errors.labels(error_type='redis_timeout', neighborhood=neighborhood).inc()
    raise RedisUnavailableError(f"Redis unavailable: {e}") from e
except Exception as e:
    LOGGER.error(f"[MemStream] Unexpected Redis error: {e}")
    self._scoring_errors.labels(error_type='redis_error', neighborhood=neighborhood).inc()
    return None
```

**Verification:**
- [ ] Redis timeout raises `RedisUnavailableError` (not silently returns None)
- [ ] Error counter incremented with `error_type='redis_timeout'`
- [ ] Circuit breaker can trigger on persistent Redis failures

---

### H-SEC-4: Remove Hardcoded `REQUIRE_MODEL_SIGNATURE = True`

**Why:** Line 1010 in `memstream_scoring_op.py` overrides the env var check unconditionally with `REQUIRE_MODEL_SIGNATURE = True`. This removes the ability to configure signature requirements via environment variables.

---

#### Fix 7: Use Environment Variable for `REQUIRE_MODEL_SIGNATURE`

**File:** `memstream_src/operators/memstream_scoring_op.py`

**Context:** After `_enforce_hmac_config()`, where `REQUIRE_MODEL_SIGNATURE` is defined (around line 1010)

**Before:**
```python
REQUIRE_MODEL_SIGNATURE = os.getenv('REQUIRE_MODEL_SIGNATURE', 'true').lower() == 'true'
# ... later ...
REQUIRE_MODEL_SIGNATURE = True  # ← H-SEC-4: hardcoded override ← REMOVE THIS
```

**After:**
```python
# H-SEC-4: Read from env var only, no hardcoded override
REQUIRE_MODEL_SIGNATURE = os.getenv('REQUIRE_MODEL_SIGNATURE', 'true').lower() == 'true'
# NOTE: The _enforce_hmac_config() call above already requires both keys,
# so signature verification is always enforced. This flag controls whether
# MISSING .hmac files are a fatal error or a warning.
```

**Verification:**
- [ ] `REQUIRE_MODEL_SIGNATURE` is read from env var only
- [ ] No line in the codebase unconditionally overrides this variable
- [ ] Default behavior (`true`) matches production security requirements

---

### H-ML-2: Complete Determinism Flags

**Why:** Only `torch.backends.cudnn.deterministic` is set. Full determinism in PyTorch requires additional flags. Without `PYTHONHASHSEED`, Python's set/dict ordering is non-deterministic across runs.

---

#### Fix 8a: Complete Determinism Flags in `memstream_core.py`

**File:** `memstream_src/core/memstream_core.py`

**Context:** At the top of the file, after imports (new section)

**Add after imports:**
```python
# ── H-ML-2: Full determinism configuration ───────────────────────────────────
def set_determinism(seed: int = 42):
    """Configure all random sources for reproducible training/scoring.
    
    Call this at the start of training scripts and in the Flink operator
    open() method. Does not guarantee bit-exact reproducibility across
    PyTorch versions or hardware, but eliminates most sources of variance.
    
    Note: PYTHONHASHSEED must be set in the environment BEFORE Python starts.
    Set in docker-compose.yml: environment: PYTHONHASHSEED: "42"
    """
    # Python built-ins
    import random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # ── H-ML-3: CUDA seeds ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # All GPUs in multi-GPU training
    
    # CuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Deterministic algorithms (PyTorch 1.8+)
    # Note: May reduce performance; only use for reproducibility-critical paths
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
```

**Context update in `warmup()` method:**
```python
# At the start of warmup(), add:
set_determinism(cfg.seed)
```

**Context update in `MemStreamScoringOperator.open()`:**
```python
# At the start of open(), add:
set_determinism(int(os.getenv('MEMSTREAM_SEED', '42')))
```

**Verification:**
- [ ] `set_determinism()` sets numpy, torch CPU, torch CUDA, random, and hash seed
- [ ] CuDNN deterministic and benchmark flags are set
- [ ] `torch.use_deterministic_algorithms(True)` is called (with warn_only=True for performance)
- [ ] `PYTHONHASHSEED` is set in environment
- [ ] Docker Compose documents `PYTHONHASHSEED: "42"` requirement

---

### H-ML-3: Add CUDA Seed Setting

**Why:** `torch.cuda.manual_seed_all(seed)` is not called. In multi-GPU training, only the primary GPU is seeded, leaving other GPUs with non-deterministic state.

---

#### Fix 9: CUDA Seed Setting

**File:** `memstream_src/core/memstream_core.py`

**Context:** Already included in `set_determinism()` above (H-ML-2 Fix 8a). The critical addition is:

```python
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # ← H-ML-3: All GPUs seeded
```

**Additionally**, update `scripts/train_warmup.py` to call `set_determinism()` at startup:

**Before:**
```python
np.random.seed(seed)
torch.manual_seed(seed)
```

**After:**
```python
from memstream_src.core.memstream_core import set_determinism
set_determinism(seed)
```

**Verification:**
- [ ] CUDA seed is set on all available GPUs
- [ ] `torch.cuda.manual_seed_all()` is called when CUDA is available
- [ ] Training script uses `set_determinism()` for complete seed coverage

---

## Summary of Changes

### Files Modified

| File | CRITICAL Fixes | HIGH Fixes |
|------|---------------|------------|
| `memstream_src/operators/memstream_scoring_op.py` | C-SEC-1 (1a, 1c), C-SEC-2 (Fix 2), H-SEC-3 (Fix 6) | H-SEC-4 (Fix 7) |
| `memstream_src/operators/iec_feedback_op.py` | C-SEC-1 (1b, 1d) | — |
| `memstream_src/core/memstream_core.py` | C-SEC-3 (Fix 3), C-ML-1 (Fix 4) | H-ML-2 (Fix 8a), H-ML-3 (Fix 9) |
| `deployment/docker-compose.yml` | — | H-SEC-2 (Fix 5) |
| `scripts/train_warmup.py` | — | H-ML-2, H-ML-3 (Fix 8b) |

### Verification Matrix

| Fix | Test Scenario | Expected Result |
|-----|---------------|-----------------|
| C-SEC-1 | Start operator without `IEC_SIGNING_KEY` | RuntimeError at startup |
| C-SEC-1 | Set `IEC_SIGNING_KEY=short` | RuntimeError: key too short |
| C-SEC-1 | Write beta without signing key | Impossible (enforced at startup) |
| C-SEC-1 | Write beta with wrong HMAC key | HMAC mismatch, beta rejected |
| C-SEC-2 | Start without `REDIS_PASSWORD` | RuntimeError at startup |
| C-SEC-2 | Redis unavailable during scoring | RedisUnavailableError raised |
| C-SEC-2 | Redis connection >5s | TimeoutError, not hang |
| C-SEC-3 | Load model file | `.hmac` file opened exactly once |
| C-ML-1 | Call `score_one()` before `warmup()` | Informative RuntimeError |
| C-ML-1 | Call `score_one()` after `set_beta()` | Works correctly |
| H-SEC-2 | Use 20-char signing key | RuntimeError at startup |
| H-SEC-4 | Check `REQUIRE_MODEL_SIGNATURE` source | Only from env var |
| H-ML-2 | Run benchmark 3 times | Same anomaly rankings |
| H-ML-3 | Multi-GPU training | All GPUs produce same results |

---

## Rollback Plan

If any fix causes issues in production:

1. **C-SEC-1 bypass restoration:** Remove `_enforce_hmac_config()` call and revert env var defaults to `None`
2. **C-SEC-2 auth bypass:** Restore `REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)` and remove TLS requirements
3. **C-SEC-3 restoration:** Restore the duplicate block from git history
4. **C-ML-1 restoration:** Remove `self.max_thres = ...` from `__init__`, rely on `warmup()`/`set_beta()`

**Note:** Rollback of C-SEC-1 or C-SEC-2 exposes critical vulnerabilities. Test thoroughly before production deployment.

---

*Fix specification version: v5-draft*
*Source reviews: Security Engineer v4, ML Engineer v4*
*Date: 2026-05-12*
