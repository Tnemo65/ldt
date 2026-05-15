# MemStream Security Documentation

**Phase**: 5A REC-10
**Audience**: DevOps, Security, Platform Engineering
**Last Updated**: 2026-05-14

---

## Overview

This document covers security-critical configuration for the MemStream/CA-DQStream deployment, including key management, rotation procedures, and emergency revocation.

---

## Security-Critical Environment Variables

| Variable | Purpose | Rotation | Storage |
|----------|---------|----------|---------|
| `MEMSTREAM_MODEL_SIGNING_KEY` | HMAC-SHA256 signing key for model checkpoint verification | Every 90 days or after suspected compromise | Vault/Secret Manager, never in source control |
| `IEC_SIGNING_KEY` | HMAC-SHA256 signing key for IEC beta updates | Every 90 days or after suspected compromise | Vault/Secret Manager, never in source control |
| `INTERNAL_API_KEY` | Bearer token for ML service `/predict` and `/retrain` endpoints | Every 90 days | Vault/Secret Manager |
| `METRICS_API_KEY` | Bearer token for Prometheus `/metrics` endpoint (falls back to `INTERNAL_API_KEY`) | Every 90 days | Vault/Secret Manager |
| `REDIS_PASSWORD` | Redis authentication password | Every 180 days or after suspected compromise | Vault/Secret Manager |
| `MINIO_SECRET_KEY` | MinIO/S3 secret access key | Every 180 days | Vault/Secret Manager |
| `MINIO_ACCESS_KEY` | MinIO/S3 access key | Every 180 days | Vault/Secret Manager |

### Minimum Key Length Requirements

| Key | Minimum Length | Notes |
|-----|----------------|-------|
| `MEMSTREAM_MODEL_SIGNING_KEY` | 32 characters (256-bit) | Hard-block: < 32 chars causes startup failure |
| `IEC_SIGNING_KEY` | 32 characters (256-bit) | Hard-block: < 32 chars causes startup failure |
| `INTERNAL_API_KEY` | 32 characters | 256-bit entropy recommended |
| `METRICS_API_KEY` | 32 characters | Falls back to `INTERNAL_API_KEY` if unset |
| Redis password | 24 characters | Alphanumeric + special characters |
| MinIO keys | 16 characters | Access key; secret key should be 32+ chars |

---

## Key Generation

### HMAC Signing Keys

Generate a new HMAC signing key using OpenSSL:

```bash
# Generate 256-bit (32-byte) HMAC key
openssl rand -hex 32

# Example output:
# a1b2c3d4e5f6... (64 hex characters = 32 bytes)
```

### API Keys (Bearer Tokens)

```bash
# Generate API key for ML service endpoints
openssl rand -hex 32

# Generate metrics API key (can be same as INTERNAL_API_KEY or separate)
openssl rand -hex 32
```

### MinIO Credentials

```bash
# Access key: 16-128 alphanumeric characters
openssl rand -hex 8   # 16 characters

# Secret key: 32+ characters
openssl rand -hex 32  # 64 hex characters = 32 bytes
```

---

## Key Rotation Schedule

### Routine Rotation (90-day cycle)

| Key | Frequency | Procedure |
|-----|-----------|-----------|
| `MEMSTREAM_MODEL_SIGNING_KEY` | Every 90 days | Coordinated with model checkpoint rotation |
| `IEC_SIGNING_KEY` | Every 90 days | Requires IEC restart |
| `INTERNAL_API_KEY` / `METRICS_API_KEY` | Every 90 days | Rolling update across services |
| Redis password | Every 180 days | Redis restart required |
| MinIO credentials | Every 180 days | MinIO restart required |

### Pre-Deployment Rotation

1. Generate new keys offline (air-gapped machine or CI/CD pipeline secrets)
2. Store new keys in Vault/Secret Manager
3. Update `.env` or secrets injection mechanism
4. Deploy new configuration
5. Verify all services start correctly with new keys
6. Delete old keys from Vault after 24-hour verification window

---

## Emergency Revocation Procedure

### Scenario 1: Suspected HMAC Key Compromise

If `MEMSTREAM_MODEL_SIGNING_KEY` or `IEC_SIGNING_KEY` is suspected to be compromised:

1. **Immediate**: Rotate to new key in Vault/Secret Manager
2. **Deploy**: Update deployment with new key (rolling restart)
3. **Verify**: Confirm no HMAC verification failures in Prometheus metrics
4. **Checkpoint**: Re-sign all model checkpoints with new key
5. **Audit**: Review logs for unauthorized model loads in the past 90 days

```bash
# Re-sign model checkpoint with new key
python scripts/resign_checkpoint.py \
    --checkpoint models/memstream_checkpoint.pt \
    --signing-key-env MEMSTREAM_MODEL_SIGNING_KEY \
    --sha256-key-env SHA256_CONTENT_KEY  # New for REC-6
```

### Scenario 2: API Key Compromise

If `INTERNAL_API_KEY` is compromised:

1. **Immediate**: Generate new key, update in Vault
2. **Deploy**: Rolling restart of all services using the key
3. **Verify**: Confirm 401 responses for old key, 200 for new key
4. **Revoke**: Disable old key in Vault after verification

### Scenario 3: Redis/MinIO Credential Compromise

1. **Immediate**: Rotate password/key in Redis/MinIO
2. **Deploy**: Update all services with new credentials
3. **Verify**: Confirm all connections establish correctly
4. **Check**: Review Redis/MinIO access logs for unauthorized access

---

## Security Verification Checklist

After any key rotation, verify:

- [ ] All MemStream scoring operators start without HMAC failures
- [ ] ML service `/health` returns `status: healthy`
- [ ] `/metrics` endpoint requires `Authorization: Bearer <token>`
- [ ] `/predict` endpoint requires `Authorization: Bearer <token>`
- [ ] Prometheus shows no `MemStream_HMACVerificationFailures` increments
- [ ] Prometheus shows no `MemStream_ModelFileTooLarge` increments
- [ ] Redis connections established (check `ml_service_redis_connected = 1`)
- [ ] Circuit breaker state is `closed` (no `MemStream_CircuitBreakerTripped` alerts)

---

## Security Features (Phase 5A REC-5 through REC-10)

| REC | Feature | Status | Description |
|-----|---------|--------|-------------|
| REC-5 | Model file size limit | Implemented | 500MB max, `MemStream_ModelFileTooLarge` alert |
| REC-6 | SHA256 content verification | Implemented | SHA256 hash embedded in checkpoint, verified before HMAC |
| REC-7 | Metrics endpoint auth | Implemented | Bearer token required for `/metrics` |
| REC-8 | Operator identity in IEC | Implemented | `operator_id` in all IEC payloads |
| REC-9 | Circuit breaker alert | Verified | `MemStream_CircuitBreakerTripped` alert in Phase 3E |
| REC-10 | Key rotation docs | This document | Full rotation procedures |

---

## Checkpoint Signing (REC-6)

Model checkpoints are signed with a dual-verification scheme:

1. **SHA256** (integrity): Hash of checkpoint content stored in last 64 bytes
2. **HMAC** (authenticity): HMAC-SHA256 of checkpoint content stored in bytes 64-127

### Checkpoint Format

```
[checkpoint_content][hmac_signature (64 bytes)][sha256_hash (64 bytes)]
```

### Verification Flow (before `torch.load()`)

```
1. Check file size <= 500MB (REC-5)
2. Read all bytes
3. Extract SHA256 from last 64 bytes
4. Compute SHA256 of content
5. Compare with constant-time comparison
6. Extract HMAC from bytes 64-127
7. Compute HMAC with signing key
8. Compare with constant-time comparison
9. Only then call torch.load()
```

### Re-signing Checkpoints After Key Rotation

```python
import hashlib
import hmac

def resign_checkpoint(checkpoint_path: str, signing_key: str, output_path: str):
    """Re-sign a checkpoint with a new key."""
    # Read checkpoint content
    with open(checkpoint_path, 'rb') as f:
        content = f.read()
    
    # Remove old signatures if present
    # (check if file has embedded signatures)
    if len(content) > 128:
        content = content[:-128]
    
    # Compute SHA256 (REC-6)
    sha256_hash = hashlib.sha256(content).hexdigest()
    
    # Compute HMAC
    hmac_sig = hmac.new(
        signing_key.encode(),
        content,
        hashlib.sha256
    ).hexdigest()
    
    # Write new signed checkpoint
    with open(output_path, 'wb') as f:
        f.write(content)
        f.write(hmac_sig.encode())  # HMAC at bytes 64-127
        f.write(sha256_hash.encode())  # SHA256 at last 64 bytes
    
    print(f"Checkpoint re-signed: {output_path}")
    print(f"  Content size: {len(content)} bytes")
    print(f"  SHA256: {sha256_hash[:16]}...")
    print(f"  HMAC: {hmac_sig[:16]}...")
```

---

## Metrics Endpoint Authentication (REC-7)

The `/metrics` endpoint requires Bearer token authentication:

```bash
# Access metrics with token
curl -H "Authorization: Bearer <METRICS_API_KEY>" \
    http://localhost:8000/metrics

# Without token → 401 Unauthorized
curl http://localhost:8000/metrics
# {"detail": "Missing Authorization header"}
```

### Prometheus Scraping Configuration

Update `prometheus.yml` to include the authorization header:

```yaml
scrape_configs:
  - job_name: 'memstream-ml-service'
    metrics_path: '/metrics'
    bearer_token: '${METRICS_API_KEY}'
    static_configs:
      - targets: ['ml-service:8000']
```

---

## Audit Trail (REC-8)

All IEC payloads include an `operator_id` field for traceability:

```json
{
  "operator_id": "iec-flink-taskmanager-1-12345-a1b2c3d4",
  "drifts_detected": [...],
  "iec_strategy": "adjust_threshold",
  "iec_confidence": 0.82,
  "iec_timestamp": "2026-05-14T10:30:00.000Z"
}
```

The `operator_id` format: `iec-{hostname}-{pid}-{short_uuid}`

This allows correlating IEC decisions to specific Flink task instances.

---

## References

- Phase 5A Plan: `.cursor/plans/memstream_migration_plan_bcb6fcf4.plan.md`
- ML Service: `src/api/ml_service.py`
- IEC Operator: `src/operators/iec_operator.py`
- Circuit Breaker: `src/iec/circuit_breaker.py`
- Prometheus Alerts: `deployment/prometheus/alert-rules/cadqstream-alerts.yml`

---

**Classification**: Internal - DevOps/Security
**Owner**: Platform Engineering
**Review Frequency**: Quarterly
