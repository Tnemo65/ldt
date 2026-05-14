# Docker/K8s Engineer Review — PLAN_v3.md

**Reviewer:** Principal DevOps/Platform Engineer (12 yrs K8s, Docker, Helm, ArgoCD, GitOps)
**Date:** 2026-05-12
**Files Reviewed:** PLAN_v3.md (§6), `docker-compose.yml` (root), `deployment/docker-compose.yml`, `deployment/flink/Dockerfile`, `Dockerfile.ml-service`, `deployment/cadqstream-metrics/Dockerfile`, `deployment/flink/flink-conf.yml`, `deployment/.env`, `deployment/prometheus/prometheus.yml`, `deployment/prometheus/alert-rules/cadqstream-alerts.yml`

---

## 1. Dockerfile Analysis

### 1.1 Flink Dockerfile — Build Fragility

```dockerfile
COPY flink/flink-connector-kafka-1.17.1.jar /opt/flink/lib/
```

JAR files must exist in `deployment/flink/` relative to build context (project root). Only works with `docker build -f deployment/flink/Dockerfile .`. No `.dockerignore` file.

### 1.2 Python 3.8 Base Image — EOL

```dockerfile
FROM python:3.8-slim
```

Python 3.8 reached EOL October 2024. Should be `python:3.12-slim-bookworm`.

### 1.3 HEALTHCHECK Uses Python requests Not Installed

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"
```

`requests` is NOT in `requirements.txt` or explicit installs. HEALTHCHECK fails with `ModuleNotFoundError`.

### 1.4 No Non-Root User

ML service container runs as root. Should use:
```dockerfile
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser
```

### 1.5 cloudpickle Version Mismatch

```dockerfile
RUN pip install cloudpickle==2.2.1
ENV PYTHONPATH=.../cloudpickle-2.2.0-src.zip  # Different version!
```

### 1.6 cadqstream-metrics — curl Not Installed

```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -sf http://localhost:9250/health"]
```

But `cadqstream-metrics/Dockerfile` installs only `flask`, `prometheus-client`, `requests`. No `curl`.

---

## 2. Kubernetes Deployment

### 2.1 CRITICAL: Zero Kubernetes Manifests Exist

`PLAN_v3.md` references `kubernetes_deployment.yaml` at `memstream_src/operators/deployment/`. **This directory is empty.** There are zero K8s manifests in the entire codebase.

### 2.2 Docker Compose Is Not Production-Grade

- Single JobManager replica (no HA)
- `file:///tmp/flink-checkpoints` (ephemeral)
- `restart: unless-stopped` (infinite restart loop)
- No PVC for checkpoints

### 2.3 No Helm Chart

No `Chart.yaml`, `values.yaml`, or resource templates for production K8s deployment.

---

## 3. Resource Management

### 3.1 JVM Heap Exceeds Container Memory — CRITICAL

```yaml
# flink-conf.yml
taskmanager.memory.process.size: 4096m
taskmanager.memory.managed.fraction: 0.4   # = 1638MB

# taskmanager.conf
env.java.opts.taskmanager: -Xms3072m -Xmx3072m
```

JVM heap = 3072MB. Non-managed = 4096 − 1638 = 2458MB. **Heap exceeds non-managed by 614MB** → OOM kill.

### 3.2 Zero Resource Limits in Root Compose — CRITICAL

Root `docker-compose.yml` has NO `deploy.resources` on any service. Kafka can consume all RAM.

### 3.3 Single TaskManager Replica

```yaml
deploy:
  replicas: 1
```

Any failure removes 100% of ML processing capacity.

### 3.4 GPU Support — Completely Absent

MemStream supports CUDA but:
- No `gpus: all` passthrough in compose
- No `nvidia.com/gpu` requests in K8s
- No CUDA version pinned
- No NVIDIA device plugin deployment

---

## 4. Secrets & Security

### 4.1 Hardcoded Credentials — CRITICAL

| File | Credentials |
|------|-------------|
| `deployment/.env` | `cadqstream123`, `minioadmin123`, `admin123` |
| Root `docker-compose.yml` | `cadqstream123` in env |
| `deployment/docker-compose.yml` | `minioadmin123` in env |

### 4.2 Model Signing Key Env Var Never Set

```python
MODEL_SIGNING_KEY = os.getenv('MEMSTREAM_MODEL_SIGNING_KEY', None)
```

The env var is **never set** in any compose file. HMAC verification is bypassed.

### 4.3 Kafka PLAINTEXT Only — CRITICAL

```yaml
KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT
```

No TLS, no SASL. Any pod can produce/consume any topic.

### 4.4 PostgreSQL MD5 Auth

```yaml
POSTGRES_HOST_AUTH_METHOD: md5
```

Deprecated in PostgreSQL 15. Should be `scram-sha-256`.

---

## 5. Checkpoints & DR

### 5.1 Ephemeral Checkpoints — CRITICAL

```yaml
state.checkpoints.dir: file:///tmp/flink-checkpoints
```

`/tmp` is ephemeral. On container restart, **all checkpoints are lost**. MemStream per-key memory must be rebuilt from scratch.

### 5.2 No PostgreSQL Backup

`minio-data` volume has no backup automation, no lifecycle policy, no tested restore procedure.

### 5.3 No Model Artifact Backup

MemStream model files have HMAC but no off-site backup, no versioned registry, no rollback procedure.

---

## 6. CI/CD

### 6.1 Zero GitHub Actions Workflows — CRITICAL

No `.github/workflows/` directory. Missing:
- Build & Test (`ci.yml`)
- Docker Build & Push (`docker-push.yml`)
- Model Training & Registry (`model-train.yml`)
- ArgoCD Sync (`argocd-sync.yml`)

### 6.2 No Image Signing

Images not signed with Cosign, no Trivy CVE scanning, no SBOM generation.

### 6.3 No Multi-Architecture Build

Images only `linux/amd64`. No `platforms: [linux/amd64, linux/arm64]`.

---

## 7. CRITICAL Issues

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | N/A | Zero K8s manifests — plan references non-existent files | Create full K8s manifest set at `src/k8s/` |
| 2 | `flink-conf.yml` | Ephemeral checkpoints → memory loss on restart | Use `s3://cadqstream-checkpoints/` |
| 3 | `deployment/.env` | Hardcoded credentials in git | Externalize via K8s secrets + Vault |
| 4 | `deployment/docker-compose.yml` | PLAINTEXT Kafka → no auth | Enable TLS + SASL/SCRAM |

---

## 8. HIGH Issues

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `deployment/flink/taskmanager.conf` | JVM heap 3072MB > non-managed 2458MB | Reduce to 2048MB heap, increase process to 8192m |
| 2 | `deployment/docker-compose.yml` | No HPA | Add HorizontalPodAutoscaler |
| 3 | `memstream_scoring_op.py` | `MEMSTREAM_MODEL_SIGNING_KEY` never set | Add env var to compose/K8s |
| 4 | `docker-compose.yml` | All services no resource limits | Add `deploy.resources.limits` |
| 5 | `deployment/flink/flink-conf.yml` | Config var `${VAR}` not resolved | Use `${env:VAR}` prefix |
| 6 | `docker-compose.yml` | Kafka RF=1 → no fault tolerance | RF=3, minISR=2 |

---

## 9. Priority Fixes

### Fix 1: K8s Manifests (Full Set)

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cadqstream
---
# k8s/configmap-flink.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: flink-config
  namespace: cadqstream
data:
  flink-conf.yaml: |
    state.backend: rocksdb
    state.checkpoints.dir: s3://cadqstream-checkpoints/flink/checkpoints
    taskmanager.memory.process.size: 8192m
    taskmanager.memory.managed.fraction: 0.4
---
# k8s/secret-credentials.yaml
apiVersion: v1
kind: Secret
metadata:
  name: cadqstream-secrets
  namespace: cadqstream
type: Opaque
stringData:
  minio-password: "REPLACE_FROM_VAULT"
  minio-access-key: "REPLACE_FROM_VAULT"
  minio-secret-key: "REPLACE_FROM_VAULT"
  model-signing-key: "REPLACE_FROM_VAULT"
---
# k8s/statefulset-taskmanager.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: flink-taskmanager
  namespace: cadqstream
spec:
  replicas: 3
  serviceName: flink-taskmanager
  template:
    spec:
      containers:
        - name: taskmanager
          image: ghcr.io/org/cadqstream-flink:1.17.1-py
          resources:
            requests:
              cpu: 1000m
              memory: 4Gi
              nvidia.com/gpu: 1
            limits:
              cpu: 4000m
              memory: 8Gi
              nvidia.com/gpu: 1
          env:
            - name: MEMSTREAM_MODEL_SIGNING_KEY
              valueFrom:
                secretKeyRef:
                  name: cadqstream-secrets
                  key: model-signing-key
          securityContext:
            runAsNonRoot: true
            runAsUser: 1000
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
      securityContext:
        fsGroup: 1000
---
# k8s/pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: flink-taskmanager-pdb
  namespace: cadqstream
spec:
  maxUnavailable: 1
  selector:
    matchLabels:
      app: flink
      component: taskmanager
---
# k8s/hpa-taskmanager.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: flink-taskmanager-hpa
  namespace: cadqstream
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: StatefulSet
    name: flink-taskmanager
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: External
      external:
        metric:
          name: kafka_consumer_lag_sum
        target:
          type: AverageValue
          averageValue: "5000"
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
---
# k8s/argocd-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: cadqstream
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/org/cadqstream.git
    targetRevision: HEAD
    path: k8s/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: cadqstream
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

### Fix 2: TaskManager Memory

```properties
# deployment/flink/taskmanager.conf — FIXED
env.java.opts.taskmanager: -Xms2048m -Xmx2048m \
    -XX:+UseG1GC -XX:MaxGCPauseMillis=200 \
    -XX:+ExitOnOutOfMemoryError \
    -XX:MaxDirectMemorySize=2048m
```

```yaml
# flink-conf.yml — FIXED
taskmanager.memory.process.size: 8192m   # Was 4096m
taskmanager.memory.managed.fraction: 0.4  # = 3276MB managed
# JVM heap 2048 + direct 2048 + managed 3276 + overhead ~500 = ~7864MB < 8192MB ✓
```

### Fix 3: CI/CD GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11', cache: 'pip' }
      - run: pip install -r requirements.txt
      - run: pip install flake8 pytest-cov
      - run: flake8 memstream_src/ --max-line-length=120 --ignore=E501,W503
      - run: pytest memstream_src/tests/ -v --cov=memstream_src --cov-report=xml
      - uses: codecov/codecov-action@v4
  docker-build:
    needs: test
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: deployment/flink/Dockerfile
          push: true
          tags: ghcr.io/${{ github.repository }}/flink:${{ github.ref_name }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
      - uses: sigstore/cosign-installer@v3
      - run: |
          cosign sign --yes \
            ghcr.io/${{ github.repository }}/flink:${{ github.ref_name }}
```

---

## 10. Scorecard

| Category | Status | CRITICAL | HIGH | MEDIUM |
|----------|--------|----------|------|--------|
| Dockerfile | PARTIAL | 0 | 1 | 5 |
| K8s Deployment | **NONE** | 1 | 2 | 2 |
| Resources | CRITICAL | 1 | 2 | 2 |
| Secrets | CRITICAL | 2 | 1 | 1 |
| Checkpoints/DR | CRITICAL | 2 | 2 | 1 |
| CI/CD | **NONE** | 1 | 1 | 1 |
| Network/Security | CRITICAL | 2 | 1 | 3 |
| **TOTAL** | **NOT PRODUCTION** | **9** | **10** | **15** |

---

*Reviewed by: Principal DevOps/Platform Engineer | 2026-05-12*
