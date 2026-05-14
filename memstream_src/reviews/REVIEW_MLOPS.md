# MLOps Engineer Review — PLAN_v3.md

**Reviewer:** Principal MLOps Engineer (12 yrs MLflow, Vertex AI, SageMaker, Kubeflow)
**Date:** 2026-05-12
**Files Reviewed:** PLAN_v3.md (§1, §4, §5, §7), `src/models/model_registry.py`, `src/ml/train_iforest.py`, `src/ml/model_registry.py`, `src/evaluation/evaluator.py`, `benchmark_v6.py`, `scripts/train_iforest.py`, `scripts/batch_predict.py`, `src/api/ml_service.py`

---

## 1. Model Versioning & Registry

### 1.1 No Model Registry — CRITICAL

Models are stored as pickle files on filesystem:
```
/models/iforest_v3.pkl
/memstream_src/models/memstream_warmed.pt
```

No MLflow Model Registry, no SageMaker Model Registry, no version history, no stage tracking (STAGING→PRODUCTION), no SHA256 hash for integrity, no model comparison.

### 1.2 `model_registry.py` — Placeholder Only

```python
# src/models/model_registry.py:
def get_latest_model():
    # PLACEHOLDER: returns hardcoded path
    return "models/iforest_v3.pkl"
```

Not implemented. Not integrated with any model registry service.

### 1.3 MemStream Model Version = "unversioned" — CRITICAL

```python
# PLAN_v3.md config.py:
MEMSTREAM_MODEL_VERSION: str = "unversioned"  # No versioning!
```

No commit SHA, no training date, no dataset version. Cannot trace which model produced which result.

---

## 2. Model Signature & Contracts

### 2.1 No ModelSignature for MemStream — CRITICAL

MemStream has no input/output schema contract. No defined `input_schema` or `output_schema`. This makes:
- Model A/B testing impossible (no schema to validate)
- Training-serving skew detection impossible
- Model registry compatibility checks impossible

### 2.2 ml_service.py — Feature Dimension Validation Bug

```python
# src/api/ml_service.py:272
if len(features) != 21:
    raise ValueError(f"Expected 21 features, got {len(features)}")
```

**BUG:** `FeatureVectorizer` was updated to 25D but ML service still validates 21D. Live inference will reject all records after the vectorizer update with `ValueError`.

### 2.3 Hardcoded Threshold in ML Service — HIGH

```python
# src/api/ml_service.py:
anomaly_score = if_model.decision_function([features])[0]
is_anomaly = anomaly_score < -0.50  # hardcoded threshold
```

Threshold should come from model metadata or model registry, not hardcoded in API.

---

## 3. A/B Testing & Canary

### 3.1 "Canary Deployment" Is Per-Record Coin Flip — HIGH

```python
# src/api/ml_service.py:
if random.random() < CANARY_RATE:
    canary_score = memstream_model.predict(...)
```

This is NOT canary deployment. True canary:
1. Routes a **population** of requests to the new model
2. Compares **population-level metrics** between old and new
3. Uses statistical significance testing (Wilcoxon, t-test)
4. Promotes only if new model is better

Per-record randomization with no population comparison produces interleaved results where individual decisions are indistinguishable from random variation.

### 3.2 No Shadow Mode Implementation

PLAN_v3.md §1.2 defines "shadow mode" but `ml_service.py` has no shadow mode. No "shadow" model that runs alongside production, no shadow result storage, no shadow-vs-production comparison.

### 3.3 No Traffic Splitting

No traffic splitting between IsolationForest and MemStream for gradual rollout. All-or-nothing switch with no rollback window.

---

## 4. Feature Store

### 4.1 No Feature Store — CRITICAL

No Feast, Tecton, or any feature store. Hardcoded baseline constants:
```python
# ml_service.py:
expected_fare_per_mile = 2.5  # Hardcoded
```

Training-serving skew undetected. Baseline features can drift independently from serving features.

### 4.2 No Training-Serving Skew Detection

No mechanism to compare features used during training vs. serving. If vectorizer changes (e.g., 21D→25D), models trained on old features produce garbage predictions with no alert.

---

## 5. Continuous Training Pipeline

### 5.1 IEC "Retrain Model" Does Nothing — CRITICAL

```python
# iec_operator.py:
elif strategy == 'retrain_model':
    LOGGER.info(f"[IEC] Strategy: retrain_model (job_id={retrain_job_id})")
    # NOTE: Actual retraining is triggered asynchronously.
    # This would launch a Kubernetes Job or Airflow DAG.
```

**This is a placeholder comment.** No Kubernetes Job is actually launched. No Airflow DAG is triggered. No ML pipeline executes. The IEC reports "retrain_model" but the model never actually trains.

### 5.2 No Kubeflow/Airflow Pipeline

No Kubeflow Pipelines, no Apache Airflow DAGs, no Metaflow. The entire continuous training infrastructure is absent.

### 5.3 No Data Quality Gate Before Training

No automated checks before model training:
- Null rate in training data < 0.5%
- Temporal ordering verified
- Anomaly rate within expected range (1%–15%)
- Feature distribution matches serving distribution (PSI < 0.1)

---

## 6. Monitoring & Drift Detection

### 6.1 Training Monitoring — Basic Only

`benchmark_v6.py` has basic metrics (AUC-PR, AUC-ROC, F1, Precision, Recall). But:
- No calibration curves
- No SHAP values for feature importance
- No per-neighborhood breakdown
- No concept drift detection on training data

### 6.2 Production Drift Detection — IEC Only

IEC detects drift via ADWIN-U but only triggers responses, not metrics. No:
- Population Stability Index (PSI) monitoring
- KS-test on feature distributions
- Label drift monitoring
- Concept drift indices

### 6.3 No Model Performance Monitoring

No tracking of model performance over time:
- True positive rate on labeled data (if available)
- False positive rate over time
- Calibration drift

---

## 7. Ensemble Management

### 7.1 Voting Ensemble — Canary Overrides ML

```python
# meta_aggregator.py:
if has_violation:
    final_decision = 'ANOMALY'  # Canary always overrides
elif is_ml_anomaly:
    final_decision = 'ANOMALY'  # ML anomaly
else:
    final_decision = 'CLEAN'
```

This is hardcoded logic. No:
- Configurable weight per model
- Per-context weight adaptation
- Confidence-weighted voting
- Online ensemble learning

### 7.2 No Model Stacking

No meta-learner that combines canary rules and ML scores. Current "ensemble" is just priority override, not learned combination.

---

## 8. CRITICAL Issues

| # | File | Line | Issue | Fix |
|---|------|------|-------|-----|
| 1 | `src/models/model_registry.py` | — | No model registry — pickle files only | Implement MLflow ModelRegistry |
| 2 | `src/api/ml_service.py` | 272 | Validates 21D but vectorizer is 25D | Update to 25D |
| 3 | `src/api/ml_service.py` | ~210 | "Canary" is per-record coin flip, not canary deployment | Implement population-based canary with statistical testing |
| 4 | `src/ml/train_iforest.py` | 139 | Validates on same data as training | Use held-out validation set |
| 5 | `iec_operator.py` | — | `retrain_model` is a placeholder comment | Implement K8s Job / Airflow trigger |
| 6 | `src/api/ml_service.py` | — | No feature store — hardcoded baseline constants | Implement Feast feature store |

---

## 9. HIGH Issues

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `ml_service.py` | Threshold hardcoded (0.50) | Load from model metadata |
| 2 | `benchmark_v6.py` | No per-neighborhood evaluation | Add neighborhood-level metrics |
| 3 | `benchmark_v6.py` | No SHAP / feature importance | Add SHAP analysis |
| 4 | `train_iforest.py` | No data quality gate | Add null rate, PSI checks |
| 5 | `ml_service.py` | No shadow mode implementation | Add parallel shadow model + storage |
| 6 | `meta_aggregator.py` | Hardcoded voting logic | Add configurable weights + confidence |

---

## 10. Recommended Fixes

### Fix 1: MLflow Model Registry

```python
# src/ml/model_registry.py
import mlflow
from mlflow.tracking import MlflowClient

class MemStreamModelRegistry:
    def __init__(self, tracking_uri="http://mlflow:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.model_name = "cadqstream-memstream"
    
    def register_model(self, model_path, metrics, params, tags):
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for k, v in tags.items():
                mlflow.set_tag(k, v)
            
            # Log model with signature
            model_info = mlflow.pyfunc.log_model(
                model_name=self.model_name,
                python_model=MemStreamWrapper(model_path),
                signature=self._get_signature(),
                artifacts={"base_model": model_path},
            )
            
            # Register version
            version = self.client.get_model_version(
                self.model_name, model_info.version
            )
            
            # SHA256 hash for tamper detection
            sha256 = self._hash_file(model_path)
            self.client.set_model_version_tag(
                self.model_name, version.version, "sha256", sha256
            )
            
            return version
    
    def get_production_model(self):
        return self.client.get_model_version(self.model_name, "Production")
    
    def _get_signature(self):
        import mlflow.types
        return mlflow.models.ModelSignature(
            inputs=mlflow.models.TensorSpec(
                type=mlflow.types.DataType.float32,
                shape=(-1, 25)  # 25D features
            ),
            outputs=mlflow.models.TensorSpec(
                type=mlflow.types.DataType.float32,
                shape=(-1, 2)  # score, is_anomaly
            )
        )
    
    def _hash_file(self, path):
        import hashlib
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()
```

### Fix 2: Canary Router with Statistical Testing

```python
# src/mlops/canary_router.py
import numpy as np
from scipy import stats

class CanaryRouter:
    """Population-based canary routing with Wilcoxon significance testing."""
    
    def __init__(self, canary_rate=0.05, min_samples=500,
                 promotion_threshold=0.05):
        self.canary_rate = canary_rate
        self.min_samples = min_samples
        self.promotion_threshold = promotion_threshold
        
        # Shadow results (production model vs candidate model)
        self._shadow_prod = []  # [(score, decision, metadata), ...]
        self._shadow_cand = []
        self._shadow_lock = threading.Lock()
    
    def route(self, record, context_key):
        """Returns (model_name, features)."""
        if random.random() < self.canary_rate:
            return ("candidate", record)
        return ("production", record)
    
    def record_shadow_result(self, record, prod_result, cand_result):
        """Record shadow mode results for comparison."""
        with self._shadow_lock:
            self._shadow_prod.append(prod_result)
            self._shadow_cand.append(cand_result)
            
            if len(self._shadow_prod) >= self.min_samples:
                return self._evaluate()
        return None
    
    def _evaluate(self):
        """Wilcoxon signed-rank test on shadow results."""
        if len(self._shadow_prod) < self.min_samples:
            return None
        
        prod_scores = [r['score'] for r in self._shadow_prod]
        cand_scores = [r['score'] for r in self._shadow_cand]
        
        # Paired test on scores
        stat, p_value = stats.wilcoxon(prod_scores, cand_scores)
        
        # Cohen's d for effect size
        diff = np.array(cand_scores) - np.array(prod_scores)
        cohens_d = diff.mean() / diff.std()
        
        result = {
            'p_value': p_value,
            'cohens_d': cohens_d,
            'n_samples': len(prod_scores),
            'can_promote': p_value < self.promotion_threshold,
        }
        
        # Reset for next evaluation window
        self._shadow_prod = []
        self._shadow_cand = []
        return result
```

### Fix 3: IEC Continuous Training Trigger

```python
# src/mlops/iec_training_trigger.py
import kubernetes
from kubernetes.client import BatchV1Api

class IECContinuousTrainingTrigger:
    """Triggers model retraining when IEC detects sustained drift."""
    
    def __init__(self, namespace="cadqstream"):
        self.namespace = namespace
        self.batch_api = BatchV1Api()
    
    def trigger_retrain(self, iec_decision, context_key):
        """Launch Kubernetes Job for model retraining."""
        if iec_decision['strategy'] != 'retrain_model':
            return None
        
        # Only trigger if confidence is high
        if iec_decision.get('confidence', 0) < 0.8:
            return None
        
        job_name = f"cadqstream-retrain-{context_key}-{int(time.time())}"
        
        job = kubernetes.client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=kubernetes.client.V1ObjectMeta(
                name=job_name,
                namespace=self.namespace,
                labels={
                    "app": "cadqstream",
                    "component": "retraining",
                    "context_key": context_key,
                }
            ),
            spec=kubernetes.client.V1JobSpec(
                ttl_seconds_after_finished=86400,  # Clean up after 24h
                backoff_limit=2,
                template=kubernetes.client.V1PodTemplateSpec(
                    spec=kubernetes.client.V1PodSpec(
                        containers=[kubernetes.client.V1Container(
                            name="retrain",
                            image="ghcr.io/org/cadqstream-mltrain:latest",
                            env=[
                                kubernetes.client.V1EnvVar(
                                    name="RETRAIN_CONTEXT_KEY",
                                    value=context_key
                                ),
                                kubernetes.client.V1EnvVar(
                                    name="DRIFT_SIGNAL",
                                    value=str(iec_decision)
                                ),
                            ],
                            resources=kubernetes.client.V1ResourceRequirements(
                                requests={"cpu": "2", "memory": "4Gi"},
                                limits={"cpu": "4", "memory": "8Gi"},
                            )
                        )],
                        restart_policy="Never",
                    )
                )
            )
        )
        
        self.batch_api.create_namespaced_job(
            namespace=self.namespace,
            body=job
        )
        return job_name
```

---

*Reviewed by: Principal MLOps Engineer | 2026-05-12*
