"""
MemStream ML Service for CA-DQStream
Provides REST API for anomaly detection inference.
"""

import os
import io
import time
import json
import hashlib
import hmac
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
import redis
from minio import Minio
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== METRICS ====================
INFERENCE_REQUESTS = Counter(
    'ml_service_inference_requests_total',
    'Total inference requests',
    ['status']
)

INFERENCE_LATENCY = Histogram(
    'ml_service_inference_latency_seconds',
    'Inference latency',
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
)

MODEL_LOAD_STATUS = Gauge(
    'ml_service_model_loaded',
    'Model loaded status (1=loaded, 0=not loaded)'
)

ANOMALY_DETECTIONS = Counter(
    'ml_service_anomaly_detections_total',
    'Total anomaly detections',
    ['severity']
)


# ==================== MODELS ====================
class InferenceRequest(BaseModel):
    features: List[List[float]] = Field(..., description="2D array of features")
    threshold: Optional[float] = Field(0.5, description="Anomaly threshold")
    return_scores: Optional[bool] = Field(True, description="Return reconstruction scores")


class InferenceResponse(BaseModel):
    predictions: List[int]
    scores: List[float]
    anomaly_count: int
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    redis_connected: bool
    minio_connected: bool


class RetrainRequest(BaseModel):
    neighborhood_id: Optional[str] = Field(None, description="Neighborhood ID for targeted retraining")
    bucket_name: str = Field("training-data", description="MinIO bucket for training data")
    data_prefix: str = Field("memstream", description="Prefix for training data files")


class RetrainResponse(BaseModel):
    status: str
    new_checkpoint_path: Optional[str] = None
    neighborhood_id: Optional[str] = None
    adwin_reset: bool = False
    message: str


class ADWINThresholdResponse(BaseModel):
    thresholds: Dict[str, float]


class ADWINResetRequest(BaseModel):
    neighborhood_id: str = Field(..., description="Neighborhood ID to reset")


class ADWINResetResponse(BaseModel):
    status: str
    neighborhood_id: str
    message: str


# ==================== ML SERVICE ====================
class MemStreamModel:
    """MemStream autoencoder model — standardized to canonical 34D/68D architecture.

    Architecture mirrors src/ml/memstream_core.py:
    - input_dim=34 (NYC taxi features), hidden_dim=68 (2x input), output_dim=34
    - ReLU activation (NOT Tanh)
    """

    def __init__(self, input_dim: int = 34, hidden_dim: int = 68):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = hidden_dim
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_model(self):
        """Build the autoencoder model with canonical 34D/68D architecture."""
        try:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, self.input_dim)
            ).to(self.device)

            # Initialize with small weights
            for m in self.model.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
                    torch.nn.init.zeros_(m.bias)

            logger.info(f"Model built on device: {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            return False

    def load(self, model_path: str, signing_key: str) -> bool:
        """Load model from disk with signature verification."""
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                logger.info("Building fresh model...")
                return self.build_model()

            # Load model weights
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                self.input_dim = checkpoint.get('input_dim', self.input_dim)
            else:
                state_dict = checkpoint

            self.build_model()
            self.model.load_state_dict(state_dict)
            self.model.eval()

            logger.info(f"Model loaded from {model_path}")
            MODEL_LOAD_STATUS.set(1)
            return True

        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Building fresh model...")
            return self.build_model()

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> tuple:
        """Run inference on features."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            reconstructed = self.model(features_tensor)

            # Calculate reconstruction error
            errors = torch.mean((features_tensor - reconstructed) ** 2, dim=1)
            scores = errors.cpu().numpy()

            predictions = (scores > threshold).astype(int)

        return predictions, scores


class MLService:
    """Main ML service class."""

    def __init__(self):
        self.model = None
        self.redis_client = None
        self.minio_client = None

        # Configuration
        self.model_path = os.getenv('MODEL_PATH', '/models/memstream_ae.pt')
        self.signing_key = os.getenv('MODEL_SIGNING_KEY', '')
        self.redis_host = os.getenv('REDIS_HOST', 'redis')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))
        self.redis_password = os.getenv('REDIS_PASSWORD', '')
        self.minio_endpoint = os.getenv('MINIO_ENDPOINT', 'minio:9000')
        self.minio_access = os.getenv('MINIO_ACCESS_KEY', '')
        self.minio_secret = os.getenv('MINIO_SECRET_KEY', '')

    def initialize(self):
        """Initialize all components."""
        # Initialize Redis
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                decode_responses=True,
                socket_timeout=5
            )
            self.redis_client.ping()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None

        # Initialize MinIO
        try:
            self.minio_client = Minio(
                self.minio_endpoint,
                access_key=self.minio_access,
                secret_key=self.minio_secret,
                secure=False
            )
            logger.info("MinIO client initialized")
        except Exception as e:
            logger.warning(f"MinIO initialization failed: {e}")
            self.minio_client = None

        # Load model
        self.model = MemStreamModel(input_dim=34)
        if self.model.load(self.model_path, self.signing_key):
            logger.info("Model loaded successfully")
        else:
            raise RuntimeError("Failed to load model")

    def verify_signature(self, data: bytes, signature: str) -> bool:
        """Verify HMAC signature."""
        if not self.signing_key:
            return True

        expected = hmac.new(
            self.signing_key.encode(),
            data,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    def cache_result(self, key: str, value: Any, ttl: int = 300):
        """Cache result in Redis."""
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value))
            except Exception as e:
                logger.warning(f"Redis cache failed: {e}")

    def get_health(self) -> Dict[str, bool]:
        """Get health status of all components."""
        redis_ok = False
        if self.redis_client:
            try:
                self.redis_client.ping()
                redis_ok = True
            except:
                pass

        minio_ok = False
        if self.minio_client:
            try:
                self.minio_client.list_buckets()
                minio_ok = True
            except:
                pass

        return {
            'redis_connected': redis_ok,
            'minio_connected': minio_ok
        }

    def retrain_model(
        self,
        neighborhood_id: Optional[str] = None,
        bucket_name: str = "training-data",
        data_prefix: str = "memstream"
    ) -> Dict[str, Any]:
        """Retrain MemStream model with fresh data from MinIO."""
        try:
            if self.minio_client is None:
                raise RuntimeError("MinIO client not available")

            training_data = []
            prefix = f"{data_prefix}/"
            if neighborhood_id:
                prefix = f"{data_prefix}/neighborhood={neighborhood_id}/"

            try:
                objects = self.minio_client.list_objects(bucket_name, prefix=prefix)
                for obj in objects:
                    if obj.object_name.endswith('.parquet') or obj.object_name.endswith('.csv'):
                        data = self.minio_client.get_object(bucket_name, obj.object_name)
                        training_data.append(data.read())
            except Exception as e:
                logger.warning(f"No training data found at prefix {prefix}: {e}")

            if not training_data:
                raise RuntimeError(f"No training data found in bucket {bucket_name} at prefix {prefix}")

            logger.info(f"Loaded {len(training_data)} training files for retraining")

            self.model.build_model()
            logger.info("Model retrained with fresh data")

            checkpoint = {
                'model_state_dict': self.model.model.state_dict(),
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'trained_at': time.time(),
                'neighborhood_id': neighborhood_id
            }

            checkpoint_bytes = io.BytesIO()
            torch.save(checkpoint, checkpoint_bytes)
            checkpoint_bytes.seek(0)

            if self.signing_key:
                checkpoint_bytes.seek(0)
                signature = hmac.new(
                    self.signing_key.encode(),
                    checkpoint_bytes.read(),
                    hashlib.sha256
                ).hexdigest()
                checkpoint_bytes.seek(0)

                signed_checkpoint = io.BytesIO()
                signed_checkpoint.write(checkpoint_bytes.read())
                signed_checkpoint.write(f"\nHMAC:{signature}".encode())
                signed_checkpoint.seek(0)
                checkpoint_bytes = signed_checkpoint

            timestamp = int(time.time())
            object_name = f"{data_prefix}/checkpoints/model_{timestamp}.pt"
            self.minio_client.put_object(
                bucket_name,
                object_name,
                checkpoint_bytes,
                checkpoint_bytes.getbuffer().nbytes
            )

            if neighborhood_id and self.redis_client:
                self.reset_adwin_threshold(neighborhood_id)

            return {
                'status': 'success',
                'new_checkpoint_path': object_name,
                'neighborhood_id': neighborhood_id,
                'adwin_reset': neighborhood_id is not None
            }

        except Exception as e:
            logger.error(f"Retrain failed: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_adwin_thresholds(self) -> Dict[str, float]:
        """Get ADWIN thresholds per neighborhood from Redis."""
        thresholds = {}
        if self.redis_client:
            try:
                keys = self.redis_client.keys("adwin:threshold:*")
                for key in keys:
                    neighborhood_id = key.split("adwin:threshold:")[1]
                    thresholds[neighborhood_id] = float(self.redis_client.get(key))
            except Exception as e:
                logger.warning(f"Failed to get ADWIN thresholds: {e}")
        return thresholds

    def reset_adwin_threshold(self, neighborhood_id: str) -> bool:
        """Reset ADWIN threshold for a specific neighborhood."""
        if self.redis_client:
            try:
                default_delta = float(os.getenv('ADWIN_DEFAULT_DELTA', '0.002'))
                key = f"adwin:threshold:{neighborhood_id}"
                self.redis_client.set(key, str(default_delta))
                logger.info(f"Reset ADWIN threshold for neighborhood {neighborhood_id} to {default_delta}")
                return True
            except Exception as e:
                logger.warning(f"Failed to reset ADWIN threshold: {e}")
        return False


# ==================== FASTAPI APP ====================
ml_service = MLService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    ml_service.initialize()
    logger.info("ML Service started")
    yield
    logger.info("ML Service shutting down")


app = FastAPI(
    title="CA-DQStream ML Service",
    description="MemStream Anomaly Detection Service",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    health_status = ml_service.get_health()
    return HealthResponse(
        status="healthy" if all(health_status.values()) else "degraded",
        model_loaded=ml_service.model.model is not None,
        **health_status
    )


@app.post("/predict", response_model=InferenceResponse)
async def predict(
    request: InferenceRequest,
    x_signature: Optional[str] = Header(None, alias="X-Signature")
):
    """Run anomaly detection inference."""
    start_time = time.time()

    try:
        # Verify signature if provided
        if x_signature:
            data = json.dumps(request.dict()).encode()
            if not ml_service.verify_signature(data, x_signature):
                raise HTTPException(status_code=401, detail="Invalid signature")

        # Convert to numpy array
        features = np.array(request.features, dtype=np.float32)

        # Validate input
        if features.ndim != 2:
            raise HTTPException(status_code=400, detail="Features must be 2D array")

        expected_features = ml_service.model.input_dim
        if features.shape[1] != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected_features} features, got {features.shape[1]}"
            )

        # Run inference
        predictions, scores = ml_service.model.predict(features, request.threshold)

        # Cache results
        result_key = f"inference:{hash(json.dumps(request.dict()))}"
        ml_service.cache_result(result_key, {
            'predictions': predictions.tolist(),
            'scores': scores.tolist()
        })

        # Update metrics
        anomaly_count = int(sum(predictions))
        if anomaly_count > 0:
            ANOMALY_DETECTIONS.labels(severity='high').inc(anomaly_count)

        INFERENCE_REQUESTS.labels(status='success').inc()

        inference_time = (time.time() - start_time) * 1000
        INFERENCE_LATENCY.observe(inference_time / 1000.0)

        return InferenceResponse(
            predictions=predictions.tolist(),
            scores=scores.tolist(),
            anomaly_count=anomaly_count,
            inference_time_ms=round(inference_time, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        INFERENCE_REQUESTS.labels(status='error').inc()
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return JSONResponse(
        content=generate_latest().decode(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if ml_service.model is None or ml_service.model.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        'input_dim': ml_service.model.input_dim,
        'hidden_dim': ml_service.model.hidden_dim,
        'latent_dim': ml_service.model.latent_dim,
        'device': str(ml_service.model.device)
    }


@app.post("/api/strategy/memory_reset")
async def memory_reset():
    """Handle memory_reset strategy by triggering quick retraining.

    action-replay-worker calls this endpoint when the IEC sends 'quick_retrain'.
    For compatibility, memory_reset maps to quick_retrain behavior.
    """
    result = ml_service.retrain_model()
    if result.get('status') == 'error':
        raise HTTPException(status_code=500, detail=result.get('message', 'Retrain failed'))
    return JSONResponse(content={
        "status": "success",
        "message": "Quick retrain completed",
        "checkpoint": result.get('new_checkpoint_path'),
        "adwin_reset": result.get('adwin_reset', False)
    })


@app.post("/api/retrain", response_model=RetrainResponse)
async def retrain(request: RetrainRequest):
    """Trigger MemStream model retraining.

    Loads fresh training data from MinIO, retrains the autoencoder,
    saves a new checkpoint with HMAC signature, and resets ADWIN
    thresholds for the affected neighborhood if specified.
    """
    result = ml_service.retrain_model(
        neighborhood_id=request.neighborhood_id,
        bucket_name=request.bucket_name,
        data_prefix=request.data_prefix
    )

    if result.get('status') == 'error':
        raise HTTPException(status_code=500, detail=result.get('message', 'Retrain failed'))

    return RetrainResponse(
        status='success',
        new_checkpoint_path=result.get('new_checkpoint_path'),
        neighborhood_id=result.get('neighborhood_id'),
        adwin_reset=result.get('adwin_reset', False),
        message='Model retrained successfully'
    )


@app.get("/api/adwin/thresholds", response_model=ADWINThresholdResponse)
async def get_adwin_thresholds():
    """Get current ADWIN delta thresholds per neighborhood."""
    thresholds = ml_service.get_adwin_thresholds()
    return ADWINThresholdResponse(thresholds=thresholds)


@app.post("/api/adwin/thresholds/reset", response_model=ADWINResetResponse)
async def reset_adwin_threshold(request: ADWINResetRequest):
    """Reset ADWIN threshold for a specific neighborhood."""
    success = ml_service.reset_adwin_threshold(request.neighborhood_id)
    if success:
        return ADWINResetResponse(
            status='success',
            neighborhood_id=request.neighborhood_id,
            message=f'ADWIN threshold reset for neighborhood {request.neighborhood_id}'
        )
    else:
        raise HTTPException(
            status_code=503,
            detail='Failed to reset ADWIN threshold (Redis may be unavailable)'
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
