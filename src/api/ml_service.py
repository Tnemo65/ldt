#!/usr/bin/env python3
"""
FastAPI ML Service - Production-Ready Model Serving.
Task 4.1-4.5: Async model cache, health checks, Prometheus metrics

Features:
- Async model loading with LRU cache (asyncache)
- Health check endpoints
- Prometheus metrics export
- Model inference API
- Strategy execution endpoints

Usage:
  uvicorn src.api.ml_service:app --host 0.0.0.0 --port 8000
  docker run -p 8000:8000 cadqstream-ml-service
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import pickle
import numpy as np
from pathlib import Path
import time
from datetime import datetime

# Async cache
try:
    from asyncache import cached
    from cachetools import LRUCache
    ASYNCACHE_AVAILABLE = True
except ImportError:
    ASYNCACHE_AVAILABLE = False
    print("⚠ asyncache not available - model caching disabled")

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("⚠ prometheus_client not available - metrics disabled")


# FastAPI app
app = FastAPI(
    title="CA-DQStream ML Service",
    description="Production ML serving for anomaly detection",
    version="1.0.0"
)


# Prometheus metrics (if available)
if PROMETHEUS_AVAILABLE:
    INFERENCE_REQUESTS = Counter(
        'inference_requests_total',
        'Total inference requests',
        ['model', 'status']
    )
    INFERENCE_LATENCY = Histogram(
        'inference_latency_seconds',
        'Inference latency in seconds',
        ['model']
    )
    MODEL_LOAD_TIME = Histogram(
        'model_load_time_seconds',
        'Model loading time in seconds'
    )
    ACTIVE_MODELS = Gauge(
        'active_models_count',
        'Number of models loaded in memory'
    )


# Request/Response models
class InferenceRequest(BaseModel):
    """Inference request schema."""
    features: List[float]
    model_name: str = "iforest_v2"


class InferenceResponse(BaseModel):
    """Inference response schema."""
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    model_name: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_loaded: int
    uptime_seconds: float


class StrategyRequest(BaseModel):
    """Strategy execution request."""
    strategy: str
    meta_metrics: Dict
    neighborhood: str


# Model cache (async if available)
if ASYNCACHE_AVAILABLE:
    @cached(cache=LRUCache(maxsize=10))
    async def load_model_cached(model_path: str):
        """Load model with async LRU cache.

        Args:
            model_path: Path to model file

        Returns:
            Loaded model
        """
        start_time = time.time()

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        load_time = time.time() - start_time

        if PROMETHEUS_AVAILABLE:
            MODEL_LOAD_TIME.observe(load_time)
            ACTIVE_MODELS.inc()

        print(f"[ModelCache] Loaded {model_path} ({load_time:.3f}s)")

        return model
else:
    # Fallback: simple dict cache
    _model_cache = {}

    async def load_model_cached(model_path: str):
        """Fallback model cache using dict."""
        if model_path in _model_cache:
            return _model_cache[model_path]

        start_time = time.time()

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        load_time = time.time() - start_time
        _model_cache[model_path] = model

        print(f"[ModelCache] Loaded {model_path} ({load_time:.3f}s)")

        return model


# Load scaler
async def load_scaler():
    """Load StandardScaler."""
    scaler_path = 'models/scaler.pkl'

    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return scaler


# Startup/Shutdown events
app_start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Warmup: preload default model."""
    print("="*60)
    print("CA-DQStream ML Service Starting...")
    print("="*60)

    try:
        # Preload default model
        await load_model_cached('models/iforest_model_v2.pkl')
        await load_scaler()
        print("✅ Default model preloaded")
    except Exception as e:
        print(f"⚠ Could not preload model: {e}")

    print("="*60)
    print("Service ready on port 8000")
    print("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\n[Shutdown] ML Service stopping...")


# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Returns service status and basic metrics.
    """
    uptime = time.time() - app_start_time

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=len(_model_cache) if not ASYNCACHE_AVAILABLE else 0,
        uptime_seconds=uptime
    )


@app.get("/health/ready")
async def readiness_check():
    """Readiness check (Kubernetes)."""
    # Check if default model is loaded
    try:
        await load_model_cached('models/iforest_model_v2.pkl')
        return {"status": "ready"}
    except:
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/health/live")
async def liveness_check():
    """Liveness check (Kubernetes)."""
    return {"status": "alive"}


# Metrics endpoint
if PROMETHEUS_AVAILABLE:
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        from fastapi.responses import Response
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Inference endpoint
@app.post("/api/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Score features with ML model.

    Args:
        request: Inference request with features and model name

    Returns:
        Anomaly score and decision
    """
    start_time = time.time()

    try:
        # Load model
        model_path = f'models/{request.model_name}.pkl'
        model = await load_model_cached(model_path)

        # Load scaler
        scaler = await load_scaler()

        # Validate features
        if len(request.features) != 21:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 21 features, got {len(request.features)}"
            )

        # Scale features
        features = np.array([request.features])
        features_scaled = scaler.transform(features)[0]

        # sklearn IsolationForest: score_samples returns negative scores (lower = more anomalous)
        # Negate so higher = more anomalous (River-like semantics)
        raw_score = model.score_samples(features_scaled.reshape(1, -1))[0]
        anomaly_score = -raw_score

        # Threshold (simplified - in production, use context-aware)
        threshold = 0.50
        is_anomaly = anomaly_score > threshold

        inference_time = (time.time() - start_time) * 1000  # ms

        # Update metrics
        if PROMETHEUS_AVAILABLE:
            INFERENCE_REQUESTS.labels(model=request.model_name, status='success').inc()
            INFERENCE_LATENCY.labels(model=request.model_name).observe(time.time() - start_time)

        return InferenceResponse(
            anomaly_score=float(anomaly_score),
            is_anomaly=bool(is_anomaly),
            threshold=float(threshold),
            model_name=request.model_name,
            inference_time_ms=inference_time
        )

    except FileNotFoundError:
        if PROMETHEUS_AVAILABLE:
            INFERENCE_REQUESTS.labels(model=request.model_name, status='model_not_found').inc()
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_name}")

    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            INFERENCE_REQUESTS.labels(model=request.model_name, status='error').inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# Strategy execution endpoints
@app.post("/api/strategy/adjust_threshold")
async def execute_adjust_threshold(request: StrategyRequest):
    """Execute threshold adjustment strategy.

    Args:
        request: Strategy execution request

    Returns:
        Execution result
    """
    # In production: update thresholds in database/config
    new_threshold = 0.55 if request.meta_metrics.get('anomaly_rate', 0) > 0.15 else 0.50

    return {
        "strategy": "adjust_threshold",
        "status": "executed",
        "neighborhood": request.neighborhood,
        "new_threshold": new_threshold,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/strategy/retrain_model")
async def execute_retrain_model(request: StrategyRequest):
    """Execute model retraining strategy.

    Args:
        request: Strategy execution request

    Returns:
        Execution result
    """
    # In production: trigger async retraining job
    return {
        "strategy": "retrain_model",
        "status": "triggered",
        "neighborhood": request.neighborhood,
        "job_id": f"retrain_{int(time.time())}",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/strategy/switch_model")
async def execute_switch_model(request: StrategyRequest):
    """Execute model switching strategy.

    Args:
        request: Strategy execution request

    Returns:
        Execution result
    """
    # In production: switch to backup model
    return {
        "strategy": "switch_model",
        "status": "executed",
        "neighborhood": request.neighborhood,
        "active_model": "iforest_backup",
        "timestamp": datetime.utcnow().isoformat()
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "CA-DQStream ML Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "inference": "/api/inference",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
