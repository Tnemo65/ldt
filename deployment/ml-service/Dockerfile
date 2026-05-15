# ============================================================
# CA-DQStream ML Service Docker Image
# Build context: c:\proj\ldt (project root)
# Run from project root: docker build -t ldt-ml-service:latest -f deployment/ml-service/Dockerfile .
# ============================================================
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# CPU-only torch from PyTorch wheel index (avoids 1.5GB GPU packages)
COPY deployment/ml-service/requirements.txt /tmp/requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the FULL application (src/api/ml_service.py is 1425 lines with Phase 5C features)
COPY src/api/ml_service.py /app/ml_service.py

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV SERVICE_PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run service directly via Python (if __name__ == "__main__" calls uvicorn)
CMD ["python", "-u", "/app/ml_service.py"]
