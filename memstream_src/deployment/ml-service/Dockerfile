# ============================================================
# CA-DQStream ML Service Docker Image
# ============================================================
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application
COPY ml_service.py /app/service.py

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SERVICE_PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run service
CMD ["python", "-u", "service.py"]
