# ============================================================
# CA-DQStream Metrics Exporter Docker Image
# ============================================================
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    prometheus-client \
    redis \
    kafka-python \
    minio \
    pyyaml \
    requests

# Copy application
COPY cadqstream_metrics.py /app/metrics.py

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV METRICS_PORT=9250

# Expose metrics port
EXPOSE 9250

# Run metrics exporter
CMD ["python", "-u", "metrics.py"]
