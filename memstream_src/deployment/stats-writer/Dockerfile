# ============================================================
# CA-DQStream Stats Writer Docker Image
# ============================================================
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    redis \
    minio \
    pyarrow \
    pandas \
    python-dateutil

# Copy application
COPY stats_writer.py /app/writer.py

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV WRITE_INTERVAL=60

# Run stats writer
CMD ["python", "-u", "writer.py"]
