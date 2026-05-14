FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir minio==7.1.0 requests==2.31.0
COPY stats_writer.py .
CMD ["python", "stats_writer.py"]
