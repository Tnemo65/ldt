FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir flask>=2.3.0 prometheus-client>=0.19.0 requests>=2.31.0
COPY cadqstream_metrics.py .
EXPOSE 9250
ENV FLASK_ENV=production
ENV FLINK_JM_URL=http://flink-jobmanager:8081
CMD ["python", "cadqstream_metrics.py"]
