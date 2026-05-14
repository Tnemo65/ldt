FROM python:3.10-slim

WORKDIR /app

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends curl liblz4-dev lz4 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY action_replay_worker.py .

ENV KAFKA_BOOTSTRAP_SERVERS=kafka:9092
ENV ML_SERVICE_URL=http://ml-service:8000
ENV MAX_RETRIES=10
ENV BACKOFF_BASE=2

CMD ["python", "action_replay_worker.py", \
     "--kafka-bootstrap", "kafka:9092", \
     "--ml-service-url", "http://ml-service:8000", \
     "--max-retries", "10", \
     "--backoff-base", "2"]
