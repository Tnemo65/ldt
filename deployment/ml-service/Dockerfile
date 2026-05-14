FROM python:3.10-slim

WORKDIR /app

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_service.py .

RUN mkdir -p models && chmod 755 models
RUN ln -sf /models /app/models

EXPOSE 8000

CMD ["uvicorn", "ml_service:app", "--host", "0.0.0.0", "--port", "8000"]
