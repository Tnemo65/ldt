#!/bin/bash
set -e

CERTS_DIR=/certs
mkdir -p "$CERTS_DIR"
cd "$CERTS_DIR"

# Idempotent guard: skip if keystore already exists
if [ -f "$CERTS_DIR/kafka.keystore.jks" ]; then
    echo "[redis-certs] Certificates already exist, skipping generation"
    exit 0
fi

# Load KAFKA_SSL_PASSWORD from environment (set in .env)
PASSWORD="${KAFKA_SSL_PASSWORD:-$(openssl rand -hex 32 2>/dev/null || echo "changeme_$(date +%s)")}"
openssl req -x509 -newkey rsa:4096 -keyout ca-key.pem -out ca-cert.pem -days 365 -nodes -subj /CN=CA-DQStream-Redis
openssl req -newkey rsa:4096 -keyout redis-key.pem -out redis.csr -nodes -subj /CN=redis
openssl x509 -req -in redis.csr -CA ca-cert.pem -CAkey ca-key.pem -out redis-cert.pem -days 365 -CAcreateserial
rm -f redis.csr ca-key.pem ca-cert.srl
echo "[redis-certs] Done"
ls -la /certs
