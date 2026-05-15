#!/bin/bash
# =============================================================================
# Generate self-signed TLS certificates for Redis
# Run BEFORE docker compose up
# =============================================================================

set -e

CERTS_DIR="$(dirname "$0")/certs"
mkdir -p "$CERTS_DIR"

echo "Generating Redis TLS certificates in $CERTS_DIR..."

# Generate CA
openssl req -x509 -newkey rsa:4096 \
    -keyout "$CERTS_DIR/ca-key.pem" \
    -out "$CERTS_DIR/ca-cert.pem" \
    -days 365 -nodes \
    -subj "/CN=CA-DQStream-Redis"

# Generate Redis key + CSR
openssl req -newkey rsa:4096 \
    -keyout "$CERTS_DIR/redis-key.pem" \
    -out "$CERTS_DIR/redis.csr" \
    -nodes \
    -subj "/CN=redis"

# Sign Redis cert with CA
openssl x509 -req \
    -in "$CERTS_DIR/redis.csr" \
    -CA "$CERTS_DIR/ca-cert.pem" \
    -CAkey "$CERTS_DIR/ca-key.pem" \
    -out "$CERTS_DIR/redis-cert.pem" \
    -days 365

# Cleanup CSR
rm -f "$CERTS_DIR/redis.csr"

echo "Redis TLS certificates generated successfully:"
echo "  CA cert:     $CERTS_DIR/ca-cert.pem"
echo "  Redis cert:  $CERTS_DIR/redis-cert.pem"
echo "  Redis key:   $CERTS_DIR/redis-key.pem"
echo ""
echo "To start the stack:"
echo "  1. Review and update passwords in deployment/.env"
echo "  2. cd deployment && docker compose up -d"
