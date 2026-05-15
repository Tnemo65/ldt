#!/bin/bash
# =============================================================================
# Generate self-signed TLS keystores and truststores for Kafka
# Run BEFORE docker compose up
# Requires: openssl, keytool (Java JDK)
# =============================================================================

set -e

CERTS_DIR="$(dirname "$0")/certs"
PASSWORD="${KAFKA_SSL_PASSWORD:-change-kafka-ssl-password}"
KEY_PASSWORD="${KAFKA_SSL_KEY_PASSWORD:-change-kafka-ssl-key-password}"

mkdir -p "$CERTS_DIR"

echo "Generating Kafka TLS certificates in $CERTS_DIR..."

# Generate CA key and certificate
openssl req -x509 -newkey rsa:4096 \
    -keyout "$CERTS_DIR/ca-key.pem" \
    -out "$CERTS_DIR/ca-cert.pem" \
    -days 365 -nodes \
    -subj "/CN=Kafka-CA-DQStream"

echo "Generated CA certificate"

# Convert CA cert to PKCS12 format for truststore creation
openssl pkcs12 -export \
    -in "$CERTS_DIR/ca-cert.pem" \
    -inkey "$CERTS_DIR/ca-key.pem" \
    -out "$CERTS_DIR/ca.p12" \
    -name "kafka-ca" \
    -passout pass:"$PASSWORD"

# Create JKS truststore from CA
keytool -importkeystore \
    -srckeystore "$CERTS_DIR/ca.p12" \
    -srcstoretype PKCS12 \
    -srcstorepass "$PASSWORD" \
    -destkeystore "$CERTS_DIR/kafka.truststore.jks" \
    -deststoretype JKS \
    -storepass "$PASSWORD" \
    -noprompt

# Generate broker key and CSR
openssl req -newkey rsa:4096 \
    -keyout "$CERTS_DIR/kafka-broker-key.pem" \
    -out "$CERTS_DIR/kafka-broker.csr" \
    -nodes \
    -subj "/CN=kafka"

# Sign broker certificate with CA
openssl x509 -req \
    -in "$CERTS_DIR/kafka-broker.csr" \
    -CA "$CERTS_DIR/ca-cert.pem" \
    -CAkey "$CERTS_DIR/ca-key.pem" \
    -out "$CERTS_DIR/kafka-broker-cert.pem" \
    -days 365

# Create PKCS12 keystore from broker cert + key
openssl pkcs12 -export \
    -in "$CERTS_DIR/kafka-broker-cert.pem" \
    -inkey "$CERTS_DIR/kafka-broker-key.pem" \
    -out "$CERTS_DIR/kafka-broker.p12" \
    -name "kafka-broker" \
    -passout pass:"$PASSWORD"

# Create JKS keystore
keytool -importkeystore \
    -srckeystore "$CERTS_DIR/kafka-broker.p12" \
    -srcstoretype PKCS12 \
    -srcstorepass "$PASSWORD" \
    -destkeystore "$CERTS_DIR/kafka.keystore.jks" \
    -deststoretype JKS \
    -storepass "$PASSWORD" \
    -noprompt

# Import CA into broker keystore (for client cert validation)
keytool -importcert \
    -alias CARoot \
    -file "$CERTS_DIR/ca-cert.pem" \
    -keystore "$CERTS_DIR/kafka.keystore.jks" \
    -storepass "$PASSWORD" \
    -noprompt

# Cleanup intermediate files (keep broker PEMs for kafka-exporter)
rm -f "$CERTS_DIR/ca.p12" \
       "$CERTS_DIR/kafka-broker.p12" \
       "$CERTS_DIR/kafka-broker.csr"

echo ""
echo "Kafka TLS certificates generated successfully:"
echo "  Truststore: $CERTS_DIR/kafka.truststore.jks"
echo "  Keystore:   $CERTS_DIR/kafka.keystore.jks"
echo "  CA cert:    $CERTS_DIR/ca-cert.pem"
echo ""
echo "Password for both stores: $PASSWORD"
echo ""
echo "To start the stack:"
echo "  1. Review and update passwords in deployment/.env"
echo "  2. cd deployment && docker compose up -d"
