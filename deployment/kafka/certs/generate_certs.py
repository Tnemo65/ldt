#!/usr/bin/env python3
"""Generate self-signed Kafka TLS certificates."""
import subprocess, os, sys

CERTS_DIR = os.path.dirname(os.path.abspath(__file__))
PASSWORD = os.environ.get('KAFKA_SSL_PASSWORD', 'change-kafka-ssl-password')
KEY_PASSWORD = os.environ.get('KAFKA_SSL_KEY_PASSWORD', 'change-kafka-ssl-key-password')

os.makedirs(CERTS_DIR, exist_ok=True)

print(f"Generating Kafka TLS certificates in {CERTS_DIR}...")

# Generate CA
subprocess.run([
    'openssl', 'req', '-x509', '-newkey', 'rsa:4096',
    '-keyout', f'{CERTS_DIR}/ca-key.pem',
    '-out', f'{CERTS_DIR}/ca-cert.pem',
    '-days', '365', '-nodes',
    '-subj', '/CN=Kafka-CA-DQStream'
], check=True)
print("Generated CA certificate")

# CA p12
subprocess.run([
    'openssl', 'pkcs12', '-export',
    '-in', f'{CERTS_DIR}/ca-cert.pem',
    '-inkey', f'{CERTS_DIR}/ca-key.pem',
    '-out', f'{CERTS_DIR}/ca.p12',
    '-name', 'kafka-ca',
    '-passout', f'pass:{PASSWORD}'
], check=True)

# Truststore JKS
subprocess.run([
    'keytool', '-importkeystore',
    '-srckeystore', f'{CERTS_DIR}/ca.p12',
    '-srcstoretype', 'PKCS12',
    '-srcstorepass', PASSWORD,
    '-destkeystore', f'{CERTS_DIR}/kafka.truststore.jks',
    '-deststoretype', 'JKS',
    '-storepass', PASSWORD,
    '-noprompt'
], check=True)
print("Generated truststore JKS")

# Broker key
subprocess.run([
    'openssl', 'req', '-newkey', 'rsa:4096',
    '-keyout', f'{CERTS_DIR}/kafka-broker-key.pem',
    '-out', f'{CERTS_DIR}/kafka-broker.csr',
    '-nodes',
    '-subj', '/CN=kafka'
], check=True)

# Sign broker cert
subprocess.run([
    'openssl', 'x509', '-req',
    '-in', f'{CERTS_DIR}/kafka-broker.csr',
    '-CA', f'{CERTS_DIR}/ca-cert.pem',
    '-CAkey', f'{CERTS_DIR}/ca-key.pem',
    '-out', f'{CERTS_DIR}/kafka-broker-cert.pem',
    '-days', '365'
], check=True)

# Broker p12
subprocess.run([
    'openssl', 'pkcs12', '-export',
    '-in', f'{CERTS_DIR}/kafka-broker-cert.pem',
    '-inkey', f'{CERTS_DIR}/kafka-broker-key.pem',
    '-out', f'{CERTS_DIR}/kafka-broker.p12',
    '-name', 'kafka-broker',
    '-passout', f'pass:{PASSWORD}'
], check=True)

# Broker keystore JKS
subprocess.run([
    'keytool', '-importkeystore',
    '-srckeystore', f'{CERTS_DIR}/kafka-broker.p12',
    '-srcstoretype', 'PKCS12',
    '-srcstorepass', PASSWORD,
    '-destkeystore', f'{CERTS_DIR}/kafka.keystore.jks',
    '-deststoretype', 'JKS',
    '-storepass', PASSWORD,
    '-noprompt'
], check=True)

# Import CA into broker keystore
subprocess.run([
    'keytool', '-importcert',
    '-alias', 'CARoot',
    '-file', f'{CERTS_DIR}/ca-cert.pem',
    '-keystore', f'{CERTS_DIR}/kafka.keystore.jks',
    '-storepass', PASSWORD,
    '-noprompt'
], check=True)
print("Generated keystore JKS")

# Cleanup intermediates
for f in ['ca.p12', 'kafka-broker.p12', 'kafka-broker.csr', 'ca-key.pem', 'kafka-broker-key.pem']:
    try:
        os.remove(f'{CERTS_DIR}/{f}')
    except FileNotFoundError:
        pass

print(f"\nKafka TLS certificates generated successfully in {CERTS_DIR}")
print(f"Password: {PASSWORD}")
