#!/usr/bin/env python3
"""Generate Kafka TLS certificates using subprocess calls."""
import subprocess, os, sys

CERTS_DIR = os.path.dirname(os.path.abspath(__file__)) or "."
PASSWORD = "changeKafkaSSL123"
KEY_PASSWORD = "changeKafkaKey456"

os.makedirs(CERTS_DIR, exist_ok=True)
os.chdir(CERTS_DIR)

def run(cmd):
    print(f"Running: {' '.join(cmd[:3])}...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"STDERR: {r.stderr}")
        print(f"STDOUT: {r.stdout}")
        raise RuntimeError(f"Command failed: {cmd[0]}")
    return r

# Generate CA
run(["openssl", "req", "-x509", "-newkey", "rsa:4096",
     "-keyout", "ca-key.pem", "-out", "ca-cert.pem",
     "-days", "365", "-nodes",
     "-subj", "/CN=Kafka-CA-DQStream"])
print("CA generated")

# CA p12
run(["openssl", "pkcs12", "-export",
     "-in", "ca-cert.pem", "-inkey", "ca-key.pem",
     "-out", "ca.p12", "-name", "kafka-ca",
     "-passout", f"pass:{PASSWORD}"])

# Truststore JKS
run(["keytool", "-importkeystore",
     "-srckeystore", "ca.p12", "-srcstoretype", "PKCS12",
     "-srcstorepass", PASSWORD,
     "-destkeystore", "kafka.truststore.jks", "-deststoretype", "JKS",
     "-storepass", PASSWORD, "-noprompt"])
print("Truststore JKS generated")

# Broker key
run(["openssl", "req", "-newkey", "rsa:4096",
     "-keyout", "kafka-broker-key.pem", "-out", "kafka-broker.csr",
     "-nodes", "-subj", "/CN=kafka"])

# Sign broker cert
run(["openssl", "x509", "-req",
     "-in", "kafka-broker.csr",
     "-CA", "ca-cert.pem", "-CAkey", "ca-key.pem",
     "-out", "kafka-broker-cert.pem", "-days", "365",
     "-CAcreateserial"])
print("Broker cert signed")

# Broker p12
run(["openssl", "pkcs12", "-export",
     "-in", "kafka-broker-cert.pem", "-inkey", "kafka-broker-key.pem",
     "-out", "kafka-broker.p12", "-name", "kafka-broker",
     "-passout", f"pass:{PASSWORD}"])

# Broker keystore JKS
run(["keytool", "-importkeystore",
     "-srckeystore", "kafka-broker.p12", "-srcstoretype", "PKCS12",
     "-srcstorepass", PASSWORD,
     "-destkeystore", "kafka.keystore.jks", "-deststoretype", "JKS",
     "-storepass", PASSWORD, "-noprompt"])

# Import CA into broker keystore
run(["keytool", "-importcert",
     "-alias", "CARoot", "-file", "ca-cert.pem",
     "-keystore", "kafka.keystore.jks",
     "-storepass", PASSWORD, "-noprompt"])
print("Keystore JKS generated")

# Cleanup intermediates
for f in ["ca.p12", "kafka-broker.p12", "kafka-broker.csr",
          "ca-key.pem", "kafka-broker-key.pem", "ca-cert.srl"]:
    try:
        os.remove(os.path.join(CERTS_DIR, f))
    except FileNotFoundError:
        pass

print(f"\nKafka TLS certificates generated in {CERTS_DIR}")
print(f"Password: {PASSWORD}")
