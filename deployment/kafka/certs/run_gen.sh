#!/bin/bash
PASSWORD=changeKafkaSSL123
CERTS_DIR=/certs
mkdir -p "$CERTS_DIR"
cd "$CERTS_DIR"
openssl req -x509 -newkey rsa:4096 -keyout ca-key.pem -out ca-cert.pem -days 365 -nodes -subj /CN=Kafka-CA-DQStream
openssl pkcs12 -export -in ca-cert.pem -inkey ca-key.pem -out ca.p12 -name kafka-ca -passout pass:"$PASSWORD"
keytool -importkeystore -srckeystore ca.p12 -srcstoretype PKCS12 -srcstorepass "$PASSWORD" -destkeystore kafka.truststore.jks -deststoretype JKS -storepass "$PASSWORD" -noprompt
openssl req -newkey rsa:4096 -keyout kafka-broker-key.pem -out kafka-broker.csr -nodes -subj /CN=kafka
openssl x509 -req -in kafka-broker.csr -CA ca-cert.pem -CAkey ca-key.pem -out kafka-broker-cert.pem -days 365 -CAcreateserial
openssl pkcs12 -export -in kafka-broker-cert.pem -inkey kafka-broker-key.pem -out kafka-broker.p12 -name kafka-broker -passout pass:"$PASSWORD"
keytool -importkeystore -srckeystore kafka-broker.p12 -srcstoretype PKCS12 -srcstorepass "$PASSWORD" -destkeystore kafka.keystore.jks -deststoretype JKS -storepass "$PASSWORD" -noprompt
keytool -importcert -alias CARoot -file ca-cert.pem -keystore kafka.keystore.jks -storepass "$PASSWORD" -noprompt
rm -f ca.p12 kafka-broker.p12 kafka-broker.csr ca-key.pem kafka-broker-key.pem ca-cert.srl
echo Done
