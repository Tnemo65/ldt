#!/bin/bash
PASSWORD=changeKafkaSSL123
CERTS_DIR=/certs
mkdir -p 
cd 
/usr/bin/openssl req -x509 -newkey rsa:4096 -keyout ca-key.pem -out ca-cert.pem -days 365 -nodes -subj /CN=Kafka-CA-DQStream
/usr/bin/openssl pkcs12 -export -in ca-cert.pem -inkey ca-key.pem -out ca.p12 -name kafka-ca -passout pass:
keytool -importkeystore -srckeystore ca.p12 -srcstoretype PKCS12 -srcstorepass  -destkeystore kafka.truststore.jks -deststoretype JKS -storepass  -noprompt
/usr/bin/openssl req -newkey rsa:4096 -keyout kafka-broker-key.pem -out kafka-broker.csr -nodes -subj /CN=kafka
/usr/bin/openssl x509 -req -in kafka-broker.csr -CA ca-cert.pem -CAkey ca-key.pem -out kafka-broker-cert.pem -days 365
/usr/bin/openssl pkcs12 -export -in kafka-broker-cert.pem -inkey kafka-broker-key.pem -out kafka-broker.p12 -name kafka-broker -passout pass:
keytool -importkeystore -srckeystore kafka-broker.p12 -srcstoretype PKCS12 -srcstorepass  -destkeystore kafka.keystore.jks -deststoretype JKS -storepass  -noprompt
keytool -importcert -alias CARoot -file ca-cert.pem -keystore kafka.keystore.jks -storepass  -noprompt
rm -f ca.p12 kafka-broker.p12 kafka-broker.csr ca-key.pem kafka-broker-key.pem
ls -la 
