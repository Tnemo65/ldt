#!/bin/bash
cd /certs
openssl req -x509 -newkey rsa:4096 -keyout ca-key.pem -out ca-cert.pem -days 365 -nodes -subj /CN=CA-DQStream-Redis
openssl req -newkey rsa:4096 -keyout redis-key.pem -out redis.csr -nodes -subj /CN=redis
openssl x509 -req -in redis.csr -CA ca-cert.pem -CAkey ca-key.pem -out redis-cert.pem -days 365 -CAcreateserial
rm -f redis.csr ca-key.pem ca-cert.srl
echo Done
ls -la /certs
