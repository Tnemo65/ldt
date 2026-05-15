mc alias set local http://localhost:9000 ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}
mc mb local/cadqstream-raw -p
mc mb local/cadqstream-violations -p
mc mb local/cadqstream-anomalies -p
mc mb local/cadqstream-metrics -p
mc mb local/cadqstream-drift -p
mc mb local/cadqstream-dlq -p
echo "All buckets:"
mc ls local/
