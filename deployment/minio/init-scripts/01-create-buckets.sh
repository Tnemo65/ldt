#!/bin/bash
set -e

MC_ALIAS="${MC_ALIAS:-local}"
MC_HOST="${MC_HOST:-http://minio:9000}"
MC_USER="${MC_USER:-minioadmin}"
MC_PASS="${MC_PASS:-minioadmin123}"

echo "[minio-init] Waiting for MinIO at ${MC_HOST}..."
until mc alias set "${MC_ALIAS}" "${MC_HOST}" "${MC_USER}" "${MC_PASS}" > /dev/null 2>&1; do
    echo "[minio-init] MinIO not ready, waiting 5s..."
    sleep 5
done
echo "[minio-init] MinIO alias set successfully!"

echo "[minio-init] Creating buckets..."

mc mb "${MC_ALIAS}/ml-models" --ignore-existing
mc mb "${MC_ALIAS}/checkpoints" --ignore-existing
mc mb "${MC_ALIAS}/artifacts" --ignore-existing
mc mb "${MC_ALIAS}/mlflow-artifacts" --ignore-existing

echo "[minio-init] Setting bucket policies..."
mc anonymous set download "${MC_ALIAS}/ml-models"
mc anonymous set download "${MC_ALIAS}/checkpoints"
mc anonymous set download "${MC_ALIAS}/artifacts"
mc anonymous set download "${MC_ALIAS}/mlflow-artifacts"

echo "[minio-init] Verifying buckets..."
mc ls "${MC_ALIAS}/"

echo "[minio-init] MinIO initialization complete!"
