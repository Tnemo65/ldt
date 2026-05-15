#!/bin/bash
set -e

MC_ALIAS="${MC_ALIAS:-local}"
MC_HOST="${MC_HOST:-http://minio:9000}"
MC_USER="${MC_USER:-${MC_MINIO_USER}}"
MC_PASS="${MC_PASS:-${MC_MINIO_PASSWORD}}"

echo "[minio-init] Waiting for MinIO at ${MC_HOST}..."
until mc alias set "${MC_ALIAS}" "${MC_HOST}" "${MC_USER}" "${MC_PASS}" > /dev/null 2>&1; do
    echo "[minio-init] MinIO not ready, waiting 5s..."
    sleep 5
done
echo "[minio-init] MinIO alias set successfully!"

# ── SECURITY FIRST: Remove public access from all buckets ───────────────────
# REMOVE PUBLIC ACCESS from cadqstream-checkpoints (CRITICAL — Flink state contains raw trip data)
mc anonymous set none "${MC_ALIAS}/cadqstream-checkpoints" 2>/dev/null || true

# REMOVE PUBLIC ACCESS from cadqstream-metrics (contains trip-level anomaly scores)
mc anonymous set none "${MC_ALIAS}/cadqstream-metrics" 2>/dev/null || true

# ALL OTHER BUCKETS: no public access by default (enforced at creation)

echo "[minio-init] Creating buckets..."

# ── Flink checkpoints + MLflow artifacts ──
mc mb "${MC_ALIAS}/cadqstream-checkpoints" --ignore-existing

# ── Flink StreamingFileSink buckets (must match minio_sink.py) ──
# cadqstream-raw/        → valid taxi trips
mc mb "${MC_ALIAS}/cadqstream-raw" --ignore-existing

# cadqstream-violations/ → schema_violations + canary_violations
mc mb "${MC_ALIAS}/cadqstream-violations" --ignore-existing

# cadqstream-anomalies/  → anomaly_scores
mc mb "${MC_ALIAS}/cadqstream-anomalies" --ignore-existing

# cadqstream-metrics/   → meta_metrics + pipeline_stats
mc mb "${MC_ALIAS}/cadqstream-metrics" --ignore-existing

# cadqstream-drift/     → drift_events + alerts
mc mb "${MC_ALIAS}/cadqstream-drift" --ignore-existing

# DLQ bucket for action-replay-worker
mc mb "${MC_ALIAS}/cadqstream-dlq" --ignore-existing

# ── Legacy zone buckets (stats-writer, etc.) ──
mc mb "${MC_ALIAS}/raw-zone" --ignore-existing
mc mb "${MC_ALIAS}/quarantine-zone" --ignore-existing
mc mb "${MC_ALIAS}/clean-zone" --ignore-existing

# ── ML platform ──
mc mb "${MC_ALIAS}/ml-models" --ignore-existing

# REMOVED: Public download was a security risk - raw taxi PII was publicly accessible
# mc anonymous set download "${MC_ALIAS}/${bucket}" 2>/dev/null || true

echo "[minio-init] Verifying buckets..."
mc ls "${MC_ALIAS}/"

# ── Bucket Lifecycle Policies ────────────────────────────────────────────────
echo "[minio-init] Applying lifecycle policies..."

# cadqstream-checkpoints: 90-day retention
mc ilm add --expire-days 90 "${MC_ALIAS}/cadqstream-checkpoints" 2>/dev/null || true

# cadqstream-drift: 30-day retention
mc ilm add --expire-days 30 "${MC_ALIAS}/cadqstream-drift" 2>/dev/null || true

# cadqstream-metrics: 30-day retention
mc ilm add --expire-days 30 "${MC_ALIAS}/cadqstream-metrics" 2>/dev/null || true

# ml-models: NO expiration — preserve every model version

# cadqstream-raw: 30-day retention + purge (NYC LL144 compliance)
mc ilm add --expire-days 30 --delete "${MC_ALIAS}/cadqstream-raw" 2>/dev/null || true

# cadqstream-anomalies: 90-day retention (compliance)
mc ilm add --expire-days 90 --delete "${MC_ALIAS}/cadqstream-anomalies" 2>/dev/null || true

echo "[minio-init] Lifecycle policies applied successfully."

echo "[minio-init] MinIO initialization complete!"
