#!/bin/bash
# scripts/setup_minio.sh
# Setup MinIO buckets for Flink checkpoints and state
# Spec: Lines 1625-1640 (S3-compatible storage for RocksDB)

set -e

MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin123}"

echo "=== MinIO Bucket Setup ==="
echo "Endpoint: $MINIO_ENDPOINT"
echo ""

# Install MinIO client if not present
if ! command -v mc &> /dev/null; then
    echo "Installing MinIO client..."
    curl -o /tmp/mc https://dl.min.io/client/mc/release/linux-amd64/mc
    chmod +x /tmp/mc
    sudo mv /tmp/mc /usr/local/bin/mc
    echo "✓ MinIO client installed"
fi

# Configure MinIO client
echo "Configuring MinIO client..."
mc alias set local $MINIO_ENDPOINT $MINIO_ACCESS_KEY $MINIO_SECRET_KEY

# Create buckets
echo ""
echo "Creating buckets..."

if mc ls local/cadqstream-checkpoints &> /dev/null; then
    echo "✓ Bucket 'cadqstream-checkpoints' already exists"
else
    mc mb local/cadqstream-checkpoints
    echo "✓ Created bucket 'cadqstream-checkpoints'"
fi

if mc ls local/cadqstream-state &> /dev/null; then
    echo "✓ Bucket 'cadqstream-state' already exists"
else
    mc mb local/cadqstream-state
    echo "✓ Created bucket 'cadqstream-state'"
fi

# Set bucket policies (allow Flink access)
echo ""
echo "Setting bucket policies..."
mc anonymous set download local/cadqstream-checkpoints
mc anonymous set upload local/cadqstream-checkpoints
mc anonymous set download local/cadqstream-state
mc anonymous set upload local/cadqstream-state
echo "✓ Bucket policies configured"

# Verify setup
echo ""
echo "=== Verification ==="
mc ls local/
echo ""
echo "✅ MinIO setup complete!"
echo ""
echo "Next steps:"
echo "1. Export S3 credentials:"
echo "   export AWS_ACCESS_KEY_ID=$MINIO_ACCESS_KEY"
echo "   export AWS_SECRET_ACCESS_KEY=$MINIO_SECRET_KEY"
echo "   export AWS_ENDPOINT=$MINIO_ENDPOINT"
echo ""
echo "2. Start Flink job with S3 filesystem JAR:"
echo "   flink run --jarfile flink-s3-fs-hadoop-*.jar job.py"
