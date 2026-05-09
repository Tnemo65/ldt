#!/bin/bash
# Flink container startup script
set -e

# Extract pyflink.zip if not already extracted
PYFLINK_ZIP="/opt/flink/opt/python/pyflink.zip"
PYFLINK_DIR="/opt/flink/pyflink_extracted"
PYFLINK_UDF_SCRIPT="$PYFLINK_DIR/pyflink/bin/pyflink-udf-runner.sh"

if [ ! -d "$PYFLINK_DIR/pyflink" ]; then
    echo "[init] Extracting pyflink.zip..."
    mkdir -p "$PYFLINK_DIR"
    python3 -c "import zipfile; z=zipfile.ZipFile('$PYFLINK_ZIP'); z.extractall('$PYFLINK_DIR')"
    echo "[init] pyflink extracted"
fi

# Create python symlink
if [ ! -e /usr/bin/python ]; then
    ln -sf /usr/bin/python3 /usr/bin/python
fi

# Add Python UDF config to flink-conf.yaml if not present
FLINK_CONF="/opt/flink/conf/flink-conf.yaml"
if ! grep -q "python.udf.runner.script" "$FLINK_CONF" 2>/dev/null; then
    echo "" >> "$FLINK_CONF"
    echo "python.udf.runner.script: $PYFLINK_UDF_SCRIPT" >> "$FLINK_CONF"
    echo "[init] Added python.udf.runner.script to flink-conf.yaml"
fi

echo "[init] Startup initialization complete"

# Execute the original entrypoint/command
exec "$@"
