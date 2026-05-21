# =============================================================================
# CA-DQStream Custom Flink Image
# Based on Apache Flink 1.18.1 with Python 3.10 support, Kafka connector,
# and all required ML libraries. MinIO used for all persistent storage.
# Build context: C:\proj\ldt (project root)
# Run from project root: docker build -t ldt-flink:1.18.1-py -f deployment/flink/Dockerfile .
#
# Layer-cache strategy:
#   1. System deps & PyFlink setup  → rarely change
#   2. Connector JARs               → moderate change
#   3. requirements.txt → pip install → MOSTLY STATIC  (cache invalidated only on version bump)
#   4. COPY src/                   → EVERY code change
# =============================================================================

FROM flink:1.18.1-scala_2.12-java11

USER root

# ── System Dependencies ─────────────────────────────────────────────────────
RUN apt-get update -qq && apt-get install -y -qq \
    python3 python3-pip python3-dev python3-venv \
    build-essential cmake git curl wget netcat \
    default-jdk-headless \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# ── PyFlink UDF Runner Setup ───────────────────────────────────────────────
RUN mkdir -p /opt/flink/pyflink_extracted && \
    python3 -c "import zipfile; z=zipfile.ZipFile('/opt/flink/opt/python/pyflink.zip'); z.extractall('/opt/flink/pyflink_extracted')" && \
    chmod +x /opt/flink/pyflink_extracted/pyflink/bin/pyflink-udf-runner.sh

# ── Connector JARs ─────────────────────────────────────────────────────────
# Download Kafka connector from Maven Central — no local files needed.
# Version strategy: connector 3.1.0 is stable for Flink 1.18.x
# NOTE: flink-s3-fs-hadoop and hadoop-aws/aws-sdk are ALREADY included in
# the base flink:1.18.1-scala_2.12-java11 image. Do NOT add duplicates
# here (would cause "multiple delegation token providers" error).
RUN cd /opt/flink/lib && \
    curl -sL https://repo1.maven.org/maven2/org/apache/flink/flink-connector-kafka/3.1.0-1.18/flink-connector-kafka-3.1.0-1.18.jar \
        -o flink-connector-kafka-3.1.0-1.18.jar && \
    curl -sL https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.5.1/kafka-clients-3.5.1.jar \
        -o kafka-clients-3.5.1.jar

# ── Python Dependencies ────────────────────────────────────────────────────
# Step 1: copy requirements.txt only → this layer is invalidated ONLY on dep version bump
COPY deployment/flink/requirements.txt /app/requirements.txt

# Step 2: install in venv → CACHED forever unless requirements.txt changes
# CPU-only PyTorch: avoid 2GB CUDA packages in Docker container
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip wheel && \
    /opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt --ignore-installed torch

# ── Environment ────────────────────────────────────────────────────────────
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONPATH=/opt/flink/pyflink_extracted:/opt/flink/opt/python/py4j-0.10.9.7-src.zip:/opt/flink/opt/python/cloudpickle-2.2.0-src.zip:/opt/venv/lib/python3.10/site-packages

# ── PyFlink Configuration ───────────────────────────────────────────────────
RUN sed -i '/^# YARN/a\
python.udf.runner.script: /opt/flink/pyflink_extracted/pyflink/bin/pyflink-udf-runner.sh\
python.client.executor.args: --python /opt/venv/bin/python3\
taskmanager.memory.task.off-heap.enabled: true\
taskmanager.memory.task.off-heap.size: 1024m\
' /opt/flink/conf/flink-conf.yaml

# ── Python Symlinks ─────────────────────────────────────────────────────────
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/python3 /usr/local/bin/python

# ── Application Directory ────────────────────────────────────────────────────
RUN mkdir -p /opt/flink/e2e/models /opt/flink/e2e/scripts /opt/flink/e2e/data \
    /opt/flink/e2e /opt/flink/checkpoints /opt/flink/savepoints /models && \
    chown -R flink:flink /opt/flink/e2e /opt/flink/checkpoints /opt/flink/savepoints /models

# ── Copy Application Source (LAST — changes most frequently) ────────────────
# This layer is invalidated on every code edit; all above layers remain cached.
COPY src/ /opt/flink/e2e/src/

# ── Non-root user ──────────────────────────────────────────────────────────
USER flink

WORKDIR /opt/flink
