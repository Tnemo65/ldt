# =============================================================================
# CA-DQStream Custom Flink Image
# Based on Apache Flink 1.18.1 with Python 3.10 support, Kafka connector,
# and all required ML libraries. MinIO used for all persistent storage.
# Build context: C:\proj\ldt (project root)
# Run from project root: docker build -t ldt-flink:1.18.1-py -f deployment/flink/Dockerfile .
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
COPY deployment/flink/flink-connector-kafka-1.17.1.jar /opt/flink/lib/
COPY deployment/flink/kafka-clients-3.5.1.jar /opt/flink/lib/
# S3/S3A filesystem for MinIO lakehouse writes
COPY deployment/flink/flink-s3-fs-hadoop-1.17.1.jar /opt/flink/lib/

# ── Python Dependencies ────────────────────────────────────────────────────
# All in one RUN so venv activation persists across all pip installs
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip wheel && \
    /opt/venv/bin/pip install "setuptools<61.0" && \
    /opt/venv/bin/pip install \
    cloudpickle==2.2.1 \
    numpy==1.24.3 \
    grpcio==1.59.0 \
    grpcio-tools==1.59.0 \
    apache-beam==2.48.0 \
    avro-python3==1.10.2 \
    kafka-python==2.0.2 \
    pickle5 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    joblib==1.3.2 \
    mmh3==3.0.0 \
    prometheus-client==0.19.0 \
    python-dotenv==1.0.0 \
    pyyaml==6.0.1 \
    requests==2.31.0 \
    fastapi==0.104.1 \
    river==0.21.0 \
    redis==5.0.1 \
    pydantic>=2.5.0 \
    boto3 \
    minio \
    torch>=2.0.0


# Verify torch installation
RUN /opt/venv/bin/python3 -c "import torch; assert torch.__version__.startswith('2.')"

# Add venv python to PATH and set PYTHONPATH
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONPATH=/opt/flink/pyflink_extracted:/opt/flink/opt/python/py4j-0.10.9.7-src.zip:/opt/flink/opt/python/cloudpickle-2.2.0-src.zip:/opt/venv/lib/python3.10/site-packages

# ── S3 / Hadoop AWS Dependencies for MinIO ─────────────────────────────────
# The flink-s3-fs-hadoop JAR requires hadoop-aws and aws-java-sdk-bundle.
# Download and place in lib/ so the S3A filesystem is available at runtime.
RUN cd /opt/flink/lib && \
    curl -sL https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar \
        -o aws-java-sdk-bundle-1.12.262.jar && \
    curl -sL https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar \
        -o hadoop-aws-3.3.4.jar && \
    curl -sL https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.3.4/hadoop-common-3.3.4.jar \
        -o hadoop-common-3.3.4.jar

# ── PyFlink Configuration ───────────────────────────────────────────────────
RUN sed -i '/^# YARN/a\
python.udf.runner.script: /opt/flink/pyflink_extracted/pyflink/bin/pyflink-udf-runner.sh\
python.client.executor.args: --python /opt/venv/bin/python3\
taskmanager.memory.task.off-heap.enabled: true\
' /opt/flink/conf/flink-conf.yaml

# ── Python Symlinks ─────────────────────────────────────────────────────────
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/python3 /usr/local/bin/python

# ── Copy Application Source ─────────────────────────────────────────────────
COPY src/ /opt/flink/e2e/src/

# ── Application Directory ────────────────────────────────────────────────────
RUN mkdir -p /opt/flink/e2e/models /opt/flink/e2e/scripts /opt/flink/e2e/data /opt/flink/e2e /opt/flink/checkpoints /opt/flink/savepoints /models && \
    chown -R flink:flink /opt/flink/e2e /opt/flink/checkpoints /opt/flink/savepoints /models

# ── Non-root user ──────────────────────────────────────────────────────────
USER flink

WORKDIR /opt/flink
