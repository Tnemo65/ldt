# =============================================================================
# CA-DQStream Custom Flink Image
# Based on Apache Flink 1.17.1 with Python 3.10 support, Kafka connector,
# JDBC connector, and all required ML libraries
# =============================================================================

FROM flink:1.17.1

USER root

# ── System Dependencies ───────────────────────────────────────────────────────
RUN apt-get update -qq && apt-get install -y -qq \
    python3 python3-pip python3-dev python3-venv \
    build-essential cmake git curl wget netcat \
    default-jdk-headless \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# ── Python Dependencies ───────────────────────────────────────────────────────
# Installed in ONE layer to minimize image size and avoid dependency conflicts
# cloudpickle version locked to match Flink's bundled version
RUN pip3 install --no-cache-dir --break-system-packages \
    cloudpickle==2.2.1 \
    numpy==1.24.3 \
    grpcio==1.59.0 \
    grpcio-tools==1.59.0 \
    apache-beam==2.48.0 \
    avro-python3==1.10.2 \
    kafka-python==2.0.2 \
    pickle5 \
    pandas==2.0.3 \
    scikit-learn==1.3.2 \
    joblib==1.3.2 \
    mmh3==3.0.0 \
    psycopg2-binary==2.9.12 \
    prometheus-client==0.19.0 \
    python-dotenv==1.0.0 \
    pyyaml==6.0.1 \
    requests==2.31.0 \
    fastapi==0.104.1 \
    2>&1 | tail -20 || true

# ── PyFlink UDF Runner Setup ─────────────────────────────────────────────────
# Extract pyflink.zip to a real filesystem directory (required for UDF runner)
RUN mkdir -p /opt/flink/pyflink_extracted && \
    python3 -c "import zipfile; z=zipfile.ZipFile('/opt/flink/opt/python/pyflink.zip'); z.extractall('/opt/flink/pyflink_extracted')" && \
    chmod +x /opt/flink/pyflink_extracted/pyflink/bin/pyflink-udf-runner.sh

# ── Connector JARs ────────────────────────────────────────────────────────────
# Note: In production, mount these from host or download during build
# JARs should be present in the lib/ directory
COPY flink-connector-kafka-1.17.1.jar /opt/flink/lib/
COPY flink-connector-jdbc-3.1.1-1.17.jar /opt/flink/lib/
COPY kafka-clients-3.5.1.jar /opt/flink/lib/
COPY postgresql-42.6.0.jar /opt/flink/lib/

# ── PyFlink Configuration ─────────────────────────────────────────────────────
RUN sed -i '/^# YARN/a\
python.udf.runner.script: /opt/flink/pyflink_extracted/pyflink/bin/pyflink-udf-runner.sh\
python.client.executor.args: --python /usr/local/bin/python3\
taskmanager.memory.task.off-heap.enabled: true\
' /opt/flink/conf/flink-conf.yaml

# ── Environment Variables ─────────────────────────────────────────────────────
ENV PYTHONPATH=/opt/flink/pyflink_extracted:/opt/flink/opt/python/py4j-0.10.9.7-src.zip:/opt/flink/opt/python/cloudpickle-2.2.0-src.zip
ENV JAVA_HOME=/opt/java/openjdk
ENV FLINK_HOME=/opt/flink
ENV PATH=$PATH:/opt/flink/bin:$JAVA_HOME/bin

# ── Python Symlinks (required by UDF runner) ─────────────────────────────────
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/python3 /usr/local/bin/python

# ── Application Directory ─────────────────────────────────────────────────────
RUN mkdir -p /opt/flink/e2e && \
    mkdir -p /opt/flink/e2e/src && \
    mkdir -p /opt/flink/e2e/models && \
    mkdir -p /opt/flink/e2e/scripts && \
    mkdir -p /opt/flink/e2e/data

# ── Prometheus Metrics Exporter ───────────────────────────────────────────────
# Flink exposes metrics on ports 9248 (JobManager) and 9249 (TaskManager)
# via the built-in PrometheusReporter

USER flink

WORKDIR /opt/flink
