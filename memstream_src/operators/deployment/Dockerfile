# ============================================================
# CA-DQStream Flink Operator Docker Image
# ============================================================
FROM flink:1.18.1-scala_2.12-java11

# Install Python and dependencies for ML operators
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /opt/flink

# Copy custom configuration
COPY flink-conf.yml /opt/flink/conf/flink-conf.yml

# Environment variables
ENV FLINK_ENV_JAVA_OPTS="-Dhttps.protocols=TLSv1.2"
ENV PYTHONPATH="/opt/flink/python:${PYTHONPATH:-}"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Expose ports
EXPOSE 8081 9249 6123

# Default command
CMD ["jobmanager"]
