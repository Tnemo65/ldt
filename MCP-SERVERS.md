# MCP Servers Setup Guide for CA-DQStream

This document describes how to set up and use MCP (Model Context Protocol) servers in the CA-DQStream project. MCP servers enable AI assistants (Cursor, Claude Desktop, VS Code, etc.) to interact with your streaming infrastructure components: Kafka, PostgreSQL, MinIO, Grafana, Prometheus, and Confluent Cloud.

---

## Table of Contents

- [Quick Setup](#quick-setup)
- [MCP Servers Overview](#mcp-servers-overview)
- [Prerequisites](#prerequisites)
- [Per-Server Setup](#per-server-setup)
  - [1. Confluent MCP Server](#1-confluent-mcp-server)
  - [2. Kafka MCP Server](#2-kafka-mcp-server)
  - [3. PostgreSQL MCP Server](#3-postgresql-mcp-server)
  - [4. MinIO MCP Server](#4-minio-mcp-server)
  - [5. Grafana MCP Server](#5-grafana-mcp-server)
  - [6. Prometheus MCP Server](#6-prometheus-mcp-server)
- [AI Client Configuration](#ai-client-configuration)
  - [Cursor](#cursor)
  - [Claude Desktop](#claude-desktop)
  - [VS Code](#vs-code)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

---

## Quick Setup

All pre-configured MCP JSON files are in `mcp-config/`:

| Client | Config File |
|--------|------------|
| Cursor | `mcp-config/cursor/mcp.json` |
| Claude Desktop | `mcp-config/claude-desktop/config.json` |
| VS Code | `mcp-config/vscode/mcp.json` |

Copy the appropriate file to your AI client's MCP config location (see [AI Client Configuration](#ai-client-configuration) below), then install the required packages:

```bash
# Install uv (required for most servers)
# Windows:
irm https://astral.sh/uv/install.ps1 | iex
# or: pip install uv

# Kafka MCP Server (Go)
# Note: go install github.com/tuannvm/kafka-mcp-server@latest has module version issues.
# Build from source instead:
#   git clone --depth 1 https://github.com/tuannvm/kafka-mcp-server.git %TEMP%\kafka-mcp-server
#   go build -o %USERPROFILE%\go\bin\kafka-mcp-server.exe %TEMP%\kafka-mcp-server\cmd
# Add %USERPROFILE%\go\bin to your PATH

# MinIO MCP Server (Docker only)
# Pull the official image:
#   docker pull quay.io/minio/aistor/mcp-server-aistor:latest
# Note: uv/pip installation NOT available for this server

# All uvx-based servers (postgres-mcp, mcp-grafana, prometheus-mcp-server)
# auto-install on first use via uvx (part of uv package)
```

---

## MCP Servers Overview

| Server | Purpose | Connection | Auth |
|--------|---------|-----------|------|
| **Confluent** | Flink, Kafka, Schema Registry, Connectors (Confluent Cloud) | Confluent Cloud REST API | API Key |
| **Kafka** | Local Kafka broker operations | localhost:9092 | SASL/TLS (optional) |
| **PostgreSQL** | Database health, index tuning, query optimization | localhost:5432 | Password |
| **MinIO** | Object storage operations | localhost:9000 | Access/Secret Key |
| **Grafana** | Dashboards, alerts, datasources | localhost:3000 | Service Account Token |
| **Prometheus** | Metrics querying via PromQL | localhost:9090 | None (local) |

---

## Prerequisites

### Required Tools

1. **uv** (Astral's fast Python package manager)
   ```bash
   pip install uv
   # or on Windows PowerShell:
   irm https://astral.sh/uv/install.ps1 | iex
   ```

2. **Go 1.21+** (for Kafka MCP Server)
   ```bash
   # Download from https://go.dev/dl/
   ```

3. **Docker** (optional, for postgres-mcp Docker method)
   ```bash
   docker --version
   ```

### Infrastructure Services

Start all infrastructure via docker-compose:

```bash
cd C:/proj/ldt
docker-compose up -d
```

Verify services are running:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

Expected containers: `ldt-kafka`, `ldt-postgres`, `ldt-minio`, `ldt-grafana`, `ldt-prometheus`, `ldt-schema-registry`, `ldt-pgbouncer`.

---

## Per-Server Setup

### 1. Confluent MCP Server

**Purpose**: Manage Flink pipelines, Kafka topics, Schema Registry, Connectors, and Confluent Cloud resources via AI.

**Package**: `@confluentinc/mcp-confluent` (already in `../mcp-confluent/`)

**Setup**:

1. Navigate to the MCP Confluent server directory:
   ```bash
   cd C:/proj/ldt/mcp-confluent
   ```

2. Install dependencies and build:
   ```bash
   npm install
   npm run build
   ```

3. Configure credentials. Copy the example env file and edit:
   ```bash
   cp .env.example .env
   ```

4. Set your Confluent Cloud API key and secret in `.env`:
   ```
   CONFLUENT_CLOUD_API_KEY=your_api_key
   CONFLUENT_CLOUD_API_SECRET=your_api_secret
   ```

**Tools Available**: Create/delete Flink statements, manage Kafka topics, read/write schemas, create connectors, query metrics, manage billing.

**Config Note**: Uses YAML config for multi-connection support. See `mcp-confluent/docs/` for full docs.

---

### 2. Kafka MCP Server

**Purpose**: Native Kafka operations — list topics, produce/consume messages, manage consumer groups, cluster health — without Confluent Cloud.

**Package**: [tuannvm/kafka-mcp-server](https://github.com/tuannvm/kafka-mcp-server) (Go, franz-go)

**Setup**:

> Note: The Go module has version mismatch issues with `go install`. Build from source instead.

```bash
# Clone and build
git clone --depth 1 https://github.com/tuannvm/kafka-mcp-server.git %TEMP%\kafka-mcp-server
go build -o %USERPROFILE%\go\bin\kafka-mcp-server.exe %TEMP%\kafka-mcp-server\cmd

# Verify installation
kafka-mcp-server.exe --help
```

**Configuration** (environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_BROKERS` | `localhost:9092` | Kafka broker addresses |
| `KAFKA_CLIENT_ID` | `kafka-mcp-server` | Client identifier |
| `KAFKA_SASL_MECHANISM` | (empty) | SASL auth: `PLAIN`, `SCRAM-SHA-256`, `SCRAM-SHA-512` |
| `KAFKA_TLS_ENABLE` | `false` | Enable TLS |
| `KAFKA_USERNAME` | (empty) | SASL username |
| `KAFKA_PASSWORD` | (empty) | SASL password |

**With SASL/SCRAM Authentication**:
```json
"kafka": {
  "command": "kafka-mcp-server",
  "env": {
    "KAFKA_BROKERS": "localhost:9092",
    "KAFKA_CLIENT_ID": "kafka-mcp-server",
    "KAFKA_SASL_MECHANISM": "SCRAM-SHA-512",
    "KAFKA_USERNAME": "myuser",
    "KAFKA_PASSWORD": "mypassword",
    "KAFKA_TLS_ENABLE": "true"
  }
}
```

**Tools Available**: `list_topics`, `describe_topic`, `create_topic`, `delete_topic`, `produce_message`, `consume_messages`, `list_consumer_groups`, `describe_consumer_group`, `get_cluster_health`.

**AI Prompts**: The server includes pre-configured prompts for common Kafka workflows.

---

### 3. PostgreSQL MCP Server

**Purpose**: Database health analysis, index tuning with industrial-strength algorithms, EXPLAIN plan analysis, hypothetical index simulation, schema intelligence, and safe SQL execution.

**Package**: [crystaldba/postgres-mcp](https://github.com/crystaldba/postgres-mcp) (Python, 2.7K stars)

**Setup**:

**Option A: uvx (recommended)**
```bash
uv tool install crystaldba/postgres-mcp
```

**Option B: pipx**
```bash
pipx install postgres-mcp
```

**Option C: Docker**
```bash
docker pull crystaldba/postgres-mcp
```

**Access Modes**:

| Mode | Description | Use Case |
|------|-------------|----------|
| `--access-mode=unrestricted` | Full read/write, schema changes | Development |
| `--access-mode=restricted` | Read-only, query time limits | Production |

**Recommended Extension Installation** (for full functionality):
```sql
-- Run in your PostgreSQL database
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS hypopg;
```
- `pg_stat_statements`: Enables query execution statistics analysis
- `hypopg`: Enables hypothetical index simulation

**Connection URI**: Uses the same connection as the pipeline — through pgbouncer at `localhost:6432` or direct at `localhost:5432`:

```
# Through pgbouncer (transaction mode, recommended for pipeline)
postgresql://cadqstream:cadqstream123@localhost:6432/dq_pipeline

# Direct to postgres
postgresql://cadqstream:cadqstream123@localhost:5432/dq_pipeline
```

**Tools Available**:
| Tool | Description |
|------|-------------|
| `list_schemas` | List all database schemas |
| `list_objects` | List tables, views, sequences, extensions in a schema |
| `get_object_details` | Table columns, constraints, indexes |
| `execute_sql` | Run SQL (guarded by access mode) |
| `explain_query` | Get execution plan with hypothetical indexes |
| `get_top_queries` | Slowest queries from pg_stat_statements |
| `analyze_workload_indexes` | Workload-based index recommendations |
| `analyze_query_indexes` | Index recommendations for specific queries |
| `analyze_db_health` | Comprehensive health checks |

**Example AI Prompts**:
- "Check the health of my database and identify any issues"
- "What are the slowest queries in my database? How can I speed them up?"
- "Analyze my database workload and suggest indexes to improve performance"
- "Help me optimize this query: SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id WHERE orders.created_at > '2023-01-01'"

---

### 4. MinIO MCP Server

**Purpose**: Object storage operations — bucket management, object upload/download, presigned URLs, metadata retrieval. Useful for managing ML model artifacts, checkpoint data, and pipeline outputs stored in MinIO.

**Package**: [miniohq/mcp-server-aistor](https://github.com/miniohq/mcp-server-aistor) (Go/Docker only)

**Setup**:

```bash
# Pull the official Docker image (no pip/uv package available)
docker pull quay.io/minio/aistor/mcp-server-aistor:latest
```

**Environment Variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO server address |
| `MINIO_ACCESS_KEY` | (required) | Access key |
| `MINIO_SECRET_KEY` | (required) | Secret key |
| `MINIO_SECURE` | `false` | Use HTTPS |

**Available Tools** (25+):
- `list_buckets`, `create_bucket`, `delete_bucket`
- `list_objects`, `upload_object`, `download_object`, `delete_object`
- `get_object_metadata`, `get_presigned_url`
- `copy_object`, `compose_object`
- `bucket_policy_management`
- `set_bucket_versioning`, `get_bucket_versioning`
- `list_multipart_uploads`, `abort_multipart_upload`

**AI Prompts for MLOps**:
- "List all model artifacts in the ml-models bucket"
- "Upload the trained model to MinIO and generate a presigned download URL"
- "Check if checkpoint files exist for run_id XYZ"

---

### 5. Grafana MCP Server

**Purpose**: Query metrics and logs, manage dashboards and alerts, work with datasources (Prometheus, Loki), interact with Grafana Incident and OnCall, and generate deeplinks to Grafana resources.

**Package**: [grafana/mcp-grafana](https://github.com/grafana/mcp-grafana)

**Setup**:

```bash
uvx mcp-grafana --help
```

**Prerequisites**:
- Grafana 9.0 or later
- A Grafana Service Account with an API token

**Creating a Service Account Token**:

1. Open Grafana at `http://localhost:3000` (admin / admin123)
2. Go to **Administration → Service Accounts → Add service account**
3. Name: `mcp-server`, Role: `Admin`
4. Click **Add service account token**, copy the token

**Environment Variables**:

| Variable | Description |
|----------|-------------|
| `GRAFANA_URL` | Grafana instance URL |
| `GRAFANA_SERVICE_ACCOUNT_TOKEN` | Service account API token |

**Tools Available**:
- Dashboards: list, search, get panel data, create, update, duplicate, delete
- Datasources: list, test connectivity, get health status
- Alerting: list rules, create/update silences, get alert groups
- Metrics: query Prometheus datasources
- Logs: query Loki datasources
- Incident/OnCall: manage incidents and on-call schedules
- Management: folder operations, team management

**Example AI Prompts**:
- "Show me the CPU usage dashboard for the last 24 hours"
- "Create an alert when Kafka consumer lag exceeds 10000"
- "Query the error rate from Prometheus for the dq-pipeline service"
- "List all dashboards that contain the word 'Flink'"

---

### 6. Prometheus MCP Server

**Purpose**: Execute PromQL queries, list available metrics, and retrieve server configuration from Prometheus. Particularly useful for AWS Managed Prometheus (AMP) with SigV4 authentication, but also works with local Prometheus.

**Package**: [awslabs/prometheus-mcp-server](https://github.com/awslabs/mcp/tree/main/src/prometheus-mcp-server) (Python)

**Setup**:

```bash
uvx awslabs.prometheus-mcp-server@latest
```

**Prerequisites**:
- Python 3.10+
- AWS credentials (for AWS Managed Prometheus)
- Local Prometheus at `http://localhost:9090`

**Environment Variables**:

| Variable | Description |
|----------|-------------|
| `FASTMCP_LOG_LEVEL` | Log level: `ERROR`, `DEBUG`, `INFO` |
| `AWS_PROFILE` | AWS CLI profile name |
| `AWS_REGION` | AWS region for AMP |

**Configuration Options**:

```json
"prometheus": {
  "command": "uvx",
  "args": [
    "awslabs.prometheus-mcp-server@latest",
    "--url", "http://localhost:9090"
  ],
  "env": {
    "FASTMCP_LOG_LEVEL": "ERROR"
  }
}
```

**Tools Available**:
| Tool | Description |
|------|-------------|
| `get_available_workspaces` | List AMP workspaces in region |
| `execute_query` | Instant PromQL query |
| `execute_range_query` | PromQL range query with time window |
| `list_metrics` | All available metric names |
| `get_server_info` | Server configuration details |

**Example AI Prompts**:
- "Get the Kafka consumer lag metrics from Prometheus"
- "Show me the Flink job manager heap memory usage over the last hour"
- "List all metrics that contain 'kafka' in their name"
- "Query the rate of processed records per second over the last 15 minutes"

---

## AI Client Configuration

### Cursor

1. Open Cursor → `Cmd/Ctrl + Shift + P` → **Cursor Settings**
2. Go to the **MCP** tab
3. Copy the content from `mcp-config/cursor/mcp.json`
4. Paste into Cursor's MCP configuration
5. Replace `<YOUR_GRAFANA_SERVICE_ACCOUNT_TOKEN>` with your actual token
6. Restart Cursor

**Or manually via file**: Edit `C:\Users\<YourUser>\.cursor\mcp.json`

### Claude Desktop

1. Locate the config file:
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Copy the content from `mcp-config/claude-desktop/config.json`
3. Merge into the existing `mcpServers` section
4. Replace `<YOUR_GRAFANA_SERVICE_ACCOUNT_TOKEN>`
5. Restart Claude Desktop

### VS Code

1. Open **Command Palette** (`Cmd/Ctrl + Shift + P`)
2. Search for **MCP** and select **Open MCP Settings (JSON)**
3. Copy the content from `mcp-config/vscode/mcp.json`
4. Replace `<YOUR_GRAFANA_SERVICE_ACCOUNT_TOKEN>`
5. VS Code will auto-reload MCP servers

**Or via workspace settings**: Create `.vscode/mcp.json` in the project root with the config.

---

## Usage Examples

### Development Workflow Example

With all MCP servers configured, you can have AI-powered conversations like:

```
User: "Check if the Flink job is running and show me the latest consumer lag"

AI (via Confluent MCP): → Queries Flink statements API
                      → Checks Kafka consumer group lag

User: "The pipeline seems slow. Can you analyze the database performance?"

AI (via PostgreSQL MCP): → Runs health checks on postgres
                      → Identifies slow queries via pg_stat_statements
                      → Suggests indexes using hypopg simulation

User: "Show me the Grafana dashboard for the Kafka metrics"

AI (via Grafana MCP): → Lists dashboards containing "Kafka"
                    → Retrieves panel data from the dashboard

User: "Upload the updated ML model to MinIO and create a Flink statement to use it"

AI (via MinIO MCP): → Uploads model artifact to MinIO
                   (via Confluent MCP): → Creates new Flink statement referencing the model
```

### Kafka Exploration

```bash
# Via kafka-mcp-server, ask AI:
"List all Kafka topics and show me the message count for each"

"Produce a test message to the dq-stream-raw topic with value 'test_event'"

"Consume the last 10 messages from dq-stream-raw topic"
```

### Database Performance Analysis

```bash
# Via postgres-mcp, ask AI:
"Run analyze_db_health and identify any critical issues"

"Analyze the top 10 slowest queries and suggest indexes"

"Help me write an EXPLAIN plan for this join query and suggest optimizations"
```

---

## Troubleshooting

### Common Issues

#### uvx command not found

```bash
# Install uv first
pip install uv
# Verify
uv --version
```

#### Kafka MCP Server: connection refused

```bash
# Verify Kafka is running
docker ps ldt-kafka
# Check broker is listening
docker exec ldt-kafka kafka-broker-api-versions --bootstrap-server localhost:9092
```

#### PostgreSQL MCP: connection timeout through pgbouncer

```bash
# Try direct connection to postgres (bypass pgbouncer)
# Change DATABASE_URI to: postgresql://cadqstream:cadqstream123@localhost:5432/dq_pipeline
```

#### Grafana MCP: 401 Unauthorized

```bash
# Verify token is valid
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:3000/api/dashboards
# If 401, regenerate token in Grafana UI: Administration → Service Accounts
```

#### MinIO MCP: SSL certificate errors

```bash
# Set MINIO_SECURE=false if using HTTP
# Verify MinIO is accessible
curl http://localhost:9000/minio/health/live
```

#### Prometheus MCP: AWS credentials not found

```bash
# Configure AWS credentials
aws configure --profile prometheus
# Or set environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### Checking MCP Server Status

In Cursor, use the MCP tab in Settings to see server status (green = running, red = error).

For debugging, set `FASTMCP_LOG_LEVEL=DEBUG` in the server's environment variables to see detailed logs.

### Updating MCP Servers

```bash
# uvx-based servers (auto-update with @latest tag)
# Restart the AI client to pick up new versions

# kafka-mcp-server (Go)
go install github.com/tuannvm/kafka-mcp-server@latest

# postgres-mcp
uv tool upgrade postgres-mcp
```

---

## Flink Agents MCP Integration (Advanced)

For integrating MCP servers with Apache Flink Agents (Python/Java), see the [Flink Agents MCP Documentation](https://nightlies.apache.org/flink/flink-agents-docs-main/docs/development/mcp/).

The pattern uses decorators to declare MCP server connections:

```python
class ReviewAnalysisAgent(Agent):

    @mcp_server
    @staticmethod
    def my_mcp_server() -> ResourceDescriptor:
        return ResourceDescriptor(
            clazz=ResourceName.MCP_SERVER,
            endpoint="http://127.0.0.1:8000/mcp",
            headers={"Authorization": "Bearer your-token"}
        )
```

This allows Flink Agents to use MCP tools as native agent capabilities.

---

## References

- [Apache Flink SQL Demo](https://flink.apache.org/2020/07/28/flink-sql-demo-building-an-end-to-end-streaming-application/) — Flink SQL e2e streaming with Kafka, MySQL, Elasticsearch
- [Flink Agents MCP](https://nightlies.apache.org/flink/flink-agents-docs-main/docs/development/mcp/) — MCP in Flink Agents (Python/Java)
- [Confluent Streaming Agents](https://github.com/confluentinc/quickstart-streaming-agents) — Event-driven agents on Flink & Kafka
- [Kafka MCP Server](https://github.com/tuannvm/kafka-mcp-server) — Native Kafka MCP (Go/franz-go)
- [PostgreSQL MCP](https://github.com/crystaldba/postgres-mcp) — Postgres with index tuning
- [MinIO MCP Server](https://github.com/miniohq/mcp-server-aistor) — Object storage MCP
- [Grafana MCP](https://grafana.com/docs/grafana/latest/developer-resources/mcp/) — Grafana integration
- [Prometheus MCP](https://awslabs.github.io/mcp/servers/prometheus-mcp-server) — AWS Managed Prometheus
