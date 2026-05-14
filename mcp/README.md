# MCP Servers - CA-DQStream

This file lists all configured MCP servers. When you need to use an MCP, check this file to find the matching server.

## Servers

### 1. Confluent (`confluent`)
**Purpose:** Manage Kafka on Confluent Cloud (topics, messages, schemas, connectors)
**Connection:** Confluent Cloud API
**Env vars:** `CONFLUENT_API_KEY`, `CONFLUENT_API_SECRET`

**Use when prompt contains:**
- "topic", "create topic", "delete topic", "list topics"
- "produce", "send message", "publish to Kafka"
- "consume", "read messages", "read from Kafka"
- "schema", "Avro schema", "schema registry"
- "Kafka", "Confluent"

---

### 2. Kafka (`kafka`)
**Purpose:** Local Kafka - topic/message operations
**Connection:** localhost:9092
**Env vars:** `KAFKA_BROKERS=localhost:9092`

**Use when prompt contains:**
- "local Kafka", "localhost Kafka"
- "local topic", "test Kafka"

---

### 3. Grafana (`grafana`)
**Purpose:** Dashboards, alerting, Explore logs/traces, datasource queries
**Connection:** localhost:3000
**Env vars:** `GRAFANA_URL`, `GRAFANA_SERVICE_ACCOUNT_TOKEN`

**Use when prompt contains:**
- "dashboard", "chart", "graph"
- "alert", "alerting"
- "logs", "log queries", "Explore"
- "traces", "tracing"
- "metrics", "PromQL"
- "Grafana"

---

### 4. Prometheus (`prometheus`)
**Purpose:** Metrics queries and metadata
**Connection:** Prometheus server (usually localhost:9090)
**Env vars:** `FASTMCP_LOG_LEVEL`

**Use when prompt contains:**
- "Prometheus", "metrics", "PromQL"
- "query metrics", "metric metadata"
- "explore labels", "label values"

---

### 5. MinIO (`minio`)
**Purpose:** Object storage (S3-compatible), file operations
**Connection:** localhost:9000
**Env vars:** `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`

**Use when prompt contains:**
- "MinIO", "S3", "object storage"
- "upload file", "download file", "list buckets"
- "bucket", "data lake"

---

## Config Locations

### Cursor
`mcp/cursor.json`

### Claude Desktop
`mcp/claude-desktop.json`

### VS Code
`mcp/vscode.json`

## Setup
1. Copy `.env.example` to `.env`
2. Fill in all credentials
3. Restart IDE to apply changes
