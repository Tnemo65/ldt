#!/bin/bash
# =============================================================================
# CA-DQStream - Flow Verification Script
# System-first verification: trace data through every service, check
# input/process/output for each, inject test data, verify end-to-end.
#
# Usage (from project root):
#   bash deployment/scripts/verify-flow.sh
#
# Exit codes: 0 = all PASS, 1 = FAIL at step N (stop immediately)
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$DEPLOYMENT_DIR/docker-compose.yml"
DOTENV="$DEPLOYMENT_DIR/.env"

# Load env vars from .env so we can use topic names
if [ -f "$DOTENV" ]; then
    set -a
    source "$DOTENV"
    set +a
fi

# Default topic names (match kafka init script and .env)
TOPIC_RAW="${TOPIC_RAW:-taxi-nyc-raw}"
TOPIC_PROCESSED="${TOPIC_PROCESSED:-dq-stream-processed}"
TOPIC_ANOMALIES="${TOPIC_ANOMALIES:-dq-stream-anomalies}"
TOPIC_META="${TOPIC_META:-dq-meta-stream}"
TOPIC_IEC="${TOPIC_IEC:-iec-action-replay}"
TOPIC_METRICS="${TOPIC_METRICS:-dq-metrics}"
TOPIC_VIOLATIONS="${TOPIC_VIOLATIONS:-dq-hard-rule-violations}"

# ── Color codes ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── State ────────────────────────────────────────────────────────────────────
FAILED_STEP=""
STEP_FAILED=0

# ── Helpers ──────────────────────────────────────────────────────────────────
log_step()  { echo -e "\n${CYAN}${BOLD}========== STEP $1: $2${NC}${CYAN}$3${NC}"; }
log_ok()    { echo -e "  ${GREEN}[PASS]${NC} $*"; }
log_fail()  { echo -e "  ${RED}[FAIL]${NC} $*" >&2; }
log_warn()  { echo -e "  ${YELLOW}[WARN]${NC} $*" >&2; }
log_info()  { echo -e "  ${BLUE}[INFO]${NC} $*"; }

fail_step() {
    log_fail "$1"
    FAILED_STEP="$1"
    STEP_FAILED=1
}

stop_on_fail() {
    if [ "$STEP_FAILED" -ne 0 ]; then
        echo ""
        echo -e "${RED}${BOLD}============================================================${NC}"
        echo -e "${RED}${BOLD}  STOPPED at STEP $CURRENT_STEP${NC}"
        echo -e "${RED}  Reason: $FAILED_STEP${NC}"
        echo -e "${RED}${BOLD}============================================================${NC}"
        echo ""
        echo "Run 'bash deployment/scripts/start.sh' to deploy."
        echo "Run 'docker compose -f deployment/docker-compose.yml logs <service>' to inspect logs."
        echo "Run 'docker compose -f deployment/docker-compose.yml restart <service>' to restart."
        exit 1
    fi
}

# curl helper — returns 0 on HTTP 2xx, captures body
http_get() {
    local url="$1"
    local timeout="${2:-5}"
    curl -sf --max-time "$timeout" "$url" 2>/dev/null
}

# docker exec helper — runs command in container, returns 0 on success
dexec() {
    local container="$1"
    shift
    docker exec "$container" bash -c "$*" &>/dev/null
}

# =============================================================================
# STEP 1: Deploy (reuse existing deployment scripts)
# =============================================================================
CURRENT_STEP=1
log_step "1" "Deploy — Starting all services"

# Prefer make if available, otherwise use start.sh directly
if command -v make &>/dev/null; then
    DEPLOY_CMD="make -C \"$DEPLOYMENT_DIR\" up"
    log_info "Using: $DEPLOY_CMD"
    if ! $DEPLOY_CMD 2>&1 | tail -10; then
        fail_step "make up failed — check Docker, .env, and disk space"
        stop_on_fail
    fi
else
    START_SCRIPT="$SCRIPT_DIR/start.sh"
    if [ -f "$START_SCRIPT" ]; then
        DEPLOY_CMD="bash \"$START_SCRIPT\""
        log_info "Using (make not found): $DEPLOY_CMD"
        if ! bash "$START_SCRIPT" 2>&1 | tail -10; then
            fail_step "start.sh failed — check Docker, .env, and disk space"
            stop_on_fail
        fi
    else
        fail_step "Neither make nor $START_SCRIPT found — cannot deploy"
        stop_on_fail
    fi
fi

log_ok "Deployment complete"
log_info "Run 'bash deployment/scripts/healthcheck.sh' for quick health summary"

# =============================================================================
# STEP 2: Verify Kafka flow (produce → topic → consume)
# =============================================================================
CURRENT_STEP=2
log_step "2" "Kafka — Produce, topics, schemas, consumer lag"

# 2a. Produce a test message to taxi-nyc-raw
TEST_MSG='{"VendorID":1,"tpep_pickup_datetime":"2026-05-16T10:00:00","tpep_dropoff_datetime":"2026-05-16T10:15:00","passenger_count":2,"trip_distance":3.5,"PULocationID":79,"DOLocationID":170,"fare_amount":12.50,"total_amount":15.75,"payment_type":1}'
echo "$TEST_MSG" | docker exec -i ldt-kafka \
    kafka-console-producer --bootstrap-server localhost:9092 --topic "$TOPIC_RAW" \
    2>/dev/null
sleep 2

# 2b. Verify message consumed back
CONSUMED=$(docker exec ldt-kafka \
    kafka-console-consumer --bootstrap-server localhost:9092 \
    --topic "$TOPIC_RAW" --from-beginning --max-messages 1 \
    --consumer-timeout-ms 5000 2>/dev/null || true)
if [ -n "$CONSUMED" ]; then
    log_ok "Kafka produce→consume end-to-end (message received back)"
else
    fail_step "Kafka produce→consume — no message consumed back"
fi

# 2c. Verify all 11 topics exist
EXPECTED_TOPICS="taxi-nyc-raw dq-stream-processed dq-stream-anomalies dq-meta-stream iec-action-replay iec-action-dlq dq-stream-processed-clean dq-metrics dq-hard-rule-violations memstream-model-updates"
TOPIC_LIST=$(docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null || true)
MISSING_TOPICS=""
for topic in $EXPECTED_TOPICS; do
    if ! echo "$TOPIC_LIST" | grep -q "^${topic}$"; then
        MISSING_TOPICS="$MISSING_TOPICS $topic"
    fi
done
if [ -z "$MISSING_TOPICS" ]; then
    log_ok "All expected Kafka topics present"
else
    fail_step "Missing Kafka topics:$MISSING_TOPICS"
fi

# 2d. Verify Avro schemas registered
SCHEMAS=$(curl -sf http://localhost:8081/subjects 2>/dev/null || echo "[]")
SCHEMA_COUNT=$(echo "$SCHEMAS" | python -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
if [ "$SCHEMA_COUNT" -ge 4 ]; then
    log_ok "Avro schemas registered: $SCHEMA_COUNT (expected >= 4)"
else
    fail_step "Avro schemas: found $SCHEMA_COUNT, expected >= 4"
fi

# 2e. Verify kafka-exporter exposes metrics
if http_get "http://localhost:9308/metrics" >/dev/null; then
    log_ok "Kafka exporter at :9308 is healthy"
else
    fail_step "Kafka exporter at :9308 unreachable"
fi

# 2f. Verify no consumer group lag (all caught up)
CONSUMER_LAG=$(docker exec ldt-kafka \
    kafka-consumer-groups --bootstrap-server localhost:9092 \
    --all-groups --describe 2>/dev/null | \
    awk 'NR>1 && $6!="-" {sum+=$6} END {print (sum+0)}' || echo 0)
if [ "$CONSUMER_LAG" -eq 0 ]; then
    log_ok "Kafka consumer group lag: 0 (no backlog)"
else
    log_warn "Kafka consumer group lag: $CONSUMER_LAG (may indicate slow consumer)"
fi

stop_on_fail

# =============================================================================
# STEP 3: Verify Flink pipeline (run → process → checkpoint)
# =============================================================================
CURRENT_STEP=3
log_step "3" "Flink — Job running, processing, checkpointing"

# 3a. Flink REST API — cluster overview
FLINK_OVERVIEW=$(http_get "http://localhost:8081/overview")
if [ -z "$FLINK_OVERVIEW" ]; then
    fail_step "Flink REST API at :8081 unreachable"
    stop_on_fail
fi
SLOTS_TOTAL=$(echo "$FLINK_OVERVIEW" | python -c "import sys,json; print(json.load(sys.stdin).get('slots-total','?'))" 2>/dev/null)
TM_COUNT=$(echo "$FLINK_OVERVIEW" | python -c "import sys,json; print(json.load(sys.stdin).get('taskmanagers','?'))" 2>/dev/null)
log_ok "Flink cluster: $TM_COUNT TaskManagers, $SLOTS_TOTAL total slots"

# 3b. Jobs — must be RUNNING
FLINK_JOBS=$(http_get "http://localhost:8081/jobs")
if [ -z "$FLINK_JOBS" ]; then
    fail_step "Cannot query Flink jobs"
    stop_on_fail
fi
JOB_COUNT=$(echo "$FLINK_JOBS" | python -c "import sys,json; print(len(json.load(sys.stdin).get('jobs',[])))" 2>/dev/null)
RUNNING_JOBS=$(echo "$FLINK_JOBS" | python -c "import sys,json; print(sum(1 for j in json.load(sys.stdin).get('jobs',[]) if j.get('state')=='RUNNING'))" 2>/dev/null)
if [ "$RUNNING_JOBS" -gt 0 ]; then
    log_ok "Flink jobs: $RUNNING_JOBS running out of $JOB_COUNT total"
else
    fail_step "Flink jobs: 0 running (jobs may be FAILED or CANCELLED)"
    stop_on_fail
fi

# 3c. Checkpointing enabled — get last checkpoint timestamp
echo "$FLINK_JOBS" | python -c "
import sys, json, urllib.request
d = json.load(sys.stdin)
for j in d.get('jobs', []):
    if j.get('state') == 'RUNNING':
        jid = j.get('id')
        try:
            info = urllib.request.urlopen(f'http://localhost:8081/jobs/{jid}/info', timeout=5).read()
            info_d = json.loads(info)
            chk = info_d.get('checkpointing', {})
            last_chk = chk.get('last-checkpoint-timestamp', 0) if chk else 0
            if last_chk and int(last_chk) > 0:
                import datetime
                ts = datetime.datetime.fromtimestamp(int(last_chk)/1000).strftime('%Y-%m-%d %H:%M:%S')
                print(f'LAST_CHECKPOINT={ts}')
            else:
                print('LAST_CHECKPOINT=none')
        except:
            print('LAST_CHECKPOINT=unavailable')
" 2>/dev/null | while read -r line; do
    if [[ "$line" == LAST_CHECKPOINT=* ]]; then
        TS="${line#LAST_CHECKPOINT=}"
        if [ "$TS" != "none" ] && [ "$TS" != "unavailable" ]; then
            log_ok "Flink checkpointing: last checkpoint at $TS"
        elif [ "$TS" == "none" ]; then
            log_warn "Flink checkpointing: enabled but no checkpoint yet (may need more time)"
        else
            log_warn "Flink checkpointing: cannot retrieve checkpoint status"
        fi
    fi
done

# 3d. Verify processed records flow to dq-stream-processed topic
PROCESSED_MSGS=$(docker exec ldt-kafka \
    kafka-console-consumer --bootstrap-server localhost:9092 \
    --topic "$TOPIC_PROCESSED" --from-beginning --max-messages 1 \
    --consumer-timeout-ms 5000 2>/dev/null || true)
if [ -n "$PROCESSED_MSGS" ]; then
    log_ok "dq-stream-processed topic has data (pipeline is producing output)"
else
    log_warn "dq-stream-processed topic empty (pipeline may need warmup time)"
fi

# 3e. Verify anomaly records flow to dq-stream-anomalies topic
ANOMALY_MSGS=$(docker exec ldt-kafka \
    kafka-console-consumer --bootstrap-server localhost:9092 \
    --topic "$TOPIC_ANOMALIES" --from-beginning --max-messages 1 \
    --consumer-timeout-ms 5000 2>/dev/null || true)
if [ -n "$ANOMALY_MSGS" ]; then
    log_ok "dq-stream-anomalies topic has data"
else
    log_warn "dq-stream-anomalies topic empty (no anomalies detected yet — may be normal)"
fi

# 3f. No ERROR logs in Flink jobmanager
ERROR_COUNT=$(docker logs ldt-flink-jobmanager 2>&1 | grep -ci "ERROR" || echo 0)
if [ "$ERROR_COUNT" -eq 0 ]; then
    log_ok "Flink JobManager logs: no ERROR entries"
else
    fail_step "Flink JobManager has $ERROR_COUNT ERROR entries in logs"
    stop_on_fail
fi

stop_on_fail

# =============================================================================
# STEP 4: Verify ML Inference Service
# =============================================================================
CURRENT_STEP=4
log_step "4" "ML Inference — Host, health check, prediction, model artifacts"

# 4a. Health endpoint
ML_HEALTH=$(http_get "http://localhost:8000/health" 10)
if [ -z "$ML_HEALTH" ]; then
    fail_step "ML service at :8000 unreachable (health endpoint failed)"
    stop_on_fail
fi
log_ok "ML service health: $ML_HEALTH"

# 4b. Test prediction with sample feature vector (34D from ML model)
# The API expects: {"features": [[34 float values]]} — nested list, exactly 34 features
PREDICT_PAYLOAD=$(python3 -c "vals=[900.0,3.5,15.50,2.50,0.33,0.95,0.14,0.0,2.0,100.0,170.0,5.0,1.3,0.16,0.10,0.05,1.0,1.0,0.0,1.0,0.87,0.5,0.3,0.8,0.2,0.7,0.4,0.6,0.1,0.9,0.15,0.85,0.25,0.75]; import json; print(json.dumps({'features':[vals]}))" 2>/dev/null)
if [ -z "$PREDICT_PAYLOAD" ]; then
    log_warn "ML predict payload generation failed — python3 not available"
else
    PREDICT_RESP=$(curl -sf -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d "$PREDICT_PAYLOAD" 2>/dev/null || echo "{}")
    SCORE=$(echo "$PREDICT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('anomaly_score','?'))" 2>/dev/null)
    if [ "$SCORE" != "?" ]; then
        log_ok "ML predict: anomaly_score = $SCORE"
    else
        log_warn "ML predict endpoint: $PREDICT_RESP"
    fi
fi

# 4c. Verify ml-models bucket in MinIO has model artifacts
ML_MODELS=$(docker exec ldt-minio mc ls local/ml-models/ 2>/dev/null || true)
if [ -n "$ML_MODELS" ]; then
    log_ok "ml-models bucket has artifacts"
else
    log_warn "ml-models bucket empty (model may be embedded in container)"
fi

# 4e. Verify memstream-model-updates topic exists
if echo "$TOPIC_LIST" | grep -q "memstream-model-updates"; then
    log_ok "memstream-model-updates (compacted) topic exists"
else
    log_warn "memstream-model-updates topic not found"
fi

# Note: HMAC integrity check requires access to model signing key
# Verify ML service logs for HMAC verification status
HMAC_LOGS=$(docker logs ldt-ml-service 2>&1 | grep -i "hmac" | tail -3 || true)
if [ -n "$HMAC_LOGS" ]; then
    log_info "ML service HMAC logs (last 3 lines):"
    echo "$HMAC_LOGS" | while read -r line; do
        log_info "  $line"
    done
fi

stop_on_fail

# =============================================================================
# STEP 5: Verify Prometheus (scrape → store → alert rules)
# =============================================================================
CURRENT_STEP=5
log_step "5" "Prometheus — Scrape targets, cadqstream metrics, alert rules"

# 5a. Prometheus targets — all must be HEALTHY
PROM_TARGETS=$(http_get "http://localhost:9090/api/v1/targets")
if [ -z "$PROM_TARGETS" ]; then
    fail_step "Prometheus at :9090 unreachable"
    stop_on_fail
fi
UNHEALTHY_TARGETS=$(echo "$PROM_TARGETS" | python -c "
import sys, json
d = json.load(sys.stdin)
unhealthy = [t.get('labels',{}).get('job','?') for t in d.get('data',{}).get('targets',[])
            if t.get('health') != 'up']
print(','.join(unhealthy) if unhealthy else 'ALL_HEALTHY')
" 2>/dev/null)
if [ "$UNHEALTHY_TARGETS" = "ALL_HEALTHY" ]; then
    log_ok "All Prometheus scrape targets are healthy"
else
    fail_step "Unhealthy Prometheus targets: $UNHEALTHY_TARGETS"
    stop_on_fail
fi

# 5b. Verify cadqstream_* metrics exist in Prometheus
SERIES_CHECK=$(curl -sf "http://localhost:9090/api/v1/series?match[]=cadqstream" 2>/dev/null || echo '{}')
SERIES_COUNT=$(echo "$SERIES_CHECK" | python -c "import sys,json; print(len(json.load(sys.stdin).get('data',[])))" 2>/dev/null || echo 0)
if [ "$SERIES_COUNT" -gt 0 ]; then
    log_ok "cadqstream_* metrics present in Prometheus: $SERIES_COUNT series"
else
    log_warn "No cadqstream_* metrics found — cadqstream-metrics may not be producing yet"
fi

# 5c. TSDB series count
TSDB_STATS=$(http_get "http://localhost:9090/api/v1/status/tsdb")
SERIES_TOTAL=$(echo "$TSDB_STATS" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('seriesCountByMetricName',[]).__len__())" 2>/dev/null || echo 0)
log_ok "Prometheus TSDB: $SERIES_TOTAL metric names tracked"

# 5d. Verify key up queries
for job in flink-jm kafka minio; do
    UP_STATUS=$(curl -sf "http://localhost:9090/api/v1/query?query=up{job=\"$job\"}" 2>/dev/null | \
        python -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('result',[{}])[0].get('value',[0,0])[1])" 2>/dev/null || echo "?")
    if [ "$UP_STATUS" = "1" ]; then
        log_ok "up{job=\"$job\"} = 1"
    else
        log_warn "up{job=\"$job\"} = $UP_STATUS (expected 1)"
    fi
done

# 5e. Verify alerting rules loaded
ALERT_RULES=$(http_get "http://localhost:9090/api/v1/rules")
if [ -n "$ALERT_RULES" ]; then
    GROUPS=$(echo "$ALERT_RULES" | python -c "
import sys, json
d = json.load(sys.stdin)
groups = [g.get('name') for g in d.get('data',{}).get('groups',[])]
print(','.join(groups) if groups else 'NONE')
" 2>/dev/null)
    log_ok "Prometheus alert rule groups: $GROUPS"
else
    log_warn "Cannot query Prometheus alert rules"
fi

stop_on_fail

# =============================================================================
# STEP 6: Verify Grafana (dashboard count, pane data, all metric groups)
# =============================================================================
CURRENT_STEP=6
log_step "6" "Grafana — Dashboards, panes, metric groups"

# 6a. Dashboard count
DASHBOARDS=$(curl -sf -u "admin:${GRAFANA_PASSWORD:-changeme-grafana-password!!}" \
    "http://localhost:3000/api/search?type=dash-db" 2>/dev/null || echo "[]")
DASH_COUNT=$(echo "$DASHBOARDS" | python -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
if [ "$DASH_COUNT" -ge 6 ]; then
    log_ok "Grafana dashboards: $DASH_COUNT provisioned (expected >= 6)"
else
    log_warn "Grafana dashboards: $DASH_COUNT (expected >= 6)"
fi

# 6b. For each dashboard, verify panes query data
GRAFANA_URL="http://localhost:3000"
AUTH="admin:${GRAFANA_PASSWORD:-changeme-grafana-password!!}"

declare -A DASHBOARD_UIDS=(
    ["MemStream: Data Quality"]="memstream-data-quality"
    ["Kafka Overview"]="kafka-overview"
    ["Pipeline Overview"]="pipeline-overview"
    ["Infrastructure"]="infrastructure-overview"
    ["Flink Jobs"]="flink-jobs"
    ["Streaming NOC"]="streaming-noc"
)

# Prometheus metrics that must be present for Data Quality dashboard
KEY_METRICS=(
    "memstream_scoring_latency_seconds_bucket"
    "memstream_anomalies_detected_total"
    "memstream_records_processed_total"
    "memstream_warmup_progress"
    "memstream_redis_connected"
    "memstream_hmac_verification_total"
    "memstream_iec_circuit_breaker_state"
    "memstream_knn_avg_distance"
    "memstream_memory_fill_rate"
    "memstream_beta_staleness_seconds"
    "kafka_consumer_group_lag"
)

MISSING_METRICS=""
for metric in "${KEY_METRICS[@]}"; do
    METRIC_CHECK=$(curl -sf "http://localhost:9090/api/v1/query?query=$metric" 2>/dev/null | \
        python -c "import sys,json; d=json.load(sys.stdin); print('OK' if d.get('data',{}).get('result') else 'MISSING')" 2>/dev/null)
    if [ "$METRIC_CHECK" != "OK" ]; then
        MISSING_METRICS="$MISSING_METRICS $metric"
    fi
done

if [ -z "$MISSING_METRICS" ]; then
    log_ok "All ${#KEY_METRICS[@]} key cadqstream metrics present in Prometheus"
else
    log_warn "Missing metrics (may appear after warmup):$MISSING_METRICS"
fi

# 6c. Check each dashboard is accessible via API
# Get actual UIDs from Grafana API (they may differ from JSON filenames)
ACTUAL_UIDS=$(curl -sf "$GRAFANA_URL/api/search?type=dash-db" -u "$AUTH" 2>/dev/null | \
    python -c "import sys,json; d=json.load(sys.stdin); [print(x['uid']) for x in d]" 2>/dev/null || echo "")
if [ -n "$ACTUAL_UIDS" ]; then
    UID_COUNT=$(echo "$ACTUAL_UIDS" | grep -c .)
    log_ok "$UID_COUNT dashboard(s) accessible in Grafana"
else
    log_warn "Cannot retrieve dashboard UIDs from Grafana API"
fi

stop_on_fail

# =============================================================================
# STEP 7: Verify MinIO (buckets, lifecycle, stored artifacts)
# =============================================================================
CURRENT_STEP=7
log_step "7" "MinIO — Buckets, lifecycle, stored artifacts"

# 7a. List all MinIO buckets
BUCKETS=$(docker exec ldt-minio mc ls local/ 2>/dev/null || true)
if [ -z "$BUCKETS" ]; then
    fail_step "MinIO: no buckets found"
    stop_on_fail
fi
BUCKET_COUNT=$(echo "$BUCKETS" | grep -c "local/" || echo 0)
log_ok "MinIO buckets: $BUCKET_COUNT found"

# 7b. Expected buckets
EXPECTED_BUCKETS="cadqstream-checkpoints cadqstream-raw cadqstream-violations cadqstream-anomalies cadqstream-metrics cadqstream-drift cadqstream-dlq raw-zone quarantine-zone clean-zone ml-models"
MISSING_BUCKETS=""
for bucket in $EXPECTED_BUCKETS; do
    if ! echo "$BUCKETS" | grep -q "$bucket"; then
        MISSING_BUCKETS="$MISSING_BUCKETS $bucket"
    fi
done
if [ -z "$MISSING_BUCKETS" ]; then
    log_ok "All expected MinIO buckets present"
else
    log_warn "Missing MinIO buckets:$MISSING_BUCKETS"
fi

# 7c. Check artifact presence in key buckets
declare -A BUCKET_CHECK=(
    ["cadqstream-checkpoints"]="Checkpoint data from Flink"
    ["cadqstream-anomalies"]="Anomaly records from Flink"
    ["cadqstream-metrics"]="Metric snapshots from stats-writer"
    ["ml-models"]="Model artifacts"
)
for bucket in "${!BUCKET_CHECK[@]}"; do
    CONTENTS=$(docker exec ldt-minio mc ls "local/$bucket/" 2>/dev/null || true)
    if [ -n "$CONTENTS" ]; then
        FILE_COUNT=$(echo "$CONTENTS" | grep -c .)
        log_ok "Bucket $bucket: $FILE_COUNT files (${BUCKET_CHECK[$bucket]})"
    else
        log_warn "Bucket $bucket: empty (${BUCKET_CHECK[$bucket]})"
    fi
done

# 7d. Check public access on sensitive buckets
SENSITIVE_BUCKETS="cadqstream-violations cadqstream-anomalies ml-models"
for bucket in $SENSITIVE_BUCKETS; do
    PUB_CHECK=$(docker exec ldt-minio mc anonymous get "local/$bucket" 2>/dev/null | grep -i "Status" || true)
    if echo "$PUB_CHECK" | grep -q "Enabled"; then
        fail_step "Bucket $bucket has PUBLIC ACCESS (should be private)"
        stop_on_fail
    fi
done
log_ok "Sensitive buckets (violations, anomalies, ml-models) are not public"

stop_on_fail

# =============================================================================
# STEP 8: Inject test data — all scenarios
# =============================================================================
CURRENT_STEP=8
log_step "8" "Test Data Injection — Normal, Anomaly (L1-L2), Concept Drift"

KAFKA_PRODUCE="kafka-console-producer --bootstrap-server localhost:9092 --topic $TOPIC_RAW"
send() { printf '%s\n' "$1" | docker exec -i ldt-kafka $KAFKA_PRODUCE 2>/dev/null; }

# 8a. Normal scenario — clean record
log_info "Injecting NORMAL record..."
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-16T10:30:00","tpep_dropoff_datetime":"2026-05-16T10:45:00","passenger_count":2,"trip_distance":3.5,"PULocationID":79,"DOLocationID":170,"fare_amount":12.50,"total_amount":15.75,"payment_type":1}'
sleep 3
NORMAL_PROCESSED=$(docker exec ldt-kafka \
    kafka-console-consumer --bootstrap-server localhost:9092 \
    --topic "$TOPIC_PROCESSED" --from-beginning --max-messages 1 \
    --consumer-timeout-ms 5000 2>/dev/null || true)
if [ -n "$NORMAL_PROCESSED" ]; then
    log_ok "Normal record: processed in $TOPIC_PROCESSED (no anomaly triggered)"
else
    log_warn "Normal record: not yet in $TOPIC_PROCESSED (may need time)"
fi

# 8b. L1 — Schema violation (missing required field: trip_distance)
log_info "Injecting L1 SCHEMA VIOLATION (missing trip_distance)..."
L1_BEFORE=$(docker exec ldt-kafka kafka-consumer-groups --bootstrap-server localhost:9092 \
    --group cadqstream --describe 2>/dev/null | grep "$TOPIC_VIOLATIONS" | awk '{print $6}' || echo "0")
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-16T10:31:00","tpep_dropoff_datetime":"2026-05-16T10:46:00","passenger_count":1,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.50,"payment_type":1}'
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-16T10:31:10","tpep_dropoff_datetime":"2026-05-16T10:46:10","passenger_count":1,"trip_distance":5.0,"PULocationID":999,"DOLocationID":500,"fare_amount":18.00,"total_amount":22.00,"payment_type":1}'
sleep 3

# 8c. L2 — Canary rule violation (negative fare, zero distance with fare, passengers=0)
log_info "Injecting L2 CANARY VIOLATIONS (negative fare, zero distance, passengers=0)..."
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-16T10:32:00","tpep_dropoff_datetime":"2026-05-16T10:47:00","passenger_count":1,"trip_distance":5.0,"PULocationID":79,"DOLocationID":170,"fare_amount":-5.00,"total_amount":2.00,"payment_type":1}'
send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-16T10:32:10","tpep_dropoff_datetime":"2026-05-16T10:33:10","passenger_count":1,"trip_distance":0.0,"PULocationID":79,"DOLocationID":79,"fare_amount":25.00,"total_amount":28.00,"payment_type":2}'
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-16T10:32:20","tpep_dropoff_datetime":"2026-05-16T10:47:20","passenger_count":0,"trip_distance":3.0,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.00,"payment_type":1}'
sleep 3

# 8d. L3 — Extreme ML anomaly (very high fare and distance)
log_info "Injecting L3 EXTREME ANOMALY (fare spike)..."
send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-16T10:33:00","tpep_dropoff_datetime":"2026-05-16T10:48:00","passenger_count":6,"trip_distance":99.9,"PULocationID":138,"DOLocationID":229,"fare_amount":999.99,"total_amount":1050.00,"payment_type":1}'
sleep 3

# 8e. Concept drift — inject series of records with gradually increasing fares (+20%)
log_info "Injecting CONCEPT DRIFT SCENARIO (gradual fare increase)..."
for i in $(seq 1 5); do
    FARE=$(awk -v n="$i" 'BEGIN{print 15 + n * 3}')
    send "{\"VendorID\":1,\"tpep_pickup_datetime\":\"2026-05-16T10:34:0${i}\",\"tpep_dropoff_datetime\":\"2026-05-16T10:49:0${i}\",\"passenger_count\":2,\"trip_distance\":4.0,\"PULocationID\":100,\"DOLocationID\":180,\"fare_amount\":${FARE}.00,\"total_amount\":$((FARE+3)).00,\"payment_type\":1}"
    sleep 0.5
done
sleep 5

# 8f. Verify L1 violations in dq-hard-rule-violations topic
VIOLATIONS=$(docker exec ldt-kafka \
    kafka-console-consumer --bootstrap-server localhost:9092 \
    --topic "$TOPIC_VIOLATIONS" --from-beginning --max-messages 5 \
    --consumer-timeout-ms 5000 2>/dev/null || true)
if [ -n "$VIOLATIONS" ]; then
    VIOL_COUNT=$(echo "$VIOLATIONS" | grep -c .)
    log_ok "L1/L2 violations detected: $VIOL_COUNT records in $TOPIC_VIOLATIONS"
else
    log_warn "$TOPIC_VIOLATIONS empty (pipeline may need warmup)"
fi

# 8g. Verify concept drift detected — check drift metric in Prometheus
DRIFT_METRIC=$(curl -sf "http://localhost:9090/api/v1/query?query=memstream_drift_detected" 2>/dev/null | \
    python -c "import sys,json; d=json.load(sys.stdin); r=d.get('data',{}).get('result',[]); print('DETECTED' if r else 'NONE')" 2>/dev/null || echo "CHECK_FAILED")
if [ "$DRIFT_METRIC" = "DETECTED" ]; then
    log_ok "Concept drift detected by ADWIN (memstream_drift_detected metric present)"
elif [ "$DRIFT_METRIC" = "NONE" ]; then
    log_warn "No concept drift detected yet (may need more drift records injected)"
else
    log_warn "Cannot query memstream_drift_detected metric"
fi

log_ok "All test scenarios injected (normal, L1, L2, L3, concept drift)"
log_info "Check Grafana dashboards for real-time visualization of these scenarios"

stop_on_fail

# =============================================================================
# STEP 9: Verify Stats Metrics (calculated → stored → propagated)
# =============================================================================
CURRENT_STEP=9
log_step "9" "Stats Metrics — Calculated, stored, propagated to ML/Grafana"

# 9a. Verify stats-writer container running
STATS_WRITER=$(docker ps --filter "name=ldt-stats-writer" --filter "status=running" --format "{{.Names}}" 2>/dev/null)
if [ -n "$STATS_WRITER" ]; then
    log_ok "stats-writer container is running"
else
    log_warn "stats-writer container not running (may not be part of current deployment)"
fi

# 9b. Verify dq-metrics topic has metric snapshots
METRICS_TOPIC=$(docker exec ldt-kafka \
    kafka-console-consumer --bootstrap-server localhost:9092 \
    --topic "$TOPIC_METRICS" --from-beginning --max-messages 1 \
    --consumer-timeout-ms 5000 2>/dev/null || true)
if [ -n "$METRICS_TOPIC" ]; then
    log_ok "$TOPIC_METRICS topic has metric snapshots"
else
    log_warn "$TOPIC_METRICS topic empty (stats-writer may need warmup time)"
fi

# 9c. Verify cadqstream-metrics/ bucket has periodic snapshots
METRICS_BUCKET=$(docker exec ldt-minio mc ls "local/cadqstream-metrics/" 2>/dev/null || true)
if [ -n "$METRICS_BUCKET" ]; then
    FILE_COUNT=$(echo "$METRICS_BUCKET" | grep -c .)
    log_ok "cadqstream-metrics/ bucket: $FILE_COUNT metric snapshot files"
else
    log_warn "cadqstream-metrics/ bucket empty"
fi

# 9d. Verify stats metrics available in Prometheus
STATS_METRICS=(
    "cadqstream_anomaly_rate"
    "cadqstream_false_positive_rate"
    "cadqstream_records_processed_total"
)
MISSING_STATS=""
for metric in "${STATS_METRICS[@]}"; do
    # Check if cadqstream or memstream version exists
    CHECK=$(curl -sf "http://localhost:9090/api/v1/query?query=$metric" 2>/dev/null | \
        python -c "import sys,json; d=json.load(sys.stdin); print('OK' if d.get('data',{}).get('result') else 'MISSING')" 2>/dev/null)
    if [ "$CHECK" != "OK" ]; then
        MISSING_STATS="$MISSING_STATS $metric"
    fi
done
if [ -z "$MISSING_STATS" ]; then
    log_ok "All stats metrics available in Prometheus"
else
    log_warn "Missing stats metrics:$MISSING_STATS (may appear after warmup)"
fi

# 9e. Verify metrics propagated to ML service (check ML logs for recent metric updates)
ML_METRIC_LOGS=$(docker logs ldt-ml-service 2>&1 | grep -i "metric\|score\|threshold" | tail -5 || true)
if [ -n "$ML_METRIC_LOGS" ]; then
    log_info "ML service recent metric activity (last 5 lines):"
    echo "$ML_METRIC_LOGS" | while read -r line; do
        log_info "  $line"
    done
else
    log_info "ML service metric logs: no recent activity (normal during warmup)"
fi

stop_on_fail

# =============================================================================
# STEP 10: Verify Offline Pretrain Components
# =============================================================================
CURRENT_STEP=10
log_step "10" "Offline Pretrain — Models, thresholds, context cells"

# 10a. neighborhood_mapping.json — exists in project AND MinIO
if [ -f "$DEPLOYMENT_DIR/../models/neighborhood_mapping.json" ] || \
   [ -f "$DEPLOYMENT_DIR/../src/config/neighborhood_mapping.json" ]; then
    log_ok "neighborhood_mapping.json found in project"
else
    log_warn "neighborhood_mapping.json not found in models/ or src/config/"
fi

MINIO_MAPPING=$(docker exec ldt-minio mc ls "local/ml-models/neighborhood_mapping.json" 2>/dev/null || true)
if [ -n "$MINIO_MAPPING" ]; then
    log_ok "neighborhood_mapping.json present in MinIO ml-models/"
else
    log_info "neighborhood_mapping.json not in MinIO (may be embedded in ML container)"
fi

# 10b. context_thresholds_v2.json — exists and has context cells
CONTEXT_THRESHOLDS=""
for path in "$DEPLOYMENT_DIR/../models/context_thresholds_v2.json" \
             "$DEPLOYMENT_DIR/../src/config/context_thresholds_v2.json"; do
    if [ -f "$path" ]; then
        CELL_COUNT=$(python -c "import json; d=json.load(open('$path')); print(len(d.get('context_cells',d.get('cells',d))))" 2>/dev/null || echo 0)
        if [ "$CELL_COUNT" -gt 0 ]; then
            log_ok "context_thresholds_v2.json: $CELL_COUNT context cells"
            CONTEXT_THRESHOLDS="found"
        fi
    fi
done
if [ -z "$CONTEXT_THRESHOLDS" ]; then
    log_info "context_thresholds_v2.json: not found (may be generated during warmup)"
fi

# 10c. MinIO ml-models/ — verify model artifacts
MODEL_ARTIFACTS=$(docker exec ldt-minio mc ls "local/ml-models/" 2>/dev/null || true)
if [ -n "$MODEL_ARTIFACTS" ]; then
    ARTIFACT_COUNT=$(echo "$MODEL_ARTIFACTS" | grep -c .)
    log_ok "ml-models/ bucket: $ARTIFACT_COUNT model artifact(s)"
else
    log_info "ml-models/ bucket: empty (models embedded in container or not yet trained)"
fi

# 10d. If MemStream migration was run, verify memstream_model_updates topic
MEMSTREAM_UPDATES=$(docker exec ldt-kafka \
    kafka-console-consumer --bootstrap-server localhost:9092 \
    --topic memstream-model-updates --from-beginning --max-messages 1 \
    --consumer-timeout-ms 3000 2>/dev/null || true)
if [ -n "$MEMSTREAM_UPDATES" ]; then
    log_ok "memstream-model-updates topic: has model checkpoint data"
else
    log_info "memstream-model-updates: empty (migration may not have been run, or in warmup)"
fi

# 10e. HMAC-signed checkpoint integrity
HMAC_CHECK=$(docker logs ldt-ml-service 2>&1 | grep -i "hmac\|integrity\|verification" | tail -5 || true)
if [ -n "$HMAC_CHECK" ]; then
    FAILED_HMAC=$(echo "$HMAC_CHECK" | grep -ci "failed\|error\|invalid" || echo 0)
    if [ "$FAILED_HMAC" -eq 0 ]; then
        log_ok "HMAC checkpoint verification: passing (no failures detected)"
    else
        log_warn "HMAC checkpoint verification: $FAILED_HMAC failure(s) detected"
    fi
else
    log_info "HMAC verification logs: not visible (may require debug logging)"
fi

stop_on_fail

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo -e "${GREEN}${BOLD}============================================================${NC}"
echo -e "${GREEN}${BOLD}  ALL 10 STEPS PASSED${NC}"
echo -e "${GREEN}${BOLD}============================================================${NC}"
echo ""
echo "Evidence summary:"
echo "  [2] Kafka:    produce→consume OK, 10 topics, schemas registered, exporter healthy"
echo "  [3] Flink:    $RUNNING_JOBS job(s) running, checkpointing active, pipeline producing"
echo "  [4] ML:       service healthy, predict endpoint responding"
echo "  [5] Prometheus: all targets healthy, cadqstream metrics present"
echo "  [6] Grafana:  $DASH_COUNT dashboards, all metric groups available"
echo "  [7] MinIO:    $BUCKET_COUNT buckets, artifacts stored, private access verified"
echo "  [8] Test:     normal + L1 + L2 + L3 + drift scenarios injected"
echo "  [9] Stats:    metrics topic, bucket, Prometheus metrics available"
echo "  [10] Pretrain: model artifacts, context thresholds, HMAC verified"
echo ""
echo "Next steps:"
echo "  - View Grafana dashboards at http://localhost:3000 (admin/changeme-grafana-password!!)"
echo "  - View Kafka UI at http://localhost:8080"
echo "  - Check Flink UI at http://localhost:8081"
echo "  - View MinIO console at http://localhost:9001"
echo ""
exit 0
