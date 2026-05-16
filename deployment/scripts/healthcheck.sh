#!/bin/bash
# =============================================================================
# CA-DQStream - Health Check Script
# Comprehensive health verification with parallel checks and detailed reporting
# Exit codes: 0 = all healthy, 1 = critical failures, 2 = warnings only
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$DEPLOYMENT_DIR/docker-compose.yml"

# ── Color codes ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# ── Counters ────────────────────────────────────────────────────────────────
CRITICAL_FAILURES=0
WARNINGS=0
SERVICES_CHECKED=0
SERVICES_HEALTHY=0

# ── State files for parallel checks ────────────────────────────────────────
TMPDIR="${TMPDIR:-/tmp}"
CHECK_STATE="$TMPDIR/cadqstream_health_$$"
mkdir -p "$CHECK_STATE"

# Cleanup on exit
trap 'rm -rf "$CHECK_STATE"' EXIT

# ── Logging helpers ──────────────────────────────────────────────────────────
log_ok()    { echo -e "  ${GREEN}✓${NC} $*"; }
log_fail()  { echo -e "  ${RED}✗${NC} $*" >&2; }
log_warn()  { echo -e "  ${YELLOW}⚠${NC} $*" >&2; }
log_info()  { echo -e "  ${BLUE}•${NC} $*"; }
log_section(){ echo -e "\n${CYAN}${BOLD}  $1${NC}"; }

# ── Parallel check runner ────────────────────────────────────────────────────
# Runs a named check in background, saves result to file
# Args: check_name, cmd, timeout
run_check() {
    local name="$1"
    local cmd="$2"
    local timeout="${3:-10}"

    (
        set +e
        local output
        output=$(timeout "$timeout" bash -c "$cmd" 2>&1)
        local exit_code=$?
        echo "$exit_code" > "$CHECK_STATE/${name}.exit"
        echo "$output" > "$CHECK_STATE/${name}.out"
    ) &
    echo $! > "$CHECK_STATE/${name}.pid"
}

# Wait for a named check to complete
wait_check() {
    local name="$1"
    local pid
    pid=$(cat "$CHECK_STATE/${name}.pid" 2>/dev/null || echo "")
    if [ -n "$pid" ]; then
        wait "$pid" 2>/dev/null || true
    fi
}

# Get result of a named check
check_result() {
    local name="$1"
    cat "$CHECK_STATE/${name}.exit" 2>/dev/null || echo "999"
}

check_output() {
    local name="$1"
    cat "$CHECK_STATE/${name}.out" 2>/dev/null || echo ""
}

# =============================================================================
# HEADER
# =============================================================================
echo ""
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}  CA-DQStream Health Report${NC}"
echo -e "${CYAN}${BOLD}  $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"

# =============================================================================
# A. CONTAINER STATUS (all ldt-* containers)
# =============================================================================
log_section "A. Container Status"

CONTAINER_STATS=$(docker ps --filter "name=ldt-" --format "{{.Names}}\t{{.Status}}" 2>/dev/null | sort)
TOTAL_CONTAINERS=$(echo "$CONTAINER_STATS" | grep -c . || echo 0)
RUNNING_CONTAINERS=$(echo "$CONTAINER_STATS" | grep -c "Up " || echo 0)
STOPPED_CONTAINERS=$(echo "$CONTAINER_STATS" | grep -vc "Up " || echo 0)

if [ "$STOPPED_CONTAINERS" -gt 0 ]; then
    echo -e "  ${RED}⚠${NC} $STOPPED_CONTAINERS container(s) not running:"
    echo "$CONTAINER_STATS" | while IFS=$'\t' read -r name status; do
        if [[ ! "$status" =~ ^Up ]]; then
            echo -e "    ${RED}✗${NC} $name: $status"
            CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
        fi
    done
else
    echo -e "  ${GREEN}✓${NC} All $TOTAL_CONTAINERS containers running"
fi

# Show running containers in compact format
if [ -n "$CONTAINER_STATS" ]; then
    echo "$CONTAINER_STATS" | while IFS=$'\t' read -r name status; do
        if [[ "$status" =~ ^Up ]]; then
            echo "    $name"
        fi
    done
fi

# =============================================================================
# B. DOCKER NETWORK INSPECTION
# =============================================================================
log_section "B. Docker Network"

NETWORK_NAME="cadqstream-net"
NETWORK_EXISTS=$(docker network ls --filter "name=$NETWORK_NAME" --format "{{.Name}}" 2>/dev/null | grep -q "^${NETWORK_NAME}$" && echo "yes" || echo "no")

if [ "$NETWORK_EXISTS" = "yes" ]; then
    echo -e "  ${GREEN}✓${NC} Network '$NETWORK_NAME' exists"

    # List containers in the network
    NETWORK_CONTAINERS=$(docker network inspect "$NETWORK_NAME" --format '{{range .Containers}}{{.Name}} {{end}}' 2>/dev/null | tr ' ' '\n' | sort | grep -v '^$')
    NETWORK_CONTAINER_COUNT=$(echo "$NETWORK_CONTAINERS" | grep -c . || echo 0)
    echo -e "  ${GREEN}✓${NC} $NETWORK_CONTAINER_COUNT container(s) in network"

    # Check for expected containers
    for expected in ldt-kafka ldt-zookeeper ldt-flink-jobmanager ldt-minio ldt-redis ldt-prometheus ldt-grafana; do
        if echo "$NETWORK_CONTAINERS" | grep -q "^${expected}$"; then
            :  # ok
        else
            echo -e "  ${YELLOW}⚠${NC} Expected container '$expected' not in '$NETWORK_NAME'"
            WARNINGS=$((WARNINGS + 1))
        fi
    done
else
    echo -e "  ${RED}✗${NC} Network '$NETWORK_NAME' does not exist"
    CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
fi

# =============================================================================
# C. PARALLEL HEALTH CHECKS
# =============================================================================
log_section "C. Service Health Checks"

echo "  Running parallel health checks..."

# Start all checks in parallel
# Format: run_check "name" "command" [timeout]

# ── Critical Infrastructure ──
run_check "zookeeper"   "docker exec ldt-zookeeper bash -c 'echo ruok | nc localhost 2181 | grep -q imok' 2>/dev/null"
run_check "kafka"       "docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null | grep -q ."
run_check "schema-reg"  "curl -sf http://localhost:8081/subjects"
run_check "kafka-ui"    "curl -sf http://localhost:8080"
run_check "kafka-exp"   "curl -sf http://localhost:9308/metrics"

# ── Storage ──
REDIS_PASSWORD="${REDIS_PASSWORD:-redis_password}"
run_check "redis"       "docker exec ldt-redis redis-cli -a \"$REDIS_PASSWORD\" ping 2>/dev/null | grep -q PONG"
run_check "minio"       "docker exec ldt-minio mc ready local 2>/dev/null"

# ── Flink ──
run_check "flink-jm"    "curl -sf http://localhost:8081/overview"
run_check "flink-tm"    "curl -sf http://localhost:8081/taskmanagers 2>/dev/null | grep -q '\"id\"'"
run_check "flink-jobs"  "curl -sf http://localhost:8081/jobs 2>/dev/null"

# ── ML Service ──
run_check "ml-service"  "curl -sf http://localhost:8000/health"

# ── Observability ──
run_check "prometheus"  "wget --spider -q http://localhost:9090/-/healthy"
run_check "grafana"     "curl -sf http://localhost:3000/api/health"
run_check "node-exp"    "curl -sf http://localhost:9100/metrics"
run_check "cadqmetrics" "curl -sf http://localhost:9250/health"

# ── Data Layer ──
run_check "stats-writer"    "docker ps --filter 'name=ldt-stats-writer' --filter 'status=running' --format '{{.Names}}' | grep -q ."
run_check "kafka-producer"  "docker ps --filter 'name=ldt-kafka-producer' --filter 'status=running' --format '{{.Names}}' | grep -q ."
run_check "action-replay"    "docker ps --filter 'name=ldt-action-replay-worker' --filter 'status=running' --format '{{.Names}}' | grep -q ."

# Wait for all parallel checks
ALL_CHECKS=(
    zookeeper kafka schema-reg kafka-ui kafka-exp
    redis minio
    flink-jm flink-tm flink-jobs
    ml-service
    prometheus grafana node-exp cadqmetrics
    stats-writer kafka-producer action-replay
)

echo "  Waiting for checks to complete..."
for check in "${ALL_CHECKS[@]}"; do
    wait_check "$check"
done
echo "  All checks complete."

# =============================================================================
# D. PROCESS AND DISPLAY RESULTS
# =============================================================================

# Service metadata: name, group, critical (yes/no)
declare -A SERVICE_GROUPS=(
    [zookeeper]="Infrastructure|yes"
    [kafka]="Infrastructure|yes"
    [schema-reg]="Infrastructure|yes"
    [kafka-ui]="Infrastructure|no"
    [kafka-exp]="Observability|no"
    [redis]="Storage|yes"
    [minio]="Storage|yes"
    [flink-jm]="Flink|yes"
    [flink-tm]="Flink|yes"
    [flink-jobs]="Flink|no"
    [ml-service]="ML Service|yes"
    [prometheus]="Observability|no"
    [grafana]="Observability|no"
    [node-exp]="Observability|no"
    [cadqmetrics]="Observability|no"
    [stats-writer]="Data Layer|no"
    [kafka-producer]="Data Layer|no"
    [action-replay]="Data Layer|no"
)

# Display each check result
echo ""

HEALTHY_BY_GROUP=""
WARN_BY_GROUP=""
FAIL_BY_GROUP=""

for check in "${ALL_CHECKS[@]}"; do
    SERVICES_CHECKED=$((SERVICES_CHECKED + 1))
    local exit_code
    exit_code=$(check_result "$check")
    local output
    output=$(check_output "$check")

    local metadata="${SERVICE_GROUPS[$check]:-Unknown|yes}"
    local group="${metadata%%|*}"
    local critical="${metadata##*|}"

    local display_name="$check"
    case "$check" in
        zookeeper)   display_name="Zookeeper" ;;
        kafka)       display_name="Kafka" ;;
        schema-reg)  display_name="Schema Reg" ;;
        kafka-ui)    display_name="Kafka UI" ;;
        kafka-exp)   display_name="Kafka Exp" ;;
        redis)       display_name="Redis" ;;
        minio)       display_name="MinIO" ;;
        flink-jm)    display_name="Flink JM" ;;
        flink-tm)    display_name="Flink TM" ;;
        flink-jobs)  display_name="Flink Jobs" ;;
        ml-service)  display_name="ML Service" ;;
        prometheus)  display_name="Prometheus" ;;
        grafana)     display_name="Grafana" ;;
        node-exp)    display_name="Node Exp" ;;
        cadqmetrics) display_name="Metrics" ;;
        stats-writer) display_name="Stats Writer" ;;
        kafka-producer) display_name="Kafka Prod" ;;
        action-replay) display_name="Action Replay" ;;
    esac

    if [ "$exit_code" = "0" ]; then
        SERVICES_HEALTHY=$((SERVICES_HEALTHY + 1))
        echo -e "  ${GREEN}✓${NC} $display_name  [${CYAN}$group${NC}]"
    elif [ "$exit_code" = "124" ]; then
        # Timeout
        if [ "$critical" = "yes" ]; then
            echo -e "  ${RED}✗${NC} $display_name  [${CYAN}$group${NC}]  ${RED}(TIMEOUT - critical)${NC}"
            CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
        else
            echo -e "  ${YELLOW}⚠${NC} $display_name  [${CYAN}$group${NC}]  ${YELLOW}(TIMEOUT)${NC}"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        if [ "$critical" = "yes" ]; then
            echo -e "  ${RED}✗${NC} $display_name  [${CYAN}$group${NC}]  ${RED}(unreachable - critical)${NC}"
            CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
        else
            echo -e "  ${YELLOW}⚠${NC} $display_name  [${CYAN}$group${NC}]  ${YELLOW}(unreachable)${NC}"
            WARNINGS=$((WARNINGS + 1))
        fi
    fi
done

# =============================================================================
# E. KAFKA TOPICS WITH RECORD COUNTS
# =============================================================================
log_section "E. Kafka Topics"

if kafka_exit=$(check_result "kafka") && [ "$kafka_exit" = "0" ]; then
    TOPICS_OUTPUT=$(docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null)
    if [ -n "$TOPICS_OUTPUT" ]; then
        TOPIC_COUNT=$(echo "$TOPICS_OUTPUT" | grep -c .)
        echo -e "  ${GREEN}✓${NC} $TOPIC_COUNT topic(s) found:"
        while IFS= read -r topic; do
            # Get partition count and consumer group lag (if available)
            local partitions=$(docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --topic "$topic" --describe 2>/dev/null | grep "PartitionCount" | awk '{print $3}' | head -1 || echo "?")
            echo -e "    ${CYAN}$topic${NC}  (partitions: $partitions)"
        done <<< "$TOPICS_OUTPUT"
    else
        echo -e "  ${YELLOW}⚠${NC} No topics found"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "  ${RED}✗${NC} Kafka not reachable — cannot list topics"
    WARNINGS=$((WARNINGS + 1))
fi

# =============================================================================
# F. MINIO BUCKETS WITH SIZES
# =============================================================================
log_section "F. MinIO Buckets"

if minio_exit=$(check_result "minio") && [ "$minio_exit" = "0" ]; then
    BUCKETS_OUTPUT=$(docker exec ldt-minio mc ls local/ 2>/dev/null)
    if [ -n "$BUCKETS_OUTPUT" ]; then
        BUCKET_COUNT=$(echo "$BUCKETS_OUTPUT" | grep -c .)
        echo -e "  ${GREEN}✓${NC} $BUCKET_COUNT bucket(s) found:"
        echo "$BUCKETS_OUTPUT" | while IFS= read -r line; do
            # Parse ls output: [date] [time] [size] [bucket/]
            local bucket=$(echo "$line" | awk '{print $NF}' | sed 's|/$||')
            local size=$(echo "$line" | awk '{print $5}')
            if [ -n "$bucket" ]; then
                echo -e "    ${CYAN}$bucket${NC}  ${size:-?}"
            fi
        done
    else
        echo -e "  ${YELLOW}⚠${NC} No buckets found"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "  ${RED}✗${NC} MinIO not reachable — cannot list buckets"
    WARNINGS=$((WARNINGS + 1))
fi

# =============================================================================
# G. FLINK JOB DETAILS
# =============================================================================
log_section "G. Flink Jobs"

if flink_jm_exit=$(check_result "flink-jm") && [ "$flink_jm_exit" = "0" ]; then
    FLINK_OVERVIEW=$(curl -sf http://localhost:8081/overview 2>/dev/null)
    if [ -n "$FLINK_OVERVIEW" ]; then
        local slots_total=$(echo "$FLINK_OVERVIEW" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('slots-total','?'))" 2>/dev/null || echo "?")
        local slots_used=$(echo "$FLINK_OVERVIEW" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('slots-used','?'))" 2>/dev/null || echo "?")
        local tm_count=$(echo "$FLINK_OVERVIEW" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('taskmanagers','?'))" 2>/dev/null || echo "?")
        echo -e "  ${GREEN}✓${NC} Flink cluster:"
        echo -e "    TaskManagers: $tm_count"
        echo -e "    Slots total:  $slots_total"
        echo -e "    Slots used:   $slots_used"
    fi

    FLINK_JOBS=$(curl -sf http://localhost:8081/jobs 2>/dev/null)
    if [ -n "$FLINK_JOBS" ]; then
        local job_count
        job_count=$(echo "$FLINK_JOBS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('jobs',[])))" 2>/dev/null || echo "0")
        if [ "$job_count" -gt 0 ]; then
            echo -e "  ${GREEN}✓${NC} $job_count Flink job(s):"
            echo "$FLINK_JOBS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for j in d.get('jobs', []):
    jid = j.get('id', '?')[:16]
    state = j.get('state', '?')
    name = j.get('name', 'unknown')
    parallelism = j.get('parallelism', '?')
    print(f'    {jid}... [{state}] - {name} (parallelism={parallelism})')
" 2>/dev/null

            # Get checkpoint info for each running job
            echo "$FLINK_JOBS" | python3 -c "
import sys, json, urllib.request
d = json.load(sys.stdin)
for j in d.get('jobs', []):
    if j.get('state') == 'RUNNING':
        jid = j.get('id')
        try:
            info = urllib.request.urlopen(f'http://localhost:8081/jobs/{jid}/info', timeout=5).read()
            info_d = json.loads(info)
            chk = info_d.get('checkpointing', {})
            if chk:
                last_chk = chk.get('last-checkpoint-timestamp', None)
                if last_chk and int(last_chk) > 0:
                    import datetime
                    ts = datetime.datetime.fromtimestamp(int(last_chk)/1000).strftime('%Y-%m-%d %H:%M:%S')
                    print(f'    Last checkpoint: {ts}')
                else:
                    print(f'    Checkpointing: enabled (no checkpoint yet)')
            else:
                print(f'    Checkpointing: not configured')
        except:
            print(f'    Checkpoint info: unavailable')
" 2>/dev/null
        else
            echo -e "  ${YELLOW}⚠${NC} No Flink jobs running"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} Cannot query Flink jobs"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "  ${RED}✗${NC} Flink JobManager not reachable"
    WARNINGS=$((WARNINGS + 1))
fi

# =============================================================================
# H. REDIS INFO
# =============================================================================
log_section "H. Redis Info"

if redis_exit=$(check_result "redis") && [ "$redis_exit" = "0" ]; then
    REDIS_INFO=$(docker exec ldt-redis redis-cli -a "$REDIS_PASSWORD" info clients 2>/dev/null | grep -E "^(connected_clients|blocked_clients|cluster_enabled):" || true)
    REDIS_VERSION=$(docker exec ldt-redis redis-cli -a "$REDIS_PASSWORD" info server 2>/dev/null | grep "^redis_version:" | cut -d: -f2 | tr -d '\r' || echo "?")
    REDIS_MEM=$(docker exec ldt-redis redis-cli -a "$REDIS_PASSWORD" info memory 2>/dev/null | grep "^used_memory_human:" | cut -d: -f2 | tr -d '\r' || echo "?")

    echo -e "  ${GREEN}✓${NC} Redis v${REDIS_VERSION}:"
    if [ -n "$REDIS_INFO" ]; then
        echo "$REDIS_INFO" | while IFS=: read -r key val; do
            val=$(echo "$val" | tr -d '\r')
            echo "    $key: $val"
        done
    fi
    [ -n "$REDIS_MEM" ] && echo "    Memory used: $REDIS_MEM"
else
    echo -e "  ${RED}✗${NC} Redis not reachable"
    WARNINGS=$((WARNINGS + 1))
fi

# =============================================================================
# I. PROMETHEUS SCRAPE STATUS
# =============================================================================
log_section "I. Prometheus"

if prom_exit=$(check_result "prometheus") && [ "$prom_exit" = "0" ]; then
    PROM_STATS=$(curl -sf http://localhost:9090/api/v1/status/tsdb 2>/dev/null)
    if [ -n "$PROM_STATS" ]; then
        local series_count
        series_count=$(echo "$PROM_STATS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('seriesCountByMetricName',[]).__len__())" 2>/dev/null || echo "?")
        echo -e "  ${GREEN}✓${NC} Prometheus is healthy"
        echo "    Time series: $series_count+ metrics"

        # Get scrape interval from targets
        PROM_TARGETS=$(curl -sf http://localhost:9090/api/v1/targets 2>/dev/null)
        if [ -n "$PROM_TARGETS" ]; then
            local scrape_interval
            scrape_interval=$(echo "$PROM_TARGETS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for t in d.get('data', {}).get('targets', []):
    labels = t.get('labels', {})
    if 'job' in labels:
        scrape = t.get('lastScrapeDuration', '?')
        print(f'  {labels.get(\"job\",\"?\")}: scrape_duration={scrape}s')
        break
" 2>/dev/null || true)
            [ -n "$scrape_interval" ] && echo "$scrape_interval"
        fi
    fi
else
    echo -e "  ${RED}✗${NC} Prometheus not reachable"
    WARNINGS=$((WARNINGS + 1))
fi

# =============================================================================
# J. GRAFANA STATUS
# =============================================================================
log_section "J. Grafana"

if grafana_exit=$(check_result "grafana") && [ "$grafana_exit" = "0" ]; then
    GRAFANA_HEALTH=$(curl -sf http://localhost:3000/api/health 2>/dev/null)
    if [ -n "$GRAFANA_HEALTH" ]; then
        local version
        version=$(echo "$GRAFANA_HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('version','?'))" 2>/dev/null || echo "?")
        echo -e "  ${GREEN}✓${NC} Grafana v$version is healthy"

        # Count provisioned dashboards
        GRAFANA_DASHBOARDS=$(curl -sf -u "admin:${GRAFANA_PASSWORD:-admin}" http://localhost:3000/api/search?type=dash-db 2>/dev/null || echo "[]")
        DASH_COUNT=$(echo "$GRAFANA_DASHBOARDS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d))" 2>/dev/null || echo "?")
        echo "    Dashboards: $DASH_COUNT provisioned"
    fi
else
    echo -e "  ${RED}✗${NC} Grafana not reachable"
    WARNINGS=$((WARNINGS + 1))
fi

# =============================================================================
# K. CADQSTREAM METRICS ENDPOINT
# =============================================================================
log_section "K. CA-DQStream Metrics"

if cadq_exit=$(check_result "cadqmetrics") && [ "$cadq_exit" = "0" ]; then
    METRICS_MISC=$(curl -sf http://localhost:9250/metrics 2>/dev/null | head -5)
    if [ -n "$METRICS_MISC" ]; then
        echo -e "  ${GREEN}✓${NC} cadqstream-metrics is healthy"
        local metric_count
        metric_count=$(curl -sf http://localhost:9250/metrics 2>/dev/null | grep -c "^cadqstream" || echo 0)
        echo "    Exposed metrics: $metric_count"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} cadqstream-metrics not reachable"
    WARNINGS=$((WARNINGS + 1))
fi

# =============================================================================
# L. ML SERVICE STATUS
# =============================================================================
log_section "L. ML Service"

if ml_exit=$(check_result "ml-service") && [ "$ml_exit" = "0" ]; then
    ML_HEALTH=$(curl -sf http://localhost:8000/health 2>/dev/null)
    if [ -n "$ML_HEALTH" ]; then
        echo -e "  ${GREEN}✓${NC} ML service is healthy"
        echo "$ML_HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'  {k}: {v}') for k,v in d.items()]" 2>/dev/null || true
    fi
else
    echo -e "  ${YELLOW}⚠${NC} ML service not reachable"
    WARNINGS=$((WARNINGS + 1))
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}  Summary${NC}"
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Service count summary
echo -e "  ${CYAN}Service Counts:${NC}"
echo -e "    Services checked:  $SERVICES_CHECKED"
echo -e "    Healthy:           ${GREEN}$SERVICES_HEALTHY${NC}"
echo -e "    Warnings:          ${YELLOW}$WARNINGS${NC}"
echo -e "    Critical failures: ${RED}$CRITICAL_FAILURES${NC}"
echo ""

# Kafka topics summary
if kafka_exit=$(check_result "kafka") && [ "$kafka_exit" = "0" ]; then
    TOPIC_SUMMARY=$(docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null | tr '\n' ' ')
    TOPIC_COUNT=$(docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null | grep -c . || echo 0)
    if [ -n "$TOPIC_SUMMARY" ]; then
        echo -e "  ${CYAN}Kafka Topics ($TOPIC_COUNT):${NC}"
        echo "    $TOPIC_SUMMARY"
    fi
fi

# MinIO buckets summary
if minio_exit=$(check_result "minio") && [ "$minio_exit" = "0" ]; then
    BUCKET_SUMMARY=$(docker exec ldt-minio mc ls local/ 2>/dev/null | awk '{print $NF}' | sed 's|/$||' | tr '\n' ' ')
    BUCKET_COUNT=$(docker exec ldt-minio mc ls local/ 2>/dev/null | grep -c . || echo 0)
    if [ -n "$BUCKET_SUMMARY" ]; then
        echo ""
        echo -e "  ${CYAN}MinIO Buckets ($BUCKET_COUNT):${NC}"
        echo "    $BUCKET_SUMMARY"
    fi
fi

# Flink job summary
if flink_jm_exit=$(check_result "flink-jm") && [ "$flink_jm_exit" = "0" ]; then
    FLINK_JOBS_SUMMARY=$(curl -sf http://localhost:8081/jobs 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
jobs = d.get('jobs', [])
running = [j for j in jobs if j.get('state') == 'RUNNING']
if running:
    for j in running:
        name = j.get('name', 'unknown')
        state = j.get('state', '?')
        par = j.get('parallelism', '?')
        print(f'    {name} [{state}] (parallelism={par})')
else:
    print('    No running jobs')
" 2>/dev/null || echo "    (unavailable)")
    echo ""
    echo -e "  ${CYAN}Flink Jobs:${NC}"
    echo "$FLINK_JOBS_SUMMARY"
fi

echo ""
echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"

# Exit code logic
if [ $CRITICAL_FAILURES -gt 0 ]; then
    echo -e "  ${RED}${BOLD}Status: CRITICAL — $CRITICAL_FAILURES critical failure(s)${NC}"
    echo ""
    echo -e "  ${RED}Action required:${NC}"
    echo "    1. Check container logs: docker compose -f deployment/docker-compose.yml logs [service]"
    echo "    2. Restart failed services: docker compose -f deployment/docker-compose.yml restart [service]"
    echo "    3. For full restart: bash deployment/scripts/stop.sh && bash deployment/scripts/start.sh"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "  ${YELLOW}${BOLD}Status: WARNINGS — $WARNINGS warning(s), no critical failures${NC}"
    echo ""
    echo "    All critical services are healthy. Some non-critical services may need attention."
    exit 2
else
    echo -e "  ${GREEN}${BOLD}Status: HEALTHY — all $SERVICES_HEALTHY services healthy${NC}"
    echo ""
    echo -e "  ${GREEN}CA-DQStream stack is fully operational.${NC}"
    exit 0
fi
