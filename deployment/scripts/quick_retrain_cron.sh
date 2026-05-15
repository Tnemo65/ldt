#!/bin/bash
# =============================================================================
# Quick Retrain Cron Script - Phase 5C
# =============================================================================
# Schedule: every 24 hours via cron
#
# This script checks for retrain trigger conditions:
# - Trigger B: Anomaly rate > 15% for 3 consecutive days
#
# If conditions are met, it calls POST /api/strategy/retrain_model
# with Authorization: Bearer ${INTERNAL_API_KEY}
#
# Crontab entry (run every 24 hours at midnight):
#   0 0 * * * /opt/memstream/scripts/quick_retrain_cron.sh >> /var/log/memstream/quick_retrain.log 2>&1
#
# =============================================================================

set -euo pipefail

# Configuration
ML_SERVICE_URL="${ML_SERVICE_URL:-http://localhost:8000}"
INTERNAL_API_KEY="${INTERNAL_API_KEY:-}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAFANA_TOKEN="${GRAFANA_TOKEN:-}"
GRAFANA_DATASOURCE="${GRAFANA_DATASOURCE:-Prometheus}"
ANOMALY_RATE_THRESHOLD="${ANOMALY_RATE_THRESHOLD:-0.15}"
CONSECUTIVE_DAYS="${CONSECUTIVE_DAYS:-3}"

# Logging
LOG_FILE="${LOG_FILE:-/var/log/memstream/quick_retrain_cron.log}"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

log() {
    echo "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$TIMESTAMP] ERROR: $1" | tee -a "$LOG_FILE" >&2
}

# =============================================================================
# Check Prerequisites
# =============================================================================

check_prerequisites() {
    if [[ -z "$INTERNAL_API_KEY" ]]; then
        log_error "INTERNAL_API_KEY environment variable is not set"
        exit 1
    fi

    if [[ ! -x "$(command -v curl)" ]]; then
        log_error "curl is required but not installed"
        exit 1
    fi
}

# =============================================================================
# Query Grafana for Anomaly Rate
# =============================================================================

query_grafana_anomaly_rate() {
    local days=$1
    local neighborhood=$2

    log "Querying Grafana for anomaly rate (last ${days} days, neighborhood=${neighborhood})"

    # Prometheus query for average anomaly rate
    # This assumes metric name: memstream_anomaly_rate or meta_anomaly_rate
    local promql_query="avg_over_time(meta_anomaly_rate{neighborhood=\"${neighborhood}\"}[${days}d])"

    # URL-encode the query
    local encoded_query=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${promql_query}'))" 2>/dev/null || echo "${promql_query}")

    local grafana_api_url="${GRAFANA_URL}/api/ds/query"

    local response
    if [[ -n "$GRAFANA_TOKEN" ]]; then
        response=$(curl -s -X POST \
            -H "Authorization: Bearer ${GRAFANA_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{\"queries\":[{\"refId\":\"A\",\"expr\":\"${promql_query}\",\"datasource\":{\"type\":\"prometheus\",\"uid\":\"${GRAFANA_DATASOURCE}\"}}],\"from\":\"now-${days}d\",\"to\":\"now\"}" \
            "${grafana_api_url}" 2>/dev/null || echo '{"error": "curl failed"}')
    else
        # Try without auth (for local development)
        response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "{\"queries\":[{\"refId\":\"A\",\"expr\":\"${promql_query}\",\"datasource\":{\"type\":\"prometheus\",\"uid\":\"${GRAFANA_DATASOURCE}\"}}],\"from\":\"now-${days}d\",\"to\":\"now\"}" \
            "${grafana_api_url}" 2>/dev/null || echo '{"error": "curl failed"}')
    fi

    echo "$response"
}

# =============================================================================
# Parse Anomaly Rate from Response
# =============================================================================

parse_anomaly_rate() {
    local response=$1
    local default_value=0.0

    # Try to extract the average value from Prometheus response
    # Expected format: {"data":{"result":[{"value":[timestamp, value]}]}}
    local rate
    rate=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'data' in data and 'result' in data['data']:
        results = data['data']['result']
        if results and 'value' in results[0]:
            print(results[0]['value'][1])
        else:
            print('${default_value}')
    else:
        print('${default_value}')
except Exception as e:
    print('${default_value}')
" 2>/dev/null || echo "${default_value}")

    echo "$rate"
}

# =============================================================================
# Check Consecutive Days Above Threshold
# =============================================================================

check_consecutive_days() {
    local neighborhood=$1
    local threshold=$2
    local days=$3

    log "Checking consecutive days above threshold: neighborhood=${neighborhood}, threshold=${threshold}, days=${days}"

    # Query for each day individually
    local consecutive_count=0
    local days_above_threshold=0

    for ((i=1; i<=days; i++)); do
        local response
        response=$(query_grafana_anomaly_rate 1 "$neighborhood")

        local rate
        rate=$(parse_anomaly_rate "$response")

        log "  Day $i: anomaly_rate=${rate}"

        # Compare as floats
        local is_above
        is_above=$(python3 -c "
import sys
rate = float('${rate}')
threshold = float('${threshold}')
print('1' if rate > threshold else '0')
" 2>/dev/null || echo "0")

        if [[ "$is_above" == "1" ]]; then
            ((consecutive_count++))
            if [[ $consecutive_count -ge $days ]]; then
                days_above_threshold=$consecutive_count
                break
            fi
        else
            consecutive_count=0
        fi
    done

    echo "$consecutive_count"
}

# =============================================================================
# Trigger Retrain
# =============================================================================

trigger_retrain() {
    local neighborhood=$1
    local trigger_reason=$2

    log "Triggering retrain: neighborhood=${neighborhood}, reason=${trigger_reason}"

    local response
    response=$(curl -s -X POST \
        -H "Authorization: Bearer ${INTERNAL_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"neighborhood\":\"${neighborhood}\",\"data_source\":\"recent\",\"epochs\":50}" \
        "${ML_SERVICE_URL}/api/strategy/retrain_model" 2>/dev/null || echo '{"error": "curl failed"}')

    log "Retrain response: $response"

    # Check for success
    local status
    status=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('status', 'unknown'))
except:
    print('error')
" 2>/dev/null || echo "error")

    if [[ "$status" == "completed" ]] || [[ "$status" == "queued" ]]; then
        log "Retrain triggered successfully for ${neighborhood}"
        return 0
    else
        log_error "Failed to trigger retrain: status=${status}"
        return 1
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    log "========================================"
    log "Quick Retrain Cron Started"
    log "========================================"

    check_prerequisites

    # Neighborhoods to check
    local neighborhoods="${NEIGHBORHOODS:-manhattan,brooklyn,queens_lower,queens_upper,bronx,staten_island,ewr,jfk,nalp,unknown}"
    IFS=',' read -ra NB_ARRAY <<< "$neighborhoods"

    local retrain_count=0

    for nb in "${NB_ARRAY[@]}"; do
        log "----------------------------------------"
        log "Checking neighborhood: ${nb}"

        # Check consecutive days above threshold
        local consecutive
        consecutive=$(check_consecutive_days "$nb" "$ANOMALY_RATE_THRESHOLD" "$CONSECUTIVE_DAYS")

        log "Consecutive days above threshold: ${consecutive}/${CONSECUTIVE_DAYS}"

        if [[ $consecutive -ge $CONSECUTIVE_DAYS ]]; then
            log "Trigger B met for ${nb}: anomaly rate > ${ANOMALY_RATE_THRESHOLD} for ${CONSECUTIVE_DAYS} consecutive days"

            if trigger_retrain "$nb" "trigger_b_anomaly_rate"; then
                ((retrain_count++))
            fi
        else
            log "No retrain trigger for ${nb}"
        fi
    done

    log "========================================"
    log "Quick Retrain Cron Completed"
    log "Retrain jobs triggered: ${retrain_count}"
    log "========================================"

    exit 0
}

# Run main
main "$@"
