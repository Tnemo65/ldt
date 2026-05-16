#!/bin/bash
# =============================================================================
# CA-DQStream - Reset Script
# Cleanup/reset only — does NOT build images or start services
# Idempotent: safe to run multiple times
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging functions
log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $*"; }

# Critical ports that must be free for startup to succeed
CRITICAL_PORTS=(9092 8081)
# All ports used by the deployment
ALL_PORTS=(2181 9092 8081 8080 8082 3000 9090 9100 9308 9000 9001 9250)
PORT_NAMES=(
    "Zookeeper"
    "Kafka"
    "Flink UI"
    "Kafka UI"
    "Schema Registry"
    "Grafana"
    "Prometheus"
    "Node Exporter"
    "Kafka Exporter"
    "MinIO API"
    "MinIO Console"
    "Stats Writer"
)

# =============================================================================
# PHASE 1: Stop containers gracefully
# =============================================================================
phase_stop_containers() {
    log_info "Stopping docker-compose stack..."
    docker compose -f "$DEPLOYMENT_DIR/docker-compose.yml" down --remove-orphans 2>/dev/null || true

    log_info "Stopping orphan ldt-* containers..."
    local orphans
    orphans=$(docker ps -a --filter "name=^ldt-" --format "{{.Names}}" 2>/dev/null || true)
    if [ -n "$orphans" ]; then
        echo "$orphans" | while read -r container; do
            if [ -n "$container" ]; then
                log_info "Stopping container: $container"
                docker stop -t 30 "$container" 2>/dev/null || true
            fi
        done
    else
        log_info "No orphan ldt-* containers found"
    fi
}

# =============================================================================
# PHASE 2: Cleanup resources (containers, networks, volumes)
# =============================================================================
phase_cleanup_resources() {
    # Remove all containers (including stopped ones)
    log_info "Removing all ldt-* containers..."
    local containers
    containers=$(docker ps -a --filter "name=^ldt-" --format "{{.Names}}" 2>/dev/null || true)
    if [ -n "$containers" ]; then
        echo "$containers" | while read -r container; do
            if [ -n "$container" ]; then
                log_info "Removing container: $container"
                docker rm -f "$container" 2>/dev/null || true
            fi
        done
    else
        log_info "No ldt-* containers to remove"
    fi

    # Remove dangling networks
    log_info "Removing dangling networks..."
    local networks=(
        "cadqstream-net"
        "ldt_cadqstream-net"
        "deployment_cadqstream-net"
        "deployment-cadqstream-net"
    )
    for net in "${networks[@]}"; do
        if docker network ls --format "{{.Name}}" | grep -q "^${net}$"; then
            log_info "Removing network: $net"
            docker network rm "$net" 2>/dev/null || true
        else
            log_info "Network already absent: $net"
        fi
    done

    # Remove orphan volumes (ldt- prefix)
    log_info "Removing orphan volumes..."
    local volumes
    volumes=$(docker volume ls --filter "name=^ldt-" --format "{{.Name}}" 2>/dev/null || true)
    if [ -n "$volumes" ]; then
        echo "$volumes" | while read -r volume; do
            if [ -n "$volume" ]; then
                log_info "Removing volume: $volume"
                docker volume rm "$volume" 2>/dev/null || true
            fi
        done
    else
        log_info "No orphan volumes to remove"
    fi
}

# =============================================================================
# PHASE 3: Port verification
# =============================================================================
phase_verify_ports() {
    log_info "Checking required ports..."

    local critical_in_use=()
    local non_critical_in_use=()

    for port in "${ALL_PORTS[@]}"; do
        local idx
        idx=$(printf '%s\n' "${ALL_PORTS[@]}" | grep -n "^${port}$" | cut -d: -f1)
        idx=$((idx - 1))
        local name="${PORT_NAMES[$idx]:-unknown}"

        if netstat -tlnp 2>/dev/null | grep -q ":${port} " || ss -tlnp 2>/dev/null | grep -q ":${port} "; then
            local proc_info
            proc_info=$(netstat -tlnp 2>/dev/null | grep ":${port} " || ss -tlnp 2>/dev/null | grep ":${port} " || echo "process info unavailable")
            log_warn "Port $port ($name) is in use: $proc_info"

            if printf '%s\n' "${CRITICAL_PORTS[@]}" | grep -q "^${port}$"; then
                critical_in_use+=("$port ($name)")
            else
                non_critical_in_use+=("$port ($name)")
            fi
        else
            log_ok "Port $port ($name) is free"
        fi
    done

    # Report summary
    echo ""
    if [ ${#critical_in_use[@]} -gt 0 ]; then
        log_err "Critical ports still occupied:"
        for item in "${critical_in_use[@]}"; do
            echo -e "  ${RED}  $item${NC}" >&2
        done
        log_err "Cannot proceed. Stop the blocking processes before starting services."
        return 1
    fi

    if [ ${#non_critical_in_use[@]} -gt 0 ]; then
        log_warn "Non-critical ports still occupied:"
        for item in "${non_critical_in_use[@]}"; do
            echo -e "  ${YELLOW}  $item${NC}"
        done
        log_warn "Startup may fail for affected services."
    fi

    echo ""
    log_ok "Port verification complete."
}

# =============================================================================
# PHASE 4: Validate preconditions for startup
# =============================================================================
phase_validate_preconditions() {
    log_info "Validating preconditions for startup..."

    local errors=0

    # Check docker is running
    log_info "Checking Docker daemon..."
    if ! docker info &>/dev/null; then
        log_err "Docker daemon is not running or not accessible"
        errors=$((errors + 1))
    else
        log_ok "Docker daemon is running"
    fi

    # Check docker compose plugin
    log_info "Checking docker compose plugin..."
    if ! docker compose version &>/dev/null; then
        log_err "docker compose plugin not found"
        log_err "Install Docker Desktop or docker-compose package"
        errors=$((errors + 1))
    else
        local compose_version
        compose_version=$(docker compose version 2>/dev/null || echo "unknown")
        log_ok "docker compose plugin available: $compose_version"
    fi

    # Check .env file
    log_info "Checking .env file..."
    if [ ! -f "$DEPLOYMENT_DIR/.env" ]; then
        if [ -f "$DEPLOYMENT_DIR/.env.example" ]; then
            log_warn ".env not found — copying from .env.example"
            cp "$DEPLOYMENT_DIR/.env.example" "$DEPLOYMENT_DIR/.env"
            log_warn "Edit $DEPLOYMENT_DIR/.env with your configuration before starting"
        else
            log_err ".env file not found at $DEPLOYMENT_DIR/.env"
            errors=$((errors + 1))
        fi
    else
        log_ok ".env file exists"
    fi

    # Check Flink image
    log_info "Checking Flink image..."
    local flink_image="ldt-flink:1.18.1-py"
    if docker image ls "$flink_image" &>/dev/null; then
        log_ok "Flink image found: $flink_image"
    else
        log_warn "Flink image not found: $flink_image"
        log_warn "Run 'make build' or 'make up' to build the image before starting"
    fi

    # Print startup readiness
    echo ""
    log_step "Startup readiness:"
    if [ $errors -eq 0 ]; then
        echo "  Services ready to start:"
        echo "    ${GREEN}make up${NC}   — Full production deployment"
        echo "    ${GREEN}make dev${NC}   — Development mode (faster)"
        echo ""
        echo "  Individual commands:"
        echo "    make build   — Build Flink image only"
        echo "    make down    — Stop containers only"
        echo "    make clean   — Stop + remove volumes"
        echo "    make health  — Check service health"
        echo "    make info    — System information"
    else
        log_err "$errors precondition(s) failed — fix errors before proceeding"
        return 1
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  CA-DQStream Reset — Cleanup Only${NC}"
    echo -e "${CYAN}  No build, no startup${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Trap to ensure we don't leave the terminal in a broken state
    trap 'log_err "Reset interrupted"; exit 1' INT TERM

    log_step "PHASE 1: Stop containers gracefully"
    phase_stop_containers
    log_ok "Phase 1 complete"
    echo ""

    log_step "PHASE 2: Cleanup resources"
    phase_cleanup_resources
    log_ok "Phase 2 complete"
    echo ""

    log_step "PHASE 3: Verify ports"
    if ! phase_verify_ports; then
        log_err "Port verification failed"
        exit 1
    fi
    echo ""

    log_step "PHASE 4: Validate preconditions"
    if ! phase_validate_preconditions; then
        log_err "Precondition validation failed"
        exit 1
    fi
    echo ""

    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Reset complete${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    log_ok "Run 'make up' or 'make dev' to start services"
    echo ""
}

main "$@"
