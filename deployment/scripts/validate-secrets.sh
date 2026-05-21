#!/bin/bash
# =============================================================================
# CA-DQStream Secret Validation Script
# BLOCKS deployment if any secrets are still using placeholder values.
# Run BEFORE docker compose up, or include in start.sh.
#
# Usage:
#   bash deployment/scripts/validate-secrets.sh
#   ./validate-secrets.sh  # from deployment/scripts/
#   source <(./validate-secrets.sh --dry-run)  # dry-run mode
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$DEPLOYMENT_DIR/.env"

# Load .env if it exists
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
fi

# ANSI colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DRY_RUN="${1:-}"
if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "[dry-run] Would check the following secrets..."
fi

ERRORS=0

check_secret() {
    local name="$1"
    local value="$2"
    local trimmed_value
    # Trim leading/trailing whitespace for comparison
    trimmed_value="$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

    if [ -z "$trimmed_value" ]; then
        echo -e "${RED}[FATAL]${NC} $name is EMPTY — must be set in .env"
        ERRORS=$((ERRORS + 1))
        return 1
    fi

    # Check for placeholder patterns
        local is_placeholder=0
    case "$trimmed_value" in
        changeme*)
            is_placeholder=1
            ;;
        *changeme*)
            is_placeholder=1
            ;;
        *!!*)
            is_placeholder=1
            ;;
        *_local)
            is_placeholder=1
            ;;
        *_admin)
            is_placeholder=1
            ;;
    esac

    if [ "$is_placeholder" -eq 1 ]; then
        echo -e "${RED}[FATAL]${NC} $name is still a placeholder value: $trimmed_value"
        echo "         Please set a strong random value in .env"
        echo "         Generate with: openssl rand -hex 32"
        ERRORS=$((ERRORS + 1))
        return 1
    fi

    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo -e "${GREEN}[OK]${NC}   $name: set"
    fi
    return 0
}

echo ""
echo -e "${YELLOW}Validating secrets in $ENV_FILE${NC}"
echo "=============================================="
echo ""

# All required secrets
check_secret "MINIO_ROOT_PASSWORD" "${MINIO_ROOT_PASSWORD:-}"
check_secret "GRAFANA_PASSWORD" "${GRAFANA_PASSWORD:-}"
check_secret "REDIS_PASSWORD" "${REDIS_PASSWORD:-}"
check_secret "INTERNAL_API_KEY" "${INTERNAL_API_KEY:-}"
check_secret "METRICS_API_KEY" "${METRICS_API_KEY:-}"
check_secret "MEMSTREAM_MODEL_SIGNING_KEY" "${MEMSTREAM_MODEL_SIGNING_KEY:-}"
check_secret "IEC_SIGNING_KEY" "${IEC_SIGNING_KEY:-}"

echo ""

if [ "$DRY_RUN" = "--dry-run" ]; then
    if [ $ERRORS -eq 0 ]; then
        echo -e "${GREEN}All secrets look good!${NC}"
    else
        echo -e "${RED}WARNING: $ERRORS secret(s) still need to be updated.${NC}"
    fi
    exit 0
fi

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}==============================================${NC}"
    echo -e "${RED}[FATAL] $ERRORS secret(s) still use placeholder values.${NC}"
    echo -e "${RED}==============================================${NC}"
    echo ""
    echo "Please update deployment/.env with strong random secrets before deploying."
    echo "Generate secrets with: openssl rand -hex 32"
    echo ""
    exit 1
fi

echo -e "${GREEN}[OK] All secrets validated successfully.${NC}"
echo ""
exit 0
