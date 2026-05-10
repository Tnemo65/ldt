#!/bin/bash
# =============================================================================
# CA-DQStream - Stop Script
# Gracefully stops all containers and optionally removes volumes
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

REMOVE_VOLUMES="${1:-}"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Stopping CA-DQStream Stack${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"

echo "Stopping all containers..."
docker compose -f "$DEPLOYMENT_DIR/docker-compose.yml" down 2>/dev/null || true

if [ "$REMOVE_VOLUMES" == "--remove-volumes" ] || [ "$REMOVE_VOLUMES" == "-v" ]; then
    echo "Removing named volumes (data will be lost)..."
    docker compose -f "$DEPLOYMENT_DIR/docker-compose.yml" down -v --remove-orphans 2>/dev/null || true
    echo -e "${GREEN}Volumes removed.${NC}"
else
    echo "Named volumes preserved (data persisted). Use '--remove-volumes' to delete data."
fi

echo "Removing network..."
docker network rm cadqstream-net 2>/dev/null || true

echo ""
echo -e "${GREEN}CA-DQStream stack stopped.${NC}"
echo "To start again: bash deployment/scripts/start.sh"
