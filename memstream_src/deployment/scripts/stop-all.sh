#!/bin/bash
# =============================================================================
# CA-DQStream + MemStream v5 - Stop Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping CA-DQStream + MemStream v5...${NC}"
echo ""

# Stop and remove containers
docker compose down

echo ""
echo -e "${GREEN}CA-DQStream + MemStream v5 stopped.${NC}"
echo ""
echo "To remove volumes (data loss!): docker compose down -v"
echo "To start again: ./scripts/start-all.sh"
