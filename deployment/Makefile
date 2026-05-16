# =============================================================================
# CA-DQStream Deployment Makefile
# Deterministic Docker orchestration for CA-DQStream streaming pipeline
#
# Usage (from project root c:\proj\ldt):
#   make [target]
#
# Or from deployment directory:
#   cd deployment && make [target]
#
# Override timeouts at runtime:
#   WAIT_TIMEOUT=60 make up
#   FLINK_TIMEOUT=180 make up
#   BOOT_TIMEOUT=10 make up
#
# Override dev mode (shorter timeouts + verbose output):
#   DEV_MODE=true make dev
# =============================================================================

.PHONY: help reset build up dev down clean health logs ps init restart shell info verify

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
MAKEFILE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
COMPOSE_FILE := $(MAKEFILE_DIR)/docker-compose.yml
SCRIPTS_DIR := $(MAKEFILE_DIR)/scripts

# ── Timeouts (override at runtime: WAIT_TIMEOUT=60 make up) ──────────────────
WAIT_TIMEOUT   ?= 120
BOOT_TIMEOUT   ?= 30
FLINK_TIMEOUT  ?= 120

# ── Image names ────────────────────────────────────────────────────────────────
FLINK_IMAGE := ldt-flink:1.17.1-py

# ── Terminal colours (fallback to no-op on Windows Git Bash / WSL / macOS) ────
BLUE   := $(shell tput setaf 4 2>/dev/null || echo '')
GREEN  := $(shell tput setaf 2 2>/dev/null || echo '')
RED    := $(shell tput setaf 1 2>/dev/null || echo '')
YELLOW := $(shell tput setaf 3 2>/dev/null || echo '')
CYAN   := $(shell tput setaf 6 2>/dev/null || echo '')
NC     := $(shell tput sgr0 2>/dev/null || echo '')

# ── Compose helpers ───────────────────────────────────────────────────────────
DOCKER_COMPOSE       := docker compose -f $(COMPOSE_FILE)
DOCKER_COMPOSE_EXEC := $(DOCKER_COMPOSE) exec -T

# ── Macro: step banner ───────────────────────────────────────────────────────
define step
	@echo ""
	@echo -e "$(CYAN)--- $1 ---$(NC)"
endef

# ════════════════════════════════════════════════════════════════════════════════
# HELP — show all targets with descriptions
# ════════════════════════════════════════════════════════════════════════════════
#   make help
#
# Targets:
#
#   make reset         — Cleanup only, no build, no startup (run reset.sh)
#   make build        — Build custom Flink image (ldt-flink:1.17.1-py)
#   make up           — Full deterministic startup (reset → build → up → health)
#   make dev          — Dev mode (same as up but shorter timeouts, verbose)
#   make down         — Stop all containers, remove orphans, keep volumes
#   make clean        — Stop + remove volumes + prune networks (full data wipe)
#   make health       — Run healthcheck script and report
#   make logs         — Tail logs for all services
#   make logs SERVICE=kafka  — Tail logs for a specific service
#   make ps           — Show container status
#   make init         — Run bootstrap.sh only (assumes containers are running)
#   make restart SERVICE=flink-jobmanager  — Restart a specific service
#   make shell SERVICE=redis  — Open shell in a container
#   make info         — Show container count, docker df, networks, volumes
#   make help         — Show this help
#
help:
	@echo ""
	@echo -e "$(CYAN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo -e "$(CYAN)  CA-DQStream Deployment — Available targets$(NC)"
	@echo -e "$(CYAN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "  Lifecycle targets:"
	@echo "    make reset         Cleanup only, no build, no startup"
	@echo "    make build        Build custom Flink image (ldt-flink:1.17.1-py)"
	@echo "    make up           Full deterministic startup (reset → build → up → verify)"
	@echo "    make dev          Dev mode (shorter timeouts, verbose logging)"
	@echo ""
	@echo "  Container targets:"
	@echo "    make down         Stop containers, keep volumes"
	@echo "    make clean        Stop + remove volumes + prune networks (full wipe)"
	@echo "    make ps           Show container status"
	@echo "    make restart SERVICE=flink-jobmanager   Restart specific service"
	@echo "    make shell SERVICE=redis               Open shell in container"
	@echo ""
	@echo "  Inspection targets:"
	@echo "    make health       Run healthcheck script and report"
	@echo "    make logs         Tail logs for all services"
	@echo "    make logs SERVICE=kafka    Tail logs for specific service"
	@echo "    make info         Docker system df, networks, volumes"
	@echo ""
	@echo "  Init targets:"
	@echo "    make init         Run bootstrap.sh only (assumes containers running)"
	@echo ""
	@echo "  Override timeouts at runtime:"
	@echo "    WAIT_TIMEOUT=60 make up       (default: 120)"
	@echo "    FLINK_TIMEOUT=180 make up     (default: 120)"
	@echo "    BOOT_TIMEOUT=10 make up       (default: 30)"
	@echo ""
	@echo "    DEV_MODE=true make dev        (pre-configured short timeouts)"
	@echo ""

# ════════════════════════════════════════════════════════════════════════════════
# VERIFY — pre-flight checks (called automatically by `make up` and `make dev`)
# ════════════════════════════════════════════════════════════════════════════════
#   make verify
#
verify:
	@echo -e "$(BLUE)[VERIFY]${NC}  Pre-flight checks..."
	@if ! docker info &>/dev/null; then \
		echo -e "$(RED)[ERROR]${NC}  Docker is not running or not accessible"; \
		exit 1; \
	fi
	@if ! docker compose version &>/dev/null; then \
		echo -e "$(RED)[ERROR]${NC}  docker compose plugin not found"; \
		exit 1; \
	fi
	@if [ ! -f $(COMPOSE_FILE) ]; then \
		echo -e "$(RED)[ERROR]${NC}  docker-compose.yml not found at $(COMPOSE_FILE)"; \
		exit 1; \
	fi
	@if [ ! -f $(MAKEFILE_DIR)/.env ]; then \
		echo -e "$(YELLOW)[WARN]${NC}  .env file not found at $(MAKEFILE_DIR)/.env"; \
		echo -e "$(YELLOW)[WARN]${NC}  Copy .env.example to .env and configure required variables"; \
	fi
	@echo -e "$(GREEN)[VERIFY]${NC}  Pre-flight checks passed"

# ════════════════════════════════════════════════════════════════════════════════
# RESET — cleanup only, no build, no startup
# ════════════════════════════════════════════════════════════════════════════════
#   make reset
#
reset:
	@echo -e "$(BLUE)[RESET]${NC}  Cleanup only — no build, no startup"
	@bash $(SCRIPTS_DIR)/reset.sh

# ════════════════════════════════════════════════════════════════════════════════
# BUILD — build custom Flink image only
# ════════════════════════════════════════════════════════════════════════════════
#   make build
#
#   Build context is the project root (c:\proj\ldt) because COPY paths in the
#   Dockerfile reference deployment/flink/*.jar relative to the build context.
#
build:
	@echo -e "$(BLUE)[BUILD]${NC}  Building $(FLINK_IMAGE)"
	@echo "  Dockerfile: $(MAKEFILE_DIR)/flink/Dockerfile"
	@echo "  Build context: $(PROJECT_ROOT)"
	@docker build \
		-t $(FLINK_IMAGE) \
		-f $(MAKEFILE_DIR)/flink/Dockerfile \
		$(PROJECT_ROOT)
	@echo -e "$(GREEN)[BUILD]${NC}  Done — $(FLINK_IMAGE) ready"

# ════════════════════════════════════════════════════════════════════════════════
# UP — full deterministic startup: reset → build → start → bootstrap → verify
# ════════════════════════════════════════════════════════════════════════════════
#   make up
#   WAIT_TIMEOUT=60 make up    (override wait timeout)
#
up: verify
	@echo -e "$(CYAN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo -e "$(CYAN)  UP — Deterministic startup$(NC)"
	@echo -e "$(CYAN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""

	$(call step,"PHASE 1: Reset environment")
	@bash $(SCRIPTS_DIR)/reset.sh

	$(call step,"PHASE 2: Build Flink image")
	@docker build \
		-t $(FLINK_IMAGE) \
		-f $(MAKEFILE_DIR)/flink/Dockerfile \
		$(PROJECT_ROOT)

	$(call step,"PHASE 3: Start infrastructure (zookeeper, kafka, schema-registry, kafka-ui, kafka-exporter)")
	@$(DOCKER_COMPOSE) up -d zookeeper kafka schema-registry kafka-ui kafka-exporter

	$(call step,"PHASE 4: Wait for Kafka (port 9092)")
	@bash $(SCRIPTS_DIR)/wait-for.sh localhost:9092 --timeout=$(WAIT_TIMEOUT)

	$(call step,"PHASE 5: Start storage + cache (minio, redis)")
	@$(DOCKER_COMPOSE) up -d minio redis

	$(call step,"PHASE 6: Wait for MinIO (port 9000)")
	@bash $(SCRIPTS_DIR)/wait-for.sh localhost:9000 --timeout=$(WAIT_TIMEOUT)

	$(call step,"PHASE 7: Run init containers (kafka-init, minio-init)")
	@$(DOCKER_COMPOSE) up -d kafka-init minio-init
	@sleep $(BOOT_TIMEOUT)

	$(call step,"PHASE 8: Start Flink (jobmanager, taskmanager)")
	@$(DOCKER_COMPOSE) up -d flink-jobmanager flink-taskmanager

	$(call step,"PHASE 9: Wait for Flink REST API (port 8081)")
	@bash $(SCRIPTS_DIR)/wait-for.sh localhost:8081 --timeout=$(FLINK_TIMEOUT) --path=/overview

	$(call step,"PHASE 10: Run Flink init (pipeline job submission)")
	@$(DOCKER_COMPOSE) up flink-init
	@sleep $(BOOT_TIMEOUT)

	$(call step,"PHASE 11: Start ML service (ml-service, action-replay-worker)")
	@$(DOCKER_COMPOSE) up -d ml-service action-replay-worker

	$(call step,"PHASE 12: Start observability stack (prometheus, grafana, exporters)")
	@$(DOCKER_COMPOSE) up -d prometheus grafana node-exporter cadqstream-metrics stats-writer kafka-producer

	$(call step,"PHASE 13: Wait for monitoring services (prometheus, grafana)")
	@bash $(SCRIPTS_DIR)/wait-for.sh localhost:9090 --timeout=$(BOOT_TIMEOUT)
	@bash $(SCRIPTS_DIR)/wait-for.sh localhost:3000 --timeout=$(BOOT_TIMEOUT)

	$(call step,"PHASE 14: Verify health")
	@bash $(SCRIPTS_DIR)/check-health.sh

	@echo ""
	@echo -e "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo -e "$(GREEN)  UP complete — all services started and verified$(NC)"
	@echo -e "$(GREEN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""

# ════════════════════════════════════════════════════════════════════════════════
# DEV — development mode (shorter timeouts, verbose logging)
# ════════════════════════════════════════════════════════════════════════════════
#   make dev
#   or
#   DEV_MODE=true make dev
#
dev: verify
	@echo -e "$(YELLOW)[DEV]${NC}  Development mode — shorter timeouts, verbose output"
	@$(MAKE) up \
		WAIT_TIMEOUT=30 \
		FLINK_TIMEOUT=60 \
		BOOT_TIMEOUT=10

# ════════════════════════════════════════════════════════════════════════════════
# DOWN — stop all containers, remove orphans, keep volumes
# ════════════════════════════════════════════════════════════════════════════════
#   make down
#
down:
	@echo -e "$(BLUE)[DOWN]${NC}  Stopping all containers, keeping volumes"
	@$(DOCKER_COMPOSE) down --remove-orphans
	@echo -e "$(GREEN)[DOWN]${NC}  Containers stopped. Volumes preserved."

# ════════════════════════════════════════════════════════════════════════════════
# CLEAN — full data wipe: stop + remove volumes + prune networks
# ════════════════════════════════════════════════════════════════════════════════
#   make clean
#
clean:
	@echo -e "$(RED)[CLEAN]${NC}  Full data wipe — stopping, removing volumes, pruning networks"
	@$(DOCKER_COMPOSE) down -v --remove-orphans
	@docker network prune -f 2>/dev/null || true
	@echo -e "$(GREEN)[CLEAN]${NC}  Done — all volumes and networks removed."

# ════════════════════════════════════════════════════════════════════════════════
# HEALTH — run healthcheck script and report
# ════════════════════════════════════════════════════════════════════════════════
#   make health
#
health:
	@bash $(SCRIPTS_DIR)/check-health.sh

# ════════════════════════════════════════════════════════════════════════════════
# LOGS — tail logs for all services or a specific service
# ════════════════════════════════════════════════════════════════════════════════
#   make logs              # all services
#   make logs SERVICE=kafka  # specific service
#
logs:
ifndef SERVICE
	@echo -e "$(BLUE)[LOGS]${NC}  Tailing logs for all CA-DQStream services"
	@$(DOCKER_COMPOSE) logs -f --tail=100
else
	@echo -e "$(BLUE)[LOGS]${NC}  Tailing logs for $(SERVICE)"
	@$(DOCKER_COMPOSE) logs -f --tail=200 $(SERVICE)
endif

# ════════════════════════════════════════════════════════════════════════════════
# PS — show container status
# ════════════════════════════════════════════════════════════════════════════════
#   make ps
#
ps:
	@$(DOCKER_COMPOSE) ps

# ════════════════════════════════════════════════════════════════════════════════
# INIT — run bootstrap.sh only (assumes containers are already running)
# ════════════════════════════════════════════════════════════════════════════════
#   make init
#
init:
	@echo -e "$(BLUE)[INIT]${NC}  Running bootstrap.sh (assumes containers are running)"
	@bash $(SCRIPTS_DIR)/bootstrap.sh

# ════════════════════════════════════════════════════════════════════════════════
# RESTART — restart a specific service
# ════════════════════════════════════════════════════════════════════════════════
#   make restart SERVICE=flink-jobmanager
#
restart:
ifndef SERVICE
	@echo -e "$(RED)[ERROR]${NC}  SERVICE is required. Example: make restart SERVICE=flink-jobmanager"
	@exit 1
endif
	@echo -e "$(YELLOW)[RESTART]${NC}  Restarting $(SERVICE)"
	@$(DOCKER_COMPOSE) restart $(SERVICE)
	@echo -e "$(GREEN)[RESTART]${NC}  Done"

# ════════════════════════════════════════════════════════════════════════════════
# SHELL — open shell (exec) in a container
# ════════════════════════════════════════════════════════════════════════════════
#   make shell SERVICE=redis
#   make shell SERVICE=kafka
#   make shell SERVICE=flink-jobmanager
#
shell:
ifndef SERVICE
	@echo -e "$(RED)[ERROR]${NC}  SERVICE is required. Example: make shell SERVICE=redis"
	@exit 1
endif
	@echo -e "$(BLUE)[SHELL]${NC}  Opening shell in $(SERVICE)"
	@$(DOCKER_COMPOSE) exec $(SERVICE) /bin/bash

# ════════════════════════════════════════════════════════════════════════════════
# INFO — Docker system info: container count, space usage, networks, volumes
# ════════════════════════════════════════════════════════════════════════════════
#   make info
#
info:
	@echo ""
	@echo -e "$(CYAN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo -e "$(CYAN)  CA-DQStream System Info$(NC)"
	@echo -e "$(CYAN)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo -e "$(BLUE)[Running Containers]${NC}"
	@docker ps --filter "name=ldt-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo -e "$(BLUE)[Running containers: $$(docker ps --filter 'name=ldt-' --format '{{.Names}}' | wc -l)]${NC}"
	@echo ""
	@echo -e "$(BLUE)[Docker Space Usage]${NC}"
	@docker system df
	@echo ""
	@echo -e "$(BLUE)[Networks]${NC}"
	@docker network ls --filter "name=cadqstream" --format "table {{.Name}}\t{{.Driver}}\t{{.Scope}}"
	@docker network ls --filter "name=ldt" --format "table {{.Name}}\t{{.Driver}}\t{{.Scope}}"
	@echo ""
	@echo -e "$(BLUE)[Volumes]${NC}"
	@docker volume ls --filter "name=ldt-" --format "table {{.Name}}\t{{.Mountpoint}}\t{{.Size}}"
	@echo ""
