.PHONY: help install install-dev test lint format clean docker-up docker-down

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies (production)
	python -m pip install --upgrade pip
	pip install -e .

install-dev:  ## Install dependencies (development)
	python -m pip install --upgrade pip
	pip install -e ".[dev,notebook]"

test:  ## Run tests
	pytest test/

test-cov:  ## Run tests with coverage
	pytest test/ --cov=src --cov-report=html

lint:  ## Run linters
	ruff check src/ test/
	black --check src/ test/

format:  ## Format code
	black src/ test/
	ruff check --fix src/ test/

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov/ .coverage

docker-up:  ## Start Docker services
	docker compose up -d

docker-down:  ## Stop Docker services
	docker compose down -v

validate-phase0:  ## Run Phase 0 validation
	python scripts/validate_phase0.py

setup-db:  ## Initialize PostgreSQL schema
	./scripts/init_postgres.sh

setup-kafka:  ## Create Kafka topics
	./scripts/create_topics.sh

setup-schema:  ## Register Avro schema
	python scripts/register_schema.py

flink-run:  ## Run Flink job
	python src/flink_job.py
