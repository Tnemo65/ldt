@echo off
REM ============================================================
REM CA-DQStream Startup Script (Windows)
REM ============================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ============================================================
echo   CA-DQStream + MemStream Deployment
echo ============================================================
echo.

REM Check for .env file
if not exist .env (
    echo Warning: .env file not found!
    echo Creating from .env.example...
    copy .env.example .env >nul
    echo.
    echo ============================================================
    echo ERROR: Please edit .env and set all required secrets!
    echo ============================================================
    echo Required variables:
    echo   - MEMSTREAM_MODEL_SIGNING_KEY
    echo   - IEC_SIGNING_KEY
    echo   - REDIS_PASSWORD
    echo   - MINIO_SECRET_KEY
    echo.
    echo Generate keys with: openssl rand -hex 32
    echo ============================================================
    exit /b 1
)

REM Check Docker
where docker >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed!
    exit /b 1
)

where docker compose >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Compose is not installed!
    exit /b 1
)

echo Starting CA-DQStream...
echo.

REM Start the cluster
docker compose up -d

echo.
echo ============================================================
echo   Services Started!
echo ============================================================
echo.
echo Access URLs:
echo   - Kafka UI:        http://localhost:8080
echo   - Flink Dashboard: http://localhost:8081
echo   - MinIO Console:   http://localhost:9001
echo   - Grafana:        http://localhost:3000
echo   - Prometheus:     http://localhost:9090
echo   - ML Service:     http://localhost:8000
echo.
echo To view logs: docker compose logs -f
echo To stop:     docker compose down
echo ============================================================

endlocal
