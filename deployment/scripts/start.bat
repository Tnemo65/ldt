@echo off
setlocal enabledelayedexpansion

set "DEPLOYMENT_DIR=%~dp0.."

echo.
echo ================================================================
echo   CA-DQStream Production Deployment - Windows Startup
echo   4-Layer Streaming Pipeline on Apache Flink 1.17.1
echo ================================================================
echo.

:: ─── Step 0: Docker Reset ───────────────────────────────────────────
echo [STEP 0] Docker Reset - Cleaning environment...

if exist "%DEPLOYMENT_DIR%\docker-compose.yml" (
    docker compose -f "%DEPLOYMENT_DIR%\docker-compose.yml" down --remove-orphans 2^>nul
)

for /f "tokens=*" %%c in ('docker ps -q 2^>nul') do (
    docker stop %%c 2^>nul
)
for /f "tokens=*" %%c in ('docker ps -aq 2^>nul') do (
    docker rm -f %%c 2^>nul
)

docker network rm cadqstream-net 2^>nul
docker network rm deployment_cadqstream-net 2^>nul
docker network rm ldt_cadqstream-net 2^>nul
docker network prune -f 2^>nul

echo [OK] Docker reset complete.
echo.

:: ─── Step 0b: Check Ports ────────────────────────────────────────
echo [STEP 0b] Checking required ports...
set PORTS=2181 9092 8081 8080 8082 3000 9090 9100 9308 9000 9001
for %%p in (%PORTS%) do (
    netstat -ano 2^>nul | findstr ":%%p " >nul 2>&1
    if errorlevel 1 (
        echo [OK] Port %%p is free
    ) else (
        echo [ERROR] Port %%p is in use!
    )
)
echo.

:: ─── Step 1: Build Custom Flink Image ─────────────────────────────
echo [STEP 1] Building custom Flink image (ldt-flink:1.17.1-py)...

if not exist "%DEPLOYMENT_DIR%\flink\flink-connector-kafka-1.17.1.jar" (
    echo [ERROR] flink-connector-kafka-1.17.1.jar NOT FOUND
    exit /b 1
)
if not exist "%DEPLOYMENT_DIR%\flink\kafka-clients-3.5.1.jar" (
    echo [ERROR] kafka-clients-3.5.1.jar NOT FOUND
    exit /b 1
)
echo [OK] All required JAR files found.

echo Building image...
docker build -t ldt-flink:1.17.1-py "%DEPLOYMENT_DIR%" -f "%DEPLOYMENT_DIR%\flink\Dockerfile"
if errorlevel 1 (
    echo [ERROR] Flink image build failed!
    exit /b 1
)
echo [OK] Flink image built successfully.
echo.

:: ─── Step 2: Start Infrastructure ────────────────────────────────
echo [STEP 2] Starting infrastructure services...
docker compose -f "%DEPLOYMENT_DIR%\docker-compose.yml" up -d zookeeper kafka schema-registry
echo.

:: ─── Step 3: Start Storage ────────────────────────────────────────
echo [STEP 3] Starting storage services (MinIO only)...
docker compose -f "%DEPLOYMENT_DIR%\docker-compose.yml" up -d minio
echo.

:: ─── Step 4: Wait for Services ──────────────────────────────────
echo [STEP 4] Waiting for services to become healthy (up to 120s)...

:: Kafka: wait 40s then check
echo Waiting 40s for Kafka to initialize...
timeout /t 40 /nobreak >nul
docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>nul
if errorlevel 1 (
    echo [WARNING] Kafka not ready yet, waiting 20s more...
    timeout /t 20 /nobreak >nul
    docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>nul
)
echo [OK] Kafka ready or checked.

:: MinIO: wait 15s then check
echo Waiting 15s for MinIO to initialize...
timeout /t 15 /nobreak >nul
docker exec ldt-minio mc ready local 2>nul
if errorlevel 1 (
    echo [WARNING] MinIO not ready yet, waiting 15s more...
    timeout /t 15 /nobreak >nul
    docker exec ldt-minio mc ls local/ 2>nul
)
echo [OK] MinIO ready or checked.
echo.

:: ─── Step 5: Run Init Containers ─────────────────────────────────
echo [STEP 5] Running init containers (Kafka topics, MinIO buckets)...

docker compose -f "%DEPLOYMENT_DIR%\docker-compose.yml" up -d kafka-init
timeout /t 10 /nobreak >nul
docker compose -f "%DEPLOYMENT_DIR%\docker-compose.yml" up -d minio-init
timeout /t 15 /nobreak >nul

echo.
echo [OK] Kafka topics:
docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>nul
echo.
echo [OK] MinIO buckets:
docker exec ldt-minio mc ls local/ 2>nul
echo.

:: ─── Step 6: Start Flink ─────────────────────────────────────────
echo [STEP 6] Starting Flink services...
docker compose -f "%DEPLOYMENT_DIR%\docker-compose.yml" up -d flink-jobmanager flink-taskmanager
echo.

echo Waiting 60s for Flink REST API...
timeout /t 60 /nobreak >nul
curl -s http://localhost:8081/overview 2>nul
if errorlevel 1 (
    echo [WARNING] Flink REST not responding, waiting 30s more...
    timeout /t 30 /nobreak >nul
)
echo [OK] Flink step complete.

:: ─── Step 7: Submit Flink Job ─────────────────────────────────────
echo [STEP 7] Submitting Flink pipeline job (one-shot init container)...
docker compose -f "%DEPLOYMENT_DIR%\docker-compose.yml" up flink-init
echo.

:: ─── Step 8: Start Monitoring ────────────────────────────────────
echo [STEP 8] Starting monitoring stack...
docker compose -f "%DEPLOYMENT_DIR%\docker-compose.yml" up -d prometheus grafana kafka-exporter node-exporter
timeout /t 20 /nobreak >nul

:: ─── Final Status ────────────────────────────────────────────────
echo.
echo ================================================================
echo   DEPLOYMENT COMPLETE!
echo ================================================================
echo.
echo   Service          URL / Endpoint              Credentials
echo   -------          ---------------              ------------
echo   Kafka UI        http://localhost:8080       (no auth)
echo   Flink UI        http://localhost:8081       (no auth)
echo   Grafana         http://localhost:3000       admin / admin123
echo   Prometheus      http://localhost:9090        (no auth)
echo   MinIO Console   http://localhost:9001       minioadmin / minioadmin123
echo.
echo   Kafka Topics:
docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>nul
echo.
echo   MinIO Buckets:
docker exec ldt-minio mc ls local/ 2>nul
echo.
echo   Running Containers:
docker ps --filter "name=ldt-" --format "  %%name %%status"
echo.
echo.
echo   To start data producer:  deployment\scripts\start-producer.bat
echo   To check health:         deployment\scripts\check-health.bat
echo   To stop stack:           deployment\scripts\stop.bat
echo.

