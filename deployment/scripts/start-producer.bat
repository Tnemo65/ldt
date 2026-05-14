@echo off
setlocal enabledelayedexpansion

set "DEPLOYMENT_DIR=%~dp0.."

echo.
echo ================================================================
echo   CA-DQStream - Start Data Producer
echo ================================================================
echo.

echo Starting Kafka data producer (synthetic NYC taxi data)...

:: Check if Kafka is running
docker ps --filter "name=ldt-kafka" --filter "status=running" --format "%%{Names}" | findstr /C:"ldt-kafka" >nul
if errorlevel 1 (
    echo [ERROR] Kafka is not running. Please run start.bat first.
    exit /b 1
)

:: Build producer image
echo Building producer image...
docker build -t ldt-producer:latest "%DEPLOYMENT_DIR%\kafka" -f "%DEPLOYMENT_DIR%\kafka\Dockerfile.producer"
if errorlevel 1 (
    echo [ERROR] Producer image build failed!
    exit /b 1
)

:: Run producer
echo Running producer - sending 100,000 taxi records to Kafka...
docker run --rm --network cadqstream-net `
    -e KAFKA_BOOTSTRAP=kafka:9092 `
    -e TOPIC=taxi-nyc-raw `
    -e MESSAGES=100000 `
    ldt-producer:latest

echo.
echo [OK] Data producer complete!
echo.
echo Now the Flink pipeline should be consuming data.
echo Check Grafana at http://localhost:3000 to see pipeline metrics.
echo Check Flink UI at http://localhost:8081 to see job status.
echo Check MinIO data: docker exec ldt-minio mc ls local/raw-zone/
