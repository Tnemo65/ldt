@echo off
setlocal enabledelayedexpansion

set "DEPLOYMENT_DIR=%~dp0.."

echo.
echo ================================================================
echo   CA-DQStream - Health Check (MinIO-only storage)
echo ================================================================
echo.

set HEALTHY=0
set WARNINGS=0
set UNHEALTHY=0

:: Helper to check service
goto :skip_helper

:check_container
set "NAME=%~1"
set "PORT=%~2"
set "CMD=%~3"

docker ps --filter "name=%NAME%" --filter "status=running" --format "%%{Names}" | findstr /C:"%NAME%" >nul
if errorlevel 1 (
    docker ps --filter "name=%NAME%" --format "%%{Names}" | findstr /C:"%NAME%" >nul
    if errorlevel 1 (
        echo   %NAME%: NOT FOUND ^(unhealthy^)
        set /a UNHEALTHY+=1
    ) else (
        echo   %NAME%: STOPPED ^(unhealthy^)
        set /a UNHEALTHY+=1
    )
) else (
    if not "%CMD%"=="" (
        %CMD% >nul 2>&1
        if errorlevel 1 (
            echo   %NAME%: RUNNING ^(app starting^)
            set /a WARNINGS+=1
        ) else (
            echo   %NAME%: HEALTHY
            set /a HEALTHY+=1
        )
    ) else (
        echo   %NAME%: RUNNING
        set /a HEALTHY+=1
    )
)
exit /b 0

:skip_helper

:: Kafka Infrastructure
echo [Kafka Infrastructure]
call :check_container ldt-zookeeper "" "docker exec ldt-zookeeper bash -c echo ruok^|nc localhost 2181 2^>nul"
call :check_container ldt-kafka "" "docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list"
call :check_container ldt-schema-registry "" "curl -s http://localhost:8081/subjects"
call :check_container ldt-kafka-ui "" "curl -s http://localhost:8080"
call :check_container ldt-kafka-exporter "" "curl -s http://localhost:9308/metrics"
echo.

:: Storage (MinIO only)
echo [Storage - MinIO Only]
call :check_container ldt-minio "" "docker exec ldt-minio mc ready local"
echo.

:: Streaming
echo [Streaming - Flink]
call :check_container ldt-flink-jobmanager "" "curl -s http://localhost:8081/overview"
call :check_container ldt-flink-taskmanager "" ""
call :check_container ldt-flink-init "" ""
echo.

:: ML Platform
echo [ML Platform]
call :check_container ldt-mlflow "" "curl -s http://localhost:5000"
echo.

:: Observability
echo [Observability]
call :check_container ldt-prometheus "" "curl -s http://localhost:9090/-/healthy"
call :check_container ldt-grafana "" "curl -s http://localhost:3000/api/health"
call :check_container ldt-node-exporter "" "curl -s http://localhost:9100/metrics"
echo.

:: Kafka Topics
echo [Kafka Topics]
docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>nul | findstr /v "^$" | sort
if errorlevel 1 echo   ^(no topics found^)
echo.

:: MinIO Buckets
echo [MinIO Buckets]
docker exec ldt-minio mc ls local/ 2>nul
if errorlevel 1 echo   ^(could not list buckets^)
echo.

:: Summary
echo ================================================================
echo   Health: %HEALTHY% healthy, %WARNINGS% warnings, %UNHEALTHY% unhealthy
echo ================================================================
if %UNHEALTHY% gtr 0 (
    echo   Some services are unhealthy.
    exit /b 1
) else if %WARNINGS% gtr 0 (
    echo   All containers running, some services may still be initializing.
    exit /b 0
) else (
    echo   All services are healthy!
    exit /b 0
)
