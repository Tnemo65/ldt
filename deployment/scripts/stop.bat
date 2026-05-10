@echo off
setlocal enabledelayedexpansion

set "DEPLOYMENT_DIR=%~dp0.."

echo.
echo ================================================================
echo   CA-DQStream - Stop Script
echo ================================================================
echo.

echo Stopping all containers...
docker compose -f "%DEPLOYMENT_DIR%\docker-compose.yml" down 2>nul

echo.
echo CA-DQStream stack stopped.
echo To start again: deployment\scripts\start.bat
echo To remove volumes: deployment\scripts\stop.bat --remove-volumes
