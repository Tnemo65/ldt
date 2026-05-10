@echo off
setlocal enabledelayedexpansion

set "DEPLOYMENT_DIR=%~dp0.."

echo.
echo ================================================================
echo   CA-DQStream - Stop Script (with volumes)
echo ================================================================
echo.

echo Stopping and removing all containers...
docker compose -f "%DEPLOYMENT_DIR%\docker-compose.yml" down -v --remove-orphans 2>nul

echo Removing network...
docker network rm cadqstream-net 2>nul

echo.
echo CA-DQStream stack stopped with ALL volumes deleted.
echo All data is lost - fresh start next time.
echo.
