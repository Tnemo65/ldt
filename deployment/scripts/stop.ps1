# CA-DQStream Stop Script (PowerShell)
# Stop all containers and optionally remove volumes

param(
    [switch]$RemoveVolumes
)

$DEPLOYMENT_DIR = Split-Path -Parent $PSScriptRoot

Write-Host "Stopping CA-DQStream services..."
Push-Location $DEPLOYMENT_DIR

if ($RemoveVolumes) {
    Write-Host "Removing volumes (data will be lost)..."
    docker compose -f "docker-compose.yml" down -v 2>&1
} else {
    docker compose -f "docker-compose.yml" down 2>&1
}

# Stop and remove any leftover ldt- containers
$ldtContainers = docker ps -aq --filter "name=ldt-" 2>$null
if ($ldtContainers) {
    Write-Host "Removing leftover containers..."
    docker rm -f $ldtContainers 2>&1 | Out-Null
}

# Remove producer container
docker rm -f ldt-kafka-producer 2>$null | Out-Null

Pop-Location
Write-Host "Done."
