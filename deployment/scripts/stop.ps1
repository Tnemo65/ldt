# CA-DQStream Stop Script (PowerShell)
# Stop all containers and optionally remove volumes

param(
    [switch]$RemoveVolumes,
    [switch]$CleanKafka
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

# ─── Step 1b: Kafka cleanup (only with -CleanKafka) ─────────────────
if ($CleanKafka) {
    Write-Host "[STEP 1b] Cleaning Kafka state..."

    # Bring up kafka container temporarily to run admin commands
    docker compose -f "docker-compose.yml" up -d kafka 2>&1 | Out-Null
    Start-Sleep -Seconds 5

    # Reset consumer group offsets to beginning
    $groups = @(
        "cadqstream-complete-pipeline",
        "cadqstream-meta-consumer",
        "cadqstream-iec-consumer"
    )

    foreach ($group in $groups) {
        try {
            docker exec ldt-kafka kafka-consumer-groups `
                --bootstrap-server localhost:9092 `
                --group $group `
                --reset-offsets `
                --to-earliest `
                --all-topics `
                --execute 2>&1 | Out-Null
            Write-Host "  [RESET] Consumer group: $group"
        } catch {
            Write-Host "  [SKIP] Consumer group not found or reset failed: $group"
        }
    }

    # Optionally delete and recreate topics (only if -RemoveVolumes is also used)
    if ($RemoveVolumes) {
        $topics = @(
            "taxi-nyc-raw",
            "dq-stream-processed",
            "dq-stream-anomalies",
            "dq-meta-stream",
            "dq-hard-rule-violations",
            "dq-metrics",
            "iec-action-replay"
        )

        foreach ($topic in $topics) {
            try {
                docker exec ldt-kafka kafka-topics `
                    --bootstrap-server localhost:9092 `
                    --delete `
                    --topic $topic 2>&1 | Out-Null
                Write-Host "  [DELETED] Topic: $topic"
            } catch {
                Write-Host "  [SKIP] Topic not found: $topic"
            }
        }

        Write-Host "[NOTE] Topics will be recreated on next start by kafka-init container."
    }

    Write-Host "[OK] Kafka state cleaned."
    Write-Host ""
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
Write-Host ""
Write-Host "TIP: For a clean demo reset, use:  .\stop.ps1 -CleanKafka -RemoveVolumes"
Write-Host "     Then start fresh with:         .\start.ps1"
