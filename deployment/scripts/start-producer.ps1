# CA-DQStream Data Producer Script (PowerShell)
# Generate synthetic NYC taxi data into Kafka

param(
    [int]$Messages = 10000,
    [string]$Topic = "taxi-nyc-raw",
    [string]$BootstrapServers = "kafka:9092"
)

$DEPLOYMENT_DIR = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "================================================================"
Write-Host "  CA-DQStream Data Producer"
Write-Host "  Sending $Messages messages to topic '$Topic'"
Write-Host "================================================================"
Write-Host ""

# Check if kafka-producer image exists
$img = docker images -q ldt-kafka-producer 2>$null
if (-not $img) {
    Write-Host "[BUILD] Building Kafka producer image..."
    docker build -t ldt-kafka-producer (Join-Path $DEPLOYMENT_DIR "kafka") -f (Join-Path $DEPLOYMENT_DIR "kafka\Dockerfile.producer") 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Build failed!"
        exit 1
    }
}

# Remove old producer
docker rm -f ldt-kafka-producer 2>$null | Out-Null

# Run producer
Write-Host "[RUN] Starting producer..."
docker run --network cadqstream-net --name ldt-kafka-producer -d ldt-kafka-producer python3 /app/fast_producer.py $Messages $BootstrapServers $Topic 2>&1 | Out-Null

# Wait for completion
Start-Sleep -Seconds 3

# Check logs
$log = docker logs ldt-kafka-producer 2>&1
Write-Host $log

# Show result
if ($log -match "DONE") {
    Write-Host ""
    Write-Host "[OK] Producer finished successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[INFO] Producer still running or finished." -ForegroundColor Yellow
}
