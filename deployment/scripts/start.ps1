# CA-DQStream Deployment Script (PowerShell)
# 4-Layer Streaming Pipeline on Apache Flink 1.17.1
# Run from project root: powershell -ExecutionPolicy Bypass -File deployment/scripts/start.ps1

param(
    [switch]$SkipBuild,
    [int]$ProducerMessages = 0,
    [switch]$TrainModel
)

$ErrorActionPreference = "Continue"

$DEPLOYMENT_DIR = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "================================================================"
Write-Host "  CA-DQStream Production Deployment - PowerShell"
Write-Host "  4-Layer Streaming Pipeline on Apache Flink 1.17.1"
Write-Host "================================================================"
Write-Host ""

# ─── Step 0: Docker Reset ───────────────────────────────────────────
Write-Host "[STEP 0] Docker Reset..."

Push-Location $DEPLOYMENT_DIR

# Stop all containers
docker compose -f "docker-compose.yml" down --remove-orphans 2>$null | Out-Null

$ldtContainers = docker ps -q --filter "name=ldt-" 2>$null
foreach ($c in $ldtContainers) { docker stop $c 2>$null | Out-Null }
foreach ($c in $ldtContainers) { docker rm -f $c 2>$null | Out-Null }

# Remove old networks
@("cadqstream-net", "deployment_cadqstream-net", "ldt_cadqstream-net") | ForEach-Object {
    docker network rm $_ 2>$null | Out-Null
}

Write-Host "[OK] Docker reset complete."
Write-Host ""

# ─── Step 1: Build Flink Image ─────────────────────────────────────
Write-Host "[STEP 1] Building custom Flink image..."

$jars = @(
    "flink\flink-connector-kafka-1.17.1.jar",
    "flink\flink-connector-jdbc-3.1.1-1.17.jar",
    "flink\kafka-clients-3.5.1.jar",
    "flink\postgresql-42.6.0.jar"
)
foreach ($j in $jars) {
    if (-not (Test-Path (Join-Path $DEPLOYMENT_DIR $j))) {
        Write-Host "[ERROR] NOT FOUND: $j"
        exit 1
    }
}

if (-not $SkipBuild) {
    Write-Host "Building (3-5 min on first run)..."
    $build = docker build -t ldt-flink:1.17.1-py $DEPLOYMENT_DIR -f (Join-Path $DEPLOYMENT_DIR "flink\Dockerfile") 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] Build failed!"; exit 1 }
    Write-Host "[OK] Image built."
} else {
    Write-Host "[SKIP] Build skipped."
}
Write-Host ""

# ─── Step 2: Start All Services ────────────────────────────────────
Write-Host "[STEP 2] Starting all services..."

docker compose -f "docker-compose.yml" up -d 2>&1 | Out-Null
Write-Host "[OK] All services started."
Write-Host ""

# ─── Step 3: Wait for Core Services ────────────────────────────────
Write-Host "[STEP 3] Waiting for services (5 min total)..."

$services = @(
    @{Name="ldt-kafka"; Wait=180; Interval=10; Desc="Kafka"},
    @{Name="ldt-postgres"; Wait=120; Interval=10; Desc="PostgreSQL"},
    @{Name="ldt-minio"; Wait=60; Interval=10; Desc="MinIO"},
    @{Name="ldt-cadqstream-metrics"; Wait=30; Interval=5; Desc="cadqstream-metrics"}
)

foreach ($svc in $services) {
    Write-Host "  Waiting for $($svc.Desc)..."
    $elapsed = 0
    while ($elapsed -lt $svc.Wait) {
        $running = (docker ps --filter "name=$($svc.Name)" --format "{{.Names}}" 2>$null) -ne ""
        $healthy = (docker ps --filter "name=$($svc.Name)" --format "{{.Status}}" 2>$null) -match "healthy"
        if ($running -and $healthy) {
            Write-Host "  [OK] $($svc.Desc) is healthy."
            break
        }
        Start-Sleep -Seconds $svc.Interval
        $elapsed += $svc.Interval
    }
    if ($elapsed -ge $svc.Wait) {
        Write-Host "  [WARN] $($svc.Desc) did not become healthy in ${$svc.Wait}s."
    }
}

Write-Host ""

# ─── Step 4: Run Init Containers ───────────────────────────────────
Write-Host "[STEP 4] Running init containers..."

docker compose -f "docker-compose.yml" up -d kafka-init 2>&1 | Out-Null
Start-Sleep -Seconds 10

docker compose -f "docker-compose.yml" up -d minio-init 2>&1 | Out-Null
Start-Sleep -Seconds 15

Write-Host ""

# ─── Step 5: Wait for Flink ────────────────────────────────────────
Write-Host "[STEP 5] Waiting for Flink REST API..."
$flinkReady = $false
$elapsed = 0
while ($elapsed -lt 180) {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:8081/overview" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($resp.StatusCode -eq 200) {
            $flinkReady = $true
            break
        }
    } catch {}
    Start-Sleep -Seconds 10
    $elapsed += 10
    Write-Host "  Waiting... ${elapsed}s"
}
if ($flinkReady) {
    Write-Host "[OK] Flink REST API ready."
} else {
    Write-Host "[WARN] Flink REST API did not respond."
}
Write-Host ""

# ─── Step 6: Submit Flink Job ──────────────────────────────────────
Write-Host "[STEP 6] Submitting Flink pipeline job..."

$sourceMounted = docker exec ldt-flink-jobmanager bash -c "test -f /opt/flink/e2e/src/flink_job_complete.py && echo YES" 2>$null
if ($sourceMounted -notmatch "YES") {
    Write-Host "[ERROR] Source files not mounted in Flink container!"
    Write-Host "  The docker-compose.yml volume paths need to point to C:/proj/ldt/src"
    Write-Host "  Check that 'C:/proj/ldt/src:/opt/flink/e2e/src:ro' is in flink-jobmanager volumes"
} else {
    Write-Host "  Source files mounted correctly."

    # Check river package
    $riverOK = docker exec ldt-flink-jobmanager bash -c "python3 -c 'import river; print(river.__version__)'" 2>$null
    if ($riverOK -notmatch "^\d") {
        Write-Host "  Installing river package..."
        docker exec ldt-flink-jobmanager bash -c "pip3 install --upgrade setuptools && pip3 install river==0.21.0" 2>$null | Out-Null
        docker exec ldt-flink-taskmanager bash -c "pip3 install --upgrade setuptools && pip3 install river==0.21.0" 2>$null | Out-Null
    }

    # Check for existing jobs
    $existing = docker exec ldt-flink-jobmanager bash -c "curl -s http://localhost:8081/jobs 2>&1" 2>$null
    $jobCount = ($existing | ConvertFrom-Json).jobs.Count
    if ($jobCount -gt 0) {
        Write-Host "  [INFO] $jobCount job(s) already running. Skipping submission."
    } else {
        Write-Host "  Submitting job..."
        $result = docker exec ldt-flink-jobmanager bash -c "cd /opt/flink && flink run -d -pyfs /opt/flink/e2e -python /opt/flink/e2e/src/flink_job_complete.py 2>&1" 2>$null
        if ($result -match "Job has been submitted") {
            Write-Host "  [OK] Flink job submitted successfully!"
        } else {
            Write-Host "  [WARN] Submission result: $result"
        }
    }
}
Write-Host ""

# ─── Step 6b: Train ML Model (optional) ─────────────────────────────
if ($TrainModel) {
    Write-Host "[STEP 6b] Training ML model..."

    # Run training inside the Flink jobmanager container (has all Python deps)
    $trainResult = docker exec ldt-flink-jobmanager bash -c "cd /opt/flink/e2e && python3 scripts/train_model.py --version v1 --n-samples 50000 2>&1" 2>$null
    if ($trainResult -match "Training Complete" -or $LASTEXITCODE -eq 0) {
        Write-Host "  [OK] ML model trained and uploaded to MinIO."
    } else {
        Write-Host "  [WARN] Model training output: $trainResult"
    }

    # ─── Step 6c: Broadcast model to Kafka ───────────────────────────
    Write-Host "[STEP 6c] Broadcasting model to pipeline..."
    $loadResult = docker exec ldt-flink-jobmanager bash -c "cd /opt/flink/e2e && python3 scripts/load_model_to_broadcast.py --version v1 --bootstrap kafka:9092 2>&1" 2>$null
    if ($loadResult -match "SUCCESS" -or $LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Model broadcast to Kafka."
    } else {
        Write-Host "  [WARN] Model broadcast output: $loadResult"
    }
}
Write-Host ""

# ─── Step 7: Start Monitoring ───────────────────────────────────────
Write-Host "[STEP 7] Starting monitoring stack..."
Start-Sleep -Seconds 10
Write-Host "[OK] Monitoring stack started (started with all services)."
Write-Host ""

# ─── Step 8: Run Data Producer (optional) ─────────────────────────
if ($ProducerMessages -gt 0) {
    Write-Host "[STEP 8] Starting data producer..."

    # Build producer if needed
    $producerImg = docker images -q ldt-kafka-producer 2>$null
    if (-not $producerImg) {
        Write-Host "  Building producer image..."
        docker build -t ldt-kafka-producer (Join-Path $DEPLOYMENT_DIR "kafka") -f (Join-Path $DEPLOYMENT_DIR "kafka\Dockerfile.producer") 2>&1 | Out-Null
    }

    # Remove old producer
    docker rm -f ldt-kafka-producer 2>$null | Out-Null

    # Run producer
    docker run --network cadqstream-net --name ldt-kafka-producer -d ldt-kafka-producer python3 /app/fast_producer.py $ProducerMessages kafka:9092 taxi-nyc-raw 2>&1 | Out-Null
    Start-Sleep -Seconds 3

    $log = docker logs ldt-kafka-producer 2>&1
    Write-Host "  Producer: $log"
}
Write-Host ""

# ─── Step 9: Final Verification ─────────────────────────────────────
Write-Host "[STEP 9] Final verification..."
Write-Host ""
Write-Host "[INFO] Running containers:"
docker ps --filter "name=ldt-" --format "  {0,-35} {1}" --filter "name=ldt-" 2>$null | ForEach-Object { Write-Host "  $_" }
Write-Host ""

Write-Host "[INFO] Kafka topics:"
docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>$null | ForEach-Object { Write-Host "  $_" }
Write-Host ""

Write-Host "[INFO] PostgreSQL tables:"
docker exec ldt-postgres psql -U cadqstream -d dq_pipeline -c "\dt" 2>$null | Select-Object -Last 12 | ForEach-Object { Write-Host "  $_" }
Write-Host ""

Write-Host "[INFO] MinIO buckets:"
docker exec ldt-minio mc ls local/ 2>$null | ForEach-Object { Write-Host "  $_" }
Write-Host ""

Write-Host "[INFO] Flink jobs:"
try {
    $jobs = Invoke-WebRequest -Uri "http://localhost:8081/jobs" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
    if ($jobs.jobs.Count -gt 0) {
        $jobs.jobs | ForEach-Object { Write-Host "  $($_.id.Substring(0,16))... - $($_.state)" }
    } else {
        Write-Host "  No jobs running."
    }
} catch {
    Write-Host "  (Flink REST not accessible)"
}
Write-Host ""

# ─── Summary ──────────────────────────────────────────────────────
Write-Host "================================================================"
Write-Host "  DEPLOYMENT COMPLETE!"
Write-Host "================================================================"
Write-Host ""
Write-Host "  Service          URL / Port                Credentials"
Write-Host "  -------          ---------------              ------------"
Write-Host "  Kafka UI        http://localhost:8080       (no auth)"
Write-Host "  Flink UI        http://localhost:8081       (no auth)"
Write-Host "  Grafana         http://localhost:3000      admin / admin123"
Write-Host "  Prometheus      http://localhost:9090        (no auth)"
Write-Host "  MinIO Console   http://localhost:9001      minioadmin / minioadmin123"
Write-Host "  MLflow          http://localhost:5000       (no auth)"
Write-Host "  PostgreSQL      localhost:5432             cadqstream / cadqstream123"
Write-Host "  Kafka           localhost:9092              (no auth)"
Write-Host ""
Write-Host "  Kafka Topics: taxi-nyc-raw, dq-stream-processed, dq-stream-anomalies,"
Write-Host "    dq-meta-stream, dq-hard-rule-violations, iec-action-replay, dq-metrics"
Write-Host ""
Write-Host "  To start producer:  docker run --network cadqstream-net --name ldt-kafka-producer"
Write-Host "                        -d ldt-kafka-producer python3 /app/fast_producer.py"
Write-Host "                        10000 kafka:9092 taxi-nyc-raw"
Write-Host ""
Write-Host "  To check health:   powershell -File deployment/scripts/check-health.ps1"
Write-Host "  To stop:           powershell -File deployment/scripts/stop.ps1"
Write-Host ""
Write-Host "  Optional flags:"
Write-Host "    -SkipBuild      Skip Docker image build"
Write-Host "    -TrainModel     Train ML model and broadcast to pipeline"
Write-Host "    -ProducerMsgs N Start producer with N messages"
Write-Host ""

Pop-Location
