# =============================================================================
# CA-DQStream Deployment Script (PowerShell)
# 4-Layer Streaming Pipeline on Apache Flink 1.17.1 with Python support
# Run from project root: powershell -ExecutionPolicy Bypass -File deployment/scripts/start.ps1
# =============================================================================

param(
    [switch]$SkipBuild,
    [switch]$TrainModel,
    [switch]$ForceRestartFlinkJob
)

$ErrorActionPreference = "Continue"

$DEPLOYMENT_DIR = Split-Path -Parent $PSScriptRoot

Write-Host ""
Write-Host "================================================================"
Write-Host "  CA-DQStream Production Deployment - PowerShell"
Write-Host "  4-Layer Streaming Pipeline on Apache Flink 1.17.1 + Python"
Write-Host "================================================================"
Write-Host ""

# ─── Step 0: Docker Reset ───────────────────────────────────────────
Write-Host "[STEP 0] Docker Reset..."

Push-Location $DEPLOYMENT_DIR

docker compose -f "$DEPLOYMENT_DIR\docker-compose.yml" down --remove-orphans 2>$null | Out-Null

$ldtContainers = docker ps -q --filter "name=ldt-" 2>$null
foreach ($c in $ldtContainers) { docker stop $c 2>$null | Out-Null }
foreach ($c in $ldtContainers) { docker rm -f $c 2>$null | Out-Null }

@("cadqstream-net", "deployment_cadqstream-net", "ldt_cadqstream-net") | ForEach-Object {
    docker network rm $_ 2>$null | Out-Null
}

Write-Host "[OK] Docker reset complete."
Write-Host ""

# ─── Step 1: Check Required JAR Files ─────────────────────────────────────
Write-Host "[STEP 1] Checking required JAR files..."

$jars = @(
    "flink\flink-connector-kafka-1.17.1.jar",
    "flink\flink-connector-jdbc-3.1.1-1.17.jar",
    "flink\kafka-clients-3.5.1.jar"
)

$missing = $false
foreach ($j in $jars) {
    $path = Join-Path $DEPLOYMENT_DIR $j
    if (-not (Test-Path $path)) {
        Write-Host "  [MISSING] $j" -ForegroundColor Red
        $missing = $true
    } else {
        $size = (Get-Item $path).Length / 1MB
        Write-Host "  [OK] $j ($([math]::Round($size,1)) MB)"
    }
}

# S3/S3A connector JARs (required for MinIO checkpoint + lakehouse storage)
$s3Jars = @(
    "flink\flink-s3-fs-hadoop-1.17.1.jar",
    "flink\kafka-clients-3.5.1.jar"
)
foreach ($j in $s3Jars) {
    $path = Join-Path $DEPLOYMENT_DIR $j
    if (-not (Test-Path $path)) {
        Write-Host "  [MISSING - OPTIONAL] $j" -ForegroundColor Yellow
        Write-Host "    -> Download from Maven: flink-s3-fs-presto, hadoop-common, hadoop-aws"
    } else {
        $size = (Get-Item $path).Length / 1MB
        Write-Host "  [OK] $j ($([math]::Round($size,1)) MB)"
    }
}

if ($missing) {
    Write-Host "[ERROR] Required JAR files are missing. Cannot proceed."
    Write-Host "         Download them and place in deployment/flink/"
    exit 1
}

$s3Missing = $false
foreach ($j in $s3Jars) {
    $path = Join-Path $DEPLOYMENT_DIR $j
    if (-not (Test-Path $path)) {
        Write-Host "  [MISSING] $j" -ForegroundColor Yellow
        $s3Missing = $true
    }
}
if ($s3Missing) {
    Write-Host "  [WARN] S3/MinIO JARs missing - Hadoop deps must be in Flink image Dockerfile." -ForegroundColor Yellow
}
Write-Host ""

# ─── Step 2: Build Images ─────────────────────────────────────
Write-Host "[STEP 2] Building Docker images..."

# Build Flink image
if (-not $SkipBuild) {
    Write-Host "  Building ldt-flink:1.17.1-py (5-10 min on first run)..."
    $build = docker build -t ldt-flink:1.17.1-py $DEPLOYMENT_DIR -f (Join-Path $DEPLOYMENT_DIR "flink\Dockerfile") 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Flink image build failed!" -ForegroundColor Red
        Write-Host $build
        exit 1
    }
    Write-Host "  [OK] ldt-flink:1.17.1-py built."
} else {
    $img = docker images -q ldt-flink:1.17.1-py 2>$null
    if (-not $img) {
        Write-Host "[ERROR] Image ldt-flink:1.17.1-py not found. Run without -SkipBuild."
        exit 1
    }
    Write-Host "  [SKIP] Flink build skipped."
}

# Build cadqstream-metrics image (fixed Prometheus registry)
Write-Host "  Building cadqstream-metrics image..."
$metricsBuild = docker build -t cadqstream-metrics:latest (Join-Path $DEPLOYMENT_DIR "cadqstream-metrics") 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] cadqstream-metrics build failed!" -ForegroundColor Red
    Write-Host $metricsBuild
    exit 1
}
Write-Host "  [OK] cadqstream-metrics:latest built."

# Build ML service image
Write-Host "  Building ml-service image..."
$mlBuild = docker build -t ml-service:latest (Join-Path $DEPLOYMENT_DIR "ml-service") 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] ml-service build failed!" -ForegroundColor Red
    Write-Host $mlBuild
    exit 1
}
Write-Host "  [OK] ml-service:latest built."

# Build action-replay-worker image
Write-Host "  Building action-replay-worker image..."
$arwBuild = docker build -t action-replay-worker:latest (Join-Path $DEPLOYMENT_DIR "action-replay-worker") 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] action-replay-worker build failed!" -ForegroundColor Red
    Write-Host $arwBuild
    exit 1
}
Write-Host "  [OK] action-replay-worker:latest built."

# Build stats-writer image
Write-Host "  Building stats-writer image..."
$swBuild = docker build -t stats-writer:latest (Join-Path $DEPLOYMENT_DIR "stats-writer") 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] stats-writer build failed!" -ForegroundColor Red
    Write-Host $swBuild
    exit 1
}
Write-Host "  [OK] stats-writer:latest built."
Write-Host ""

# ─── Step 3: Start All Services ────────────────────────────────────
Write-Host "[STEP 3] Starting all services via docker compose..."
docker compose -f "$DEPLOYMENT_DIR\docker-compose.yml" up -d 2>&1 | Out-Null
Write-Host "[OK] All services started."
Write-Host ""

# ─── Step 4: Wait for Core Services ────────────────────────────────
Write-Host "[STEP 4] Waiting for core services..."

$services = @(
    @{Name="ldt-kafka"; Wait=180; Interval=10; Desc="Kafka"},
    @{Name="ldt-minio"; Wait=60; Interval=10; Desc="MinIO"},
    @{Name="ldt-cadqstream-metrics"; Wait=60; Interval=5; Desc="cadqstream-metrics"},
    @{Name="ldt-ml-service"; Wait=60; Interval=5; Desc="ML Service (FastAPI)"},
    @{Name="ldt-stats-writer"; Wait=30; Interval=5; Desc="Stats Writer"}
)

foreach ($svc in $services) {
    Write-Host "  Waiting for $($svc.Desc)..."
    $elapsed = 0
    $success = $false
    while ($elapsed -lt $svc.Wait) {
        $running = (docker ps --filter "name=$($svc.Name)" --format "{{.Names}}" 2>$null) -ne ""
        $healthy = (docker ps --filter "name=$($svc.Name)" --format "{{.Status}}" 2>$null) -match "healthy"

        # For cadqstream-metrics, also check HTTP endpoint
        if ($svc.Name -eq "ldt-cadqstream-metrics" -and $running) {
            try {
                $healthResp = Invoke-WebRequest -Uri "http://localhost:9250/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
                if ($healthResp.StatusCode -eq 200) {
                    $healthy = $true
                    # Check metrics endpoint has cadqstream_ metrics
                    $metricsResp = Invoke-WebRequest -Uri "http://localhost:9250/metrics" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
                    if ($metricsResp.Content -match "cadqstream_records_valid_total") {
                        Write-Host "    [METRICS OK] cadqstream_metrics endpoint verified."
                    }
                }
            } catch {}
        }

        if ($running -and $healthy) {
            Write-Host "  [OK] $($svc.Desc) is healthy."
            $success = $true
            break
        }
        Start-Sleep -Seconds $svc.Interval
        $elapsed += $svc.Interval
    }
    if (-not $success) {
        Write-Host "  [WARN] $($svc.Desc) did not become healthy in ${$svc.Wait}s." -ForegroundColor Yellow
        docker logs $svc.Name --tail 10 2>$null | ForEach-Object { Write-Host "    $_" }
    }
}
Write-Host ""

# ─── Step 5: Wait for Init Containers ──────────────────────────────
Write-Host "[STEP 5] Waiting for init containers..."

$initContainers = @("ldt-kafka-init", "ldt-minio-init")
foreach ($c in $initContainers) {
    Write-Host "  Checking $c..."
    $maxWait = 120
    $elapsed = 0
    $done = $false
    while ($elapsed -lt $maxWait) {
        $state = docker inspect --format='{{.State.Status}}' $c 2>$null
        if ($state -eq "exited") {
            Write-Host "  [OK] $c completed."
            $done = $true
            break
        }
        # Also check if container doesn't exist (it may have completed and been removed)
        if (-not $state) {
            Write-Host "  [OK] $c already finished (container removed)."
            $done = $true
            break
        }
        Start-Sleep -Seconds 5
        $elapsed += 5
    }
    if (-not $done) {
        Write-Host "  [WARN] $c did not complete in ${maxWait}s." -ForegroundColor Yellow
    }
}
Write-Host ""

# ─── Step 6: Wait for Flink REST API ──────────────────────────────
Write-Host "[STEP 6] Waiting for Flink REST API..."
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
    Write-Host "[WARN] Flink REST API did not respond." -ForegroundColor Yellow
    docker logs ldt-flink-jobmanager --tail 20 2>$null | ForEach-Object { Write-Host "  $_" }
}
Write-Host ""

# ─── Step 7: Install Python Dependencies in Flink ────────────────────
Write-Host "[STEP 7] Installing Python dependencies..."

$deps = @(
    @{Name="river"; Version="0.21.0"},
    @{Name="boto3"; Version=""},
    @{Name="minio"; Version=""}
)

foreach ($dep in $deps) {
    $verArg = if ($dep.Version) { "==$($dep.Version)" } else { "" }
    $installed = docker exec ldt-flink-jobmanager bash -c "python3 -c 'import $($dep.Name)'" 2>$null
    if (-not $installed) {
        Write-Host "  Installing $($dep.Name)..."
        docker exec ldt-flink-jobmanager bash -c "pip3 install --quiet $($dep.Name)$verArg" 2>$null | Out-Null
        docker exec ldt-flink-taskmanager bash -c "pip3 install --quiet $($dep.Name)$verArg" 2>$null | Out-Null
    } else {
        Write-Host "  [SKIP] $($dep.Name) already installed."
    }
}
Write-Host ""

# ─── Step 8: Submit / Restart Flink Job ─────────────────────────────────
Write-Host "[STEP 8] Submitting Flink pipeline job..."

$sourceMounted = docker exec ldt-flink-jobmanager bash -c "test -f /opt/flink/e2e/src/flink_job_complete.py && echo YES" 2>$null
if ($sourceMounted -notmatch "YES") {
    Write-Host "[ERROR] Source files not mounted! Check docker-compose.yml volumes." -ForegroundColor Red
} else {
    Write-Host "  [OK] Source files mounted correctly."

    # Cancel existing job if requested or if job is in a bad state
    $existing = docker exec ldt-flink-jobmanager bash -c "curl -s http://localhost:8081/jobs 2>&1" 2>$null
    try {
        $jobsData = $existing | ConvertFrom-Json
        $runningJobs = @()
        foreach ($job in $jobsData.jobs) {
            if ($job.state -eq "RUNNING" -or $job.state -eq "FAILING") {
                $runningJobs += $job
            }
        }
    } catch {
        $runningJobs = @()
    }

    # Force restart: cancel all running jobs first
    if ($ForceRestartFlinkJob -and $runningJobs.Count -gt 0) {
        Write-Host "  [INFO] Force restart: canceling existing jobs..."
        foreach ($job in $runningJobs) {
            $jobId = $job.id
            Write-Host "    Canceling: $($jobId.Substring(0,16))... (state: $($job.state))"
            docker exec ldt-flink-jobmanager bash -c "curl -s -X PATCH http://localhost:8081/jobs/$jobId" 2>$null | Out-Null
            Start-Sleep -Seconds 2
        }
        Write-Host "  [OK] Existing jobs cancelled."
    } elseif ($runningJobs.Count -gt 0) {
        Write-Host "  [INFO] $($runningJobs.Count) job(s) already running. Skipping submission."
        foreach ($job in $runningJobs) {
            Write-Host "    $($job.id.Substring(0,16))... - $($job.state)"
        }
    }

    # Submit new job (only if no jobs are running)
    $checkAgain = docker exec ldt-flink-jobmanager bash -c "curl -s http://localhost:8081/jobs 2>&1" 2>$null
    try {
        $jobsData2 = $checkAgain | ConvertFrom-Json
        $activeJobs = @($jobsData2.jobs | Where-Object { $_.state -eq "RUNNING" })
    } catch {
        $activeJobs = @()
    }

    if ($activeJobs.Count -eq 0) {
        Write-Host "  Submitting job..."
        $submitResult = docker exec ldt-flink-jobmanager bash -c "
            export PYTHONPATH=/opt/flink/pyflink_extracted:/opt/flink/opt/python/py4j-0.10.9.7-src.zip:/opt/flink/opt/python/cloudpickle-2.2.0-src.zip:/opt/flink/e2e &&
            cd /opt/flink/e2e &&
            flink run -d -pyfs /opt/flink/e2e -python /opt/flink/e2e/src/flink_job_complete.py 2>&1
        " 2>$null

        if ($submitResult -match "Job has been submitted" -or $submitResult -match "Job ID") {
            Write-Host "  [OK] Flink job submitted!"
            Write-Host "    $submitResult"
        } else {
            Write-Host "  [WARN] Submission output: $submitResult" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  [INFO] Active jobs detected, skipping submission."
    }
}
Write-Host ""

# ─── Step 9: Verify cadqstream-metrics Endpoint ─────────────────────────
Write-Host "[STEP 9] Verifying cadqstream-metrics endpoint..."

try {
    $metricsResp = Invoke-WebRequest -Uri "http://localhost:9250/metrics" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($metricsResp.StatusCode -eq 200) {
        $content = $metricsResp.Content
        $hasValid = $content -match "cadqstream_records_valid_total"
        $hasViolation = $content -match "cadqstream_records_violation_total"
        $hasTypeLine = $content -match "# TYPE cadqstream_records_valid_total"
        Write-Host "  [OK] /metrics endpoint responds with 200."
        Write-Host "    cadqstream_records_valid_total: $(if($hasValid){'[YES]'}else{'[MISSING]'})"
        Write-Host "    cadqstream_records_violation_total: $(if($hasViolation){'[YES]'}else{'[MISSING]'})"
        Write-Host "    # TYPE lines: $(if($hasTypeLine){'[YES]'}else{'[MISSING]'})"
        if (-not $hasTypeLine) {
            Write-Host "  [ERROR] Prometheus TYPE lines missing! Check cadqstream-metrics logs." -ForegroundColor Red
            docker logs ldt-cadqstream-metrics --tail 20 2>$null | ForEach-Object { Write-Host "    $_" }
        }
    } else {
        Write-Host "  [ERROR] /metrics returned $($metricsResp.StatusCode)" -ForegroundColor Red
    }
} catch {
    Write-Host "  [ERROR] Cannot reach cadqstream-metrics:9250/metrics" -ForegroundColor Red
    docker logs ldt-cadqstream-metrics --tail 10 2>$null | ForEach-Object { Write-Host "    $_" }
}
Write-Host ""

# ─── Step 10: Train ML Model (optional) ─────────────────────────────
if ($TrainModel) {
    Write-Host "[STEP 10] Training ML model..."

    } else {
        Write-Host "  [WARN] ML Service not ready. Skipping model training." -ForegroundColor Yellow
    }
}
Write-Host ""

# ─── Step 11: Final Verification ─────────────────────────────────────
Write-Host "[STEP 11] Final verification..."
Write-Host ""
Write-Host "[INFO] Running containers:"
docker ps --filter "name=ldt-" --format "  {0,-35} {1}" 2>$null | ForEach-Object { Write-Host "  $_" }
Write-Host ""

Write-Host "[INFO] Kafka topics:"
docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>$null | ForEach-Object { Write-Host "  $_" }
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

Write-Host "[INFO] cadqstream-metrics quick check:"
try {
    $m = Invoke-WebRequest -Uri "http://localhost:9250/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
    Write-Host "  /health: $($m.Content)"
} catch {
    Write-Host "  [ERROR] cadqstream-metrics not reachable." -ForegroundColor Red
}
Write-Host ""

# ─── Summary ──────────────────────────────────────────────────────
Write-Host "================================================================"
Write-Host "  DEPLOYMENT COMPLETE!"
Write-Host "================================================================"
Write-Host ""
Write-Host "  Service             URL / Port                  Credentials"
Write-Host "  -------             ---------------              ------------"
Write-Host "  Kafka UI            http://localhost:8080       (no auth)"
Write-Host "  Flink UI            http://localhost:8081       (no auth)"
Write-Host "  Grafana             http://localhost:3000       admin / admin123"
Write-Host "  Prometheus          http://localhost:9090       (no auth)"
Write-Host "  MinIO Console       http://localhost:9001       minioadmin / minioadmin123"
Write-Host "  ML Service          http://localhost:8000       FastAPI"
Write-Host "  cadqstream-metrics  localhost:9250/metrics      Prometheus scrape target"
Write-Host ""
Write-Host "  Next steps:"
Write-Host "    1. Open Grafana at http://localhost:3000 (admin/admin123)"
Write-Host "    2. Wait 1-2 minutes for data to flow through pipeline"
Write-Host "    3. Check MinIO data: docker exec ldt-minio mc ls local/raw-zone/"
Write-Host "    4. Check metrics: curl http://localhost:9250/metrics | findstr cadqstream"
Write-Host "    5. Check IEC actions: docker logs ldt-action-replay-worker --tail 20"
Write-Host "    6. Inject drift: kafka-producer should be running with anomaly data"
Write-Host ""
Write-Host "  To restart Flink job: powershell -File deployment/scripts/start.ps1 -ForceRestartFlinkJob"
Write-Host "  To stop:             powershell -File deployment/scripts/stop.ps1"
Write-Host ""

Pop-Location
