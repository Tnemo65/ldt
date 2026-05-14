# =============================================================================
# CA-DQStream Full Deployment + E2E Demo Script (PowerShell)
# Storage: MinIO only
# Pipeline: Kafka -> Flink 4-Layer -> MinIO Lakehouse
# Run from project root:
#   powershell -ExecutionPolicy Bypass -File deployment/scripts/run_full_demo.ps1
# =============================================================================

param(
    [switch]$SkipBuild,          # Skip Docker image builds
    [switch]$SkipTraining,        # Skip ML model training
    [switch]$SkipFlinkSubmit,    # Skip Flink job submission
    [switch]$AnomalyBurst,       # Run anomaly burst demo after startup
    [switch]$AnomalyDrift,       # Run drift injection demo after startup
    [switch]$ContinuousDemo,      # Run continuous anomaly injection demo
    [switch]$ForceRestart        # Force restart all containers
)

$ErrorActionPreference = "Continue"

$DEPLOYMENT_DIR = Split-Path -Parent $PSScriptRoot
$COMPOSE_FILE = Join-Path $DEPLOYMENT_DIR "docker-compose.yml"
Push-Location $DEPLOYMENT_DIR

# =============================================================================
# HELPERS
# =============================================================================

function Write-Banner {
    param([string]$Text)
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$Text)
    Write-Host "[STEP] $Text" -ForegroundColor Yellow
}

function Wait-ForContainer {
    param([string]$Name, [string]$Desc, [int]$MaxWait=120, [int]$Interval=5)
    Write-Host "  Waiting for $Desc..." -NoNewline
    $elapsed = 0
    while ($elapsed -lt $MaxWait) {
        $running = (docker ps --filter "name=$Name" --format "{{.Names}}" 2>$null) -ne ""
        $healthy = (docker ps --filter "name=$Name" --format "{{.Status}}" 2>$null) -match "healthy"
        if ($running -and $healthy) {
            Write-Host " [OK]" -ForegroundColor Green
            return $true
        }
        Start-Sleep -Seconds $Interval
        $elapsed += $Interval
    }
    Write-Host " [WARN - running but not healthy]" -ForegroundColor Yellow
    return $false
}

function Wait-ForHttp {
    param([string]$Url, [string]$Desc, [int]$MaxWait=60, [int]$Interval=5)
    Write-Host "  Waiting for $Desc..." -NoNewline
    $elapsed = 0
    while ($elapsed -lt $MaxWait) {
        try {
            $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($resp.StatusCode -eq 200) {
                Write-Host " [OK]" -ForegroundColor Green
                return $true
            }
        } catch {}
        Start-Sleep -Seconds $Interval
        $elapsed += $Interval
    }
    Write-Host " [TIMEOUT]" -ForegroundColor Red
    return $false
}

function Get-MinioBucket {
    # Returns bucket listing from MinIO via mc alias "local"
    docker exec ldt-minio mc ls local/ 2>$null
}

function Get-MinioPath {
    param([string]$Bucket, [string]$Path)
    docker exec ldt-minio mc ls "local/$Bucket/$Path" 2>$null
}

# =============================================================================
# PHASE 0: DOCKER RESET
# =============================================================================
Write-Banner "PHASE 0: Docker Reset"

if ($ForceRestart) {
    Write-Step "Stopping all containers and cleaning networks..."
    docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>$null | Out-Null
    $ldtContainers = docker ps -aq --filter "name=ldt-" 2>$null
    foreach ($c in $ldtContainers) { docker rm -f $c 2>$null | Out-Null }
    @("cadqstream-net", "deployment_cadqstream-net", "ldt_cadqstream-net") | ForEach-Object {
        docker network rm $_ 2>$null | Out-Null
    }
    Write-Host "  [OK] All containers and networks cleaned."
} else {
    Write-Host "  [SKIP] Use -ForceRestart to stop all containers first."
}

# =============================================================================
# PHASE 1: BUILD DOCKER IMAGES
# =============================================================================
Write-Banner "PHASE 1: Build Docker Images"

if (-not $SkipBuild) {
    Write-Step "Building ldt-flink:1.17.1-py (first run: 5-10 min)..."
    $build = docker build -t ldt-flink:1.17.1-py $DEPLOYMENT_DIR -f "$DEPLOYMENT_DIR\flink\Dockerfile" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Flink image build failed!" -ForegroundColor Red
        Write-Host $build
        exit 1
    }
    Write-Host "  [OK] ldt-flink:1.17.1-py built."

    foreach ($img in @(
        @{Name="cadqstream-metrics:latest";   Path="$DEPLOYMENT_DIR\cadqstream-metrics";   Desc="cadqstream-metrics"},
        @{Name="ml-service:latest";           Path="$DEPLOYMENT_DIR\ml-service";           Desc="ML Service (FastAPI)"},
        @{Name="action-replay-worker:latest"; Path="$DEPLOYMENT_DIR\action-replay-worker"; Desc="Action Replay Worker"},
        @{Name="stats-writer:latest";         Path="$DEPLOYMENT_DIR\stats-writer";        Desc="Stats Writer"},
        @{Name="ldt-kafka-producer";         Path="$DEPLOYMENT_DIR\kafka";               Desc="Kafka Producer"; Dockerfile="$DEPLOYMENT_DIR\kafka\Dockerfile.producer"}
    )) {
        $dockerfileArg = if ($img.Dockerfile) { "-f $($img.Dockerfile)" } else { "" }
        Write-Step "Building $($img.Desc)..."
        $b = docker build -t $img.Name $img.Path $dockerfileArg 2>&1
        if ($LASTEXITCODE -ne 0) { Write-Host "  [WARN] Build failed" -ForegroundColor Yellow }
        else { Write-Host "  [OK] $($img.Name) built." }
    }
} else {
    Write-Host "  [SKIP] Docker builds skipped (-SkipBuild)."
}

# =============================================================================
# PHASE 2: START ALL SERVICES
# =============================================================================
Write-Banner "PHASE 2: Start All Services"

Write-Step "Starting docker compose..."
docker compose -f "$COMPOSE_FILE" up -d 2>&1 | Out-Null
Write-Host "  [OK] docker compose up -d complete."
Start-Sleep -Seconds 5

# =============================================================================
# PHASE 3: WAIT FOR CORE SERVICES
# =============================================================================
Write-Banner "PHASE 3: Wait for Core Services"

$services = @(
    @{Name="ldt-kafka";                  Desc="Kafka";                   MaxWait=180; Interval=10}
    @{Name="ldt-minio";                  Desc="MinIO";                   MaxWait=60;  Interval=5}
    @{Name="ldt-cadqstream-metrics";     Desc="cadqstream-metrics";      MaxWait=60;  Interval=5}
    @{Name="ldt-ml-service";            Desc="ML Service (FastAPI)";     MaxWait=60;  Interval=5}
    @{Name="ldt-mlflow";                Desc="MLflow";                  MaxWait=120; Interval=10}
    @{Name="ldt-stats-writer";          Desc="Stats Writer";            MaxWait=30;  Interval=5}
    @{Name="ldt-action-replay-worker";   Desc="Action Replay Worker";     MaxWait=30;  Interval=5}
)

foreach ($svc in $services) {
    Wait-ForContainer -Name $svc.Name -Desc $svc.Desc -MaxWait $svc.MaxWait -Interval $svc.Interval
}

Wait-ForHttp -Url "http://localhost:8081/overview"  -Desc "Flink REST API"   -MaxWait 180
Wait-ForHttp -Url "http://localhost:5000/"         -Desc "MLflow"           -MaxWait 60

# =============================================================================
# PHASE 4: INIT CONTAINERS (Kafka topics, MinIO buckets)
# =============================================================================
Write-Banner "PHASE 4: Initialize Kafka Topics & MinIO Buckets"

$initContainers = @("ldt-kafka-init", "ldt-minio-init")
foreach ($c in $initContainers) {
    Write-Host "  Checking $c..." -NoNewline
    $elapsed = 0
    $done = $false
    while ($elapsed -lt 120) {
        $state = docker inspect --format='{{.State.Status}}' $c 2>$null
        if ($state -eq "exited" -or -not $state) {
            Write-Host " [OK - completed]" -ForegroundColor Green
            $done = $true; break
        }
        Start-Sleep -Seconds 5
        $elapsed += 5
    }
    if (-not $done) { Write-Host " [WARN - timed out]" -ForegroundColor Yellow }
}

Write-Host ""
Write-Host "  Kafka topics:"
docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>$null | ForEach-Object {
    Write-Host "    $_"
}

Write-Host ""
Write-Host "  MinIO buckets:"
docker exec ldt-minio mc ls local/ 2>$null | ForEach-Object {
    Write-Host "    $_"
}

# =============================================================================
# PHASE 5: TRAIN ML MODEL
# =============================================================================
Write-Banner "PHASE 5: Train ML Model"

if (-not $SkipTraining) {
    Write-Step "Training IsolationForest model (50K synthetic trips)..."
    $trainResult = docker exec ldt-flink-jobmanager bash -c "
        cd /opt/flink/e2e &&
        python3 deployment/scripts/train_model.py --version v1 --n-samples 50000 --n-estimators 100 2>&1
    " 2>$null

    if ($trainResult -match "Training Complete" -or $LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Model trained and uploaded to MinIO." -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Training output (first 5 lines):" -ForegroundColor Yellow
        $trainResult -split "`n" | Select-Object -First 5 | ForEach-Object { Write-Host "    $_" }
    }

    Write-Step "Broadcasting model to Kafka (if-model-updates topic)..."
    $loadResult = docker exec ldt-flink-jobmanager bash -c "
        cd /opt/flink/e2e &&
        python3 deployment/scripts/load_model_to_broadcast.py --version v1 --bootstrap kafka:9092 2>&1
    " 2>$null

    if ($loadResult -match "SUCCESS" -or $LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Model broadcast to Kafka." -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Broadcast: $($loadResult | Select-Object -First 3)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [SKIP] Model training skipped (-SkipTraining)."
}

# =============================================================================
# PHASE 6: SUBMIT FLINK JOB
# =============================================================================
Write-Banner "PHASE 6: Submit Flink Pipeline Job"

if (-not $SkipFlinkSubmit) {
    $sourceMounted = docker exec ldt-flink-jobmanager bash -c "test -f /opt/flink/e2e/src/flink_job_complete.py && echo YES" 2>$null
    if ($sourceMounted -notmatch "YES") {
        Write-Host "  [ERROR] Source files not mounted in container!" -ForegroundColor Red
        Write-Host "    Expected: /opt/flink/e2e/src/flink_job_complete.py" -ForegroundColor Red
        Write-Host "    Check docker-compose.yml volume mounts." -ForegroundColor Red
    } else {
        Write-Host "  [OK] Source files mounted correctly."

        $check = docker exec ldt-flink-jobmanager bash -c "curl -s http://localhost:8081/jobs 2>&1" 2>$null
        try {
            $jobsData = $check | ConvertFrom-Json
            $runningJobs = @($jobsData.jobs | Where-Object { $_.state -eq "RUNNING" })
        } catch { $runningJobs = @() }

        if ($runningJobs.Count -gt 0) {
            Write-Host "  [INFO] $($runningJobs.Count) job(s) already running:"
            foreach ($job in $runningJobs) {
                Write-Host "    $($job.id.Substring(0,20))... - $($job.state)"
            }
            Write-Host "  [SKIP] Use -ForceRestart to cancel and resubmit."
        } else {
            Write-Step "Submitting Flink job (this may take 30-60s)..."
            $submit = docker exec ldt-flink-jobmanager bash -c "
                export PYTHONPATH=/opt/flink/pyflink_extracted:/opt/flink/opt/python/py4j-0.10.9.7-src.zip:/opt/flink/opt/python/cloudpickle-2.2.0-src.zip:/opt/flink/e2e &&
                cd /opt/flink/e2e &&
                flink run -d -pyfs /opt/flink/e2e -python /opt/flink/e2e/src/flink_job_complete.py 2>&1
            " 2>$null

            if ($submit -match "Job has been submitted" -or $submit -match "Job ID") {
                Write-Host "  [OK] Flink job submitted!" -ForegroundColor Green
                Write-Host "    $submit"
            } else {
                Write-Host "  [INFO] Submit output:" -ForegroundColor Yellow
                $submit -split "`n" | Select-Object -First 5 | ForEach-Object { Write-Host "    $_" }
            }
        }
    }
} else {
    Write-Host "  [SKIP] Flink job submission skipped (-SkipFlinkSubmit)."
}

# =============================================================================
# PHASE 7: START DATA PRODUCER
# =============================================================================
Write-Banner "PHASE 7: Start Data Producer"

docker rm -f ldt-kafka-producer 2>$null | Out-Null
docker rm -f ldt-kafka-anomaly-producer 2>$null | Out-Null

Write-Step "Starting continuous Kafka producer (normal data, no anomaly injection)..."
$producerOut = docker run --network cadqstream-net --name ldt-kafka-producer -d ldt-kafka-producer `
    python3 /app/fast_producer.py 100 kafka:9092 taxi-nyc-raw 2>&1
Start-Sleep -Seconds 3

Write-Host "  Container: $($producerOut.Substring(0,12))"
$producerStatus = docker ps --filter "name=ldt-kafka-producer" --format "{{.Status}}"
Write-Host "  Status: $producerStatus"

# =============================================================================
# PHASE 8: VERIFY PIPELINE IS FLOWING
# =============================================================================
Write-Banner "PHASE 8: Verify Pipeline Data Flow"

Start-Sleep -Seconds 20

# --- Flink job status ---
Write-Host "  Flink job status:"
try {
    $jobs = Invoke-WebRequest -Uri "http://localhost:8081/jobs" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
    if ($jobs.jobs.Count -gt 0) {
        foreach ($job in $jobs.jobs) {
            $stateColor = if ($job.state -eq "RUNNING") { "Green" } else { "Yellow" }
            Write-Host "    $($job.id.Substring(0,20))... - $($job.state)" -ForegroundColor $stateColor
        }
    } else {
        Write-Host "    No jobs running yet (pipeline starting up)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "    (Could not reach Flink REST API)" -ForegroundColor Yellow
}

# --- Kafka consumer lag ---
Write-Host ""
Write-Host "  Kafka consumer groups:"
try {
    $groups = docker exec ldt-kafka kafka-consumer-groups --bootstrap-server localhost:9092 --list 2>$null
    if ($groups) {
        $groups -split "`n" | Where-Object { $_ -match "cadqstream" } | ForEach-Object { Write-Host "    $_" }
    }
} catch {}

# --- MinIO lakehouse data (check each zone) ---
Write-Host ""
Write-Host "  MinIO Lakehouse (checking for data files)..."

$zones = @(
    @{Bucket="raw-zone";        Path="taxi_trips_raw";       Desc="Valid taxi trips"}
    @{Bucket="quarantine-zone"; Path="canary_violations";   Desc="Canary rule violations"}
    @{Bucket="quarantine-zone"; Path="schema_violations";   Desc="Schema violations"}
    @{Bucket="clean-zone";      Path="anomaly_scores";       Desc="ML anomaly scores"}
    @{Bucket="clean-zone";      Path="meta_metrics";        Desc="Windowed meta-metrics"}
    @{Bucket="clean-zone";      Path="drift_events";        Desc="IEC drift events"}
    @{Bucket="clean-zone";      Path="alerts";               Desc="IEC/pipeline alerts"}
)

foreach ($z in $zones) {
    try {
        $files = docker exec ldt-minio mc ls "local/$($z.Bucket)/$($z.Path)/" 2>$null
        $count = ($files -split "`n" | Where-Object { $_ -match "\.parquet" }).Count
        if ($count -gt 0) {
            Write-Host "    $($z.Desc): $count files" -ForegroundColor Green
        } else {
            Write-Host "    $($z.Desc): 0 files (still initializing)" -ForegroundColor DarkGray
        }
    } catch {
        Write-Host "    $($z.Desc): (bucket/path not yet created)" -ForegroundColor DarkGray
    }
}

# =============================================================================
# PHASE 9: ANOMALY INJECTION DEMO
# =============================================================================
Write-Banner "PHASE 9: Anomaly Injection Demo"

$demoMode = "none"
if ($AnomalyBurst)     { $demoMode = "burst" }
elseif ($AnomalyDrift) { $demoMode = "drift" }
elseif ($ContinuousDemo) { $demoMode = "continuous" }

if ($demoMode -eq "none") {
    Write-Host "No anomaly mode selected. Use one of:" -ForegroundColor Yellow
    Write-Host "  -AnomalyBurst      : 500 normal -> 200 anomalies -> 500 normal" -ForegroundColor White
    Write-Host "  -AnomalyDrift     : 1000 normal -> 10x fare spike (300 recs) -> 100 normal" -ForegroundColor White
    Write-Host "  -ContinuousDemo   : Infinite stream with 8% anomaly rate" -ForegroundColor White
    Write-Host ""
    Write-Host "Commands below show how to run each demo manually." -ForegroundColor Cyan
} else {
    docker rm -f ldt-kafka-anomaly-producer 2>$null | Out-Null

    if ($demoMode -eq "burst") {
        Write-Host "[MODE: BURST ANOMALY]" -ForegroundColor Magenta
        Write-Host "Sending: 500 normal -> 200 anomalies -> 500 normal" -ForegroundColor White
        Write-Host ""
        $producerOut = docker run --network cadqstream-net --name ldt-kafka-anomaly-producer -d ldt-kafka-producer `
            python3 /app/anomaly_producer.py --mode burst --bootstrap kafka:9092 2>&1
        Write-Host "  Container: $($producerOut.Substring(0,12))"
        Write-Host "  Waiting 30s for burst to complete..."
        Start-Sleep -Seconds 30
        Write-Host ""
        Write-Host "  Producer logs:"
        docker logs ldt-kafka-anomaly-producer --tail 15 2>$null | ForEach-Object { Write-Host "    $_" }

    } elseif ($demoMode -eq "drift") {
        Write-Host "[MODE: DRIFT INJECTION]" -ForegroundColor Magenta
        Write-Host "Sending: 1000 normal -> 10x fare spike (300 records) -> 100 normal" -ForegroundColor White
        Write-Host "Trigger: ADWIN-U drift detection + METER strategy prediction" -ForegroundColor White
        Write-Host ""
        $producerOut = docker run --network cadqstream-net --name ldt-kafka-anomaly-producer -d ldt-kafka-producer `
            python3 /app/anomaly_producer.py --mode drift_inject --bootstrap kafka:9092 2>&1
        Write-Host "  Container: $($producerOut.Substring(0,12))"
        Write-Host "  Waiting 60s for drift injection to complete..."
        Start-Sleep -Seconds 60
        Write-Host ""
        Write-Host "  Producer logs:"
        docker logs ldt-kafka-anomaly-producer --tail 15 2>$null | ForEach-Object { Write-Host "    $_" }

    } elseif ($demoMode -eq "continuous") {
        Write-Host "[MODE: CONTINUOUS ANOMALY INJECTION]" -ForegroundColor Magenta
        Write-Host "Infinite stream: 8% anomaly rate, normal 92%" -ForegroundColor White
        Write-Host ""
        $producerOut = docker run --network cadqstream-net --name ldt-kafka-anomaly-producer -d ldt-kafka-producer `
            python3 /app/anomaly_producer.py --mode continuous --rate 0.08 --bootstrap kafka:9092 2>&1
        Write-Host "  Container: $($producerOut.Substring(0,12))"
        Write-Host "  Running in detached mode." -ForegroundColor Green
        Write-Host "  View logs: docker logs ldt-kafka-anomaly-producer -f"
    }
}

# =============================================================================
# PHASE 9.1: MANUAL ANOMALY COMMANDS (always printed)
# =============================================================================
Write-Banner "PHASE 9.1: Manual Anomaly Demo Commands"

Write-Host "Burst Demo (quick -- 1200 total records):" -ForegroundColor Yellow
Write-Host '  docker rm -f ldt-kafka-anomaly-producer 2>$null'
Write-Host '  docker run --network cadqstream-net --name ldt-kafka-anomaly-producer -d ldt-kafka-producer python3 /app/anomaly_producer.py --mode burst --bootstrap kafka:9092'
Write-Host ""

Write-Host "Drift Injection Demo (10x fare spike -> ADWIN-U triggers):" -ForegroundColor Yellow
Write-Host '  docker rm -f ldt-kafka-anomaly-producer 2>$null'
Write-Host '  docker run --network cadqstream-net --name ldt-kafka-anomaly-producer -d ldt-kafka-producer python3 /app/anomaly_producer.py --mode drift_inject --bootstrap kafka:9092'
Write-Host ""

Write-Host "Continuous (8% anomaly rate):" -ForegroundColor Yellow
Write-Host '  docker rm -f ldt-kafka-anomaly-producer 2>$null'
Write-Host '  docker run --network cadqstream-net --name ldt-kafka-anomaly-producer -d ldt-kafka-producer python3 /app/anomaly_producer.py --mode continuous --rate 0.08 --bootstrap kafka:9092'
Write-Host ""

Write-Host "Continuous (20% anomaly rate):" -ForegroundColor Yellow
Write-Host '  docker rm -f ldt-kafka-anomaly-producer 2>$null'
Write-Host '  docker run --network cadqstream-net --name ldt-kafka-anomaly-producer -d ldt-kafka-producer python3 /app/anomaly_producer.py --mode continuous --rate 0.20 --bootstrap kafka:9092'
Write-Host ""

# =============================================================================
# PHASE 9.2: ALL HANDLED ANOMALY TYPES
# =============================================================================
Write-Banner "PHASE 9.2: All 8 Handled Anomaly Types"

Write-Host "The CA-DQStream system detects these anomaly types:" -ForegroundColor White
Write-Host ""

$anomalies = @(
    @{N=1;  Type="NEGATIVE_FARE";             Logic='fare_amount < 0';             Severity="critical"; Weight="15%"; Trigger="Rule 1"}
    @{N=2;  Type="EXTREME_FARE";              Logic='fare_amount > $1000';        Severity="warning";  Weight="10%"; Trigger="Rule 2"}
    @{N=3;  Type="HIGH_FARE_PER_MIN";         Logic='fare/dur_min > $5.0/min';   Severity="warning";  Weight="n/a";  Trigger="Rule 3"}
    @{N=4;  Type="EXTREME_SPEED";              Logic='speed_mph > 80';             Severity="warning";  Weight="n/a";  Trigger="Rule 4"}
    @{N=5;  Type="ZERO_DISTANCE_WITH_FARE";    Logic='dist=0 but fare>0';          Severity="warning";  Weight="15%"; Trigger="Rule 5"}
    @{N=6;  Type="INVALID_PASSENGERS";         Logic='pax < 1 OR pax > 6';        Severity="warning";  Weight="15%"; Trigger="Rule 6"}
    @{N=7;  Type="CREDIT_NO_TIP";             Logic='payment_type=1 AND tip=0';   Severity="warning";  Weight="15%"; Trigger="Rule 7"}
    @{N=8;  Type="INVALID_ZONE";               Logic='LocationID outside 1-263';    Severity="warning";  Weight="20%"; Trigger="Layer 1 schema"}
)

$sevColor = @{
    "critical" = "Red"
    "warning"  = "Yellow"
}

foreach ($a in $anomalies) {
    $color = $sevColor[$a.Severity]
    Write-Host "  [$($a.N)] $($a.Type)" -ForegroundColor $color
    Write-Host "       Logic:    $($a.Logic)" -ForegroundColor DarkGray
    Write-Host "       Severity: $($a.Severity.ToUpper()) | Inject weight: $($a.Weight) | Trigger: $($a.Trigger)" -ForegroundColor DarkGray
    Write-Host ""
}

Write-Host "  [9] DRIFT_SPIKE (ADWIN-U drift detection)" -ForegroundColor Cyan
Write-Host "       Logic:    10x fare increase for 300 consecutive records" -ForegroundColor DarkGray
Write-Host "       Severity: CRITICAL | Triggers ADWIN-U + METER response" -ForegroundColor DarkGray
Write-Host "       Run: drift_inject mode" -ForegroundColor DarkGray
Write-Host ""

# =============================================================================
# PHASE 10: VERIFICATION & MONITORING COMMANDS (MinIO-based)
# =============================================================================
Write-Banner "PHASE 10: Verification & Monitoring Commands"

Write-Host "--- MinIO: List all buckets ---" -ForegroundColor Yellow
Write-Host "  docker exec ldt-minio mc ls local/"
Write-Host ""

Write-Host "--- MinIO: List raw-zone taxi trips ---" -ForegroundColor Yellow
Write-Host "  docker exec ldt-minio mc ls local/raw-zone/taxi_trips_raw/"
Write-Host ""

Write-Host "--- MinIO: List quarantine-zone (canary violations) ---" -ForegroundColor Yellow
Write-Host "  docker exec ldt-minio mc ls local/quarantine-zone/canary_violations/"
Write-Host ""

Write-Host "--- MinIO: List clean-zone (anomaly scores) ---" -ForegroundColor Yellow
Write-Host "  docker exec ldt-minio mc ls local/clean-zone/anomaly_scores/"
Write-Host ""

Write-Host "--- MinIO: List clean-zone (drift events / alerts) ---" -ForegroundColor Yellow
Write-Host "  docker exec ldt-minio mc ls local/clean-zone/drift_events/"
Write-Host "  docker exec ldt-minio mc ls local/clean-zone/alerts/"
Write-Host ""

Write-Host "--- Kafka: View raw records ---" -ForegroundColor Yellow
Write-Host "  docker exec ldt-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic taxi-nyc-raw --from-beginning --max-messages 5"
Write-Host ""

Write-Host "--- Kafka: Consumer group lag ---" -ForegroundColor Yellow
Write-Host "  docker exec ldt-kafka kafka-consumer-groups --bootstrap-server localhost:9092 --list"
Write-Host ""

Write-Host "--- Prometheus: Query pipeline metrics ---" -ForegroundColor Yellow
Write-Host '  curl -s "http://localhost:9090/api/v1/query?query=cadqstream_records_input_total" | ConvertFrom-Json'
Write-Host ""

Write-Host "--- cadqstream-metrics: Check Prometheus metrics ---" -ForegroundColor Yellow
Write-Host "  curl -s http://localhost:9250/metrics | Select-String 'cadqstream'"
Write-Host ""

Write-Host "--- Flink: Check job status ---" -ForegroundColor Yellow
Write-Host "  curl -s http://localhost:8081/jobs | ConvertFrom-Json | Select-Object -ExpandProperty jobs"
Write-Host ""

Write-Host "--- Flink: Check job logs ---" -ForegroundColor Yellow
Write-Host "  docker logs ldt-flink-jobmanager --tail 50 -f"
Write-Host ""

Write-Host "--- Anomaly Producer: View logs ---" -ForegroundColor Yellow
Write-Host "  docker logs ldt-kafka-anomaly-producer --tail 30 -f"
Write-Host ""

Write-Host "--- ML Service: Health check ---" -ForegroundColor Yellow
Write-Host "  curl http://localhost:8000/health"
Write-Host ""

Write-Host "--- Grafana: Pipeline dashboards ---" -ForegroundColor Yellow
Write-Host "  Open http://localhost:3000 (admin / admin123)"
Write-Host "  Dashboard: CA-DQStream Pipeline Overview"
Write-Host ""

# =============================================================================
# MINIO DATA PATH REFERENCE
# =============================================================================
Write-Banner "MinIO Lakehouse Data Paths"

Write-Host "All pipeline outputs are written as Parquet files to MinIO:" -ForegroundColor White
Write-Host ""
Write-Host "  raw-zone/" -ForegroundColor White
Write-Host "    taxi_trips_raw/       Valid records (Layer 1 output)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  quarantine-zone/" -ForegroundColor White
Write-Host "    schema_violations/   Records failing JSON/schema validation" -ForegroundColor DarkGray
Write-Host "    canary_violations/   Records failing 7 canary rules" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  clean-zone/" -ForegroundColor White
Write-Host "    anomaly_scores/      ML anomaly scores + thresholds" -ForegroundColor DarkGray
Write-Host "    meta_metrics/         1-min windowed aggregates per neighborhood" -ForegroundColor DarkGray
Write-Host "    drift_events/         IEC drift decisions (ADWIN-U results)" -ForegroundColor DarkGray
Write-Host "    alerts/               IEC + pipeline alerts (severity + strategy)" -ForegroundColor DarkGray
Write-Host "    pipeline_stats/       Periodic pipeline throughput stats" -ForegroundColor DarkGray
Write-Host ""
Write-Host "View files in MinIO console: http://localhost:9001" -ForegroundColor Cyan
Write-Host "  (minioadmin / minioadmin123)" -ForegroundColor Cyan
Write-Host ""

# =============================================================================
# FINAL SUMMARY
# =============================================================================
Write-Banner "DEPLOYMENT & DEMO COMPLETE!"

Write-Host "Service             URL / Port                   Credentials" -ForegroundColor White
Write-Host "-------             ---------------               ------------" -ForegroundColor DarkGray
Write-Host "Grafana            http://localhost:3000        admin / admin123" -ForegroundColor White
Write-Host "Flink UI           http://localhost:8081         (no auth)" -ForegroundColor White
Write-Host "Kafka UI           http://localhost:8080         (no auth)" -ForegroundColor White
Write-Host "Prometheus         http://localhost:9090         (no auth)" -ForegroundColor White
Write-Host "MinIO Console      http://localhost:9001         minioadmin / minioadmin123" -ForegroundColor White
Write-Host "MLflow             http://localhost:5000        (no auth)" -ForegroundColor White
Write-Host "ML Service         http://localhost:8000         FastAPI /docs" -ForegroundColor White
Write-Host "cadqstream-metrics localhost:9250/metrics        Prometheus scrape" -ForegroundColor White
Write-Host ""

Write-Host "Quick Start Commands:" -ForegroundColor White
Write-Host "  1. Burst anomaly demo:    powershell -File deployment/scripts/run_full_demo.ps1 -AnomalyBurst" -ForegroundColor Cyan
Write-Host "  2. Drift injection demo:  powershell -File deployment/scripts/run_full_demo.ps1 -AnomalyDrift" -ForegroundColor Cyan
Write-Host "  3. Continuous demo:       powershell -File deployment/scripts/run_full_demo.ps1 -ContinuousDemo" -ForegroundColor Cyan
Write-Host ""

Write-Host "Full restart (clean slate):" -ForegroundColor White
Write-Host "  powershell -File deployment/scripts/run_full_demo.ps1 -ForceRestart -AnomalyDrift" -ForegroundColor Cyan
Write-Host ""

Write-Host "Stop all services:" -ForegroundColor White
Write-Host "  powershell -File deployment/scripts/stop.ps1" -ForegroundColor Cyan
Write-Host ""

Write-Host "System Architecture (MinIO-only):" -ForegroundColor White
Write-Host "  Producer -> Kafka (taxi-nyc-raw)" -ForegroundColor DarkGray
Write-Host "    Layer 1: Validation (JSON parse, MurmurHash3 dedup, schema check)" -ForegroundColor DarkGray
Write-Host "    Layer 2: Canary Branch (7 rules) + Complex Branch (ML scoring)" -ForegroundColor DarkGray
Write-Host "    Layer 3: Rendezvous sync + 1-min TumblingWindow meta-aggregation" -ForegroundColor DarkGray
Write-Host "    Layer 4: IEC (ADWIN-U drift detection + METER strategy)" -ForegroundColor DarkGray
Write-Host "  Output: MinIO Lakehouse (raw-zone / quarantine-zone / clean-zone)" -ForegroundColor DarkGray
Write-Host "  Metadata: Kafka topics (dq-meta-stream, iec-action-replay, alerts)" -ForegroundColor DarkGray
Write-Host "  Metrics: Prometheus (cadqstream-metrics:9250) + Grafana dashboards" -ForegroundColor DarkGray
Write-Host ""

Pop-Location
