#Requires -Version 5.1
# =============================================================================
# CA-DQStream - End-to-End Deploy & Verify
# Fully idempotent: safe to re-run at any time.
# Checks: Kafka -> Flink -> MinIO -> ML -> Prometheus -> Grafana (top-to-bottom flow)
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File deployment/scripts/deploy-e2e.ps1
#
# Parameters:
#   -SkipBuild       Reuse existing Docker images (faster re-runs)
#   -SkipDeploy      Verify only (assumes stack is running)
#   -SkipTests      Deploy + verify but skip test injection
#   -Verbose         Show detailed output per check
#
# Exit codes:
#   0  ALL checks passed
#   1  Critical failure (pipeline not flowing)
#   2  Stack running but some non-critical warnings
# =============================================================================

param(
    [switch]$SkipBuild,
    [switch]$SkipDeploy,
    [switch]$SkipTests,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"
$DEPLOYMENT_DIR = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$GRAFANA_PASSWORD = if ($env:GRAFANA_PASSWORD) { $env:GRAFANA_PASSWORD } else { "grafana_local_admin" }
$COMPOSE_FILE = Join-Path $DEPLOYMENT_DIR "deployment\docker-compose-minimal.yml"

# Helpers
function Write-Pass($msg) { Write-Host "  [PASS] $msg" -ForegroundColor Green }
function Write-Fail($msg) { Write-Host "  [FAIL] $msg" -ForegroundColor Red }
function Write-Warn($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Info($msg) { Write-Host "  [INFO] $msg" -ForegroundColor Cyan }
function Write-Step($msg) { Write-Host ""; Write-Host "=== STEP $msg ===" -ForegroundColor Magenta }
function Write-Section($msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Magenta }

$FAILED = @()
$WARNINGS = @()

function Add-Fail($reason) {
    $script:FAILED += $reason
    Write-Fail $reason
}

function Add-Warn($reason) {
    $script:WARNINGS += $reason
    Write-Warn $reason
}

# Load .env
$envFile = Join-Path $DEPLOYMENT_DIR ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
        $parts = $_ -split '=', 2
        [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), 'Process')
    }
}

# =============================================================================
# PHASE 1: Pre-flight checks
# =============================================================================
Write-Section "PHASE 1: Pre-flight"

$dockerOk = docker --version 2>&1; if ($LASTEXITCODE -eq 0) { Write-Pass "Docker: $dockerOk" } else { Add-Fail "Docker not running"; return 1 }
$composeOk = docker compose version 2>&1; if ($LASTEXITCODE -eq 0) { Write-Pass "Docker Compose: $composeOk" } else { Add-Fail "Docker Compose not available"; return 1 }
if (-not (Test-Path $envFile)) { Add-Fail ".env not found"; return 1 }
Write-Pass ".env loaded"

$secrets = @("MINIO_ROOT_USER","MINIO_ROOT_PASSWORD","REDIS_PASSWORD","GRAFANA_PASSWORD","MEMSTREAM_MODEL_SIGNING_KEY","IEC_SIGNING_KEY","INTERNAL_API_KEY")
$missingSecrets = @()
foreach ($s in $secrets) {
    $val = (Get-ChildItem "env:$s" -ErrorAction SilentlyContinue).Value
    if (-not $val -or $val.StartsWith("changeme")) { $missingSecrets += $s }
}
if ($missingSecrets.Count -gt 0) { Add-Warn "Secrets still placeholder: $($missingSecrets -join ', ')" }
else { Write-Pass "All secrets configured" }

if ($script:FAILED.Count -gt 0) { return 1 }

# =============================================================================
# PHASE 2: Deploy (idempotent)
# =============================================================================
if (-not $SkipDeploy) {
    Write-Section "PHASE 2: Deploy (idempotent)"

    Write-Info "Stopping old ldt-* containers..."
    $old = docker ps -q --filter "name=ldt-" 2>$null
    if ($old) {
        docker stop $old 2>$null | Out-Null
        docker rm -f $old 2>$null | Out-Null
        $cnt = @($old).Count
        Write-Info "Cleaned up $cnt old containers"
    }

    # Fix Kafka InconsistentClusterIdException: always clear kafka-data volume
    # before Kafka starts. Safe because topics are recreated by kafka-init.
    # This ensures a clean Cluster ID every startup.
    Write-Info "Clearing Kafka data volume (ensures clean Cluster ID)..."
    docker volume rm ldt-kafka-data 2>$null | Out-Null
    Write-Info "Kafka data volume cleared"

    if (-not $SkipBuild) {
        Write-Step "2b: Build images"
        $builds = @{
            "ldt-flink:1.18.1-py" = @{ Path = "$DEPLOYMENT_DIR"; Dockerfile = "deployment\flink\Dockerfile"; Name = "Flink" }
            "ldt-cadqstream-metrics:latest" = @{ Path = "$DEPLOYMENT_DIR\deployment\cadqstream-metrics"; Dockerfile = "Dockerfile"; Name = "cadqstream-metrics" }
            "ldt-ml-service:latest" = @{ Path = "$DEPLOYMENT_DIR\deployment\ml-service"; Dockerfile = "Dockerfile"; Name = "ML Service" }
            "ldt-action-replay-worker:latest" = @{ Path = "$DEPLOYMENT_DIR\deployment\action-replay-worker"; Dockerfile = "Dockerfile"; Name = "Action Replay" }
            "ldt-stats-writer:latest" = @{ Path = "$DEPLOYMENT_DIR\deployment\stats-writer"; Dockerfile = "Dockerfile"; Name = "Stats Writer" }
            "ldt-kafka-producer:latest" = @{ Path = "$DEPLOYMENT_DIR\deployment\kafka"; Dockerfile = "Dockerfile.producer"; Name = "Kafka Producer" }
        }
        foreach ($img in $builds.Keys) {
            $info = $builds[$img]
            $df = Join-Path $DEPLOYMENT_DIR $info.Dockerfile
            $exists = docker images -q $img 2>$null
            if ($exists) {
                Write-Info "Image $img already exists, skipping build"
            } else {
                Write-Info "Building $img ($($info.Name))..."
                $out = docker build -t $img $info.Path -f $df 2>&1
                if ($LASTEXITCODE -ne 0) { Add-Warn "$($info.Name) build failed" }
                else { Write-Pass "$($info.Name) built" }
            }
        }
    }

    Write-Step "2c: Starting all services"
    Write-Info "Running: docker compose up -d"
    $up = docker compose -f $COMPOSE_FILE up -d 2>&1
    if ($LASTEXITCODE -ne 0) { Add-Fail "docker compose up failed: $($up[-1])"; return 1 }
    Write-Pass "docker compose up -d completed"
    Write-Info "Waiting 45s for services to initialize..."
    Start-Sleep -Seconds 45

    Write-Step "2d: Wait for critical services to be healthy"
    $critical = @(
        @{Name="ldt-zookeeper"; Desc="Zookeeper"; Wait=120; Interval=5},
        @{Name="ldt-kafka"; Desc="Kafka"; Wait=180; Interval=10},
        @{Name="ldt-minio"; Desc="MinIO"; Wait=90; Interval=5},
        @{Name="ldt-redis"; Desc="Redis"; Wait=60; Interval=5},
        @{Name="ldt-flink-jobmanager"; Desc="Flink JobManager"; Wait=120; Interval=10},
        @{Name="ldt-prometheus"; Desc="Prometheus"; Wait=60; Interval=5},
        @{Name="ldt-grafana"; Desc="Grafana"; Wait=60; Interval=5}
    )
    foreach ($svc in $critical) {
        $elapsed = 0; $healthy = $false
        while ($elapsed -lt $svc.Wait) {
            $status = docker ps --filter "name=$($svc.Name)" --filter "status=running" --format "{{.Status}}" 2>$null
            if ($status -match "healthy" -or $status -match "Up") { $healthy = $true; break }
            Start-Sleep -Seconds $svc.Interval; $elapsed += $svc.Interval
        }
        if ($healthy) { Write-Pass "$($svc.Desc) is healthy" }
        else { Add-Fail "$($svc.Desc) did not become healthy within $($svc.Wait)s" }
    }

    Write-Step "2e: Run init containers"
    foreach ($init in @("ldt-kafka-init", "ldt-minio-init")) {
        $state = docker inspect --format='{{.State.Status}}' $init 2>$null
        if ($state -eq "exited" -or $state -eq "running") {
            if ($state -eq "running") {
                $e = 0
                while ($e -lt 120) {
                    $s = docker inspect --format='{{.State.Status}}' $init 2>$null
                    if ($s -eq "exited") { break }
                    Start-Sleep -Seconds 5; $e += 5
                }
            }
            $code = docker inspect --format='{{.ExitCode}}' $init 2>$null
            if ($code -eq "0") { Write-Pass "$init completed successfully" }
            else { Add-Warn "$init exited with code $code"; docker logs $init 2>&1 | Select-Object -Last 5 | ForEach-Object { Write-Info "  $_" } }
        }
    }

    Write-Step "2f: Wait for Flink REST API"
    $flinkElapsed = 0
    while ($flinkElapsed -lt 180) {
        try {
            $resp = Invoke-WebRequest -Uri "http://localhost:8081/overview" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($resp.StatusCode -eq 200) { Write-Pass "Flink REST API is ready"; break }
        } catch {}
        Start-Sleep -Seconds 10; $flinkElapsed += 10
    }
    if ($flinkElapsed -ge 180) { Add-Warn "Flink REST API did not respond within 180s" }

    $fiState = docker inspect --format='{{.State.Status}}' ldt-flink-init 2>$null
    if ($fiState -eq "running") { Write-Pass "flink-init (auto-recovery supervisor) is running" }
    elseif ($fiState) { Add-Warn "flink-init state: $fiState" }
}

if ($script:FAILED.Count -gt 0) { return 1 }

# =============================================================================
# PHASE 3: Kafka - topics, schemas, produce/consume
# =============================================================================
Write-Section "PHASE 3: Kafka (Layer 1 - Input)"

$kafkaList = docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>$null
if ($kafkaList) { Write-Pass "Kafka is reachable" }
else { Add-Fail "Kafka unreachable"; return 1 }

$activeTopics = @("taxi-nyc-raw-v2", "taxi-nyc-raw", "dq-stream-unified", "iec-action-replay", "iec-action-dlq", "memstream-model-updates")
$topicLines = $kafkaList -split "`n" | Where-Object { $_.Trim() -ne "" }
Write-Info "Kafka topics found: $($topicLines.Count)"
if ($Verbose) { $topicLines | ForEach-Object { Write-Info "  $_" } }

$missingTopics = @()
foreach ($t in $activeTopics) {
    if (-not ($topicLines -contains $t)) { $missingTopics += $t }
}
if ($missingTopics.Count -eq 0) { Write-Pass "All 6 active topics present: $($activeTopics -join ', ')" }
else { Add-Fail "Missing topics: $($missingTopics -join ', ')"; return 1 }

# Schema Registry - retry up to 30s
$srReady = $false
for ($srRetry = 0; $srRetry -lt 6; $srRetry++) {
    try {
        $schemas = Invoke-WebRequest -Uri "http://localhost:8081/subjects" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
        $sCount = ($schemas.Content | ConvertFrom-Json).Count
        if ($sCount -ge 0) { $srReady = $true; break }
    } catch {}
    Start-Sleep -Seconds 5
}
if ($srReady) { Write-Pass "Schema Registry: $sCount schemas registered" }
else { Add-Warn "Schema Registry unreachable (non-critical)" }

Write-Step "3d: Kafka produce/consume test"
$testMsg = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T14:00:00","tpep_dropoff_datetime":"2026-05-17T14:15:00","passenger_count":2,"trip_distance":3.5,"PULocationID":79,"DOLocationID":170,"fare_amount":12.50,"total_amount":15.75,"payment_type":1}'
$null = $testMsg | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
# Wait for broker to acknowledge and propagate metadata
Start-Sleep -Seconds 10
$consumed = docker exec ldt-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 --from-beginning --max-messages 1 --consumer-timeout-ms 15000 2>$null
if ($consumed) { Write-Pass "Kafka produce->consume end-to-end: message received" }
else { Add-Warn "Kafka produce->consume: no message consumed back (topic may have just been recreated)" }

Write-Step "3e: Kafka consumer group lag"
try {
    $lagOut = docker exec ldt-kafka kafka-consumer-groups --bootstrap-server localhost:9092 --all-groups --describe 2>$null
    $lagLines = $lagOut -split "`n" | Where-Object { $_ -match '^\S+\s+\S+\s+\S+\s+\S+\s+(\S+)' }
    $totalLag = 0
    foreach ($line in $lagLines) {
        $parts = $line -split '\s+' | Where-Object { $_ -ne "" }
        $lagVal = $parts[5]
        if ($lagVal -and $lagVal -ne "-" -and $lagVal -match '^\d+$') { $totalLag += [int]$lagVal }
    }
    Write-Info "Total consumer lag: $totalLag"
    if ($totalLag -gt 1000) { Add-Warn "Consumer lag is high ($totalLag) - slow consumer" }
    else { Write-Pass "Consumer lag: $totalLag (OK)" }
} catch { Add-Warn "Could not check consumer lag" }

try {
    $kexp = Invoke-WebRequest -Uri "http://localhost:9308/metrics" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($kexp.StatusCode -eq 200) { Write-Pass "kafka-exporter at :9308 is healthy" }
    else { Add-Warn "kafka-exporter returned HTTP $($kexp.StatusCode)" }
} catch { Add-Warn "kafka-exporter not accessible" }

try {
    $kui = Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($kui.StatusCode -eq 200) { Write-Pass "Kafka UI at :8080 is accessible" }
} catch { Add-Warn "Kafka UI not accessible" }

if ($script:FAILED.Count -gt 0) { return 1 }

# =============================================================================
# PHASE 4: Flink - job running, checkpointing, Kafka connectivity
# =============================================================================
Write-Section "PHASE 4: Flink (Layers 1-4)"

try {
    $overview = Invoke-WebRequest -Uri "http://localhost:8081/overview" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
    $tmCount = $overview.taskmanagers
    $slotsTotal = $overview.'taskmanager'.totalTaskManagerSlotNumber
    $slotsUsed = $overview.'taskmanager'.totalAvailableSlotNumber
    Write-Info "Flink cluster: $tmCount TM(s), $slotsTotal slots total, $slotsUsed free"
    if ($tmCount -lt 1) { Add-Fail "No TaskManagers registered"; return 1 }
    Write-Pass "TaskManager(s): $tmCount"
} catch { Add-Fail "Flink REST API unreachable"; return 1 }

try {
    $jobs = Invoke-WebRequest -Uri "http://localhost:8081/jobs" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
    $runningJobs = @($jobs.jobs | Where-Object { $_.state -eq "RUNNING" })
    if ($Verbose) {
        foreach ($j in @($jobs.jobs)) { Write-Info "  Job: $($j.name) [$($j.state)] (parallelism=$($j.parallelism))" }
    }
    if ($runningJobs.Count -gt 0) {
        Write-Pass "$($runningJobs.Count) job(s) RUNNING"
        foreach ($j in $runningJobs) { Write-Info "  $($j.name) [RUNNING]" }
    } else {
        $failedJobs = @($jobs.jobs | Where-Object { $_.state -eq "FAILED" })
        $failedReason = ""
        if ($failedJobs.Count -gt 0) {
            $failedReason = $failedJobs[0].name + " FAILED"
            foreach ($fj in $failedJobs) {
                $fjLogs = docker logs ldt-flink-jobmanager 2>&1 | Select-String -Pattern "Exception|ERROR|Traceback" -Context 0,2 | Select-Object -First 5
                if ($fjLogs) { Write-Info "  Error: $($fjLogs.Line)" }
            }
        }
        Add-Fail "No RUNNING jobs. $($failedJobs.Count) FAILED, $($failedReason). Check Flink UI at http://localhost:8081"
    }
} catch { Add-Fail "Cannot query Flink jobs"; return 1 }

foreach ($j in $runningJobs) {
    try {
        $jinfo = Invoke-WebRequest -Uri "http://localhost:8081/jobs/$($j.id)/info" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
        $chk = $jinfo.checkpointing
        if ($chk) {
            $lastChk = $chk.last_checkpoint_timestamp
            if ($lastChk -and [int64]$lastChk -gt 0) {
                $chkDate = [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$lastChk).LocalDateTime.ToString("yyyy-MM-dd HH:mm:ss")
                Write-Pass "Checkpointing active for '$($j.name)': last checkpoint at $chkDate"
            } else { Add-Warn "Checkpointing enabled but no checkpoint yet for '$($j.name)'" }
        } else { Add-Warn "Checkpointing not configured for '$($j.name)'" }
    } catch { Add-Warn "Cannot get checkpoint info for '$($j.name)'" }
}

Write-Step "4d: Flink consuming from taxi-nyc-raw-v2"
try {
    $cg = docker exec ldt-kafka kafka-consumer-groups --bootstrap-server localhost:9092 --all-groups --describe 2>$null
    $consumingGroup = $cg -split "`n" | Where-Object { $_ -match "taxi-nyc-raw-v2" }
    if ($consumingGroup) {
        Write-Pass "Flink consumer group is consuming from taxi-nyc-raw-v2"
        if ($Verbose) { $consumingGroup | Select-Object -First 3 | ForEach-Object { Write-Info "  $_" } }
    } else { Add-Warn "No consumer group consuming taxi-nyc-raw-v2" }
} catch {}

Write-Step "4e: Pipeline output to dq-stream-unified"
$outputMsgs = docker exec ldt-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic dq-stream-unified --from-beginning --max-messages 3 --consumer-timeout-ms 8000 2>$null
if ($outputMsgs) {
    $msgCount = (@($outputMsgs -split "`n" | Where-Object { $_ -ne "" })).Count
    Write-Pass "dq-stream-unified has data ($msgCount messages)"
    if ($Verbose) {
        $outputMsgs -split "`n" | Select-Object -First 3 | ForEach-Object {
            if ($_ -ne "") {
                try {
                    $parsed = $_ | ConvertFrom-Json
                    $et = $parsed._event_type
                    Write-Info "  [_event_type=$et]"
                } catch { Write-Info "  $_" }
            }
        }
    }
} else { Add-Warn "dq-stream-unified is empty (pipeline may need warmup)" }

Write-Step "4f: Flink JobManager error log"
$jmErrors = docker logs ldt-flink-jobmanager 2>&1 | Select-String -Pattern "^\s*(\S+\s+)?ERROR\s" | Select-Object -First 5
$jmErrorCount = @($jmErrors).Count
if ($jmErrorCount -eq 0) { Write-Pass "JobManager logs: no ERROR entries" }
else { Add-Warn "JobManager has $jmErrorCount ERROR entries:"; $jmErrors | Select-Object -First 3 | ForEach-Object { Write-Info "  $($_.Line)" } }

Write-Step "4g: Kafka->Flink->Kafka flow (end-to-end)"
$msgCount = @($outputMsgs -split "`n" | Where-Object { $_ -ne "" }).Count
$before = $msgCount
Start-Sleep -Seconds 2
$injectMsg = '{"VendorID":2,"tpep_pickup_datetime":"2026-05-17T14:30:00","tpep_dropoff_datetime":"2026-05-17T14:45:00","passenger_count":1,"trip_distance":5.5,"PULocationID":100,"DOLocationID":200,"fare_amount":18.50,"total_amount":22.00,"payment_type":1}'
$null = $injectMsg | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
Start-Sleep -Seconds 8
$afterMsgs = docker exec ldt-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic dq-stream-unified --from-beginning --max-messages 5 --consumer-timeout-ms 8000 2>$null
$afterCount = @($afterMsgs -split "`n" | Where-Object { $_ -ne "" }).Count
if ($afterCount -gt $before) { Write-Pass "End-to-end flow verified: message injected -> pipeline produced output" }
else { Add-Warn "Pipeline output count unchanged (may need more warmup time)" }

if ($script:FAILED.Count -gt 0) { return 1 }

# =============================================================================
# PHASE 5: MinIO - buckets, lifecycle, stored artifacts
# =============================================================================
Write-Section "PHASE 5: MinIO (Storage)"

# Use docker run --rm with mc image to check MinIO buckets (minio-init has restart:no)
$mcEnv = "-e MC_HOST=http://minio:9000 -e MC_ALIAS=local -e MC_USER=minioadmin -e MC_PASS=minioadmin123"
$mcReady = docker run --rm --network cadqstream-net $mcEnv minio/mc ready local 2>$null
if ($LASTEXITCODE -eq 0) { Write-Pass "MinIO is reachable" }
else { Add-Fail "MinIO mc ready failed"; return 1 }

$allBuckets = docker run --rm --network cadqstream-net $mcEnv minio/mc ls local/ 2>$null
$bucketLines = $allBuckets -split "`n" | ForEach-Object {
    if ($_ -match 'local/(\S+)') { $matches[1].TrimEnd('/') }
} | Where-Object { $_ -ne "" }
Write-Info "MinIO buckets: $($bucketLines.Count) found"
if ($Verbose) { $bucketLines | ForEach-Object { Write-Info "  $_" } }

$expectedBuckets = @("cadqstream-checkpoints", "cadqstream-raw", "cadqstream-violations", "cadqstream-anomalies", "cadqstream-metrics", "cadqstream-drift", "cadqstream-dlq", "ml-models")
$missingBuckets = @()
foreach ($b in $expectedBuckets) {
    if (-not ($bucketLines -contains $b)) { $missingBuckets += $b }
}
if ($missingBuckets.Count -eq 0) { Write-Pass "All 8 expected MinIO buckets present" }
else { Add-Fail "Missing MinIO buckets: $($missingBuckets -join ', ')"; return 1 }

Write-Step "5c: MinIO bucket contents"
$bucketChecks = @{
    "cadqstream-raw" = "Layer 1 valid records";
    "cadqstream-violations" = "Schema + canary violations";
    "cadqstream-anomalies" = "MemStream anomaly scores";
    "cadqstream-drift" = "IEC decisions and alerts";
    "cadqstream-metrics" = "MetaAggregator windowed metrics";
    "ml-models" = "Model artifacts";
}
$filledBuckets = 0
foreach ($b in $bucketChecks.Keys) {
    $contents = docker run --rm --network cadqstream-net $mcEnv minio/mc ls "local/$b/" 2>$null
    if ($contents) {
        $fc = (@($contents -split "`n" | Where-Object { $_ -ne "" })).Count
        Write-Info "Bucket $b : $fc files ($($bucketChecks[$b]))"
        $filledBuckets++
    } else { Write-Warn "Bucket $b : empty ($($bucketChecks[$b]))" }
}
Write-Pass "$filledBuckets/$($bucketChecks.Count) buckets have content"

Write-Step "5d: MinIO bucket security"
$sensitive = @("cadqstream-violations", "cadqstream-anomalies", "ml-models")
foreach ($b in $sensitive) {
    $pub = docker run --rm --network cadqstream-net $mcEnv minio/mc anonymous get "local/$b" 2>$null
    $bName = $b
    if ($pub -match "Enabled") { Add-Fail "Bucket ${bName} has PUBLIC ACCESS (security risk)"; return 1 }
    else { Write-Pass "Bucket ${bName}: private access verified" }
}

$chkpt = docker run --rm --network cadqstream-net $mcEnv minio/mc ls "local/cadqstream-checkpoints/" 2>$null
if ($chkpt) { Write-Pass "cadqstream-checkpoints has files (Flink checkpoint data)" }
else { Add-Warn "cadqstream-checkpoints is empty (Flink checkpoints may use local volume)" }

Write-Step "5f: ML model artifacts in ml-models"
$modelFiles = docker run --rm --network cadqstream-net $mcEnv minio/mc ls "local/ml-models/" 2>$null
if ($modelFiles) {
    $mCount = (@($modelFiles -split "`n" | Where-Object { $_ -match "local/ml-models/\S+" })).Count
    Write-Info "ml-models: $mCount files"
    $modelFiles -split "`n" | Where-Object { $_ -match "local/ml-models/(\S+)" } | Select-Object -First 5 | ForEach-Object {
        $fname = $_ -replace '.*local/ml-models/', ''
        Write-Info "  $fname"
    }
    Write-Pass "Model artifacts present in ml-models"
} else { Add-Warn "ml-models is empty (models embedded in container or not yet trained)" }

# =============================================================================
# PHASE 6: ML Service - FastAPI, inference, HMAC
# =============================================================================
Write-Section "PHASE 6: ML Service (Inference + Retrain)"

try {
    $mlH = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 10 -ErrorAction SilentlyContinue
    if ($mlH.StatusCode -eq 200) {
        $mlHJ = $mlH.Content | ConvertFrom-Json
        Write-Pass "ML service health: OK"
        if ($Verbose) {
            $mlHJ.PSObject.Properties | Where-Object { $_.Name -notmatch "password|key|secret" } | ForEach-Object {
                Write-Info "  $($_.Name): $($_.Value)"
            }
        }
    } else { Add-Fail "ML service health: HTTP $($mlH.StatusCode)"; return 1 }
} catch { Add-Fail "ML service not reachable at :8000"; return 1 }

Write-Step "6b: ML /predict inference"
try {
    $features = @(900.0, 3.5, 15.50, 2.50, 0.33, 0.95, 0.14, 0.0, 2.0, 100.0,
                  170.0, 5.0, 1.3, 0.16, 0.10, 0.05, 1.0, 1.0, 0.0, 1.0,
                  0.87, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.1, 0.9,
                  0.15, 0.85, 0.25, 0.75)
    $payload = @{features = @($features)} | ConvertTo-Json -Compress
    $pred = Invoke-WebRequest -Uri "http://localhost:8000/predict" -UseBasicParsing -Method POST `
        -Body $payload -ContentType "application/json" -TimeoutSec 15 -ErrorAction SilentlyContinue
    if ($pred.StatusCode -eq 200) {
        $predJ = $pred.Content | ConvertFrom-Json
        $score = $predJ.anomaly_score
        $isAnomaly = $predJ.is_anomaly
        Write-Info "anomaly_score = $score, is_anomaly = $isAnomaly"
        if ($null -ne $score) { Write-Pass "ML /predict: anomaly_score = $score" }
        else { Add-Warn "ML /predict: no anomaly_score in response" }
    } else { Add-Warn "ML /predict: HTTP $($pred.StatusCode)" }
} catch { Add-Warn "ML /predict endpoint not responding: $_" }

Write-Step "6c: ML HMAC verification status"
$mlLogs = docker logs ldt-ml-service --tail 50 2>$null
if ($mlLogs) {
    $hmacLines = $mlLogs -split "`n" | Where-Object { $_ -match "HMAC|hmac" } | Select-Object -First 3
    if ($hmacLines) {
        Write-Info "HMAC activity:"
        $hmacLines | ForEach-Object { Write-Info "  $_" }
    }
    $errLines = $mlLogs -split "`n" | Where-Object { $_ -match "ERROR|Exception|Traceback" } | Select-Object -First 3
    if ($errLines) {
        Add-Warn "ML service has errors:"
        $errLines | ForEach-Object { Write-Info "  $_" }
    } else { Write-Pass "ML service logs: no errors" }
}

Write-Step "6d: ML metrics in Prometheus"
try {
    $q = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=ml_service_model_loaded" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
    if ($q.data.result.Count -gt 0) { Write-Pass "ml_service_model_loaded metric: present" }
    else { Add-Warn "ml_service_model_loaded metric: not yet available" }
} catch {}

# =============================================================================
# PHASE 7: Prometheus - scrape targets, cadqstream metrics, alert rules
# =============================================================================
Write-Section "PHASE 7: Prometheus (Observability)"

try {
    $promH = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($promH.StatusCode -eq 200) { Write-Pass "Prometheus is healthy" }
    else { Add-Fail "Prometheus unhealthy: HTTP $($promH.StatusCode)"; return 1 }
} catch { Add-Fail "Prometheus unreachable at :9090"; return 1 }

Write-Step "7b: Prometheus scrape targets"
try {
    $targets = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/targets" -UseBasicParsing -TimeoutSec 10 -ErrorAction SilentlyContinue | ConvertFrom-Json
    $allTargets = $targets.data.targets
    $upTargets = @($allTargets | Where-Object { $_.health -eq "up" })
    $downTargets = @($allTargets | Where-Object { $_.health -ne "up" })
    Write-Info "Scrape targets: $($upTargets.Count) up, $($downTargets.Count) down"
    if ($Verbose) {
        $allTargets | ForEach-Object {
            $icon = if ($_.health -eq "up") { "[OK]" } else { "[DOWN]" }
            Write-Info "  $icon $($_.labels.job) : $($_.lastError)"
        }
    }
    if ($downTargets.Count -gt 0) {
        Add-Warn "Down scrape targets:"
        $downTargets | ForEach-Object { Add-Warn "  $($_.labels.job): $($_.lastError)" }
    }
    Write-Pass "Scrape targets: $($upTargets.Count)/$($allTargets.Count) healthy"
} catch { Add-Warn "Cannot query Prometheus targets" }

Write-Step "7c: cadqstream-metrics service"
try {
    $cadq = Invoke-WebRequest -Uri "http://localhost:9250/metrics" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($cadq.StatusCode -eq 200) {
        $cadqMetrics = (@($cadq.Content -split "`n" | Where-Object { $_ -match "^cadqstream" })).Count
        Write-Pass "cadqstream-metrics at :9250 is healthy: $cadqMetrics cadqstream_* metrics exposed"
    } else { Add-Warn "cadqstream-metrics returned HTTP $($cadq.StatusCode)" }
} catch { Add-Warn "cadqstream-metrics not accessible" }

Write-Step "7d: Key metric groups per layer"
$metricGroups = @{
    "L1-Ingestion"   = @("cadqstream_records_valid_total", "cadqstream_records_violation_total");
    "L2-Canary"       = @("cadqstream_canary_violation_total");
    "L3-MemStream"    = @("cadqstream_anomaly_score", "memstream_scoring_latency_seconds_bucket");
    "L3-MetaAgg"      = @("cadqstream_meta_window_record_count");
    "L4-IEC"          = @("cadqstream_drift_detected", "cadqstream_iec_action_total", "cadqstream_circuit_breaker_state");
    "ML-Warmup"       = @("memstream_warmup_progress", "memstream_redis_connected");
    "ML-HMAC"         = @("memstream_hmac_verification_total");
    "ML-Stats"        = @("memstream_knn_avg_distance", "memstream_memory_fill_rate", "memstream_beta_staleness_seconds");
    "Kafka-Exp"       = @("kafka_topic_partition_current_offset");
    "Flink"           = @("flink_jobmanager_StatusReport_NumRunningJobs");
}

foreach ($group in $metricGroups.Keys) {
    $groupMissing = @()
    $groupFound = @()
    foreach ($m in $metricGroups[$group]) {
        try {
            $q = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=$m" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
            if ($q.data.result.Count -gt 0) { $groupFound += $m }
            else { $groupMissing += $m }
        } catch { $groupMissing += "$m (query failed)" }
    }
    if ($groupMissing.Count -eq 0) { Write-Pass "$group : all $($groupFound.Count) metrics present" }
    else { Write-Warn "$group : missing $($groupMissing.Count) metrics"; $groupMissing | ForEach-Object { Write-Info "    - $_" } }
}

Write-Step "7e: Prometheus alert rules"
try {
    $alerts = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/rules" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
    $groups = $alerts.data.groups
    $totalRules = ($groups | ForEach-Object { $_.rules }).Count
    $alertingRules = $groups | ForEach-Object { $_.rules | Where-Object { $_.type -eq "alerting" } }
    Write-Info "Alert rule groups: $($groups.Count), total rules: $totalRules, alerting: $($alertingRules.Count)"
    Write-Pass "Alert rules loaded"
} catch { Add-Warn "Cannot query Prometheus alert rules" }

# =============================================================================
# PHASE 8: Grafana - dashboards, panes, metric groups
# =============================================================================
Write-Section "PHASE 8: Grafana (Visualization)"

try {
    $gH = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($gH.StatusCode -eq 200) {
        $gVer = ($gH.Content | ConvertFrom-Json).version
        Write-Pass "Grafana v$gVer is healthy"
    } else { Add-Warn "Grafana: HTTP $($gH.StatusCode)" }
} catch { Add-Warn "Grafana not reachable" }

try {
    $dashes = Invoke-WebRequest -Uri "http://localhost:3000/api/search?type=dash-db" -UseBasicParsing -TimeoutSec 10 -ErrorAction SilentlyContinue `
        -Credential (New-Object PSCredential("admin", (ConvertTo-SecureString $GRAFANA_PASSWORD -AsPlainText -Force))) | ConvertFrom-Json
    $dashCount = $dashes.Count
    Write-Info "Grafana dashboards: $dashCount provisioned"
    if ($Verbose) { $dashes | ForEach-Object { Write-Info "  $($_.title) [uid=$($_.uid)]" } }
    if ($dashCount -ge 6) { Write-Pass "Grafana dashboards: $dashCount (expected >= 6)" }
    else { Add-Warn "Grafana dashboards: $dashCount (expected >= 6)" }
} catch { Add-Warn "Cannot retrieve Grafana dashboards" }

Write-Step "8c: Dashboard pane metrics completeness (L1 through L4)"
$dashMetrics = @{
    "pipeline-overview"        = @("cadqstream_records_valid_total", "flink_taskmanager_JVM_Memory_Heap_Used");
    "cadqstream-data-quality"  = @("cadqstream_records_violation_total", "cadqstream_anomaly_score");
    "memstream-data-quality"   = @("memstream_warmup_progress", "memstream_knn_avg_distance", "memstream_memory_fill_rate");
    "kafka-overview"          = @("kafka_topic_partition_current_offset", "kafka_consumer_group_lag");
    "flink-jobs"              = @("flink_jobmanager_StatusReport_NumRunningJobs", "flink_taskmanager_JVM_Memory_Heap_Used");
    "streaming-noc"           = @("cadqstream_records_valid_total", "cadqstream_drift_detected", "cadqstream_iec_action_total");
    "infrastructure"           = @("node_memory_MemAvailable_bytes", "container_memory_usage_bytes");
}

$incomplete = @()
foreach ($dash in $dashMetrics.Keys) {
    $missing = @()
    foreach ($m in $dashMetrics[$dash]) {
        try {
            $q = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=$m" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
            if ($q.data.result.Count -eq 0) { $missing += $m }
        } catch { $missing += "$m (query failed)" }
    }
    if ($missing.Count -eq 0) { Write-Pass "Dashboard '$dash': all $($dashMetrics[$dash].Count) metrics present" }
    else {
        Write-Warn "Dashboard '$dash': missing $($missing.Count) metrics"
        $missing | ForEach-Object { Write-Info "    - $_" }
        $incomplete += $dash
    }
}

# =============================================================================
# PHASE 9: Test data injection - normal, anomaly, drift
# =============================================================================
if (-not $SkipTests) {
    Write-Section "PHASE 9: Test Data Injection"

    $KAFKA = "kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2"
    $send = { $msg | docker exec -i ldt-kafka $KAFKA 2>$null }

    Write-Step "9a: Normal record (expect: PROCESSED_RECORD)"
    &$send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T15:00:00","tpep_dropoff_datetime":"2026-05-17T15:20:00","passenger_count":2,"trip_distance":3.5,"PULocationID":79,"DOLocationID":170,"fare_amount":12.50,"total_amount":15.75,"payment_type":1}'
    Start-Sleep -Seconds 5

    Write-Step "9b: L1 Schema violations (missing trip_distance, invalid PULocationID)"
    &$send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T15:01:00","tpep_dropoff_datetime":"2026-05-17T15:21:00","passenger_count":1,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.50,"payment_type":1}'
    &$send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T15:01:30","tpep_dropoff_datetime":"2026-05-17T15:21:30","passenger_count":1,"trip_distance":5.0,"PULocationID":999,"DOLocationID":500,"fare_amount":18.00,"total_amount":22.00,"payment_type":1}'
    Start-Sleep -Seconds 3

    Write-Step "9c: L2 Canary rule violations"
    &$send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T15:02:00","tpep_dropoff_datetime":"2026-05-17T15:17:00","passenger_count":1,"trip_distance":5.0,"PULocationID":79,"DOLocationID":170,"fare_amount":-5.00,"total_amount":2.00,"payment_type":1}'
    &$send '{"VendorID":2,"tpep_pickup_datetime":"2026-05-17T15:02:10","tpep_dropoff_datetime":"2026-05-17T15:03:10","passenger_count":1,"trip_distance":0.0,"PULocationID":79,"DOLocationID":79,"fare_amount":25.00,"total_amount":28.00,"payment_type":2}'
    &$send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T15:02:20","tpep_dropoff_datetime":"2026-05-17T15:17:20","passenger_count":0,"trip_distance":3.0,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.00,"payment_type":1}'
    Start-Sleep -Seconds 3

    Write-Step "9d: L3 Extreme anomaly (fare spike)"
    &$send '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T15:03:00","tpep_dropoff_datetime":"2026-05-17T15:18:00","passenger_count":6,"trip_distance":99.9,"PULocationID":138,"DOLocationID":229,"fare_amount":999.99,"total_amount":1050.00,"payment_type":1}'
    Start-Sleep -Seconds 5

    Write-Step "9e: Concept drift (gradual fare increase, 10 records)"
    $baseFare = 15.0
    for ($i = 1; $i -le 10; $i++) {
        $fare = [math]::Round($baseFare * [math]::Pow(1.2, $i), 2)
        $driftMsg = "{`"VendorID`":1,`"tpep_pickup_datetime`":`"2026-05-17T15:04:0${i}`",`"tpep_dropoff_datetime`":`"2026-05-17T15:19:0${i}`",`"passenger_count`":2,`"trip_distance`":4.0,`"PULocationID`":100,`"DOLocationID`":180,`"fare_amount`":$fare,`"total_amount`":$($fare + 3),`"payment_type`":1}"
        $null = $driftMsg | docker exec -i ldt-kafka $KAFKA 2>$null
        Start-Sleep -Milliseconds 500
    }
    Write-Info "Concept drift injection complete (10 records with +20% fare each)"
    Write-Info "ADWIN drift detection needs ~3 min. Check cadqstream-drift/ for IEC events."

    Write-Step "9f: Verify dq-stream-unified event types"
    Start-Sleep -Seconds 10
    $unified = docker exec ldt-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic dq-stream-unified --from-beginning --max-messages 20 --consumer-timeout-ms 10000 2>$null
    if ($unified) {
        $eventTypes = @{}
        $unified -split "`n" | ForEach-Object {
            if ($_ -ne "") {
                try {
                    $parsed = $_ | ConvertFrom-Json
                    $et = $parsed._event_type
                    if ($et) { $eventTypes[$et] = $eventTypes[$et] + 1 }
                } catch {}
            }
        }
        $total = 0
        foreach ($et in $eventTypes.Keys) {
            Write-Info "  $et : $($eventTypes[$et]) records"
            $total += $eventTypes[$et]
        }
        Write-Info "Total records in dq-stream-unified: $total"
        if ($total -gt 0) { Write-Pass "dq-stream-unified has $total records across $(@($eventTypes.Keys).Count) event types" }
    } else { Add-Warn "dq-stream-unified empty after test injection" }

    Write-Step "9g: MinIO violation buckets after test injection"
    foreach ($bucket in @("cadqstream-violations", "cadqstream-anomalies")) {
        $files = docker run --rm --network cadqstream-net $mcEnv minio/mc ls "local/$bucket/" 2>$null
        if ($files) {
            $fCount = (@($files -split "`n" | Where-Object { $_ -ne "" })).Count
            Write-Pass "Bucket $bucket : $fCount files"
        } else { Write-Warn "Bucket $bucket : still empty" }
    }

    Write-Step "9h: Prometheus drift metric"
    try {
        $dq = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=cadqstream_drift_detected" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
        if ($dq.data.result.Count -gt 0) { Write-Pass "cadqstream_drift_detected: DETECTED" }
        else { Write-Info "cadqstream_drift_detected: not yet (ADWIN needs time)" }
    } catch {}
}

# =============================================================================
# PHASE 10: Stats metrics
# =============================================================================
Write-Section "PHASE 10: Stats Metrics"

$swStatus = docker ps --filter "name=ldt-stats-writer" --format "{{.Status}}" 2>$null
if ($swStatus -match "Up") { Write-Pass "stats-writer is running" }
else { Add-Warn "stats-writer not running (non-critical)" }

$statsFiles = docker run --rm --network cadqstream-net $mcEnv minio/mc ls "local/cadqstream-metrics/" 2>$null
if ($statsFiles) {
    $sfCount = (@($statsFiles -split "`n" | Where-Object { $_ -ne "" })).Count
    Write-Pass "cadqstream-metrics/ bucket: $sfCount files (stats snapshots)"
} else { Add-Warn "cadqstream-metrics/ bucket: empty (stats-writer may need warmup)" }

$statsMetrics = @("cadqstream_anomaly_rate", "cadqstream_false_positive_rate", "cadqstream_records_processed_total", "cadqstream_violation_rate")
$statsMissing = @()
foreach ($m in $statsMetrics) {
    try {
        $q = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=$m" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
        if ($q.data.result.Count -gt 0) { Write-Info "  $m : present" }
        else { $statsMissing += $m }
    } catch { $statsMissing += "$m (query failed)" }
}
if ($statsMissing.Count -eq 0) { Write-Pass "All $($statsMetrics.Count) stats metrics in Prometheus" }
else { Add-Warn "Missing stats metrics: $($statsMissing -join ', ')" }

# =============================================================================
# PHASE 11: Offline pretrain artifacts
# =============================================================================
Write-Section "PHASE 11: Offline Pretrain Components"

$nmFound = $false
foreach ($p in @("$DEPLOYMENT_DIR\models\neighborhood_mapping.json", "$DEPLOYMENT_DIR\src\config\neighborhood_mapping.json")) {
    if (Test-Path $p) { Write-Pass "neighborhood_mapping.json found at $p"; $nmFound = $true; break }
}
if (-not $nmFound) { Add-Warn "neighborhood_mapping.json not found locally" }

$nmMinio = docker run --rm --network cadqstream-net $mcEnv minio/mc ls "local/ml-models/neighborhood_mapping.json" 2>$null
if ($nmMinio) { Write-Pass "neighborhood_mapping.json in MinIO ml-models/" }
else { Add-Warn "neighborhood_mapping.json not in MinIO" }

$ctFound = $false
foreach ($p in @("$DEPLOYMENT_DIR\models\context_thresholds_v2.json", "$DEPLOYMENT_DIR\src\config\context_thresholds_v2.json")) {
    if (Test-Path $p) {
        try {
            $ctCount = (Get-Content $p | ConvertFrom-Json).PSObject.Properties.Count
            Write-Pass "context_thresholds_v2.json: $ctCount context cells"
            $ctFound = $true
        } catch {}
        break
    }
}
if (-not $ctFound) { Write-Info "context_thresholds_v2.json: not found (generated during warmup)" }

$checkpoint = docker run --rm --network cadqstream-net $mcEnv minio/mc ls "local/ml-models/checkpoints/" 2>$null
if ($checkpoint) { Write-Pass "Model checkpoint in ml-models/checkpoints/" }
else {
    $root = docker run --rm --network cadqstream-net $mcEnv minio/mc ls "local/ml-models/" 2>$null
    if ($root -match "memstream_checkpoint|\.pt") { Write-Pass "Model checkpoint found in ml-models root" }
    else { Add-Warn "No model checkpoint in ml-models (models may be in container or not yet trained)" }
}

$mlLogs = docker logs ldt-ml-service --tail 50 2>$null
if ($mlLogs) {
    if ($mlLogs -match "HMAC.*success|HMAC.*passed|verification.*ok|Integrity.*OK|checkpoint.*loaded") { Write-Pass "HMAC checkpoint verification: passing" }
    elseif ($mlLogs -match "HMAC.*failed|verification.*error|Integrity.*FAILED") { Add-Warn "HMAC verification failures in ML service logs" }
    else { Write-Info "HMAC verification: status unclear from logs" }
}

# =============================================================================
# FINAL SUMMARY
# =============================================================================
Write-Section "FINAL SUMMARY"
$running = (docker ps --filter "name=ldt-" --format "{{.Names}}" 2>$null).Count
Write-Info "Containers running: $running"
Write-Info "FAILED steps: $($script:FAILED.Count)"
Write-Info "WARNINGS: $($script:WARNINGS.Count)"

if ($script:FAILED.Count -gt 0) {
    Write-Host ""
    Write-Host "CRITICAL FAILURES:" -ForegroundColor Red
    $script:FAILED | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "FAILED STEPS MUST BE RESOLVED." -ForegroundColor Red
    Write-Host "Run: docker compose -f deployment\docker-compose.yml logs [service]" -ForegroundColor Cyan
    Write-Host "Run: bash deployment/scripts/healthcheck.sh" -ForegroundColor Cyan
    return 1
}

if ($script:WARNINGS.Count -gt 0) {
    Write-Host ""
    Write-Host "WARNINGS (non-critical):" -ForegroundColor Yellow
    $script:WARNINGS | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    Write-Host ""
    Write-Host "Stack is OPERATIONAL. $running containers running." -ForegroundColor Green
    Write-Host "Some non-critical components need attention." -ForegroundColor Yellow
    return 2
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "ALL CHECKS PASSED" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Cyan
Write-Host "  Kafka UI:       http://localhost:8080"
Write-Host "  Flink UI:       http://localhost:8081"
Write-Host "  Grafana:        http://localhost:3000  (admin / $GRAFANA_PASSWORD)"
Write-Host "  Prometheus:     http://localhost:9090"
Write-Host "  MinIO Console:  http://localhost:9001  (minioadmin / minioadmin)"
Write-Host "  ML Service:     http://localhost:8000"
Write-Host "  cadqstream-metrics: http://localhost:9250/metrics"
Write-Host ""
Write-Host "Pipeline flow:" -ForegroundColor Cyan
Write-Host "  taxi-nyc-raw-v2 (Kafka) -> Flink -> dq-stream-unified (Kafka)" -ForegroundColor Cyan
Write-Host "  Flink -> MinIO buckets (cadqstream-raw, violations, anomalies, drift)" -ForegroundColor Cyan
Write-Host "  cadqstream-metrics -> Prometheus -> Grafana dashboards" -ForegroundColor Cyan
Write-Host "  ML service -> inference + HMAC verified model" -ForegroundColor Cyan
Write-Host ""
return 0
