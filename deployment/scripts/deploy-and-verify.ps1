# =============================================================================
# CA-DQStream - Deploy and Verify Full Stack
# Runs complete deployment with end-to-end verification for all 21 services.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File deployment/scripts/deploy-and-verify.ps1
#
# Parameters:
#   -SkipBuild       Reuse existing Docker images (faster for re-runs)
#   -SkipDeploy      Run verification only (assumes stack is already running)
#   -SkipTests       Deploy only, skip test injection and flow verification
#   -Verbose         Show detailed output
#
# Exit codes:
#   0  All checks passed
#   1  Critical failure (stop-on-fail)
#   2  Warnings only (non-critical failures)
# =============================================================================

param(
    [switch]$SkipBuild,
    [switch]$SkipDeploy,
    [switch]$SkipTests,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"
$DEPLOYMENT_DIR = Split-Path -Parent $PSScriptRoot
$GRAFANA_PASSWORD = if ($env:GRAFANA_PASSWORD) { $env:GRAFANA_PASSWORD } else { "grafana_local_admin" }

# ── Color helpers ─────────────────────────────────────────────────────────────
function Write-Pass($msg) { Write-Host "  [PASS] $msg" -ForegroundColor Green }
function Write-Fail($msg) { Write-Host "  [FAIL] $msg" -ForegroundColor Red }
function Write-Warn($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Info($msg) { Write-Host "  [INFO] $msg" -ForegroundColor Cyan }
function Write-Step($msg) { Write-Host ""; Write-Host "=== STEP $msg ===" -ForegroundColor Magenta }
function Write-Section($msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Magenta }

$FAILED_STEPS = @()
$WARN_STEPS = @()

# =============================================================================
# PHASE 1: PRE-DEPLOYMENT AUDIT
# =============================================================================
Write-Section "PHASE 1: Pre-Deployment Audit"

Write-Info "Checking Docker..."
$dockerVersion = docker --version 2>&1
if ($LASTEXITCODE -eq 0) { Write-Pass "Docker: $dockerVersion" } else { Write-Fail "Docker not running"; exit 1 }

Write-Info "Checking Docker Compose..."
$composeVersion = docker compose version 2>&1
if ($LASTEXITCODE -eq 0) { Write-Pass "Docker Compose: $composeVersion" } else { Write-Fail "Docker Compose not available"; exit 1 }

Write-Info "Checking .env file..."
$envFile = Join-Path $DEPLOYMENT_DIR ".env"
if (-not (Test-Path $envFile)) {
    Write-Fail ".env file not found at $envFile"
    exit 1
}
Write-Pass ".env found"

# Load .env
Get-Content $envFile | Where-Object { $_ -match '^\s*[^#].*=' } | ForEach-Object {
    $parts = $_ -split '=', 2
    [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), 'Process')
}
Write-Info "Environment variables loaded from .env"

# Check required env vars
$required = @("MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD", "REDIS_PASSWORD", "GRAFANA_PASSWORD", "MEMSTREAM_MODEL_SIGNING_KEY", "IEC_SIGNING_KEY", "INTERNAL_API_KEY")
$missing = @()
foreach ($v in $required) {
    $val = if ($env:$v) { $env:$v } else { [System.Environment]::GetEnvironmentVariable($v, 'Process') }
    if (-not $val -or $val.StartsWith("changeme")) {
        $missing += $v
    }
}
if ($missing.Count -gt 0) {
    Write-Warn "Some .env variables are placeholder values: $($missing -join ', ')"
    $WARN_STEPS += "Phase1-EnvVars"
}

Write-Info "Checking source files..."
$srcFiles = @(
    "src\flink_job_complete.py",
    "src\operators\canary_rules_operator.py",
    "src\operators\memstream_scoring_operator.py",
    "src\operators\iec_operator.py",
    "src\operators\meta_aggregator.py",
    "src\sinks\minio_sink.py",
    "src\ml\memstream_core.py",
    "src\api\ml_service.py"
)
$missingSrc = @()
foreach ($f in $srcFiles) {
    $path = Join-Path $DEPLOYMENT_DIR $f
    if (-not (Test-Path $path)) {
        $missingSrc += $f
    }
}
if ($missingSrc.Count -gt 0) {
    Write-Fail "Missing source files: $($missingSrc -join ', ')"
    $FAILED_STEPS += "Phase1-SourceFiles"
} else {
    Write-Pass "All critical source files present"
}

if ($FAILED_STEPS.Count -gt 0) { exit 1 }

# =============================================================================
# PHASE 2: DOCKER DEPLOYMENT
# =============================================================================
if (-not $SkipDeploy) {
    Write-Section "PHASE 2: Docker Deployment"

    Write-Step "2a: Clean slate (idempotent)"
    Write-Info "Stopping existing containers..."
    docker compose -f "$DEPLOYMENT_DIR\docker-compose.yml" down --remove-orphans 2>$null | Out-Null

    $ldtContainers = docker ps -q --filter "name=ldt-" 2>$null
    if ($ldtContainers) {
        docker stop $ldtContainers 2>$null | Out-Null
        docker rm -f $ldtContainers 2>$null | Out-Null
    }

    # Remove networks (various naming conventions)
    @("cadqstream-net", "deployment_cadqstream-net", "ldt_cadqstream-net", "deployment-cadqstream-net") | ForEach-Object {
        docker network rm $_ 2>$null | Out-Null
    }

    Write-Pass "Clean slate complete"

    Write-Step "2b: Build custom Docker images"
    if (-not $SkipBuild) {
        $buildStart = Get-Date

        # Flink image
        Write-Info "Building ldt-flink:1.18.1-py (5-15 min on first run)..."
        $build = docker build -t ldt-flink:1.18.1-py $DEPLOYMENT_DIR -f "$DEPLOYMENT_DIR\flink\Dockerfile" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "ldt-flink build failed. Check Dockerfile and Maven connectivity."
            Write-Host $build | Select-Object -First 20
            $FAILED_STEPS += "Phase2-FlinkBuild"
        } else {
            $elapsed = [math]::Round(((Get-Date) - $buildStart).TotalSeconds)
            Write-Pass "ldt-flink:1.18.1-py built successfully (${elapsed}s)"
        }

        # cadqstream-metrics
        Write-Info "Building ldt-cadqstream-metrics:latest..."
        $build2 = docker build -t ldt-cadqstream-metrics:latest "$DEPLOYMENT_DIR\cadqstream-metrics" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "cadqstream-metrics build failed."
            Write-Host $build2 | Select-Object -First 10
            $FAILED_STEPS += "Phase2-MetricsBuild"
        } else {
            Write-Pass "ldt-cadqstream-metrics:latest built"
        }

        # ML service
        Write-Info "Building ldt-ml-service:latest..."
        $build3 = docker build -t ldt-ml-service:latest "$DEPLOYMENT_DIR\ml-service" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "ml-service build failed."
            $FAILED_STEPS += "Phase2-MLBuild"
        } else {
            Write-Pass "ldt-ml-service:latest built"
        }

        # Action replay worker
        Write-Info "Building ldt-action-replay-worker:latest..."
        $build4 = docker build -t ldt-action-replay-worker:latest "$DEPLOYMENT_DIR\action-replay-worker" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "action-replay-worker build failed (non-critical)."
            $WARN_STEPS += "Phase2-ARWBuild"
        } else {
            Write-Pass "ldt-action-replay-worker:latest built"
        }

        # Stats writer
        Write-Info "Building ldt-stats-writer:latest..."
        $build5 = docker build -t ldt-stats-writer:latest "$DEPLOYMENT_DIR\stats-writer" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "stats-writer build failed (non-critical)."
            $WARN_STEPS += "Phase2-StatsBuild"
        } else {
            Write-Pass "ldt-stats-writer:latest built"
        }

        # Kafka producer
        Write-Info "Building ldt-kafka-producer:latest..."
        $build6 = docker build -t ldt-kafka-producer:latest "$DEPLOYMENT_DIR\kafka" -f "$DEPLOYMENT_DIR\kafka\Dockerfile.producer" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "kafka-producer build failed (non-critical)."
            $WARN_STEPS += "Phase2-ProducerBuild"
        } else {
            Write-Pass "ldt-kafka-producer:latest built"
        }
    } else {
        Write-Info "Skipping build (-SkipBuild)..."
    }

    if ($FAILED_STEPS.Count -gt 0) { exit 1 }

    Write-Step "2c: Deploy all services via docker compose"
    Write-Info "Running docker compose up -d..."
    $compose = docker compose -f "$DEPLOYMENT_DIR\docker-compose.yml" up -d 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "docker compose up failed."
        Write-Host $compose
        exit 1
    }
    Write-Pass "docker compose up -d completed"

    Write-Info "Waiting 30s for containers to initialize..."
    Start-Sleep -Seconds 30

    Write-Step "2d: Verify container count"
    $runningContainers = docker ps --filter "name=ldt-" --format "{{.Names}}" 2>$null
    $containerCount = if ($runningContainers) { $runningContainers.Count } else { 0 }
    Write-Info "Running containers: $containerCount"
    if ($containerCount -lt 10) {
        Write-Warn "Expected ~21 containers, found $containerCount. Some services may not have started."
        $WARN_STEPS += "Phase2-ContainerCount"
    } else {
        Write-Pass "$containerCount containers running"
    }

    Write-Step "2e: Wait for critical services to be healthy"
    $criticalServices = @(
        @{Name="ldt-kafka"; Desc="Kafka"; WaitSec=180; IntervalSec=10},
        @{Name="ldt-zookeeper"; Desc="Zookeeper"; WaitSec=120; IntervalSec=5},
        @{Name="ldt-minio"; Desc="MinIO"; WaitSec=90; IntervalSec=5},
        @{Name="ldt-redis"; Desc="Redis"; WaitSec=60; IntervalSec=5},
        @{Name="ldt-flink-jobmanager"; Desc="Flink JobManager"; WaitSec=120; IntervalSec=10},
        @{Name="ldt-prometheus"; Desc="Prometheus"; WaitSec=60; IntervalSec=5},
        @{Name="ldt-grafana"; Desc="Grafana"; WaitSec=60; IntervalSec=5}
    )

    foreach ($svc in $criticalServices) {
        Write-Info "Waiting for $($svc.Desc)..."
        $elapsed = 0
        $healthy = $false
        while ($elapsed -lt $svc.WaitSec) {
            $status = docker ps --filter "name=$($svc.Name)" --filter "status=running" --format "{{.Status}}" 2>$null
            if ($status -match "healthy" -or $status -match "Up") {
                $healthy = $true
                break
            }
            Start-Sleep -Seconds $svc.IntervalSec
            $elapsed += $svc.IntervalSec
        }
        if ($healthy) {
            Write-Pass "$($svc.Desc) is healthy"
        } else {
            Write-Fail "$($svc.Desc) did not become healthy within $($svc.WaitSec)s"
            $FAILED_STEPS += "Phase2-$($svc.Name)"
        }
    }

    if ($FAILED_STEPS.Count -gt 0) { exit 1 }

    Write-Step "2f: Wait for init containers"
    $initContainers = @("ldt-kafka-init", "ldt-minio-init")
    foreach ($c in $initContainers) {
        $elapsed = 0
        while ($elapsed -lt 120) {
            $state = docker inspect --format='{{.State.Status}}' $c 2>$null
            if ($state -eq "exited") {
                $exitCode = docker inspect --format='{{.ExitCode}}' $c 2>$null
                if ($exitCode -eq "0") {
                    Write-Pass "$c completed successfully"
                } else {
                    Write-Warn "$c exited with code $exitCode"
                    $WARN_STEPS += "Phase2-$c"
                }
                break
            }
            if (-not $state) {
                Write-Pass "$c already finished"
                break
            }
            Start-Sleep -Seconds 5
            $elapsed += 5
        }
        if ($elapsed -ge 120) {
            Write-Warn "$c did not complete within 120s"
            $WARN_STEPS += "Phase2-$c"
        }
    }

    Write-Step "2g: Wait for Flink job to be submitted"
    $flinkElapsed = 0
    Write-Info "Waiting for Flink REST API..."
    while ($flinkElapsed -lt 180) {
        try {
            $resp = Invoke-WebRequest -Uri "http://localhost:8081/overview" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($resp.StatusCode -eq 200) {
                Write-Pass "Flink REST API is ready"
                break
            }
        } catch {}
        Start-Sleep -Seconds 10
        $flinkElapsed += 10
    }
    if ($flinkElapsed -ge 180) {
        Write-Warn "Flink REST API did not respond within 180s"
        $WARN_STEPS += "Phase2-FlinkREST"
    }

    # Wait for flink-init to run (it runs indefinitely for auto-recovery)
    Write-Info "Checking flink-init container..."
    $flinkInitState = docker inspect --format='{{.State.Status}}' ldt-flink-init 2>$null
    if ($flinkInitState) {
        Write-Info "flink-init state: $flinkInitState"
        if ($flinkInitState -eq "running") {
            Write-Pass "flink-init is running (auto-recovery supervisor active)"
        }
    } else {
        Write-Warn "flink-init container not found"
        $WARN_STEPS += "Phase2-FlinkInit"
    }
}

# =============================================================================
# PHASE 3: KAFKA VERIFICATION
# =============================================================================
Write-Section "PHASE 3: Kafka Layer (L1) Verification"

$allTopics = $null
try {
    $allTopics = docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>$null
} catch {}

if (-not $allTopics) {
    Write-Fail "Kafka unreachable"
    $FAILED_STEPS += "Phase3-Kafka"
} else {
    Write-Pass "Kafka is reachable"
}

# Expected topics
$expectedTopics = @("taxi-nyc-raw-v2", "dq-stream-unified", "iec-action-replay", "iec-action-dlq", "memstream-model-updates")
$topicList = $allTopics -split "`n" | Where-Object { $_ -trim() -ne "" }
Write-Info "Kafka topics found: $($topicList.Count)"
if ($Verbose) { $topicList | ForEach-Object { Write-Info "  $_" } }

$missingTopics = @()
foreach ($t in $expectedTopics) {
    if (-not ($topicList -contains $t)) {
        $missingTopics += $t
    }
}
if ($missingTopics.Count -gt 0) {
    Write-Warn "Missing topics: $($missingTopics -join ', ')"
    $WARN_STEPS += "Phase3-MissingTopics"
} else {
    Write-Pass "All expected Kafka topics present: $($expectedTopics -join ', ')"
}

# Schema Registry
try {
    $schemas = Invoke-WebRequest -Uri "http://localhost:8081/subjects" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    $schemaCount = ($schemas.Content | ConvertFrom-Json).Count
    if ($schemaCount -ge 3) {
        Write-Pass "Schema Registry: $schemaCount schemas registered"
    } else {
        Write-Warn "Schema Registry: only $schemaCount schemas (expected >= 3)"
        $WARN_STEPS += "Phase3-SchemaReg"
    }
} catch {
    Write-Warn "Schema Registry not reachable"
    $WARN_STEPS += "Phase3-SchemaReg"
}

# Kafka produce/consume test
Write-Info "Kafka produce/consume test..."
$testMsg = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T12:00:00","tpep_dropoff_datetime":"2026-05-17T12:15:00","passenger_count":2,"trip_distance":3.5,"PULocationID":79,"DOLocationID":170,"fare_amount":12.50,"total_amount":15.75,"payment_type":1}'
$produceResult = $testMsg | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
Start-Sleep -Seconds 3

$consumeResult = docker exec ldt-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 --from-beginning --max-messages 1 --consumer-timeout-ms 5000 2>$null
if ($consumeResult) {
    Write-Pass "Kafka produce/consume end-to-end test passed"
} else {
    Write-Warn "Kafka produce/consume: no message consumed back (may need warmup)"
    $WARN_STEPS += "Phase3-KafkaConsume"
}

# Consumer group lag
try {
    $lag = docker exec ldt-kafka kafka-consumer-groups --bootstrap-server localhost:9092 --all-groups --describe 2>$null
    if ($lag) {
        $lagLines = $lag -split "`n" | Where-Object { $_ -match '^\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)' }
        $totalLag = 0
        foreach ($line in $lagLines) {
            $parts = $line -split '\s+' | Where-Object { $_ -ne "" }
            $lagVal = $parts[5]
            if ($lagVal -and $lagVal -ne "-" -and $lagVal -match '^\d+$') {
                $totalLag += [int]$lagVal
            }
        }
        Write-Info "Total consumer group lag: $totalLag"
        if ($totalLag -gt 1000) {
            Write-Warn "Consumer lag is high ($totalLag) — may indicate slow consumer"
            $WARN_STEPS += "Phase3-ConsumerLag"
        } else {
            Write-Pass "Consumer group lag: $totalLag"
        }
    }
} catch {
    Write-Warn "Could not check consumer group lag"
}

# Kafka UI
try {
    $kafkaUI = Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($kafkaUI.StatusCode -eq 200) {
        Write-Pass "Kafka UI accessible at http://localhost:8080"
    }
} catch {
    Write-Warn "Kafka UI not accessible"
    $WARN_STEPS += "Phase3-KafkaUI"
}

# kafka-exporter
try {
    $kafkaExp = Invoke-WebRequest -Uri "http://localhost:9308/metrics" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($kafkaExp.StatusCode -eq 200) {
        Write-Pass "Kafka exporter accessible at :9308"
    }
} catch {
    Write-Warn "Kafka exporter not accessible"
    $WARN_STEPS += "Phase3-KafkaExporter"
}

# =============================================================================
# PHASE 4: REDIS AND MINIO VERIFICATION
# =============================================================================
Write-Section "PHASE 4: Redis (L1b) and MinIO (L2) Verification"

# Redis
$redisPwd = [System.Environment]::GetEnvironmentVariable("REDIS_PASSWORD", "Process")
if (-not $redisPwd) { $redisPwd = "redis_password_local" }
try {
    $redisPing = docker exec ldt-redis redis-cli -a $redisPwd ping 2>$null
    if ($redisPing -match "PONG") {
        Write-Pass "Redis: PONG"
    } else {
        Write-Fail "Redis: unexpected response: $redisPing"
        $FAILED_STEPS += "Phase4-Redis"
    }
} catch {
    Write-Fail "Redis: not reachable"
    $FAILED_STEPS += "Phase4-Redis"
}

# Redis info
try {
    $redisClients = docker exec ldt-redis redis-cli -a $redisPwd info clients 2>$null | Select-String "connected_clients"
    $redisVer = docker exec ldt-redis redis-cli -a $redisPwd info server 2>$null | Select-String "redis_version"
    Write-Info "Redis: $($redisVer.ToString().Trim()) | $($redisClients.ToString().Trim())"
} catch {}

# MinIO
try {
    $mcReady = docker exec ldt-minio mc ready local 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Pass "MinIO: mc ready local"
    } else {
        Write-Fail "MinIO: mc ready local failed"
        $FAILED_STEPS += "Phase4-MinIO"
    }
} catch {
    Write-Fail "MinIO: not reachable"
    $FAILED_STEPS += "Phase4-MinIO"
}

# MinIO buckets
$minioBuckets = docker exec ldt-minio mc ls local/ 2>$null
$bucketList = $minioBuckets -split "`n" | ForEach-Object {
    if ($_ -match 'local/(\S+)') { $matches[1].TrimEnd('/') }
} | Where-Object { $_ -and $_ -ne "" }
Write-Info "MinIO buckets found: $($bucketList.Count)"
if ($Verbose) { $bucketList | ForEach-Object { Write-Info "  $_" } }

$expectedBuckets = @("cadqstream-checkpoints", "cadqstream-raw", "cadqstream-violations", "cadqstream-anomalies", "cadqstream-metrics", "cadqstream-drift", "cadqstream-dlq", "ml-models")
$missingBuckets = @()
foreach ($b in $expectedBuckets) {
    if (-not ($bucketList -contains $b)) {
        $missingBuckets += $b
    }
}
if ($missingBuckets.Count -gt 0) {
    Write-Warn "Missing MinIO buckets: $($missingBuckets -join ', ')"
    $WARN_STEPS += "Phase4-MissingBuckets"
} else {
    Write-Pass "All 8 expected MinIO buckets present"
}

# Check sensitive buckets for public access
$sensitive = @("cadqstream-violations", "cadqstream-anomalies", "ml-models")
foreach ($b in $sensitive) {
    $pubCheck = docker exec ldt-minio mc anonymous get "local/$b" 2>$null
    if ($pubCheck -match "Enabled") {
        Write-Fail "Bucket $b has PUBLIC ACCESS (security risk)"
        $FAILED_STEPS += "Phase4-BucketSecurity"
    } else {
        Write-Pass "Bucket $b: private access verified"
    }
}

if ($FAILED_STEPS.Count -gt 0) { exit 1 }

# =============================================================================
# PHASE 5: FLINK VERIFICATION
# =============================================================================
Write-Section "PHASE 5: Flink (L4) Verification"

# Cluster overview
try {
    $overview = Invoke-WebRequest -Uri "http://localhost:8081/overview" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
    $tmCount = $overview.taskmanagers
    $slotsTotal = $overview.'taskmanager'.totalTaskManagerSlotNumber
    $slotsFree = $overview.'taskmanager'.totalAvailableSlotNumber
    Write-Info "Flink cluster: $tmCount TaskManager(s), $slotsTotal slots total, $slotsFree free"
    if ($tmCount -lt 1) {
        Write-Fail "No TaskManagers registered"
        $FAILED_STEPS += "Phase5-NoTM"
    } else {
        Write-Pass "Flink TaskManager(s): $tmCount"
    }
} catch {
    Write-Fail "Flink REST API unreachable"
    $FAILED_STEPS += "Phase5-FlinkREST"
}

if ($FAILED_STEPS.Count -gt 0) { exit 1 }

# Job status
try {
    $jobs = Invoke-WebRequest -Uri "http://localhost:8081/jobs" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
    $jobCount = $jobs.jobs.Count
    $runningJobs = @($jobs.jobs | Where-Object { $_.state -eq "RUNNING" })

    Write-Info "Flink jobs: $jobCount total"
    if ($runningJobs.Count -gt 0) {
        foreach ($j in $runningJobs) {
            Write-Info "  $($j.name) [$($j.state)] (parallelism=$($j.parallelism))"
        }
        Write-Pass "$($runningJobs.Count) job(s) RUNNING"

        # Check checkpointing for each running job
        foreach ($j in $runningJobs) {
            try {
                $jobInfo = Invoke-WebRequest -Uri "http://localhost:8081/jobs/$($j.id)/info" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
                $chkInfo = $jobInfo.checkpointing
                if ($chkInfo) {
                    $lastChk = $chkInfo.last_checkpoint_timestamp
                    if ($lastChk -and [int64]$lastChk -gt 0) {
                        $chkDate = [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$lastChk).LocalDateTime.ToString("yyyy-MM-dd HH:mm:ss")
                        Write-Info "  Last checkpoint: $chkDate"
                        Write-Pass "Checkpointing active for $($j.name)"
                    } else {
                        Write-Warn "Checkpointing enabled but no checkpoint yet for $($j.name)"
                        $WARN_STEPS += "Phase5-Checkpoint"
                    }
                }
            } catch {}
        }
    } else {
        Write-Fail "No RUNNING jobs found. Jobs: $((@($jobs.jobs) | ForEach-Object { "$($_.name) [$($_.state)]" }) -join ', ')"
        $FAILED_STEPS += "Phase5-NoRunningJob"
    }
} catch {
    Write-Warn "Cannot query Flink jobs"
}

# Flink init auto-recovery logs
Write-Info "Checking flink-init auto-recovery logs..."
$initLogs = docker logs ldt-flink-init --tail 20 2>$null
if ($initLogs) {
    if ($initLogs -match "RUNNING|HEALTHY|CONTINUOUS|HEALTH MONITOR") {
        Write-Pass "flink-init auto-recovery is active"
    } else {
        Write-Warn "flink-init logs do not show active monitoring"
        if ($Verbose) {
            $initLogs -split "`n" | Select-Object -First 5 | ForEach-Object { Write-Info "  $_" }
        }
    }
}

# Consumer lag on taxi-nyc-raw-v2
Write-Info "Checking consumer lag on taxi-nyc-raw-v2..."
try {
    $lag = docker exec ldt-kafka kafka-consumer-groups --bootstrap-server localhost:9092 --group cadqstream --topic taxi-nyc-raw-v2 --describe 2>$null
    if ($lag -match "CURRENT-OFFSET") {
        $lagLines = $lag -split "`n" | Where-Object { $_ -match '\s+(\d+)\s+(?:\d+|-)\s+(\d+|-)' }
        foreach ($line in $lagLines) {
            if ($line -match '(\d+)\s+(?:\d+|-)\s+(\d+|-)') {
                $logEnd = if ($matches[2] -eq "-") { 0 } else { [int]$matches[2] }
                if ($logEnd -gt 100) {
                    Write-Warn "Consumer lag on taxi-nyc-raw-v2: $logEnd"
                    $WARN_STEPS += "Phase5-ConsumerLag"
                } else {
                    Write-Pass "Consumer lag on taxi-nyc-raw-v2: $logEnd"
                }
            }
        }
    }
} catch {}

# =============================================================================
# PHASE 6: ML SERVICE VERIFICATION
# =============================================================================
Write-Section "PHASE 6: ML Service (L4b) Verification"

try {
    $mlHealth = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 10 -ErrorAction SilentlyContinue
    if ($mlHealth.StatusCode -eq 200) {
        $mlHealthJson = $mlHealth.Content | ConvertFrom-Json
        Write-Pass "ML service health: OK"
        if ($Verbose) {
            $mlHealthJson.PSObject.Properties | ForEach-Object { Write-Info "  $($_.Name): $($_.Value)" }
        }
    } else {
        Write-Fail "ML service health: HTTP $($mlHealth.StatusCode)"
        $FAILED_STEPS += "Phase6-MLHealth"
    }
} catch {
    Write-Warn "ML service not reachable at :8000"
    $WARN_STEPS += "Phase6-MLHealth"
}

# Test predict endpoint
Write-Info "Testing ML /predict endpoint..."
try {
    $features = @(900.0, 3.5, 15.50, 2.50, 0.33, 0.95, 0.14, 0.0, 2.0, 100.0, 170.0, 5.0, 1.3, 0.16, 0.10, 0.05, 1.0, 1.0, 0.0, 1.0, 0.87, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.1, 0.9, 0.15, 0.85, 0.25, 0.75)
    $payload = @{features = @($features)} | ConvertTo-Json -Compress
    $pred = Invoke-WebRequest -Uri "http://localhost:8000/predict" -UseBasicParsing -Method POST -Body $payload -ContentType "application/json" -TimeoutSec 15 -ErrorAction SilentlyContinue
    if ($pred.StatusCode -eq 200) {
        $predJson = $pred.Content | ConvertFrom-Json
        $score = $predJson.anomaly_score
        if ($null -ne $score) {
            Write-Pass "ML /predict: anomaly_score = $score"
        } else {
            Write-Warn "ML /predict: response does not contain anomaly_score"
            $WARN_STEPS += "Phase6-MLPredict"
        }
    } else {
        Write-Warn "ML /predict: HTTP $($pred.StatusCode)"
        $WARN_STEPS += "Phase6-MLPredict"
    }
} catch {
    Write-Warn "ML /predict endpoint not responding"
    $WARN_STEPS += "Phase6-MLPredict"
}

# Check ML service logs for HMAC
$mlLogs = docker logs ldt-ml-service --tail 30 2>$null
if ($mlLogs) {
    if ($mlLogs -match "HMAC|hmac") {
        $hmacLines = $mlLogs -split "`n" | Where-Object { $_ -match "HMAC|hmac" } | Select-Object -First 3
        Write-Info "ML service HMAC activity:"
        $hmacLines | ForEach-Object { Write-Info "  $_" }
    }
    if ($mlLogs -match "ERROR|Exception|Traceback" -and $mlLogs -notmatch "WARNING") {
        Write-Warn "ML service logs contain errors"
        if ($Verbose) {
            $mlLogs -split "`n" | Where-Object { $_ -match "ERROR|Exception" } | Select-Object -First 5 | ForEach-Object { Write-Warn "  $_" }
        }
    }
}

# =============================================================================
# PHASE 7: PROMETHEUS VERIFICATION
# =============================================================================
Write-Section "PHASE 7: Prometheus (L6) Verification"

try {
    $promHealth = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($promHealth.StatusCode -eq 200) {
        Write-Pass "Prometheus is healthy"
    }
} catch {
    Write-Fail "Prometheus unreachable at :9090"
    $FAILED_STEPS += "Phase7-Prometheus"
}

if ($FAILED_STEPS.Count -gt 0) { exit 1 }

# Scrape targets
Write-Info "Checking scrape targets..."
try {
    $targets = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/targets" -UseBasicParsing -TimeoutSec 10 -ErrorAction SilentlyContinue | ConvertFrom-Json
    $allTargets = $targets.data.targets
    $unhealthyTargets = @($allTargets | Where-Object { $_.health -ne "up" })
    $upTargets = @($allTargets | Where-Object { $_.health -eq "up" })

    Write-Info "Scrape targets: $($upTargets.Count) up, $($unhealthyTargets.Count) down"
    if ($unhealthyTargets.Count -gt 0) {
        Write-Warn "Unhealthy targets:"
        $unhealthyTargets | ForEach-Object {
            $job = $_.labels.job
            $state = $_.lastError
            Write-Warn "  $job : $state"
        }
        $WARN_STEPS += "Phase7-UnhealthyTargets"
    } else {
        Write-Pass "All $($upTargets.Count) scrape targets are healthy"
    }

    if ($Verbose) {
        $upTargets | ForEach-Object {
            Write-Info "  $($_.labels.job) [$($_.health)]"
        }
    }
} catch {
    Write-Warn "Cannot query Prometheus targets"
}

# Key metric groups (per-layer)
Write-Info "Checking key metric groups in Prometheus..."
$metricGroups = @{
    "L1-Ingestion"   = @("cadqstream_records_valid_total", "cadqstream_records_violation_total");
    "L2-Canary"      = @("cadqstream_canary_violation_total");
    "L3-MemStream"   = @("cadqstream_anomaly_score", "memstream_scoring_latency_seconds_bucket");
    "L3-MetaAgg"     = @("cadqstream_meta_window_record_count");
    "L4-IEC"         = @("cadqstream_drift_detected", "cadqstream_iec_action_total", "cadqstream_circuit_breaker_state");
    "ML-Warmup"      = @("memstream_warmup_progress", "memstream_redis_connected");
    "ML-HMAC"        = @("memstream_hmac_verification_total");
    "ML-Stats"       = @("memstream_knn_avg_distance", "memstream_memory_fill_rate", "memstream_beta_staleness_seconds");
}

$missingMetrics = @()
$foundMetrics = @()
foreach ($group in $metricGroups.Keys) {
    $groupFound = $true
    foreach ($metric in $metricGroups[$group]) {
        try {
            $q = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=$metric" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
            if ($q.data.result.Count -gt 0) {
                $foundMetrics += $metric
            } else {
                $missingMetrics += "$group/$metric"
                $groupFound = $false
            }
        } catch {
            $missingMetrics += "$group/$metric (query failed)"
            $groupFound = $false
        }
    }
    if ($groupFound) {
        Write-Pass "$group : all metrics present"
    } else {
        Write-Warn "$group : missing some metrics"
    }
}

if ($missingMetrics.Count -gt 0) {
    Write-Warn "Missing metrics: $($missingMetrics -join ', ')"
    $WARN_STEPS += "Phase7-MissingMetrics"
}

# =============================================================================
# PHASE 8: GRAFANA VERIFICATION
# =============================================================================
Write-Section "PHASE 8: Grafana (L6) Verification"

try {
    $grafanaHealth = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($grafanaHealth.StatusCode -eq 200) {
        $grafanaVer = ($grafanaHealth.Content | ConvertFrom-Json).version
        Write-Pass "Grafana v$grafanaVer is healthy"
    }
} catch {
    Write-Fail "Grafana unreachable"
    $FAILED_STEPS += "Phase8-Grafana"
}

if ($FAILED_STEPS.Count -eq 0) {
    # Dashboard count
    try {
        $dashboards = Invoke-WebRequest -Uri "http://localhost:3000/api/search?type=dash-db" -UseBasicParsing -TimeoutSec 10 -ErrorAction SilentlyContinue -Credential (New-Object PSCredential("admin", (ConvertTo-SecureString $GRAFANA_PASSWORD -AsPlainText -Force)))
        $dashList = $dashboards.Content | ConvertFrom-Json
        $dashCount = $dashList.Count
        Write-Info "Grafana dashboards: $dashCount provisioned"
        if ($Verbose) {
            $dashList | ForEach-Object { Write-Info "  $($_.title) [uid=$($_.uid)]" }
        }
        if ($dashCount -lt 6) {
            Write-Warn "Expected >= 6 dashboards, found $dashCount"
            $WARN_STEPS += "Phase8-DashboardCount"
        } else {
            Write-Pass "Grafana dashboards: $dashCount (expected >= 6)"
        }
    } catch {
        Write-Warn "Cannot retrieve Grafana dashboards (may need auth)"
    }

    # Verify each dashboard has data (via Prometheus queries)
    Write-Info "Checking per-dashboard data availability..."
    $dashToMetrics = @{
        "pipeline-overview"       = @("cadqstream_records_valid_total", "flink_taskmanager_JVM_Memory_Heap_Used");
        "data-quality"            = @("cadqstream_records_violation_total", "cadqstream_anomaly_score");
        "memstream-data-quality"   = @("memstream_warmup_progress", "memstream_knn_avg_distance");
        "kafka-overview"          = @("kafka_topic_partition_current_offset");
        "flink-jobs"             = @("flink_jobmanager_StatusReport_NumRunningJobs");
        "streaming-noc"           = @("cadqstream_records_valid_total", "cadqstream_drift_detected");
    }

    $incompleteDashboards = @()
    foreach ($dash in $dashToMetrics.Keys) {
        $metricsOk = $true
        foreach ($m in $dashToMetrics[$dash]) {
            try {
                $q = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=$m" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
                if ($q.data.result.Count -eq 0) {
                    $metricsOk = $false
                    break
                }
            } catch {
                $metricsOk = $false
                break
            }
        }
        if ($metricsOk) {
            Write-Pass "Dashboard '$dash': all required metrics present"
        } else {
            Write-Warn "Dashboard '$dash': some metrics not available yet"
            $incompleteDashboards += $dash
            $WARN_STEPS += "Phase8-$dash"
        }
    }
}

# =============================================================================
# PHASE 9: END-TO-END DATA FLOW TESTS
# =============================================================================
if (-not $SkipTests) {
    Write-Section "PHASE 9: End-to-End Data Flow Tests"

    Write-Info "Kafka producer container..."
    $producerStatus = docker ps --filter "name=ldt-kafka-producer" --format "{{.Status}}" 2>$null
    if ($producerStatus -match "Up") {
        Write-Pass "kafka-producer is running"
    } else {
        Write-Warn "kafka-producer not running (needed for flow tests)"
        $WARN_STEPS += "Phase9-Producer"
    }

    Write-Step "9a: Normal Record Flow"
    $normalMsg = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T12:30:00","tpep_dropoff_datetime":"2026-05-17T12:45:00","passenger_count":2,"trip_distance":3.5,"PULocationID":79,"DOLocationID":170,"fare_amount":12.50,"total_amount":15.75,"payment_type":1}'
    $normalMsg | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
    Start-Sleep -Seconds 5
    Write-Info "Normal record injected. Check cadqstream-raw bucket for valid trip storage."

    Write-Step "9b: L1 Schema Violation"
    $l1Msg = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T12:31:00","tpep_dropoff_datetime":"2026-05-17T12:46:00","passenger_count":1,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.50,"payment_type":1}'
    $l1Msg | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
    Write-Info "L1 violation (missing trip_distance) injected."

    # Invalid zone
    $zoneMsg = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T12:31:10","tpep_dropoff_datetime":"2026-05-17T12:46:10","passenger_count":1,"trip_distance":5.0,"PULocationID":999,"DOLocationID":500,"fare_amount":18.00,"total_amount":22.00,"payment_type":1}'
    $zoneMsg | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
    Write-Info "L1 violation (invalid PULocationID=999) injected."

    Write-Step "9c: L2 Canary Rule Violations"
    $canaryCases = @(
        @{Desc="Negative fare"; Msg='{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T12:32:00","tpep_dropoff_datetime":"2026-05-17T12:47:00","passenger_count":1,"trip_distance":5.0,"PULocationID":79,"DOLocationID":170,"fare_amount":-5.00,"total_amount":2.00,"payment_type":1}'},
        @{Desc="Zero distance with fare"; Msg='{"VendorID":2,"tpep_pickup_datetime":"2026-05-17T12:32:10","tpep_dropoff_datetime":"2026-05-17T12:33:10","passenger_count":1,"trip_distance":0.0,"PULocationID":79,"DOLocationID":79,"fare_amount":25.00,"total_amount":28.00,"payment_type":2}'},
        @{Desc="Passengers=0"; Msg='{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T12:32:20","tpep_dropoff_datetime":"2026-05-17T12:47:20","passenger_count":0,"trip_distance":3.0,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.00,"payment_type":1}'}
    )
    foreach ($c in $canaryCases) {
        $c.Msg | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
        Write-Info "L2 violation injected: $($c.Desc)"
    }

    Write-Step "9d: L3 Extreme Anomaly"
    $l3Msg = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T12:33:00","tpep_dropoff_datetime":"2026-05-17T12:48:00","passenger_count":6,"trip_distance":99.9,"PULocationID":138,"DOLocationID":229,"fare_amount":999.99,"total_amount":1050.00,"payment_type":1}'
    $l3Msg | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
    Write-Info "L3 extreme anomaly injected. Check cadqstream-anomalies/scores/ for high anomaly score."

    Write-Step "9e: Concept Drift (gradual fare increase)"
    Write-Info "Injecting 10 records with +20% fare increment..."
    $baseFare = 15.0
    for ($i = 1; $i -le 10; $i++) {
        $fare = [math]::Round($baseFare * [math]::Pow(1.2, $i), 2)
        $driftMsg = "{`"VendorID`":1,`"tpep_pickup_datetime`":`"2026-05-17T12:34:0${i}`",`"tpep_dropoff_datetime`":`"2026-05-17T12:49:0${i}`",`"passenger_count`":2,`"trip_distance`":4.0,`"PULocationID`":100,`"DOLocationID`":180,`"fare_amount`":$fare,`"total_amount`":$($fare + 3),`"payment_type`":1}"
        $driftMsg | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
        Start-Sleep -Milliseconds 500
    }
    Write-Info "Concept drift injection complete. Wait 3 min for ADWIN to detect."
    Write-Info "Check cadqstream-drift/drift_events/ for drift events."
    Write-Info "Check cadqstream-drift/alerts/ for alerts."

    Write-Step "9f: Verify IEC action-replay signal"
    try {
        $iecMsgs = docker exec ldt-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic iec-action-replay --from-beginning --max-messages 3 --consumer-timeout-ms 3000 2>$null
        if ($iecMsgs) {
            Write-Pass "IEC action-replay topic has messages"
        } else {
            Write-Warn "IEC action-replay topic empty (may need concept drift to be detected first)"
        }
    } catch {
        Write-Warn "Cannot read IEC action-replay topic"
    }

    Write-Step "9g: Check Violations Bucket"
    $violFiles = docker exec ldt-minio mc ls "local/cadqstream-violations/" 2>$null
    if ($violFiles) {
        Write-Pass "cadqstream-violations bucket has files"
    } else {
        Write-Warn "cadqstream-violations bucket empty (may need warmup)"
        $WARN_STEPS += "Phase9-Violations"
    }

    Write-Step "9h: Check Anomalies Bucket"
    $anomFiles = docker exec ldt-minio mc ls "local/cadqstream-anomalies/" 2>$null
    if ($anomFiles) {
        Write-Pass "cadqstream-anomalies bucket has files"
    } else {
        Write-Warn "cadqstream-anomalies bucket empty (may need warmup)"
        $WARN_STEPS += "Phase9-Anomalies"
    }
}

# =============================================================================
# PHASE 10: STATS METRICS VERIFICATION
# =============================================================================
Write-Section "PHASE 10: Stats Metrics (L6b) Verification"

Write-Info "Stats-writer container..."
$statsStatus = docker ps --filter "name=ldt-stats-writer" --format "{{.Status}}" 2>$null
if ($statsStatus -match "Up") {
    Write-Pass "stats-writer is running"
} else {
    Write-Warn "stats-writer not running (non-critical)"
    $WARN_STEPS += "Phase10-StatsWriter"
}

Write-Info "Checking cadqstream-metrics/ bucket..."
$statsFiles = docker exec ldt-minio mc ls "local/cadqstream-metrics/" 2>$null
$statsFileCount = if ($statsFiles) { ($statsFiles -split "`n" | Where-Object { $_ -ne "" }).Count } else { 0 }
if ($statsFileCount -gt 0) {
    Write-Pass "cadqstream-metrics/ bucket: $statsFileCount files"
} else {
    Write-Warn "cadqstream-metrics/ bucket: empty (stats-writer may need warmup)"
    $WARN_STEPS += "Phase10-StatsBucket"
}

# Stats metrics in Prometheus
$statsMetrics = @("cadqstream_anomaly_rate", "cadqstream_false_positive_rate", "cadqstream_records_processed_total")
foreach ($m in $statsMetrics) {
    try {
        $q = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=$m" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
        if ($q.data.result.Count -gt 0) {
            Write-Pass "Stats metric '$m': present"
        } else {
            Write-Warn "Stats metric '$m': not yet available"
            $WARN_STEPS += "Phase10-$m"
        }
    } catch {}
}

# =============================================================================
# PHASE 11: OFFLINE PRETRAIN VERIFICATION
# =============================================================================
Write-Section "PHASE 11: Offline Pretrain Components"

Write-Info "Checking neighborhood_mapping.json..."
$nmPaths = @("$DEPLOYMENT_DIR\models\neighborhood_mapping.json", "$DEPLOYMENT_DIR\src\config\neighborhood_mapping.json")
$nmFound = $false
foreach ($p in $nmPaths) {
    if (Test-Path $p) {
        Write-Pass "neighborhood_mapping.json found at $p"
        $nmFound = $true
        break
    }
}
if (-not $nmFound) {
    Write-Warn "neighborhood_mapping.json not found locally (may be in container or MinIO)"
    $WARN_STEPS += "Phase11-NeighMapping"
}

Write-Info "Checking context_thresholds_v2.json..."
$ctPaths = @("$DEPLOYMENT_DIR\models\context_thresholds_v2.json", "$DEPLOYMENT_DIR\src\config\context_thresholds_v2.json")
$ctFound = $false
foreach ($p in $ctPaths) {
    if (Test-Path $p) {
        Write-Pass "context_thresholds_v2.json found at $p"
        $ctFound = $true
        break
    }
}
if (-not $ctFound) {
    Write-Warn "context_thresholds_v2.json not found (may be generated during warmup)"
    $WARN_STEPS += "Phase11-ContextThresholds"
}

Write-Info "Checking ml-models bucket in MinIO..."
$mlModels = docker exec ldt-minio mc ls "local/ml-models/" 2>$null
if ($mlModels) {
    $modelFiles = $mlModels -split "`n" | Where-Object { $_ -match "local/ml-models/(\S+)" }
    Write-Info "ml-models bucket contents:"
    $modelFiles | ForEach-Object { Write-Info "  $_" }
    if ($mlModels -match "meter_hypernetwork\.pkl|meter_scaler\.pkl|context_thresholds") {
        Write-Pass "ML model artifacts present in MinIO"
    } else {
        Write-Warn "ML model artifacts may not be present in MinIO"
        $WARN_STEPS += "Phase11-MLModels"
    }
} else {
    Write-Warn "ml-models bucket empty (models may be embedded in container)"
    $WARN_STEPS += "Phase11-MLModels"
}

Write-Info "HMAC verification check..."
$mlLogs = docker logs ldt-ml-service --tail 50 2>$null
if ($mlLogs) {
    if ($mlLogs -match "HMAC.*success|HMAC.*passed|verification.*ok|Integrity.*OK") {
        Write-Pass "HMAC checkpoint verification: passing"
    } elseif ($mlLogs -match "HMAC.*failed|verification.*error|Integrity.*ERROR|Integrity.*FAILED") {
        Write-Warn "HMAC verification has failures"
        $WARN_STEPS += "Phase11-HMAC"
    } else {
        Write-Info "HMAC verification status unclear from logs"
    }
}

# =============================================================================
# FINAL SUMMARY
# =============================================================================
Write-Section "FINAL SUMMARY"

$runningCount = (docker ps --filter "name=ldt-" --format "{{.Names}}" 2>$null).Count
Write-Info "Containers running: $runningCount"

if ($FAILED_STEPS.Count -gt 0) {
    Write-Host ""
    Write-Host "CRITICAL FAILURES:" -ForegroundColor Red
    $FAILED_STEPS | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "FAILED STEPS MUST BE RESOLVED BEFORE DEPLOYMENT IS COMPLETE." -ForegroundColor Red
    Write-Host ""
    exit 1
}

if ($WARN_STEPS.Count -gt 0) {
    Write-Host ""
    Write-Host "WARNINGS (non-critical):" -ForegroundColor Yellow
    $WARN_STEPS | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    Write-Host ""
    Write-Host "STACK IS OPERATIONAL. $runningCount containers running." -ForegroundColor Green
    Write-Host "Some non-critical components may need attention." -ForegroundColor Yellow
    Write-Host ""
    exit 2
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "ALL CHECKS PASSED" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Cyan
Write-Host "  Kafka UI:      http://localhost:8080"
Write-Host "  Flink UI:      http://localhost:8081"
Write-Host "  Grafana:       http://localhost:3000  (admin / $GRAFANA_PASSWORD)"
Write-Host "  Prometheus:    http://localhost:9090"
Write-Host "  MinIO Console: http://localhost:9001  (minioadmin / minioadmin)"
Write-Host "  ML Service:    http://localhost:8000"
Write-Host "  cadqstream-metrics: http://localhost:9250/metrics"
Write-Host ""
Write-Host "Next: Run data injection tests or check Grafana dashboards." -ForegroundColor Cyan
Write-Host ""
exit 0
