# =============================================================================
# CA-DQStream - Flow Verification Script (PowerShell)
# Verifies end-to-end data flow without redeploying.
# Run from project root: powershell -ExecutionPolicy Bypass -File deployment/scripts/verify-flow.ps1
#
# Exit codes: 0=pass, 1=fail (stop-on-fail), 2=warnings only
# =============================================================================

param(
    [switch]$Quick,
    [switch]$Inject,
    [string]$GrafanaPassword
)

$ErrorActionPreference = "Continue"
$DEPLOYMENT_DIR = Split-Path -Parent $PSScriptRoot
if (-not $env:GRAFANA_PASSWORD) {
    Write-Host "ERROR: GRAFANA_PASSWORD environment variable not set" -ForegroundColor Red
    Write-Host "Set it via: \$env:GRAFANA_PASSWORD = 'your_password'"
    exit 1
}
$GrafanaPassword = $env:GRAFANA_PASSWORD

function Write-Pass($msg) { Write-Host "  [PASS] $msg" -ForegroundColor Green }
function Write-Fail($msg) { Write-Host "  [FAIL] $msg" -ForegroundColor Red }
function Write-Warn($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Info($msg) { Write-Host "  [INFO] $msg" -ForegroundColor Cyan }
function Write-Hdr($msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Magenta }
function Write-Sec($msg) { Write-Host ""; Write-Host "== $msg ==" -ForegroundColor Magenta }

$FAILED = @()
$WARNED = @()

# =============================================================================
# HEADER
# =============================================================================
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  CA-DQStream - Flow Verification" -ForegroundColor Cyan
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# =============================================================================
# A. CONTAINER STATUS
# =============================================================================
Write-Hdr "A. Container Status"
$containers = docker ps --filter "name=ldt-" --format "{{.Names}}|{{.Status}}" 2>$null
if (-not $containers) {
    Write-Fail "No ldt-* containers found. Is the stack deployed?"
    $FAILED += "A-Containers"
} else {
    $up = 0; $down = 0
    $containers -split "`n" | ForEach-Object {
        if ($_ -match "\|Up ") { $up++ } else { $down++ }
    }
    Write-Info "Running: $up, Down/Not running: $down"
    if ($down -gt 0) {
        Write-Warn "Some containers not running:"
        $containers -split "`n" | Where-Object { $_ -notmatch "\|Up " } | ForEach-Object { Write-Warn "  $_" }
        $WARNED += "A-SomeContainersDown"
    } else {
        Write-Pass "All containers running"
    }
}

# =============================================================================
# B. KAFKA FLOW
# =============================================================================
Write-Hdr "B. Kafka Flow"

Write-Sec "B1. Kafka Cluster"
try {
    $topics = docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>$null
    if ($topics) {
        $tCount = ($topics -split "`n" | Where-Object { $_ -ne "" }).Count
        Write-Pass "Kafka reachable: $tCount topics"
    }
} catch {
    Write-Fail "Kafka unreachable"
    $FAILED += "B-Kafka"
}

Write-Sec "B2. Topic Validation"
$expected = @("taxi-nyc-raw-v2", "dq-stream-unified", "iec-action-replay", "iec-action-dlq", "memstream-model-updates")
if ($topics) {
    $tList = $topics -split "`n" | Where-Object { $_ -ne "" }
    foreach ($t in $expected) {
        if ($tList -contains $t) { Write-Pass "Topic: $t" }
        else { Write-Warn "Topic missing: $t"; $WARNED += "B-MissingTopic-$t" }
    }
}

Write-Sec "B3. Avro Schemas"
try {
    $schemas = Invoke-WebRequest -Uri "http://localhost:8081/subjects" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    $sCount = ($schemas.Content | ConvertFrom-Json).Count
    if ($sCount -ge 3) { Write-Pass "Schemas registered: $sCount (expected >= 3)" }
    else { Write-Warn "Schemas: $sCount (expected >= 3)"; $WARNED += "B-SchemaCount" }
} catch {
    Write-Warn "Schema Registry unreachable"
}

Write-Sec "B4. Kafka Exporter"
try {
    $kExp = Invoke-WebRequest -Uri "http://localhost:9308/metrics" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($kExp.StatusCode -eq 200) { Write-Pass "Kafka exporter at :9308" }
} catch {
    Write-Warn "Kafka exporter unreachable"
}

Write-Sec "B5. Consumer Group Lag"
try {
    $lagOut = docker exec ldt-kafka kafka-consumer-groups --bootstrap-server localhost:9092 --all-groups --describe 2>$null
    if ($lagOut) {
        $totalLag = 0
        $lagOut -split "`n" | ForEach-Object {
            if ($_ -match '^\S+\s+\S+\s+\S+\s+\S+\s+(\d+|-)\s+(\d+|-)') {
                $end = if ($matches[2] -eq "-") { 0 } else { [int]$matches[2] }
                $totalLag += $end
            }
        }
        if ($totalLag -eq 0) { Write-Pass "Consumer lag: 0 (all caught up)" }
        elseif ($totalLag -lt 1000) { Write-Info "Consumer lag: $totalLag (low)" }
        else { Write-Warn "Consumer lag: $totalLag (high)"; $WARNED += "B-ConsumerLag" }
    }
} catch {}

# =============================================================================
# C. FLINK PIPELINE
# =============================================================================
Write-Hdr "C. Flink Pipeline (L4)"

Write-Sec "C1. Cluster Status"
try {
    $overview = Invoke-WebRequest -Uri "http://localhost:8081/overview" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
    $tms = $overview.taskmanagers
    $slots = $overview.'taskmanager'.totalTaskManagerSlotNumber
    Write-Info "TaskManagers: $tms, Slots total: $slots"
    if ($tms -lt 1) { Write-Fail "No TaskManagers"; $FAILED += "C-NoTM" }
    else { Write-Pass "TaskManager(s): $tms" }
} catch {
    Write-Fail "Flink REST API unreachable"
    $FAILED += "C-FlinkREST"
}

if ($FAILED -contains "C-FlinkREST") {}
else {
    Write-Sec "C2. Job Status"
    try {
        $jobs = Invoke-WebRequest -Uri "http://localhost:8081/jobs" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
        $running = @($jobs.jobs | Where-Object { $_.state -eq "RUNNING" })

        if ($running.Count -gt 0) {
            Write-Pass "$($running.Count) job(s) RUNNING"
            foreach ($j in $running) {
                Write-Info "  $($j.name) [parallelism=$($j.parallelism)]"
            }
        } else {
            $failed = @($jobs.jobs | Where-Object { $_.state -eq "FAILED" })
            $cancelled = @($jobs.jobs | Where-Object { $_.state -eq "CANCELED" })
            if ($failed.Count -gt 0) {
                Write-Fail "Job FAILED: $((@($jobs.jobs) | Where-Object { $_.state -eq 'FAILED' } | ForEach-Object { $_.name }) -join ', ')"
                $FAILED += "C-JobFailed"
            }
            if ($cancelled.Count -gt 0) {
                Write-Warn "Job CANCELED: $((@($jobs.jobs) | Where-Object { $_.state -eq 'CANCELED' } | ForEach-Object { $_.name }) -join ', ')"
                $WARNED += "C-JobCancelled"
            }
            if ($jobs.jobs.Count -eq 0) { Write-Warn "No Flink jobs found" }
            else { Write-Fail "No RUNNING jobs found"; $FAILED += "C-NoRunningJob" }
        }
    } catch {
        Write-Warn "Cannot query Flink jobs"
    }

    Write-Sec "C3. Checkpointing"
    try {
        $jobs = Invoke-WebRequest -Uri "http://localhost:8081/jobs" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
        $running = @($jobs.jobs | Where-Object { $_.state -eq "RUNNING" })
        $chkActive = $false
        foreach ($j in $running) {
            try {
                $ji = Invoke-WebRequest -Uri "http://localhost:8081/jobs/$($j.id)/info" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
                $chk = $ji.checkpointing
                if ($chk) {
                    $lastTs = $chk.last_checkpoint_timestamp
                    if ($lastTs -and [int64]$lastTs -gt 0) {
                        $chkDate = [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$lastTs).LocalDateTime.ToString("yyyy-MM-dd HH:mm:ss")
                        Write-Info "Last checkpoint for $($j.name): $chkDate"
                        $chkActive = $true
                    }
                }
            } catch {}
        }
        if ($chkActive) { Write-Pass "Checkpointing active" }
        else { Write-Warn "No recent checkpoint found"; $WARNED += "C-NoCheckpoint" }
    } catch {}

    Write-Sec "C4. flink-init Auto-Recovery"
    $initLogs = docker logs ldt-flink-init --tail 15 2>$null
    if ($initLogs) {
        if ($initLogs -match "RUNNING|HEALTHY|CONTINUOUS|HEALTH MONITOR|AUTO-RECOVERY") {
            Write-Pass "flink-init auto-recovery is active"
        } else {
            Write-Info "flink-init logs (last 15 lines):"
            $initLogs -split "`n" | Select-Object -Last 15 | ForEach-Object { Write-Info "  $_" }
        }
    } else {
        Write-Warn "flink-init logs unavailable"
    }

    Write-Sec "C5. Consumer Lag on taxi-nyc-raw-v2"
    try {
        $lagOut = docker exec ldt-kafka kafka-consumer-groups --bootstrap-server localhost:9092 --all-groups --topic taxi-nyc-raw-v2 --describe 2>$null
        if ($lagOut) {
            $lagOut -split "`n" | ForEach-Object {
                if ($_ -match 'CURRENT-OFFSET') {
                    $parts = $_ -split '\s+'
                    $lagIdx = 5
                    if ($parts[$lagIdx] -match '^\d+$') {
                        $lag = [int]$parts[$lagIdx]
                        if ($lag -eq 0) { Write-Pass "Lag on taxi-nyc-raw-v2: 0" }
                        elseif ($lag -lt 500) { Write-Info "Lag on taxi-nyc-raw-v2: $lag" }
                        else { Write-Warn "Lag on taxi-nyc-raw-v2: $lag"; $WARNED += "C-TopicLag" }
                    }
                }
            }
        }
    } catch {}
}

# =============================================================================
# D. ML SERVICE
# =============================================================================
Write-Hdr "D. ML Service (L4b)"
Write-Sec "D1. Health"
try {
    $h = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 10 -ErrorAction SilentlyContinue
    if ($h.StatusCode -eq 200) {
        $hj = $h.Content | ConvertFrom-Json
        Write-Pass "ML service healthy"
        if ($VerbosePreference -or $Quick -eq $false) {
            $hj.PSObject.Properties | ForEach-Object { Write-Info "  $($_.Name): $($_.Value)" }
        }
    } else {
        Write-Fail "ML service: HTTP $($h.StatusCode)"; $FAILED += "D-MLHealth"
    }
} catch {
    Write-Warn "ML service unreachable"; $WARNED += "D-MLHealth"
}

Write-Sec "D2. Predict Endpoint"
if ($FAILED -notcontains "D-MLHealth") {
    try {
        $feats = @(900.0, 3.5, 15.50, 2.50, 0.33, 0.95, 0.14, 0.0, 2.0, 100.0, 170.0, 5.0, 1.3, 0.16, 0.10, 0.05, 1.0, 1.0, 0.0, 1.0, 0.87, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.1, 0.9, 0.15, 0.85, 0.25, 0.75)
        $pay = @{features = @($feats)} | ConvertTo-Json -Compress
        $p = Invoke-WebRequest -Uri "http://localhost:8000/predict" -UseBasicParsing -Method POST -Body $pay -ContentType "application/json" -TimeoutSec 15 -ErrorAction SilentlyContinue
        if ($p.StatusCode -eq 200) {
            $pj = $p.Content | ConvertFrom-Json
            if ($null -ne $pj.anomaly_score) {
                Write-Pass "ML predict: anomaly_score = $($pj.anomaly_score)"
            } else {
                Write-Warn "ML predict: no anomaly_score in response"
            }
        } else {
            Write-Warn "ML predict: HTTP $($p.StatusCode)"; $WARNED += "D-MLPredict"
        }
    } catch {
        Write-Warn "ML predict: exception"; $WARNED += "D-MLPredict"
    }
}

Write-Sec "D3. ML Model Artifacts"
$mlModels = docker exec ldt-minio mc ls "local/ml-models/" 2>$null
if ($mlModels) {
    $files = $mlModels -split "`n" | Where-Object { $_ -match "local/ml-models/(\S+)" }
    Write-Info "ml-models bucket contents:"
    $files | ForEach-Object { Write-Info "  $_" }
    if ($mlModels -match "meter_hypernetwork|meter_scaler|context_thresholds") {
        Write-Pass "Model artifacts present in MinIO"
    } else {
        Write-Warn "Model artifacts not visible in ml-models bucket"
    }
} else {
    Write-Info "ml-models bucket empty (models embedded in container)"
}

Write-Sec "D4. HMAC Status"
$mlLogs = docker logs ldt-ml-service --tail 20 2>$null
if ($mlLogs) {
    if ($mlLogs -match "HMAC.*success|HMAC.*OK|verif.*ok") {
        Write-Pass "HMAC verification: passing"
    } elseif ($mlLogs -match "HMAC.*failed|verif.*error") {
        Write-Warn "HMAC verification failures detected"
        $WARNED += "D-HMAC"
    } else {
        Write-Info "HMAC status unclear from logs"
    }
}

# =============================================================================
# E. PROMETHEUS
# =============================================================================
Write-Hdr "E. Prometheus (L6)"

Write-Sec "E1. Scrape Targets"
try {
    $tgt = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/targets" -UseBasicParsing -TimeoutSec 10 -ErrorAction SilentlyContinue | ConvertFrom-Json
    $unhealthy = @($tgt.data.targets | Where-Object { $_.health -ne "up" })
    $upCount = @($tgt.data.targets | Where-Object { $_.health -eq "up" }).Count
    if ($unhealthy.Count -eq 0) {
        Write-Pass "All $upCount scrape targets healthy"
    } else {
        Write-Warn "$upCount up, $($unhealthy.Count) down"
        $unhealthy | ForEach-Object { Write-Warn "  $($_.labels.job): $($_.lastError)" }
        $WARNED += "E-UnhealthyTargets"
    }
} catch {
    Write-Warn "Cannot query Prometheus targets"
}

Write-Sec "E2. Key Metrics (per layer)"
$metrics = @(
    @{Layer="L1-Ingestion"; M="cadqstream_records_valid_total"},
    @{Layer="L1-Violation"; M="cadqstream_records_violation_total"},
    @{Layer="L2-Canary"; M="cadqstream_canary_violation_total"},
    @{Layer="L3-MemStream"; M="cadqstream_anomaly_score"},
    @{Layer="L3-MemStream"; M="memstream_scoring_latency_seconds_bucket"},
    @{Layer="L4-IEC"; M="cadqstream_drift_detected"},
    @{Layer="L4-IEC"; M="cadqstream_iec_action_total"},
    @{Layer="L4-CircuitBrk"; M="cadqstream_circuit_breaker_state"},
    @{Layer="ML-Warmup"; M="memstream_warmup_progress"},
    @{Layer="ML-HMAC"; M="memstream_hmac_verification_total"},
    @{Layer="ML-kNN"; M="memstream_knn_avg_distance"},
    @{Layer="ML-Memory"; M="memstream_memory_fill_rate"},
    @{Layer="ML-Beta"; M="memstream_beta_staleness_seconds"}
)
$missed = 0
foreach ($m in $metrics) {
    try {
        $q = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=$($m.M)" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
        if ($q.data.result.Count -gt 0) { }
        else { Write-Warn "  $($m.Layer) / $($m.M): not found"; $missed++ }
    } catch { Write-Warn "  $($m.M): query failed" }
}
if ($missed -eq 0) { Write-Pass "All 13 key metrics present" }
else { Write-Warn "$missed metric(s) missing"; $WARNED += "E-MissingMetrics" }

# =============================================================================
# F. GRAFANA
# =============================================================================
Write-Hdr "F. Grafana (L6)"

try {
    $gh = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($gh.StatusCode -eq 200) {
        $gver = ($gh.Content | ConvertFrom-Json).version
        Write-Pass "Grafana v$gver healthy"
    }
} catch {
    Write-Fail "Grafana unreachable"; $FAILED += "F-Grafana"
}

if ($FAILED -notcontains "F-Grafana") {
    try {
        $creds = New-Object PSCredential("admin", (ConvertTo-SecureString $GrafanaPassword -AsPlainText -Force))
        $dashes = Invoke-WebRequest -Uri "http://localhost:3000/api/search?type=dash-db" -UseBasicParsing -TimeoutSec 10 -Credential $creds -ErrorAction SilentlyContinue
        $dList = $dashes.Content | ConvertFrom-Json
        $dCount = $dList.Count
        Write-Info "Dashboards provisioned: $dCount"
        if ($dCount -ge 6) { Write-Pass "Dashboard count: $dCount (>= 6)" }
        else { Write-Warn "Dashboard count: $dCount (expected >= 6)"; $WARNED += "F-DashboardCount" }

        if (-not $Quick) {
            Write-Info "Dashboard list:"
            $dList | ForEach-Object { Write-Info "  $($_.title) [uid=$($_.uid)]" }
        }
    } catch {
        Write-Warn "Cannot retrieve dashboards (auth issue)"
    }

    # Per-pane validation: check that each dashboard's key metrics have data
    Write-Sec "F2. Per-Dashboard Data Check"
    $dashMetrics = @{
        "pipeline-overview"        = @("cadqstream_records_valid_total", "flink_taskmanager_JVM_Memory_Heap_Used");
        "data-quality"           = @("cadqstream_records_violation_total", "cadqstream_anomaly_score");
        "memstream-data-quality" = @("memstream_warmup_progress", "memstream_knn_avg_distance");
        "kafka-overview"         = @("kafka_consumer_group_lag");
        "flink-jobs"             = @("flink_jobmanager_StatusReport_NumRunningJobs");
        "streaming-noc"          = @("cadqstream_records_valid_total", "cadqstream_drift_detected");
    }
    foreach ($dash in $dashMetrics.Keys) {
        $ok = $true
        foreach ($m in $dashMetrics[$dash]) {
            try {
                $q = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=$m" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
                if ($q.data.result.Count -eq 0) { $ok = $false; break }
            } catch { $ok = $false; break }
        }
        if ($ok) { Write-Pass "$dash : data flowing" }
        else { Write-Warn "$dash : data NOT flowing (some panels may be empty)"; $WARNED += "F-$dash-Data" }
    }
}

# =============================================================================
# G. MINIO STORAGE
# =============================================================================
Write-Hdr "G. MinIO Storage (L2)"

Write-Sec "G1. Buckets"
# Use docker run --rm with mc image to check MinIO buckets (minio-init has restart:no)
$envVars = @(
    "MC_HOST=http://minio:9000",
    "MC_ALIAS=local",
    "MC_USER=minioadmin",
    "MC_PASS=minioadmin123"
)
$envArg = ($envVars | ForEach-Object { "-e", $_ }) -join " "
$buckets = docker run --rm --network cadqstream-net $envArg minio/mc ls local/ 2>$null
if ($buckets) {
    $bCount = ($buckets -split "`n" | Where-Object { $_ -ne "" }).Count
    Write-Info "MinIO buckets: $bCount"
    $expected = @("cadqstream-checkpoints", "cadqstream-raw", "cadqstream-violations", "cadqstream-anomalies", "cadqstream-metrics", "cadqstream-drift", "cadqstream-dlq", "ml-models")
    $bList = $buckets -split "`n" | ForEach-Object {
        if ($_ -match 'local/(\S+)') { $matches[1].TrimEnd('/') }
    } | Where-Object { $_ }
    $missB = @()
    foreach ($e in $expected) {
        if ($bList -notcontains $e) { $missB += $e }
    }
    if ($missB.Count -eq 0) { Write-Pass "All 8 expected buckets present" }
    else { Write-Warn "Missing buckets: $($missB -join ', ')"; $WARNED += "G-MissingBuckets" }
} else {
    Write-Fail "MinIO: cannot list buckets"; $FAILED += "G-MinIO"
}

Write-Sec "G2. Bucket Contents"
if ($buckets) {
    $checkBuckets = @("cadqstream-raw", "cadqstream-violations", "cadqstream-anomalies", "cadqstream-metrics", "cadqstream-drift", "cadqstream-dlq", "cadqstream-checkpoints", "ml-models")
    foreach ($b in $checkBuckets) {
        $contents = docker run --rm --network cadqstream-net $envArg minio/mc ls "local/$b/" 2>$null
        $fCount = if ($contents) { ($contents -split "`n" | Where-Object { $_ -ne "" }).Count } else { 0 }
        if ($fCount -gt 0) {
            Write-Info "Bucket $b : $fCount file(s)"
        } else {
            Write-Info "Bucket $b : empty (may be normal for cold bucket)"
        }
    }

    Write-Sec "G3. Sensitive Bucket Access"
    $sensitive = @("cadqstream-violations", "cadqstream-anomalies", "ml-models")
    $publicFound = $false
    foreach ($b in $sensitive) {
        $pubCheck = docker run --rm --network cadqstream-net $envArg minio/mc anonymous get "local/$b" 2>$null
        if ($pubCheck -match "Enabled") {
            Write-Fail "Bucket $b has PUBLIC ACCESS (security risk!)"
            $FAILED += "G-BucketPublic-$b"
            $publicFound = $true
        }
    }
    if (-not $publicFound) { Write-Pass "Sensitive buckets are private" }
}

# =============================================================================
# H. REDIS
# =============================================================================
Write-Hdr "H. Redis (L1b)"
$redisPwd = [System.Environment]::GetEnvironmentVariable("REDIS_PASSWORD", "Process")
if (-not $redisPwd) { $redisPwd = "redis_password_local" }
try {
    $pong = docker exec ldt-redis redis-cli -a $redisPwd ping 2>$null
    if ($pong -match "PONG") { Write-Pass "Redis: PONG" }
    else { Write-Warn "Redis response: $pong"; $WARNED += "H-Redis" }
} catch {
    Write-Warn "Redis unreachable"; $WARNED += "H-Redis"
}

# =============================================================================
# I. STATS WRITER
# =============================================================================
Write-Hdr "I. Stats Writer (L6b)"
$statsStatus = docker ps --filter "name=ldt-stats-writer" --format "{{.Status}}" 2>$null
if ($statsStatus -match "Up") {
    Write-Pass "stats-writer running"
} else {
    Write-Warn "stats-writer not running"; $WARNED += "I-StatsWriter"
}
$statsFiles = docker exec ldt-minio mc ls "local/cadqstream-metrics/" 2>$null
$statsFileCount = if ($statsFiles) { ($statsFiles -split "`n" | Where-Object { $_ -ne "" }).Count } else { 0 }
if ($statsFileCount -gt 0) { Write-Pass "cadqstream-metrics/ : $statsFileCount file(s)" }
else { Write-Warn "cadqstream-metrics/ : empty"; $WARNED += "I-StatsBucket" }

# =============================================================================
# J. ACTION REPLAY WORKER
# =============================================================================
Write-Hdr "J. Action Replay Worker (L4b)"
$arwStatus = docker ps --filter "name=ldt-action-replay-worker" --format "{{.Status}}" 2>$null
if ($arwStatus -match "Up") {
    Write-Pass "action-replay-worker running"
} else {
    Write-Warn "action-replay-worker not running"; $WARNED += "J-ARW"
}

# =============================================================================
# K. DATA INJECTION TESTS
# =============================================================================
if ($Inject) {
    Write-Hdr "K. Data Injection Tests"

    Write-Info "Injecting NORMAL record..."
    $normal = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T13:00:00","tpep_dropoff_datetime":"2026-05-17T13:15:00","passenger_count":2,"trip_distance":3.5,"PULocationID":79,"DOLocationID":170,"fare_amount":12.50,"total_amount":15.75,"payment_type":1}'
    $normal | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
    Start-Sleep -Seconds 3
    Write-Info "Normal record injected."

    Write-Info "Injecting L1 Schema Violation (missing trip_distance)..."
    $l1 = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T13:01:00","tpep_dropoff_datetime":"2026-05-17T13:16:00","passenger_count":1,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.50,"payment_type":1}'
    $l1 | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
    Write-Info "L1 violation injected."

    Write-Info "Injecting L2 Canary Violations (negative fare, zero distance, passengers=0)..."
    $negFare = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T13:02:00","tpep_dropoff_datetime":"2026-05-17T13:17:00","passenger_count":1,"trip_distance":5.0,"PULocationID":79,"DOLocationID":170,"fare_amount":-5.00,"total_amount":2.00,"payment_type":1}'
    $zeroDist = '{"VendorID":2,"tpep_pickup_datetime":"2026-05-17T13:02:10","tpep_dropoff_datetime":"2026-05-17T13:03:10","passenger_count":1,"trip_distance":0.0,"PULocationID":79,"DOLocationID":79,"fare_amount":25.00,"total_amount":28.00,"payment_type":2}'
    $zeroPass = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T13:02:20","tpep_dropoff_datetime":"2026-05-17T13:17:20","passenger_count":0,"trip_distance":3.0,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.00,"payment_type":1}'
    $negFare | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
    $zeroDist | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
    $zeroPass | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
    Write-Info "L2 violations injected."

    Write-Info "Injecting L3 Extreme Anomaly..."
    $l3 = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-17T13:03:00","tpep_dropoff_datetime":"2026-05-17T13:48:00","passenger_count":6,"trip_distance":99.9,"PULocationID":138,"DOLocationID":229,"fare_amount":999.99,"total_amount":1050.00,"payment_type":1}'
    $l3 | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
    Write-Info "L3 anomaly injected."

    Write-Info "Injecting Concept Drift (5 records with +20% fare increment)..."
    $baseFare = 15.0
    for ($i = 1; $i -le 5; $i++) {
        $fare = [math]::Round($baseFare * [math]::Pow(1.2, $i), 2)
        $driftMsg = "{`"VendorID`":1,`"tpep_pickup_datetime`":`"2026-05-17T13:04:0${i}`",`"tpep_dropoff_datetime`":`"2026-05-17T13:19:0${i}`",`"passenger_count`":2,`"trip_distance`":4.0,`"PULocationID`":100,`"DOLocationID`":180,`"fare_amount`":$fare,`"total_amount`":$($fare + 3),`"payment_type`":1}"
        $driftMsg | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 2>$null
        Start-Sleep -Milliseconds 500
    }
    Write-Info "Concept drift injection complete. Check Grafana dashboards for drift detection."

    Start-Sleep -Seconds 5
    Write-Info "Verifying iec-action-replay topic..."
    try {
        $iecMsgs = docker exec ldt-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic iec-action-replay --from-beginning --max-messages 3 --consumer-timeout-ms 3000 2>$null
        if ($iecMsgs) { Write-Pass "IEC action-replay has messages" }
        else { Write-Warn "IEC action-replay empty (drift may not be detected yet)" }
    } catch {}

    Write-Info "Checking cadqstream-violations/..."
    $vFiles = docker exec ldt-minio mc ls "local/cadqstream-violations/" 2>$null
    if ($vFiles) { Write-Pass "cadqstream-violations: files present" }
    else { Write-Warn "cadqstream-violations: still empty" }
}

# =============================================================================
# FINAL
# =============================================================================
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Verification Complete" -ForegroundColor Cyan
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

if ($FAILED.Count -gt 0) {
    Write-Host "CRITICAL FAILURES:" -ForegroundColor Red
    $FAILED | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host ""
    exit 1
}
if ($WARNED.Count -gt 0) {
    Write-Host "WARNINGS ($($WARNED.Count)):" -ForegroundColor Yellow
    $WARNED | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    Write-Host ""
    Write-Host "Stack is operational with warnings." -ForegroundColor Yellow
    Write-Host ""
    exit 2
}

Write-Host "ALL CHECKS PASSED" -ForegroundColor Green
Write-Host ""
exit 0
