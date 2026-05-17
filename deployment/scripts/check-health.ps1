# =============================================================================
# CA-DQStream Health Check (PowerShell)
# Verifies all 21 services are running and healthy.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File deployment/scripts/check-health.ps1
#
# Exit codes: 0=healthy, 1=critical failures, 2=warnings only
# =============================================================================

$ErrorActionPreference = "Continue"
$DEPLOYMENT_DIR = Split-Path -Parent $PSScriptRoot
if (-not $env:GRAFANA_PASSWORD) {
    Write-Host "ERROR: GRAFANA_PASSWORD environment variable not set" -ForegroundColor Red
    Write-Host "Set it via: \$env:GRAFANA_PASSWORD = 'your_password'"
    exit 1
}
$GRAFANA_PASSWORD = $env:GRAFANA_PASSWORD

function Write-OK($msg)   { Write-Host "  [OK]   $msg" -ForegroundColor Green }
function Write-DOWN($msg) { Write-Host "  [DOWN] $msg" -ForegroundColor Red }
function Write-WARN($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-INFO($msg) { Write-Host "  [INFO] $msg" -ForegroundColor Cyan }
function Write-SEC($msg)  { Write-Host ""; Write-Host "== $msg ==" -ForegroundColor Magenta }

$FAIL_COUNT = 0
$WARN_COUNT = 0

# =============================================================================
# HEADER
# =============================================================================
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  CA-DQStream Health Check" -ForegroundColor Cyan
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# =============================================================================
# A. CONTAINER STATUS
# =============================================================================
Write-SEC "A. Container Status"

$CRITICAL_SERVICES = @(
    "ldt-zookeeper",
    "ldt-kafka",
    "ldt-schema-registry",
    "ldt-redis",
    "ldt-minio",
    "ldt-flink-jobmanager",
    "ldt-flink-taskmanager",
    "ldt-prometheus",
    "ldt-grafana"
)

$ALL_SERVICES = @(
    "ldt-zookeeper",
    "ldt-kafka",
    "ldt-kafka-init",
    "ldt-schema-registry",
    "ldt-kafka-ui",
    "ldt-kafka-exporter",
    "ldt-redis",
    "ldt-minio",
    "ldt-minio-init",
    "ldt-flink-jobmanager",
    "ldt-flink-taskmanager",
    "ldt-flink-init",
    "ldt-ml-service",
    "ldt-action-replay-worker",
    "ldt-prometheus",
    "ldt-grafana",
    "ldt-node-exporter",
    "ldt-cadqstream-metrics",
    "ldt-stats-writer",
    "ldt-kafka-producer"
)

$containerStatus = @{}
docker ps --filter "name=ldt-" --format "{{.Names}}|{{.Status}}" 2>$null -split "`n" | ForEach-Object {
    if ($_ -match "^(.+)\|(.+)$") {
        $containerStatus[$matches[1]] = $matches[2].Trim()
    }
}

# Show critical services first
foreach ($svc in $CRITICAL_SERVICES) {
    $status = $containerStatus[$svc]
    if ($status) {
        if ($status -match "healthy|Up") {
            Write-OK "$svc : $status"
        } else {
            Write-DOWN "$svc : $status"
            $FAIL_COUNT++
        }
    } else {
        Write-DOWN "$svc : NOT RUNNING"
        $FAIL_COUNT++
    }
}

Write-INFO "Optional services:"
foreach ($svc in $ALL_SERVICES) {
    if ($svc -notin $CRITICAL_SERVICES) {
        $status = $containerStatus[$svc]
        if ($status) {
            if ($status -match "healthy|Up") {
                Write-OK "$svc : $status"
            } else {
                Write-WARN "$svc : $status"
                $WARN_COUNT++
            }
        } else {
            Write-WARN "$svc : NOT RUNNING"
            $WARN_COUNT++
        }
    }
}

# =============================================================================
# B. DOCKER NETWORK
# =============================================================================
Write-SEC "B. Docker Network"
$networks = docker network ls --filter "name=cadqstream" --format "{{.Name}}" 2>$null
if ($networks) {
    Write-OK "Network found: $networks"
    $netContainers = docker network inspect $networks.Split("`n")[0] --format '{{range .Containers}}{{.Name}} {{end}}' 2>$null
    Write-INFO "$($netContainers.Split(' ').Count) container(s) in network"
} else {
    Write-WARN "No cadqstream network found"
    $WARN_COUNT++
}

# =============================================================================
# C. KAFKA
# =============================================================================
Write-SEC "C. Kafka"

# Topic list
try {
    $topics = docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>$null
    $tCount = ($topics -split "`n" | Where-Object { $_ -ne "" }).Count
    Write-OK "Kafka reachable: $tCount topics"
} catch {
    Write-DOWN "Kafka unreachable"
    $FAIL_COUNT++
}

# Consumer group lag
Write-INFO "Consumer group lag:"
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
        if ($totalLag -eq 0) { Write-OK "Total consumer lag: 0" }
        elseif ($totalLag -lt 1000) { Write-INFO "Total consumer lag: $totalLag" }
        else { Write-WARN "Total consumer lag: $totalLag"; $WARN_COUNT++ }
    }
} catch {
    Write-INFO "Could not check consumer lag"
}

# Topic partition info
if ($topics) {
    $topicList = $topics -split "`n" | Where-Object { $_ -ne "" }
    Write-INFO "Topics:"
    $topicList | Select-Object -First 10 | ForEach-Object {
        $t = $_.Trim()
        if ($t) {
            $desc = docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --topic $t --describe 2>$null
            $parts = if ($desc -match "PartitionCount:\s*(\d+)") { $matches[1] } else { "?" }
            Write-INFO "  $t (partitions=$parts)"
        }
    }
    if ($topicList.Count -gt 10) {
        Write-INFO "  ... and $($topicList.Count - 10) more"
    }
}

# Schema Registry
try {
    $schemas = Invoke-WebRequest -Uri "http://localhost:8081/subjects" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    $sCount = ($schemas.Content | ConvertFrom-Json).Count
    if ($sCount -ge 3) { Write-OK "Schema Registry: $sCount schemas" }
    else { Write-WARN "Schema Registry: only $sCount schemas"; $WARN_COUNT++ }
} catch {
    Write-WARN "Schema Registry: not reachable"
    $WARN_COUNT++
}

# kafka-exporter
try {
    $kExp = Invoke-WebRequest -Uri "http://localhost:9308/metrics" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($kExp.StatusCode -eq 200) { Write-OK "Kafka exporter (:9308): responding" }
} catch {
    Write-WARN "Kafka exporter (:9308): not responding"
    $WARN_COUNT++
}

# kafka-ui
try {
    $kUi = Invoke-WebRequest -Uri "http://localhost:8080" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($kUi.StatusCode -eq 200) { Write-OK "Kafka UI (:8080): responding" }
} catch {
    Write-WARN "Kafka UI (:8080): not responding"
    $WARN_COUNT++
}

# =============================================================================
# D. FLINK
# =============================================================================
Write-SEC "D. Flink"

try {
    $overview = Invoke-WebRequest -Uri "http://localhost:8081/overview" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
    $tms = $overview.taskmanagers
    $slotsTotal = $overview.'taskmanager'.totalTaskManagerSlotNumber
    $slotsFree = $overview.'taskmanager'.totalAvailableSlotNumber
    Write-OK "Flink cluster: $tms TM(s), $slotsTotal slots ($slotsFree free)"
} catch {
    Write-DOWN "Flink REST API: unreachable"
    $FAIL_COUNT++
}

if ($null -ne $overview) {
    # Jobs
    try {
        $jobs = Invoke-WebRequest -Uri "http://localhost:8081/jobs" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
        $jobCount = $jobs.jobs.Count
        $running = @($jobs.jobs | Where-Object { $_.state -eq "RUNNING" })
        $failed = @($jobs.jobs | Where-Object { $_.state -eq "FAILED" })

        Write-INFO "Jobs: $jobCount total, $($running.Count) RUNNING, $($failed.Count) FAILED"

        if ($running.Count -gt 0) {
            foreach ($j in $running) {
                Write-OK "  $($j.name) [RUNNING] (parallelism=$($j.parallelism))"
            }
            # Checkpoint info
            foreach ($j in $running) {
                try {
                    $ji = Invoke-WebRequest -Uri "http://localhost:8081/jobs/$($j.id)/info" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
                    $chk = $ji.checkpointing
                    if ($chk) {
                        $lastTs = $chk.last_checkpont_timestamp
                        if ($lastTs -and [int64]$lastTs -gt 0) {
                            $chkDate = [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$lastTs).LocalDateTime.ToString("yyyy-MM-dd HH:mm:ss")
                            Write-INFO "  Checkpoint: last at $chkDate"
                        } else {
                            Write-WARN "  Checkpoint: enabled but no checkpoint yet"
                            $WARN_COUNT++
                        }
                    }
                } catch {}
            }
        }

        if ($failed.Count -gt 0) {
            foreach ($j in $failed) {
                Write-DOWN "  $($j.name) [FAILED]"
            }
            $FAIL_COUNT++
        }
    } catch {
        Write-WARN "Cannot query Flink jobs"
        $WARN_COUNT++
    }

    # TaskManagers
    try {
        $tms2 = Invoke-WebRequest -Uri "http://localhost:8081/taskmanagers" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
        $tmList = $tms2.taskmanagers
        Write-INFO "$($tmList.Count) TaskManager(s) registered:"
        foreach ($tm in $tmList) {
            $slotsUsed = $tm.slotsNumber
            $slotsTotal = $tm.slotsNumber
            $tmId = $tm.id.Substring(0, [Math]::Min(16, $tm.id.Length))
            Write-INFO "  $tmId... (slots=$slotsTotal)"
        }
    } catch {
        Write-WARN "Cannot query TaskManagers"
    }

    # flink-init auto-recovery
    $initLogs = docker logs ldt-flink-init --tail 10 2>$null
    if ($initLogs) {
        if ($initLogs -match "RUNNING|HEALTHY|CONTINUOUS|HEALTH MONITOR") {
            Write-OK "flink-init auto-recovery: active"
        } else {
            Write-WARN "flink-init auto-recovery: check logs"
            $initLogs -split "`n" | Select-Object -Last 5 | ForEach-Object { Write-INFO "  $_" }
        }
    }
}

# =============================================================================
# E. ML SERVICE
# =============================================================================
Write-SEC "E. ML Service"

try {
    $mlH = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 10 -ErrorAction SilentlyContinue
    if ($mlH.StatusCode -eq 200) {
        $mlHj = $mlH.Content | ConvertFrom-Json
        Write-OK "ML service: healthy"
        $mlHj.PSObject.Properties | ForEach-Object { Write-INFO "  $($_.Name): $($_.Value)" }
    } else {
        Write-DOWN "ML service: HTTP $($mlH.StatusCode)"
        $FAIL_COUNT++
    }
} catch {
    Write-DOWN "ML service: not reachable"
    $FAIL_COUNT++
}

# Predict endpoint
try {
    $feats = @(900.0, 3.5, 15.50, 2.50, 0.33, 0.95, 0.14, 0.0, 2.0, 100.0, 170.0, 5.0, 1.3, 0.16, 0.10, 0.05, 1.0, 1.0, 0.0, 1.0, 0.87, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.1, 0.9, 0.15, 0.85, 0.25, 0.75)
    $pay = @{features = @($feats)} | ConvertTo-Json -Compress
    $mlP = Invoke-WebRequest -Uri "http://localhost:8000/predict" -UseBasicParsing -Method POST -Body $pay -ContentType "application/json" -TimeoutSec 15 -ErrorAction SilentlyContinue
    if ($mlP.StatusCode -eq 200) {
        $mlPj = $mlP.Content | ConvertFrom-Json
        $score = $mlPj.anomaly_score
        if ($null -ne $score) { Write-OK "ML predict: anomaly_score = $score" }
        else { Write-WARN "ML predict: no anomaly_score" }
    } else {
        Write-WARN "ML predict: HTTP $($mlP.StatusCode)"
    }
} catch {
    Write-WARN "ML predict: not responding"
}

# ML logs
$mlLogs = docker logs ldt-ml-service --tail 10 2>$null
if ($mlLogs) {
    if ($mlLogs -match "ERROR|Exception") {
        Write-WARN "ML service logs contain errors:"
        $mlLogs -split "`n" | Where-Object { $_ -match "ERROR|Exception" } | Select-Object -First 3 | ForEach-Object { Write-WARN "  $_" }
    }
}

# =============================================================================
# F. PROMETHEUS
# =============================================================================
Write-SEC "F. Prometheus"

try {
    $promH = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($promH.StatusCode -eq 200) { Write-OK "Prometheus: healthy" }
} catch {
    Write-DOWN "Prometheus: not reachable"
    $FAIL_COUNT++
}

# Targets
try {
    $targets = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/targets" -UseBasicParsing -TimeoutSec 10 -ErrorAction SilentlyContinue | ConvertFrom-Json
    $unhealthy = @($targets.data.targets | Where-Object { $_.health -ne "up" })
    $upTargets = @($targets.data.targets | Where-Object { $_.health -eq "up" })
    Write-INFO "Scrape targets: $($upTargets.Count) up, $($unhealthy.Count) down"
    foreach ($t in $upTargets) {
        Write-OK "  $($t.labels.job)"
    }
    foreach ($t in $unhealthy) {
        Write-DOWN "  $($t.labels.job) : $($t.lastError)"
        $FAIL_COUNT++
    }
} catch {
    Write-WARN "Cannot query scrape targets"
}

# =============================================================================
# G. GRAFANA
# =============================================================================
Write-SEC "G. Grafana"

try {
    $gH = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($gH.StatusCode -eq 200) {
        $gver = ($gH.Content | ConvertFrom-Json).version
        Write-OK "Grafana v$gver: healthy"
    }
} catch {
    Write-DOWN "Grafana: not reachable"
    $FAIL_COUNT++
}

# Dashboards
try {
    $creds = New-Object PSCredential("admin", (ConvertTo-SecureString $GRAFANA_PASSWORD -AsPlainText -Force))
    $dashes = Invoke-WebRequest -Uri "http://localhost:3000/api/search?type=dash-db" -UseBasicParsing -TimeoutSec 10 -Credential $creds -ErrorAction SilentlyContinue
    $dList = $dashes.Content | ConvertFrom-Json
    $dCount = $dList.Count
    if ($dCount -ge 6) { Write-OK "Dashboards: $dCount (>= 6)" }
    else { Write-WARN "Dashboards: $dCount (expected >= 6)"; $WARN_COUNT++ }

    $dList | ForEach-Object { Write-INFO "  $($_.title)" }
} catch {
    Write-WARN "Cannot retrieve Grafana dashboards"
}

# =============================================================================
# H. MINIO
# =============================================================================
Write-SEC "H. MinIO"

try {
    $mcReady = docker exec ldt-minio mc ready local 2>$null
    if ($LASTEXITCODE -eq 0) { Write-OK "MinIO: mc ready" }
    else { Write-DOWN "MinIO: mc ready failed"; $FAIL_COUNT++ }
} catch {
    Write-DOWN "MinIO: not reachable"
    $FAIL_COUNT++
}

# Buckets
$buckets = docker exec ldt-minio mc ls local/ 2>$null
if ($buckets) {
    $bList = $buckets -split "`n" | Where-Object { $_ -ne "" }
    $bCount = $bList.Count
    Write-INFO "MinIO buckets: $bCount"
    $bList | ForEach-Object {
        if ($_ -match 'local/(\S+)') {
            $bn = $matches[1].TrimEnd('/')
            $contents = docker exec ldt-minio mc ls "local/$bn/" 2>$null
            $fCount = if ($contents) { ($contents -split "`n" | Where-Object { $_ -ne "" }).Count } else { 0 }
            $bName = $bn.PadRight(30)
            if ($fCount -gt 0) { Write-INFO "  $bName : $fCount file(s)" }
            else { Write-INFO "  $bName : empty" }
        }
    }
} else {
    Write-WARN "Cannot list MinIO buckets"
}

# Sensitive bucket access
$sensitive = @("cadqstream-violations", "cadqstream-anomalies", "ml-models")
foreach ($b in $sensitive) {
    $pubCheck = docker exec ldt-minio mc anonymous get "local/$b" 2>$null
    if ($pubCheck -match "Enabled") {
        Write-DOWN "Bucket $b has PUBLIC ACCESS (security risk!)"
        $FAIL_COUNT++
    }
}

# =============================================================================
# I. REDIS
# =============================================================================
Write-SEC "I. Redis"
$redisPwd = [System.Environment]::GetEnvironmentVariable("REDIS_PASSWORD", "Process")
if (-not $redisPwd) { $redisPwd = "redis_password_local" }
try {
    $pong = docker exec ldt-redis redis-cli -a $redisPwd ping 2>$null
    if ($pong -match "PONG") {
        Write-OK "Redis: PONG"
        $clients = docker exec ldt-redis redis-cli -a $redisPwd info clients 2>$null | Select-String "connected_clients"
        $ver = docker exec ldt-redis redis-cli -a $redisPwd info server 2>$null | Select-String "redis_version"
        Write-INFO "  $($ver.ToString().Trim()) | $($clients.ToString().Trim())"
    } else {
        Write-DOWN "Redis: unexpected response: $pong"
        $FAIL_COUNT++
    }
} catch {
    Write-DOWN "Redis: not reachable"
    $FAIL_COUNT++
}

# =============================================================================
# J. CADQSTREAM METRICS
# =============================================================================
Write-SEC "J. cadqstream-metrics"

try {
    $cmH = Invoke-WebRequest -Uri "http://localhost:9250/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($cmH.StatusCode -eq 200) {
        Write-OK "cadqstream-metrics: healthy"
    }
} catch {
    Write-WARN "cadqstream-metrics: not reachable"
    $WARN_COUNT++
}

try {
    $cmM = Invoke-WebRequest -Uri "http://localhost:9250/metrics" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    if ($cmM.StatusCode -eq 200) {
        $metricCount = ($cmM.Content -split "`n" | Where-Object { $_ -match "^cadqstream" }).Count
        Write-INFO "cadqstream_metrics endpoint: $metricCount cadqstream_* metrics"
    }
} catch {
    Write-WARN "cadqstream-metrics /metrics: not responding"
}

# =============================================================================
# K. ACTION REPLAY WORKER & STATS WRITER
# =============================================================================
Write-SEC "K. Action Replay & Stats Writer"

$arwStatus = docker ps --filter "name=ldt-action-replay-worker" --format "{{.Status}}" 2>$null
if ($arwStatus -match "Up") {
    Write-OK "action-replay-worker: running"
} else {
    Write-WARN "action-replay-worker: not running"
    $WARN_COUNT++
}

$swStatus = docker ps --filter "name=ldt-stats-writer" --format "{{.Status}}" 2>$null
if ($swStatus -match "Up") {
    Write-OK "stats-writer: running"
} else {
    Write-WARN "stats-writer: not running"
    $WARN_COUNT++
}

# =============================================================================
# L. KAFKA PRODUCER
# =============================================================================
Write-SEC "L. Kafka Producer"

$prodStatus = docker ps --filter "name=ldt-kafka-producer" --format "{{.Status}}" 2>$null
if ($prodStatus -match "Up") {
    Write-OK "kafka-producer: running"
} else {
    Write-WARN "kafka-producer: not running"
    $WARN_COUNT++
}

# =============================================================================
# SUMMARY
# =============================================================================
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Summary" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$running = (docker ps --filter "name=ldt-" --format "{{.Names}}" 2>$null).Count
Write-Host "  Containers running: $running / $($ALL_SERVICES.Count)"
Write-Host "  Critical failures: $FAIL_COUNT"
Write-Host "  Warnings:          $WARN_COUNT"
Write-Host ""

if ($FAIL_COUNT -gt 0) {
    Write-Host "  Status: CRITICAL - $FAIL_COUNT failure(s)" -ForegroundColor Red
    Write-Host ""
    Write-Host "  To investigate: docker compose -f deployment/docker-compose.yml logs <service>" -ForegroundColor Yellow
    Write-Host "  To restart:     docker compose -f deployment/docker-compose.yml restart <service>" -ForegroundColor Yellow
    exit 1
}
if ($WARN_COUNT -gt 0) {
    Write-Host "  Status: WARNINGS - $WARN_COUNT warning(s), no critical failures" -ForegroundColor Yellow
    Write-Host "  Stack is operational." -ForegroundColor Green
    exit 2
}

Write-Host "  Status: HEALTHY - all services operational" -ForegroundColor Green
Write-Host ""
exit 0
