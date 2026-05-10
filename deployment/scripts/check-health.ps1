# CA-DQStream Health Check Script (PowerShell)

$services = @(
    @{Name="ldt-zookeeper"; Url=""; Port=2181; Protocol="tcp"; Desc="ZooKeeper"},
    @{Name="ldt-kafka"; Url=""; Port=9092; Protocol="tcp"; Desc="Kafka"},
    @{Name="ldt-postgres"; Url=""; Port=5432; Protocol="tcp"; Desc="PostgreSQL"},
    @{Name="ldt-minio"; Url="http://localhost:9000/minio/health/live"; Port=9000; Protocol="http"; Desc="MinIO"},
    @{Name="ldt-mlflow"; Url="http://localhost:5000"; Port=5000; Protocol="http"; Desc="MLflow"},
    @{Name="ldt-schema-registry"; Url=""; Port=8082; Protocol="tcp"; Desc="Schema Registry"},
    @{Name="ldt-flink-jobmanager"; Url="http://localhost:8081/overview"; Port=8081; Protocol="http"; Desc="Flink UI"},
    @{Name="ldt-prometheus"; Url="http://localhost:9090/-/healthy"; Port=9090; Protocol="http"; Desc="Prometheus"},
    @{Name="ldt-grafana"; Url="http://localhost:3000/api/health"; Port=3000; Protocol="http"; Desc="Grafana"},
    @{Name="ldt-kafka-ui"; Url="http://localhost:8080"; Port=8080; Protocol="http"; Desc="Kafka UI"}
)

Write-Host ""
Write-Host "================================================================"
Write-Host "  CA-DQStream Health Check"
Write-Host "================================================================"
Write-Host ""

$allHealthy = $true

foreach ($svc in $services) {
    $status = docker ps --filter "name=$($svc.Name)" --format "{{.Status}}" 2>$null

    if (-not $status) {
        Write-Host "  [DOWN]   $($svc.Name) - not running" -ForegroundColor Red
        $allHealthy = $false
        continue
    }

    $healthy = $status -match "healthy|Up"
    if ($healthy) {
        Write-Host "  [OK]     $($svc.Name) - $status" -ForegroundColor Green
    } else {
        Write-Host "  [WARN]   $($svc.Name) - $status" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "[INFO] Kafka topics:"
$topics = docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>$null
if ($topics) {
    $topics | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "  (unable to list)" -ForegroundColor Red
}

Write-Host ""
Write-Host "[INFO] PostgreSQL tables:"
$tables = docker exec ldt-postgres psql -U cadqstream -d dq_pipeline -c "\dt" 2>$null
if ($tables) {
    $tables | Select-Object -Last 10 | ForEach-Object { Write-Host "  $_" }
}

Write-Host ""
Write-Host "[INFO] MinIO buckets:"
$buckets = docker exec ldt-minio mc ls local/ 2>$null
if ($buckets) {
    $buckets | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "  (none or access denied)" -ForegroundColor Red
}

Write-Host ""
Write-Host "[INFO] Flink jobs:"
try {
    $jobs = Invoke-WebRequest -Uri "http://localhost:8081/jobs" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
    if ($jobs.jobs.Count -gt 0) {
        foreach ($job in $jobs.jobs) {
            Write-Host "  $($job.id.Substring(0,16))...  State: $($job.state)  Name: $($job.name)" -ForegroundColor Green
        }
    } else {
        Write-Host "  No jobs running" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  Flink REST API not accessible" -ForegroundColor Red
}

Write-Host ""
Write-Host "[INFO] Kafka consumer lag (via kafka-exporter):"
try {
    $metrics = Invoke-WebRequest -Uri "http://localhost:9308/metrics" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
    $lag = ($metrics.Content -split "`n" | Where-Object { $_ -match "kafka_consumer_group_lag" -and $_ -notmatch "^#" } | Select-Object -First 3) -join "`n"
    if ($lag) {
        $lag | ForEach-Object { Write-Host "  $_" }
    } else {
        Write-Host "  No consumer groups active yet"
    }
} catch {
    Write-Host "  kafka-exporter not accessible"
}

Write-Host ""
if ($allHealthy) {
    Write-Host "All services healthy!" -ForegroundColor Green
} else {
    Write-Host "Some services are down or unhealthy." -ForegroundColor Yellow
}
Write-Host ""
