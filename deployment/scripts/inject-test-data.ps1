# =============================================================================
# CA-DQStream - Test Data Injection Script (PowerShell)
# Injects test records for each failure scenario and verifies propagation.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File deployment/scripts/inject-test-data.ps1
#   powershell -ExecutionPolicy Bypass -File deployment/scripts/inject-test-data.ps1 -Scenario All -Count 5
#
# Scenarios:
#   Normal       Clean valid record (baseline)
#   L1           Schema violation (missing trip_distance)
#   L2           Canary rule violations (negative fare, zero distance, etc.)
#   L3           Extreme anomaly (ML-triggering values)
#   Drift        Gradual fare increase (ADWIN concept drift)
#   All          Run all scenarios sequentially
# =============================================================================

param(
    [ValidateSet("Normal", "L1", "L2", "L3", "Drift", "All")]
    [string]$Scenario = "All",

    [ValidateRange(1, 100)]
    [int]$Count = 3,

    [ValidateRange(0, 60)]
    [int]$DelayMs = 500,

    [switch]$Verify,
    [switch]$Watch
)

$ErrorActionPreference = "Continue"
$TOPIC = "taxi-nyc-raw-v2"
$BASE_TIME = "2026-05-17T14:00:00"
$GRAFANA_PASSWORD = if ($env:GRAFANA_PASSWORD) { $env:GRAFANA_PASSWORD } else { "grafana_local_admin" }

function Write-Pass($msg) { Write-Host "  [PASS] $msg" -ForegroundColor Green }
function Write-Fail($msg) { Write-Host "  [FAIL] $msg" -ForegroundColor Red }
function Write-Warn($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Info($msg) { Write-Host "  [INFO] $msg" -ForegroundColor Cyan }
function Write-Hdr($msg) { Write-Host ""; Write-Host "=== $msg ===" -ForegroundColor Magenta }

# ── Helpers ──────────────────────────────────────────────────────────────────

function Inject($msg, $desc) {
    $msg | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic $TOPIC 2>$null
    Write-Info "$desc injected"
    if ($DelayMs -gt 0) { Start-Sleep -Milliseconds $DelayMs }
}

function VerifyBucket($bucket, $desc) {
    $files = docker exec ldt-minio mc ls "local/$bucket/" 2>$null
    $fCount = if ($files) { ($files -split "`n" | Where-Object { $_ -ne "" }).Count } else { 0 }
    if ($fCount -gt 0) { Write-Pass "$desc: $fCount file(s) in $bucket" }
    else { Write-Warn "$desc: $bucket still empty" }
    return $fCount
}

function VerifyMetric($metric, $desc) {
    try {
        $q = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=$metric" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | ConvertFrom-Json
        if ($q.data.result.Count -gt 0) {
            Write-Pass "$desc ($metric): data present"
            return $true
        } else {
            Write-Warn "$desc ($metric): no data yet"
            return $false
        }
    } catch {
        Write-Warn "$desc ($metric): query failed"
        return $false
    }
}

# ── Scenarios ────────────────────────────────────────────────────────────────

function Scenario-Normal {
    param([int]$n = 1)
    Write-Hdr "Scenario: NORMAL ($n records)"
    for ($i = 1; $i -le $n; $i++) {
        $ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
        $msg = @"
{"VendorID":1,"tpep_pickup_datetime":"$ts","tpep_dropoff_datetime":"$((Get-Date).AddMinutes(15).ToString("yyyy-MM-ddTHH:mm:ss"))","passenger_count":2,"trip_distance":3.5,"PULocationID":79,"DOLocationID":170,"fare_amount":12.50,"total_amount":15.75,"payment_type":1}
"@
        Inject $msg "Normal record #$i"
    }
    Write-Info "Verify: record should appear in cadqstream-raw (valid trips)"
    Write-Info "Expected: NO violations triggered"
}

function Scenario-L1 {
    param([int]$n = 1)
    Write-Hdr "Scenario: L1 SCHEMA VIOLATION ($n records)"
    Write-Info "L1a: Missing trip_distance field"

    # Case 1: missing trip_distance
    $ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
    Inject @"
{"VendorID":1,"tpep_pickup_datetime":"$ts","tpep_dropoff_datetime":"$((Get-Date).AddMinutes(15).ToString("yyyy-MM-ddTHH:mm:ss"))","passenger_count":1,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.50,"payment_type":1}
"@ "L1a: missing trip_distance"

    # Case 2: invalid PULocationID (not in NYC TLC zones)
    $ts2 = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
    Inject @"
{"VendorID":1,"tpep_pickup_datetime":"$ts2","tpep_dropoff_datetime":"$((Get-Date).AddMinutes(15).ToString("yyyy-MM-ddTHH:mm:ss"))","passenger_count":1,"trip_distance":5.0,"PULocationID":999,"DOLocationID":500,"fare_amount":18.00,"total_amount":22.00,"payment_type":1}
"@ "L1b: invalid PULocationID=999"

    Write-Info "Expected: records in cadqstream-violations/schema/"
}

function Scenario-L2 {
    param([int]$n = 1)
    Write-Hdr "Scenario: L2 CANARY RULE VIOLATIONS"
    Write-Info "Triggering 7 canary rules..."

    $ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
    Inject @"
{"VendorID":1,"tpep_pickup_datetime":"$ts","tpep_dropoff_datetime":"$((Get-Date).AddMinutes(15).ToString("yyyy-MM-ddTHH:mm:ss"))","passenger_count":1,"trip_distance":5.0,"PULocationID":79,"DOLocationID":170,"fare_amount":-5.00,"total_amount":2.00,"payment_type":1}
"@ "L2a: negative fare (-5.00)"

    Inject @"
{"VendorID":2,"tpep_pickup_datetime":"$ts","tpep_dropoff_datetime":"$((Get-Date).AddMinutes(1).ToString("yyyy-MM-ddTHH:mm:ss"))","passenger_count":1,"trip_distance":0.0,"PULocationID":79,"DOLocationID":79,"fare_amount":25.00,"total_amount":28.00,"payment_type":2}
"@ "L2b: zero distance with fare (speed anomaly)"

    Inject @"
{"VendorID":1,"tpep_pickup_datetime":"$ts","tpep_dropoff_datetime":"$((Get-Date).AddMinutes(15).ToString("yyyy-MM-ddTHH:mm:ss"))","passenger_count":0,"trip_distance":3.0,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.00,"payment_type":1}
"@ "L2c: passenger_count=0"

    Inject @"
{"VendorID":1,"tpep_pickup_datetime":"$ts","tpep_dropoff_datetime":"$((Get-Date).AddMinutes(1).ToString("yyyy-MM-ddTHH:mm:ss"))","passenger_count":2,"trip_distance":3.0,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.00,"payment_type":1}
"@ "L2d: trip duration unrealistic (<2 min)"

    Inject @"
{"VendorID":1,"tpep_pickup_datetime":"$ts","tpep_dropoff_datetime":"$((Get-Date).AddMinutes(15).ToString("yyyy-MM-ddTHH:mm:ss"))","passenger_count":4,"trip_distance":50.0,"PULocationID":138,"DOLocationID":229,"fare_amount":650.00,"total_amount":700.00,"payment_type":1}
"@ "L2e: fare > $500"

    Inject @"
{"VendorID":1,"tpep_pickup_datetime":"$ts","tpep_dropoff_datetime":"$((Get-Date).AddMinutes(15).ToString("yyyy-MM-ddTHH:mm:ss"))","passenger_count":1,"trip_distance":3.0,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":-5.00,"payment_type":1}
"@ "L2f: negative total_amount"

    Inject @"
{"VendorID":1,"tpep_pickup_datetime":"$ts","tpep_dropoff_datetime":"$((Get-Date).AddMinutes(15).ToString("yyyy-MM-ddTHH:mm:ss"))","passenger_count":1,"trip_distance":30.0,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.00,"payment_type":1}
"@ "L2g: speed > 100mph (30 miles in 15 min)"

    Write-Info "Expected: records in cadqstream-violations/canary/"
}

function Scenario-L3 {
    param([int]$n = 1)
    Write-Hdr "Scenario: L3 EXTREME ANOMALY (ML-triggering)"
    Write-Info "Values are technically valid but trigger MemStream anomaly detection"

    $ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
    Inject @"
{"VendorID":1,"tpep_pickup_datetime":"$ts","tpep_dropoff_datetime":"$((Get-Date).AddMinutes(45).ToString("yyyy-MM-ddTHH:mm:ss"))","passenger_count":6,"trip_distance":99.9,"PULocationID":138,"DOLocationID":229,"fare_amount":999.99,"total_amount":1050.00,"payment_type":1}
"@ "L3a: extreme fare spike ($999.99, 99.9 miles)"

    Inject @"
{"VendorID":1,"tpep_pickup_datetime":"$ts","tpep_dropoff_datetime":"$((Get-Date).AddMinutes(15).ToString("yyyy-MM-ddTHH:mm:ss"))","passenger_count":10,"trip_distance":0.1,"PULocationID":79,"DOLocationID":170,"fare_amount":200.00,"total_amount":210.00,"payment_type":1}
"@ "L3b: high fare for very short distance"

    Write-Info "Expected: anomaly_score > threshold in cadqstream-anomalies/scores/"
    Write-Info "Check MinIO: cadqstream-anomalies/scores/ for high anomaly scores"
}

function Scenario-Drift {
    param([int]$n = 5)
    Write-Hdr "Scenario: CONCEPT DRIFT (gradual fare increase)"
    Write-Info "Injecting $n records with +20% fare increment each"
    Write-Info "ADWIN should detect distribution shift in ~3-5 minutes"

    $baseFare = 15.0
    for ($i = 1; $i -le $n; $i++) {
        $fare = [math]::Round($baseFare * [math]::Pow(1.2, $i), 2)
        $total = [math]::Round($fare + 3, 2)
        $ts = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
        Inject @"
{"VendorID":1,"tpep_pickup_datetime":"$ts","tpep_dropoff_datetime":"$((Get-Date).AddMinutes(15).ToString("yyyy-MM-ddTHH:mm:ss"))","passenger_count":2,"trip_distance":4.0,"PULocationID":100,"DOLocationID":180,"fare_amount":$fare,"total_amount":$total,"payment_type":1}
"@ "Drift #$i: fare=\$$fare (+20% from previous)"
    }

    Write-Info "Wait 3-5 minutes for ADWIN to accumulate and detect drift."
    Write-Info "Expected: cadqstream_drift_detected metric = 1 in Prometheus"
    Write-Info "Expected: drift events in cadqstream-drift/drift_events/"
    Write-Info "Expected: alerts in cadqstream-drift/alerts/"
    Write-Info "Expected: iec-action-replay Kafka message for quick_retrain signal"
}

# ── Verification ─────────────────────────────────────────────────────────────

function Run-Verification {
    Write-Hdr "Verification"
    Write-Info "Checking buckets and metrics after injection..."

    Write-Sec "MinIO Buckets"
    $hasViol = VerifyBucket "cadqstream-violations" "L1/L2 Violations"
    $hasAnom = VerifyBucket "cadqstream-anomalies" "L3 Anomalies"
    $hasRaw  = VerifyBucket "cadqstream-raw" "Valid trips"

    Write-Sec "Prometheus Metrics"
    $hasValidMetric = VerifyMetric "cadqstream_records_valid_total" "L1 Valid records"
    $hasViolMetric  = VerifyMetric "cadqstream_records_violation_total" "L1/L2 Violations"
    $hasCanaryMetric = VerifyMetric "cadqstream_canary_violation_total" "L2 Canary violations"
    $hasAnomMetric  = VerifyMetric "cadqstream_anomaly_score" "L3 Anomalies"
    $hasDriftMetric = VerifyMetric "cadqstream_drift_detected" "Concept Drift (ADWIN)"

    Write-Sec "Kafka Topics"
    try {
        $dqTopic = docker exec ldt-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic dq-stream-unified --from-beginning --max-messages 1 --consumer-timeout-ms 3000 2>$null
        if ($dqTopic) { Write-Pass "dq-stream-unified has data" }
        else { Write-Warn "dq-stream-unified empty (pipeline warmup needed)" }
    } catch { Write-Warn "Cannot read dq-stream-unified topic" }

    try {
        $iecTopic = docker exec ldt-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic iec-action-replay --from-beginning --max-messages 1 --consumer-timeout-ms 3000 2>$null
        if ($iecTopic) { Write-Pass "iec-action-replay has data" }
        else { Write-Info "iec-action-replay empty (drift detection pending)" }
    } catch { Write-Info "Cannot read iec-action-replay topic" }
}

# ── Watch Mode ───────────────────────────────────────────────────────────────

function Watch-Flow {
    Write-Hdr "Watch Mode: Monitoring flow for 60 seconds..."
    Write-Info "Press Ctrl+C to stop."
    Write-Info ""

    $sw = [Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt 60) {
        $ts = Get-Date -Format "HH:mm:ss"
        Write-Host -NoNewline "[$ts] " -ForegroundColor DarkGray

        # Check violations bucket
        $vCount = (docker exec ldt-minio mc ls "local/cadqstream-violations/" 2>$null -split "`n" | Where-Object { $_ -ne "" }).Count
        Write-Host -NoNewline "violations=$vCount " -ForegroundColor $(if ($vCount -gt 0) { "Green" } else { "DarkGray" })

        # Check anomalies bucket
        $aCount = (docker exec ldt-minio mc ls "local/cadqstream-anomalies/" 2>$null -split "`n" | Where-Object { $_ -ne "" }).Count
        Write-Host -NoNewline "anomalies=$aCount " -ForegroundColor $(if ($aCount -gt 0) { "Green" } else { "DarkGray" })

        # Check Kafka topic lag
        try {
            $lag = docker exec ldt-kafka kafka-consumer-groups --bootstrap-server localhost:9092 --all-groups --describe 2>$null
            $totalLag = 0
            $lag -split "`n" | ForEach-Object {
                if ($_ -match '^\S+\s+\S+\s+\S+\s+\S+\s+(\d+|-)\s+(\d+|-)') {
                    $end = if ($matches[2] -eq "-") { 0 } else { [int]$matches[2] }
                    $totalLag += $end
                }
            }
            Write-Host -NoNewline "kafka_lag=$totalLag " -ForegroundColor $(if ($totalLag -lt 100) { "Green" } elseif ($totalLag -lt 1000) { "Yellow" } else { "Red" })
        } catch {}

        # Check metrics
        try {
            $q = Invoke-WebRequest -Uri "http://localhost:9090/api/v1/query?query=cadqstream_records_valid_total" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue | ConvertFrom-Json
            if ($q.data.result.Count -gt 0) {
                $val = $q.data.result[0].value[1]
                Write-Host -NoNewline "valid_records=$val " -ForegroundColor Green
            }
        } catch {}

        Write-Host ""
        Start-Sleep -Seconds 5
    }
    Write-Host "Watch complete."
}

# ── Main ─────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  CA-DQStream - Test Data Injection" -ForegroundColor Cyan
Write-Host "  Scenario: $Scenario | Count: $Count | Delay: ${DelayMs}ms" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Pre-flight check
try {
    docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>$null | Out-Null
} catch {
    Write-Fail "Kafka unreachable. Is the stack deployed?"
    exit 1
}

switch ($Scenario) {
    "Normal" { Scenario-Normal -n $Count }
    "L1"     { Scenario-L1 -n $Count }
    "L2"     { Scenario-L2 -n $Count }
    "L3"     { Scenario-L3 -n $Count }
    "Drift"  { Scenario-Drift -n $Count }
    "All"    {
        Write-Info "Running ALL scenarios sequentially..."
        Scenario-Normal -n 2
        Scenario-L1
        Scenario-L2
        Scenario-L3
        Scenario-Drift -n 5
    }
}

Write-Host ""
Write-Info "Injection complete."
Write-Info "Waiting 5 seconds for pipeline to process..."
Start-Sleep -Seconds 5

if ($Verify) { Run-Verification }

if ($Watch) { Watch-Flow }

if (-not $Verify -and -not $Watch) {
    Write-Host ""
    Write-Host "To verify, run with -Verify:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File deployment/scripts/inject-test-data.ps1 -Scenario All -Verify"
    Write-Host ""
    Write-Host "To watch flow live, run with -Watch:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File deployment/scripts/inject-test-data.ps1 -Watch"
    Write-Host ""
    Write-Host "To inject a specific scenario:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File deployment/scripts/inject-test-data.ps1 -Scenario L3"
    Write-Host "  powershell -ExecutionPolicy Bypass -File deployment/scripts/inject-test-data.ps1 -Scenario Drift -Count 10"
}

Write-Host ""
