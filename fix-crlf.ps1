$files = @(
  'c:/proj/ldt/deployment/kafka/init-scripts/init.sh',
  'c:/proj/ldt/deployment/kafka/init-scripts/01-create-topics.sh',
  'c:/proj/ldt/deployment/kafka/init-scripts/02-fix-compact-topics.sh',
  'c:/proj/ldt/deployment/minio/init-scripts/01-create-buckets.sh',
  'c:/proj/ldt/deployment/flink/flink-init.sh',
  'c:/proj/ldt/deployment/scripts/start.sh',
  'c:/proj/ldt/deployment/scripts/verify-flow.sh',
  'c:/proj/ldt/deployment/scripts/healthcheck.sh',
  'c:/proj/ldt/deployment/scripts/bootstrap.sh',
  'c:/proj/ldt/deployment/scripts/stop.sh',
  'c:/proj/ldt/deployment/scripts/reset.sh',
  'c:/proj/ldt/deployment/scripts/init-all.sh',
  'c:/proj/ldt/deployment/scripts/wait-for.sh',
  'c:/proj/ldt/deployment/scripts/migrate_to_memstream.sh',
  'c:/proj/ldt/deployment/scripts/quick_retrain_cron.sh',
  'c:/proj/ldt/deployment/kafka/continuous_producer.sh',
  'c:/proj/ldt/deployment/kafka/generate-certs.sh',
  'c:/proj/ldt/deployment/kafka/certs/gen.sh',
  'c:/proj/ldt/deployment/kafka/certs/run_gen.sh',
  'c:/proj/ldt/deployment/redis/generate-certs.sh',
  'c:/proj/ldt/deployment/redis/certs/run_redis.sh',
  'c:/proj/ldt/deployment/inject_anomalies.sh',
  'c:/proj/ldt/deployment/post_silence.sh',
  'c:/proj/ldt/deployment/check_flink.sh'
)
$converted = 0
$skipped = 0
foreach ($f in $files) {
  if (Test-Path $f) {
    $bytes = [System.IO.File]::ReadAllBytes($f)
    $hasCrlf = $false
    for ($i = 0; $i -lt $bytes.Length - 1; $i++) {
      if ($bytes[$i] -eq 13 -and $bytes[$i+1] -eq 10) {
        $hasCrlf = $true
        break
      }
    }
    if ($hasCrlf) {
      $content = [System.IO.File]::ReadAllText($f) -replace "`r`n", "`n"
      [System.IO.File]::WriteAllText($f, $content)
      Write-Host "[LF] $f"
      $converted++
    } else {
      Write-Host "[OK]  $f (already LF)"
      $skipped++
    }
  } else {
    Write-Host "[SKIP] $f not found"
  }
}
Write-Host ""
Write-Host "Converted: $converted, Already LF: $skipped"
