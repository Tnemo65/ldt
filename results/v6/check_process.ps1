$p = Get-Process python -ErrorAction SilentlyContinue
if ($p) {
    Write-Host "Python process: PID=$($p.Id), CPU=$($p.CPU)s, RAM=$([math]::Round($p.WorkingSet64/1MB,1))MB"
} else {
    Write-Host "No Python process found"
}
