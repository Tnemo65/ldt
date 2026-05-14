$p = Get-Process -Id 17956 -ErrorAction SilentlyContinue
if ($p) {
    $m = [math]::Round($p.WorkingSet64/1MB, 0)
    Write-Host "PID=$($p.Id) Responding=$($p.Responding) CPU=$($p.CPU)s RAM=${m}MB"
} else {
    Write-Host "Process 17956 not found"
}
