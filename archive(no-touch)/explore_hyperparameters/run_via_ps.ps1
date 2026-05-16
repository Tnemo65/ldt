$ErrorActionPreference = "Continue"
$output = & python "C:\proj\ldt\explore_hyperparameters\run_configs_inline.py" 2>&1
$output | Out-File -FilePath "C:\proj\ldt\explore_hyperparameters\results\ps_output.txt" -Encoding utf8
Write-Host "Exit code: $LASTEXITCODE"
Write-Host "Output length: $($output.Length)"
Write-Host "Output:"
$output
