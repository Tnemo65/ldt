$ErrorActionPreference = "SilentlyContinue"
$output = python "C:\proj\ldt\explore_hyperparameters\run_config_c1.py" 2>&1 | Out-String
Write-Host "Output length: $($output.Length)"
Write-Host $output
