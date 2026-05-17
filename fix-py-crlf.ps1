$pyFiles = Get-ChildItem -Path 'c:/proj/ldt/src' -Recurse -Include *.py
$converted = 0
foreach ($f in $pyFiles) {
  $bytes = [System.IO.File]::ReadAllBytes($f.FullName)
  $hasCrlf = $false
  for ($i = 0; $i -lt $bytes.Length - 1; $i++) {
    if ($bytes[$i] -eq 13 -and $bytes[$i+1] -eq 10) {
      $hasCrlf = $true
      break
    }
  }
  if ($hasCrlf) {
    $content = [System.IO.File]::ReadAllText($f.FullName) -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($f.FullName, $content)
    Write-Host "[LF] $($f.FullName)"
    $converted++
  }
}
Write-Host ""
Write-Host "Converted: $converted Python files to LF"
