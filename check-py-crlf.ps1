$pyFiles = Get-ChildItem -Path 'c:/proj/ldt/src' -Recurse -Include *.py
$crlf = @()
foreach ($f in $pyFiles) {
  $bytes = [System.IO.File]::ReadAllBytes($f.FullName)
  for ($i = 0; $i -lt $bytes.Length - 1; $i++) {
    if ($bytes[$i] -eq 13 -and $bytes[$i+1] -eq 10) {
      $crlf += $f.FullName
      break
    }
  }
}
Write-Host "Python files with CRLF: $($crlf.Count)"
$crlf | ForEach-Object { Write-Host "  $_" }
