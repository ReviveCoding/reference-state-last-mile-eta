$ErrorActionPreference = "Stop"

$Repo = $PSScriptRoot
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$LaDe = $env:LADE_DATA_ROOT

if ([string]::IsNullOrWhiteSpace($LaDe)) {
    throw "LADE_DATA_ROOT is required. Set `$env:LADE_DATA_ROOT to the LaDe-D data directory before running this script."
}

if (-not (Test-Path -LiteralPath $LaDe -PathType Container)) {
    throw "LADE_DATA_ROOT does not exist or is not a directory: $LaDe"
}
$Output = Join-Path $Repo "reports\lade_raw_profile"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment Python not found: $Python"
}

if (-not (Test-Path -LiteralPath $LaDe)) {
    throw "LaDe-D input folder not found: $LaDe"
}

Set-Location -LiteralPath $Repo

Write-Host "===== VALIDATE PROFILER ====="
& $Python -m compileall -q "scripts\profile_lade_raw.py"
if ($LASTEXITCODE -ne 0) {
    throw "LaDe raw profiler compile validation failed."
}

Write-Host ""
Write-Host "===== PROFILE LaDe-D RAW PARQUET ====="
& $Python "scripts\profile_lade_raw.py" `
    --input-dir $LaDe `
    --output-dir $Output `
    --batch-size 100000

if ($LASTEXITCODE -ne 0) {
    throw "LaDe raw profile failed."
}

Write-Host ""
Write-Host "===== PROFILE SUMMARY PATH ====="
Write-Host "$Output\lade_raw_profile.json"
