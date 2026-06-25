$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$LaDe = $env:LADE_DATA_ROOT

if ([string]::IsNullOrWhiteSpace($LaDe)) {
    throw "LADE_DATA_ROOT is required. Set `$env:LADE_DATA_ROOT to the LaDe-D data directory before running this script."
}

if (-not (Test-Path -LiteralPath $LaDe -PathType Container)) {
    throw "LADE_DATA_ROOT does not exist or is not a directory: $LaDe"
}
$OutputDir = Join-Path $Repo "artifacts\lade_raw_pilot"
$OutputCsv = Join-Path $OutputDir "lade_delivery_normalized.csv"
$ReportPath = Join-Path $Repo "reports\lade_raw_pilot\normalization_report.json"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment Python was not found: $Python"
}

if (-not (Test-Path -LiteralPath $LaDe)) {
    throw "LaDe-D input folder was not found: $LaDe"
}

Set-Location -LiteralPath $Repo
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ReportPath) | Out-Null

Write-Host "===== VALIDATE NORMALIZER ====="
& $Python -m ruff check scripts\normalize_lade_raw_pilot.py
& $Python -m ruff format --check scripts\normalize_lade_raw_pilot.py
& $Python -m compileall -q scripts\normalize_lade_raw_pilot.py

Write-Host ""
Write-Host "===== NORMALIZE DETERMINISTIC LaDe-D PILOT ====="
& $Python scripts\normalize_lade_raw_pilot.py `
    --input-dir $LaDe `
    --output-csv $OutputCsv `
    --report-path $ReportPath `
    --cities Hangzhou Shanghai Chongqing `
    --max-courier-days 2000 `
    --max-duration-minutes 720 `
    --anchor-year 2022 `
    --seed 42

Write-Host ""
Write-Host "===== OUTPUTS ====="
Write-Host "NORMALIZED_CSV=$OutputCsv"
Write-Host "NORMALIZATION_REPORT=$ReportPath"
Get-Content -LiteralPath $ReportPath
