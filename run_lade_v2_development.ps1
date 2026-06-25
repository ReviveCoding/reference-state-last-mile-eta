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
$Normalized = Join-Path $Repo "artifacts\lade_raw_v2_development\lade_delivery_normalized.csv"
$NormalizeReport = Join-Path $Repo "reports\lade_raw_v2_development\normalization_report.json"
$Artifacts = Join-Path $Repo "artifacts\lade_v2_development_v1"
$Reports = Join-Path $Repo "reports\lade_v2_development_v1"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)] [string] $Label,
        [Parameter(Mandatory = $true)] [scriptblock] $Action
    )
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment Python not found: $Python"
}
if (-not (Test-Path -LiteralPath $LaDe)) {
    throw "LaDe-D raw directory not found: $LaDe"
}

Set-Location -LiteralPath $Repo

Write-Host "===== VERIFY V2 DEVELOPMENT DEPENDENCIES ====="
& $Python -c "import lightgbm, numpy, pandas, pyarrow; print('LIGHTGBM=' + lightgbm.__version__); print('PYARROW=' + pyarrow.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "V2 dependencies are missing."
}

Write-Host ""
Write-Host "===== FORMAT + VALIDATE V2 SCRIPTS ====="
Invoke-Checked "Ruff check" { & $Python -m ruff check scripts\normalize_lade_raw_v2_development.py scripts\run_lade_v2_development.py --fix }
Invoke-Checked "Ruff format" { & $Python -m ruff format scripts\normalize_lade_raw_v2_development.py scripts\run_lade_v2_development.py }
Invoke-Checked "Compile validation" { & $Python -m compileall -q scripts\normalize_lade_raw_v2_development.py scripts\run_lade_v2_development.py }

Remove-Item -LiteralPath $Normalized -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $NormalizeReport -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $Artifacts -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $Reports -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "===== BUILD FULLY DISJOINT V2 DEVELOPMENT COHORT ====="
Invoke-Checked "V2 development normalization" {
    & $Python scripts\normalize_lade_raw_v2_development.py `
        --input-dir $LaDe `
        --output-csv $Normalized `
        --report-path $NormalizeReport `
        --seed 271828 `
        --max-courier-days 2000
}

Write-Host ""
Write-Host "===== RUN PRE-SPECIFIED EVENT-STATE V2 BENCHMARK ====="
Invoke-Checked "V2 development benchmark" {
    & $Python scripts\run_lade_v2_development.py `
        --input-csv $Normalized `
        --artifact-dir $Artifacts `
        --report-dir $Reports `
        --seed 271828
}

Write-Host ""
Write-Host "===== V2 DEVELOPMENT SUMMARY ====="
Get-Content -LiteralPath "$Reports\v2_development_summary.json"

Write-Host ""
Write-Host "===== V2 DEVELOPMENT SCORECARD ====="
Import-Csv -LiteralPath "$Reports\v2_development_scorecard.csv" | Format-Table -AutoSize

Write-Host ""
Write-Host "===== TOP V2 FEATURE IMPORTANCE ====="
Import-Csv -LiteralPath "$Reports\v2_event_feature_importance.csv" |
    Select-Object -First 20 | Format-Table -AutoSize

Write-Host ""
Write-Host "V2_DEVELOPMENT_EXIT=0"
