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
$Pilot = Join-Path $Repo "artifacts\lade_raw_pilot\lade_delivery_normalized.csv"
$Confirmatory = Join-Path $Repo "artifacts\lade_raw_confirmatory\lade_delivery_normalized.csv"
$Development = Join-Path $Repo "artifacts\lade_raw_v2_development\lade_delivery_normalized.csv"
$Normalized = Join-Path $Repo "artifacts\lade_raw_v2_final_confirmation\lade_delivery_normalized.csv"
$NormalizationReport = Join-Path $Repo "reports\lade_raw_v2_final_confirmation\normalization_report.json"
$Artifacts = Join-Path $Repo "artifacts\lade_v2_final_confirmation_v1"
$Reports = Join-Path $Repo "reports\lade_v2_final_confirmation_v1"

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

foreach ($Path in @($Python, $LaDe, $Pilot, $Confirmatory, $Development)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path was not found: $Path"
    }
}

Set-Location -LiteralPath $Repo

Write-Host "===== VERIFY V2 FINAL-CONFIRMATION DEPENDENCIES ====="
& $Python -c "import lightgbm, numpy, pandas, pyarrow; print('LIGHTGBM=' + lightgbm.__version__); print('PYARROW=' + pyarrow.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "V2 final-confirmation dependencies are missing."
}

Write-Host ""
Write-Host "===== FORMAT + VALIDATE FINAL-CONFIRMATION SCRIPTS ====="
Invoke-Checked "Ruff check" { & $Python -m ruff check scripts\normalize_lade_raw_v2_final_confirmation.py scripts\run_lade_v2_final_confirmation.py --fix }
Invoke-Checked "Ruff format" { & $Python -m ruff format scripts\normalize_lade_raw_v2_final_confirmation.py scripts\run_lade_v2_final_confirmation.py }
Invoke-Checked "Compile validation" { & $Python -m compileall -q scripts\normalize_lade_raw_v2_final_confirmation.py scripts\run_lade_v2_final_confirmation.py }

Remove-Item -LiteralPath $Normalized -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $NormalizationReport -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $Artifacts -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $Reports -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "===== BUILD UNTOUCHED V2 FINAL-CONFIRMATION COHORT ====="
Invoke-Checked "V2 final normalization" {
    & $Python scripts\normalize_lade_raw_v2_final_confirmation.py `
        --input-dir $LaDe `
        --output-csv $Normalized `
        --report-path $NormalizationReport `
        --pilot-csv $Pilot `
        --confirmatory-csv $Confirmatory `
        --development-csv $Development `
        --seed 1618033 `
        --max-courier-days 2000
}

Write-Host ""
Write-Host "===== RUN LOCKED V2 FINAL CONFIRMATION ====="
Invoke-Checked "V2 final benchmark" {
    & $Python scripts\run_lade_v2_final_confirmation.py `
        --input-csv $Normalized `
        --normalization-report $NormalizationReport `
        --artifact-dir $Artifacts `
        --report-dir $Reports `
        --seed 1618033 `
        --minimum-relative-mae-improvement 0.01 `
        --tail-mae-tolerance 0.02
}

Write-Host ""
Write-Host "===== V2 FINAL-CONFIRMATION SUMMARY ====="
Get-Content -LiteralPath "$Reports\v2_final_confirmation_summary.json"

Write-Host ""
Write-Host "===== V2 FINAL-CONFIRMATION SCORECARD ====="
Import-Csv -LiteralPath "$Reports\v2_final_confirmation_scorecard.csv" | Format-Table -AutoSize

Write-Host ""
Write-Host "V2_FINAL_CONFIRMATION_EXIT=0"
