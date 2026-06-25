$Repo = $PSScriptRoot
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$Pilot = Join-Path $Repo "artifacts\lade_raw_pilot_baseline_v1\data"
$Confirm = Join-Path $Repo "artifacts\lade_raw_confirmatory_baseline_v1\data"
$Output = Join-Path $Repo "reports\lade_feature_signal_audit"

Set-Location -LiteralPath $Repo
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $Pilot)) {
    throw "Pilot partition directory not found: $Pilot"
}
if (-not (Test-Path -LiteralPath $Confirm)) {
    throw "Confirmatory partition directory not found: $Confirm"
}

Write-Host "===== FORMAT + VALIDATE FEATURE AUDIT ====="
& $Python -m ruff check scripts\audit_lade_feature_signal.py --fix
if ($LASTEXITCODE -ne 0) { throw "Ruff check failed." }
& $Python -m ruff format scripts\audit_lade_feature_signal.py
if ($LASTEXITCODE -ne 0) { throw "Ruff format failed." }
& $Python -m compileall -q scripts\audit_lade_feature_signal.py
if ($LASTEXITCODE -ne 0) { throw "Compile validation failed." }

Write-Host ""
Write-Host "===== AUDIT PILOT + CONFIRMATORY FEATURE SIGNAL ====="
& $Python scripts\audit_lade_feature_signal.py `
    --data-dir $Pilot `
    --label exploratory_pilot `
    --data-dir $Confirm `
    --label independent_confirmation `
    --output-dir $Output
if ($LASTEXITCODE -ne 0) { throw "Feature signal audit failed." }

Write-Host ""
Write-Host "===== AUDIT SUMMARY ====="
Get-Content -LiteralPath "$Output\feature_signal_audit.json"

Write-Host ""
Write-Host "===== CONSTANT / LOW-VARIANCE FEATURES ====="
Import-Csv -LiteralPath "$Output\numeric_feature_signal.csv" |
    Where-Object { $_.constant_or_near_constant -eq "True" } |
    Format-Table cohort, feature, unique_values, nonzero_rate, std -AutoSize

Write-Host ""
Write-Host "===== CATEGORICAL UNSEEN RATES ====="
Import-Csv -LiteralPath "$Output\categorical_generalization.csv" |
    Format-Table cohort, feature, split, train_unique, split_unique, unseen_rows, unseen_rate -AutoSize

Write-Host ""
Write-Host "FEATURE_SIGNAL_AUDIT_EXIT=0"
