$ErrorActionPreference = "Stop"

$Repo = $PSScriptRoot
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$Experiment = Join-Path $Repo "scripts\run_gated_rcot_experiment.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment Python not found: $Python"
}

if (-not (Test-Path -LiteralPath $Experiment)) {
    throw "Experiment file not found: $Experiment"
}

Set-Location -LiteralPath $Repo

Write-Host "===== VALIDATE SCRIPT ====="
& $Python -m ruff check scripts\run_gated_rcot_experiment.py --fix
if ($LASTEXITCODE -ne 0) {
    throw "Ruff check failed."
}

& $Python -m ruff format scripts\run_gated_rcot_experiment.py
if ($LASTEXITCODE -ne 0) {
    throw "Ruff format failed."
}

& $Python -m compileall -q scripts\run_gated_rcot_experiment.py
if ($LASTEXITCODE -ne 0) {
    throw "Compile check failed."
}

Write-Host ""
Write-Host "===== RUN GATED RCOT EXPERIMENT ====="
& $Python scripts\run_gated_rcot_experiment.py `
    --config configs\smoke.yaml `
    --data-dir artifacts\data `
    --output-dir reports\gated_rcot_experiment `
    --oof-splits 5

if ($LASTEXITCODE -ne 0) {
    throw "Gated RCOT experiment failed."
}

Write-Host ""
Write-Host "===== SUMMARY ====="
Get-Content -LiteralPath `
    "$Repo\reports\gated_rcot_experiment\gated_rcot_summary.json"

Write-Host ""
Write-Host "===== TEST SCORECARD ====="
Import-Csv `
    "$Repo\reports\gated_rcot_experiment\gated_rcot_scorecard.csv" |
    Format-Table -AutoSize

Write-Host ""
Write-Host "===== TOP VALIDATION CANDIDATES ====="
Import-Csv `
    "$Repo\reports\gated_rcot_experiment\gated_rcot_validation_grid.csv" |
    Sort-Object { [double]$_.validation_mae } |
    Select-Object -First 12 `
        threshold, `
        clip_minutes, `
        blend, `
        activation_rate, `
        validation_mae, `
        relative_mae_improvement, `
        promote, `
        promotion_reason |
    Format-Table -AutoSize
