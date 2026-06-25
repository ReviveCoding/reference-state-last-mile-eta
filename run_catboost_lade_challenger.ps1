$Repo = $PSScriptRoot
$Python = "$Repo\.venv\Scripts\python.exe"
$Namespace = "lade_raw_pilot_catboost_v1"

Set-Location -LiteralPath $Repo
$ErrorActionPreference = "Stop"

$Train = "$Repo\artifacts\lade_raw_pilot_baseline_v1\data\train_snapshots.csv"
$Validation = "$Repo\artifacts\lade_raw_pilot_baseline_v1\data\validation_snapshots.csv"
$Test = "$Repo\artifacts\lade_raw_pilot_baseline_v1\data\test_snapshots.csv"
$Config = "$Repo\configs\lade_raw_pilot.yaml"
$Output = "$Repo\reports\$Namespace"
$Script = "$Repo\scripts\run_catboost_lade_challenger.py"

foreach ($Path in @($Train, $Validation, $Test, $Config, $Script)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing required file: $Path"
    }
}

Write-Host "===== INSTALL / VERIFY CATBOOST ====="
& $Python -m pip install --upgrade "catboost>=1.2,<2"
if ($LASTEXITCODE -ne 0) {
    throw "CatBoost installation failed."
}

& $Python -c "import catboost; print('CATBOOST=' + catboost.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "CatBoost import validation failed."
}

Write-Host ""
Write-Host "===== VALIDATE CHALLENGER SCRIPT ====="
& $Python -m ruff check scripts\run_catboost_lade_challenger.py --fix
if ($LASTEXITCODE -ne 0) {
    throw "Ruff check failed."
}

& $Python -m ruff format scripts\run_catboost_lade_challenger.py
if ($LASTEXITCODE -ne 0) {
    throw "Ruff format failed."
}

& $Python -m compileall -q scripts\run_catboost_lade_challenger.py
if ($LASTEXITCODE -ne 0) {
    throw "Compile validation failed."
}

Write-Host ""
Write-Host "===== RUN CPU DETERMINISTIC CATBOOST CHALLENGER ====="
& $Python scripts\run_catboost_lade_challenger.py `
    --config $Config `
    --train $Train `
    --validation $Validation `
    --test $Test `
    --output-dir $Output `
    --seed 42

if ($LASTEXITCODE -ne 0) {
    throw "CatBoost challenger experiment failed."
}

Write-Host ""
Write-Host "===== SUMMARY ====="
Get-Content -LiteralPath "$Output\catboost_challenger_summary.json"

Write-Host ""
Write-Host "===== TEST SCORECARD ====="
Import-Csv -LiteralPath "$Output\catboost_challenger_scorecard.csv" |
    Format-Table -AutoSize

Write-Host ""
Write-Host "===== VALIDATION CANDIDATES ====="
Import-Csv -LiteralPath "$Output\catboost_challenger_validation.csv" |
    Sort-Object { [double]$_.validation_mae } |
    Format-Table -AutoSize

Write-Host ""
Write-Host "CATBOOST_LADE_CHALLENGER_EXIT=0"
