$Repo = $PSScriptRoot
$Python = "$Repo\.venv\Scripts\python.exe"
$Namespace = "lade_raw_confirmatory_baseline_v1"
$DataOutput = "$Repo\artifacts\lade_raw_confirmatory\lade_delivery_normalized.csv"
$NormalizeReport = "$Repo\reports\lade_raw_confirmatory\normalization_report.json"
$ConfirmReports = "$Repo\reports\lade_raw_confirmatory_citybase_v1"
$ExcludeCsv = "$Repo\artifacts\lade_raw_pilot\lade_delivery_normalized.csv"

Set-Location -LiteralPath $Repo
$ErrorActionPreference = "Stop"

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

Write-Host "===== VERIFY DEPENDENCIES (CITYBASE CONFIRMATORY v1.2) ====="
Invoke-PythonChecked -Label "Dependency verification" -Arguments @(
    "-c",
    "import pyarrow, catboost; print('PYARROW=' + pyarrow.__version__); print('CATBOOST=' + catboost.__version__)"
)

Write-Host ""
Write-Host "===== FORMAT + VALIDATE CONFIRMATION SCRIPTS ====="
Invoke-PythonChecked -Label "Ruff check" -Arguments @(
    "-m", "ruff", "check",
    "scripts\normalize_lade_raw_confirmatory.py",
    "scripts\run_catboost_citybase_confirmatory.py",
    "--fix"
)
Invoke-PythonChecked -Label "Ruff format" -Arguments @(
    "-m", "ruff", "format",
    "scripts\normalize_lade_raw_confirmatory.py",
    "scripts\run_catboost_citybase_confirmatory.py"
)
Invoke-PythonChecked -Label "Compile validation" -Arguments @(
    "-m", "compileall", "-q",
    "scripts\normalize_lade_raw_confirmatory.py",
    "scripts\run_catboost_citybase_confirmatory.py"
)

Write-Host ""
Write-Host "===== BUILD INDEPENDENT CONFIRMATORY COHORT ====="
Invoke-PythonChecked -Label "Confirmatory cohort normalization" -Arguments @(
    "scripts\normalize_lade_raw_confirmatory.py",
    "--output-csv", $DataOutput,
    "--report-path", $NormalizeReport,
    "--seed", "314159",
    "--max-courier-days", "2000",
    "--exclude-normalized-csv", $ExcludeCsv
)

Write-Host ""
Write-Host "===== BUILD FIXED CHRONOLOGICAL PARTITIONS ====="
Invoke-PythonChecked -Label "Confirmatory baseline partitions" -Arguments @(
    "scripts\run_pipeline.py",
    "--config", "$Repo\configs\lade_raw_confirmatory.yaml",
    "--output-namespace", $Namespace
)

Write-Host ""
Write-Host "===== RUN PRE-SPECIFIED CATBOOST CITYBASE CONFIRMATION ====="
Invoke-PythonChecked -Label "CatBoost CityBase confirmation" -Arguments @(
    "scripts\run_catboost_citybase_confirmatory.py",
    "--data-dir", "$Repo\artifacts\$Namespace\data",
    "--output-dir", $ConfirmReports,
    "--seed", "42"
)

Write-Host ""
Write-Host "===== CONFIRMATORY SUMMARY ====="
Get-Content -LiteralPath "$ConfirmReports\confirmatory_summary.json"

Write-Host ""
Write-Host "===== CONFIRMATORY SCORECARD ====="
Import-Csv -LiteralPath "$ConfirmReports\confirmatory_scorecard.csv" |
    Format-Table -AutoSize

Write-Host ""
Write-Host "CITYBASE_CONFIRMATORY_EXIT=0"
