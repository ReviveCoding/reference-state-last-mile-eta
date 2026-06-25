$Repo = $PSScriptRoot
$Python = "$Repo\.venv\Scripts\python.exe"
$Output = "$Repo\reports\lade_raw_confirmatory\confirmatory_membership_diagnosis.json"

Set-Location -LiteralPath $Repo
$ErrorActionPreference = "Stop"

Write-Host "===== VALIDATE DIAGNOSTIC ====="
& $Python -m ruff check scripts\diagnose_lade_confirmatory_membership.py --fix
if ($LASTEXITCODE -ne 0) { throw "Ruff check failed." }
& $Python -m ruff format scripts\diagnose_lade_confirmatory_membership.py
if ($LASTEXITCODE -ne 0) { throw "Ruff format failed." }
& $Python -m compileall -q scripts\diagnose_lade_confirmatory_membership.py
if ($LASTEXITCODE -ne 0) { throw "Compile check failed." }

Write-Host ""
Write-Host "===== DIAGNOSE CONFIRMATORY COHORT MEMBERSHIP ====="
& $Python scripts\diagnose_lade_confirmatory_membership.py --output $Output
if ($LASTEXITCODE -ne 0) { throw "Confirmatory cohort diagnosis failed." }

Write-Host ""
Write-Host "DIAGNOSIS_PATH=$Output"
