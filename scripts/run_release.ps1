$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    python scripts/tasks.py release
    if ($LASTEXITCODE -ne 0) {
        throw "Release failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
