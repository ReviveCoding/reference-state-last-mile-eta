# Local LaDe-D data configuration

The public repository intentionally excludes raw and normalized LaDe-D data.

Set the local data directory before executing a raw-data normalization, profile, development, or final-confirmation runner.

## PowerShell

```powershell
$env:LADE_DATA_ROOT = "C:\path\to\LaDe-D_dataset\data"
```

## Shell

```text
LADE_DATA_ROOT=/path/to/LaDe-D_dataset/data
```

All runners resolve the repository root from their own location. Do not hardcode a user home directory in committed scripts.