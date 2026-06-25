# Windows PowerShell runbook

Use Python 3.11 or 3.13 in a fresh virtual environment. `make` is not required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -c constraints/ci-py311.txt -e ".[dev]"
python scripts/tasks.py lint
python scripts/tasks.py coverage
python scripts/tasks.py smoke
python scripts/tasks.py train-gpu
python scripts/tasks.py package-check
python scripts/tasks.py release
```

The equivalent convenience wrapper is:

```powershell
.\scripts\run_release.ps1
```

The cross-platform runner creates a Windows process group for each child process. On interruption or
step timeout, the release watchdog attempts `taskkill /T /F` so descendant Python, LightGBM, and
PyTorch processes are not left running.

Paths containing spaces are supported because commands are passed as argument arrays rather than
shell-composed strings. Do not run project commands from an elevated shell unless required by the
local Python installation.
