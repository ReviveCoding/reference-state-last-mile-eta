from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS: list[tuple[str, list[str]]] = [
    ("pip-check", [sys.executable, "-m", "pip", "check"]),
    ("lint", [sys.executable, "scripts/tasks.py", "lint"]),
    ("typecheck", [sys.executable, "scripts/tasks.py", "typecheck"]),
    ("coverage", [sys.executable, "scripts/tasks.py", "coverage"]),
    ("smoke", [sys.executable, "scripts/tasks.py", "smoke"]),
    ("train-gpu", [sys.executable, "scripts/tasks.py", "train-gpu"]),
    ("lade-smoke", [sys.executable, "scripts/tasks.py", "lade-smoke"]),
    ("amazon-smoke", [sys.executable, "scripts/tasks.py", "amazon-smoke"]),
    ("runtime-api", [sys.executable, "scripts/verify_runtime_api.py"]),
    ("api-benchmark", [sys.executable, "scripts/tasks.py", "api-benchmark"]),
    ("locking-check", [sys.executable, "scripts/tasks.py", "locking-check"]),
    ("repro-check", [sys.executable, "scripts/tasks.py", "repro-check"]),
    ("package-check", [sys.executable, "scripts/tasks.py", "package-check"]),
    ("sbom", [sys.executable, "scripts/tasks.py", "sbom"]),
    ("verify-manifest", [sys.executable, "scripts/tasks.py", "verify-manifest"]),
    ("release-manifest", [sys.executable, "scripts/tasks.py", "release-manifest"]),
    ("clean-candidate", [sys.executable, "scripts/tasks.py", "clean-candidate"]),
    ("candidate-handoff", [sys.executable, "scripts/tasks.py", "candidate-handoff"]),
]


def _run(name: str, argv: list[str], env: dict[str, str]) -> dict[str, object]:
    started = time.perf_counter()
    print(f"== {name} ==", flush=True)
    print("$ " + subprocess.list2cmdline(argv), flush=True)
    completed = subprocess.run(argv, cwd=ROOT, env=env, text=True)
    elapsed = time.perf_counter() - started
    result = {
        "name": name,
        "command": argv,
        "exit_code": completed.returncode,
        "duration_seconds": round(elapsed, 3),
    }
    if completed.returncode != 0:
        raise SystemExit(json.dumps({"status": "FAIL", "failed_step": result}, indent=2))
    return result


def main() -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src if not existing else os.pathsep.join([src, existing])
    results = []
    for name, argv in COMMANDS:
        command_env = env.copy()
        if name == "pip-check":
            command_env.pop("PYTHONPATH", None)
        results.append(_run(name, argv, command_env))
    output = ROOT / "reports" / "local_qualification_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"status": "PASS", "commands": results}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "summary": str(output.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
