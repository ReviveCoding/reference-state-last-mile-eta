from __future__ import annotations

import hashlib
from pathlib import Path

from reference_eta.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".locks",
    ".mypy_cache",
    "__pycache__",
    "build",
}
OUTPUT = ROOT / "artifacts" / "release_manifest.json"
MUTABLE_LOG = ROOT / "reports" / "final_release_log.txt"
DYNAMIC_QUALIFICATION_FILES = {
    ROOT / "qualification_manifest.json",
    ROOT / "release_bundle_manifest.json",
    ROOT / "release_candidate_handoff.json",
    ROOT / "release_candidate_handoff.json.sha256",
    ROOT / "reports" / "clean_candidate_validation.json",
    ROOT / "reports" / "local_qualification_summary.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(
            part in EXCLUDED_PARTS or part.startswith(".venv") or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.is_symlink():
            raise SystemExit(f"Release manifest refuses symbolic links: {relative}")
        if not path.is_file():
            continue
        if (
            path.name == ".coverage"
            or path.name.startswith(".coverage.")
            or path.name.endswith((".pyc", ".pyo"))
            or path in {OUTPUT, MUTABLE_LOG}
            or path in DYNAMIC_QUALIFICATION_FILES
        ):
            continue
        files.append(
            {
                "path": str(relative),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    OUTPUT.parent.mkdir(exist_ok=True)
    atomic_write_json(OUTPUT, files)
    print(f"Wrote {len(files)} release entries")


if __name__ == "__main__":
    main()
