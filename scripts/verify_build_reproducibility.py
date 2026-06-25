from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from reference_eta.io import atomic_write_json
from scripts.normalize_sdist import normalize_sdist

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_build_metadata() -> None:
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    for path in (ROOT / "src").glob("*.egg-info"):
        shutil.rmtree(path, ignore_errors=True)


def verify_build_reproducibility(dist_dir: Path, *, epoch: int) -> dict[str, object]:
    dist_dir = dist_dir.resolve()
    expected = sorted([*dist_dir.glob("*.whl"), *dist_dir.glob("*.tar.gz")])
    if len(expected) != 2:
        raise ValueError(
            "Expected exactly one wheel and one sdist for reproducibility verification"
        )
    expected_hashes = {path.name: _sha256(path) for path in expected}

    try:
        with tempfile.TemporaryDirectory(prefix="reference-eta-rebuild-") as temporary:
            rebuilt_dir = Path(temporary)
            _clean_build_metadata()
            env = os.environ.copy()
            env["SOURCE_DATE_EPOCH"] = str(epoch)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--no-isolation",
                    "--sdist",
                    "--wheel",
                    "--outdir",
                    str(rebuilt_dir),
                    ".",
                ],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            rebuilt_sdists = list(rebuilt_dir.glob("*.tar.gz"))
            if len(rebuilt_sdists) != 1:
                raise ValueError("Rebuild did not produce exactly one sdist")
            normalize_sdist(rebuilt_sdists[0], epoch=epoch)
            rebuilt = sorted([*rebuilt_dir.glob("*.whl"), *rebuilt_dir.glob("*.tar.gz")])
            rebuilt_hashes = {path.name: _sha256(path) for path in rebuilt}
    finally:
        _clean_build_metadata()

    if expected_hashes != rebuilt_hashes:
        raise ValueError(
            "Distribution builds are not bitwise reproducible: "
            + json.dumps({"expected": expected_hashes, "rebuilt": rebuilt_hashes}, sort_keys=True)
        )
    return {
        "build_reproducibility_schema_version": 1,
        "status": "PASS",
        "source_date_epoch": epoch,
        "artifacts": expected_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild wheel/sdist and compare SHA-256 digests")
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    parser.add_argument(
        "--epoch",
        type=int,
        default=int(os.getenv("SOURCE_DATE_EPOCH", "1704067200")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/distribution_reproducibility.json",
    )
    args = parser.parse_args()
    result = verify_build_reproducibility(args.dist_dir, epoch=args.epoch)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
