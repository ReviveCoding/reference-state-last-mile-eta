from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(manifest: Path) -> int:
    records: Any = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest must contain a non-empty list: {manifest}")
    failures: list[str] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not {"path", "size_bytes", "sha256"}.issubset(record):
            failures.append(f"invalid record in {manifest}")
            continue
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"unsafe path: {relative}")
            continue
        if str(relative) in seen:
            failures.append(f"duplicate record: {relative}")
            continue
        seen.add(str(relative))
        is_serving_bundle_manifest = (
            manifest.parent.parent.name == "serving_bundles"
            and manifest.parent.parent.parent.name == "artifacts"
        )
        base_dir = manifest.parent if is_serving_bundle_manifest else ROOT
        unresolved_path = base_dir / relative
        if unresolved_path.is_symlink():
            failures.append(f"symbolic links are not allowed: {relative}")
            continue
        path = unresolved_path.resolve()
        allowed_root = base_dir.resolve()
        if allowed_root not in path.parents and path != allowed_root:
            failures.append(f"path escapes manifest root: {relative}")
            continue
        if not path.exists() or not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        if path.resolve() == manifest.resolve():
            failures.append(f"manifest cannot hash itself: {relative}")
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != int(record["size_bytes"]):
            failures.append(f"size mismatch: {relative}")
        if actual_hash != str(record["sha256"]):
            failures.append(f"sha256 mismatch: {relative}")
    if failures:
        raise ValueError("Artifact manifest verification failed:\n" + "\n".join(failures))
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--all", action="store_true", dest="verify_all")
    args = parser.parse_args()
    if args.manifest is not None and args.verify_all:
        raise SystemExit("Use either --manifest or --all, not both")
    if args.manifest is not None:
        manifests = [args.manifest if args.manifest.is_absolute() else ROOT / args.manifest]
    elif args.verify_all:
        manifests = sorted((ROOT / "artifacts").rglob("artifact_manifest.json"))
    else:
        manifests = [ROOT / "artifacts" / "artifact_manifest.json"]
    if not manifests:
        raise SystemExit("No artifact manifests found; run the pipelines first")
    total = 0
    try:
        for manifest in manifests:
            if not manifest.exists():
                raise FileNotFoundError(f"Missing manifest: {manifest}")
            count = verify_manifest(manifest)
            total += count
            print(f"Verified {count} records: {manifest.relative_to(ROOT)}")
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Verified {len(manifests)} manifests and {total} total records")


if __name__ == "__main__":
    main()
