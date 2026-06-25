from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from reference_eta import __version__  # noqa: E402
from reference_eta.io import atomic_write_json  # noqa: E402
from reference_eta.locking import ExclusiveFileLock, LockTimeoutError  # noqa: E402
from scripts.run_pipeline import _publish_serving_bundle  # noqa: E402

REQUIRED = (
    "rcot.joblib",
    "quantile_champion.joblib",
    "cqr_calibrator.joblib",
    "tail_thresholds.joblib",
    "release_decision.json",
    "run_provenance.json",
)


def _hold_lock(path: str, ready: mp.synchronize.Event, release: mp.synchronize.Event) -> None:
    with ExclusiveFileLock(Path(path), purpose="subprocess holder"):
        ready.set()
        release.wait(timeout=20.0)


def _verify_cross_process_exclusion(root: Path) -> dict[str, object]:
    lock_path = root / "cross-process.lock"
    context = mp.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=_hold_lock, args=(str(lock_path), ready, release))
    process.start()
    try:
        if not ready.wait(timeout=10.0):
            raise RuntimeError("Lock holder did not become ready")
        rejected = False
        try:
            ExclusiveFileLock(
                lock_path,
                timeout_seconds=0.2,
                poll_seconds=0.02,
                purpose="contender",
            ).acquire()
        except LockTimeoutError:
            rejected = True
        if not rejected:
            raise RuntimeError("Concurrent lock contender was not rejected")
    finally:
        release.set()
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
    if process.exitcode != 0:
        raise RuntimeError(f"Lock holder failed with exit code {process.exitcode}")
    if lock_path.exists():
        raise RuntimeError("Lock file remained after holder exit")
    return {"concurrent_writer_rejected": True, "holder_exit_code": process.exitcode}


def _write_source_artifacts(root: Path) -> None:
    for index, name in enumerate(REQUIRED):
        if name == "run_provenance.json":
            (root / name).write_text(
                json.dumps(
                    {
                        "provenance_schema_version": 2,
                        "project": {
                            "name": "reference-state-last-mile-eta",
                            "version": __version__,
                        },
                    }
                ),
                encoding="utf-8",
            )
        else:
            (root / name).write_bytes(f"payload-{index}".encode())


def _verify_atomic_pointer_failure(root: Path) -> dict[str, object]:
    _write_source_artifacts(root)
    first = _publish_serving_bundle(root)
    pointer_path = root / "current_bundle.json"
    before = pointer_path.read_text(encoding="utf-8")
    (root / "release_decision.json").write_bytes(b"changed-release")
    injected = False
    try:
        _publish_serving_bundle(root, fault_stage="after_bundle_before_pointer")
    except RuntimeError as error:
        if "Injected publish failure" not in str(error):
            raise
        injected = True
    if not injected:
        raise RuntimeError("Publish failure injection did not trigger")
    after = pointer_path.read_text(encoding="utf-8")
    if before != after:
        raise RuntimeError("Pointer changed after injected publish failure")
    if list((root / "serving_bundles").glob(".*.tmp")):
        raise RuntimeError("Temporary serving bundle remained after failure")
    return {
        "failure_injected": True,
        "pointer_unchanged": True,
        "original_bundle_id": first["bundle_id"],
    }


def verify() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="reference-eta-locking-") as temporary:
        root = Path(temporary)
        exclusion = _verify_cross_process_exclusion(root / "lock")
        publication_root = root / "publish"
        publication_root.mkdir(parents=True)
        pointer = _verify_atomic_pointer_failure(publication_root)
    return {
        "locking_verification_schema_version": 1,
        "status": "PASS",
        "cross_process_exclusion": exclusion,
        "atomic_pointer_failure": pointer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify locking and atomic publish recovery")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "locking_recovery_report.json"
    )
    parser.add_argument(
        "--force-process-exit",
        action="store_true",
        help="Flush evidence and bypass multiprocessing/native shutdown after completion",
    )
    args = parser.parse_args()
    result = verify()
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if args.force_process_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    mp.freeze_support()
    main()
