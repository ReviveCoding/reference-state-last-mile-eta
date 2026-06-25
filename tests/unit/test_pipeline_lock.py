from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_pipeline as pipeline
from reference_eta.locking import ExclusiveFileLock, LockTimeoutError


def test_root_pipeline_lock_blocks_second_writer(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setenv("REFERENCE_ETA_LOCK_TIMEOUT_SECONDS", "0")
    lock_path = tmp_path / "artifacts" / ".locks" / "root.lock"
    held = ExclusiveFileLock(lock_path, purpose="held").acquire()
    called = False

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(pipeline, "_run_unlocked", fake_run)
    try:
        with pytest.raises(LockTimeoutError):
            pipeline.run(Path("config.yaml"), "smoke")
    finally:
        held.release()
    assert not called
