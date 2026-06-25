from __future__ import annotations

import sys
from pathlib import Path

import pytest

import scripts.release as release
from reference_eta.locking import ExclusiveFileLock, LockTimeoutError


def test_full_release_lock_blocks_concurrent_release(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "LOG_PATH", tmp_path / "reports" / "final_release_log.txt")
    monkeypatch.setattr(release, "_preflight", lambda: None)
    monkeypatch.setenv("REFERENCE_ETA_LOCK_TIMEOUT_SECONDS", "0")
    monkeypatch.setattr(sys, "argv", ["release.py"])
    called = False

    def fake_execute(args):  # noqa: ANN001, ANN202
        nonlocal called
        called = True

    monkeypatch.setattr(release, "_execute_release", fake_execute)
    held = ExclusiveFileLock(
        tmp_path / "artifacts" / ".locks" / "release.lock", purpose="held"
    ).acquire()
    try:
        with pytest.raises(LockTimeoutError):
            release.main()
    finally:
        held.release()
    assert not called
