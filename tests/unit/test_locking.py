from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

import reference_eta.locking as locking
from reference_eta.locking import ExclusiveFileLock, LockTimeoutError


def test_exclusive_lock_records_owner_and_releases(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    with ExclusiveFileLock(path, purpose="test"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        assert metadata["pid"] == os.getpid()
        assert metadata["purpose"] == "test"
        assert metadata["token"]
    assert not path.exists()


def test_second_live_owner_times_out(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    first = ExclusiveFileLock(path, purpose="first").acquire()
    try:
        with pytest.raises(LockTimeoutError, match="owner="):
            ExclusiveFileLock(
                path,
                timeout_seconds=0.05,
                poll_seconds=0.01,
                purpose="second",
            ).acquire()
    finally:
        first.release()


def test_dead_local_pid_is_recovered_even_before_age_timeout(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    path.write_text(
        json.dumps(
            {
                "lock_schema_version": 1,
                "token": "stale",
                "pid": 999_999_999,
                "hostname": __import__("socket").gethostname(),
                "started_unix_seconds": time.time(),
                "purpose": "stale",
            }
        ),
        encoding="utf-8",
    )
    with ExclusiveFileLock(path, purpose="replacement"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        assert metadata["purpose"] == "replacement"


def test_non_owner_does_not_remove_replaced_lock(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    lock = ExclusiveFileLock(path, purpose="original").acquire()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["token"] = "other-owner"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    lock.release()
    assert path.exists()


def test_windows_pid_probe_avoids_console_control_event(monkeypatch) -> None:
    observed: list[list[str]] = []

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        observed.append(command)

        return subprocess.CompletedProcess(
            command,
            0,
            stdout='"python.exe","1234","Console","1","10 K"\n',
            stderr="",
        )

    def forbidden_kill(pid: int, sig: int) -> None:
        raise AssertionError(f"os.kill must not be used for Windows probing: {pid}, {sig}")

    monkeypatch.setattr(locking.subprocess, "run", fake_run)
    monkeypatch.setattr(locking.os, "kill", forbidden_kill)

    assert locking._pid_is_running(1234, platform_name="nt")
    assert observed == [["tasklist", "/FI", "PID eq 1234", "/FO", "CSV", "/NH"]]
