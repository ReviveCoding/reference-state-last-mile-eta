from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


class LockTimeoutError(TimeoutError):
    pass


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@dataclass
class ExclusiveFileLock:
    path: Path
    timeout_seconds: float = 0.0
    poll_seconds: float = 0.1
    stale_after_seconds: float = 3600.0
    purpose: str = "exclusive operation"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.timeout_seconds < 0.0:
            raise ValueError("timeout_seconds must be nonnegative")
        if not 0.01 <= self.poll_seconds <= 10.0:
            raise ValueError("poll_seconds must be between 0.01 and 10")
        if self.stale_after_seconds <= 0.0:
            raise ValueError("stale_after_seconds must be positive")
        self._token: str | None = None

    def _metadata(self, token: str) -> dict[str, object]:
        return {
            "lock_schema_version": 1,
            "token": token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_unix_seconds": time.time(),
            "purpose": self.purpose,
        }

    def _read_metadata(self) -> dict[str, object] | None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return raw if isinstance(raw, dict) else None

    def _is_stale(self) -> bool:
        try:
            age = max(0.0, time.time() - self.path.stat().st_mtime)
        except FileNotFoundError:
            return False
        metadata = self._read_metadata()
        if metadata is None:
            return age > self.stale_after_seconds
        hostname = str(metadata.get("hostname", ""))
        try:
            pid = int(str(metadata.get("pid", -1)))
        except (TypeError, ValueError):
            return age > self.stale_after_seconds
        if hostname == socket.gethostname():
            return not _pid_is_running(pid)
        return age > self.stale_after_seconds

    def acquire(self) -> ExclusiveFileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            token = uuid.uuid4().hex
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError as error:
                if self._is_stale():
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    owner = self._read_metadata()
                    raise LockTimeoutError(
                        f"Timed out acquiring {self.purpose} lock: {self.path}; owner={owner}"
                    ) from error
                time.sleep(self.poll_seconds)
                continue
            try:
                payload = json.dumps(self._metadata(token), indent=2, sort_keys=True).encode(
                    "utf-8"
                )
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._token = token
            return self

    def release(self) -> None:
        if self._token is None:
            return
        metadata = self._read_metadata()
        if metadata is not None and str(metadata.get("token", "")) == self._token:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self._token = None

    def __enter__(self) -> ExclusiveFileLock:
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        self.release()
