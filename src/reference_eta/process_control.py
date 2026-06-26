from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Callable
from typing import Any, cast

CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
WINDOWS_CTRL_BREAK_EVENT = cast(
    signal.Signals | int,
    getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM),
)


def _kill_process_group(pid: int, sig: signal.Signals | int) -> None:
    """Call POSIX os.killpg without exposing Unix-only attrs to Windows mypy."""

    killpg = cast(
        "Callable[[int, signal.Signals | int], None] | None",
        getattr(os, "killpg", None),
    )
    if killpg is None:
        raise ProcessLookupError("process groups are not supported on this platform")
    killpg(pid, sig)


def popen_group_kwargs(platform_name: str | None = None) -> dict[str, Any]:
    platform_name = os.name if platform_name is None else platform_name
    if platform_name == "nt":
        return {"creationflags": CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def windows_taskkill_command(pid: int) -> list[str]:
    if pid <= 0:
        raise ValueError("pid must be positive")
    return ["taskkill", "/PID", str(pid), "/T", "/F"]


def terminate_process_tree(
    process: subprocess.Popen[Any],
    *,
    platform_name: str | None = None,
    grace_seconds: float = 5.0,
) -> None:
    if process.poll() is not None:
        return

    platform_name = os.name if platform_name is None else platform_name

    if platform_name == "nt":
        try:
            process.send_signal(WINDOWS_CTRL_BREAK_EVENT)
        except (AttributeError, OSError, ValueError):
            try:
                process.terminate()
            except OSError:
                return
    else:
        try:
            _kill_process_group(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass

    if platform_name == "nt":
        process.kill()
    else:
        try:
            _kill_process_group(
                process.pid,
                getattr(signal, "SIGKILL", signal.SIGTERM),
            )
        except ProcessLookupError:
            return

    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
