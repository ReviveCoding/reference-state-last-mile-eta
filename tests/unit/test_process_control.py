from __future__ import annotations

import subprocess

import pytest

from reference_eta.process_control import popen_group_kwargs, windows_taskkill_command


def test_platform_process_group_kwargs() -> None:
    assert popen_group_kwargs("posix") == {"start_new_session": True}
    windows = popen_group_kwargs("nt")
    assert windows["creationflags"] == getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)


def test_windows_taskkill_command_is_argument_safe() -> None:
    assert windows_taskkill_command(1234) == ["taskkill", "/PID", "1234", "/T", "/F"]
    with pytest.raises(ValueError):
        windows_taskkill_command(0)
