from __future__ import annotations

import pytest

from scripts import tasks


def test_cross_platform_runner_exposes_required_commands() -> None:
    assert tasks._commands("smoke")[0].argv[-2:] == [
        "--require-release-pass",
        "--force-process-exit",
    ]
    assert tasks._commands("release")[0].argv[-1] == "scripts/release.py"
    assert any(
        "verify_build_reproducibility" in " ".join(step.argv)
        for step in tasks._commands("package-check")
    )


def test_task_commands_are_argument_arrays_without_shell_strings() -> None:
    for name in ("lint", "coverage", "smoke", "train-gpu", "lade-smoke", "amazon-smoke"):
        for command in tasks._commands(name):
            assert isinstance(command.argv, list)
            assert all(isinstance(item, str) and item for item in command.argv)


def test_serve_task_uses_validated_worker_configuration(monkeypatch) -> None:
    monkeypatch.setenv("REFERENCE_ETA_WORKERS", "3")
    command = tasks._commands("serve")[0].argv
    assert command[command.index("--workers") + 1] == "3"
    monkeypatch.setenv("REFERENCE_ETA_WORKERS", "0")
    with pytest.raises(ValueError, match="between 1 and 8"):
        tasks._commands("serve")


def test_locking_task_uses_forced_process_exit() -> None:
    commands = tasks._commands("locking-check")
    assert len(commands) == 1
    assert "--force-process-exit" in commands[0].argv


def test_package_command_construction_does_not_delete_existing_dist(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tasks, "ROOT", tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "candidate.whl"
    wheel.write_bytes(b"wheel")
    commands = tasks._commands("package-check")
    assert wheel.is_file()
    assert commands[0].argv[-1] == "scripts/clean_distribution.py"


def test_package_check_command_construction_is_valid() -> None:
    from scripts.tasks import _commands

    commands = _commands("package-check")
    assert [command.name for command in commands] == [
        "clean-distribution",
        "distribution-build",
        "normalize-sdist",
        "verify-distribution",
        "reproducible-distribution",
        "clean-build-metadata",
    ]
    assert all(isinstance(command.argv, list) for command in commands)
    assert commands[2].argv[:3] == [commands[2].argv[0], "-m", "scripts.normalize_sdist"]
