from __future__ import annotations

import os
from pathlib import Path

import pytest

from reference_eta import cli


def test_project_root_uses_explicit_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_pipeline.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("REFERENCE_ETA_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)
    assert cli._project_root() == tmp_path.resolve()


def test_subprocess_environment_preserves_explicit_thread_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    environment = cli._subprocess_env(tmp_path)
    assert environment["OMP_NUM_THREADS"] == "7"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["PYTHONPATH"] == str(tmp_path / "src")


def test_smoke_command_dispatches_release_guarded_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripts = tmp_path / "scripts"
    configs = tmp_path / "configs"
    scripts.mkdir()
    configs.mkdir()
    (scripts / "run_pipeline.py").write_text("", encoding="utf-8")
    (configs / "smoke.yaml").write_text("seed: 1\n", encoding="utf-8")
    monkeypatch.setenv("REFERENCE_ETA_HOME", str(tmp_path))
    monkeypatch.setattr(os.sys, "argv", ["reference-eta", "smoke"])
    captured: dict[str, object] = {}

    def fake_run(command: list[str], root: Path | None, *, exit_on_complete: bool = True) -> int:
        captured["command"] = command
        captured["root"] = root
        captured["exit_on_complete"] = exit_on_complete
        raise SystemExit(0)

    monkeypatch.setattr(cli, "_run", fake_run)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    command = captured["command"]
    assert isinstance(command, list)
    assert "scripts/run_pipeline.py" in command
    assert "--require-release-pass" in command
    assert captured["root"] == tmp_path.resolve()


def test_serve_command_does_not_require_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        os.sys,
        "argv",
        [
            "reference-eta",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "9001",
            "--workers",
            "2",
        ],
    )
    captured: dict[str, object] = {}

    def fake_run(command: list[str], root: Path | None, *, exit_on_complete: bool = True) -> int:
        captured["command"] = command
        captured["root"] = root
        raise SystemExit(0)

    monkeypatch.setattr(cli, "_run", fake_run)
    with pytest.raises(SystemExit):
        cli.main()
    command = captured["command"]
    assert isinstance(command, list)
    assert "uvicorn" in command
    assert "127.0.0.1" in command
    assert "9001" in command
    assert "--workers" in command
    assert "2" in command
    assert captured["root"] is None


def test_serve_rejects_invalid_worker_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os.sys, "argv", ["reference-eta", "serve", "--workers", "0"])
    with pytest.raises(SystemExit, match="workers"):
        cli.main()
