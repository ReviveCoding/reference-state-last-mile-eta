from pathlib import Path


def test_clean_candidate_verification_checks_tests_wheel_and_manifests() -> None:
    source = Path("scripts/verify_clean_candidate.py").read_text(encoding="utf-8")
    assert '"--no-deps"' in source
    assert '"-p", "no:cacheprovider"' in source
    assert '"pytest", "-q"' not in source
    assert '"--all"' in source
    assert '"artifacts/release_manifest.json"' in source


def test_clean_candidate_timeout_is_configurable_and_guarded(monkeypatch) -> None:
    from scripts.verify_clean_candidate import _command_timeout_seconds

    monkeypatch.delenv("REFERENCE_ETA_CLEAN_COMMAND_TIMEOUT_SECONDS", raising=False)
    assert _command_timeout_seconds() == 600
    monkeypatch.setenv("REFERENCE_ETA_CLEAN_COMMAND_TIMEOUT_SECONDS", "120")
    assert _command_timeout_seconds() == 120
    monkeypatch.setenv("REFERENCE_ETA_CLEAN_COMMAND_TIMEOUT_SECONDS", "59")
    import pytest

    with pytest.raises(ValueError, match="at least 60"):
        _command_timeout_seconds()
    monkeypatch.setenv("REFERENCE_ETA_CLEAN_COMMAND_TIMEOUT_SECONDS", "bad")
    with pytest.raises(ValueError, match="must be an integer"):
        _command_timeout_seconds()
