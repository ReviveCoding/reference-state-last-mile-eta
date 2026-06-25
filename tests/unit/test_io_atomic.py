from __future__ import annotations

import json
from pathlib import Path

import pytest

from reference_eta.io import atomic_copy, atomic_write_json, atomic_write_text


def test_atomic_text_and_json_replace_existing_content_without_temp_files(tmp_path: Path) -> None:
    text_path = tmp_path / "nested" / "value.txt"
    atomic_write_text(text_path, "first")
    atomic_write_text(text_path, "second")
    assert text_path.read_text(encoding="utf-8") == "second"

    json_path = tmp_path / "value.json"
    atomic_write_json(json_path, {"b": 2, "a": 1})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert not list(tmp_path.rglob("*.tmp"))


def test_atomic_copy_replaces_destination_and_preserves_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "out" / "destination.bin"
    source.write_bytes(b"new-content")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old")
    atomic_copy(source, destination)
    assert destination.read_bytes() == b"new-content"
    assert not list(tmp_path.rglob("*.tmp"))


def test_atomic_copy_does_not_publish_when_source_is_missing(tmp_path: Path) -> None:
    destination = tmp_path / "destination.bin"
    with pytest.raises(FileNotFoundError):
        atomic_copy(tmp_path / "missing.bin", destination)
    assert not destination.exists()
    assert not list(tmp_path.rglob("*.tmp"))
