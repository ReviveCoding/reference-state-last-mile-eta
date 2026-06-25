from __future__ import annotations

from pathlib import Path

from scripts.build_release_candidate_handoff import source_diff, source_fingerprint


def test_source_fingerprint_is_order_stable() -> None:
    first = [
        {"path": "b.py", "size_bytes": 2, "sha256": "b"},
        {"path": "a.py", "size_bytes": 1, "sha256": "a"},
    ]
    second = list(reversed(first))
    assert source_fingerprint(first) == source_fingerprint(second)


def test_source_diff_reports_added_modified_deleted_and_stable_checksum() -> None:
    baseline = [
        {"path": "same.py", "size_bytes": 1, "sha256": "a"},
        {"path": "changed.py", "size_bytes": 1, "sha256": "b"},
        {"path": "deleted.py", "size_bytes": 1, "sha256": "c"},
    ]
    candidate = [
        {"path": "same.py", "size_bytes": 1, "sha256": "a"},
        {"path": "changed.py", "size_bytes": 2, "sha256": "d"},
        {"path": "added.py", "size_bytes": 1, "sha256": "e"},
    ]
    result = source_diff(baseline, candidate)
    assert result["added"] == ["added.py"]
    assert result["modified"] == ["changed.py"]
    assert result["deleted"] == ["deleted.py"]
    assert len(str(result["diff_checksum"])) == 64


def test_handoff_files_are_intentionally_outside_repository_manifest_cycle() -> None:
    source = Path("scripts/build_release_manifest.py").read_text(encoding="utf-8")
    assert "release_candidate_handoff.json" in source


def test_source_fingerprint_excludes_machine_generated_qualification_manifests(tmp_path):
    from scripts.build_release_candidate_handoff import _source_records

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "code.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "qualification_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "release_bundle_manifest.json").write_text("{}", encoding="utf-8")
    records = _source_records(tmp_path)
    paths = {str(record["path"]) for record in records}
    assert "src/code.py" in paths
    assert "qualification_manifest.json" not in paths
    assert "release_bundle_manifest.json" not in paths
