from __future__ import annotations

import gzip
import hashlib
import io
import os
import tarfile
from pathlib import Path

from scripts.normalize_sdist import normalize_sdist


def _write_sdist(path: Path, *, mtime: int) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="source.tar", mode="wb", fileobj=raw, mtime=mtime) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                data = b"hello\n"
                info = tarfile.TarInfo("project/file.txt")
                info.size = len(data)
                info.mtime = mtime
                archive.addfile(info, io.BytesIO(data))


def test_normalized_sdist_is_bitwise_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    _write_sdist(first, mtime=100)
    _write_sdist(second, mtime=200)
    normalize_sdist(first, epoch=1234)
    normalize_sdist(second, epoch=1234)
    assert (
        hashlib.sha256(first.read_bytes()).hexdigest()
        == hashlib.sha256(second.read_bytes()).hexdigest()
    )


def test_cli_dist_dir_requires_at_least_one_archive(tmp_path: Path) -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "scripts.normalize_sdist", "--dist-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "No source distributions" in result.stderr


def _write_custom_sdist(path: Path, names: list[str]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="source.tar", mode="wb", fileobj=raw, mtime=1) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for name in names:
                    data = name.encode("utf-8")
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))


def test_normalizer_rejects_unsafe_member_paths(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.tar.gz"
    _write_custom_sdist(path, ["../escape.txt"])
    import pytest

    with pytest.raises(ValueError, match="Unsafe source-distribution member path"):
        normalize_sdist(path, epoch=1)


def test_normalizer_rejects_duplicate_members(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.tar.gz"
    _write_custom_sdist(path, ["project/file.txt", "project/file.txt"])
    import pytest

    with pytest.raises(ValueError, match="Duplicate source-distribution member"):
        normalize_sdist(path, epoch=1)


def test_normalized_sdist_is_world_readable(tmp_path: Path) -> None:
    path = tmp_path / "readable.tar.gz"
    _write_sdist(path, mtime=100)
    normalize_sdist(path, epoch=1234)
    if os.name == "nt":
        assert path.exists()
    else:
        assert path.stat().st_mode & 0o777 == 0o644
