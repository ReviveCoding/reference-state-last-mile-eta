from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write text atomically within the destination filesystem."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(
    path: Path,
    value: Any,
    *,
    default: Callable[[Any], Any] | None = None,
) -> None:
    atomic_write_text(
        Path(path),
        json.dumps(value, indent=2, sort_keys=True, default=default),
    )


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy a file and atomically publish the completed copy."""

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target_handle:
            descriptor = -1
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, target_handle)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        shutil.copystat(source, temporary_path)
        os.replace(temporary_path, destination)
    except BaseException:
        if descriptor != -1:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise
