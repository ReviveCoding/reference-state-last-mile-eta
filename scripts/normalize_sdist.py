from __future__ import annotations

import argparse
import gzip
import io
import os
import tarfile
import tempfile
from pathlib import Path


def normalize_sdist(path: Path, *, epoch: int) -> None:
    path = Path(path)
    if not path.is_file() or not path.name.endswith(".tar.gz"):
        raise ValueError(f"Expected an existing .tar.gz source distribution: {path}")
    if epoch < 0:
        raise ValueError("SOURCE_DATE_EPOCH must be nonnegative")

    with tarfile.open(path, "r:gz") as source:
        members = source.getmembers()
        member_names: set[str] = set()
        payloads: dict[str, bytes] = {}
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe source-distribution member path: {member.name}")
            if member.name in member_names:
                raise ValueError(f"Duplicate source-distribution member: {member.name}")
            member_names.add(member.name)
            if member.issym() or member.islnk():
                raise ValueError(f"Source distributions cannot contain links: {member.name}")
            if member.isfile():
                extracted = source.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Could not read source-distribution member: {member.name}")
                payloads[member.name] = extracted.read()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                ) as target:
                    for original in sorted(members, key=lambda item: item.name):
                        member = tarfile.TarInfo(name=original.name)
                        member.type = original.type
                        member.mtime = epoch
                        member.uid = 0
                        member.gid = 0
                        member.uname = ""
                        member.gname = ""
                        member.mode = 0o755 if original.isdir() or original.mode & 0o111 else 0o644
                        member.pax_headers = {}
                        if original.isdir():
                            member.size = 0
                            target.addfile(member)
                        elif original.isfile():
                            data = payloads[original.name]
                            member.size = len(data)
                            target.addfile(member, io.BytesIO(data))
                        else:
                            raise ValueError(
                                f"Unsupported source-distribution member type: {original.name}"
                            )
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize sdist metadata for reproducible builds")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--dist-dir", type=Path)
    parser.add_argument(
        "--epoch",
        type=int,
        default=int(os.getenv("SOURCE_DATE_EPOCH", "1704067200")),
    )
    args = parser.parse_args()
    paths = list(args.paths)
    if args.dist_dir is not None:
        paths.extend(sorted(args.dist_dir.glob("*.tar.gz")))
    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique_paths:
        raise SystemExit("No source distributions were provided or found")
    for path in unique_paths:
        normalize_sdist(path, epoch=args.epoch)
        print(f"Normalized source distribution: {path}")


if __name__ == "__main__":
    main()
