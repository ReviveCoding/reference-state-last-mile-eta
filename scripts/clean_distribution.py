from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    target = ROOT / "dist"
    shutil.rmtree(target, ignore_errors=True)
    print(f"Removed distribution directory: {target}")


if __name__ == "__main__":
    main()
