from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    removed: list[str] = []
    for path in [ROOT / "build", *ROOT.glob("*.egg-info"), *(ROOT / "src").glob("*.egg-info")]:
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path.relative_to(ROOT)))
    print("Removed build metadata: " + (", ".join(removed) if removed else "none"))


if __name__ == "__main__":
    main()
