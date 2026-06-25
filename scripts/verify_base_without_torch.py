from __future__ import annotations

import importlib.abc
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):  # noqa: ANN001, ANN201
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch intentionally blocked for base-dependency validation")
        return None


def main() -> None:
    sys.meta_path.insert(0, _BlockTorch())
    from scripts.run_pipeline import run  # noqa: E402

    namespace = "base_without_torch"
    try:
        summary = run(ROOT / "configs/smoke.yaml", "smoke", output_namespace=namespace)
        if not str(summary["release_gate"]).startswith("PASS_"):
            raise SystemExit(f"Base-without-torch release gate failed: {summary['release_gate']}")
        if summary.get("advanced") is not None:
            raise SystemExit("Base-without-torch smoke unexpectedly executed the advanced model")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "release_gate": summary["release_gate"],
                    "torch_import_blocked": True,
                },
                indent=2,
            )
        )
    finally:
        shutil.rmtree(ROOT / "artifacts" / namespace, ignore_errors=True)
        shutil.rmtree(ROOT / "reports" / namespace, ignore_errors=True)


if __name__ == "__main__":
    main()
