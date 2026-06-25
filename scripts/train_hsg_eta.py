from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from scripts.run_pipeline import run

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/smoke.yaml")
    parser.add_argument("--require-release-pass", action="store_true")
    parser.add_argument(
        "--force-process-exit",
        action="store_true",
        help="Flush output and bypass native-library interpreter shutdown after completion",
    )
    args = parser.parse_args()
    summary = run(args.config, "gpu", output_namespace="gpu_smoke")
    print(json.dumps(summary, indent=2, default=str))
    release_passed = str(summary["release_gate"]).startswith("PASS_")
    if args.force_process_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if (release_passed or not args.require_release_pass) else 1)
    if args.require_release_pass and not release_passed:
        raise SystemExit(f"Release gate did not pass: {summary['release_gate']}")


if __name__ == "__main__":
    main()
