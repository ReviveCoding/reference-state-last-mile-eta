from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    specification = {
        "role_arn": args.role_arn,
        "bucket": args.bucket,
        "entry_point": "scripts/run_pipeline.py",
        "config": "configs/smoke.yaml",
        "output": "s3://<bucket>/reference-eta/artifacts/",
    }
    if args.dry_run:
        print(json.dumps(specification, indent=2).replace("<bucket>", args.bucket))
        return
    raise SystemExit(
        "Install and configure the SageMaker SDK in an AWS-authenticated environment, then "
        "replace this guarded scaffold with the approved organization-specific estimator."
    )


if __name__ == "__main__":
    main()
