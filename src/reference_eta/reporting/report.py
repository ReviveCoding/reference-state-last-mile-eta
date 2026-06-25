from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_markdown_report(
    path: str | Path,
    *,
    summary: dict[str, Any],
    model_metrics: pd.DataFrame,
    capacity_metrics: pd.DataFrame,
    drift_report: pd.DataFrame,
) -> None:
    path = Path(path)
    lines = [
        "# Reference-State Last-Mile ETA Smoke Report",
        "",
        "## Run summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Model scorecard",
        "",
        model_metrics.to_markdown(index=False),
        "",
        "## Capacity-constrained triage",
        "",
        capacity_metrics.to_markdown(index=False),
        "",
        "## Drift report",
        "",
        drift_report.to_markdown(index=False),
        "",
        "## Claim boundary",
        "",
        "- Predictive metrics are offline results on deterministic synthetic event-time data.",
        "- Capacity metrics measure at-risk workload identification, not actual delay prevention.",
        "- Intervention utility is not claimed without an explicit treatment-effect assumption.",
        "- LaDe and Amazon adapters are provided, but public data are not redistributed in this bundle.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
