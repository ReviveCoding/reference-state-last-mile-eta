from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
import uuid
from collections import deque
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from reference_eta import __version__
from reference_eta.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "reference-state-last-mile-eta"
PROJECT_REF = f"pkg:pypi/{PROJECT_NAME}@{__version__}"


def _root_requirement_names(pyproject_path: Path, *, include_dev: bool) -> set[str]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data["project"]
    requirements = list(project.get("dependencies", []))
    if include_dev:
        requirements.extend(project.get("optional-dependencies", {}).get("dev", []))
    return {canonicalize_name(Requirement(item).name) for item in requirements}


def _installed_dependency_closure(
    root_names: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queue = deque(sorted(root_names))
    seen: set[str] = set()
    component_by_name: dict[str, dict[str, Any]] = {}
    dependency_names: dict[str, set[str]] = {}
    while queue:
        canonical_name = queue.popleft()
        if canonical_name in seen:
            continue
        seen.add(canonical_name)
        try:
            dist = distribution(canonical_name)
        except PackageNotFoundError as error:
            raise RuntimeError(f"SBOM dependency is not installed: {canonical_name}") from error
        metadata = dist.metadata
        name = metadata.get("Name", canonical_name)
        package_version = dist.version
        purl = f"pkg:pypi/{canonicalize_name(name)}@{package_version}"
        license_text = metadata.get("License") or "NOASSERTION"
        license_text = re.sub(r"\s+", " ", license_text).strip()[:512] or "NOASSERTION"
        component_by_name[canonical_name] = {
            "bom-ref": purl,
            "type": "library",
            "name": name,
            "version": package_version,
            "purl": purl,
            "licenses": [{"license": {"name": license_text}}],
        }
        dependencies: set[str] = set()
        for requirement_text in dist.requires or []:
            requirement = Requirement(requirement_text)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            dependency_name = canonicalize_name(requirement.name)
            dependencies.add(dependency_name)
            if dependency_name not in seen:
                queue.append(dependency_name)
        dependency_names[canonical_name] = dependencies

    components = [component_by_name[name] for name in sorted(component_by_name)]
    dependency_graph = []
    for name in sorted(component_by_name):
        dependency_graph.append(
            {
                "ref": component_by_name[name]["bom-ref"],
                "dependsOn": sorted(
                    component_by_name[dependency]["bom-ref"]
                    for dependency in dependency_names.get(name, set())
                    if dependency in component_by_name
                ),
            }
        )
    return components, dependency_graph


def build_sbom(pyproject_path: Path, *, include_dev: bool) -> dict[str, Any]:
    roots = _root_requirement_names(pyproject_path, include_dev=include_dev)
    components, dependency_graph = _installed_dependency_closure(roots)
    component_by_name = {
        canonicalize_name(str(component["name"])): component for component in components
    }
    dependency_graph.insert(
        0,
        {
            "ref": PROJECT_REF,
            "dependsOn": sorted(
                component_by_name[name]["bom-ref"] for name in roots if name in component_by_name
            ),
        },
    )
    identity_payload = json.dumps(
        {
            "project_ref": PROJECT_REF,
            "components": components,
            "dependencies": dependency_graph,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    identity_digest = hashlib.sha256(identity_payload).hexdigest()
    serial = uuid.uuid5(uuid.NAMESPACE_URL, f"{PROJECT_REF}:{identity_digest}")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "component": {
                "bom-ref": PROJECT_REF,
                "type": "application",
                "name": PROJECT_NAME,
                "version": __version__,
                "purl": PROJECT_REF,
            },
            "properties": [
                {"name": "generated-by", "value": "scripts/generate_sbom.py"},
                {"name": "development-dependencies-included", "value": str(include_dev).lower()},
            ],
        },
        "components": components,
        "dependencies": dependency_graph,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic CycloneDX SBOM")
    parser.add_argument("--pyproject", type=Path, default=ROOT / "pyproject.toml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/sbom.cdx.json")
    parser.add_argument("--include-dev", action="store_true")
    args = parser.parse_args()
    sbom = build_sbom(args.pyproject, include_dev=args.include_dev)
    atomic_write_json(args.output, sbom)
    print(f"Wrote {len(sbom['components'])} SBOM components to {args.output}")


if __name__ == "__main__":
    main()
