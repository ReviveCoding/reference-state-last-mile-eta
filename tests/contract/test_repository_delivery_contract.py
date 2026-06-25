from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_github_workflow_uses_current_official_action_majors() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert "actions/upload-artifact@v6" in workflow
    assert "make package-check verify-manifest" in workflow
    assert "docker build" in workflow
    assert "make repro-check" in workflow
    assert "make coverage" in workflow
    assert "make sbom" in workflow


def test_docker_runtime_is_non_root_and_release_ready() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "AS builder" in dockerfile
    assert "AS runtime" in dockerfile
    assert "--require-release-pass" in dockerfile
    assert "USER appuser" in dockerfile
    assert "/ready" in dockerfile
    assert "PYTHONPATH=src python scripts/run_pipeline.py" not in dockerfile
    assert "python scripts/run_pipeline.py" in dockerfile


def test_make_release_uses_guarded_orchestrator() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "release-preflight:" in makefile
    assert "release-bootstrap:" in makefile
    assert "scripts/release.py" in makefile
    assert "scripts.normalize_sdist" in makefile
    assert "scripts.verify_build_reproducibility" in makefile
    assert "ruff format --check" in makefile
    assert "--cov-fail-under=70" in makefile


def test_classical_pipeline_does_not_import_optional_torch_at_module_import_time() -> None:
    import ast

    source = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = [
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    assert "reference_eta.models.hsg_eta" not in top_level_imports
    assert "optional 'advanced' or 'gpu' extra" in source


def test_release_chain_verifies_bitwise_distribution_reproducibility() -> None:
    source = (ROOT / "scripts" / "release.py").read_text(encoding="utf-8")
    assert '"Normalize source distribution"' in source
    assert '"Distribution reproducibility"' in source
    assert "scripts.verify_build_reproducibility" in source


def test_release_native_model_steps_use_forced_cli_exit_and_polling_watchdog() -> None:
    release_source = (ROOT / "scripts" / "release.py").read_text(encoding="utf-8")
    assert "process.poll()" in release_source
    assert "--force-process-exit" in release_source
    train_source = (ROOT / "scripts" / "train_hsg_eta.py").read_text(encoding="utf-8")
    pipeline_source = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    assert "os._exit" in train_source
    assert "os._exit" in pipeline_source


def test_release_manifest_excludes_ephemeral_coverage_database() -> None:
    source = (ROOT / "scripts" / "build_release_manifest.py").read_text(encoding="utf-8")
    assert '".coverage"' in source


def test_ci_has_cross_platform_python_matrix_and_concurrency_benchmark() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "windows-latest" in workflow
    assert 'python-version: ["3.11", "3.13"]' in workflow
    assert "python scripts/tasks.py smoke" in workflow
    assert "api_concurrency_benchmark.json" in workflow
    assert "cancel-in-progress: true" in workflow


def test_security_automation_and_dependabot_are_configured() -> None:
    codeql = (ROOT / ".github" / "workflows" / "codeql.yml").read_text(encoding="utf-8")
    dependabot = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    assert "github/codeql-action/init@v4" in codeql
    assert "github/codeql-action/analyze@v4" in codeql
    assert "security-events: write" in codeql
    assert "package-ecosystem: pip" in dependabot
    assert "package-ecosystem: github-actions" in dependabot


def test_windows_runbook_and_powershell_wrapper_exist() -> None:
    runbook = (ROOT / "docs" / "WINDOWS_RUNBOOK.md").read_text(encoding="utf-8")
    powershell = (ROOT / "scripts" / "run_release.ps1").read_text(encoding="utf-8")
    assert "python scripts/tasks.py release" in runbook
    assert "python scripts/tasks.py release" in powershell
    assert "make` is not required" in runbook


def test_release_manifest_excludes_active_lock_files() -> None:
    source = (ROOT / "scripts" / "build_release_manifest.py").read_text(encoding="utf-8")
    assert '".locks"' in source


def test_release_includes_locking_and_api_load_evidence() -> None:
    source = (ROOT / "scripts" / "release.py").read_text(encoding="utf-8")
    assert '"Locking and publish recovery"' in source
    assert '"Concurrent API benchmark"' in source
    assert "locking_recovery_report.json" in source
    assert "api_concurrency_benchmark.json" in source


def test_release_attestation_binds_sbom_to_wheel() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-attest.yml").read_text(encoding="utf-8")
    assert "uses: actions/attest@v4" in workflow
    assert "subject-path: dist/*.whl" in workflow
    assert "sbom-path: reports/sbom.cdx.json" in workflow


def test_release_locking_verification_uses_forced_process_exit() -> None:
    source = (ROOT / "scripts" / "release.py").read_text(encoding="utf-8")
    locking_section = source.split('"Locking and publish recovery"', maxsplit=1)[1]
    locking_section = locking_section.split('"Deterministic replay"', maxsplit=1)[0]
    assert '"--force-process-exit"' in locking_section


def test_release_and_source_fingerprints_exclude_mypy_cache() -> None:
    release_source = (ROOT / "scripts" / "build_release_manifest.py").read_text(encoding="utf-8")
    handoff_source = (ROOT / "scripts" / "build_release_candidate_handoff.py").read_text(
        encoding="utf-8"
    )
    clean_source = (ROOT / "scripts" / "verify_clean_candidate.py").read_text(encoding="utf-8")
    assert '".mypy_cache"' in release_source
    assert '".mypy_cache"' in handoff_source
    assert '".mypy_cache"' in clean_source


def test_release_manifest_excludes_dynamic_qualification_outputs() -> None:
    source = (ROOT / "scripts" / "build_release_manifest.py").read_text(encoding="utf-8")
    assert "qualification_manifest.json" in source
    assert "release_bundle_manifest.json" in source
    assert "clean_candidate_validation.json" in source
    assert "local_qualification_summary.json" in source
