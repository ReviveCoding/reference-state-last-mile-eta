# Operations

This repository is an offline/local release-candidate benchmark for reference-state last-mile ETA, uncertainty, and operational triage. It does not require external credentials for the synthetic, LaDe-style, Amazon-shaped replay, API, package, and clean-copy qualification paths.

Canonical local qualification is:

```bash
python scripts/qualify_local.py
```

The service reads the published artifact pointer from `artifacts/current_bundle.json` and verifies bundle manifests before serving predictions. Failed or partial publication must not be treated as a valid serving bundle.

Unsupported as local evidence: actual GitHub-hosted runner success, Docker build success, cloud deployment, official full dataset performance, and CUDA throughput unless those paths are separately executed and recorded.
