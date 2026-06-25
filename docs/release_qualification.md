# Release Qualification

Qualification profile: STANDARD local with Q3 built-artifact clean-copy validation. GitHub-hosted Q4 validation is not claimed in this artifact.

## Verdict

CONDITIONALLY QUALIFIED for the local/Linux/source-ZIP/built-wheel scope. RELEASE QUALIFIED still requires exact-commit GitHub-hosted workflow success.

## Evidence summary

- Source fingerprint: `9ebda0ede508b50ce9030474bd9aebb1b52d6abd686062ac88240a43a2dac708`
- Python: `3.13.5`
- Local qualification script: `reports/local_qualification_summary.json` = PASS
- Tests: 138 passed
- Coverage: 76.4%
- Clean-copy candidate: PASS
- Repository manifest records: 275
- API benchmark: 64 requests, concurrency 8, errors 0, p50 82.04 ms, p95 127.90 ms
- Wheel SHA-256: `6594f4d421828f730335fec6baca8ae275b8ac9d0ac242b83d188da2692344aa`
- sdist SHA-256: `c9ff765e11d849327c6fcd79b13af33d38c766cd7ba41286b21a1650f7e941c3`

## Canonical command

```bash
python scripts/qualify_local.py
```

## GitHub status

GitHub configuration is present but not executed in this run. Do not claim GitHub PASS until the exact candidate source is pushed and the hosted workflow run succeeds.

## Claim boundary

This is offline synthetic/small-data, local operational release evidence. It does not prove production deployment, real customer-delay prevention, official full-dataset performance, Docker runtime success, Windows hosted parity, or CUDA throughput.
