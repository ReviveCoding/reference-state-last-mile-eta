# Known limitations

## High-severity external qualification items

### Hosted Windows execution

Windows process-tree and task-runner branches are implemented and contract-tested, but an actual
Windows hosted runner has not executed this candidate. The next qualification must run Python 3.11
and 3.13 Windows jobs and retain their logs.

### Remote GitHub and Docker qualification

GitHub Actions, CodeQL, Dependabot, Docker build, and artifact-attestation workflows are configured
but have not run for the exact candidate source in this environment. E4 evidence requires the exact
candidate commit and retained hosted artifacts.

## Medium-severity evidence items

### Full public datasets

The bundled LaDe-style and Amazon official-shaped data are deterministic generated fixtures for
correctness and integration. They are not official full-data benchmark results.

### CUDA evidence

The HSG path is CUDA/AMP capable, but the candidate does not include actual RTX throughput, VRAM,
or CPU-vs-GPU comparison evidence. PyTorch determinism is only claimed within the tested software and
hardware environment.

### Intervention impact

The decision layer measures offline risk capture and simulated utility. It does not establish causal
production delay reduction or customer impact.

## Intentionally not implemented

- Additional ETA neural architectures
- Additional calibration families
- Forced RCOT promotion
- More decision policies without new operational evidence
- Cloud wrappers that cannot be executed in the candidate environment

These additions would increase complexity without addressing a release blocker.
