# Data and claim boundary

## Runnable smoke evidence

The default pipeline generates deterministic synthetic courier-day events. Its metrics demonstrate
pipeline integrity, RCOT mechanics, calibration, triage evaluation, API compatibility, and tests.
They are not real-world delivery performance claims.

## LaDe

LaDe is the intended predictive benchmark. The repository does not redistribute it. Users must
normalize official city/scenario files to the schema in `docs/lade_adapter.md`. All features must
satisfy `feature_available_at <= query_time`.

## Amazon Last Mile Routing Challenge

Amazon challenge data are reserved for offline route and time-window replay. They are not used as
if they contained observed stop-arrival labels. Respect the dataset's CC BY-NC 4.0 terms.

## Allowed claims

- Offline ETA and interval metrics on identified datasets.
- Capacity-constrained capture of observed or derived tail-risk workload.
- Simulated utility only under explicitly documented action-effect assumptions.

## Disallowed claims

- Actual Amazon production deployment or internal data use.
- Actual delay prevention without treatment/outcome evidence.
- Individual conditional coverage guarantees under arbitrary distribution shift.
