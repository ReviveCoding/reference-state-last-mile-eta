# V2 Final Confirmation: Event-State ETA Challenger

## Purpose

This document records the locked, offline final-confirmation result for the V2 event-state ETA challenger against the matched V1 clean baseline.

The evaluation uses whole-courier-day cohort isolation and a fully disjoint final cohort. The scope is offline ETA evaluation only; it does not establish production dispatch, delay-prevention, or customer-impact outcomes.

## Locked final-confirmation outcome

**Decision: HOLD_BASELINE**

The V2 challenger produced favorable point estimates, but the pre-specified clustered-bootstrap promotion gate did not confirm superiority because the reported interval crossed zero.

| Metric | V1 clean baseline | V2 event-state challenger | Change |
|---|---:|---:|---:|
| Validation MAE | 84.50 | 80.64 | -3.86 (-4.57%) |
| Final held-out MAE | 75.43 | 73.98 | -1.45 (-1.92%) |
| Final held-out median absolute error | 41.83 | 40.13 | -1.71 (-4.08%) |
| Final held-out tail MAE | 244.14 | 240.83 | -3.31 (-1.36%) |

The observed challenger-minus-baseline final MAE difference was -1.526. The reported clustered-bootstrap interval was [-3.375, +0.343], so the final promotion rule retained the baseline.

## Cohort and protocol

- Final normalized cohort: 1,716 courier-days and 39,941 normalized rows.
- Final held-out evaluation: 2,921 snapshots.
- Earlier pilot, confirmation, and V2 development cohorts had zero overlap with the final cohort.
- Both models used the matched LightGBM comparison and the same median-residual calibration workflow.
- Tail error and uncertainty were evaluated in addition to average error.

## Interpretation

The event-state representation is promising because it improved multiple held-out point metrics without worsening the tail metric. The result is intentionally reported as an observed improvement rather than statistically confirmed superiority. The project therefore demonstrates a reliability-first challenger-promotion workflow: preserve the baseline when the evidence threshold is not met.

## Reproduction boundary

Raw LaDe-D data, normalized data, trained model artifacts, prediction-level CSVs, and local runtime logs are intentionally excluded from version control. See [local-data.md](../local-data.md) for local dataset configuration.