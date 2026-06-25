# Leakage threat model

1. The same `courier_id + work_date` group cannot appear in more than one split.
2. Reference curves, bins, and support counts are fit on training rows only.
3. Tail thresholds are estimated from training outcomes only.
4. Conformal correction is fit on a dedicated calibration period.
5. Test outcomes are read only by evaluation and delayed-label reconciliation.
6. Future task order and finish times are targets, never snapshot features.
7. SQL/Python feature parity must be tested before release.
