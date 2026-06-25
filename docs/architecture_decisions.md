# Architecture decisions

## ADR-001: LaDe for ETA, Amazon for replay
Amazon challenge data provide route structure but not the realized event-time labels needed for the
primary ETA task. Predictive training and route replay remain separate evidence tracks.

## ADR-002: Strong classical champion before HSG-ETA
LightGBM and quantile LightGBM are mandatory release candidates. HSG-ETA is promoted only when it
earns complexity through predictive, reliability, decision, and latency gates.

## ADR-003: RCOT is support-aware
A reference time without support and dispersion information can create false confidence. RCOT is
therefore emitted with support, dispersion, OOD probability, and trust.

## ADR-004: Triage before intervention claims
The primary decision metric is at-risk workload capture under capacity. Intervention cost reduction
is simulation-only until treatment/outcome data exist.
