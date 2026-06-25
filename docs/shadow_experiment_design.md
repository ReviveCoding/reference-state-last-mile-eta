# Shadow experiment design

- Unit: courier-route snapshot, grouped by courier-day for analysis.
- Treatment in a future online test: reliability-adjusted triage queue.
- Control: current heuristic or q50 ETA ranking.
- Primary metric: captured tail workload at fixed review capacity.
- Guardrails: false-review burden, route coverage, low-support selection, latency, operator workload.
- Rollout: offline replay, shadow queue, limited operator-visible pilot, controlled rollout.
- Contamination risk: multiple snapshots from the same route and shared operator capacity.
- Stop/rollback: guardrail breach or persistent subgroup regression.
