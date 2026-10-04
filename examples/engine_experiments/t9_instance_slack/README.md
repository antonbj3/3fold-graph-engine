# T9_INSTANCE_SLACK_GRAPH

Research drivers live here; the isolated Graph branch is in `worktree/`.
Read RESULTS.md and the preregistrations before interpreting results. All data,
including local python-flint 0.9.0 for independent references, are under the real
en scratch-disk via `$T9_DATA_DIR`.
Source graphs and T1/T7 are read-only. No remote push or merge.

Drivers' tracked snapshots will be in `worktree/examples/engine_experiments/t9_instance_slack`.
Use `T9_LANE` to point copied drivers at this lane, and optional `T9_DATA_DIR` for
new outputs. At most two threads and one heavy_run job. New data directories
need python-flint for F1 references (`pip --target "$T9_DATA_DIR/vendor"`).
The candidate and checker use Python Fraction, regardless of reference backend.

Public optional starts in `graph_engine.certified_resistance`:
`initial_flow='shortest'|'edge_disjoint'|'retained_sp'|'routed'`,
`initial_potential='distance'|'hub_harmonic'|'tree_voltage'|'distance_quotient'|'port_conditioned_quotient'`.
Default behavior is exactly T1. Routed policy is frozen at maxdegree/mean ≤3.245.
Reduced quotient is capped at 64 groups; on a small graph with singleton groups
it can coincide with a full solve. Quotient work is startup work, explicitly
measured, and is not called a free statistic or uniform time speedup.
