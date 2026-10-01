# Typed triples and exact interface reference

`graph_engine.typed_throws` integrates the supplied field prototype with native
imports. Supporting modules preserve its port, operator, regime, composition,
signature and sampling contracts. Johnson T2 supplies `cover_triples`; NT2
branch B supplies the exact edge-native Schur construction and rational kernel.
There are no new model dependencies.

`Part` declares inputs/outputs (`Port`), operator tokens, unit dimensions,
categorical regimes, validity boxes, assumptions and provenance reach.
`evaluate_chain` checks adjacent links, every required input's typed supply,
all assumptions and the whole regime. Supply may come from an earlier part,
including a non-adjacent part, and must obey the existing unit-dimension and
port-kind link rules. One compatible adjacent link cannot satisfy a second mismatched
required input. The head's external inputs retain their declared types;
explicitly given inputs need no earlier producer. `closure_value` follows
directed AND inputs and refuses incompatible sets. Source inputs must be
explicitly given for closed execution.

`StructuralImage.value_Q` and `closure_value` are structural information models:
unit elasticities and reach-dependent noise are declared modeling choices. They
are not calibrated physical information or worker success probabilities. Raw
Gaussian rows can use an operator inversely or cancel correlated inputs;
directed closure is the appropriate gate for port execution.

`set_value_Q_bits(form, H, sigma, Q)` composes the native Gaussian target-value
chain rule without mutating the form. `form.best_set(candidates, k, Q=Q)` searches
all sets up to `max_sets` (default 100000); above this explicit limit it raises.
The unrestricted API retains greedy selection. Its cardinality approximation
guarantee applies to unrestricted Gaussian log-det with equal costs. With Q,
`per_cost=True` maximizes joint target bits divided by the total set cost.

```python
from graph_engine.next_actions import EngineState, next_actions

state = EngineState(form=form, candidates=candidates, triple_requests=[{
    "members": ["a", "b", "c"], "target": "task", "valuation": "value_Q", "Q": Q
}])
actions = next_actions(state)
```

`triple_bundle` also supports `closure` and `port_value_Q` with `typed_parts`,
`triple_given` and `triple_costs`. These uncalibrated prices use `structural_bits`
and appear in `Actions.other`. Gaussian `value_Q` uses ordinary bits. Applying
the latter validates all three values before updating the form; stale precision
is rejected. Applying a structural bundle records its combination/evidence;
it does not turn a construction report into a measured entropy drop.
`chain_throw_bundle(..., parts=...)` optionally gates a geometric path by its
typed parts; incompatible paths raise with the regime/missing-input diagnosis.

`Federation.inferred_links` retains a derived link when its endpoint pair is
already asserted; `InferredLink.contradicts` names compatible opposite claims.
`stress_points` returns `CYCLE` probes with complete witness claim IDs. Separate
trees per instance and active validity context prevent an incompatible shortcut
from hiding a cycle. The BFS is linear per context plus witness reconstruction;
context enumeration is explicitly bounded. Fundamental witnesses do not list
every cycle or solve the frustration-index optimization problem. An aggregate
CYCLE needs edge-specific follow-up claims through `add_graph`; `record` refuses
to treat one sign as resolving the whole witness.
The `next_actions` follow-up measures the probe's single explicit edge; `apply`
adds that claim at the measured point with the common instance/regime and keeps
the original witness IDs in its receipt. It does not remove the original cycle.

`draw_triples` returns declared-universe indices and exact inclusion probabilities.
`cover_triples(n, k, seed, budget)` constructs the reused greedy cover, truncates
to an optional prefix and independently permutes the points. `triple_pi` is the
conditional coverage probability for every triple over relabeling; worker
detection sensitivity requires a separate empirical measurement.

`GraphInterface.export_edges(n, edges, weights, shared)` preserves the exact
binary64 conductances in fractions before Schur reduction. `cross_resistance`
tests every nullspace mode before solving: infeasible injection returns infinity.
The sparse-L API automatically uses this edge-native path for combinatorial
Laplacians with at most 16 nodes. Larger/estimated interfaces retain a floating
spectral guard; extremely weak positive couplings require explicit `export_edges`.
Rational elimination is a reference operation with higher CPU/byte cost. The
boundary must have identical ordering in both exporters, and every interior
component must reach it. `nbytes()` includes the exact payload.

Pre-wave DENSE3 cards currently lack a common typed port/observation/geometry
bank. Newly produced relation edges (`feeds`, `same_operator`, `contradicts`,
etc.) also lack explicit derivative signs and shared object domains. These
fields cannot be filled from a PASS verdict. Provide canonical signatures and
signed same-domain claims before using the new API for prospective worker
ranking and semantic cycle evaluation.
