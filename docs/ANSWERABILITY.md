# Exact answerability before compression

`graph_engine.answerability` certifies a declared **linear** state/observation/query model using exact rational arithmetic. It does not infer scientific semantics from graph edges or prose. Queries and observation rows may be dense sequences or sparse dictionaries. Floats denote their exact binary value.

```python
from graph_engine.answerability import (
    LinearStateGraph, answerable_before_compression, max_compression,
    query_reparameterization, repair_compression,
)
graph = LinearStateGraph(2, {'x': [1, 0], 'y': [0, 1], 'sum': [1, 1]})
Q = {'x_minus_y': [1, -1]}
rank, lost_queries, witness = answerable_before_compression(graph, Q, ['sum'])
assert rank == 0 and lost_queries == ['x_minus_y']
# The collision delta has identical retained observation and different query.
assert witness['collisions']['x_minus_y']['query_difference'] != 0
optimal = max_compression(graph, Q)
assert optimal['minimum_kept'] == 2
repair = repair_compression(graph, Q, ['sum'])
assert repair['minimum_additional_observations'] == 1
```

`basis` optionally supplies columns of a known state map `x=Tz`; this is the exact nonorthogonal counterpart of J1's state quotient. All guarantees are on this declared admissible state span. Noise, nonlinear constraints, unknown laws and roundoff in executing a recovery are separate obligations.

For a compression matrix directly, pass `Projection(rows)` instead of record IDs. For each answered question, `witness['recovery']` contains exact weights on the kept observations. For each lost question, `witness['collisions']` contains an admissible coordinate/state delta with zero retained observations and a nonzero query difference. Zero and delta are two same-data/different-answer states.

The returned rank is `dim(row(QT) ∩ row(A_keep T))`. It may be positive even when every named query is individually lost: `[1,1]` and `[1,-1]` share a recoverable combination after keeping `[1,0]`. Do not replace the rank by the count of answered question IDs.

`max_compression` finds a global maximum deletion of observation records. Independent coordinate records have a direct support proof. General records require an increasing-cardinality exhaustive search; `SearchBudgetExceeded` means no optimum has been certified. Set `max_subsets=None` only when prepared to pay the combinatorial cost. Greedy inclusion-minimal sets are not global optima.

`query_reparameterization` builds a redesigned minimal linear sketch of dimension `rank(QT)`. `repair_compression` appends a minimal number of new exact scalar observations, equal to `rank(QT)-intersection_rank`. These two operations permit new rows, whereas record deletion does not.

`harmonic_graph` reuses `GraphInterface.export_edges` and its exact binary64 conductance semantics. It declares zero interior sources and arbitrary boundary potentials, and observes node potentials or edge potential drops. Conductances and the source law remain required metadata. Deleting these observations does not certify deleting an edge from the physical Laplacian.

The row-space/nullspace criterion and query-preserving compression are established mathematics. This module supplies exact scoped certificates and source adapters, not a new observability theorem.
