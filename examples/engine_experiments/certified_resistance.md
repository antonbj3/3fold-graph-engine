# Exact intervals before a graph solve

This integrates classical Thomson/Dirichlet certificates into the graph API.
It does not introduce a new resistance algorithm. For a passive undirected
network, supply oriented edge rows `(tail, head, positive_conductance)`:

```python
from fractions import Fraction
from graph_engine.graph_interface import (
    certified_cross_resistance, ResistanceBudget, verify_resistance_witness)

edges = [(0, 1, 1), (1, 2, 1), (0, 2, 1)]
lo, hi, witness = certified_cross_resistance(edges, 0, 2, 0)
assert verify_resistance_witness(edges, 0, 2, lo, hi, witness)
assert lo <= Fraction(2, 3) <= hi

# A deterministic number of local refinement attempts; same inputs and larger
# integer budgets produce nested intervals. Start construction is still paid.
lo, hi, witness = certified_cross_resistance(edges, 0, 2, 128)

# A total soft wall deadline, including input validation and construction.
# CG proposes fields; exact conservation and exact energies certify them.
lo, hi, witness = certified_cross_resistance(
    edges, 0, 2, ResistanceBudget(seconds=0.05, refinement="cg"))
assert verify_resistance_witness(edges, 0, 2, lo, hi, witness)
```

The endpoints are `Fraction` or positive infinity. Floats supplied as
conductances mean their actual represented binary values. Integers retain all
bits; `Fraction` inputs can represent arbitrarily weak positive bridges.
Self-loops have zero incidence; parallel rows are separate conductances.
Nodes are nonnegative integer IDs; isolated ports need not appear in an edge.

The witness has an exact unit flow (`+1` divergence at source, `−1` at sink)
and a potential with source-minus-sink drop exactly one. Upper energy is
`sum(f_e**2 / c_e)`; lower bound is
`1 / sum(c_e * (phi[u] - phi[v])**2)`. The checker recomputes every condition
against the original ordered edges. It also checks a separating support cut
for disconnected ports, yielding `[infinity, infinity]`, and `[0,0]` for the
same port. Changing the input graph requires rechecking the witness.

Refinement modes are `cycle` (default), `tree` (coordinate potential plus
exact tree correction), and `cg` (untrusted incomplete preconditioned CG plus
exact tree correction). The candidate calls no full solve, pseudoinverse,
Schur exporter or matrix factorization. CG residuals never establish safety;
a numerical breakdown leaves the retained exact interval. The separate
`exact_edge_resistance` helper uses NT2 rational elimination as a modest-graph
reference and is never called by the certificate builder.

Time budgets are soft. Deadline checks occur between units of work, and a
rational operation, a NumPy proposal setup or final witness copying can
overrun. `witness.elapsed_seconds` includes final copying;
`witness.startup_seconds` marks the first finite path/potential construction.
If even that cannot finish, the return is `[0,infinity]`, kind
`uninformative`, and may be only partially input-validated. This universal
interval is conditional on the positive-graph contract. The checker fully
validates the original graph even for that return. Verification, imports,
input acquisition and witness serialization have additional measured cost.
Tiny budgets cannot promise a useful interval before an edge scan.

Witnesses contain private node fields and edge flows. They can be checked
locally and only the scalar interval released. This is not a zero-knowledge
proof, an authenticated export, or a multi-owner interface protocol. Existing
`GraphInterface.export_edges` remains the exact full-solve exporter; invoking
it inside this operation would defeat the incomplete-solve capability.
