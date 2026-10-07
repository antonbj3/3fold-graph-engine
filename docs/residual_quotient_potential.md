# Residual-directed resistance bounds

`certified_cross_resistance` can optionally construct its initial potential by
solving an exact quotient of the original positive-conductance graph. It starts
with sink-distance classes and separate terminal classes, then splits individual
vertices according to their residuals. Each split enlarges the feasible potential
space; exact reoptimization cannot increase its Dirichlet energy. The reciprocal
energy is a resistance lower bound, checked against the original graph.

```python
from graph_engine.certified_resistance import (
    certified_cross_resistance, verify_resistance_witness,
)

edges = [(0, 1, 1), (1, 2, 2), (0, 2, 1), (2, 3, 1)]
costs = {}
lo, hi, witness = certified_cross_resistance(
    edges, 0, 3, 16,
    initial_potential="residual_quotient",
    potential_splits=3,
    costs=costs,
)
assert verify_resistance_witness(edges, 0, 3, lo, hi, witness)
```

The existing flow supplies the upper bound. Returned endpoints and witness
arithmetic remain exact fractions. Existing defaults and the `stop` callback
are preserved. `potential_splits` must be a nonnegative integer; it limits
initial quotient refinement separately from the subsequent flow update budget.
`costs` optionally records construction work and timings.

Quotients are capped at 64 classes. If the initial partition already exceeds
that cap, construction retains the original distance potential and reports
`quotient_fallback=True`. This also works for graphs with more than 64 vertices.
On a small graph, singleton classes can make the quotient the full problem.
Construction is charged to the existing soft time budget, which is checked
between operations; it is not a hard real-time deadline.

The default diagonal residual score is a heuristic for choosing the next
vertex. It can select a worse split than the more expensive Schur score exposed
by the low-level `residual_quotient_potential` function. Exact validity and nested
energy do not imply a speedup or efficient convergence. The low-level function
expects validated rows, adjacency, and sink distances from the connected port
component; it returns `(None, receipt)` when its class cap is exceeded.
