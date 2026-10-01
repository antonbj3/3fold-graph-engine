# Certified decisions before a full solution

`graph_engine.graph_interface.certified_decision(edges, a, b, theta, budget)`
returns a dict with `verdict`, `updates_used`, and `witness`. Conductances are
positive integers, Fractions, or their exact represented binary values.
Threshold comparisons and witness checking are rational and strict.

```python
from fractions import Fraction
from graph_engine.graph_interface import certified_decision, verify_decision_witness

edges = [(0, 1, 1), (1, 2, 1), (0, 2, 1)]
result = certified_decision(edges, 0, 2, Fraction(1), 100)
verify_decision_witness(edges, 0, 2, Fraction(1), result)
assert result["verdict"] == "LESS"
```

The shared T1 iterator stops immediately when its exact flow upper bound is
below theta or its potential lower bound is above theta. Equality gives
`UNDECIDED`. An update budget pays mandatory initialization; a
`ResistanceBudget(seconds=...)` is a soft total generation deadline,
including graph validation and hashing. Verification is separate and must
also fit the consumer's total budget. An incomplete startup yields [0, inf],
which decides no finite positive threshold.

The SHA256 binding covers the ordered, oriented rational edge rows, including
selfloops and parallel rows. The checker independently recomputes flow
conservation, potential normalization, both energies, query binding and the
verdict. Changing even an energetically irrelevant loop invalidates the old
certificate. Hash integrity does not authenticate who supplied a graph.

`certified_decisions(..., thetas, ...)` shares one state and witness across
several thresholds for the **same** graph and port pair. It returns `verdicts`
in input order and uses `verify_decision_batch`. All startup/checking costs
must be charged to that batch; amortization is appropriate only for a consumer
that actually needs those questions. The optional `initialization="bfs"`
uses a hop tree and endpoint-supported potentials instead of Dijkstra.

`certified_quadratic_decision(J, h, theta, max_updates)` supports strictly
diagonally dominant symmetric forms through an exact Gershgorin lower bound
alpha>0. For any rational proposed x,

```
lower = 2*h.T*x - x.T*J*x
upper = lower + ||h-J*x||^2 / alpha
```

bounds h.T*J^-1*h. `verify_quadratic_witness` proves alpha from J and rechecks
the two expressions. This includes positive offdiagonal precision forms
outside positive conductance Laplacians. It does not cover arbitrary PSD
forms or general log determinants without further spectral information.

T7 measurements found no independent 1% capability against the faster
fraction-free exact fullsolve. Near-threshold decisions usually cost as many
updates as a 10% interval. Graph resistance also does not certify semantic
claim incompatibility without a supplied theorem connecting those quantities.
