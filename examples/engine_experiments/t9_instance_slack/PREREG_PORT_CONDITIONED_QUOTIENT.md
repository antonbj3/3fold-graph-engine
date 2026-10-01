# Port conditioning of the quotient — before measurement

Unconditioned distance quotient removes >100x local-attempt slack on the three
hard F1 cases, but still groups source with unrelated vertices at identical sink
distance. The partial lower bound can remain too low to certify nearby theta.

Split BOTH queried ports into singleton groups; keep all other vertices grouped
by exact precomputed distance. This expands the previous subspace by at most two
groups (sink's distance zero was already unique). Thus D_cond≤D_quot≤D_distance
and L_cond≥L_quot≥L_distance at initialization, with the same SP or ED flow. The
operation changes a boundary representation, not the update budget. Extra O(m)
aggregation and a ≤64-variable small exact solve remain charged as construction.

Four F1 cases, same 0/32/128/512 local attempts and 0/4/12 CG iterations; both
flow starts. Compare exact endpoints to all already saved controls. Diagnose
first signs at .8R/.95R/1.05R/1.2R; do not train a new holdout router. If the
lower side still has no meaningful sign gain, the remaining within-class graph
geometry is binding. Classical port-preserving graph aggregation; novelty claim
is only this measured causal correction and engine witness operation.
