# Changed representation after tree-voltage failure — distance quotient

Tree voltage worsens F1 cycle slack, and all tested warm starts are overwritten
by useful CG proposals by 12 iterations. The binding defect is non-tree chord
energy; removing zero-current tree edges does not make potentials harmonic on
cross-links. Keep the parent goal and change the potential's representation.

Group nodes by their ALREADY computed exact sink-distance, then minimize potential
energy exactly in this small group-constant subspace, with source/sink groups at
1/0. Aggregate conductances between groups; solve only the (k−2)-variable
Dirichlet quotient, capped at k≤64, otherwise fall back to distance. Original
clipped-distance phi belongs to this subspace, so the optimum energy D_q≤D_0
(proved under positive conductance), hence L_q≥L_0 at budget zero. Feasible flow
and every subsequent T1 update remain unchanged. Lift group potentials onto all
original vertices and use the original exact verifier. This is classical Ritz /
Galerkin graph aggregation, not a new variational theorem or full-graph solve.

This pays an EXTRA O(m) aggregation and small exact elimination as witness
construction; report quotient dimension, preprocessing wall time and rational
multiply-subtract count. No claim of equal TOTAL work from equality of refinement
attempts. Compare same 0,32,128,512 cycle attempts and 0,4,12 CG iterations, with
unchanged SP and ED flow starts, on all four F1 cases and exact references. Freeze
all earlier router choices. Aim: ≥10x F1 gap reduction at fixed refinement budget;
measure whole creation+verification at the same budget before any timing claim.
One containment breach stops. No 10x effect means the distance partition itself
loses decision-relevant within-shell geometry; successor must change partition.
