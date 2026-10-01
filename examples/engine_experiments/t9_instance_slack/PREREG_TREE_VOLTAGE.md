# Successor — tree-induced voltage, before measurement

The two-hub harmonic startup changes full-bank cycle gap by about 5% and leaves
CG12 unchanged; leaf support changes the full-bank gap by under 1%. This exposes
an initialization defect distributed across the shortest-path tree, not just the
largest hub. The exact full-bank factorization U/R=1.824 and R/L=777.862 diagnoses
the dual side independently of any changed candidate.

Replace the distance potential by the voltage induced by the SAME initial SP
flow on the SAME spanning tree: along the source-sink path phi=distance/P;
every off-path tree branch inherits its attachment potential. Thus all zero-flow
tree edges have zero potential drop. Non-tree chords still contribute their exact
energy, so no resistance approximation is used as a certificate. This classical
tree-induced voltage is related to Kelner et al. 1301.6628's tree/cycle framework;
no new theorem or solver claim. The new local measurement is whether it removes
F1's observed dual defect at fixed updates.

Construction uses parent traversal with memoization, O(n) visits instead of the
old O(n) clipped-distance copy; zero extra relaxation iterations. Flow, tree,
chord/coordinate order, precision and update budgets unchanged. Measure on all
four F1 cases at 0,32,128,512 cycle attempts and 0,4,12 CG iterations, against the
already measured unchanged SP/CG control and exact references. Include creation
and verification cost. Do not refit the frozen holdout router. Falsifier: any
containment violation stops; failure to obtain 10x gap reduction means the
initial potential representation alone does not remove the F1 barrier.
