# T9 preregistration — Form B, before implementation

## Decomposition first

Desired capability: choose a useful two-terminal certificate from cheap instance
information, sufficient for many strict R−theta decisions, ideally removing an
order of magnitude of certificate slack without increasing refinement budget.

1. Edge e=(u,v,c), c>0 exact rational; incidence B; injection b=e_s−e_t.
2. Conserved feasible flow Bf=b; upper U=sum(f_e²/c_e).
3. Potential phi_s−phi_t=1; dual energy D=sum(c_e(delta phi_e)²);
   lower L=1/D. Cauchy-Schwarz plus Thomson/Dirichlet gives L≤R≤U.
4. Initial flow is one minimum-resistance path; initial potential is clipped
   resistance distance from sink. They fail separately: a hub can be essential
   for optimal flow while distance potential energizes irrelevant dangling edges.
5. A refinement attempt is exactly T1's alternating fundamental-cycle and
   potential-coordinate update. Forms share the same tree, coordinate/chord order,
   bit precision, and budget. Startup path packing is extra work and is charged.
6. Observable primary gap is (U−L)/L, matching T1 LARGE's cited numbers. Also
   report (U−L)/R_exact; the ratio's denominator can change conclusions.
7. Strict sign is certified only by U<theta or L>theta. Equality is undecided.
   R_exact only constructs diagnostic thresholds and checks containment;
   it never enters routing, candidate proposals or the certificate verifier.

All mathematical leaves DERIVED_UNDER_ASSUMPTIONS of a finite undirected passive
positive rational network. Document-edge conductances/physical interpretation
UNKNOWN. Stop decomposition at incidence and exact arithmetic because further
physical detail changes the model, not this algorithm's validity.

## Known work and narrow novelty

Verified originals (2026-10-01): Kelner/Orecchia/Sidford/Zhu,
https://arxiv.org/abs/1301.6628, cycle solvers and tree-dependent convergence;
Deweese et al., https://arxiv.org/abs/1609.02957, empirical cycle-toggling study;
von Luxburg/Radl/Hein, https://arxiv.org/abs/1003.1266, degree and spectral regime
can determine resistance approximations. No new variational principle, graph
solver, Menger theorem or general open-problem claim. The open local question is
whether free statistics predict the slack of THIS engine's witnesses and enable
routing on fresh adversarial graphs and F1. A correlation alone is no new theorem.

## Falsifiers and design

G1 fails if no eligible free statistic has |Spearman rho|≥0.5 at 128 updates.
Report all correlations, primary and exact-normalized gaps, development and
holdout, family strata, and costs. Correlation is distribution-dependent, not a
universal graph law. F1 and matched counterexamples challenge causal hub claims.
G2 requires better F1 gap at identical T1 update budget; startup is explicitly
charged and a same-update improvement alone is not a time-speedup claim.
G3 requires the frozen router's fresh-holdout mean log1p(primary gap) to be below
both single forms at 128 updates. Also report medians, tail, wins/losses, total
startup/refinement/check cost. Router is a single-threshold decision on one free
statistic, trained only on development; no holdout model reselection.
G4 measures first strict sign update for theta/R in {0.8,0.95,1.05,1.2}, stopping
at 512 attempts; censored values use 513 ONLY as a labeled ordinal diagnostic.
Report correlation and censoring, separately by threshold side/margin.

400 graphs: 10 families × 40 seeds, half development/half unseen holdout,
independent seeds. n≈12–36: path, cycle, grid, star, hub+ring, preferential
attachment, random sparse, dense, weak bridge/barbell, and parallel alternatives.
Weights include rationals and weak links; all connected, ports distinct. Add
port swaps, relabel/order permutations, and dangling-branch counterexamples.
Reference: independent grounded Fraction Gaussian elimination from T1 tests;
for large F1 use independent exact sparse star-mesh elimination if feasible,
with full exact backsubstitution residual verification. No float reference is
called exact. One containment violation stops the lane immediately.

Eligible cheap predictors: n,m; max degree/mean; degree Gini; max incident-edge
share; endpoint degree and weighted conductance; shortest resistance length;
length/sink eccentricity proxy (already in T1 Dijkstra); path hub exposure.
Initial U/L is a separately labeled certificate diagnostic, not independent
structural evidence. Exact Menger count and normalized spectral gap are measured
with cost; not called free without cost evidence. Exact resistance distance and
exact diameter are rejected as free predictors (solve/all-pairs traversal).

Witness forms: SP is unchanged T1. ED removes the maximum-degree nonport hub,
then greedily packs up to 8 edge-disjoint shortest-resistance paths, and splits
unit current with inverse-path-resistance weights. If no avoided-hub path exists,
fall back to SP. This is a bounded greedy packing, NOT the exact Menger count.
Potential and all subsequent updates are unchanged. Include a safe retained-SP
variant as dominance control: min-energy initialization cannot lose at budget 0,
but does not prove dominance after identical coordinate/cycle dynamics.
Budgets {0,32,128,512}; trace in one run, exact check every returned checkpoint
and each first decision. Startup and verification measured separately.

## Negative-outcome successor, preregistered now

If hub avoidance/router fails, change the representation to two-terminal support:
prune nonterminal degree-one branches and harmonic-extend potentials back. Such
edges carry exactly zero electrical flow. Compare unchanged SP, ED and pruned SP
at identical attempts and count all traversal/extension costs. On a star this
is a decisive counterexample: the hub is required, the distance-potential energy
on dangling leaves is the defect. If leaves are insufficient, use articulation
block support (known network reduction), charged separately and preregistered
before its run. Parent objective remains cheap predictive useful certification.

Resource rules: ≤2 threads; one compute job; pilot RSS first; >1min heavy_run;
large data in /mnt/games-240/research/claude24h_night/T9_INSTANCE_SLACK_GRAPH.
No edits to T1/T7, no pushes/merges/shared graph changes.
