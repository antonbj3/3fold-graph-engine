# Successor preregistration — dual hub potential, before its measurements

Evidence that triggers this branch: initial full F1 exact R=7.127059649019;
SP and avoided-hub ED have identical gaps at 0,32,128,512 updates. ED packs
only one path. The unchanged update order reaches mostly irrelevant coordinates.
The lower endpoint, not necessarily the feasible flow, can dominate U/L.

Change the witness's potential construction, keep SP flow and the exact same
subsequent T1 schedule/budget. After distance initialization, visit the two highest
weighted-degree nonport nodes once, in descending degree; set phi_u to the exact
weighted neighbor average. Each such replacement decreases D exactly. Count and
report these two startup coordinate operations and every touched incident row;
they are additional witness-construction work, not invisible refinement attempts.
Compare raw distance and warmed potential at fixed 0,32,128,512 local attempts,
and at 0,4,12 CG iterations separately. Do NOT compare a local attempt to CG.
No speedup claim from attempt equality alone. One containment violation stops.
If it gives only a small gain, retain the counterexample and support-pruning
successor as the executed representation change; no further tuning on holdout.

F1 exact references are reused, never used to propose fields. Small synthetic
star counterexamples have an independent exact harmonic potential reference;
F1 effects remain separate from synthetic effects. This branch is mechanism
measurement, not a newly selected holdout router and not a new graph theorem.
