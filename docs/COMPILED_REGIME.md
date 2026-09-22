# Compiled regime bridge (Graph → native Kernel)

`compiled_regime.py` re-expresses the finite `RegimePosterior` family as a natural-statistic
exponential family decoded by an **injected native kernel**. It is exact on the supported
subset and refuses the rest. It creates no graph store, compiles nothing at import time and
imports no cross-repo code.

```python
from graph_engine.regime_posterior import RegimePosterior
from graph_engine.compiled_regime import CompiledRegime
from kernel_engine.inference.finite_regime import FiniteRegimeKernel

kernel = FiniteRegimeKernel(build_dir)          # explicit caller injection
regime = RegimePosterior(0.0, 1.0, n_grid=16, p_two=0.05)
regime.add_claim(0.2, 0.4, +1, n_eff=3.0, reliability=0.8)
regime.add_probe(0.55, +1, reliability=0.9)

comp = CompiledRegime.from_regime(regime, kernel)
comp.p_plus(0.3)                 # predictive field
comp.potential_value()           # ∫ u(P₊) dx, bits (domain-length integral)
comp.task_risk(loss="log")       # normalized deployment risk on caller query weights
comp.log_evidence()              # log marginal likelihood
comp.family_mass()               # 0/1/2-transition posterior
comp.collision()                 # two-transition mass + Bayes factor
x, gain = comp.best_probe(0.95)  # exact hypothetical OED
comp.add_probe(x, +1, 0.95, evidence_id="instrument/run1/reading1")
```

## Supported subset and declared differences

* **Preserved**: partition (claim edges + `n_grid` cells), all-±/one-transition/two-transition
  hypotheses, per-family priors `p0/2`, `p1/(2Q)`, `p2/(2·C(Q,2))`, epsilon floors
  (`p0,p1 ≥ 1e−12`, two-transition `≥ 1e−300`), pointwise box likelihood, entropy/error
  potential, domain-length integration vs normalized risk.
* `claim_model="majority"` is nonlocal → `NotImplementedError`. **RegimeMarkov is a different
  family and is not a substitute.**
* Rejected instead of silently handled: reversed/zero-width/out-of-domain claims, out-of-domain
  probes, `sign ∉ {−1,+1}`, `reliability ∉ [0.5,1)`, `weight ≤ 0`, `n_eff < 0`, repeated
  evidence id.
* **Frozen partition**: `add_claim` is accepted only when its bounds are already edges; a new
  edge raises `FrozenPartitionError` (freeze-and-refuse, no silent repartition).
* `from_regime` reads only public fields; it never calls `_solve`/`_with_probes`/F.

## State and cost

The numeric sufficient state is `A_q, B_q` (2Q floats) on the fixed partition; `add_probe` is
O(1). Provenance/raw history grows with observations and is not the posterior. Decode is O(Q)
in the native transfer method; the same backend's `method="enumerate"` is the strong O(Q²)
baseline. `natural_statistics()` exposes A, B, the folded claim coefficients and the kernel
parameters; `d logZ/dA_q = E[S_q]`, `d logZ/dB_q = E[S_q²]`.

See `jobs/native_graph_bridge/RESULTS.md` for measured parity and benchmarks, and
`examples/regime_bridge/run_bridge.py` for an env-path example.
