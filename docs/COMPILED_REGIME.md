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
comp.log_evidence()              # generalized evidence, preserving legacy prior floors
comp.log_evidence(normalized_prior=True)  # normalize the floored prior first
comp.family_mass()               # 0/1/2-transition posterior
comp.collision()                 # two-transition mass + Bayes factor
x, gain = comp.best_probe(0.95)  # exact hypothetical OED
comp.add_probe(x, +1, 0.95, evidence_id="instrument/run1/reading1")
```

## Optional bounded-tail predictive readout

`approximate_p_plus_cells(max_error)` is an opt-in readout on the **same**
`CompiledRegime` owner. It returns a dictionary with a detached `p_plus` vector,
`mode` (`"native"` or `"flat"`), `error_bound_estimate`,
`numerical_certificate=False`, `owner_revision`, and anchor tail/likelihood-
contrast metadata. `max_error` must be finite with `0 <= max_error < 1`; zero
always requests the ordinary full native answer. The first call, or a call when
a current full native posterior is already cached, returns that full answer and
anchors the optional representation. `p_plus_cells`, `p_plus`, risk and exact
acquisition retain their existing native behavior.

The derived flat-family state uses `H=sum(kernel_A)` and predicts the same
`sigmoid(2H)` at every cell, conditional on zero transitions and the two equal
initial-orientation priors. At a full native anchor, the owner records
`epsilon=P(k=1)+P(k=2)` by **summing** both posterior family masses. After
common positive-likelihood probes, it accumulates
`sum(weight*abs(logit(reliability)))`. The ordinary-float odds envelope
`sigmoid(logit(epsilon)+contrast)` bounds the mathematical tail mass under the
declared finite family, fixed prior/claim semantics and common evidence. If it
exceeds the caller tolerance, or its inputs are unresolved/nonfinite, the
method falls back to full native decoding and refreshes the anchor. A rounded
zero tail with positive prior support is unresolved. No original hypothesis,
natural coordinate or evidence record is deleted.

This estimate is **not** an outward-rounded numerical certificate or a
guarantee for changed sensors, priors, claims, calibrations, state-dependent
action policies or future rounded computations. `add_claim` and
`set_claim_reliabilities` invalidate the anchor; a successful actual probe
advances its owner revision and contrast. Repeated same-revision answers return
copies, and a tighter tolerance cannot reuse a flat answer that fails it.
Nonunit probe weight is a positive power likelihood, not a newly normalized
physical observation channel. Hypothetical acquisition branches never become
actual anchors. The full native path remains the fallback and exact API.

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
  Near-equal coordinates are not silently snapped. Configuration is fixed; construct a
  new instance to change the prior/domain. Returned arrays are detached copies.
* `expected_gain` evaluates a **unit-weight physical probe**. A tempered likelihood with
  unchanged Bernoulli outcome probabilities is not an exact Bayes experiment; nonunit
  weights are accepted for evidence assimilation but refused for expected acquisition gain.
* `from_regime` reads only public fields; it never calls `_solve`/`_with_probes`/F.

## State and cost

The effective numeric sufficient state is `A_q, B_q` (2Q floats) on the fixed partition.
`add_probe` uses O(log Q) cell lookup followed by an O(1) numeric update. Static claim
coefficients, geometry, cached readouts and provenance require additional storage;
raw history grows with observations. Decode is O(Q)
in the native transfer method; the same backend's `method="enumerate"` is the strong O(Q²)
baseline. `natural_statistics()` exposes A, B, the folded claim coefficients and the kernel
parameters; `d logZ/dA_q = E[S_q]`, `d logZ/dB_q = E[S_q²]`.

See `examples/regime_bridge/run_bridge.py` for an env-path example. Tests compare the bridge
against the existing solver and reject use of its dense internal path. Research benchmark
artifacts are kept separately from this public source tree; there is no general-language
inference or model-quality speed claim in this module.
