# Certified answers over a family

A certified answer to one question is rarely cheaper than an uncertified one. A certified answer to a
*family* of questions can be, because the expensive part is a basis that the whole family shares. Two
independent implementations in this repository's surrounding lanes measured the same cost law from
opposite ends, so it is stated once here, with the two gates that decide when it applies.

## The cost law

Three quantities, not one factor:

| Quantity | What it is | Why it matters |
|---|---|---|
| `start` | cost of the first certificate, including graph validation, hashing and witness verification | paid once per basis, and it is what kills single-question use |
| `step` | cost of each further member once the basis exists | usually three orders of magnitude below `start` |
| `reach` | how many members one basis covers before the certificate stops binding | sets the achievable factor; it is a property of the problem, not of the implementation |

For a family of `m` members needing `b = ceil(m / reach)` bases, the amortized cost per answer is
`(b · start + m · step) / m`, against an uncertified control cost `c` per answer. Writing `reach = m/b`,
the factor is

    factor = c / (start / reach + step)

so `m` cancels: the family's size does not set the factor, `reach` does. Reporting a single factor
without `reach` is therefore not reproducible — the same implementation gives 10.76× and 47.1× on the
same `start` and `step`.

Checked against three families whose only common parameter is `step`. `start` and `c` are recomputed
here from each family's raw wall-clock totals (`(total − m · step) / b` and the control's own median),
not taken from the reported factors:

| family | `m` | bases | `reach` | `start` per basis | `c` | factor from the law | reported |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1, against control B | 125 | 5 | 25.0 | 0.452–0.514 s | 0.213 s | **10.76×** | 10.76× |
| S1, against the strongest control | 125 | 5 | 25.0 | 0.452–0.514 s | 0.11–0.13 s | **5.56–6.57×** | 4.6–8.1×, median ~7× |
| S2 (as first reported) | 125 | 103 | 1.21 | 0.173–0.219 s | ≤ 0.133 s | ≤ 0.82× | 0.83× |
| S2 (corrected admission) | 125 | 125 | **1.00** | 0.173–0.219 s | ≤ 0.133 s | **≤ 0.68×** | — |
| S3 | 102 | 30 | 3.40 | 0.051–0.066 s | ≤ 0.035 s | **≤ 1.97×** | 2.0× |

S2's row moved after review, and the reason is the span rule again. Its 103 bases were admitted by a
check that did not identify the moments, the same defect the field lane's own review found in its three
isotropic controls. Under the corrected admission S2 admits nothing from any of its 125 bases, so
`reach = 1.00`; allowing fresh answers for the 48 targets no basis covers gives an optimistic
`reach = 1.4205`. The law then gives 0.68× and 0.96× instead of 0.82×. S2 is a loss under all three
readings, which sharpens rather than weakens the point: `reach` is what decides, and a `reach` near 1
cannot be rescued by any `start` or `step`. Source: `J3B_COVER_TRANSFER`, reviewed in
`root_review/J3B_COVER_TRANSFER/RESULTS.md`.

A second measured caution belongs with it, because it is the opposite of what a covering design is
usually sold on. **Coverage is not low variance.** On the same consumer, a randomized design that
covers 100 % of the 300 target rows within the real 360-cell budget, against 66.18 % for the sampling
it replaces, carries **17.34× higher** global Horvitz-Thompson variance, because the unchosen points
have inclusion probability zero. Only coverage plus population-proportional extra draws turns that
into 2.07× lower variance, and then at 5 760 cells rather than 360 — sixteen times the budget. A design
chosen for coverage and read for a mean is a trap.

The S1 control matters more than the law does, and it moved under review. Control B used a fixed mesh
whose widths were 1.26–1.43 % and so left the declared 2 % tolerance unused. A two-level control at the
same tolerance — `n = 3`, then `n = 4` for the 29 of 125 that miss 2 % — is 1.6–1.9× cheaper than B, and
against it S1 is **4.6–8.1×, median about 7×**, not 10.76×. The law is unchanged; `c` was measured
against a control that was not the strongest one available. S2 and S3 were only compared against B, so
their `c` is an upper bound and their factors are upper bounds too.

The three families differ in mesh, in basis cost by an order of magnitude, and in control by a factor
of six, and the one formula reproduces all three medians from the raw totals. S2 is the informative row: at `reach = 1.21`
no `start` and no `step` can produce a win, because almost every member pays for its own basis. That is
what a family-basis method has to move; nothing else in the law is adjustable.

## The two measured instances

**Guaranteed scalar over a parameter family** (field lane `SOL_FALT_VERIFIERFONSTER_20261001`,
`raw/gate_summary.json`, `raw/pilot_share_v1.json`). Strength tolerance frozen at 2 % width:

The three rows above are that lane's S1, S2 and S3 gates. Two further measurements from it:

| | `m` | bases | per answer | control | factor |
|---|---:|---:|---:|---:|---:|
| share pilot, no width acceptance | 125 | 1 | 4.52 ms | 213 ms | 47.1× |
| single answer, no family | 1 | 1 | — | — | 7.7× |

The pilot's 47.1× is the ceiling one basis would give, not an achieved factor; its shared fraction is
`U(2) = 0.998` and `U(64) = 0.945`. The single answer's 7.7× has 66 % of its cost in the solve itself.
`step = 0.46 ms` is measured on the S1 pilot only; on S2 and S3 it is not separately measured and is
negligible in both.

**Certified decisions over a threshold family** (`certified_decision`, lane
`T7_DECIDE_WITHOUT_SOLVING`, reviewed in `root_review/T7_DECIDE_WITHOUT_SOLVING/RESULTS.md`).
Two-sided Thomson/Dirichlet bounds on a port resistance, stopping as soon as `lo > θ` or `hi < θ`:

| | result |
|---|---|
| single question at 1 % of full-solve cost | 0 of 680 decided; graph hash and witness verification consume the budget |
| same runs without that fixed overhead | 60 of 340 decided |
| 17 thresholds sharing one state and one witness, `n = 144` | 17 of 17 decided in 53.18 ms, inside a shared 17 % budget |
| median decision-updates / gap-updates over 216 cases | 0.7162 |

The median of 0.7162 refuted the hypothesis that deciding a sign is generally cheaper than shrinking
the gap. It was the wrong statistic: half the θ grid lay within 5 % of the value, which forces a median
near one. The dependence is the result.

## Gate 1: distance from the question to the answer

A certified bound decides a threshold only while the threshold is far enough from the value. Measured
on the same 216 cases, median decision-updates / gap-updates by `θ/R`:

| `θ/R` | 1/2 | 4/5 | 9/10 | 19/20 | 99/100 | 101/100 | 21/20 | 6/5 | 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ratio | 0.036 | 0.429 | 0.667 | 0.833 | 1.000 | 1.167 | 1.000 | 0.580 | 0.199 |

Below `|θ − R| / R ≈ 0.05` there is nothing to win. The field lane's width tolerance is the same gate
in its own units: a variant far from its basis leaves the certificate's reach. `reach` and this
distance are one quantity seen twice.

## Gate 2: corners are only valid under per-coordinate monotonicity

Both instances evaluate a guaranteed quantity over a box. Enumerating the box's corners is exact for
some quantities and wrong for others, and the dividing property is monotonicity in each coordinate
separately.

Exact: the largest total variation between `softmax(base)` and `softmax(base + d)` over `d ∈ [l, u]`.
Since `TV = max_S Σ_{i∈S} (q_i − p_i)` and `Σ_{i∈S} q_i` increases in `d_i` for `i ∈ S` and decreases
in `d_j` otherwise, the maximum sits at the corner `d_i = u_i` (`i ∈ S`), `d_j = l_j` (otherwise). For
16 tokens all 65 536 corners are enumerable, which is how the meter's slack was measured at 2.086×
against the exact optimum (`root_review/T3_WITCERT_EXTERNAL/`).

Wrong: the field lane's contact energy over a stiffness box has its minimum in the interior, and all
16 corners miss it; 16 random draws inside the box missed 55 % of the true interval.

A corner sweep without an argument for per-coordinate monotonicity is a sample, not a bound. The
degenerate case is part of the same gate: a box of zero width still has a non-zero centre, so a
point-box can carry a non-zero value. Returning zero there was a real defect, found by
`T6_PARAMETRIC_METER` on `l = u = (0, 1)`, where the true value is `(e − 1) / (2(e + 1)) = 0.2310585786`.

Monotonicity is one sufficient condition. There is a second, with a narrower reach that has to be
stated, because it is not an alternative for arbitrary functions. The field lane's stiffness-box
theorem (`STYVHETSINTERVALL`) requires the quantity to be a **rational function whose numerator and
denominator are both multiaffine** in the box coordinates, that is of degree at most 1 in each `d_i`.
When the denominator -- there the corner determinant -- has the same non-zero sign at every corner it
cannot vanish inside the box, and then the whole value set lies in the convex hull of the corner
values. The multiaffine structure is the load-bearing premise, not the sign test.

It holds for fixed slip directions, where the solution is linear in each `d_i`. It does not hold for
the contact energy, which is quadratic in the solution, nor for the non-linear Coulomb model -- which
is why that energy's minimum sits in the interior and all 16 corners miss it. Nor does it hold for the
total variation above, where `exp(d_i)` is not affine; there monotonicity is the right gate. Neither
condition subsumes the other.

So the gate reads:

- **(a)** establish monotonicity in each coordinate, or
- **(b)** establish that the quantity is rational with multiaffine numerator and denominator, and that
  the denominator has one sign at every corner, or
- **(c)** treat the sweep as a sample.

The two halves of (b) do different work, and the distinction generalises past this theorem: **the
structure requirement classifies the quantity, the sign test classifies the instance.** Every routing
rule in this document has that shape. `pmax` sorts instances of one quantity; what licenses routing at
all is a property of the quantity, established once. A cheap per-instance test without a per-quantity
licence is a sample with a number attached.

## Gate 3: green controls check only the directions their rows span

A set of controls that all pass validates a linear, or linearised, model only in the directions their
coefficient rows span. A question outside that span is unchecked however many controls are green. The
necessary condition is `rank ≥ the number of free parameters the question depends on`.

The minimal case (field lane, `SOL_FALT_VERIFIERFONSTER_20261001` REVIEW.md §F2, L-prism P2): the
compared bounds are linear in `G_ii` and `Q_ii`, with coefficient rows `(det / s_i²)` and `(s_i² / det)`.
Three isotropic controls `diag(c, c, c)` for `c = 7/8, 9/8, 5/4` give rows all proportional to
`(1, 1, 1)`, so they span the trace direction only. Forging the moments by swapping the y and z axes in
both `G` and `Q` leaves the trace untouched, passes all three controls, and the admitted entry then
answers `[1.7657, 1.8589]` at `s = (1, 5/4, 3/4)` where an independent certificate gives
`[1.8850, 1.9089]` — disjoint.

This gate and gate 2 are the same question asked twice: what does a finite set of evaluations actually
determine. Monotonicity makes corners sufficient for finding an extremum; spanning makes controls
sufficient for admitting a model.

Applied here, the rule is not only about field meshes. The typed gate on federated claims
(`Federation.inferred_links(typecheck=True)`) decides four directions — unit dimension, unknown unit,
scale, and statement kind — and its first test covered three of them. Running a row per direction is
what exposed that `value` was being dropped at ingest exactly as `units` and `scales` had been, so the
sign-versus-number check could never fire while reading as covered in the receipt. A gate that cannot
fail is worse than no gate.

## Gate 4: measure dominance before building a selector

A selector between two sound bounds can never beat their pointwise minimum. So before routing by any
instance statistic, measure whether one branch dominates the other, and measure the **ceiling** a
perfect selector would reach rather than the performance of the selector in hand.

Measured on 185 808 evaluation cells per bit width and witness, three tolerances, two witnesses: routing
the total-variation meter on `pmax` gives Δ = 0 pp with a paired 95 % interval of [0, 0] against the best
single meter in all eighteen combinations. The ceiling is the decisive number: a perfect selector between
the two branches would take **six additional cells out of 265 104**, 0.002263 pp, in its best
combination, and 0 or 1 in every other. No statistic can reach a 0.5 pp criterion on that data, so the
question is closed rather than merely unanswered.

The looseness this was meant to exploit is real and persists in every prefix regime — it is simply not
exploitable for coverage, because the expensive branch already dominates. What routing can still buy is
**cost at identical decisions**: gating on the range theorem `TV ≤ tanh((max u − min l)/4)`, with the
threshold derived from the tolerance rather than fitted, reproduced the cascade's pass decisions exactly
at **24.69×** less meter time [22.73, 26.68], and 33.28× against range plus the full check.

So the gate reads: establish dominance first. If one branch dominates pointwise, the minimum is free and
a selector can only buy cost; if neither dominates, measure the oracle ceiling before choosing a
statistic. Reporting a selector's own result without its ceiling cannot distinguish a weak statistic
from an unreachable target.

## Gate 5: the adjustable quantity is the deciding operation, not the solver

Two lanes on unrelated substrates measured the same shape on the same day, each after a direction
built on the solver side had been exhausted.

A certified Dirichlet lane spent three rounds on the solver: a quotient potential as a warm start, then
as a deflation subspace, with spectral mode certificates for the cluster. All of it was outperformed by
leaving the solver alone and changing **when the exact check runs**. Generating 160 unreliable numerical
proposals and converting only the final potential — one exact `Fraction` energy scan instead of 160 —
gives plain cold CG excess **3.6245110366905793e−26 in 0.945–1.182 s**, meeting both the sign and the
precision requirement in 3/3, where the best solver-side variant needed 7.9 s to reach 1.12e−19. The
quotient still works; it is 5.5 % better than cold, inside the repetition spread. The diagnosis that
had motivated the whole direction — a warm start destroying CG's superlinear phase — was wrong: the
phase was not lost, it arrived later, behind a verification cost that dominated the budget.

A total-variation meter lane split its own factor the same way. Sharing the MPFR preparation across
calls, with the deciding operation unchanged, gives **1.123–1.289×**. Changing the deciding operation
on the same sharing gives **1.419–5.919×**, and the two together 1.635–7.611× at τ = 0.20. The
preparation was the part that looked expensive; it was not the part that was adjustable.

The cost law above says `reach` is the only adjustable quantity in `factor = c/(start/reach + step)`.
These two measurements say where `reach` lives in practice: in the operation that decides, not in the
one that computes. So the gate reads: before optimising a solver, a basis, a preconditioner or a shared
preparation, measure what the **deciding and verifying** operations cost in the same budget. If
verification is a fixed multiple of iterations, the cheapest correct change is to run it once.

A caveat the lanes themselves recorded, and which this gate inherits: both measurements are on one
substrate each, against thresholds that are diagnostic rather than native, and a failed construction on
the solver side never proves that no solver-side construction works.

## Gate 6: check the quantifier, not only the quantity

A requirement declared over a set and evaluated at a sample of that set is not a measurement error. It
is a different statement, and the gap can decide every instance in the set.

A declared spring-graph fixture requires transmission > 0.30 over the band ω ∈ [0.35, 0.61] rad/s. Its
own code carries `FREQS = (.38, .58, 1.35, 1.7)` and evaluates the requirement at the two in-band
frequencies. All five surviving designs pass at both sampled points, and all five fail over the band:
the best, mask 3695, reaches a band minimum of **0.29905929** against the floor of 0.30, short by
0.3136 %. At the sampled frequencies the same design reads 0.361421 and 1.245276, comfortably above.
The fixture's own band scan already recorded zero valid designs at this threshold over a 105-point
grid per window, so the sample and the declaration had been disagreeing in the same directory.

This is Gate 1's shape — distance from the question to the answer — applied to the quantifier rather
than the quantity. The quantity was computed correctly every time. What differed was *over what* it was
required to hold.

A caution on the cheap repair: on a 20 001-point grid the minimum sits at the left endpoint ω = 0.35
for all five masks, which invites the conclusion that one exact rational evaluation at ω = 7/20 settles
the requirement and no root isolation is needed. The transmission is **not** monotone on the band — it
dips and returns without going below the endpoint value — so the endpoint being the band minimum is a
grid observation, not a certificate. What follows is weaker: the endpoint gives the candidate value, and
what remains is a one-sided band bound, with no interior extremum to locate. Cheaper than isolating a
critical point, and not free.

So the gate reads: before computing a quantity against a threshold, read what set the requirement is
declared over and what set the code evaluates. If they differ, the verdict is undetermined no matter how
exact the arithmetic is, and a sample that passes is not evidence that the declaration holds.

### The census: this is not an exception

Measured across all 175 requirements in the corpus. **167 carry both an identifiable declared domain and
an identifiable evaluated set**; 7 are point requirements with no declared domain and 1 has a domain with
no locatable evaluation. Of the 167, **101 (60.5 %) evaluate a strict subset of what they declare**.

| form | count | flips | holds | undetermined |
|---|---:|---:|---:|---:|
| finite sample of a continuum | 48 | 11 | 23 | 14 |
| grid over a continuum | 27 | 5 | 12 | 10 |
| subset of a finite set | 20 | 4 | 7 | 9 |
| boundary vs interior | 6 | 0 | 2 | 4 |
| **total** | **101** | **20** | **44** | **37** |

**20 verdicts flip.** The 44 that hold each do so on a stated basis — exact coverage, proved monotonicity,
a corner theorem, exhaustive enumeration, or witness sufficiency — and never on a bare sample. The corpus
declares its domains explicitly, often in a frozen pre-registration: one job records that a ratio "is the
*continuum* value" while shipping a seven-value grid.

### The asymmetry that makes the repair cheap

One exactly evaluated rational point inside the declared set refutes a universal claim. No number of
points establishes one. So the flip risk sits almost entirely on requirements asserted to be **met**, and
refuting a false MET needs neither monotonicity nor root isolation — just one exact point.

Measured on the fixture above: all 57 point-feasible designs are exactly violated, 47 at ω = 7/20 and 10
at ω = 13/10, none left undetermined.

The asymmetry is also what the gate should be made of. Joining the census onto a 122-row threshold
ledger by node and refusing **MET** where the evaluated set is a strict subset with no stated basis —
while leaving VIOLATED alone, since a subset suffices to refute — blocks **11 of 46 MET verdicts
(24 %)** and none of the 51 VIOLATED. That is the allocation consequence in one number: the exploitable
error is concentrated in asserted-met claims, and each costs one exact point to test. The starkest case is the stopband the two-point test never
examined. Mask 3247 passes both sampled frequencies — 0.2253 at ω = 1.35 and 0.1891 at ω = 1.70 against
a 0.25 cap — and reaches **0.7503750725724 at ω = 41/26, three times the cap**, inside the declared band.
A design booked as meeting the requirement exceeds it threefold.

This also bounds the earlier caution correctly. That the band minimum sits at an endpoint on a grid is
indeed not a certificate — but only for establishing MET. For refutation the endpoint is not needed at
all: any single exact interior point that violates the bound closes the question.

## Gate 7: tell a weak selector from a degenerate criterion

Gate 4 says to measure the ceiling before building a selector. This is what happens when that is skipped,
measured three times in one day on three unrelated substrates.

| what was built | arena | measured ceiling |
|---|---|---|
| routing a total-variation meter on an instance statistic | 265 104 evaluation cells | **6 cells**, 0.002263 pp |
| choosing a solver against a direct sparse factorisation | 3 457 unknowns, 1–8 s budgets | the incumbent already answered in ~1 s |
| ranking next actions on a research graph | 3 939 rank rows | **1 row** |

The third is the clearest, because its own report concluded "superiority not shown" — which reads as a
verdict on the selector. Measuring the control instead reverses it. The selector discriminates: 904
distinct scores over 1 443 rows. The criterion does not: the unlock column is identically zero in all
1 443 rows, one distinct value, and the equally informed control carries one nonzero score across 3 939
rows with zero immediate unlocks. No selector can be superior where there is nothing to be superior at.

So the gate reads: when a selector fails to beat its control, measure the **control's own discriminating
power** before concluding anything about the selector. Two different failures look identical in a
head-to-head number:

- a **weak selector** — the control spreads, the selector does not;
- a **degenerate criterion** — neither spreads, because the arena has no signal to rank.

The cheap test is a spread count on the criterion column: distinct values, and the share that are
nonzero. One distinct value is not a close contest, it is an absent one. Reporting the head-to-head
without it cannot distinguish the two, which is the same defect Gate 4 names on the other side: a
selector's result without its ceiling cannot distinguish a weak statistic from an unreachable target.

A caution on the inference: a ceiling of one row bounds what a selector can win **on that arena with
that criterion**. It is not evidence that the question is unanswerable, and three instances in one
corpus do not make it a general rate.

## What is not claimed

No speed claim for a single certified answer: 7.7× for the field lane's single answer has 66 % of its
cost in the solve itself, and the decision instance is not cheaper at all near the threshold. No claim
that `reach` is predictable before it is measured; S2 and S3 show it collapsing with no change of
method. The cost law is arithmetic over three measured quantities, not a theorem about certificates.
