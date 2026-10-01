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
| S1 | 125 | 5 | 25.0 | 0.452–0.514 s | 0.213 s | **10.76×** | 10.76× |
| S2 | 125 | 103 | 1.21 | 0.173–0.219 s | 0.133 s | **0.82×** | 0.83× |
| S3 | 102 | 30 | 3.40 | 0.051–0.066 s | 0.035 s | **1.97×** | 2.0× |

The three families differ in mesh, in basis cost by an order of magnitude, and in control by a factor
of six, and the one formula reproduces all three medians. S2 is the informative row: at `reach = 1.21`
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

## What is not claimed

No speed claim for a single certified answer: 7.7× for the field lane's single answer has 66 % of its
cost in the solve itself, and the decision instance is not cheaper at all near the threshold. No claim
that `reach` is predictable before it is measured; S2 and S3 show it collapsing with no change of
method. The cost law is arithmetic over three measured quantities, not a theorem about certificates.
