#!/usr/bin/env python3
"""
guarantees.py — what the greedy probe rule is worth, and why the value rule cannot buy the collision probe.

Two results about `regime_posterior.RegimePosterior`, on its FIXED partition. Notation throughout:
cells c with widths w_c, hypotheses h with posterior π, F[h,c] = share of cell c where h is +,
marginal p_c = Σ_h π_h F[h,c], potential U = Σ_c w_c u(p_c) (u = binary entropy, or u = min(p,1−p)),
a probe in cell c with reliability r emits + with probability a_h = r F[h,c] + (1−r)(1−F[h,c]).

────────────────────────────────────────────────────────────────────────────────────────────────
1. THE GREEDY GUARANTEE, AND THE EXACT POINT WHERE IT LAPSES

The cell-wise identity (proved, test_cell_drop_is_mutual_information). Let Y be the answer of one probe
and Z_c ∈ {+,−} the sign at a uniform point of cell c, P(Z_c = + | h) = F[h,c], so Z_c ← h → Y is a
Markov chain and Y is binary. Then the expected drop of that cell's entropy term is exactly a mutual
information, which is exactly a Jensen gap of the binary entropy h₂ over the posterior of a_h:

    E_Y[ h₂(p_c) − h₂(p_c|Y) ] = I(Z_c;Y) = H(Y) − H(Y|Z_c) = h₂(E a) − E h₂(a),   a = a_{Z_c}     (1)
    Δ_U(x) = Σ_c w_c I(Z_c; Y_x).                                                                  (2)

(a) WHAT IS ADAPTIVE SUBMODULAR, AND WHAT IS NOT.

ADAPTIVE MONOTONE: yes, for every concave u, every reliability, every potential here. The posterior
is a martingale (E_Y[p_c|Y] = p_c on the fixed hypothesis space), so by Jensen every cell term drops
by ≥ 0 and Δ_U(x|ψ) ≥ 0 for every partial realization ψ. PROVED; test_adaptive_monotone_always.

ADAPTIVE SUBMODULAR: NO — for both potentials, with noisy answers, and the failure is structural, not
a corner case. The one-step gain at a cell is, by (1), a function of the marginal alone in the
noiseless-target limit: for a Bernoulli(p) target read through a BSC(r),
    γ_r(p) = h₂(p) − [ a h₂(pr/a) + (1−a) h₂(p(1−r)/(1−a)) ],  a = pr + (1−p)(1−r),                 (3)
which is symmetric about p = ½ and STRICTLY INCREASING on (0, ½) (proved: γ_r(p) = h₂(a) − h₂(r) by
(1), a is affine and strictly monotone in p, and h₂ is strictly increasing below ½; test
test_bsc_gain_increasing_below_half). Adaptive submodularity demands Δ(x|ψ) ≥ Δ(x|ψ′) for every
sub-realization ψ ⊆ ψ′ — pointwise in the OUTCOMES, not on average. A surprising answer in ψ′ ∖ ψ
moves p_c back toward ½, and (3) then makes the later gain strictly larger. So the counterexample is
generic rather than delicate: it needs only one probe whose answer contradicts the current belief.
EXACT INSTANCE (n_grid = 4, one claim [0, ½] with sign +, weight 3, claim reliability 0.9, probe
reliability 0.95, p_two = 0; test_adaptive_submodularity_counterexample). Entropy potential, probe in
cell 2: Δ(x | ∅) = 0.05112 bits, Δ(x | one − answer in cell 1) = 0.22849 bits — the inequality is
violated by a factor 4.47. Error potential, probe in cell 2: 0.004776 → 0.069488, factor 14.5; and
probe in cell 1: Δ(x | ∅) = 0 EXACTLY (no single answer can flip that cell's decision) while
Δ(x | one − answer in cell 3) = 0.007170, an infinite violation ratio.
This is the noisy-observation obstruction of Golovin, Krause & Ray (NIPS 2010): under noise the
version-space/equivalence-class objectives (EC²) are adaptive submodular, plain information gain is
not. Our U is neither — it is ∫ of per-cell binary entropies, a sum of concave functions of MARGINALS
— and it inherits the same obstruction through (1)+(3).
NOT adaptive submodular ⇒ the Golovin–Krause (JAIR 42, 2011) 1−1/e bound does not apply. The bound is
withdrawn for the noisy case; regime_posterior's docstring already restricts it to noiseless answers.

(b) `adaptive_submodularity_check` measures how often the inequality actually fails on random
sub-realizations A ⊆ B with answers drawn from the model. MEASURED, not proved (e33, 200 draws per
setting, probe reliability 0.95, |B| ≤ 4):
                                        entropy potential        error potential
    n_grid 6, 3 claims                  7.0 % violated, ≤ 3.6×   6.5 % violated, ≤ 3.8×
    n_grid 48, 2 claims, 12 candidates  11.0 %,         ≤ 3.0×   15.5 %,         ≤ 60.1×
Mean excess where violated: 0.052 / 0.028 bits (entropy), 0.021 / 0.017 (error). Violations are a
steady minority of histories — one in ten to one in six — not a measure-zero artifact.

(c) `brute_force_adaptive_optimum` enumerates every adaptive policy (a decision tree over the binary
answers, candidates = cell midpoints) by exact dynamic programming and returns the optimal expected
drop; `greedy_sequence_value` runs `best_probe`'s rule adaptively on the same candidate set. MEASURED
on 50 random small cases (n_grid 4–6 → 8–10 cells, budget 2–3, 2–3 claims, r = 0.95):
    entropy potential   mean greedy/optimal 0.9906, min 0.9600   (50/50 above 1−1/e = 0.6321)
    error potential     mean greedy/optimal 0.9770, min 0.9068   (50/50 above 1−1/e)
    greedy is strictly suboptimal in 33/50 (entropy) and 40/50 (error) cases — it is not optimal,
    it is close.
So: the inequality that would GUARANTEE 1−1/e fails on 7–16 % of histories, and greedy still lands
within 4 % (entropy) / 10 % (error) of the adaptive optimum on every case tried. The guarantee is
absent; the performance is not. That is a measurement over small partitions with ≤ 3 probes and is
not a bound — nothing here rules out a worse ratio at a larger budget.

────────────────────────────────────────────────────────────────────────────────────────────────
2. THE ALLOCATION LEMMA — why the value rule never buys the collision probe

The measured hole (e21): the value rule buys probes that lower the sign potential, and it flagged 0 of
25 two-transition pairs. The lemma says that is forced, not a tuning failure.

SETUP. One hypothesis h* (a single transition) carries posterior mass ≥ 1 − δ. q = posterior mass of
the two-transition family, T = the family indicator, and
    s = max_c | F̄_two(c) − F̄_one(c) |,  F̄_t(c) = mass-weighted mean of F[·,c] inside family t.

f — UPPER BOUND ON THE VALUE OF ANY SINGLE PROBE (proved, `entropy_drop_bound`).
By data processing on Z_c ← h → Y, I(Z_c;Y) ≤ I(h;Y) = h₂(E a_h) − E h₂(a_h), a Jensen gap of h₂ over
the posterior of a_h. h₂ has |h₂″| = 1/(ln2·a(1−a)) ≤ 1/(ln2·r(1−r)) on the reachable range
a ∈ [1−r, r], and a Jensen gap of a function with |f″| ≤ M is ≤ (M/2)·Var. With mass ≥ 1−δ on one
hypothesis, Var_π(F[·,c]) ≤ δ(1−δ) (a [0,1] variable with mass 1−δ at one point) and
Var_π(a_h) = (2r−1)²Var_π(F[·,c]), so for EVERY probe x and both potentials
    Δ_U(x) ≤ f(δ) = (hi−lo) · (2r−1)² / (2 ln2 · r(1−r)) · δ(1−δ)   bits.                          (4)
For u = min(p,1−p) the curvature argument does not apply (u″ is a point mass at ½), and the bound is
instead the Lipschitz one, `error_drop_bound`: |p_c − F[h*,c]| ≤ δ before the answer and ≤ δ′ after,
with δ′ ≤ δr/((1−δ)(1−r)) (the likelihood ratio of any answer is in [(1−r)/r, r/(1−r)]), so
Δ_err(x) ≤ (hi−lo)·(δ + min(δ′,1)) — also O(δ), with constant 1 + r/(1−r) = 20 at r = 0.95.
At r = 0.95, width 1:
f(δ) = 12.30·δ(1−δ). PROVED; test_f_bound_holds checks it on 200 random one-transition-explained
configurations — it holds on all 200, with measured/bound ≤ 0.072 (median 0.022), i.e. loose by
14× at its tightest and 45× typically. What it is for is the RATE, Δ_U = O(δ).

g — LOWER BOUND ON THE T-VALUE (proved, `family_drop_floor`). The drop of H(T) that
`model_check_probe` buys is again exact: Y|T=t is Bernoulli(a_t), a_t = E[a_h | T=t], so
    Δ_T(x) = I(T;Y) = h₂(q a_two + (1−q) a_one) − [q h₂(a_two) + (1−q) h₂(a_one)],
a Jensen gap over a TWO-point distribution. Here |h₂″| ≥ 4/ln2 everywhere, and a gap with |f″| ≥ m is
≥ (m/2)·Var, Var = q(1−q)(a_two − a_one)² = q(1−q)(2r−1)²(F̄_two − F̄_one)². Maximizing over cells,
    max_x Δ_T(x) ≥ g(q, r, s) = (2/ln2) · q(1−q) · (2r−1)² · s².                                   (5)
At r = 0.95, s = 1: g = 2.337·q(1−q). PROVED; test_g_bound_holds (200 configurations, realized
model_check_probe value / g ≥ 1.49, median 1.92 — the floor is real and within 2× of the value).

THE SEPARATION, and why it does not follow from (4) and (5) alone. δ ≥ q always (h* is a
single-transition hypothesis, so the whole two-transition family sits in the 1−π(h*) remainder), and
both bounds are linear in their small parameter, so f and g do not by themselves order the two
rules. What separates them is that (2) is WIDTH-WEIGHTED: a hypothesis that disagrees with h* only
inside a region of measure m contributes at most m·(2r−1)²/(2 ln2·r(1−r))·q to Δ_U, whereas H(T)
carries no width factor — a second transition worth q of posterior mass is worth g(q,r,s) whether its
excursion spans half the domain or one cell.
THE CONFIGURATION THAT MAKES IT BITE is exactly e21's: two SAME-SIGN claims with a closure-filled span
between them. Every one-transition rival of h* = "all σ" is penalized by the claims, so δ → 0 as the
claim weight grows; a two-transition hypothesis whose excursion sits strictly inside the unclaimed
span has the IDENTICAL likelihood — no claim touches it — so q does not decay at all. MEASURED (e33,
gap = 2 of 12 cells, claim weight n_eff swept 0.5 → 64, r = 0.95, entropy potential):
    n_eff   δ        q        max Δ_U (bits)   max Δ_T (bits)   Δ_T/Δ_U
    0.5     0.2636   0.0358   0.16556          0.01251          0.076
    2       0.0628   0.0121   0.01125          0.00862          0.77
    8       0.0112   0.0038   0.00069          0.00588          8.6
    32      0.0029   0.0029   0.00055          0.00544          9.9
    64      0.0029   0.0029   0.00055          0.00544          9.9
δ collapses onto q (in the limit the ONLY thing left unexplained is the hidden excursion), max Δ_U
falls by 300× and plateaus at the partition's own irreducible term, max Δ_T falls by 2.3× and
plateaus at a floor set by q — and their ratio grows monotonically to ≈ 10. Over 200 random
configurations of this shape the median Δ_T/Δ_U by δ-band is 0.064 (δ > 0.15), 0.45 (0.05–0.15),
4.10 (0.02–0.05), 5.28 (0.005–0.02), 7.37 (δ < 0.005).
WHERE THE TWO RULES POINT is a second, weaker matter: over the 200 configurations they name different
cells 63.5 % of the time. In the remaining 36.5 % — which includes the high-claim-weight end of the
sweep above — the value rule names the RIGHT cell and prices it 10× too low (0.00055 against 0.00544
bits at n_eff = 32). That is what sinks it in e21, where the budget is allocated across pairs by
expected drop per cost: a pair whose only open question is a hidden excursion is worth 5e−4 bits to
the value rule and loses every comparison, whatever cell it would have probed.
So the value rule going blind is forced by the objective, not by tuning: `model_check_probe` is the
quantity that survives the δ → 0 limit. The monotone growth of the ratio is MEASURED on these
configurations; what is proved is Δ_U ≤ f(δ) → 0 and max Δ_T ≥ g(q,r,s) > 0 whenever q > 0, s > 0.

WHAT OF THIS LEMMA TRANSFERS OUT OF THIS POSTERIOR — measured against a budget split across the
independent sub-problems of an iterative solver (two islands with no shared variables, so the operator is
block-diagonal; island A well conditioned with per-sweep residual factor ρ = 0.853554, island B
ill conditioned with ρ = 0.999003), where the currency is a residual rather than a posterior.
  • THE CLOSED FORMS DO NOT. f(δ) and g(q,r,s) are bits over a reliability r and a posterior mass δ; a
    residual is a non-negative real in impulse units and carries neither. Importing the constants is not
    justified, and `f(0.05, 1, 0.95) = 0.584291` / `g(0.05, 0.95, 1) = 0.111015` bits mean nothing there.
  • GREEDY IS NOT OPTIMAL THERE EITHER, by a different mechanism. The objective is separable across the
    islands, but a residual is not convex in the sweep count during the transient, so the marginal drops
    are not non-increasing and the set function is not submodular. Measured greedy/optimal over budgets
    4–64: 1.00–1.17, worst at K = 8 (greedy (6,2) for Σ res 9.4747e-2 against the optimum (7,1) for
    8.1076e-2). Same shape as (a)–(c) above, different reason — a non-geometric transient, not noise.
  • THE MECHANISM REPRODUCES EXACTLY. Allocating by residual DROP starves the ill-conditioned island: at
    K = 128 greedy (126, 2) leaves island A at relative error 3.58e-9 and island B at 1.0 — unchanged —
    while an equal split (64, 64) leaves 6.58e-5 and 0.941. The currency collapses while what matters does
    not, which is f(δ) → 0 in this file's terms. Re-weighting the drop does not repair it (the identical
    split, because a per-island constant cannot change which island has the larger marginal drop); a rule
    on the error-bound LEVEL does change the split, at a worse total residual. The analogue of g > 0 is
    therefore the condition number, and it has to enter as a level, not as a drop.
  • WHAT TRANSFERS IS A DIFFERENT RULE: a per-island budget from the asymptotic rate. Measure ρ_i once and
    allocate k_i = log(tol/r_i0) / log ρ_i. Measured against sweeps to 1e-8: 107.6 predicted against 108
    (island A) and 11 539.7 against 11 542 (island B) — 0.4 % and 0.0 %.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PosteriorState", "state_of", "potential_of", "update", "one_step_drops",
    "bsc_gain", "adaptive_submodularity_check", "brute_force_adaptive_optimum",
    "greedy_sequence_value", "greedy_vs_optimal", "entropy_drop_bound",
    "error_drop_bound", "family_drop_floor", "allocation_report",
]

LN2 = math.log(2.0)


# ── the state a probe acts on ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PosteriorState:
    """The (cells, widths, F, posterior) tuple of a RegimePosterior WITH its probes applied.

    Everything below works on this tuple instead of on RegimePosterior objects: the enumeration of
    adaptive policies visits thousands of posteriors and re-solving the claims at each node would
    dominate. `update` reproduces RegimePosterior._with_probes' likelihood step exactly (test)."""
    cells: np.ndarray
    w: np.ndarray
    F: np.ndarray
    post: np.ndarray
    n_one: int
    potential: str

    @property
    def n_cells(self) -> int:
        return len(self.w)

    def midpoints(self) -> np.ndarray:
        return self.cells.mean(1)

    def with_potential(self, potential: str) -> "PosteriorState":
        return PosteriorState(self.cells, self.w, self.F, self.post, self.n_one, potential)


def state_of(posterior, potential: str | None = None) -> PosteriorState:
    """Read a RegimePosterior into a PosteriorState (public read `_with_probes`, plus `_n_one`)."""
    cells, w, F, post = posterior._with_probes()
    return PosteriorState(cells, w, F, post, int(posterior._n_one),
                          posterior.potential if potential is None else potential)


def _u(pp: np.ndarray, potential: str) -> np.ndarray:
    if potential == "error":
        return np.minimum(pp, 1 - pp)
    q = np.clip(pp, 1e-12, 1 - 1e-12)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def potential_of(st: PosteriorState) -> float:
    """U = Σ_c w_c u(p_c), the same number as RegimePosterior.potential_value()."""
    return float(_u(st.post @ st.F, st.potential) @ st.w)


def update(st: PosteriorState, cell: int, sign: int, reliability: float, weight: float = 1.0) -> PosteriorState:
    """The posterior after one probe answer in `cell` — RegimePosterior._with_probes' step, verbatim."""
    fp = st.F[:, cell]
    f = fp if sign > 0 else 1 - fp
    lp = np.log(st.post + 1e-300) + weight * np.log(f * reliability + (1 - f) * (1 - reliability))
    p = np.exp(lp - lp.max())
    return PosteriorState(st.cells, st.w, st.F, p / p.sum(), st.n_one, st.potential)


def _outcomes(st: PosteriorState, cell: int, reliability: float):
    """[(P(answer), posterior | answer)] over the two answers."""
    f = st.F[:, cell]
    out = []
    for sg in (1, -1):
        ff = f if sg > 0 else 1 - f
        like = ff * reliability + (1 - ff) * (1 - reliability)
        pout = float(st.post @ like)
        if pout <= 0:
            continue
        q = st.post * like / pout
        out.append((sg, pout, PosteriorState(st.cells, st.w, st.F, q / q.sum(), st.n_one, st.potential)))
    return out


def one_step_drops(st: PosteriorState, reliability: float = 0.95) -> np.ndarray:
    """Expected drop of U for a probe in each cell — `best_probe`'s objective, per cell (test)."""
    now = potential_of(st)
    d = np.empty(st.n_cells)
    for c in range(st.n_cells):
        after = 0.0
        for _sg, pout, nxt in _outcomes(st, c, reliability):
            after += pout * potential_of(nxt)
        d[c] = now - after
    return d


def family_entropy(st: PosteriorState) -> float:
    """H(P(two transitions)) — `model_check_probe`'s objective."""
    q = float(st.post[st.n_one:].sum())
    if q <= 0 or q >= 1:
        return 0.0
    return -(q * math.log2(q) + (1 - q) * math.log2(1 - q))


def family_drops(st: PosteriorState, reliability: float = 0.95) -> np.ndarray:
    """Expected drop of H(P(two transitions)) per cell — `model_check_probe`'s objective, per cell."""
    now = family_entropy(st)
    d = np.empty(st.n_cells)
    for c in range(st.n_cells):
        after = 0.0
        for _sg, pout, nxt in _outcomes(st, c, reliability):
            after += pout * family_entropy(nxt)
        d[c] = now - after
    return d


# ── 1(a): the mechanism of the counterexample ──────────────────────────────────────────────────
def bsc_gain(p: float, r: float) -> float:
    """Expected binary-entropy drop of one BSC(r) reading of a Bernoulli(p) target, eq. (3).

    Equals h₂(a) − h₂(r) with a = pr + (1−p)(1−r): symmetric about p = ½, strictly increasing on
    (0, ½). That monotonicity IS the failure of adaptive submodularity — an answer that pushes p back
    toward ½ makes every later probe worth strictly more."""
    a = p * r + (1 - p) * (1 - r)
    h = lambda t: 0.0 if t <= 0 or t >= 1 else -(t * math.log2(t) + (1 - t) * math.log2(1 - t))
    return h(a) - h(r)


# ── 1(b): the diminishing-returns inequality, measured ─────────────────────────────────────────
def adaptive_submodularity_check(posterior, probes, n_random: int = 200, reliability: float = 0.95,
                                 potentials=("entropy", "error"), seed: int = 0) -> dict:
    """Test Δ(x | A) ≥ Δ(x | B) for random sub-realizations A ⊆ B, for both potentials.

    `probes` = candidate probe locations x (the histories are drawn from them; the tested probe is one
    that is not in B). A and B are sub-realizations, i.e. probe/answer PAIRS, with A's answers equal to
    B's on the shared probes — that is what Golovin & Krause's definition requires, and it is why the
    answers are drawn from the model (`rng` picks the answer with its predictive probability) rather
    than assigned at random.

    Returns per potential: violation rate, mean and worst violation ratio Δ(x|B)/Δ(x|A), and the
    absolute worst excess Δ(x|B) − Δ(x|A)."""
    rng = np.random.default_rng(seed)
    base = state_of(posterior)
    cells = base.cells
    cand = [min(int(np.searchsorted(cells[:, 1], float(x), side="left")), len(cells) - 1) for x in probes]
    cand = sorted(set(cand))
    if len(cand) < 3:
        raise ValueError("need at least 3 distinct probe cells")
    out = {}
    for pot in potentials:
        st0 = base.with_potential(pot)
        viol = 0
        ratios, excess = [], []
        n_eff = 0
        for _ in range(n_random):
            k = int(rng.integers(1, min(4, len(cand) - 1) + 1))        # |B| ∈ 1..4
            B = list(rng.choice(cand, size=k, replace=False))
            rest = [c for c in cand if c not in B]
            x = int(rng.choice(rest))
            # draw B's answers from the model, sequentially (a genuine sub-realization)
            st = st0
            hist = []
            for c in B:
                outs = _outcomes(st, c, reliability)
                probs = np.array([o[1] for o in outs])
                j = int(rng.choice(len(outs), p=probs / probs.sum()))
                hist.append((c, outs[j][0]))
                st = outs[j][2]
            m = int(rng.integers(0, len(hist)))                        # A = a random prefix-subset of B
            keep = sorted(rng.choice(len(hist), size=m, replace=False).tolist())
            stA = st0
            for i in keep:
                c, sg = hist[i]
                stA = update(stA, c, sg, reliability)
            dA = float(one_step_drops(stA, reliability)[x])
            dB = float(one_step_drops(st, reliability)[x])
            n_eff += 1
            if dB > dA + 1e-12:
                viol += 1
                excess.append(dB - dA)
                ratios.append(dB / dA if dA > 1e-12 else float("inf"))
        finite = [r for r in ratios if math.isfinite(r)]
        out[pot] = {
            "n": n_eff,
            "violation_rate": viol / max(n_eff, 1),
            "worst_ratio": max(ratios) if ratios else 1.0,
            "worst_finite_ratio": max(finite) if finite else 1.0,
            "mean_excess_bits": float(np.mean(excess)) if excess else 0.0,
            "worst_excess_bits": max(excess) if excess else 0.0,
            "n_infinite_ratio": len(ratios) - len(finite),
        }
    return out


# ── 1(c): the adaptive optimum by full enumeration ─────────────────────────────────────────────
def _optimal_final_U(st: PosteriorState, budget: int, cands: list[int], r: float) -> float:
    """min over adaptive policies of E[U after `budget` probes] — exact DP over the decision tree."""
    if budget == 0:
        return potential_of(st)
    best = float("inf")
    for c in cands:
        val = 0.0
        for _sg, pout, nxt in _outcomes(st, c, r):
            val += pout * _optimal_final_U(nxt, budget - 1, cands, r)
        best = min(best, val)
    return best


def brute_force_adaptive_optimum(posterior, budget: int, candidates=None, reliability: float = 0.95,
                                 potential: str | None = None) -> dict:
    """Expected drop of U achieved by the OPTIMAL adaptive policy of `budget` probes.

    Full enumeration: a policy is a decision tree over the binary answers, so the optimum is the DP
    min over candidates of the answer-averaged value-to-go. Cost ~ (2|C|)^budget · |C| posterior
    updates — for tiny partitions only (n_grid ≤ 6, budget ≤ 3). `candidates` are x values; they are
    mapped to cells, and the default is every cell midpoint."""
    st = state_of(posterior, potential)
    if candidates is None:
        cands = list(range(st.n_cells))
    else:
        cands = sorted({min(int(np.searchsorted(st.cells[:, 1], float(x), side="left")), st.n_cells - 1)
                        for x in candidates})
    now = potential_of(st)
    fin = _optimal_final_U(st, budget, cands, reliability)
    return {"U0": now, "optimal_final_U": fin, "optimal_value": now - fin,
            "budget": budget, "n_candidates": len(cands), "n_cells": st.n_cells}


def greedy_sequence_value(posterior, budget: int, candidates=None, reliability: float = 0.95,
                          potential: str | None = None) -> dict:
    """Expected drop of U achieved by the GREEDY adaptive policy (`best_probe`'s rule at every node),
    averaged exactly over all 2^budget answer paths."""
    st = state_of(posterior, potential)
    if candidates is None:
        cands = list(range(st.n_cells))
    else:
        cands = sorted({min(int(np.searchsorted(st.cells[:, 1], float(x), side="left")), st.n_cells - 1)
                        for x in candidates})

    def rec(s: PosteriorState, k: int) -> float:
        if k == 0:
            return potential_of(s)
        d = one_step_drops(s, reliability)
        c = max(cands, key=lambda i: d[i])
        return sum(pout * rec(nxt, k - 1) for _sg, pout, nxt in _outcomes(s, c, reliability))

    now = potential_of(st)
    fin = rec(st, budget)
    return {"U0": now, "greedy_final_U": fin, "greedy_value": now - fin, "budget": budget}


def greedy_vs_optimal(posterior, budget: int, candidates=None, reliability: float = 0.95,
                      potential: str | None = None) -> dict:
    """The ratio Golovin & Krause bound below by 1−1/e when the objective is adaptive submodular.
    Ours is not (see the module docstring), so this is a measurement."""
    o = brute_force_adaptive_optimum(posterior, budget, candidates, reliability, potential)
    g = greedy_sequence_value(posterior, budget, candidates, reliability, potential)
    ratio = g["greedy_value"] / o["optimal_value"] if o["optimal_value"] > 1e-15 else 1.0
    return {**o, **g, "ratio": ratio, "one_minus_1_over_e": 1 - 1 / math.e}


# ── 2: the allocation lemma ────────────────────────────────────────────────────────────────────
def entropy_drop_bound(delta: float, width: float, reliability: float) -> float:
    """f(δ), eq. (4): an upper bound on the expected potential drop of ANY single probe when one
    hypothesis carries posterior mass ≥ 1 − δ. Proved via data processing + the Jensen-gap bound
    (M/2)·Var with M = max|h₂″| = 1/(ln2·r(1−r)) on the reachable range a ∈ [1−r, r]."""
    r = float(reliability)
    return width * (2 * r - 1) ** 2 / (2 * LN2 * r * (1 - r)) * delta * (1 - delta)


def error_drop_bound(delta: float, width: float, reliability: float) -> float:
    """The u = min(p,1−p) counterpart of f(δ): u is 1-Lipschitz, the marginal sits within δ of
    F[h*,·] before the answer and within δ′ ≤ δr/((1−δ)(1−r)) after it, so no probe can drop the
    error potential by more than width·(δ + min(δ′,1)). Proved; O(δ) with constant 1 + r/(1−r)."""
    r = float(reliability)
    if delta <= 0:
        return 0.0
    dp = min(1.0, delta * r / max((1 - delta) * (1 - r), 1e-300))
    return min(width * (delta + dp), 0.5 * width)


def family_drop_floor(q: float, reliability: float, s: float) -> float:
    """g(q, r, s), eq. (5): a lower bound on max_x (expected drop of H(P(two transitions))).
    Proved via the Jensen-gap bound (m/2)·Var with m = min|h₂″| = 4/ln2."""
    r = float(reliability)
    return (2.0 / LN2) * q * (1 - q) * (2 * r - 1) ** 2 * s ** 2


def allocation_report(posterior, reliability: float = 0.95) -> dict:
    """The lemma's quantities on one posterior: δ, q, s, the bounds f and g, and what the two rules
    actually buy (max entropy-potential drop over cells, max family-entropy drop over cells)."""
    st = state_of(posterior, "entropy")
    post, F = st.post, st.F
    h_star = int(np.argmax(post))
    delta = float(1.0 - post[h_star])
    q = float(post[st.n_one:].sum())
    one, two = post[:st.n_one], post[st.n_one:]
    if q > 0 and one.sum() > 0:
        f1 = (one @ F[:st.n_one]) / one.sum()
        f2 = (two @ F[st.n_one:]) / two.sum()
        s = float(np.max(np.abs(f2 - f1)))
    else:
        s = 0.0
    du = one_step_drops(st, reliability)
    dt = family_drops(st, reliability)
    width = float(st.cells[-1, 1] - st.cells[0, 0])
    return {
        "delta": delta, "q_two": q, "separation_s": s, "width": width,
        "f_bound": entropy_drop_bound(delta, width, reliability),
        "g_bound": family_drop_floor(q, reliability, s),
        "max_entropy_drop": float(du.max()), "max_family_drop": float(dt.max()),
        "argmax_entropy_cell": int(np.argmax(du)), "argmax_family_cell": int(np.argmax(dt)),
        "ratio_T_over_U": float(dt.max() / du.max()) if du.max() > 1e-15 else float("inf"),
        "expected_error": float(np.minimum(post @ F, 1 - post @ F) @ st.w),
    }
