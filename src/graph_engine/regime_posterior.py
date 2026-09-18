#!/usr/bin/env python3
"""
regime_posterior.py — all claims on one (subject, object) pair along one condition variable, read jointly.

DECLARED ASSUMPTION (the thing the collision probe must guard): along the variable the sign
changes AT MOST ONCE. Then {x : sign = +} is an interval that touches one end of the domain, and:

  closure      every point between two same-sign claims has that sign — no measurement needed.
  order conf   the pattern + − + along the axis is impossible: at least one claim is wrong, or the
               assumption is (a second transition). Disjoint boxes can contradict each other.
  span probes  for a pair with one sign only, the probe goes deep into the larger unknown span and then to the other side:
               for a + claim at 0.4–0.5 the exact value picks 0.87 then 0.07 (claim weight 3) or 0.91 then 0.07 (weight 8);
               a + answer decides everything up to the probe by closure, a − answer brackets the transition.
               Two earlier statements here were artifacts of a coarse partition that offered one candidate per span and
               are withdrawn: "the middle of the span (0.75) first", and "a weak claim is confirmed first".
  bisection    a probe inside the gap between opposite signs halves it. The gain per gap decays
               geometrically, so taking the largest remaining expected gain first is optimal among
               allocations (separable concave rewards).

Exact Bayes over hypotheses h ∈ {all +, all −, (+→− at t), (−→+ at t)}, t on the midpoints of a FIXED partition:
the claim edges plus `n_grid` uniform cells.

FIXED HYPOTHESIS SPACE. A probe is assigned to the cell it falls in; it never adds a cell. A first version inserted every
probe point as a new cell edge. That changed the set of hypotheses and re-spread the prior after each probe, with two
measured consequences (found in review): the expected potential could RISE after a probe (one-step "drop" −0.023 on a
three-claim case), and `best_probe` predicted 0.0385 where 0.0236 was realized. With the space fixed, the posterior is a
martingale, the expected potential cannot rise (Jensen), and the predicted value is the realized expectation (test). A claim with box B, sign σ and weight n (the caller
passes claim_federation's lineage count) has likelihood r^{n f} (1−r)^{n (1−f)}, f = share of B where h has sign σ.
A probe is a point observation with its own reliability.

value of a probe at x = expected drop of  U = ∫ u(P₊(x)) dx, computed exactly over the two outcomes.
u = binary entropy by default. u = min(p, 1−p) (expected wrongly-signed measure) is available but is
piecewise LINEAR: since E[posterior] = prior, its expected drop is exactly 0 for every probe whose
single answer cannot flip the decision in any cell, so a pair that is decided everywhere gets value 0
everywhere and the choice is arbitrary (test: one strong claim covering the whole domain). A strictly concave u gives every informative probe a
positive value. Measured in e4 on the fixed partition, min(p,1−p) nevertheless scored 0.001–0.006 HIGHER in accuracy
than entropy after 60 probes in all four settings; entropy is the default for the reasons above, not for that score.

Fisher coordinates. Since E[posterior] = prior, the value of a probe is a Jensen gap: ≈ −½·u″(p)·Var(Δp). For
binary entropy u″(p) = −1/(ln2·p(1−p)), the Fisher metric of a yes/no quantity, so value ≈ Var(Δp)/(2 ln2·p(1−p)).
In θ = 2·arcsin√p that metric is flat (Δθ ≈ Δp/√(p(1−p)); Wootters 1981, Braunstein & Caves 1994), hence
    value ≈ E[Δθ²] / (2 ln 2):
the expected entropy drop is the expected squared distance the belief travels, measured in the coordinates where
straight lines are geodesics. Exact to second order; the ratio is 0.99–1.00 for weak answers and 0.79 for a
near-certain answer at p = 0.5 (test).

Greedy guarantee, and its scope. With NOISELESS answers, potential "error" and p_two = 0, the objective is a version-space
measure, adaptive monotone submodular, and the greedy choice is within (1−1/e) of the optimal adaptive policy (Golovin &
Krause, JAIR 42, 2011). For noisy answers — the default probe reliability is 0.95 — no bound is claimed; the test pins
the near-noiseless case only.

COLLISION PROBE. A full-rank local sensitivity never proves that a reading is globally unique; a uniqueness claim needs a search for a second reading that fits equally well. The two-transition family (+ − +, − + −) stays in the hypothesis space with
prior `p_two`. `collision()` reports its posterior mass and the Bayes factor. Because the probe value is
computed over the whole space, mass on the collision family automatically makes probes INSIDE
closure-filled spans worth something — that is where a second transition would hide.
Tried first and rejected: counting independent origins left unexplained by the best single-transition
fit. On TEST-GRAPHS with 40 % two-transition pairs it flagged nothing right (precision 0, recall 0):
with 1–3 claims per pair the + − + pattern is almost never present passively. It has to be probed for
(e4b: recall 0.25 / 0.89 / 0.99 at 4 / 8 / 16 probes on one pair, false flags ≤ 0.007).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["RegimePosterior"]


@dataclass
class RegimePosterior:
    lo: float
    hi: float
    reliability: float = 0.75
    p_flip: float = 0.3                      # prior P(the pair has a transition inside the domain)
    potential: str = "entropy"               # "entropy" (strictly concave) or "error" = min(p, 1−p)
    n_grid: int = 48                         # fixed uniform cells; with the claim edges they are the ONLY places a
                                             # transition can sit. Probes never add cells (see FIXED HYPOTHESIS SPACE).
    p_two: float = 0.05                      # prior P(TWO transitions): the collision family, kept in
                                             # the hypothesis space so the data can speak for it. 0 = assume it away.
    claims: list = field(default_factory=list)   # (a, b, sign, n_eff)
    probes: list = field(default_factory=list)   # (x, sign, reliability)
    _cache: tuple | None = None

    def add_claim(self, a: float, b: float, sign: int, n_eff: float = 1.0, reliability: float | None = None) -> None:
        """reliability: this claim's own (e.g. from source_reliability); None = the posterior's default."""
        r = self.reliability if reliability is None else float(reliability)
        if not 0.5 <= r < 1.0:
            raise ValueError(f"claim reliability must be in [0.5, 1): {r}")
        self.claims.append((max(a, self.lo), min(b, self.hi), 1 if sign > 0 else -1, float(n_eff), r))
        self._cache = None

    def set_claim_reliabilities(self, rs: list[float]) -> None:
        """Replace every claim's reliability in order (used when reliabilities are re-estimated during a loop)."""
        assert len(rs) == len(self.claims)
        self.claims = [(a, b, sg, n, float(r)) for (a, b, sg, n, _), r in zip(self.claims, rs)]
        self._cache = None

    def add_probe(self, x: float, sign: int, reliability: float = 0.95, weight: float = 1.0) -> None:
        """weight < 1 tempers the likelihood: K answers from one deterministic judge whose errors are
        correlated along x carry N_eff = K / (1 + (K−1)ρ) answers' worth, so each enters with N_eff / K."""
        self.probes.append((float(x), 1 if sign > 0 else -1, reliability, float(weight)))
        self._cache = None

    # -- hypothesis space --------------------------------------------------------------------------
    def _solve(self):
        if self._cache is not None:
            return self._cache
        pts = {self.lo, self.hi}
        for a, b, *_ in self.claims:
            pts.update((a, b))
        pts.update(np.linspace(self.lo, self.hi, self.n_grid + 1).tolist())
        edges = np.array(sorted(pts))
        cells = np.stack([edges[:-1], edges[1:]], 1)             # P₊ is constant on each cell
        cells = cells[cells[:, 1] > cells[:, 0]]
        thr = cells.mean(1)                                       # one candidate threshold per cell
        nt = len(thr)
        # hypotheses: [all+, all−, (+→−)@t..., (−→+)@t...]; S[h, cell] = sign of h on the cell.
        # a threshold inside a cell splits it; for the sign table the cell takes the sign of its left part
        # and the split is handled through the measure weights W below.
        mid = cells.mean(1)
        S = np.ones((2 + 2 * nt, len(cells)))
        S[1] = -1
        for k, t in enumerate(thr):
            S[2 + k] = np.where(mid < t, 1, -1)
            S[2 + nt + k] = np.where(mid < t, -1, 1)
        # a threshold cell is half + and half −: F[h, cell] = share of the cell where h is +
        F = (S > 0).astype(float)
        for k in range(nt):
            F[2 + k, k] = 0.5
            F[2 + nt + k, k] = 0.5
        n_one = len(F)
        if self.p_two > 0 and nt >= 3:                            # + − + and − + − with transitions in cells i < j
            ii, jj = np.triu_indices(nt, 1)
            idx = np.arange(nt)[None, :]
            inner = ((idx > ii[:, None]) & (idx < jj[:, None])).astype(float)
            inner[np.arange(len(ii)), ii] = 0.5
            inner[np.arange(len(jj)), jj] = 0.5
            F = np.vstack([F, 1 - inner, inner])                  # (+ − +): + outside; (− + −): + inside
        n_two = len(F) - n_one
        p1 = self.p_flip * (1 - (self.p_two if n_two else 0)); p0 = (1 - self.p_flip) * (1 - (self.p_two if n_two else 0))
        logp = np.full(len(F), math.log(max(self.p_two, 1e-300) / max(n_two, 1)))
        logp[:2] = math.log(max(p0, 1e-12) / 2)
        logp[2:n_one] = math.log(max(p1, 1e-12) / (2 * max(nt, 1)))
        self._n_one = n_one
        w = cells[:, 1] - cells[:, 0]
        for a, b, sg, n, rel in self.claims:
            lr, lq = math.log(rel), math.log(1 - rel)
            ov = np.clip(np.minimum(cells[:, 1], b) - np.maximum(cells[:, 0], a), 0, None)
            if ov.sum() <= 0:                                     # zero-width claim: use the containing cell
                ov = ((cells[:, 0] <= a) & (a <= cells[:, 1])).astype(float)
            ov = ov / ov.sum()
            fplus = F @ ov
            f = fplus if sg > 0 else 1 - fplus
            logp += n * (f * lr + (1 - f) * lq)
        post = np.exp(logp - logp.max()); post /= post.sum()
        self._cache = (cells, w, F, post, logp)
        return self._cache

    # -- reads -------------------------------------------------------------------------------------
    def _with_probes(self):
        cells, w, F, post, logp = self._solve()
        if not self.probes:
            return cells, w, F, post
        lp = np.log(post + 1e-300)
        for x, sg, rel, wt in self.probes:
            c = min(int(np.searchsorted(cells[:, 1], x, side="left")), len(cells) - 1)
            fp = F[:, c]
            f = fp if sg > 0 else 1 - fp
            lp += wt * np.log(f * rel + (1 - f) * (1 - rel))
        p = np.exp(lp - lp.max())
        return cells, w, F, p / p.sum()

    def p_plus(self, x: float) -> float:
        cells, _, F, post = self._with_probes()
        c = min(int(np.searchsorted(cells[:, 1], x, side="left")), len(cells) - 1)
        return float(post @ F[:, c])

    def _u(self, pp: np.ndarray) -> np.ndarray:
        if self.potential == "error":
            return np.minimum(pp, 1 - pp)
        q = np.clip(pp, 1e-12, 1 - 1e-12)
        return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))

    def potential_value(self) -> float:
        """U = ∫ u(P₊(x)) dx in the units `best_probe` reports (bits for "entropy", wrongly-signed measure for "error")."""
        _, w, F, post = self._with_probes()
        return float(self._u(post @ F) @ w)

    def expected_error(self) -> float:
        _, w, F, post = self._with_probes()
        pp = post @ F
        return float(np.minimum(pp, 1 - pp) @ w)

    def best_probe(self, reliability: float = 0.95) -> tuple[float, float]:
        """(x, expected drop of expected_error) — exact over both outcomes, one candidate per cell
        plus the two ends of the domain."""
        cells, w, F, post = self._with_probes()
        pp = post @ F
        now = float(self._u(pp) @ w)
        xs = list(cells.mean(1)) + [self.lo, self.hi]
        cs = list(range(len(cells))) + [0, len(cells) - 1]
        best = (xs[0], -1.0)
        for x, c in zip(xs, cs):
            f = F[:, c]
            after = 0.0
            for sg in (1, -1):
                like = (f if sg > 0 else 1 - f) * reliability + (1 - (f if sg > 0 else 1 - f)) * (1 - reliability)
                pout = float(post @ like)
                if pout <= 0:
                    continue
                q = post * like / pout
                qq = q @ F
                after += pout * float(self._u(qq) @ w)
            if now - after > best[1]:
                best = (float(x), now - after)
        return best

    def model_check_probe(self, reliability: float = 0.95) -> tuple[float, float]:
        """(x, expected drop of the entropy of the FAMILY indicator one-vs-two transitions). The value rule `best_probe`
        buys probes that lower the sign potential; when the one-transition family already explains the claims it never
        buys the probes that would expose a second transition (e21: 0 of 25 two-transition pairs flagged). This probe is
        bought against the model error instead: the expected drop of H(P(two transitions)) over the two outcomes, exact
        on the fixed partition. Zero when the collision family is off (p_two = 0)."""
        cells, w, F, post = self._with_probes()
        if len(post) == self._n_one:
            return (0.5 * (self.lo + self.hi), 0.0)
        two = np.zeros(len(post)); two[self._n_one:] = 1.0
        h = lambda p: 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        now = h(float(post @ two))
        best = (float(cells[0].mean()), -1.0)
        for c in range(len(cells)):
            f = F[:, c]; after = 0.0
            for sg in (1, -1):
                like = (f if sg > 0 else 1 - f) * reliability + (1 - (f if sg > 0 else 1 - f)) * (1 - reliability)
                pout = float(post @ like)
                if pout <= 0:
                    continue
                q = post * like / pout
                after += pout * h(float(q @ two))
            if now - after > best[1]:
                best = (float(cells[c].mean()), now - after)
        return best

    def collision(self, flag_at: float = 0.5) -> dict:
        """Posterior mass on the collision family (two transitions) and the Bayes factor against
        the declared single-transition reading. A pair is flagged when that mass passes `flag_at`:
        report it as an open gauge, do not report a unique regime boundary."""
        _, _, _, post = self._with_probes()
        m2 = float(post[self._n_one:].sum())
        prior = self.p_two if len(post) > self._n_one else 0.0
        bf = (m2 / max(1 - m2, 1e-300)) / (prior / (1 - prior)) if 0 < prior < 1 else float("nan")
        return {"p_two_transitions": m2, "bayes_factor_two_vs_one": bf, "flag": m2 >= flag_at}
