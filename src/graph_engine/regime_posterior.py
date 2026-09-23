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
    claim_model: str = "pointwise"           # how a box claim is read: "pointwise" (r-accurate at every point of the box, likelihood
                                             # r^{n f}(1−r)^{n(1−f)}) or "majority" (the source reported the majority sign of its box:
                                             # likelihood r·S(k(f−½)) + (1−r)(1−S(k(f−½))), S logistic, k = 20). e21 found the
                                             # loop's over-confidence lives in the pointwise reading, not in r (see closed_loop).
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
            if self.claim_model == "majority":
                sm = 1.0 / (1.0 + np.exp(-20.0 * (f - 0.5)))
                logp += n * np.log(rel * sm + (1 - rel) * (1 - sm))
            else:
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

    # -- one-step probe choice: the per-candidate exact loop, and an optional moment screen ----------
    #
    # Every hypothetical answer multiplies the posterior by like = (1−r) + (2r−1)·f_c (answer +) or r − (2r−1)·f_c
    # (answer −), f_c = F[:, c]. Its predictive P₊ over all cells is therefore
    #     q@F = ((1−r)·pp + (2r−1)·S[c]) / pout₊,   S = Fᵀ diag(post) F,   pout₊ = (1−r) + (2r−1)·pp_c
    # (and the mirror for −), and the family mass q@two uses the vector t = Fᵀ(post·two) the same way. So ONE
    # second-moment matrix S prices every candidate; the per-candidate loop does 2 H×Q products per candidate.
    # method="moments" screens with S and then re-decides, with the ORIGINAL per-candidate code, every candidate
    # whose screened gain is within a band of the screened maximum (strict ">" first-maximum order, as the loop).
    # The returned (x, gain) is therefore the direct result bit for bit. `last_probe_info` records the path.
    _SCREEN_BAND = 1e-9
    _SCREEN_CHUNK = 65536
    last_probe_info = None                   # set by the probe choosers: which path decided (not a dataclass field)

    def _after_u(self, post, F, w, f, reliability):
        after = 0.0
        for sg in (1, -1):
            like = (f if sg > 0 else 1 - f) * reliability + (1 - (f if sg > 0 else 1 - f)) * (1 - reliability)
            pout = float(post @ like)
            if pout <= 0:
                continue
            q = post * like / pout
            qq = q @ F
            after += pout * float(self._u(qq) @ w)
        return after

    def _after_tv(self, post, F, w, f, two, h, reliability):
        after_u = after_f = 0.0
        for sg in (1, -1):
            like = (f if sg > 0 else 1 - f) * reliability + (1 - (f if sg > 0 else 1 - f)) * (1 - reliability)
            pout = float(post @ like)
            if pout <= 0:
                continue
            q = post * like / pout
            after_u += pout * float(self._u(q @ F) @ w)
            after_f += pout * (h(float(q @ two)) if len(post) > self._n_one else 0.0)
        return after_u, after_f

    def _after_mc(self, post, f, two, h, reliability):
        after = 0.0
        for sg in (1, -1):
            like = (f if sg > 0 else 1 - f) * reliability + (1 - (f if sg > 0 else 1 - f)) * (1 - reliability)
            pout = float(post @ like)
            if pout <= 0:
                continue
            q = post * like / pout
            after += pout * h(float(q @ two))
        return after

    @staticmethod
    def _check_method(method):
        if method not in ("direct", "moments"):
            raise ValueError(f"method must be 'direct' or 'moments': {method!r}")

    def _second_moments(self, post, F):
        """S = Fᵀ diag(post) F, accumulated over row chunks (memory O(chunk·Q) beyond F)."""
        Q = F.shape[1]
        S = np.zeros((Q, Q))
        for a in range(0, F.shape[0], self._SCREEN_CHUNK):
            Fa = F[a:a + self._SCREEN_CHUNK]
            S += Fa.T @ (post[a:a + self._SCREEN_CHUNK, None] * Fa)
        return 0.5 * (S + S.T)

    @staticmethod
    def _screen_ok(reliability):
        r = float(reliability)
        return math.isfinite(r) and 0.0 < r < 1.0

    def _screen_after_u(self, pp, S, w, cs, r):
        """Screened E[U after] for candidate cells `cs` (both answers), vectorized."""
        a = 2.0 * r - 1.0
        pc = pp[cs]
        pout_p = (1.0 - r) + a * pc
        pout_m = r - a * pc
        Sc = S[cs]
        up = self._u(((1.0 - r) * pp[None, :] + a * Sc) / pout_p[:, None]) @ w
        um = self._u((r * pp[None, :] - a * Sc) / pout_m[:, None]) @ w
        return pout_p * up + pout_m * um, pout_p, pout_m

    @staticmethod
    def _screen_family(P2, tvec, cs, r, pout_p, pout_m, hv):
        a = 2.0 * r - 1.0
        tp = ((1.0 - r) * P2 + a * tvec[cs]) / pout_p
        tm = (r * P2 - a * tvec[cs]) / pout_m
        return pout_p * hv(tp) + pout_m * hv(tm)

    @staticmethod
    def _hvec(p):
        p = np.asarray(p, float)
        out = np.zeros_like(p)
        m = (p > 0) & (p < 1)
        pm = p[m]
        out[m] = -(pm * np.log2(pm) + (1 - pm) * np.log2(1 - pm))
        return out

    @staticmethod
    def _resolve(order, screened, band, exact_gain, init):
        """Re-decide by the exact per-candidate gain every candidate within `band` of the screened max, in the
        original candidate order, with the original strict-">" first-maximum rule. Returns (index, gain, n)."""
        g = np.asarray(screened, float)
        top = float(np.max(g))
        cand = [i for i in order if g[i] >= top - band]
        best_i, best_g = None, init
        for i in cand:
            gi = exact_gain(i)
            if gi > best_g:
                best_i, best_g = i, gi
        return best_i, best_g, len(cand)

    def best_probe(self, reliability: float = 0.95, *, method: str = "direct") -> tuple[float, float]:
        """(x, expected drop of expected_error) — exact over both outcomes, one candidate per cell
        plus the two ends of the domain. method="moments": the second-moment screen above, same result."""
        self._check_method(method)
        cells, w, F, post = self._with_probes()
        pp = post @ F
        now = float(self._u(pp) @ w)
        xs = list(cells.mean(1)) + [self.lo, self.hi]
        cs = list(range(len(cells))) + [0, len(cells) - 1]
        if method == "moments" and self._screen_ok(reliability):
            r = float(reliability)
            after, _, _ = self._screen_after_u(pp, self._second_moments(post, F), w, np.asarray(cs), r)
            gains = now - after
            if np.all(np.isfinite(gains)):
                band = self._SCREEN_BAND * (self.hi - self.lo)
                i, g, n = self._resolve(range(len(xs)), gains, band,
                                        lambda i: now - self._after_u(post, F, w, F[:, cs[i]], reliability), -1.0)
                if i is not None:
                    self.last_probe_info = {"method": "moments", "path": "moments", "resolved": n}
                    return (float(xs[i]), g)
            self.last_probe_info = {"method": "moments", "path": "direct", "reason": "screen not usable"}
        elif method == "moments":
            self.last_probe_info = {"method": "moments", "path": "direct", "reason": "reliability outside (0, 1)"}
        else:
            self.last_probe_info = {"method": "direct", "path": "direct"}
        best = (xs[0], -1.0)
        for x, c in zip(xs, cs):
            after = self._after_u(post, F, w, F[:, c], reliability)
            if now - after > best[1]:
                best = (float(x), now - after)
        return best

    def total_value_probe(self, reliability: float = 0.95, lam: float = 1.0, *, method: str = "direct") -> tuple[float, float]:
        """(x, gain) maximizing ΔU(x) + λ·ΔH_family(x): the sign potential's expected drop PLUS the expected drop of the entropy
        of the one-vs-two-transition indicator, both in bits, exact over the two outcomes. e33 proved the value rule prices a
        collision probe ~10× too low when the one-transition family already explains the claims (the hole in e21 is the price,
        not the location); adding the family entropy to the same currency is the correctly priced rule. λ = 1 = same bits."""
        self._check_method(method)
        cells, w, F, post = self._with_probes()
        pp = post @ F; now_u = float(self._u(pp) @ w)
        two = np.zeros(len(post)); two[self._n_one:] = 1.0
        h = lambda p: 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        now_f = h(float(post @ two)) if len(post) > self._n_one else 0.0

        def exact(c):
            after_u, after_f = self._after_tv(post, F, w, F[:, c], two, h, reliability)
            return (now_u - after_u) + lam * (now_f - after_f)

        if method == "moments" and self._screen_ok(reliability) and math.isfinite(float(lam)):
            r = float(reliability); cs = np.arange(len(cells))
            after_u, pout_p, pout_m = self._screen_after_u(pp, self._second_moments(post, F), w, cs, r)
            if len(post) > self._n_one:
                after_f = self._screen_family(float(post @ two), F.T @ (post * two), cs, r, pout_p, pout_m, self._hvec)
            else:
                after_f = np.zeros(len(cs))
            gains = (now_u - after_u) + lam * (now_f - after_f)
            if np.all(np.isfinite(gains)):
                band = self._SCREEN_BAND * ((self.hi - self.lo) + abs(float(lam)))
                i, g, n = self._resolve(range(len(cells)), gains, band, exact, -1.0)
                if i is not None:
                    self.last_probe_info = {"method": "moments", "path": "moments", "resolved": n}
                    return (float(cells[i].mean()), g)
            self.last_probe_info = {"method": "moments", "path": "direct", "reason": "screen not usable"}
        else:
            self.last_probe_info = {"method": method, "path": "direct"}
        best = (float(cells[0].mean()), -1.0)
        for c in range(len(cells)):
            gain = exact(c)
            if gain > best[1]:
                best = (float(cells[c].mean()), gain)
        return best

    def model_check_probe(self, reliability: float = 0.95, *, method: str = "direct") -> tuple[float, float]:
        """(x, expected drop of the entropy of the FAMILY indicator one-vs-two transitions). The value rule `best_probe`
        buys probes that lower the sign potential; when the one-transition family already explains the claims it never
        buys the probes that would expose a second transition (e21: 0 of 25 two-transition pairs flagged). This probe is
        bought against the model error instead: the expected drop of H(P(two transitions)) over the two outcomes, exact
        on the fixed partition. Zero when the collision family is off (p_two = 0)."""
        self._check_method(method)
        cells, w, F, post = self._with_probes()
        if len(post) == self._n_one:
            self.last_probe_info = {"method": method, "path": "no two-transition family"}
            return (0.5 * (self.lo + self.hi), 0.0)
        two = np.zeros(len(post)); two[self._n_one:] = 1.0
        h = lambda p: 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        now = h(float(post @ two))
        if method == "moments" and self._screen_ok(reliability):
            gains = now - self._family_screen(post, F, two, reliability)
            if np.all(np.isfinite(gains)):
                i, g, n = self._resolve(range(len(cells)), gains, self._SCREEN_BAND,
                                        lambda c: now - self._after_mc(post, F[:, c], two, h, reliability), -1.0)
                if i is not None:
                    self.last_probe_info = {"method": "moments", "path": "moments", "resolved": n}
                    return (float(cells[i].mean()), g)
            self.last_probe_info = {"method": "moments", "path": "direct", "reason": "screen not usable"}
        else:
            self.last_probe_info = {"method": method, "path": "direct"}
        best = (float(cells[0].mean()), -1.0)
        for c in range(len(cells)):
            after = self._after_mc(post, F[:, c], two, h, reliability)
            if now - after > best[1]:
                best = (float(cells[c].mean()), now - after)
        return best

    def _family_screen(self, post, F, two, reliability):
        """Screened E[H_family after] for every cell: needs only P2 = post·two and t = Fᵀ(post·two), O(H·Q) once."""
        r = float(reliability); a = 2.0 * r - 1.0
        pp = post @ F
        pout_p = (1.0 - r) + a * pp
        pout_m = r - a * pp
        return self._screen_family(float(post @ two), F.T @ (post * two), np.arange(F.shape[1]), r,
                                   pout_p, pout_m, self._hvec)

    def model_check_pair(self, reliability: float = 0.95, *, method: str = "direct") -> tuple[tuple[float, float], float]:
        """((x1, x2), expected drop of H(P(two transitions)) over the FOUR joint outcomes) — the two probes bought
        together. `model_check_probe` buys one probe at a time, and a second transition is only exposed by probes on
        both sides of both transitions: after one answer the single-transition family usually still explains everything,
        so the next guard probe goes to another pair (e21: the guard share flags 3-4 of 25 two-transition pairs, 21 are
        missed). Exact on the fixed partition — the same point likelihood as `model_check_probe`, applied twice, so the
        predicted gain is the realized expectation (test). Cost: cells² × 4 outcomes, so the candidates are the top-12
        cells by single-probe model-check gain. Zero when the collision family is off (p_two = 0)."""
        cells, w, F, post = self._with_probes()
        mid = 0.5 * (self.lo + self.hi)
        if len(post) == self._n_one:
            self.last_probe_info = {"method": method, "path": "no two-transition family"}
            return ((mid, mid), 0.0)
        two = np.zeros(len(post)); two[self._n_one:] = 1.0
        h = lambda p: 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        now = h(float(post @ two))

        def like_of(f, sg):
            ff = f if sg > 0 else 1 - f
            return ff * reliability + (1 - ff) * (1 - reliability)

        self._check_method(method)
        rows = range(len(cells))
        if method == "moments" and self._screen_ok(reliability):
            # the single stage only needs the exact top-12 order: re-decide exactly every cell whose screened
            # gain is within the band of the 12th screened value or above it; the rest cannot enter the top 12.
            scr = now - self._family_screen(post, F, two, reliability)
            if np.all(np.isfinite(scr)):
                kth = float(np.sort(scr)[::-1][min(12, len(scr)) - 1])
                rows = [c for c in range(len(cells)) if scr[c] >= kth - self._SCREEN_BAND]
                self.last_probe_info = {"method": "moments", "path": "moments", "resolved": len(rows)}
            else:
                self.last_probe_info = {"method": "moments", "path": "direct", "reason": "screen not usable"}
        else:
            self.last_probe_info = {"method": method, "path": "direct"}
        single = []
        for c in rows:
            f = F[:, c]; after = 0.0
            for sg in (1, -1):
                like = like_of(f, sg)
                pout = float(post @ like)
                if pout <= 0:
                    continue
                after += pout * h(float((post * like / pout) @ two))
            single.append((now - after, c))
        cand = [c for _, c in sorted(single, reverse=True)[:12]]
        best = ((float(cells[cand[0]].mean()), float(cells[cand[0]].mean())), -1.0)

        def pair_after(c1, c2):
            f1 = F[:, c1]
            f2 = F[:, c2]; after = 0.0
            for s1 in (1, -1):
                l1 = like_of(f1, s1)
                p1 = float(post @ l1)
                if p1 <= 0:
                    continue
                q1 = post * l1 / p1
                for s2 in (1, -1):
                    l2 = like_of(f2, s2)
                    p2 = float(q1 @ l2)
                    if p2 <= 0:
                        continue
                    after += p1 * p2 * h(float((q1 * l2 / p2) @ two))
            return after

        pairs = [(c1, c2) for i, c1 in enumerate(cand) for c2 in cand[i + 1:]]
        if method == "moments" and self._screen_ok(reliability) and pairs:
            # four joint answers of two probes from the restricted second moments (plain and family-weighted)
            r = float(reliability); a = 2.0 * r - 1.0
            cc = np.asarray(cand)
            Fc = F[:, cc]
            S = Fc.T @ (post[:, None] * Fc)
            pt = post * two
            S2 = Fc.T @ (pt[:, None] * Fc)
            m = post @ Fc; t = pt @ Fc; P2 = float(pt.sum())
            pos = {c: k for k, c in enumerate(cand)}
            i1 = np.array([pos[c1] for c1, _ in pairs]); i2 = np.array([pos[c2] for _, c2 in pairs])
            scr_after = np.zeros(len(pairs))
            for s1 in (1, -1):
                al1, be1 = ((1.0 - r), a) if s1 > 0 else (r, -a)
                for s2 in (1, -1):
                    al2, be2 = ((1.0 - r), a) if s2 > 0 else (r, -a)
                    p12 = al1 * al2 + al1 * be2 * m[i2] + be1 * al2 * m[i1] + be1 * be2 * S[i1, i2]
                    t12 = al1 * al2 * P2 + al1 * be2 * t[i2] + be1 * al2 * t[i1] + be1 * be2 * S2[i1, i2]
                    scr_after += np.where(p12 > 0, p12 * self._hvec(np.where(p12 > 0, t12 / np.where(p12 > 0, p12, 1.0), 0.0)), 0.0)
            gains = now - scr_after
            if np.all(np.isfinite(gains)):
                k, g, n = self._resolve(range(len(pairs)), gains, self._SCREEN_BAND,
                                        lambda k: now - pair_after(*pairs[k]), -1.0)
                if k is not None:
                    c1, c2 = pairs[k]
                    x1, x2 = float(cells[c1].mean()), float(cells[c2].mean())
                    self.last_probe_info = dict(self.last_probe_info, pair_path="moments", pair_resolved=n)
                    return ((min(x1, x2), max(x1, x2)), g)
            self.last_probe_info = dict(self.last_probe_info, pair_path="direct")
        for c1, c2 in pairs:
            after = pair_after(c1, c2)
            if now - after > best[1]:
                x1, x2 = float(cells[c1].mean()), float(cells[c2].mean())
                best = ((min(x1, x2), max(x1, x2)), now - after)
        return best

    def bundle_value(self, reliability: float = 0.95, k: int = 2, lam: float = 1.0, top: int = 12) -> dict:
        """Exactly priced BUNDLES: the best single probe and the best PAIR of probes, both valued in the same bits as
        `total_value_probe` (the TOTAL potential U + λ·H_family), so a bundle can compete with a single probe on gain
        per cost and no guard share is needed.

        Why a bundle needs its own price. e33 proved the value rule is one-step by construction and exhibited the
        non-submodular case it cannot see: a probe toward a second transition has ≈ 0 marginal value ALONE (the
        one-transition family still explains the claims) and a large value GIVEN a first probe on the other side —
        increasing returns. e35 gave the exact value of a SET of probes on one belief by the chain rule
        Σ_k [H(X | Y_<k) − H(X | Y_≤k)], which telescopes to the exact set value in any order. Here that set value is
        computed directly: for a pair (x1, x2) the gain is the expected drop of U + λ·H_fam over the FOUR joint
        outcomes, probe 1 then probe 2 with the posterior updated in between — exact on the fixed partition, the same
        computation as `model_check_pair` applied to the total potential instead of the family entropy alone.

        Candidates: the `top` cells by the per-cell single-probe total gain (`total_value_probe`'s objective), so the
        cost is top²/2 × 4 posterior evaluations. Returns
            {"single": (x, gain, 1.0), "pair": ((x1, x2), gain, 2.0), "top_singles": [(x, gain), ...], "now": ...}
        where the third entry is the number of probes: the caller multiplies by the instrument cost c, giving 1·c and
        2·c. `pair` is None for k = 1. Works with the collision family off (p_two = 0); then H_fam ≡ 0 and the price is
        the sign potential alone."""
        if k not in (1, 2):
            raise ValueError(f"bundle_value supports k = 1 or 2: {k}")
        cells, w, F, post = self._with_probes()
        two = np.zeros(len(post))
        has_fam = len(post) > self._n_one
        if has_fam:
            two[self._n_one:] = 1.0
        h = lambda p: 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        now = float(self._u(post @ F) @ w) + (lam * h(float(post @ two)) if has_fam else 0.0)

        def like_of(f, sg):
            ff = f if sg > 0 else 1 - f
            return ff * reliability + (1 - ff) * (1 - reliability)

        def tot(q):                                            # the TOTAL potential of a (normalized) posterior
            return float(self._u(q @ F) @ w) + (lam * h(float(q @ two)) if has_fam else 0.0)

        single = []
        for c in range(len(cells)):
            f = F[:, c]; after = 0.0
            for sg in (1, -1):
                like = like_of(f, sg)
                pout = float(post @ like)
                if pout <= 0:
                    continue
                after += pout * tot(post * like / pout)
            single.append((now - after, c))
        single.sort(reverse=True)
        g1, c1b = single[0]
        out = {"single": (float(cells[c1b].mean()), float(g1), 1.0),
               "top_singles": [(float(cells[c].mean()), float(g)) for g, c in single[:top]],
               "now": now, "pair": None}
        if k == 1:
            return out
        cand = [c for _, c in single[:top]]
        best = (((float(cells[cand[0]].mean()), float(cells[cand[0]].mean()))), -1.0)
        for i, c1 in enumerate(cand):
            f1 = F[:, c1]
            for c2 in cand[i + 1:]:
                f2 = F[:, c2]; after = 0.0
                for s1 in (1, -1):
                    l1 = like_of(f1, s1)
                    p1 = float(post @ l1)
                    if p1 <= 0:
                        continue
                    q1 = post * l1 / p1
                    for s2 in (1, -1):
                        l2 = like_of(f2, s2)
                        p2 = float(q1 @ l2)
                        if p2 <= 0:
                            continue
                        after += p1 * p2 * tot(q1 * l2 / p2)
                if now - after > best[1]:
                    x1, x2 = float(cells[c1].mean()), float(cells[c2].mean())
                    best = ((min(x1, x2), max(x1, x2)), now - after)
        out["pair"] = (best[0], float(best[1]), 2.0)
        return out

    def weakest_direction_probe(self, reliability: float = 0.95, mix: bool = False,
                                eps: float = 0.01, n_max: int = 30, lam: float | None = None) -> tuple[float, float]:
        """E-OPTIMAL probe: (x, gain), the probe that most raises the SMALLEST pairwise discrimination between the
        hypotheses that still carry posterior mass. `best_probe` and `model_check_probe` are D-optimal in spirit — they
        buy the largest expected drop of an entropy (of the sign potential, of the family indicator), which is an
        AVERAGE over the hypothesis space and can be bought entirely along the directions that are already well
        separated. The second transition of a collision pair is the opposite case: one direction that no probe in the
        schedule separates at all. E-optimality (max-min) is the criterion that cannot ignore it.

        Concretely. At a probe in cell c with reliability r a hypothesis h emits + with probability
        a_h = r·F[h,c] + (1−r)(1−F[h,c]) — a two-outcome distribution, so the expected log-likelihood-ratio
        contribution of that probe to telling h from h' is the KL divergence between Bern(a_h) and Bern(a_h'). Summed
        over the probes already made (each with its own cell, reliability and weight) that gives a discrimination
        matrix D(h,h'); symmetrized (Jeffreys, KL(h‖h') + KL(h'‖h)) because the pair is unordered. The candidate x
        contributes ΔD(h,h') ≥ 0 exactly — KL is already an expectation over the two outcomes, so no outcome average
        is needed and the gain is deterministic:
            gain(x) = min_{h,h'} [D + ΔD(x)] − min_{h,h'} D.
        Hypotheses with posterior < `eps` are dropped (they are not directions anyone is being confused along) and the
        set is capped at the `n_max` most probable, so the cost is cells × n_max².

        mix=True returns the MIXED rule, value = (expected entropy drop, exactly `best_probe`'s objective)
        + λ·(min-discrimination increase), with λ fixed on the first call at max-drop / max-gain over the candidates
        so the two terms have equal scale on the first step; the same λ is then reused (cached on the object) so later
        steps are compared on the first step's units. Pass `lam` to fix it by hand.
        Returns (midpoint, 0.0) when fewer than two hypotheses carry mass."""
        cells, w, F, post = self._with_probes()
        mid = 0.5 * (self.lo + self.hi)
        keep = np.where(post >= eps)[0]
        if len(keep) > n_max:
            keep = keep[np.argsort(post[keep])[::-1][:n_max]]
        if len(keep) < 2:                        # early on the mass is spread over every threshold cell and nothing
            if len(post) < 2:                    # clears ε; fall back to the two most probable so the rule is defined
                return (mid, 0.0)
            keep = np.argsort(post)[::-1][:min(n_max, len(post))]
        Fk = F[keep]

        def dmat(c: float, r: float, wt: float = 1.0) -> np.ndarray:
            """Jeffreys divergence between the two-outcome emissions of every pair of kept hypotheses at cell c."""
            a = np.clip(r * Fk[:, c] + (1 - r) * (1 - Fk[:, c]), 1e-12, 1 - 1e-12)
            la, lb = np.log(a), np.log(1 - a)
            return wt * (a[:, None] - a[None, :]) * ((la[:, None] - la[None, :]) - (lb[:, None] - lb[None, :]))

        D = np.zeros((len(keep), len(keep)))
        for x, _sg, rel, wt in self.probes:
            c = min(int(np.searchsorted(cells[:, 1], x, side="left")), len(cells) - 1)
            D += dmat(c, rel, wt)
        iu = np.triu_indices(len(keep), 1)
        now_min = float(D[iu].min())
        xs = list(cells.mean(1)) + [self.lo, self.hi]
        cs = list(range(len(cells))) + [0, len(cells) - 1]
        weak = D[iu] <= now_min + 1e-9                    # the tied weakest directions (usually many: each pair of
        gains, ties = [], []                              # neighbouring thresholds is separated only by its own cell)
        for c in cs:
            dd = dmat(c, reliability)[iu]
            gains.append(float((D[iu] + dd).min()) - now_min)
            ties.append(float(dd[weak].mean()))
        gains, ties = np.array(gains), np.array(ties)
        if gains.max() <= 1e-12:
            # Max-min is FLAT here: the minimum is attained by many pairs at once and one probe lifts only one of them,
            # so the exact one-step increase of min D is 0 for every candidate and the criterion cannot choose. Rank by
            # the mean lift of the tied weakest set instead (the flat-limit derivative of the soft-min surrogate
            # −β⁻¹ log Σ e^{−βD}); same units, and it reduces to the exact rule as soon as one direction is strictly
            # weakest. Reported as the value, so a caller comparing pairs compares the same quantity.
            gains = ties
        if not mix:
            k = int(np.argmax(gains))
            return (float(xs[k]), float(gains[k]))
        pp = post @ F
        now = float(self._u(pp) @ w)
        drops = []
        for c in cs:
            f = F[:, c]
            after = 0.0
            for sg in (1, -1):
                ff = f if sg > 0 else 1 - f
                like = ff * reliability + (1 - ff) * (1 - reliability)
                pout = float(post @ like)
                if pout <= 0:
                    continue
                after += pout * float(self._u((post * like / pout) @ F) @ w)
            drops.append(now - after)
        drops = np.array(drops)
        if lam is None:
            lam = getattr(self, "_eopt_lambda", None)
        if lam is None:
            lam = float(drops.max() / gains.max()) if gains.max() > 0 else 0.0
            self._eopt_lambda = lam
        val = drops + lam * gains
        k = int(np.argmax(val))
        return (float(xs[k]), float(val[k]))

    def collision(self, flag_at: float = 0.5) -> dict:
        """Posterior mass on the collision family (two transitions) and the Bayes factor against
        the declared single-transition reading. A pair is flagged when that mass passes `flag_at`:
        report it as an open gauge, do not report a unique regime boundary."""
        _, _, _, post = self._with_probes()
        m2 = float(post[self._n_one:].sum())
        prior = self.p_two if len(post) > self._n_one else 0.0
        bf = (m2 / max(1 - m2, 1e-300)) / (prior / (1 - prior)) if 0 < prior < 1 else float("nan")
        return {"p_two_transitions": m2, "bayes_factor_two_vs_one": bf, "flag": m2 >= flag_at}
