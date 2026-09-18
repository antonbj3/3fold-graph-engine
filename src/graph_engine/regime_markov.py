#!/usr/bin/env python3
"""
regime_markov.py — the sign along a condition variable as a two-state continuous-time Markov chain
(telegraph process), read by forward-backward instead of by enumerating sign hypotheses.

THE MODEL. On x ∈ [lo, hi] the sign s(x) ∈ {+1, −1} is a telegraph process: switches arrive as a Poisson
process of rate λ per unit x, s flips at each switch, and s(lo) is ± with probability ½ each. The number of
switches in the domain is Poisson(λ·(hi−lo)); λ is set from a prior on the expected number of transitions,
E[N] = p_flip + 2·p_two (the same two numbers RegimePosterior puts on its one- and two-transition families),
so the default λ = 0.4 per unit x. `lam` is exposed and overrides it.

WHAT IS EXACT.
  * The chain is discretised on the SAME fixed partition as RegimePosterior: the claim edges plus `n_grid`
    uniform cells. The state is the sign at the cell MIDPOINTS; the exact telegraph transition probability
    between two midpoints a distance d apart is q = (1 − e^{−2λd})/2, which is what the chain uses. So this
    is not a coarse approximation of the transition law — it is the exact law of the midpoint process. What
    the discretisation drops is switch PAIRS between two midpoints (≈ (λd)²/2 each) and switches in the two END
    half-cells, which no midpoint pair straddles (≈ λ(hi−lo)/n_grid): together 0.0067 of prior switch-count mass
    at λ = 0.4, n_grid = 48, all of it landing on N = 0 (test).
  * Given the cell states, the posterior over the sign of every cell, the joint distribution of the number of
    switches, and the probe likelihoods are computed by EXACT forward-backward (sum-product) on the chain:
    no sampling, no variational step, no hypothesis enumeration. Cost O(cells · cap²) per evaluation, cap =
    `count_cap` (the switch count is carried in the message so that P(N = k) comes out of the same pass).
  * Point observations (probes) are exact: a probe at x with reliability r contributes the emission r to the
    cell state that agrees with its answer and 1 − r to the other.
  * MAJORITY CLAIMS are exact (see next section), at a cost that is not O(cells): a box claim is NOT a product
    of per-cell emissions, and the exact reading needs the joint law of the path inside the box.

THE MAJORITY CLAIM, AND WHY IT IS NOT A PRODUCT. A source that reports one sign σ for a whole box [a, b] with
reliability r reports the MAJORITY sign of its box: P(claim | path) = r if the majority of [a, b] (by measure)
has sign σ, else 1 − r. That is a function of the whole path inside the box, not of one point, so it does not
factor over cells. closed_loop's World generates its claims exactly this way (World.majority_sign), and e21
located the loop's over-confidence in reading such a report as an r-accurate label at every point of the box.
Exact treatment here (claim_model="majority", the default):
  overlapping claim boxes are merged into BLOCKS of consecutive cells. Inside a block every path is enumerated
  by its switch set — the subsets of the block's interior cell boundaries of size ≤ `max_switches_in_block` —
  together with the entry sign. For each enumerated path the box majority of every claim in the block is a
  finite computation (width-weighted share of the box on which the path has sign σ), so the claim likelihood
  is exactly r or 1 − r, and the probe emissions inside the block multiply in. Summing the enumerated paths
  by (entry sign, exit sign, internal switch count) gives a 2 × 2 × (cap+1) transfer tensor for the block; the
  rest of the domain uses the ordinary chain transfer. Forward-backward over blocks and free cells is then
  exact for the model
      TELEGRAPH, CONDITIONED ON AT MOST `max_switches_in_block` SWITCHES STRICTLY INSIDE EACH CLAIM BLOCK.
  That conditioning is the one modelling liberty taken, and it is measurable, not hidden: `tail_mass()`
  returns the prior probability the conditioning removes (Poisson-binomial over the block's boundaries).
  At the default λ = 0.4 and cap 3 it is of order 1e−3 on a block spanning the whole domain.
  claim_model="pointwise" is the documented approximation instead: the claim enters as the per-cell emissions
  r^{n·ov/Σov}, (1−r)^{n·ov/Σov} (RegimePosterior's "pointwise" reading). It needs no enumeration and the whole
  read is then O(cells); the difference between the two readings is what e28 measures.

WHAT THIS BUYS OVER RegimePosterior. RegimePosterior enumerates {all +, all −, one transition at t, two
transitions at (t₁, t₂)} and puts a prior mass on each family: the two-transition family is a discrete
alternative with prior `p_two`, and three or more transitions are outside the space. Here the same objects are
moments of ONE process: P(N = 0), P(N = 1), P(N ≥ 2) are read off the switch-count distribution of the chain,
no family is privileged, and the cost is linear in the number of cells instead of quadratic in the number of
candidate transition points (the two-transition family alone is O(cells²) hypotheses).

READS (same names and units as RegimePosterior, so the two are directly comparable):
  p_plus(x)                 P(sign = + at x)
  potential_value()         U = ∫ u(P₊(x)) dx, u = binary entropy (bits) or min(p, 1−p)
  expected_error()          ∫ min(P₊, 1−P₊) dx — the measure the reader expects to get wrong
  best_probe(r)             (x, expected drop of U), exact over the two answers, one candidate per cell
  n_transitions_posterior() array P(N = k), k = 0 … count_cap, the last entry being P(N ≥ count_cap)
  collision()               P(N ≥ 2) and the Bayes factor against the prior odds of the same event
Because the hypothesis space is fixed (probes never add cells) the posterior is a martingale and the value
`best_probe` predicts is the realized expectation over the two answers (test).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

__all__ = ["RegimeMarkov"]


def _count_conv(a: np.ndarray, b: np.ndarray, cap: int) -> np.ndarray:
    """Convolve two switch-count vectors, the last entry being the ≥ cap bucket."""
    out = np.zeros(cap + 1)
    for i in range(cap + 1):
        if a[i] == 0.0:
            continue
        for j in range(cap + 1):
            if b[j] != 0.0:
                out[min(i + j, cap)] += a[i] * b[j]
    return out


@dataclass
class RegimeMarkov:
    lo: float
    hi: float
    reliability: float = 0.75                # default claim reliability
    p_flip: float = 0.3                      # prior P(one transition) — only used to set λ
    p_two: float = 0.05                      # prior P(two transitions) — only used to set λ
    lam: float | None = None                 # switching rate per unit x; None → (p_flip + 2 p_two)/(hi − lo)
    potential: str = "entropy"               # "entropy" (bits) or "error" = min(p, 1−p)
    n_grid: int = 48                         # the fixed partition, as in RegimePosterior
    claim_model: str = "majority"            # "majority" (exact, block enumeration) | "pointwise" (approximation)
    max_switches_in_block: int = 3           # the conditioning; `tail_mass()` reports what it removes
    count_cap: int = 6                       # switch counts are carried up to this, the last bucket is ≥ cap
    pattern_cap: int = 250_000               # enumeration budget per block; the switch cap is lowered to fit
    claims: list = field(default_factory=list)   # (a, b, sign, n_eff, reliability)
    probes: list = field(default_factory=list)   # (x, sign, reliability, weight)
    _geom: tuple | None = None
    _units: list | None = None
    _basecache: list | None = None

    def __post_init__(self):
        if self.hi <= self.lo:
            raise ValueError("hi must exceed lo")
        if self.lam is None:
            self.lam = (self.p_flip + 2.0 * self.p_two) / (self.hi - self.lo)
        self.lam = float(self.lam)
        if self.lam < 0:
            raise ValueError("lam must be ≥ 0")
        if self.claim_model not in ("majority", "pointwise"):
            raise ValueError(self.claim_model)

    # -- inputs ------------------------------------------------------------------------------------
    def add_claim(self, a: float, b: float, sign: int, n_eff: float = 1.0, reliability: float | None = None) -> None:
        r = self.reliability if reliability is None else float(reliability)
        if not 0.5 <= r < 1.0:
            raise ValueError(f"claim reliability must be in [0.5, 1): {r}")
        self.claims.append((max(a, self.lo), min(b, self.hi), 1 if sign > 0 else -1, float(n_eff), r))
        self._geom = None
        self._units = None
        self._basecache = None

    def set_claim_reliabilities(self, rs: list[float]) -> None:
        assert len(rs) == len(self.claims)
        self.claims = [(a, b, sg, n, float(r)) for (a, b, sg, n, _), r in zip(self.claims, rs)]
        self._units = None
        self._basecache = None

    def add_probe(self, x: float, sign: int, reliability: float = 0.95, weight: float = 1.0) -> None:
        if not 0.5 <= reliability < 1.0:
            raise ValueError(f"probe reliability must be in [0.5, 1): {reliability}")
        self.probes.append((float(x), 1 if sign > 0 else -1, float(reliability), float(weight)))
        self._basecache = None

    # -- geometry ----------------------------------------------------------------------------------
    def _geometry(self):
        """cells, widths, midpoints, and the exact midpoint-to-midpoint switch probabilities."""
        if self._geom is not None:
            return self._geom
        pts = {self.lo, self.hi}
        for a, b, *_ in self.claims:
            pts.update((a, b))
        pts.update(np.linspace(self.lo, self.hi, self.n_grid + 1).tolist())
        edges = np.array(sorted(pts))
        cells = np.stack([edges[:-1], edges[1:]], 1)
        cells = cells[cells[:, 1] > cells[:, 0]]
        w = cells[:, 1] - cells[:, 0]
        mid = cells.mean(1)
        d = np.diff(mid)
        q = 0.5 * (1.0 - np.exp(-2.0 * self.lam * d))            # P(sign differs) between adjacent midpoints
        self._geom = (cells, w, mid, q)
        return self._geom

    def _cell_of(self, x: float) -> int:
        cells, *_ = self._geometry()
        return int(min(np.searchsorted(cells[:, 1], x, side="left"), len(cells) - 1))

    def _claim_cells(self, a: float, b: float):
        """(cell indices, normalized overlap weights) for a claim box."""
        cells, *_ = self._geometry()
        ov = np.clip(np.minimum(cells[:, 1], b) - np.maximum(cells[:, 0], a), 0, None)
        if ov.sum() <= 0:
            ov = ((cells[:, 0] <= a) & (a <= cells[:, 1])).astype(float)
        idx = np.flatnonzero(ov > 0)
        return idx, ov[idx] / ov[idx].sum()

    # -- units: free cells and enumerated claim blocks ----------------------------------------------
    def _build_units(self):
        """Split the cells into blocks (merged claim boxes, path-enumerated) and free single cells."""
        if self._units is not None:
            return self._units
        cells, w, mid, q = self._geometry()
        C = len(cells)
        spans = []
        if self.claim_model == "majority":
            for a, b, sg, n, r in self.claims:
                idx, _ = self._claim_cells(a, b)
                spans.append((int(idx[0]), int(idx[-1])))
        spans.sort()
        merged: list[list[int]] = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        units = []
        c = 0
        for s, e in merged:
            while c < s:
                units.append({"kind": "cell", "i0": c, "i1": c})
                c += 1
            units.append(self._enumerate_block(s, e))
            c = e + 1
        while c < C:
            units.append({"kind": "cell", "i0": c, "i1": c})
            c += 1
        self._units = units
        return units

    def _enumerate_block(self, s: int, e: int) -> dict:
        """Enumerate the paths inside cells s..e with ≤ K interior switches; attach the claim log-likelihoods."""
        cells, w, mid, q = self._geometry()
        m = e - s + 1
        nb = m - 1                                              # interior boundaries
        K = min(self.max_switches_in_block, nb)
        while K > 0 and 2 * sum(math.comb(nb, j) for j in range(K + 1)) > self.pattern_cap:
            K -= 1
        sets = [c for j in range(K + 1) for c in combinations(range(nb), j)]
        npat_half = len(sets)
        flips = np.zeros((npat_half, nb), dtype=np.int8)
        for i, c in enumerate(sets):
            for b in c:
                flips[i, b] = 1
        par = np.concatenate([np.zeros((npat_half, 1), np.int8), np.cumsum(flips, 1, dtype=np.int8) % 2], 1)
        sig_up = np.where(par == 0, 1, -1).astype(np.int8)       # entry sign +1
        signs = np.concatenate([sig_up, -sig_up], 0).astype(np.int8)   # then entry sign −1
        nsw = np.concatenate([flips.sum(1), flips.sum(1)]).astype(int)
        qb = q[s:e] if nb else np.zeros(0)
        lw = np.zeros(npat_half)
        if nb:
            base = float(np.log1p(-qb).sum())
            ratio = np.log(np.maximum(qb, 1e-300)) - np.log1p(-qb)
            lw = base + flips.astype(float) @ ratio
        logw = np.concatenate([lw, lw])                          # entry sign prior (½ each) lives in the message
        plusT = np.ascontiguousarray((signs > 0).T.astype(np.float64))   # (m, npat): contiguous per CELL
        # claim likelihoods: exact majority of each box, per enumerated path
        lc = np.zeros(len(signs))
        for a, b, sg, n, r in self.claims:
            idx, ovn = self._claim_cells(a, b)
            if idx[0] < s or idx[-1] > e:
                continue
            fplus = ovn @ plusT[idx - s]
            f = fplus if sg > 0 else 1.0 - fplus
            like = np.where(f > 0.5 + 1e-12, r, np.where(f < 0.5 - 1e-12, 1.0 - r, 0.5))
            lc += n * np.log(like)
        # state index 0 = sign +, 1 = sign − everywhere (the convention of the emission columns)
        s0i, s1i = (signs[:, 0] < 0).astype(int), (signs[:, -1] < 0).astype(int)
        nswc = np.minimum(nsw, self.count_cap)
        return {"kind": "block", "i0": s, "i1": e, "plusT": plusT, "logw": logw + lc, "nsw": nswc,
                "s0": s0i, "s1": s1i, "g": (s0i * 2 + s1i) * (self.count_cap + 1) + nswc, "K": K, "nb": nb}

    # -- emissions ---------------------------------------------------------------------------------
    def _log_emission(self, extra=None) -> np.ndarray:
        """(cells, 2) log-likelihood of the observations assigned to each cell, column 0 = sign +, 1 = sign −."""
        cells, w, mid, q = self._geometry()
        E = np.zeros((len(cells), 2))
        for x, sg, r, wt in list(self.probes) + ([extra] if extra is not None else []):
            c = self._cell_of(x)
            E[c, 0] += wt * math.log(r if sg > 0 else 1 - r)
            E[c, 1] += wt * math.log(1 - r if sg > 0 else r)
        if self.claim_model == "pointwise":
            for a, b, sg, n, r in self.claims:
                idx, ovn = self._claim_cells(a, b)
                lr, lq = math.log(r), math.log(1 - r)
                E[idx, 0] += n * ovn * (lr if sg > 0 else lq)
                E[idx, 1] += n * ovn * (lq if sg > 0 else lr)
        return E

    def _base_logs(self):
        """Per-unit log-weights under the CURRENT claims and probes; a hypothetical probe only edits one unit."""
        if getattr(self, "_basecache", None) is not None:
            return self._basecache
        units = self._build_units()
        E = self._log_emission()
        base = []
        for u in units:
            if u["kind"] == "cell":
                base.append(E[u["i0"]].copy())
            else:
                s, e0 = u["i0"], u["i1"]
                Eb = E[s:e0 + 1]
                lp = u["logw"].copy()
                act = np.flatnonzero(np.abs(Eb).sum(1) > 0)
                if len(act):
                    lp = lp + Eb[act, 0] @ u["plusT"][act] + Eb[act, 1] @ (1.0 - u["plusT"][act])
                base.append(lp)
        self._basecache = base
        return base

    # -- forward-backward ---------------------------------------------------------------------------
    def _solve(self, extra=None, need_counts: bool = True):
        """Exact sum-product over units. Returns (p_plus per cell, P(N = k) or None, widths)."""
        cells, w, mid, q = self._geometry()
        units = self._build_units()
        base = self._base_logs()
        cap = self.count_cap
        hit, e_plus, e_minus = -1, 0.0, 0.0
        if extra is not None:
            x, sg, r, wt = extra
            hit = self._cell_of(x)
            e_plus = wt * math.log(r if sg > 0 else 1 - r)
            e_minus = wt * math.log(1 - r if sg > 0 else r)
        T2, Tk, loc = [], [], []
        for ui, u in enumerate(units):
            if u["kind"] == "cell":
                el = base[ui]
                if hit == u["i0"]:
                    el = el + np.array([e_plus, e_minus])
                e = np.exp(el - el.max())
                T2.append(np.diag(e))
                loc.append(None)
                if need_counts:
                    tk = np.zeros((2, 2, cap + 1))
                    tk[0, 0, 0], tk[1, 1, 0] = e[0], e[1]
                    Tk.append(tk)
                else:
                    Tk.append(None)
            else:
                lp = base[ui]
                if u["i0"] <= hit <= u["i1"]:
                    col = u["plusT"][hit - u["i0"]]
                    lp = lp + col * e_plus + (1.0 - col) * e_minus
                wp = np.exp(lp - lp.max())
                tk = np.bincount(u["g"], weights=wp, minlength=4 * (cap + 1)).reshape(2, 2, cap + 1)
                T2.append(tk.sum(2)); Tk.append(tk if need_counts else None); loc.append(wp)
        # forward / backward, count-marginalized (for the cell posteriors)
        U = len(units)
        Fm = [np.array([0.5, 0.5])]
        for i in range(U - 1):
            out = Fm[-1] @ T2[i]
            qb = q[units[i]["i1"]]
            Fm.append(out @ np.array([[1 - qb, qb], [qb, 1 - qb]]))
        Bm = [None] * U
        Bm[U - 1] = np.ones(2)
        for i in range(U - 2, -1, -1):
            qb = q[units[i]["i1"]]
            Bm[i] = np.array([[1 - qb, qb], [qb, 1 - qb]]) @ (T2[i + 1] @ Bm[i + 1])
        Z = float(Fm[0] @ T2[0] @ Bm[0])
        pplus = np.zeros(len(cells))
        for i, u in enumerate(units):
            if u["kind"] == "cell":
                pplus[u["i0"]] = Fm[i][0] * T2[i][0, 0] * Bm[i][0] / Z
            else:
                coef = loc[i] * Fm[i][u["s0"]] * Bm[i][u["s1"]]
                pplus[u["i0"]:u["i1"] + 1] = (u["plusT"] @ coef) / Z
        if not need_counts:
            return np.clip(pplus, 0.0, 1.0), None, w
        # forward with the switch count carried, for P(N = k)
        Fk = np.zeros((2, cap + 1)); Fk[:, 0] = 0.5
        for i in range(U):
            out = np.zeros((2, cap + 1))
            for s0 in (0, 1):
                for s1 in (0, 1):
                    out[s1] += _count_conv(Fk[s0], Tk[i][s0, s1], cap)
            if i < U - 1:
                qb = q[units[i]["i1"]]
                nxt = np.zeros((2, cap + 1))
                for s in (0, 1):
                    nxt[s] += out[s] * (1 - qb)
                    shifted = np.zeros(cap + 1)
                    shifted[1:] = out[1 - s][:-1] * qb
                    shifted[cap] += out[1 - s][cap] * qb
                    nxt[s] += shifted
                Fk = nxt
            else:
                Fk = out
        pn = Fk.sum(0)
        pn = pn / pn.sum()
        return np.clip(pplus, 0.0, 1.0), pn, w

    # -- reads -------------------------------------------------------------------------------------
    def p_plus(self, x: float) -> float:
        pp, _, _ = self._solve(need_counts=False)
        return float(pp[self._cell_of(x)])

    def p_plus_cells(self) -> np.ndarray:
        return self._solve(need_counts=False)[0]

    def _u(self, pp: np.ndarray) -> np.ndarray:
        if self.potential == "error":
            return np.minimum(pp, 1 - pp)
        qq = np.clip(pp, 1e-12, 1 - 1e-12)
        return -(qq * np.log2(qq) + (1 - qq) * np.log2(1 - qq))

    def potential_value(self) -> float:
        pp, _, w = self._solve(need_counts=False)
        return float(self._u(pp) @ w)

    def expected_error(self) -> float:
        pp, _, w = self._solve(need_counts=False)
        return float(np.minimum(pp, 1 - pp) @ w)

    def best_probe(self, reliability: float = 0.95) -> tuple[float, float]:
        """(x, expected drop of the potential) — exact over the two answers, one candidate per cell."""
        cells, w, mid, _ = self._geometry()
        pp, _, _ = self._solve(need_counts=False)
        now = float(self._u(pp) @ w)
        best = (float(mid[0]), -1.0)
        for c in range(len(cells)):
            after = 0.0
            for sg in (1, -1):
                p = pp[c] if sg > 0 else 1 - pp[c]
                pout = p * reliability + (1 - p) * (1 - reliability)
                if pout <= 0:
                    continue
                qpp, _, _ = self._solve(extra=(float(mid[c]), sg, float(reliability), 1.0), need_counts=False)
                after += pout * float(self._u(qpp) @ w)
            if now - after > best[1]:
                best = (float(mid[c]), now - after)
        return best

    def n_transitions_posterior(self) -> np.ndarray:
        """P(N = k) for k = 0 … count_cap, the last entry being P(N ≥ count_cap)."""
        _, pn, _ = self._solve(need_counts=True)
        return pn

    def collision(self, flag_at: float = 0.5) -> dict:
        """P(≥ 2 switches) and the Bayes factor against the same event's prior odds under the chain."""
        pn = self.n_transitions_posterior()
        m2 = float(pn[2:].sum())
        prior = self._prior_n()
        p0 = float(prior[2:].sum())
        bf = (m2 / max(1 - m2, 1e-300)) / (p0 / max(1 - p0, 1e-300)) if 0 < p0 < 1 else float("nan")
        return {"p_two_or_more": m2, "p_ge_two": m2, "prior_p_ge_two": p0,
                "bayes_factor_two_vs_one": bf, "flag": m2 >= flag_at}

    def _prior_n(self) -> np.ndarray:
        """Prior switch-count distribution of the DISCRETISED chain (no claims, no probes, no conditioning)."""
        _, _, _, q = self._geometry()
        cap = self.count_cap
        pn = np.zeros(cap + 1); pn[0] = 1.0
        for qb in q:
            nxt = pn * (1 - qb)
            nxt[1:] += pn[:-1] * qb
            nxt[cap] += pn[cap] * qb
            pn = nxt
        return pn

    def tail_mass(self) -> float:
        """Prior probability that the block conditioning removes: P(> max_switches_in_block switches inside
        some claim block), Poisson-binomial over each block's interior boundaries, union bound over blocks."""
        if self.claim_model != "majority":
            return 0.0
        _, _, _, q = self._geometry()
        tot = 0.0
        for u in self._build_units():
            if u["kind"] != "block" or u["nb"] == 0:
                continue
            pn = np.array([1.0])
            for qb in q[u["i0"]:u["i1"]]:
                nxt = np.zeros(len(pn) + 1)
                nxt[:-1] += pn * (1 - qb)
                nxt[1:] += pn * qb
                pn = nxt
            tot += float(pn[u["K"] + 1:].sum())
        return tot
