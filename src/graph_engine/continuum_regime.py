"""continuum_regime.py — resolution-free (n_grid -> infinity) backend for the finite regime family.

Same model as :class:`graph_engine.compiled_regime.CompiledRegime` (pointwise claims, <= 2
transitions, declared conditional independence) in the limit of an infinitely fine uniform grid:

* start sign +/- with probability 1/2 each, k in {0,1,2} transitions with the SAME effective
  prior masses (``p0, p1 >= 1e-12``, two-transition mass ``>= 1e-300``);
* transition positions uniform in LENGTH on [lo, hi] (k = 2: uniform on the triangle t1 < t2);
* claim (a, b, sign, n_eff, r): log-likelihood density ``n*sign*lambda/(2(b-a)) * S(x)`` on [a, b]
  plus the constant ``0.5*n*log(r(1-r))``, lambda = log(r/(1-r));
* probe (x, sign, r, w): log-likelihood ``w*log(r)`` if S(x) = sign else ``w*log(1-r)``.

Evidence enters only through the potential G(x) (piecewise linear between evidence edges, a jump at
every probe). Every integral of exp(+-2G) over a run has a closed form, so the posterior costs
O(R) for R evidence runs, and a query costs O(log R) from cached prefix/suffix tables. There is no
partition, hence no :class:`FrozenPartitionError`: a claim with new bounds is ordinary evidence and
a new edge that carries no evidence changes nothing (refinement consistency).

Convergence: native ``CompiledRegime(n_grid=n)`` converges to this posterior at O(1/n) (checked
against the native kernel); the default n_grid=48 differs by up to ~0.05 in p_plus and has
finite jumps when a claim edge lands next to a grid point, which this backend does not have.

Closed forms: p_plus(x), family_mass, log_evidence. Numerical quadrature (declared): integrals of
nonlinear functions of p_plus (entropy/error potential) and moments of the transition location use
Gauss-Legendre per run piece on an analytic integrand; the zeroth location moment is checked
against the closed-form normalizer and the discrepancy is reported.
"""
from __future__ import annotations

import math

import numpy as np

from .compiled_regime import (
    CompiledRegimeError, _as_finite_float, _check_reliability, _check_sign, majority_unsupported,
    _MIN_PRIOR, _MIN_TWO_PRIOR, _LN_HALF)

__all__ = ["ContinuumRegime", "continuum_tables", "continuum_query"]

_NINF = -np.inf
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)


# ---------------------------------------------------------------------------------- closed forms
def _log_e(beta, ell):
    """log int_0^ell exp(beta t) dt (vectorized); -inf for ell == 0.

    = log(ell) + max(x, 0) + log((1 - e^{-|x|}) / |x|), x = beta*ell (symmetric, no masks).
    """
    beta = np.asarray(beta, float)
    ell = np.asarray(ell, float)
    x = beta * ell
    ax = np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        big = np.log(-np.expm1(-ax)) - np.log(ax)
        small = np.log1p(ax * (-0.5 + ax * (1.0 / 6.0 - ax / 24.0)))
        return np.log(ell) + np.maximum(x, 0.0) + np.where(ax < 1e-4, small, big)


_HCOEF = np.array([(-1.0) ** n / math.factorial(n + 2) for n in range(18)])[::-1]


def _log_h(x):
    """log h(x), h(x) = (x - 1 + e^{-x}) / x^2 = sum_n (-x)^n / (n+2)!, h(0) = 1/2."""
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        series = np.log(np.polyval(_HCOEF, x))
        pos = np.log(x - 1.0 + np.exp(-x)) - 2 * np.log(x)
        neg = -x + np.log1p((x - 1.0) * np.exp(x)) - 2 * np.log(-x)
        return np.where(np.abs(x) <= 0.5, series, np.where(x > 0, pos, neg))


def _log_w(beta_k, d):
    """log of the within-run double integral d^2 h(beta_k d); -inf for d == 0."""
    d = np.asarray(d, float)
    with np.errstate(divide="ignore"):
        return 2 * np.log(d) + _log_h(beta_k * d)


def _cum(x):
    out = np.empty(len(x) + 1)
    out[0] = _NINF
    if len(x):
        out[1:] = np.logaddexp.accumulate(x)
    return out


def _rcum(x):
    return _cum(x[::-1])[::-1]


def continuum_tables(lo, hi, log_mass, claim_arr, probe_arr):
    """Build the cached run table.

    claim_arr: (C,3) rows (a, b, density); probe_arr: (P,2) rows (x, jump). Returns a dict with edges,
    G after each edge, slopes, prefix/suffix log-integrals and the family log-weights.
    """
    pts = [lo, hi]
    if len(claim_arr):
        pts.extend(claim_arr[:, 0].tolist())
        pts.extend(claim_arr[:, 1].tolist())
    if len(probe_arr):
        pts.extend(probe_arr[:, 0].tolist())
    e = np.unique(np.asarray(pts, float))
    R = len(e) - 1
    ell = np.diff(e)
    dens = np.zeros(R)
    if len(claim_arr):
        diff = np.zeros(R + 1)
        np.add.at(diff, np.searchsorted(e, claim_arr[:, 0]), claim_arr[:, 2])
        np.add.at(diff, np.searchsorted(e, claim_arr[:, 1]), -claim_arr[:, 2])
        dens = np.cumsum(diff)[:R]
    jump = np.zeros(R + 1)
    if len(probe_arr):
        np.add.at(jump, np.searchsorted(e, probe_arr[:, 0]), probe_arr[:, 1])
    g = np.empty(R + 1)
    g[0] = jump[0]
    g[1:] = np.cumsum(dens * ell) + np.cumsum(jump[1:]) + jump[0]
    Gh = float(g[R])
    gr = g[:R]
    k2 = 2 * dens
    lAp = 2 * gr + _log_e(k2, ell)
    lAm = -2 * gr + _log_e(-k2, ell)
    lWp = _log_w(k2, ell)
    lWm = _log_w(-k2, ell)
    Pp, Pm = _cum(lAp), _cum(lAm)
    DKp = _cum(np.logaddexp(Pp[:R] + lAm, lWp))
    DKm = _cum(np.logaddexp(Pm[:R] + lAp, lWm))
    Sp, Sm = _rcum(lAp), _rcum(lAm)
    DSp = _rcum(np.logaddexp(lAp + Sm[1:], lWp))
    DSm = _rcum(np.logaddexp(lAm + Sp[1:], lWm))
    L = hi - lo
    lm0, lm1, lm2 = log_mass
    lc = np.array([lm0 - math.log(2), lm1 - math.log(2 * L), lm2 - 2 * math.log(L)])
    lW = np.array([lc[0] + np.logaddexp(Gh, -Gh),
                   lc[1] + np.logaddexp(-Gh + Pp[R], Gh + Pm[R]),
                   lc[2] + np.logaddexp(Gh + DKp[R], -Gh + DKm[R])])
    lZ = float(np.logaddexp.reduce(lW))
    return {"e": e, "g": g, "dens": dens, "Gh": Gh, "Pp": Pp, "Pm": Pm, "DKp": DKp, "DKm": DKm,
            "Sp": Sp, "Sm": Sm, "DSp": DSp, "DSm": DSm, "lc": lc, "lW": lW, "lZ": lZ, "R": R}


def _split(t, xs):
    """Prefix/suffix log-integrals at arbitrary points xs (vectorized, O(log R) each)."""
    e, R = t["e"], t["R"]
    xs = np.asarray(xs, float)
    i = np.clip(np.searchsorted(e, xs, side="right") - 1, 0, R - 1)
    u = xs - e[i]
    v = e[i + 1] - xs
    u = np.maximum(u, 0.0)
    v = np.maximum(v, 0.0)
    gi = t["g"][i]
    k2 = 2 * t["dens"][i]
    gx = gi + t["dens"][i] * u
    aLp = 2 * gi + _log_e(k2, u)
    aLm = -2 * gi + _log_e(-k2, u)
    aRp = 2 * gx + _log_e(k2, v)
    aRm = -2 * gx + _log_e(-k2, v)
    Pq_p = np.logaddexp(t["Pp"][i], aLp)
    Pq_m = np.logaddexp(t["Pm"][i], aLm)
    DKq_p = np.logaddexp(np.logaddexp(t["DKp"][i], t["Pp"][i] + aLm), _log_w(k2, u))
    DKq_m = np.logaddexp(np.logaddexp(t["DKm"][i], t["Pm"][i] + aLp), _log_w(-k2, u))
    Sq_p = np.logaddexp(aRp, t["Sp"][i + 1])
    Sq_m = np.logaddexp(aRm, t["Sm"][i + 1])
    DSq_p = np.logaddexp(np.logaddexp(t["DSp"][i + 1], aRp + t["Sm"][i + 1]), _log_w(k2, v))
    DSq_m = np.logaddexp(np.logaddexp(t["DSm"][i + 1], aRm + t["Sp"][i + 1]), _log_w(-k2, v))
    return {"gx": gx, "Pp": Pq_p, "Pm": Pq_m, "DKp": DKq_p, "DKm": DKq_m,
            "Sp": Sq_p, "Sm": Sq_m, "DSp": DSq_p, "DSm": DSq_m}


def continuum_query(t, xs):
    """P(S(x) = +) at arbitrary points, closed form from cached tables (O(log R) each)."""
    e, R = t["e"], t["R"]
    xs = np.asarray(xs, float)
    i = np.clip(np.searchsorted(e, xs, side="right") - 1, 0, R - 1)
    u = np.maximum(xs - e[i], 0.0)
    v = np.maximum(e[i + 1] - xs, 0.0)
    gi = t["g"][i]
    dn = t["dens"][i]
    k2 = 2 * dn
    both = _log_e(np.concatenate([-k2, k2]), np.concatenate([u, v]))
    wboth = _log_w(np.concatenate([k2, k2]), np.concatenate([u, v]))
    n = len(xs)
    aLm = -2 * gi + both[:n]
    aRp = 2 * (gi + dn * u) + both[n:]
    Pm = np.logaddexp(t["Pm"][i], aLm)
    DKp = np.logaddexp(np.logaddexp(t["DKp"][i], t["Pp"][i] + aLm), wboth[:n])
    Sp = np.logaddexp(aRp, t["Sp"][i + 1])
    DSp = np.logaddexp(np.logaddexp(t["DSp"][i + 1], aRp + t["Sm"][i + 1]), wboth[n:])
    lc, Gh = t["lc"], t["Gh"]
    N1 = lc[1] + np.logaddexp(-Gh + Sp, Gh + Pm)
    N2 = lc[2] + np.logaddexp(Gh + np.logaddexp(DKp, DSp), -Gh + Pm + Sp)
    return np.exp(np.logaddexp(np.logaddexp(lc[0] + Gh, N1), N2) - t["lZ"])


def _nodes(t, max_slope_len=1.0, pieces_min=1):
    """Gauss-Legendre nodes/weights per run piece with |2*slope*piece| <= max_slope_len."""
    e, dens = t["e"], t["dens"]
    ell = np.diff(e)
    npieces = np.maximum(pieces_min, np.ceil(np.abs(2 * dens) * ell / max_slope_len)).astype(int)
    xs, ws = [], []
    for a, l, m in zip(e[:-1], ell, npieces):
        if l <= 0:
            continue
        h = l / m
        base = a + h * np.arange(m)
        xs.append((base[:, None] + (_GL_X[None, :] + 1) * h / 2).ravel())
        ws.append(np.tile(_GL_W * h / 2, m))
    return np.concatenate(xs), np.concatenate(ws)


def location_moments(t, center):
    """log-weight and first two central moments of T = first sign change.

    T = t1 for k >= 1; for k = 0, T = lo if S = + everywhere, else hi (argmin reading:
    slope sign negative left of the optimum). Returns (mean, var, mass_check) where mass_check is
    the relative difference between the quadrature normalizer and the closed-form Z.
    """
    lo, hi = t["e"][0], t["e"][-1]
    xs, ws = _nodes(t)
    q = _split(t, xs)
    lc, Gh = t["lc"], t["Gh"]
    gx = q["gx"]
    lf1 = lc[1] + np.logaddexp(-Gh + 2 * gx, Gh - 2 * gx)
    lf2 = lc[2] + np.logaddexp(Gh + 2 * gx + q["Sm"], -Gh - 2 * gx + q["Sp"])
    lf = np.logaddexp(lf1, lf2) - t["lZ"]
    dens = np.exp(lf) * ws
    p_lo = math.exp(lc[0] + Gh - t["lZ"])
    p_hi = math.exp(lc[0] - Gh - t["lZ"])
    d = xs - center
    m0 = dens.sum() + p_lo + p_hi
    m1 = dens @ d + p_lo * (lo - center) + p_hi * (hi - center)
    m2 = dens @ (d * d) + p_lo * (lo - center) ** 2 + p_hi * (hi - center) ** 2
    mean = m1 / m0
    var = max(m2 / m0 - mean * mean, 0.0)
    return center + mean, var, abs(m0 - 1.0)


# ---------------------------------------------------------------------------------- the regime
class ContinuumRegime:
    """Resolution-free regime posterior with the public read API of ``CompiledRegime``.

    ``backend``/``method`` are accepted for signature compatibility and ignored: no native
    partition kernel is needed. ``n_grid`` is always ``None``.
    """

    _fixed_configuration = frozenset({"lo", "hi", "reliability", "p_flip", "p_two",
                                      "potential", "n_grid", "claim_model"})

    def __setattr__(self, name, value):
        if name in self._fixed_configuration and name in self.__dict__:
            raise AttributeError(f"{name} is fixed; construct a new regime to change its model")
        super().__setattr__(name, value)

    def __init__(self, lo: float, hi: float, *, backend=None,
                 reliability: float = 0.75, p_flip: float = 0.3, p_two: float = 0.05,
                 potential: str = "entropy", n_grid=None,
                 claim_model: str = "pointwise", claims=(), probes=(),
                 probe_ids=None, query_weights=None, method: str = "transfer"):
        majority_unsupported(claim_model)
        if claim_model != "pointwise":
            raise ValueError(f"unsupported claim_model {claim_model!r}")
        if n_grid is not None:
            raise ValueError("ContinuumRegime has no grid; n_grid must be None")
        if query_weights is not None:
            raise ValueError("per-cell query_weights have no meaning without cells; pass query "
                             "points to task_risk instead")
        self.lo = _as_finite_float(lo, "lo")
        self.hi = _as_finite_float(hi, "hi")
        if not self.lo < self.hi:
            raise ValueError(f"require lo < hi, got lo={self.lo!r} hi={self.hi!r}")
        self.reliability = _check_reliability(reliability, "reliability")
        pf = _as_finite_float(p_flip, "p_flip")
        pt = _as_finite_float(p_two, "p_two")
        if not 0.0 <= pf <= 1.0:
            raise ValueError(f"p_flip must be in [0,1]: {pf}")
        if not 0.0 <= pt <= 1.0:
            raise ValueError(f"p_two must be in [0,1]: {pt}")
        self.p_flip = pf
        self.p_two = pt
        if potential not in ("entropy", "error"):
            raise ValueError(f"potential must be 'entropy' or 'error': {potential!r}")
        self.potential = potential
        self.n_grid = None
        self.claim_model = claim_model
        self._claims = []           # (a, b, sign, n, r)
        self._probes = []           # (x, sign, r, w, id)
        self._evidence_ids = set()
        self._claim_const = 0.0
        self._probe_const = 0.0     # sum w*(0.5*log(4r(1-r)) + log 1/2)
        self._probe_weight = 0.0
        self._tables = None
        for c in claims:
            c = tuple(c)
            if len(c) != 5:
                raise ValueError("a claim is (a, b, sign, n_eff, reliability)")
            self.add_claim(*c)
        probe_list = [tuple(p) for p in probes]
        if probe_ids is None:
            probe_ids = [f"legacy:{i}" for i in range(len(probe_list))]
        if len(probe_ids) != len(probe_list):
            raise ValueError("probe_ids must have one identity per imported probe")
        for p, pid in zip(probe_list, probe_ids):
            self.add_probe(p[0], p[1], p[2], p[3] if len(p) > 3 else 1.0, evidence_id=pid)

    # ------------------------------------------------------------------ inputs
    def add_claim(self, a, b, sign, n_eff=1.0, reliability=None) -> None:
        """Add a POINTWISE box claim with any bounds inside [lo, hi]; no partition to freeze."""
        a = _as_finite_float(a, "claim a")
        b = _as_finite_float(b, "claim b")
        if not (self.lo <= a < b <= self.hi):
            raise ValueError(
                f"claim box ({a}, {b}) must satisfy lo <= a < b <= hi = [{self.lo}, {self.hi}]")
        sign = _check_sign(sign, "claim sign")
        n = _as_finite_float(n_eff, "claim n_eff")
        if n < 0:
            raise ValueError(f"claim n_eff must be >= 0: {n}")
        r = self.reliability if reliability is None else _check_reliability(reliability, "claim reliability")
        self._claims.append((a, b, sign, n, r))
        self._claim_const += 0.5 * n * math.log(r * (1.0 - r))
        self._tables = None

    def set_claim_reliabilities(self, rs) -> None:
        rs = list(rs)
        if len(rs) != len(self._claims):
            raise ValueError("one reliability per claim is required")
        rs = [_check_reliability(r, "claim reliability") for r in rs]
        self._claims = [(a, b, sg, n, r) for (a, b, sg, n, _), r in zip(self._claims, rs)]
        self._claim_const = sum(0.5 * n * math.log(r * (1.0 - r)) for _, _, _, n, r in self._claims)
        self._tables = None

    def add_probe(self, x, sign, reliability=0.95, weight=1.0, *, evidence_id) -> None:
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("add_probe requires a nonempty evidence_id")
        if evidence_id in self._evidence_ids:
            raise ValueError(f"evidence id already consumed: {evidence_id!r}")
        sign = _check_sign(sign, "probe sign")
        r = _check_reliability(reliability, "probe reliability")
        wt = _as_finite_float(weight, "probe weight")
        if wt <= 0:
            raise ValueError(f"probe weight must be > 0: {wt}")
        x = _as_finite_float(x, "probe x")
        if not self.lo <= x <= self.hi:
            raise ValueError(f"probe x={x!r} is outside the declared domain [{self.lo}, {self.hi}]")
        self._probes.append((x, sign, r, wt, evidence_id))
        self._evidence_ids.add(evidence_id)
        self._probe_weight += wt
        self._probe_const += wt * (0.5 * math.log(4.0 * r * (1.0 - r)) + _LN_HALF)
        self._tables = None

    # ------------------------------------------------------------------ core
    def _effective_mass(self):
        two = self.p_two if self.p_two > 0.0 else 0.0
        p1 = self.p_flip * (1.0 - two)
        p0 = (1.0 - self.p_flip) * (1.0 - two)
        m0 = max(p0, _MIN_PRIOR)
        m1 = max(p1, _MIN_PRIOR)
        m2 = max(two, _MIN_TWO_PRIOR) if two > 0.0 else 0.0
        total = m0 + m1 + m2
        return np.array([m0 / total, m1 / total, m2 / total]), total, two

    def _arrays(self, extra_probe=None):
        cl = np.array([(a, b, n * sg * (math.log(r) - math.log1p(-r)) / (2.0 * (b - a)))
                       for a, b, sg, n, r in self._claims], dtype=float).reshape(-1, 3)
        pr = [(x, w * sg * (math.log(r) - math.log1p(-r)) / 2.0) for x, sg, r, w, _ in self._probes]
        if extra_probe is not None:
            pr.append(extra_probe)
        return cl, np.array(pr, dtype=float).reshape(-1, 2)

    def _log_mass(self):
        mass = self._effective_mass()[0]
        with np.errstate(divide="ignore"):
            return np.log(mass)

    def _build(self, extra_probe=None):
        cl, pr = self._arrays(extra_probe)
        t = continuum_tables(self.lo, self.hi, self._log_mass(), cl, pr)
        if not (math.isfinite(t["lZ"]) and np.all(np.isfinite(t["lW"][np.isfinite(t["lW"])]))):
            raise CompiledRegimeError("non-finite continuum partition function")
        return t

    def _t(self):
        if self._tables is None:
            self._tables = self._build()
        return self._tables

    # ------------------------------------------------------------------ reads
    @property
    def evidence_ids(self):
        return frozenset(self._evidence_ids)

    def evidence_ledger(self) -> dict:
        return {"schema": "compiled-regime-evidence-v1",
                "claims": tuple(self._claims), "probes": tuple(self._probes)}

    @property
    def n_runs(self):
        """Evidence runs (the size of the sufficient statistic), not a grid."""
        return self._t()["R"]

    def transition_prior(self):
        mass, total, two = self._effective_mass()
        return {"mass": mass, "normalizer": total, "two_family_enabled": two > 0.0}

    def p_plus(self, x) -> float:
        x = _as_finite_float(x, "query x")
        if not self.lo <= x <= self.hi:
            raise ValueError(f"query x={x!r} is outside the declared domain [{self.lo}, {self.hi}]")
        return float(continuum_query(self._t(), np.array([x]))[0])

    def p_plus_at(self, xs):
        xs = np.asarray(xs, dtype=float)
        if xs.ndim != 1 or not np.all(np.isfinite(xs)) or np.any(xs < self.lo) or np.any(xs > self.hi):
            raise ValueError("query points must be a finite 1-D array inside [lo, hi]")
        return np.clip(continuum_query(self._t(), xs), 0.0, 1.0)

    def family_mass(self):
        t = self._t()
        return np.exp(t["lW"] - t["lZ"])

    def transitions_posterior(self):
        return self.family_mass()

    def log_evidence(self, *, normalized_prior: bool = False) -> float:
        _, total, _ = self._effective_mass()
        return float(self._t()["lZ"] + (0.0 if normalized_prior else math.log(total))
                     + self._probe_const + self._claim_const)

    def collision(self, flag_at: float = 0.5) -> dict:
        pk = self.family_mass()
        m2 = float(pk[2])
        prior = float(self._effective_mass()[0][2])
        bf = (m2 / max(1.0 - m2, 1e-300)) / (prior / (1.0 - prior)) if 0.0 < prior < 1.0 else float("nan")
        return {"p_two_transitions": m2, "bayes_factor_two_vs_one": bf,
                "flag": m2 >= flag_at, "pk": pk.tolist()}

    def natural_statistics(self) -> dict:
        t = self._t()
        return {"edges": t["e"].copy(), "G_after_edge": t["g"].copy(), "slope": t["dens"].copy(),
                "log_partition": t["lZ"], "n_runs": t["R"], "n_probes": len(self._probes),
                "n_claims": len(self._claims)}

    # ------------------------------------------------------------------ potentials (quadrature)
    def _u(self, pp):
        pp = np.asarray(pp, dtype=float)
        if self.potential == "error":
            return np.minimum(pp, 1.0 - pp)
        q = np.clip(pp, 1e-12, 1.0 - 1e-12)
        return -(q * np.log2(q) + (1.0 - q) * np.log2(1.0 - q))

    def _potential_of(self, t):
        xs, ws = _nodes(t)
        return float(self._u(np.clip(continuum_query(t, xs), 0.0, 1.0)) @ ws)

    def potential_value(self, normalized: bool = False) -> float:
        """Integral of the potential over [lo, hi] (Gauss-Legendre per run piece)."""
        val = self._potential_of(self._t())
        return val / (self.hi - self.lo) if normalized else val

    def expected_error(self, normalized: bool = False) -> float:
        xs, ws = _nodes(self._t())
        pp = continuum_query(self._t(), xs)
        val = float(np.minimum(pp, 1.0 - pp) @ ws)
        return val / (self.hi - self.lo) if normalized else val

    def task_risk(self, query_points, query_weights=None, loss: str = "log") -> float:
        """Normalized risk at declared query points (weights default uniform)."""
        pp = self.p_plus_at(query_points)
        w = np.ones(len(pp)) if query_weights is None else np.asarray(query_weights, float)
        if w.shape != pp.shape or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("query_weights must be finite, nonnegative, one per point, not all zero")
        w = w / w.sum()
        if loss == "log":
            q = np.clip(pp, 0.0, 1.0)
            r = -(q * np.log2(np.maximum(q, 1e-300)) + (1.0 - q) * np.log2(np.maximum(1.0 - q, 1e-300)))
        elif loss == "brier":
            r = 1.0 - (pp ** 2 + (1.0 - pp) ** 2)
        elif loss == "error":
            r = np.minimum(pp, 1.0 - pp)
        else:
            raise ValueError("loss must be 'log', 'brier' or 'error'")
        return float(w @ r)

    # ------------------------------------------------------------------ OED
    def _probe_jump(self, y, r):
        return y * (math.log(r) - math.log1p(-r)) / 2.0

    def _outcomes(self, x, r):
        f = self.p_plus(x)
        p_plus_outcome = (1.0 - r) + (2.0 * r - 1.0) * f
        return ((1, p_plus_outcome), (-1, 1.0 - p_plus_outcome))

    def expected_gain(self, x, reliability: float = 0.95, weight: float = 1.0) -> dict:
        """Expected potential drop for one unit-weight probe at any real x (exact update)."""
        r = _check_reliability(reliability, "probe reliability")
        wt = _as_finite_float(weight, "probe weight")
        if wt != 1.0:
            raise ValueError("expected_gain supports a single unit-weight probe only")
        x = _as_finite_float(x, "probe x")
        if not self.lo <= x <= self.hi:
            raise ValueError("probe x outside the domain")
        now = self._potential_of(self._t())
        after, outcomes = 0.0, {}
        for y, pout in self._outcomes(x, r):
            if pout <= 0.0:
                continue
            u = self._potential_of(self._build((x, self._probe_jump(y, r))))
            after += pout * u
            outcomes[y] = {"probability": pout, "potential": u}
        return {"x": x, "gain": now - after, "before": now, "outcomes": outcomes,
                "reliability": r, "weight": wt}

    def location_posterior(self) -> dict:
        """Posterior of T = first sign change (k=0: T=lo if S=+ everywhere else hi)."""
        t = self._t()
        mean, var, check = location_moments(t, 0.5 * (self.lo + self.hi))
        return {"mean": mean, "var": var, "quadrature_mass_error": check}

    def expected_location_loss(self, x, reliability: float = 0.95) -> dict:
        """Bayes risk E[Var(T | data, y)] after one probe at x (squared loss on T)."""
        r = _check_reliability(reliability, "probe reliability")
        x = _as_finite_float(x, "probe x")
        if not self.lo <= x <= self.hi:
            raise ValueError("probe x outside the domain")
        c = 0.5 * (self.lo + self.hi)
        now = location_moments(self._t(), c)[1]
        after = 0.0
        for y, pout in self._outcomes(x, r):
            if pout <= 0.0:
                continue
            after += pout * location_moments(self._build((x, self._probe_jump(y, r))), c)[1]
        return {"x": x, "risk_after": after, "risk_now": now, "gain": now - after}

    def _continuous_argmax(self, score, n_candidates, refine):
        xs = np.linspace(self.lo, self.hi, n_candidates)
        edges = self._t()["e"]
        xs = np.unique(np.concatenate([xs, edges]))
        vals = np.array([score(float(x)) for x in xs])
        j = int(np.argmax(vals))
        best_x, best_v = float(xs[j]), float(vals[j])
        if refine:
            a = float(xs[max(j - 1, 0)])
            b = float(xs[min(j + 1, len(xs) - 1)])
            gr = (math.sqrt(5) - 1) / 2
            c1, c2 = b - gr * (b - a), a + gr * (b - a)
            f1, f2 = score(c1), score(c2)
            for _ in range(refine):
                if f1 >= f2:
                    b, c2, f2 = c2, c1, f1
                    c1 = b - gr * (b - a)
                    f1 = score(c1)
                else:
                    a, c1, f1 = c1, c2, f2
                    c2 = a + gr * (b - a)
                    f2 = score(c2)
            for xv, fv in ((c1, f1), (c2, f2)):
                if fv > best_v:
                    best_x, best_v = float(xv), float(fv)
        return best_x, best_v

    def best_probe(self, reliability: float = 0.95, n_candidates: int = 49, refine: int = 12) -> tuple:
        """(x, expected potential drop) over continuous x: candidates + golden-section refinement."""
        r = _check_reliability(reliability, "probe reliability")
        return self._continuous_argmax(lambda x: self.expected_gain(x, r)["gain"],
                                       n_candidates, refine)

    def best_location_probe(self, reliability: float = 0.95, n_candidates: int = 49,
                            refine: int = 12) -> tuple:
        """(x, expected drop of Var(T)) — task-specific OED for locating the sign change."""
        r = _check_reliability(reliability, "probe reliability")
        return self._continuous_argmax(lambda x: self.expected_location_loss(x, r)["gain"],
                                       n_candidates, refine)

    # ------------------------------------------------------------------ bridges
    @classmethod
    def from_ledger(cls, lo, hi, ledger, **config) -> "ContinuumRegime":
        """Rebuild from an ``evidence_ledger()`` snapshot of any regime (native or continuum)."""
        if ledger.get("schema") != "compiled-regime-evidence-v1":
            raise ValueError("unknown ledger schema")
        probes = [(x, s, r, w) for x, s, r, w, _ in ledger["probes"]]
        ids = [pid for *_, pid in ledger["probes"]]
        return cls(lo, hi, claims=ledger["claims"], probes=probes, probe_ids=ids, **config)

    @classmethod
    def from_compiled(cls, compiled) -> "ContinuumRegime":
        """Continuum view of a native CompiledRegime: same evidence and prior, no partition."""
        return cls.from_ledger(compiled.lo, compiled.hi, compiled.evidence_ledger(),
                               reliability=compiled.reliability, p_flip=compiled.p_flip,
                               p_two=compiled.p_two, potential=compiled.potential)
