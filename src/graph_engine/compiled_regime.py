"""compiled_regime.py — Graph-to-Kernel native inference bridge for the finite regime family.

This integration re-expresses the same finite
RegimePosterior family as a natural-statistic exponential family that is decoded by an
INJECTED native kernel (``kernel_engine.inference.finite_regime.FiniteRegimeKernel`` or any
object exposing the same ``evaluate(a, b, family_mass, *, method=...)`` contract).

WHAT IS EXACTLY PRESERVED (parity target: the live ``RegimePosterior``)
----------------------------------------------------------------------
* partition: claim edges + ``n_grid`` uniform cells (built once, then frozen);
* hypotheses: all-+/all-, one transition per cell (start sign +/-), and, when the family is
  enabled (``p_two > 0`` and ``Q >= 3``), two transitions in distinct cells i < j;
* per-hypothesis priors ``p0/2``, ``p1/(2Q)``, ``p2/(2*C(Q,2))`` and the SAME epsilon floors
  (``p0,p1 >= 1e-12``, two-transition mass ``>= 1e-300``);
* pointwise box claims: ``n*(f*ln r + (1-f)*ln(1-r))``, linear per cell in ``F``;
* probes: point observation at a cell, likelihood ``f*r + (1-f)*(1-r)`` with reliability
  ``r in [0.5, 1)`` and weight; conditional independence is DECLARED, not inferred;
* default potential is binary entropy in bits; ``potential="error"`` is ``min(p,1-p)``.

DECLARED DIFFERENCES from ``RegimePosterior`` (strict validation replaces silent behaviour)
------------------------------------------------------------------------------------------
1. ``claim_model="majority"`` is nonlocal and raises ``NotImplementedError``. It is NOT
   replaced by a different (Poisson/Markov) model; ``RegimeMarkov`` is a different family and
   is never recommended as an equivalent.
2. A claim with ``a >= b`` (zero-width or reversed) is rejected; a claim outside ``[lo, hi]``
   is rejected instead of being silently clamped. Define the domain to contain the claim.
3. A probe outside ``[lo, hi]`` is rejected instead of being folded into an end cell.
4. ``sign`` must be exactly ``-1`` or ``+1``; ``reliability`` finite with ``0.5 <= r < 1``
   (``r = 1`` would be a hard constraint and is refused); ``weight`` finite and ``> 0``;
   ``n_eff`` finite and ``>= 0``.
5. The partition is FROZEN (native mode). ``n_grid=None`` instead returns a
   ``ContinuumRegime`` (the n_grid -> infinity limit) that accepts any claim bounds.
   Otherwise: ``add_claim`` is accepted only when its bounds are already
   partition edges; a new edge raises :class:`FrozenPartitionError` (freeze-and-refuse),
   rather than silently rebuilding the hypothesis space.
6. Evidence identity is caller-declared and unique per ``add_probe``; a repeated id raises.
   Two copied answers are not deduplicated automatically — only a declared duplicate is.

The numeric state is ``A_q, B_q`` (2Q floats) on the fixed partition; provenance/raw history
grows with the number of observations and is kept only for audit, not for the posterior.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "CompiledRegime", "CompiledRegimeError", "FrozenPartitionError",
    "majority_unsupported", "DEFAULT_LOGDOMAIN",
]

_LN_HALF = math.log(0.5)
_MIN_PRIOR = 1e-12
_MIN_TWO_PRIOR = 1e-300
DEFAULT_LOGDOMAIN = (0.0, 1.0)


class CompiledRegimeError(Exception):
    """Base class for the compiled-regime bridge."""


class FrozenPartitionError(CompiledRegimeError):
    """A claim would introduce a new partition edge; the contract is freeze-and-refuse."""


def majority_unsupported(claim_model: str) -> None:
    if claim_model == "majority":
        raise NotImplementedError(
            "claim_model='majority' is NONLOCAL: P(claim|path) depends on the whole path "
            "inside the box and does not factor over cells. The compiled kernel refuses "
            "rather than silently factorizing. RegimeMarkov is a DIFFERENT family "
            "(telegraph/Poisson switch) and is not a substitute. Use claim_model='pointwise'.")


def _as_finite_float(value, what: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be a finite number: {value!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{what} must be finite: {value!r}")
    return v


def _check_sign(sign, what: str = "sign") -> int:
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise ValueError(f"{what} must be exactly -1 or +1: {sign!r}")
    return int(sign)


def _check_reliability(r, what: str) -> float:
    r = _as_finite_float(r, what)
    if not 0.5 <= r < 1.0:
        raise ValueError(
            f"{what} must be in [0.5, 1); r=1 is a hard constraint, unsupported here: {r}")
    return r


class CompiledRegime:
    """Frozen finite-regime posterior decoded by an injected native kernel.

    ``backend`` must expose ``evaluate(a, b, family_mass, *, method=...)`` returning a dict
    with keys ``mean_s``, ``mean_s2``, ``p_plus``, ``family_mass`` and ``log_partition``
    (arrays length ``Q`` and length 3). It is injected by the caller so this module imports
    no cross-repo code and never compiles anything at import time.
    """

    _fixed_configuration = frozenset({"lo", "hi", "reliability", "p_flip", "p_two",
                                      "potential", "n_grid", "claim_model"})

    def __new__(cls, *args, **kwargs):
        # ``n_grid=None`` selects the resolution-free backend (optional; see
        # ``continuum_regime.py``). It has the same read API and no frozen partition.
        if cls is CompiledRegime and "n_grid" in kwargs and kwargs["n_grid"] is None:
            from .continuum_regime import ContinuumRegime
            return ContinuumRegime(*args, **kwargs)
        return super().__new__(cls)

    def __setattr__(self, name, value):
        if name in self._fixed_configuration and name in self.__dict__:
            raise AttributeError(f"{name} is fixed; construct a new regime to change its model")
        super().__setattr__(name, value)

    def __init__(self, lo: float, hi: float, *, backend,
                 reliability: float = 0.75, p_flip: float = 0.3, p_two: float = 0.05,
                 potential: str = "entropy", n_grid: int = 48,
                 claim_model: str = "pointwise", claims=(), probes=(),
                 probe_ids=None, query_weights=None, method: str = "transfer",
                 covariance_backend=None):
        majority_unsupported(claim_model)
        if claim_model != "pointwise":
            raise ValueError(f"unsupported claim_model {claim_model!r}")
        if backend is None or not callable(getattr(backend, "evaluate", None)):
            raise TypeError("backend must expose evaluate(a, b, family_mass, *, method=...)")
        self._backend = backend
        self._method = method
        if covariance_backend is not None and not callable(getattr(covariance_backend, "cov_rows", None)):
            raise TypeError("covariance_backend must expose cov_rows(a, b, rows, family_mass)")
        self._covariance_backend = covariance_backend
        self.last_probe_path = None
        self.lo = _as_finite_float(lo, "lo")
        self.hi = _as_finite_float(hi, "hi")
        if not self.lo < self.hi:
            raise ValueError(f"require lo < hi, got lo={self.lo!r} hi={self.hi!r}")
        self.reliability = _check_reliability(reliability, "reliability")
        # p_flip / p_two are prior masses in [0,1], not read-out reliabilities.
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
        if isinstance(n_grid, bool) or not isinstance(n_grid, (int, np.integer)) or n_grid < 1:
            raise ValueError(f"n_grid must be an integer >= 1: {n_grid!r}")
        self.n_grid = int(n_grid)
        self.claim_model = claim_model

        claim_list = [tuple(c) for c in claims]
        bounds = []
        for c in claim_list:
            if len(c) != 5:
                raise ValueError("a claim is (a, b, sign, n_eff, reliability)")
            a, b = float(c[0]), float(c[1])
            bounds.append((a, b))
        self._build_geometry(bounds)

        self._coeff = np.zeros(self._Q)
        self._claim_const = 0.0
        self._A = np.zeros(self._Q)
        self._B = np.zeros(self._Q)
        self._probe_weight = 0.0
        self._claims = []
        self._probes = []
        self._evidence_ids = set()
        self._cache = None
        # Derived, producer-owned approximation state. The full A/B state,
        # evidence history, prior and exact native API remain authoritative.
        self._owner_revision = 0
        self._flat_anchor = None
        self._approx_cache = None
        self._query_weights = self._validate_query_weights(query_weights)

        for c in claim_list:
            self._append_claim(c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0,
                               c[4] if len(c) > 4 else None)
        probe_list = [tuple(p) for p in probes]
        if probe_ids is None:
            probe_ids = [f"legacy:{i}" for i in range(len(probe_list))]
        if len(probe_ids) != len(probe_list):
            raise ValueError("probe_ids must have one identity per imported probe")
        for p, pid in zip(probe_list, probe_ids):
            self.add_probe(p[0], p[1], p[2], p[3] if len(p) > 3 else 1.0, evidence_id=pid)

    # ------------------------------------------------------------------ construction helpers
    def _build_geometry(self, claim_bounds):
        pts = {self.lo, self.hi}
        for a, b in claim_bounds:
            if not (math.isfinite(a) and math.isfinite(b)):
                raise ValueError(f"claim bounds must be finite: {(a, b)!r}")
            if not (self.lo <= a < b <= self.hi):
                raise ValueError(
                    f"claim box ({a}, {b}) must satisfy lo <= a < b <= hi = "
                    f"[{self.lo}, {self.hi}]; reversed/zero-width/outside is refused")
            pts.add(a)
            pts.add(b)
        pts.update(np.linspace(self.lo, self.hi, self.n_grid + 1).tolist())
        edges = np.array(sorted(pts), dtype=float)
        cells = np.stack([edges[:-1], edges[1:]], axis=1)
        cells = cells[cells[:, 1] > cells[:, 0]]
        widths = cells[:, 1] - cells[:, 0]
        cells.setflags(write=False)
        widths.setflags(write=False)
        edges.setflags(write=False)
        self._edges = edges
        self._cells = cells
        self._widths = widths
        self._Q = len(cells)

    def _bounds_are_edges(self, a, b) -> bool:
        # A near edge is still a different partition. Splitting a tiny cell
        # changes the uniform-over-transition-cells prior by a finite amount.
        return bool(np.any(self._edges == a) and np.any(self._edges == b))

    def _cell_index(self, x) -> int:
        x = _as_finite_float(x, "probe x")
        if not self.lo <= x <= self.hi:
            raise ValueError(f"probe x={x!r} is outside the declared domain [{self.lo}, {self.hi}]")
        return int(min(np.searchsorted(self._cells[:, 1], x, side="left"), self._Q - 1))

    def _claim_overlap(self, a, b):
        ov = np.clip(np.minimum(self._cells[:, 1], b) - np.maximum(self._cells[:, 0], a),
                     0.0, None)
        total = ov.sum()
        if total <= 0:
            raise CompiledRegimeError("claim box does not intersect the partition")
        return ov / total

    # ------------------------------------------------------------------ inputs
    def add_claim(self, a, b, sign, n_eff=1.0, reliability=None) -> None:
        """Add a POINTWISE box claim. Bounds must already be partition edges (frozen)."""
        a = _as_finite_float(a, "claim a")
        b = _as_finite_float(b, "claim b")
        if not (self.lo <= a < b <= self.hi):
            raise ValueError(
                f"claim box ({a}, {b}) must satisfy lo <= a < b <= hi = [{self.lo}, {self.hi}]")
        if not self._bounds_are_edges(a, b):
            raise FrozenPartitionError(
                f"claim box ({a}, {b}) would add new partition edges; the partition is frozen. "
                "Rebuild a CompiledRegime with the new claim in `claims`, or snap the bounds "
                "to existing edges. (Freeze-and-refuse contract: no silent repartition.)")
        self._append_claim(a, b, sign, n_eff, reliability)

    def _append_claim(self, a, b, sign, n_eff, reliability):
        sign = _check_sign(sign, "claim sign")
        n = _as_finite_float(n_eff, "claim n_eff")
        if n < 0:
            raise ValueError(f"claim n_eff must be >= 0: {n}")
        r = self.reliability if reliability is None else _check_reliability(reliability, "claim reliability")
        ov = self._claim_overlap(float(a), float(b))
        lratio = math.log(r) - math.log(1.0 - r)
        self._coeff = self._coeff + (n * sign * lratio) * ov
        self._claim_const += 0.5 * n * math.log(r * (1.0 - r))
        self._claims.append((float(a), float(b), sign, n, r))
        self._cache = None
        self._invalidate_flat_anchor()

    def set_claim_reliabilities(self, rs) -> None:
        """Re-estimate every claim reliability in order. Geometry is unchanged."""
        rs = list(rs)
        if len(rs) != len(self._claims):
            raise ValueError("one reliability per claim is required")
        coeff = np.zeros(self._Q)
        const = 0.0
        new_claims = []
        for (a, b, sg, n, _), r in zip(self._claims, rs):
            r = _check_reliability(r, "claim reliability")
            ov = self._claim_overlap(a, b)
            coeff = coeff + (n * sg * (math.log(r) - math.log(1.0 - r))) * ov
            const += 0.5 * n * math.log(r * (1.0 - r))
            new_claims.append((a, b, sg, n, r))
        self._coeff = coeff
        self._claim_const = const
        self._claims = new_claims
        self._cache = None
        self._invalidate_flat_anchor()

    def _invalidate_flat_anchor(self):
        """Claims change the likelihood semantics at every covered cell."""
        self._owner_revision += 1
        self._flat_anchor = None
        self._approx_cache = None

    def add_probe(self, x, sign, reliability=0.95, weight=1.0, *, evidence_id) -> None:
        """Consume conditionally independent evidence; O(log Q) cell lookup.

        Updating the two numeric entries is O(1). A nonunit weight is a
        tempered likelihood, not necessarily a normalized observation channel.

        ``evidence_id`` must be a nonempty string, unique among consumed observations.
        Copying an answer without a new id is refused; copying it WITH a new id cannot be
        detected here and is the caller's declared responsibility.
        """
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("add_probe requires a nonempty evidence_id")
        if evidence_id in self._evidence_ids:
            raise ValueError(f"evidence id already consumed: {evidence_id!r}")
        sign = _check_sign(sign, "probe sign")
        r = _check_reliability(reliability, "probe reliability")
        wt = _as_finite_float(weight, "probe weight")
        if wt <= 0:
            raise ValueError(f"probe weight must be > 0: {wt}")
        c = self._cell_index(x)
        lratio = math.log(r) - math.log(1.0 - r)
        self._A[c] += wt * 0.5 * sign * lratio
        self._B[c] += wt * 0.5 * math.log(4.0 * r * (1.0 - r))
        self._probe_weight += wt
        self._probes.append((float(x), sign, r, wt, evidence_id))
        self._evidence_ids.add(evidence_id)
        self._cache = None
        self._owner_revision += 1
        self._approx_cache = None
        if self._flat_anchor is not None:
            da = wt * 0.5 * sign * lratio
            db = wt * 0.5 * math.log(4.0 * r * (1.0 - r))
            self._flat_anchor["h"] += da
            self._flat_anchor["log_likelihood_contrast"] += wt * abs(lratio)
            anchor = self._flat_anchor
            if anchor["delta_a"] is not None:
                old = max(0.0, abs(anchor["delta_a"][c]) + anchor["delta_b"][c])
                anchor["delta_a"][c] += da
                anchor["delta_b"][c] += db
                new = max(0.0, abs(anchor["delta_a"][c]) + anchor["delta_b"][c])
                anchor["sum_da"] += da
                anchor["sum_db"] += db
                anchor["log_upper"] += new - old
                anchor["numeric_valid"] = (anchor["numeric_valid"] and
                    all(math.isfinite(float(v)) for v in (
                        anchor["h"], anchor["log_likelihood_contrast"],
                        anchor["delta_a"][c], anchor["delta_b"][c],
                        anchor["sum_da"], anchor["sum_db"], anchor["log_upper"])))

    # ------------------------------------------------------------------ native decode
    def _effective_mass(self):
        two = self.p_two if (self.p_two > 0.0 and self._Q >= 3) else 0.0
        p1 = self.p_flip * (1.0 - two)
        p0 = (1.0 - self.p_flip) * (1.0 - two)
        m0 = max(p0, _MIN_PRIOR)
        m1 = max(p1, _MIN_PRIOR)
        m2 = max(two, _MIN_TWO_PRIOR) if two > 0.0 else 0.0
        total = m0 + m1 + m2
        return np.array([m0 / total, m1 / total, m2 / total]), total, two

    def _run(self, extra=None):
        a = self._A + 0.5 * self._coeff
        b = self._B.copy()
        if extra is not None:
            c, y, r, wt = extra
            a = a.copy()
            a[c] += wt * 0.5 * y * math.log(r / (1.0 - r))
            b[c] += wt * 0.5 * math.log(4.0 * r * (1.0 - r))
        mass, _, _ = self._effective_mass()
        res = self._backend.evaluate(a, b, mass, method=self._method)
        pp = np.asarray(res["p_plus"], dtype=float)
        ms = np.asarray(res["mean_s"], dtype=float)
        m2 = np.asarray(res["mean_s2"], dtype=float)
        fam = np.asarray(res["family_mass"], dtype=float)
        if (pp.shape != (self._Q,) or ms.shape != (self._Q,) or m2.shape != (self._Q,)
                or fam.shape != (3,)):
            raise CompiledRegimeError(
                f"backend returned arrays of the wrong shape for Q={self._Q}")
        if not (np.all(np.isfinite(pp)) and np.all(np.isfinite(ms))
                and np.all(np.isfinite(m2)) and np.all(np.isfinite(fam))
                and math.isfinite(float(res["log_partition"]))):
            raise CompiledRegimeError("backend returned non-finite posterior fields")
        tolerance = 1e-8
        if (pp.min() < -tolerance or pp.max() > 1+tolerance
                or m2.min() < -tolerance or m2.max() > 1+tolerance
                or fam.min() < -tolerance or abs(fam.sum()-1) > tolerance
                or np.max(np.abs(pp-(1+ms)*0.5)) > tolerance
                or np.max(np.abs(ms)-m2) > tolerance):
            raise CompiledRegimeError("backend returned inconsistent probability moments")
        return {"p_plus": pp, "mean_s": ms, "mean_s2": m2, "family_mass": fam,
                "log_partition": float(res["log_partition"])}

    def _posterior(self):
        if self._cache is None:
            self._cache = self._run(None)
        return self._cache

    # ------------------------------------------------------------------ reads
    @property
    def cells(self):
        return self._cells.copy()

    @property
    def widths(self):
        return self._widths.copy()

    @property
    def evidence_ids(self):
        return frozenset(self._evidence_ids)

    def evidence_ledger(self) -> dict:
        """Snapshot the actual likelihood inputs, with resolved reliabilities.

        Claims are (lo, hi, sign, n_eff, reliability); probes are
        (x, sign, reliability, weight, evidence_id). Order is preserved within
        each collection. Claim recalibration is reflected in a new snapshot.
        A constructor argument of ``None`` is never exported as a reliability.

        This is the current evidence ledger, not a chronological mutation log.
        Claims have no observation IDs in this API. Prior, partition and task
        semantics must be bound separately when sharing a predictive state.
        Returned tuples do not expose the owner's mutable lists.
        """
        return {"schema": "compiled-regime-evidence-v1",
                "claims": tuple(self._claims), "probes": tuple(self._probes)}

    @property
    def n_cells(self):
        return self._Q

    def transition_prior(self):
        mass, total, two = self._effective_mass()
        return {"mass": mass, "normalizer": total, "two_family_enabled": two > 0.0}

    def p_plus_cells(self):
        return self._posterior()["p_plus"].copy()

    @staticmethod
    def _flat_sigmoid(log_odds):
        if log_odds >= 0.0:
            return 1.0 / (1.0 + math.exp(-log_odds))
        value = math.exp(log_odds)
        return value / (1.0 + value)

    def _flat_budget(self):
        anchor = self._flat_anchor
        if anchor is None or not anchor["tail_resolved"]:
            return None
        epsilon = anchor["tail_mass_at_anchor"]
        contrast = anchor["log_likelihood_contrast"]
        if not (math.isfinite(epsilon) and 0.0 < epsilon < 1.0
                and math.isfinite(contrast) and contrast >= 0.0
                and math.isfinite(anchor["h"])):
            return None
        log_odds = math.log(epsilon) - math.log1p(-epsilon) + contrast
        return self._flat_sigmoid(log_odds) if math.isfinite(log_odds) else None

    @staticmethod
    def _logaddexp(a, b):
        high = max(a, b)
        return high + math.log1p(math.exp(-abs(a - b)))

    def _conditional_flat_budget(self):
        """Ordinary-float tail envelope using conditional flat evidence."""
        anchor = self._flat_anchor
        if (anchor is None or not anchor["tail_resolved"] or not anchor["numeric_valid"]
                or anchor["delta_a"] is None):
            return None
        h = anchor["anchor_h"]
        two_h = 2.0 * h
        if not all(math.isfinite(float(x)) for x in (
                h, two_h, anchor["sum_da"], anchor["sum_db"], anchor["log_upper"])):
            return None
        # These are log P(initial + | flat, anchor) and its opposite.
        log_plus = -self._logaddexp(0.0, -two_h)
        log_minus = -self._logaddexp(0.0, two_h)
        plus = log_plus + anchor["sum_da"]
        minus = log_minus - anchor["sum_da"]
        if not all(math.isfinite(x) for x in (log_plus, log_minus, plus, minus)):
            return None
        log_z = anchor["sum_db"] + self._logaddexp(plus, minus)
        ratio = anchor["log_upper"] - log_z
        if not all(math.isfinite(x) for x in (log_z, ratio)):
            return None
        epsilon = anchor["tail_mass_at_anchor"]
        log_odds = math.log(epsilon) - math.log1p(-epsilon) + max(0.0, ratio)
        return self._flat_sigmoid(log_odds) if math.isfinite(log_odds) else None

    def _selected_flat_budget(self):
        generic = self._flat_budget()
        conditional = self._conditional_flat_budget()
        if generic is None or conditional is None:
            return None
        if conditional < generic:
            return {"budget": conditional, "choice": "conditional",
                    "generic": generic, "conditional": conditional}
        return {"budget": generic, "choice": "generic",
                "generic": generic, "conditional": conditional}

    def _native_approx_result(self):
        """Refresh from the current owner's full posterior, never a hypothetical."""
        posterior = self._posterior()
        prior = self._effective_mass()[0]
        tail = math.fsum(float(x) for x in posterior["family_mass"][1:])
        prior_tail = math.fsum(float(x) for x in prior[1:])
        resolved = (math.isfinite(tail) and 0.0 < tail < 1.0)
        # A computed zero cannot erase positive prior support. Nor can a
        # rounded one prove the covered family absent.
        h = float(np.sum(self._A + 0.5 * self._coeff))
        delta_a = np.zeros(self._Q) if resolved else None
        delta_b = np.zeros(self._Q) if resolved else None
        self._flat_anchor = {"revision": self._owner_revision, "h": h,
                             "anchor_h": h, "delta_a": delta_a, "delta_b": delta_b,
                             "sum_da": 0.0, "sum_db": 0.0, "log_upper": 0.0,
                             "numeric_valid": math.isfinite(h),
                             "tail_mass_at_anchor": tail,
                             "prior_tail_mass": prior_tail,
                             "tail_resolved": resolved,
                             "log_likelihood_contrast": 0.0}
        result = {"p_plus": posterior["p_plus"].copy(), "mode": "native",
                  "error_bound_estimate": 0.0, "numerical_certificate": False,
                  "owner_revision": self._owner_revision,
                  "anchor_revision": self._owner_revision,
                  "tail_mass_at_anchor": tail,
                  "prior_tail_mass": prior_tail,
                  "tail_resolved": resolved,
                  "log_likelihood_contrast": 0.0,
                  "generic_bound_estimate": 0.0,
                  "conditional_bound_estimate": 0.0,
                  "bound_choice": "native",
                  "approx_anchor_numeric_bytes":
                      0 if delta_a is None else delta_a.nbytes + delta_b.nbytes}
        self._approx_cache = result
        return result

    def approximate_p_plus_cells(self, max_error):
        """Optional bounded-tail estimate with full native fallback.

        A positive max_error permits the flat-family conditional readout only
        after a prior full native anchor. The mathematical tail envelope is
        evaluated in ordinary floats, so it is NOT an outward numerical
        certificate. The original state and exact p_plus methods are intact.
        A nonunit probe weight is a power likelihood, not a newly normalized
        physical observation channel.
        """
        if isinstance(max_error, bool):
            raise ValueError("max_error must be finite with 0 <= max_error < 1")
        tolerance = _as_finite_float(max_error, "max_error")
        if not 0.0 <= tolerance < 1.0:
            raise ValueError("max_error must satisfy 0 <= max_error < 1")
        cached = self._approx_cache
        # An exact answer already available from another API is preferred and
        # becomes the latest anchor. Likewise tolerance zero always asks for it.
        if self._cache is not None or tolerance == 0.0 or self._flat_anchor is None:
            if (cached is None or cached["mode"] != "native"
                    or cached["owner_revision"] != self._owner_revision):
                cached = self._native_approx_result()
        elif cached is not None and cached["owner_revision"] == self._owner_revision:
            if cached["mode"] != "native" and cached["error_bound_estimate"] > tolerance:
                cached = self._native_approx_result()
        else:
            selected = self._selected_flat_budget()
            h = self._flat_anchor["h"]
            if (selected is None or selected["budget"] > tolerance
                    or not math.isfinite(2.0 * h)):
                cached = self._native_approx_result()
            else:
                value = self._flat_sigmoid(2.0 * h)
                if not math.isfinite(value):
                    cached = self._native_approx_result()
                else:
                    cached = {"p_plus": np.full(self._Q, value), "mode": "flat",
                              "error_bound_estimate": selected["budget"],
                              "numerical_certificate": False,
                              "owner_revision": self._owner_revision,
                              "anchor_revision": self._flat_anchor["revision"],
                              "tail_mass_at_anchor": self._flat_anchor["tail_mass_at_anchor"],
                              "prior_tail_mass": self._flat_anchor["prior_tail_mass"],
                              "tail_resolved": True,
                              "log_likelihood_contrast": self._flat_anchor["log_likelihood_contrast"],
                              "generic_bound_estimate": selected["generic"],
                              "conditional_bound_estimate": selected["conditional"],
                              "bound_choice": selected["choice"],
                              "approx_anchor_numeric_bytes":
                                  self._flat_anchor["delta_a"].nbytes +
                                  self._flat_anchor["delta_b"].nbytes}
                    self._approx_cache = cached
        result = dict(cached)
        result["p_plus"] = cached["p_plus"].copy()
        result["max_error_requested"] = tolerance
        return result

    def p_plus(self, x) -> float:
        return float(self._posterior()["p_plus"][self._cell_index(x)])

    def mean_s_cells(self):
        return self._posterior()["mean_s"].copy()

    def mean_s2_cells(self):
        return self._posterior()["mean_s2"].copy()

    def equal_f2_cells(self):
        """E[F_q^2] from E[S], E[S^2]: (1 + 2*E[S] + E[S^2]) / 4. NOT E[F] at F=1/2."""
        p = self._posterior()
        return (1.0 + 2.0 * p["mean_s"] + p["mean_s2"]) / 4.0

    def family_mass(self):
        return self._posterior()["family_mass"].copy()

    def transitions_posterior(self):
        return self.family_mass()

    def log_evidence(self, *, normalized_prior: bool = False) -> float:
        """Log evidence including observation constants.

        Default preserves RegimePosterior's unnormalized prior floors. Set
        normalized_prior=True for a normalized prior. With tempered probe or
        pointwise box likelihoods this is a generalized evidence score, not
        automatically the probability of a normalized observation channel.
        """
        p = self._posterior()
        _, total, _ = self._effective_mass()
        return float(p["log_partition"] + (0.0 if normalized_prior else math.log(total))
                     + self._probe_weight * _LN_HALF + self._claim_const)

    def _u(self, pp):
        pp = np.asarray(pp, dtype=float)
        if self.potential == "error":
            return np.minimum(pp, 1.0 - pp)
        q = np.clip(pp, 1e-12, 1.0 - 1e-12)
        return -(q * np.log2(q) + (1.0 - q) * np.log2(1.0 - q))

    def potential_value(self, normalized: bool = False) -> float:
        """Integral of the potential over the domain (bits for entropy). Domain-length units."""
        val = float(self._u(self._posterior()["p_plus"]) @ self._widths)
        return val / (self.hi - self.lo) if normalized else val

    def expected_error(self, normalized: bool = False) -> float:
        pp = self._posterior()["p_plus"]
        val = float(np.minimum(pp, 1.0 - pp) @ self._widths)
        return val / (self.hi - self.lo) if normalized else val

    def _validate_query_weights(self, query_weights):
        if query_weights is None:
            w = np.array(self._widths, dtype=float)
        else:
            w = np.array(query_weights, dtype=float, copy=True)
            if w.shape != (self._Q,):
                raise ValueError(f"query_weights must have one weight per cell ({self._Q})")
            if not np.all(np.isfinite(w)) or np.any(w < 0) or w.max() <= 0:
                raise ValueError("query_weights must be finite, nonnegative and not all zero")
        w = w / w.max()
        return w / w.sum()

    def task_risk(self, query_weights=None, loss: str = "log") -> float:
        """Normalized deployment risk on the SAME cell/query semantics as ``p_plus``.

        This is the NORMALIZED-risk view; ``potential_value``/``expected_error`` above are the
        domain-length INTEGRAL view. With width weights it equals the integral
        divided by domain length (subject to the legacy potential's endpoint clipping).
        """
        w = self._query_weights if query_weights is None else self._validate_query_weights(query_weights)
        w = w / w.sum()
        pp = self._posterior()["p_plus"]
        if loss == "log":
            q = np.clip(pp, 0.0, 1.0)
            r = -(q * np.log2(np.maximum(q, 1e-300))
                  + (1.0-q) * np.log2(np.maximum(1.0-q, 1e-300)))
        elif loss == "brier":
            r = 1.0 - (pp ** 2 + (1.0 - pp) ** 2)
        elif loss == "error":
            r = np.minimum(pp, 1.0 - pp)
        else:
            raise ValueError("loss must be 'log', 'brier' or 'error'")
        return float(w @ r)

    def collision(self, flag_at: float = 0.5) -> dict:
        pk = self.family_mass()
        m2 = float(pk[2])
        prior = float(self._effective_mass()[0][2])
        bf = (m2 / max(1.0 - m2, 1e-300)) / (prior / (1.0 - prior)) if 0.0 < prior < 1.0 else float("nan")
        return {"p_two_transitions": m2, "bayes_factor_two_vs_one": bf,
                "flag": m2 >= flag_at, "pk": pk.tolist()}

    def natural_statistics(self) -> dict:
        """The O(Q) sufficient state: probe A,B plus the folded claim linear coefficients."""
        return {"A": self._A.copy(), "B": self._B.copy(),
                "probe_weight": float(self._probe_weight),
                "claim_coeff": self._coeff.copy(), "claim_const": float(self._claim_const),
                "kernel_a": (self._A + 0.5 * self._coeff), "kernel_b": self._B.copy(),
                "n_cells": self._Q, "n_probes": len(self._probes), "n_claims": len(self._claims)}

    # ------------------------------------------------------------------ OED (exact hypothetical updates)
    def _hypothetical_pp(self, c, y, r, wt):
        return self._run((c, y, r, wt))["p_plus"]

    def expected_gain(self, x, reliability: float = 0.95, weight: float = 1.0) -> dict:
        """Expected potential drop for one unit-weight physical probe.

        A tempered posterior with an ordinary Bernoulli outcome probability
        is not a Bayes experiment. Nonunit weights require a specified joint
        observation channel and are therefore refused here.
        """
        r = _check_reliability(reliability, "probe reliability")
        wt = _as_finite_float(weight, "probe weight")
        if wt != 1.0:
            raise ValueError("expected_gain supports a single unit-weight probe only")
        c = self._cell_index(x)
        pp = self._posterior()["p_plus"]
        now = float(self._u(pp) @ self._widths)
        f = float(pp[c])
        p_plus_outcome = (1.0 - r) + (2.0 * r - 1.0) * f
        after = 0.0
        outcomes = {}
        for y, pout in ((1, p_plus_outcome), (-1, 1.0 - p_plus_outcome)):
            if pout <= 0.0:
                continue
            q = self._hypothetical_pp(c, y, r, wt)
            u = float(self._u(q) @ self._widths)
            after += pout * u
            outcomes[y] = {"probability": pout, "potential": u}
        return {"x": float(x), "cell": c, "gain": now - after, "before": now,
                "outcomes": outcomes, "reliability": r, "weight": wt}

    _PROBE_METHODS = ("direct", "batch", "covariance")
    # Screened gains within this fraction of the domain length (the entropy potential's
    # maximum, in bits) of the screened maximum are re-decided by the direct path.
    _PROBE_TIE_BAND = 1e-9
    _PROBE_MEAN_GUARD = 1e-9
    _COV_ROW_CHUNK = 256

    def _probe_candidates(self):
        xs = [float(x) for x in self._cells.mean(axis=1)] + [self.lo, self.hi]
        return xs, np.array([self._cell_index(x) for x in xs], dtype=np.int64)

    def _covariance_provider(self):
        if self._covariance_backend is not None:
            return self._covariance_backend
        if callable(getattr(self._backend, "cov_rows", None)):
            return self._backend
        return None

    def _outcome_probabilities(self, pp, r):
        """P(+) exactly as ``expected_gain`` forms it, per cell."""
        return (1.0 - r) + (2.0 * r - 1.0) * pp

    def _screen_batch(self, r):
        """Expected potential after a probe at every cell: 2Q states, one batch call."""
        pp = self._posterior()["p_plus"]
        q = self._Q
        a0 = self._A + 0.5 * self._coeff
        la = 0.5 * math.log(r / (1.0 - r))
        lb = 0.5 * math.log(4.0 * r * (1.0 - r))
        idx = np.arange(q)
        a = np.repeat(a0[None], 2 * q, axis=0)
        b = np.repeat(self._B[None], 2 * q, axis=0)
        a[idx, idx] += la
        a[q + idx, idx] -= la
        b[idx, idx] += lb
        b[q + idx, idx] += lb
        mass, _, _ = self._effective_mass()
        res = self._backend.evaluate_batch(a, b, mass)
        hp = np.asarray(res["p_plus"], dtype=float)
        if hp.shape != (2 * q, q) or not np.all(np.isfinite(hp)):
            raise CompiledRegimeError("batch backend returned malformed hypothetical marginals")
        u = self._u(hp) @ self._widths
        plus = self._outcome_probabilities(pp, r)
        return plus * u[:q] + (1.0 - plus) * u[q:]

    def _screen_covariance(self, provider, r):
        """Same quantity from one covariance: p'_j(y) = p_j + y(2r-1)Cov(g_j,g_c)/P(y)."""
        pp = self._posterior()["p_plus"]
        q = self._Q
        a0 = self._A + 0.5 * self._coeff
        mass, _, _ = self._effective_mass()
        beta = 2.0 * r - 1.0
        plus = self._outcome_probabilities(pp, r)
        minus = 1.0 - plus
        after = np.empty(q)
        w = self._widths
        for start in range(0, q, self._COV_ROW_CHUNK):
            rows = np.arange(start, min(q, start + self._COV_ROW_CHUNK), dtype=np.int64)
            out = provider.cov_rows(a0, self._B, rows, mass)
            ms = np.asarray(out["mean_s"], dtype=float)
            blk = np.asarray(out["rows"], dtype=float)
            if ms.shape != (q,) or blk.shape[0] != len(rows) or blk.shape[1] < q:
                raise CompiledRegimeError("covariance backend returned arrays of the wrong shape")
            if (not np.all(np.isfinite(ms)) or not np.all(np.isfinite(blk[:, :q]))
                    or np.max(np.abs(0.5 * (1.0 + ms) - pp)) > self._PROBE_MEAN_GUARD):
                raise CompiledRegimeError("covariance backend disagrees with the regime posterior")
            # Cov is symmetric: row c of the S block, read over j, is Cov(g_j, g_c) for probe c.
            cov = blk[:, :q] * 0.25                      # (probe c, cell j)
            pc, mc = plus[rows][:, None], minus[rows][:, None]
            up = np.clip(pp[None, :] + beta * cov / pc, 0.0, 1.0)
            dn = np.clip(pp[None, :] - beta * cov / mc, 0.0, 1.0)
            after[rows] = plus[rows] * (self._u(up) @ w) + minus[rows] * (self._u(dn) @ w)
        return after

    def probe_losses(self, reliability: float = 0.95, *, method: str = "direct") -> dict:
        """Expected potential after one unit-weight probe at every ``best_probe`` candidate.

        ``method`` selects how the 2Q hypothetical posteriors are produced: ``"direct"`` (one
        native evaluation each, as ``expected_gain``), ``"batch"`` (one ``evaluate_batch``) or
        ``"covariance"`` (one covariance block from a backend exposing ``cov_rows``). The
        screened methods agree with direct to rounding, not bitwise; ``best_probe`` resolves
        near-ties with the direct path. Falls back covariance -> batch -> direct when a
        capability is missing; the path used is in the result.
        """
        r = _check_reliability(reliability, "probe reliability")
        if method not in self._PROBE_METHODS:
            raise ValueError(f"method must be one of {self._PROBE_METHODS}: {method!r}")
        xs, cs = self._probe_candidates()
        pp = self._posterior()["p_plus"]
        before = float(self._u(pp) @ self._widths)
        after_cell, path, reason = None, "direct", None
        if method == "covariance":
            provider = self._covariance_provider()
            if provider is None:
                reason = "no cov_rows capability"
            else:
                try:
                    after_cell, path = self._screen_covariance(provider, r), "covariance"
                except (CompiledRegimeError, ArithmeticError, ValueError) as exc:
                    reason = f"covariance: {exc}"
                    method = "direct"    # a disagreeing kernel is not silently replaced by batch
        if after_cell is None and method in ("batch", "covariance"):
            if self._method != "transfer" or not callable(getattr(self._backend, "evaluate_batch", None)):
                reason = (reason + "; " if reason else "") + "no evaluate_batch for this method"
            else:
                try:
                    after_cell, path = self._screen_batch(r), "batch"
                except (CompiledRegimeError, ArithmeticError, ValueError) as exc:
                    reason = (reason + "; " if reason else "") + f"batch: {exc}"
        if after_cell is None:
            after = np.array([before - self.expected_gain(x, reliability=r)["gain"] for x in xs])
        else:
            after = after_cell[cs]
        return {"x": np.array(xs), "cell": cs, "after": after, "before": before,
                "gain": before - after, "path": path, "fallback_reason": reason,
                "reliability": r}

    def best_probe(self, reliability: float = 0.95, *, method: str = "direct") -> tuple:
        """(x, expected drop) maximizing the potential, one candidate per cell + both ends.

        Exact: each candidate is evaluated by a real hypothetical update through the native
        kernel — no pair-moment approximation and no ``E[F^2]=E[F]`` shortcut.

        ``method="direct"`` (default) is the original per-candidate loop. ``"batch"`` and
        ``"covariance"`` screen all candidates at once (see ``probe_losses``) and then apply
        the same first-maximum rule; when screened candidates within ``1e-9*(hi-lo)`` of the
        maximum lie in more than one cell, exactly those are re-decided by the direct path,
        so the returned value is then the direct value. Missing capabilities fall back to the
        direct loop. ``last_probe_path`` records what was used.
        """
        r = _check_reliability(reliability, "probe reliability")
        if method not in self._PROBE_METHODS:
            raise ValueError(f"method must be one of {self._PROBE_METHODS}: {method!r}")
        if method != "direct":
            scr = self.probe_losses(r, method=method)
            if scr["path"] != "direct":
                return self._resolve_screened(scr, r)
            fallback = scr["fallback_reason"]
        else:
            fallback = None
        self.last_probe_path = {"path": "direct", "fallback_reason": fallback, "resolved": 0}
        cells = self._cells
        xs = list(cells.mean(axis=1)) + [self.lo, self.hi]
        cs = list(range(self._Q)) + [0, self._Q - 1]
        best = (float(xs[0]), -1.0)
        for x, c in zip(xs, cs):
            gain = self.expected_gain(x, reliability=r)["gain"]
            if gain > best[1]:
                best = (float(x), float(gain))
        return best

    def _resolve_screened(self, scr, r):
        """First-maximum rule on screened gains; multi-cell near-ties re-decided directly."""
        xs, cs, gains = scr["x"], scr["cell"], scr["gain"]
        info = {"path": scr["path"], "fallback_reason": scr["fallback_reason"], "resolved": 0}
        self.last_probe_path = info
        gmax = float(gains.max())
        band = self._PROBE_TIE_BAND * (self.hi - self.lo)
        cand = [i for i in range(len(xs)) if gains[i] >= gmax - band]
        if len({int(cs[i]) for i in cand}) == 1:
            i = cand[0]
            return (float(xs[i]), float(gains[i])) if gains[i] > -1.0 else (float(xs[0]), -1.0)
        by_cell = {}
        for i in cand:
            c = int(cs[i])
            if c not in by_cell:
                by_cell[c] = self.expected_gain(xs[i], reliability=r)["gain"]
        info["resolved"] = len(by_cell)
        exact = {i: by_cell[int(cs[i])] for i in cand}
        best = (float(xs[0]), -1.0)
        for i in cand:
            if exact[i] > best[1]:
                best = (float(xs[i]), float(exact[i]))
        return best

    # ------------------------------------------------------------------ bridge
    @classmethod
    def from_regime(cls, regime, backend, *, probe_ids=None, query_weights=None,
                    method: str = "transfer", covariance_backend=None) -> "CompiledRegime":
        """Build the compiled view from a ``RegimePosterior`` WITHOUT its dense internals.

        Reads only the public configuration fields (``lo``, ``hi``, ``reliability``,
        ``p_flip``, ``p_two``, ``potential``, ``n_grid``, ``claim_model``, ``claims``,
        ``probes``). It never calls ``_solve``, ``_with_probes`` or touches ``F``.
        """
        claims = [(c[0], c[1], c[2], c[3], c[4]) for c in regime.claims]
        probes = [tuple(p) for p in regime.probes]
        return cls(regime.lo, regime.hi, backend=backend, reliability=regime.reliability,
                   p_flip=regime.p_flip, p_two=regime.p_two, potential=regime.potential,
                   n_grid=regime.n_grid, claim_model=regime.claim_model, claims=claims,
                   probes=probes, probe_ids=probe_ids, query_weights=query_weights,
                   method=method, covariance_backend=covariance_backend)
