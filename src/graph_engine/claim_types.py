#!/usr/bin/env python3
"""
claim_types.py — the CHECKER layer: a proposer proposes a claim, a checker decides whether it is well-typed, and only
then does it enter the graph.

Everything upstream in this package proposes. `numeric_rules` reads v ± σ with a unit and a surface quantity phrase out
of prose; `typed_extraction` reads mechanism atoms and a sign out of a sentence; `claim_federation` composes a → b into
an inferred link whenever the two claims share a concept and their boxes intersect. None of them asks whether the thing
proposed is even the right KIND of object: a hazard ratio carrying "mmHg", a mass carrying seconds, a two-step inference
through a variable that is in log₁₀(mg) on one side and mg on the other, all pass. They are not wrong numbers; they are
type errors, and a type error costs more downstream than a wrong number because it is invisible in every aggregate.

This module decides three things, and refuses rather than guesses:

1. DIMENSIONS. A unit grammar (`dimension`) maps a unit string to a vector over (M, L, T, I, Θ, N, J), with two extra
   kinds that are not vectors: `dimensionless` (the zero vector: %, ratio, branching fraction) and `count` (events,
   patients — a count is not a dimensionless real, adding one to a fraction is an error). SI base units with the usual
   prefixes, the derived units this repository's two corpora actually use (N, J, W, Pa, Hz, V, C, eV, barn, litre),
   clinical units (mmHg, mg/dL, mmol/L, kg/m², beats/min), and the time units prose is written in (min, h, day, week,
   month, year). Products and quotients with exponents parse: "kg/m^2", "GeV/c^2", "mL/min", "mmol/L".
   eV and its prefixes carry `natural=True`: in ħ = c = 1 every dimension reduces to one power of energy
   (M → E, L → E⁻¹, T → E⁻¹), so a mass in GeV and a cross section in GeV⁻² type-check against M and L² only when one
   side declares natural units. Outside that flag, GeV is an energy and nothing else.
   `check_numeric` applies three rules to a `numeric_rules` record: a ratio quantity (HR/OR/RR, or `log_scale=True`)
   with ANY unit is a type error; a value with a unit but no quantity phrase is rejected; and a quantity phrase that
   resolves in the canonical table (mass → M, blood pressure → pressure, cross section → area, duration → T, …) must
   agree with the unit's dimension. A quantity that does not resolve is ACCEPTED and said to be unchecked — the table is
   small on purpose, and silence about an unknown quantity is not a verdict about it.

2. CERTIFICATES. A claim may carry `certificate: {"kind": "deductive" | "statistical" | "none"}`.
   `deductive_certificate` is the Lean-style half: given a mechanism `model(x, **params)`, a box in x and in every
   parameter, and a claimed sign, it scans the whole box (the same derivative scan `hidden_variable.from_mechanism`
   uses to find sign flips, here run over the parameter PRODUCT, not one parameter at a time) and either certifies the
   sign on the box it actually verified or returns a counterexample point. It is a proof on a grid, not a proof: the
   certificate carries `n_points`, the grid, and the smallest |dy/dx| seen, so a reader can see how close the scan came
   to the flip. `statistical_certificate` wraps `record_guarantee.fit_threshold` — the distribution-free "≥ 1 − α of the
   admitted records are entirely right, with probability ≥ 1 − δ" — into a certificate on one record.
   `reliability_of` is what a reader uses instead of a flat prior: deductive → 1 − 1e-6 inside its verified box (and the
   default outside it), statistical → 1 − α, none → the caller's default. The asymmetry is the point: a claim that
   carries a proof is not the same object as a claim that carries a source.

3. LINK TYPECHECK. `typecheck_link` is the gate `claim_federation`'s inferred links should pass before they are
   proposed (this module does not import or change that module). For a: X → Y and b: Y → Z it checks that the shared
   concept resolves, through the alias table, to ONE canonical id on both sides; that the validity boxes intersect on
   every shared variable; that the shared variable is on the same SCALE on both sides (log vs linear, per-claim
   declaration overriding the graph's `scales`) — a box [1, 3] in log₁₀(mg) and a box [1, 3] in mg intersect
   numerically and mean nothing in common, which is exactly the error a numeric intersection cannot see; that the units
   declared for the shared variable have the same dimension; and that the two claims are the same kind of statement,
   both signs or both numbers with composable dimensions. Failures come back as a list of reasons, never as a repair.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "DimVector", "BASE_DIMENSIONS", "dimension", "same_dimension", "canonical_dimension", "check_numeric",
    "deductive_certificate", "statistical_certificate", "reliability_of", "typecheck_link",
]

BASE_DIMENSIONS = ("M", "L", "T", "I", "Th", "N", "J")
_Z7 = (0.0,) * 7


def _vec(**kw: float) -> tuple[float, ...]:
    return tuple(float(kw.get(b, 0.0)) for b in BASE_DIMENSIONS)


@dataclass(frozen=True)
class DimVector:
    """Exponents over (M, L, T, I, Θ, N, J). `count` marks a counting unit (events, patients): a count is NOT the same
    type as a dimensionless real. `natural` marks a unit stated in natural units (eV family), where M, L⁻¹ and T⁻¹ are
    the same dimension — comparisons then go through the energy power M − L − T."""
    exponents: tuple[float, ...] = _Z7
    count: bool = False
    natural: bool = False

    @property
    def kind(self) -> str:
        if self.count:
            return "count"
        return "dimensionless" if all(abs(e) < 1e-12 for e in self.exponents) else "physical"

    @property
    def energy_power(self) -> float | None:
        """M − L − T, the single exponent every dimension collapses to when ħ = c = 1; None if I/Θ/N/J are involved."""
        m, l, t, i, th, n, j = self.exponents
        return None if any(abs(x) > 1e-12 for x in (i, th, n, j)) else m - l - t

    def __mul__(self, other: "DimVector") -> "DimVector":
        return DimVector(tuple(a + b for a, b in zip(self.exponents, other.exponents)),
                         self.count or other.count, self.natural or other.natural)

    def __truediv__(self, other: "DimVector") -> "DimVector":
        return DimVector(tuple(a - b for a, b in zip(self.exponents, other.exponents)),
                         self.count or other.count, self.natural or other.natural)

    def __pow__(self, k: float) -> "DimVector":
        return DimVector(tuple(a * k for a in self.exponents), self.count, self.natural)

    def __str__(self) -> str:
        if self.count:
            return "count"
        parts = [b if abs(e - 1) < 1e-12 else f"{b}^{e:g}" for b, e in zip(BASE_DIMENSIONS, self.exponents)
                 if abs(e) > 1e-12]
        return ("·".join(parts) or "1") + ("*" if self.natural else "")


DIMENSIONLESS = DimVector()
COUNT = DimVector(count=True)
MASS = DimVector(_vec(M=1))
LENGTH = DimVector(_vec(L=1))
TIME = DimVector(_vec(T=1))
AREA = DimVector(_vec(L=2))
VOLUME = DimVector(_vec(L=3))
ENERGY = DimVector(_vec(M=1, L=2, T=-2))
PRESSURE = DimVector(_vec(M=1, L=-1, T=-2))
TEMPERATURE = DimVector(_vec(Th=1))
AMOUNT = DimVector(_vec(N=1))
FREQUENCY = DimVector(_vec(T=-1))
SPEED = DimVector(_vec(L=1, T=-1))

# ------------------------------------------------------------------------------------------------
# 1. the unit grammar
# ------------------------------------------------------------------------------------------------
_PREFIX = {"y": -24, "z": -21, "a": -18, "f": -15, "p": -12, "n": -9, "u": -6, "µ": -6, "μ": -6, "m": -3,
           "c": -2, "d": -1, "da": 1, "h": 2, "k": 3, "M": 6, "G": 9, "T": 12, "P": 15, "E": 18, "Z": 21, "Y": 24}

# symbol -> (DimVector, may take an SI prefix)
_ATOMS: dict[str, tuple[DimVector, bool]] = {
    # SI base
    "g": (MASS, True), "m": (LENGTH, True), "s": (TIME, True), "A": (DimVector(_vec(I=1)), True),
    "K": (TEMPERATURE, True), "mol": (AMOUNT, True), "cd": (DimVector(_vec(J=1)), False),
    # derived
    "N": (DimVector(_vec(M=1, L=1, T=-2)), True), "J": (ENERGY, True), "W": (DimVector(_vec(M=1, L=2, T=-3)), True),
    "Pa": (PRESSURE, True), "Hz": (FREQUENCY, True), "V": (DimVector(_vec(M=1, L=2, T=-3, I=-1)), True),
    "C": (DimVector(_vec(I=1, T=1)), True), "L": (VOLUME, True), "l": (VOLUME, True),
    # particle physics
    "eV": (DimVector(ENERGY.exponents, natural=True), True),
    "b": (AREA, True), "barn": (AREA, True), "c": (SPEED, False),
    # time as prose writes it
    "min": (TIME, False), "h": (TIME, False), "hr": (TIME, False), "hour": (TIME, False), "hours": (TIME, False),
    "d": (TIME, False), "day": (TIME, False), "days": (TIME, False), "wk": (TIME, False), "week": (TIME, False),
    "weeks": (TIME, False), "mo": (TIME, False), "month": (TIME, False), "months": (TIME, False),
    "y": (TIME, False), "yr": (TIME, False), "year": (TIME, False), "years": (TIME, False), "sec": (TIME, False),
    # clinical
    "mmHg": (PRESSURE, False), "mmhg": (PRESSURE, False), "cmH2O": (PRESSURE, False), "Torr": (PRESSURE, False),
    "torr": (PRESSURE, False), "atm": (PRESSURE, False), "bpm": (FREQUENCY, False),
    "degC": (TEMPERATURE, False), "°C": (TEMPERATURE, False), "celsius": (TEMPERATURE, False),
    # dimensionless and counts
    "%": (DIMENSIONLESS, False), "1": (DIMENSIONLESS, False), "ratio": (DIMENSIONLESS, False),
    "fraction": (DIMENSIONLESS, False), "unitless": (DIMENSIONLESS, False), "dimensionless": (DIMENSIONLESS, False),
    "rad": (DIMENSIONLESS, False), "sr": (DIMENSIONLESS, False),
    "count": (COUNT, False), "counts": (COUNT, False), "events": (COUNT, False), "event": (COUNT, False),
    "patients": (COUNT, False), "participants": (COUNT, False), "beats": (COUNT, False), "n": (COUNT, False),
}
_SUP = {"⁻": "-", "⁺": "+", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7",
        "⁸": "8", "⁹": "9"}
_FACTOR = re.compile(r"^(?P<u>[^\d^*]+?)(?:\s*(?:\^|\*\*)\s*(?P<e1>[-+]?\d+(?:\.\d+)?)|(?P<e2>[-+]?\d+))?$")


def _atom(sym: str) -> DimVector | None:
    sym = sym.strip()
    if not sym:
        return None
    if sym in _ATOMS:
        return _ATOMS[sym][0]
    for k in (2, 1):                                   # longest prefix first ("da" before "d")
        p, rest = sym[:k], sym[k:]
        if p in _PREFIX and rest in _ATOMS and _ATOMS[rest][1]:
            return _ATOMS[rest][0]
    return None


def dimension(unit_str: str | None) -> DimVector | None:
    """The dimension of a unit string, or None when the grammar does not know it (never a guess).
    Products: "kg m/s^2", "mg/dL", "GeV/c^2", "mmol/L", "beats/min", "fb^-1"."""
    if unit_str is None:
        return None
    u = "".join(_SUP.get(ch, ch) for ch in str(unit_str)).strip()
    u = u.replace("·", "*").replace("×", "*")
    if not u:
        return None
    if u in _ATOMS:                                    # "min", "1", "%" — before any splitting
        return _ATOMS[u][0]
    groups = u.split("/")
    out = DimVector()
    for gi, g in enumerate(groups):
        g = g.strip()
        if not g:
            return None
        for tok in re.split(r"[*\s]+", g):
            if not tok:
                continue
            m = _FACTOR.match(tok)
            if not m:
                return None
            d = _atom(m.group("u"))
            if d is None:
                return None
            e = float(m.group("e1") or m.group("e2") or 1.0)
            out = out * (d ** (e if gi == 0 else -e))
    return out


def same_dimension(a: DimVector | None, b: DimVector | None, allow_natural: bool = True) -> bool:
    """Equal exponents, and count only matches count. With `allow_natural` and either side in natural units, the
    comparison is on the energy power M − L − T (ħ = c = 1): a mass in GeV matches M, a cross section in GeV⁻² matches L²."""
    if a is None or b is None:
        return False
    if a.count != b.count:
        return False
    if all(abs(x - y) < 1e-9 for x, y in zip(a.exponents, b.exponents)):
        return True
    if allow_natural and (a.natural or b.natural):
        ea, eb = a.energy_power, b.energy_power
        return ea is not None and eb is not None and abs(ea - eb) < 1e-9
    return False


# ------------------------------------------------------------------------------------------------
# the canonical table: a quantity phrase -> the dimension(s) it may have. Small on purpose.
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CanonicalQuantity:
    name: str
    dims: tuple[DimVector, ...]
    ratio: bool = False           # a ratio measure: its CI is symmetric in the log and it NEVER carries a unit
    natural_ok: bool = False      # GeV-family units are admissible for it (mass, energy, cross section, width)


_RATIO = CanonicalQuantity("ratio measure", (DIMENSIONLESS,), ratio=True)
def _ci(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.I)


_CANONICAL: list[tuple[re.Pattern, CanonicalQuantity]] = [
    (_ci(r"\b(?:hazard|odds|risk|rate|incidence[-\s]rate|prevalence|mortality|event[-\s]rate)\s+ratios?\b"
                r"|\brelative\s+risks?\b|\b(?:a?HR|a?OR|a?RR|IRR|SHR|SIR|SMR)\b"), _RATIO),
    (_ci(r"\bbranching\s+(?:ratio|fraction)s?\b"), CanonicalQuantity("branching ratio", (DIMENSIONLESS,))),
    # any other "X ratio" is dimensionless too — "sodium/potassium ratio" is not a sodium concentration
    (_ci(r"\bratios?\b"), CanonicalQuantity("ratio", (DIMENSIONLESS,))),
    (_ci(r"\bbody[-\s]mass[-\s]index\b|\bBMI\b"), CanonicalQuantity("body mass index", (DimVector(_vec(M=1, L=-2)),))),
    (_ci(r"\b(?:systolic|diastolic|arterial|blood)\s+pressure\b|\b[SD]BP\b|\bblood\s+pressure\b"),
     CanonicalQuantity("blood pressure", (PRESSURE,))),
    (_ci(r"\bcross[-\s]sections?\b|\bsigma\s*\(|\bproduction\s+cross\b"),
     CanonicalQuantity("cross section", (AREA,), natural_ok=True)),
    (_ci(r"\bintegrated\s+luminosit(?:y|ies)\b"), CanonicalQuantity("integrated luminosity", (AREA ** -1,), natural_ok=True)),
    (_ci(r"\b(?:decay\s+)?widths?\b|\blifetimes?\b(?=.*\bwidth)"), CanonicalQuantity("width", (ENERGY, FREQUENCY), natural_ok=True)),
    (_ci(r"\bmass(?:es)?\b|^\W*m[_\s]?\{?(?:t|top|W|Z|H|b|c)\}?\}?\W*$|\bbody\s+weight\b"),
     CanonicalQuantity("mass", (MASS,), natural_ok=True)),
    (_ci(r"\b(?:transverse\s+)?momenta?\b|\bp[_\s]?T\b"), CanonicalQuantity("momentum", (DimVector(_vec(M=1, L=1, T=-1)),), natural_ok=True)),
    (_ci(r"\benerg(?:y|ies)\b|\bcentre[-\s]of[-\s]mass\b|\bcenter[-\s]of[-\s]mass\b"),
     CanonicalQuantity("energy", (ENERGY,), natural_ok=True)),
    (_ci(r"\b(?:follow[-\s]?up|duration|lifetime|half[-\s]life|survival\s+time|time\s+to|median\s+time|age)\b"),
     CanonicalQuantity("duration", (TIME,))),
    (_ci(r"\b(?:wavelength|distance|radius|diameter|thickness|height|displacement|length)s?\b"),
     CanonicalQuantity("length", (LENGTH,))),
    (_ci(r"\btemperatures?\b"), CanonicalQuantity("temperature", (TEMPERATURE,), natural_ok=True)),
    (_ci(r"\b(?:heart\s+rate|pulse)\b"), CanonicalQuantity("heart rate", (FREQUENCY, COUNT / TIME))),
    (_ci(r"\b(?:cholesterol|glucose|creatinine|sodium|potassium|h(?:a)?emoglobin|concentrations?)\b"),
     CanonicalQuantity("concentration", (MASS / VOLUME, AMOUNT / VOLUME))),
    (_ci(r"\b(?:doses?|dosage)\b"), CanonicalQuantity("dose", (MASS, MASS / TIME))),
    (_ci(r"\bpressures?\b"), CanonicalQuantity("pressure", (PRESSURE,))),
    (_ci(r"\bvelocit(?:y|ies)\b|\bspeeds?\b"), CanonicalQuantity("velocity", (SPEED,))),
    (_ci(r"\b(?:fractions?|percentages?|shares?|probabilit(?:y|ies)|efficienc(?:y|ies)|purit(?:y|ies)"
                r"|significances?|p[-\s]values?|asymmetr(?:y|ies)|correlation\s+coefficients?)\b"),
     CanonicalQuantity("fraction", (DIMENSIONLESS,))),
    (_ci(r"\bnumbers?\s+of\b|\b(?:event|patient|participant)\s+counts?\b"), CanonicalQuantity("count", (COUNT,))),
]


_PRODUCT = re.compile(r"\btimes\b|×|\bper\b\s+(?:unit|1\b)|\bdivided\s+by\b", re.I)


def canonical_dimension(quantity: str | None) -> CanonicalQuantity | None:
    """The canonical entry a quantity phrase resolves to, or None (unchecked — not a verdict)."""
    if not quantity:
        return None
    q = str(quantity)
    if _PRODUCT.search(q):
        return None      # "cross section times branching fraction" is neither factor; the table abstains on products
    for pat, c in _CANONICAL:
        if pat.search(q):
            return c
    return None


def check_numeric(record: dict[str, Any]) -> dict[str, Any]:
    """Type-check one `numeric_rules` record {value, sigma, unit, quantity, flags, log_scale}.

    Rejects: a ratio quantity (HR/OR/RR, or log_scale=True) carrying any unit; a value with a unit but no quantity
    phrase; a unit the grammar does not know; a quantity with a canonical dimension that the unit does not match; a
    canonical PHYSICAL quantity with no unit at all. A quantity outside the table is accepted as `unchecked`."""
    q = record.get("quantity")
    unit = record.get("unit")
    unit = None if unit in ("", None) else str(unit)
    canon = canonical_dimension(q)
    out = lambda ok, reason: {"ok": bool(ok), "reason": reason,
                              "quantity_kind": canon.name if canon else None,
                              "dimension": str(dimension(unit)) if unit and dimension(unit) else None}
    if not q or not str(q).strip():
        return out(False, "value_with_unit_without_quantity" if unit else "no_quantity")
    if (canon is not None and canon.ratio) or record.get("log_scale"):
        return out(False, "ratio_quantity_with_unit") if unit is not None else out(True, "ratio_dimensionless")
    if unit is not None and dimension(unit) is None:
        return out(False, "unknown_unit")
    if canon is None:
        return out(True, "unchecked_quantity")
    d = dimension(unit) if unit is not None else None
    if d is None:
        if any(c.kind == "dimensionless" for c in canon.dims):
            return out(True, "dimensionless_no_unit")
        return out(False, f"missing_unit_for_{canon.name.replace(' ', '_')}")
    if any(same_dimension(d, c, allow_natural=canon.natural_ok) for c in canon.dims):
        return out(True, "dimension_ok")
    return out(False, f"dimension_mismatch:{canon.name}_expects_{'|'.join(str(c) for c in canon.dims)}_got_{d}")


# ------------------------------------------------------------------------------------------------
# 2. certificates
# ------------------------------------------------------------------------------------------------
def deductive_certificate(model: Callable[..., float], x_range: tuple[float, float], params: dict[str, tuple[float, float]],
                          sign: int, n_x: int = 41, n_p: int = 9, max_points: int = 20000,
                          seed: int = 0) -> dict[str, Any]:
    """Certify sign(dy/dx) == `sign` on the WHOLE box x_range × Π params, by scanning the product grid (corners always
    included). Returns {"kind": "deductive", "ok": True, "x_range": the box verified, ...} or ok=False with a
    counterexample point. This is a grid proof: `n_points`, the grids and the smallest |dy/dx| seen are in the
    certificate so a reader can judge how near the scan came to a flip. `x_range` in the returned certificate is the
    largest verified interval from the low end — never the requested one when a counterexample was found."""
    import numpy as np

    if sign not in (1, -1):
        raise ValueError("sign must be +1 or -1")
    lo, hi = float(x_range[0]), float(x_range[1])
    if not hi > lo:
        raise ValueError("x_range must be non-degenerate")
    xs = np.linspace(lo, hi, n_x)
    h = (hi - lo) / (n_x * 10)
    names = sorted(params)
    grids = [np.unique(np.linspace(float(params[k][0]), float(params[k][1]), n_p)) for k in names]
    if names:
        total = int(np.prod([len(g) for g in grids]))
        if total <= max_points // max(n_x, 1) or total <= 2000:
            import itertools
            pts = [dict(zip(names, c)) for c in itertools.product(*grids)]
        else:                                     # corners + midpoints + a fixed random sample of the box
            import itertools
            rng = np.random.default_rng(seed)
            corners = [dict(zip(names, c)) for c in itertools.product(*[[g[0], g[-1]] for g in grids])]
            mids = [dict(zip(names, [float(np.median(g)) for g in grids]))]
            k = max(0, max_points // max(n_x, 1) - len(corners) - 1)
            rnd = [dict(zip(names, [float(rng.uniform(g[0], g[-1])) for g in grids])) for _ in range(k)]
            pts = corners + mids + rnd
    else:
        pts = [{}]
    min_abs = math.inf
    bad = None
    ok_upto = -1
    for i, x in enumerate(xs):
        good_here = True
        for p in pts:
            p = {k: float(v) for k, v in p.items()}
            dy = float(model(x + h, **p)) - float(model(x - h, **p))
            slope = dy / (2 * h)
            min_abs = min(min_abs, abs(slope))
            if not (slope * sign > 0):
                good_here = False
                if bad is None:
                    bad = {"x": float(x), "params": p, "dydx": float(slope)}
        if good_here and ok_upto == i - 1:
            ok_upto = i
    n_points = int(len(xs) * len(pts))
    base = {"kind": "deductive", "sign": int(sign), "params": {k: [float(params[k][0]), float(params[k][1])] for k in names},
            "n_points": n_points, "n_x": int(n_x), "n_p": int(n_p), "min_abs_slope": float(min_abs),
            "step": float(h)}
    if bad is None:
        return {**base, "ok": True, "x_range": [lo, hi], "requested_x_range": [lo, hi]}
    verified = [lo, float(xs[ok_upto])] if ok_upto >= 1 else None
    return {**base, "ok": False, "x_range": verified, "requested_x_range": [lo, hi], "counterexample": bad}


def statistical_certificate(alpha: float, delta: float, threshold_record: dict[str, Any],
                            score: float | None = None) -> dict[str, Any]:
    """Wrap a `record_guarantee.fit_threshold` result into a certificate on one record. `threshold_record` is that
    dict; `score` is the record's own score (`least_certain_field_score`) when the certificate is attached to a
    record rather than to the rule. ok = a threshold passed at (α, δ) AND this record is admitted by it.
    The guarantee is on the ADMITTED SET (≥ 1 − α entirely right, with probability ≥ 1 − δ), not on this record; that
    is what `reliability_of` reads as 1 − α."""
    thr = float(threshold_record.get("threshold", -1.0))
    n = int(threshold_record.get("n", 0))
    fitted_alpha = float(threshold_record.get("alpha", alpha))
    fitted_delta = float(threshold_record.get("delta", delta))
    cert = {"kind": "statistical", "alpha": float(alpha), "delta": float(delta), "threshold": thr,
            "n_calibration": n, "admitted_in_calibration": int(threshold_record.get("admitted_in_calibration", 0)),
            "ucb_at_threshold": float(threshold_record.get("ucb_at_threshold", 1.0)), "score": score}
    if thr < 0:
        return {**cert, "ok": False, "reason": "no_threshold_passed"}
    if fitted_alpha > alpha + 1e-12 or fitted_delta > delta + 1e-12:
        return {**cert, "ok": False, "reason": f"threshold_fitted_at_looser_level:alpha={fitted_alpha},delta={fitted_delta}"}
    if score is not None and float(score) > thr:
        return {**cert, "ok": False, "reason": "record_not_admitted"}
    return {**cert, "ok": True, "reason": "admitted"}


def reliability_of(certificate: dict[str, Any] | None, default: float, point: Any = None,
                   deductive_epsilon: float = 1e-6) -> float:
    """r for a reader. deductive → 1 − ε inside the box it verified (the default outside it), statistical → 1 − α,
    none/absent/failed → `default`. `point` is x (float) or {"x": …, **params}; None means "do not check the point"
    for a certified box."""
    if not certificate or certificate.get("kind") in (None, "none"):
        return float(default)
    kind = certificate.get("kind")
    if kind == "deductive":
        if not certificate.get("ok") and certificate.get("x_range") is None:
            return float(default)
        box = certificate.get("x_range")
        if box is None:
            return float(default)
        if point is not None:
            x = point.get("x") if isinstance(point, dict) else point
            if x is None or not (box[0] <= float(x) <= box[1]):
                return float(default)
            if isinstance(point, dict):
                for k, (plo, phi) in (certificate.get("params") or {}).items():
                    if k in point and not (plo <= float(point[k]) <= phi):
                        return float(default)
        return 1.0 - float(deductive_epsilon)
    if kind == "statistical":
        return (1.0 - float(certificate.get("alpha", 0.1))) if certificate.get("ok") else float(default)
    return float(default)


# ------------------------------------------------------------------------------------------------
# 3. link typecheck
# ------------------------------------------------------------------------------------------------
def _alias_table(concepts: Any) -> dict[str, set[str]]:
    """surface name (lowercased) -> set of canonical ids it could be. A name that reaches two ids is ambiguous."""
    table: dict[str, set[str]] = {}
    items: Iterable
    if isinstance(concepts, dict):
        items = [{"id": k, "aliases": v} for k, v in concepts.items()]
    else:
        items = concepts or []
    for c in items:
        cid = c["id"] if isinstance(c, dict) else str(c)
        names = [cid] + list((c.get("aliases") or []) if isinstance(c, dict) else [])
        for nm in names:
            table.setdefault(str(nm).strip().lower(), set()).add(cid)
    return table


def _resolve(name: str, table: dict[str, set[str]]) -> tuple[str | None, str | None]:
    ids = table.get(str(name).strip().lower())
    if not ids:
        return None, f"unresolved_concept:{name}"
    if len(ids) > 1:
        return None, f"ambiguous_alias:{name}->{sorted(ids)}"
    return next(iter(ids)), None


def _scale_of(claim: dict, var: str, scales: dict, concept: str | None = None) -> str:
    """The scale a claim uses for a variable: its own declaration first (under the name IT uses, or under the canonical
    concept id), then the graph's, then linear. Resolving through the concept matters: the graph declaring
    `{"SBP": "log"}` says nothing about a claim that calls the same variable "systolic blood pressure"."""
    src = (claim.get("scales") or {})
    for table in (src, scales or {}):
        for key in (var, concept):
            if key is not None and table.get(key):
                return str(table[key]).lower()
    return "linear"


def typecheck_link(claim_a: dict[str, Any], claim_b: dict[str, Any], concepts: Any = (),
                   scales: dict[str, str] | None = None) -> dict[str, Any]:
    """Can a: X → Y and b: Y → Z be composed into X → Z? Checks, in this order and reporting every failure:

      shared concept   a["object"] and b["subject"] resolve, through the alias table, to ONE canonical id, the same one;
      validity boxes   every variable named in both boxes has intersecting intervals (the box of the link is the
                       intersection; a variable absent from a box is unbounded there);
      scale            the shared variable is on the same scale on both sides. A claim may declare `scales`; otherwise
                       the graph's `scales` applies (default linear). log on one side and linear on the other is a
                       rejection even when the intervals intersect numerically — this is the error a numeric
                       intersection cannot see;
      units            units declared for the shared variable (claim["units"][var]) have the same dimension;
      statement kind   both claims are signs, or both are numbers with known dimensions (a sign and a number do not
                       compose into either).

    Returns {"ok", "reasons": [...], "notes": [...], "concept", "validity", "sign", "composed_dimension"}.
    `claim_federation` can call this before proposing an inferred link; this module does not import it."""
    scales = scales or {}
    table = _alias_table(concepts)
    reasons: list[str] = []
    notes: list[str] = []

    # -- shared concept
    a_obj, b_subj = claim_a.get("object"), claim_b.get("subject")
    cid_a, err_a = _resolve(a_obj, table) if a_obj is not None else (None, "missing_object_on_a")
    cid_b, err_b = _resolve(b_subj, table) if b_subj is not None else (None, "missing_subject_on_b")
    for e in (err_a, err_b):
        if e:
            reasons.append(e)
    concept = None
    if cid_a and cid_b:
        if cid_a != cid_b:
            reasons.append(f"shared_concept_mismatch:{cid_a}!={cid_b}")
        else:
            concept = cid_a

    # -- validity boxes
    box_a = claim_a.get("validity") or {}
    box_b = claim_b.get("validity") or {}
    inter: dict[str, list[float]] = {}
    for v in sorted(set(box_a) | set(box_b)):
        alo, ahi = (box_a.get(v) or (-math.inf, math.inf))
        blo, bhi = (box_b.get(v) or (-math.inf, math.inf))
        lo, hi = max(float(alo), float(blo)), min(float(ahi), float(bhi))
        if lo > hi:
            reasons.append(f"disjoint_validity:{v}:[{alo},{ahi}]x[{blo},{bhi}]")
        else:
            inter[v] = [lo, hi]
        if v in box_a and v in box_b:
            sa, sb = _scale_of(claim_a, v, scales), _scale_of(claim_b, v, scales)
            if sa != sb:
                reasons.append(f"scale_mismatch:{v}:{sa}!={sb}")

    # -- scale of the shared variable itself (it may be a box variable or only a units key)
    if a_obj is not None and b_subj is not None:
        sa = _scale_of(claim_a, a_obj, scales, concept)
        sb = _scale_of(claim_b, b_subj, scales, concept)
        name = concept or a_obj
        if sa != sb and f"scale_mismatch:{name}:{sa}!={sb}" not in reasons:
            reasons.append(f"scale_mismatch:{name}:{sa}!={sb}")

    # -- units of the shared variable
    ua = (claim_a.get("units") or {}).get(a_obj) or (claim_a.get("units") or {}).get(concept)
    ub = (claim_b.get("units") or {}).get(b_subj) or (claim_b.get("units") or {}).get(concept)
    if ua is not None and ub is not None:
        da, db = dimension(ua), dimension(ub)
        if da is None:
            reasons.append(f"unknown_unit:{ua}")
        if db is None:
            reasons.append(f"unknown_unit:{ub}")
        if da is not None and db is not None and not same_dimension(da, db):
            reasons.append(f"unit_dimension_mismatch:{ua}({da})!={ub}({db})")
    elif ua is not None or ub is not None:
        notes.append("unit_declared_on_one_side_only")

    # -- statement kind
    kind_a = "number" if claim_a.get("value") is not None else ("sign" if claim_a.get("sign") is not None else None)
    kind_b = "number" if claim_b.get("value") is not None else ("sign" if claim_b.get("sign") is not None else None)
    sign = None
    composed = None
    if kind_a is None or kind_b is None:
        reasons.append("claim_states_neither_sign_nor_value")
    elif kind_a != kind_b:
        reasons.append(f"mixed_statement_kinds:{kind_a}+{kind_b}")
    elif kind_a == "sign":
        sa, sb = int(claim_a["sign"]), int(claim_b["sign"])
        if sa not in (1, -1) or sb not in (1, -1):
            reasons.append("sign_must_be_+1_or_-1")
        else:
            sign = sa * sb
    else:                                   # both numbers: the composed slope is [Z]/[Y] · [Y]/[X] = [Z]/[X]
        dims = {}
        for tag, claim, xv, yv in (("a", claim_a, claim_a.get("subject"), a_obj), ("b", claim_b, b_subj, claim_b.get("object"))):
            us = claim.get("units") or {}
            dx, dy = dimension(us.get(xv)), dimension(us.get(yv))
            if dx is None or dy is None:
                reasons.append(f"number_claim_{tag}_needs_known_units_for_both_variables")
            else:
                dims[tag] = dy / dx
        if len(dims) == 2:
            composed = str(dims["b"] * dims["a"])
            if claim_a.get("value") is not None and claim_b.get("value") is not None:
                sign = int(math.copysign(1, claim_a["value"] * claim_b["value"])) if claim_a["value"] * claim_b["value"] else 0

    return {"ok": not reasons, "reasons": reasons, "notes": notes, "concept": concept,
            "validity": {k: tuple(v) for k, v in inter.items()}, "sign": sign, "composed_dimension": composed}
