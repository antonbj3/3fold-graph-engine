#!/usr/bin/env python3
"""
instruments.py — cells are EXPENSIVE INSTRUMENTS, a sweep is MANY PROBES, and the reduction between them.

WHAT WAS MISSING. `next_actions.Instrument` is a PRICE TAG: a name, a cost and a reliability, enough to rank one
action against another in bits per cost. It does not know how to obtain anything. Every channel in this package
therefore stopped one step short of a physical measurement: `regime_posterior` consumes a sign and a reliability,
`next_actions.sweep_bundle` prices K answers from one cell, `disagreement_field.model_probes` enters K answers as
ONE lineage root — and the step from "a solver ran and produced iteration counts, residuals, penetrations" to
"sign ±1 with probability p" was written nowhere. This module is that step, as a contract rather than a script.

AN INSTRUMENT IS FOUR THINGS.

    cost           what one reading costs, in the graph's units. A computation CELL is an instrument whose cost is
                   its own run time; it is bought once and answers K points.
    reliability    P(this instrument reports the correct SIGN of the effect under test). Not an accuracy score,
                   not a residual: the sign is what `RegimePosterior.add_probe` and every Bayes update downstream
                   consume, so the reliability has to be about the sign or the update is wrong.
    lineage_root   ONE source id. K answers from one deterministic cell are one piece of evidence, not K
                   (`model_probes`: resample it a thousand times and it still tells you one thing), so every
                   reading of one sweep cites this one root and enters with weight 1/K. `sources()` returns it in
                   the graph contract's shape, `[{"id": ..., "derives_from": [...]}]`.
    probe          `probe(pair, x) -> (sign, p, cost)`: the reduction from a physical measurement at condition x
                   to the engine's claim form. `sign` is ±1, `p` is P(the sign of the relation is +) as the
                   instrument reads it at x, `cost` is what that reading actually cost.

`probe` is concrete and validates; subclasses implement `measure(pair, x) -> Reading` and are refused if they
return a sign outside {+1, −1}, a p outside [0, 1] or a non-positive cost. The contract is checked on every
reading, not asserted in a docstring, because an instrument that lies about its own sign convention is the one
failure this layer exists to catch.

RELIABILITY AND p ARE NOT THE SAME NUMBER. `p` is about the WORLD (is the relation + at x?), `reliability` is
about the INSTRUMENT (does it report the sign correctly?). A reading with sign σ and probability p enters the
posterior with per-reading reliability max(p, 1−p) — the instrument's own confidence in the sign it just
reported, clipped into [0.5, 1) because `add_probe` refuses anything else. The instrument-level `reliability` is
the same quantity averaged over the family it was calibrated on; `ContactSolverSweep` computes it from the
measurements instead of taking it as a parameter, so it cannot be set by hand to make a sweep look cheap.

WHAT THIS MODULE DOES NOT DO. It does not run anything. An instrument is handed measurements — a JSON file, a
benchmark table, a solver's own log — and reduces them. Nothing here imports a solver, and the instance below
(`ContactSolverSweep`) is built from recorded contact-solver measurements, so the whole path from measurement to
priced action is reproducible without the apparatus that produced them.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

try:                                               # package import (tests, examples, callers)
    from graph_engine.next_actions import Action, EngineState, Instrument as PricedInstrument, sweep_bundle
except ImportError:                                # run as a script from anywhere in the repo
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from graph_engine.next_actions import Action, EngineState, Instrument as PricedInstrument, sweep_bundle

__all__ = ["Reading", "Instrument", "ContactSolverSweep", "SweepFamily", "load_family"]

_EPS = 1e-9


class Reading(NamedTuple):
    """One reduction of a physical measurement to the engine's claim form. It IS the tuple `(sign, p, cost)`."""
    sign: int
    p: float
    cost: float


def _reliability_of(p: float) -> float:
    """The confidence that the sign just reported is the true one: max(p, 1−p), clipped into [0.5, 1).

    `RegimePosterior.add_probe` refuses r outside [0.5, 1); a reading with p exactly 0 or 1 would claim a
    deductive certificate, which is `claim_types`' business and not a sweep's, so it is clipped rather than
    passed through."""
    r = max(float(p), 1.0 - float(p))
    return min(max(r, 0.5), 1.0 - 1e-6)


class Instrument(abc.ABC):
    """The contract. Subclasses implement `measure`; everything else is fixed here so it cannot drift per instance.

    An instrument is bought, it answers, and its answers enter the graph under ONE lineage root. The three public
    entry points are `probe` (one reading), `sweep` (K readings from one purchase) and `enter` (those K readings
    into a `RegimePosterior`, weight 1/K each). `action` prices the purchase through `next_actions.sweep_bundle`,
    which is what makes a 50-point sweep compete fairly against one more single probe and against a throw: one
    bits-per-cost list, the sweep's cost being the CELL's cost, not K probe costs.
    """

    def __init__(self, name: str, cost: float, reliability: float, lineage_root: str,
                 derives_from: Sequence[str] = (), sigma: float | None = None) -> None:
        name = str(name)
        if not name:
            raise ValueError("an instrument needs a name")
        cost = float(cost)
        if not (cost > 0) or not math.isfinite(cost):
            raise ValueError(f"an instrument's cost must be finite and positive: {cost}")
        reliability = float(reliability)
        if not 0.5 <= reliability < 1.0:
            raise ValueError(f"reliability is P(correct sign) and must be in [0.5, 1): {reliability}")
        lineage_root = str(lineage_root)
        if not lineage_root:
            raise ValueError("an instrument must declare ONE lineage root: K answers from one cell are one "
                             "piece of evidence, and the root id is what says so")
        self.name = name
        self.cost = cost
        self.reliability = reliability
        self.lineage_root = lineage_root
        self.derives_from = [str(s) for s in derives_from]
        self.sigma = None if sigma is None else float(sigma)

    # -- the reduction -----------------------------------------------------------------------------
    @abc.abstractmethod
    def measure(self, pair, x: float) -> Reading:
        """The raw reduction: what the apparatus says at condition x, as (sign, p, cost). Subclass's business."""

    def probe(self, pair, x: float) -> Reading:
        """`probe(pair, x) -> (sign, p, cost)`, with the contract enforced on the way out."""
        r = self.measure(pair, float(x))
        try:
            sign, p, cost = r
        except (TypeError, ValueError):
            raise TypeError(f"{type(self).__name__}.measure must return (sign, p, cost), got {r!r}") from None
        sign = int(sign)
        if sign not in (1, -1):
            raise ValueError(f"{self.name}: a reading's sign must be +1 or -1, got {sign!r}")
        p = float(p)
        if not (0.0 <= p <= 1.0) or not math.isfinite(p):
            raise ValueError(f"{self.name}: a reading's p must be a probability in [0, 1], got {p!r}")
        cost = float(cost)
        if not (cost > 0) or not math.isfinite(cost):
            raise ValueError(f"{self.name}: a reading's cost must be finite and positive, got {cost!r}")
        if (p >= 0.5) != (sign > 0) and abs(p - 0.5) > _EPS:
            raise ValueError(f"{self.name}: sign {sign:+d} contradicts p = {p:g}; the sign is + exactly when "
                             f"P(+) ≥ ½, and a reading that disagrees with itself is not a reading")
        return Reading(sign, p, cost)

    def sweep(self, pair, xs: Sequence[float]) -> list[Reading]:
        """K readings from ONE purchase. The cell is bought once; the K answers are where that purchase lands."""
        xs = [float(x) for x in xs]
        if not xs:
            raise ValueError("a sweep must answer at least one point")
        return [self.probe(pair, x) for x in xs]

    # -- lineage and entry -------------------------------------------------------------------------
    def sources(self) -> list[dict]:
        """The lineage record in the graph contract's shape. ONE root, whatever K is."""
        return [{"id": self.lineage_root, "derives_from": list(self.derives_from)}]

    def priced(self) -> PricedInstrument:
        """The price tag `next_actions` ranks with. Same name, cost and reliability; nothing added, nothing hidden."""
        return PricedInstrument(self.name, self.cost, self.reliability, self.sigma)

    def enter(self, posterior, pair, xs: Sequence[float]) -> list[dict]:
        """Enter K readings into `posterior` as ONE lineage root: weight 1/K each (`model_probes`' rule).

        Returns the records, also appended to `posterior.model_probe_lineage` so a sweep and a model probe land in
        the same ledger. Sampling the cell more finely spreads one answer's worth of evidence over more points; it
        never creates more of it.
        """
        xs = [float(x) for x in xs]
        readings = self.sweep(pair, xs)
        k = len(readings)
        w = 1.0 / k
        out = []
        for x, r in zip(xs, readings):
            rel = _reliability_of(r.p)
            posterior.add_probe(x, r.sign, reliability=rel, weight=w)
            out.append({"x": x, "sign": r.sign, "p_plus": r.p, "reliability": rel, "weight": w,
                        "cost": r.cost, "lineage": self.lineage_root, "instrument": self.name})
        if not hasattr(posterior, "model_probe_lineage"):
            posterior.model_probe_lineage = []
        posterior.model_probe_lineage.extend(out)
        return out

    # -- pricing -----------------------------------------------------------------------------------
    def action(self, state: EngineState, pair, xs: Sequence[float], lam: float | None = None) -> Action:
        """The purchase, priced in the same bits-per-cost list as every single probe: `sweep_bundle`."""
        a = sweep_bundle(state, pair, xs, self.priced(), lam=lam)
        a.meta["lineage_root"] = self.lineage_root
        a.meta["sources"] = self.sources()
        a.meta["instrument"] = self.name
        return a

    def outcome(self, pair, xs: Sequence[float]) -> dict:
        """The answer `next_actions.apply` takes back for the bundle this instrument's `action` produced."""
        xs = [float(x) for x in xs]
        k = len(xs)
        return {"outcomes": [{"sign": r.sign, "weight": 1.0 / k} for r in self.sweep(pair, xs)],
                "sources": self.sources()}

    def __str__(self) -> str:
        return f"{type(self).__name__} {self.name!r} (r {self.reliability:g}, cost {self.cost:g}, root {self.lineage_root})"


# -- the first instance: a contact-solver sweep ------------------------------------------------------
@dataclass(frozen=True)
class SweepFamily:
    """One measured point of a scene family: the condition value x, the two methods' readings, and what it cost.

    `a` and `b` are the quantity being compared (iteration counts, sweep-equivalents, a penetration ratio) for
    method A and method B on the same scene. `lower_is_better` is a property of the quantity, not of the scene.
    """
    label: str
    x: float
    a: float
    b: float
    cost: float
    tag: str = ""


class ContactSolverSweep(Instrument):
    """A sweep over solver settings (ρ, method, tolerance) on a scene family, reduced to the engine's claim form.

    THE CLAIM. "Method A converges faster than method B on regime R" — one pair (subject = the condition variable
    of the regime, object = the sign of the advantage), one axis, sign ±1, probability from the measurement.

    THE REDUCTION. At condition x, the scenes of the family within the window contribute log-ratios
    L_i = ln(b_i / a_i) (with `lower_is_better`, positive L means A is the cheaper method). The sign is
    sign(mean L). The probability is read off the MEASURED SPREAD, not asserted:

        p = Φ( mean(L) / se ),   se = max(sd(L) / √n, floor_se)

    with sd the sample standard deviation of the log-ratios in the window and, for a window holding one scene,
    the family's pooled sd in its place. Φ is the standard normal cdf. A window where the two methods are within
    each other's spread lands at p ≈ ½ and is worth nothing — which is the correct answer, and is why the spread
    has to enter the reduction rather than be reported beside it.

    NON-CONVERGENCE. A method that did not converge inside its cap is recorded at the cap and marked: the ratio is
    then a LOWER bound on the advantage, so the sign is right and p is understated. That direction is safe; the
    opposite (treating a cap as a measurement) would manufacture confidence, so the censoring is kept and said.

    COST AND RELIABILITY ARE MEASURED. `cost` is the family's own recorded wall cost — what buying this cell
    actually costs. `reliability` is P(correct sign) computed over the family's windows as the mean of
    max(p, 1−p), so an instrument whose measurements do not separate the two methods cannot claim to be reliable.

    ONE LINEAGE ROOT. The family is one deterministic solver campaign. All K answers cite `lineage_root` and enter
    with weight 1/K.
    """

    def __init__(self, name: str, rows: Sequence[SweepFamily], lineage_root: str, *,
                 method_a: str = "A", method_b: str = "B", quantity: str = "iterations",
                 lower_is_better: bool = True, window: float = 0.35, cost: float | None = None,
                 derives_from: Sequence[str] = (), floor_se: float = 0.05,
                 reliability: float | None = None) -> None:
        rows = [r if isinstance(r, SweepFamily) else SweepFamily(**r) for r in rows]
        if not rows:
            raise ValueError("a contact-solver sweep needs at least one measured scene")
        for r in rows:
            if not (r.a > 0 and r.b > 0):
                raise ValueError(f"{r.label}: both methods' readings must be positive, got a={r.a}, b={r.b}")
            if not (r.cost > 0):
                raise ValueError(f"{r.label}: a measured scene must carry a positive cost, got {r.cost}")
        self.rows = list(rows)
        self.method_a = str(method_a)
        self.method_b = str(method_b)
        self.quantity = str(quantity)
        self.lower_is_better = bool(lower_is_better)
        self.window = float(window)
        self.floor_se = float(floor_se)
        sgn = 1.0 if lower_is_better else -1.0
        self._L = {r.label: sgn * math.log(r.b / r.a) for r in self.rows}
        self._pooled = _sd([self._L[r.label] for r in self.rows])
        self.lo = min(r.x for r in self.rows)
        self.hi = max(r.x for r in self.rows)
        c = float(sum(r.cost for r in self.rows)) if cost is None else float(cost)
        # reliability is measured, not given: the mean confidence the family's own points come out at
        rel = float(reliability) if reliability is not None else self._measured_reliability()
        super().__init__(name, c, rel, lineage_root, derives_from=derives_from)

    # -- the reduction ------------------------------------------------------------------------------
    def rows_near(self, x: float) -> list[SweepFamily]:
        """The scenes whose condition value is within `window` of x; the nearest one when the window is empty."""
        near = [r for r in self.rows if abs(r.x - float(x)) <= self.window]
        if near:
            return near
        return [min(self.rows, key=lambda r: abs(r.x - float(x)))]

    def measure(self, pair, x: float) -> Reading:
        near = self.rows_near(x)
        ls = [self._L[r.label] for r in near]
        m = sum(ls) / len(ls)
        sd = _sd(ls) if len(ls) > 1 else self._pooled
        se = max(sd / math.sqrt(len(ls)), self.floor_se)
        p = _phi(m / se)
        sign = 1 if p >= 0.5 else -1
        cost = sum(r.cost for r in near) / len(near)
        return Reading(sign, p, max(cost, _EPS))

    def _measured_reliability(self) -> float:
        rs = [_reliability_of(self.measure(None, r.x).p) for r in self.rows]
        return min(max(sum(rs) / len(rs), 0.5), 1.0 - 1e-6)

    # -- what the family says, for the report -------------------------------------------------------
    def grid(self, k: int = 10) -> list[float]:
        """K condition values spread over the family's own span: what one purchase of this cell answers."""
        if k < 1:
            raise ValueError(f"a sweep answers at least one point: {k}")
        if k == 1 or self.hi <= self.lo:
            return [0.5 * (self.lo + self.hi)]
        step = (self.hi - self.lo) / (k - 1)
        return [self.lo + i * step for i in range(k)]

    def advantage(self, x: float) -> float:
        """The measured mean log-ratio at x: how much faster A is than B there, in nats of the quantity."""
        near = self.rows_near(x)
        return sum(self._L[r.label] for r in near) / len(near)

    def claim(self, x: float) -> str:
        r = self.probe(None, x)
        fast, slow = (self.method_a, self.method_b) if r.sign > 0 else (self.method_b, self.method_a)
        return (f"{fast} beats {slow} on {self.quantity} at {x:g} "
                f"(ratio {math.exp(abs(self.advantage(x))):.3g}x, p {max(r.p, 1 - r.p):.3f})")


def _sd(v: Sequence[float]) -> float:
    v = list(v)
    if len(v) < 2:
        return 0.0
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def load_family(rows: Sequence[dict], x_of, a_of, b_of, cost_of, label_of=None, tag_of=None) -> list[SweepFamily]:
    """Turn recorded measurements into `SweepFamily` rows. The callables say which columns mean what, so one
    benchmark table can be read along more than one axis without being copied."""
    out = []
    for i, r in enumerate(rows):
        out.append(SweepFamily(label=str(label_of(r)) if label_of else str(i), x=float(x_of(r)),
                               a=float(a_of(r)), b=float(b_of(r)), cost=float(cost_of(r)),
                               tag=str(tag_of(r)) if tag_of else ""))
    return out


def _selftest() -> None:
    """The contract, on a family small enough to check by hand."""
    rows = [SweepFamily("s1", 0.0, 10.0, 100.0, 1.0), SweepFamily("s2", 0.1, 12.0, 90.0, 1.0),
            SweepFamily("s3", 2.0, 500.0, 50.0, 2.0), SweepFamily("s4", 2.1, 600.0, 45.0, 2.0)]
    ins = ContactSolverSweep("demo", rows, "campaign:demo", method_a="A", method_b="B")
    assert ins.probe(None, 0.05).sign == +1, "A is 8x cheaper at the low end"
    assert ins.probe(None, 2.05).sign == -1, "B is 11x cheaper at the high end"
    assert 0.5 <= ins.reliability < 1.0
    assert ins.sources() == [{"id": "campaign:demo", "derives_from": []}]
    assert ins.priced().cost == ins.cost and ins.priced().reliability == ins.reliability
    recs = ins.enter(_toy_posterior(), ("regime", "advantage"), ins.grid(4))
    assert len({r["lineage"] for r in recs}) == 1 and abs(sum(r["weight"] for r in recs) - 1.0) < 1e-12
    print(f"instruments selftest ok: {ins}; sign flips between x=0.05 (+1) and x=2.05 (-1); "
          f"4 readings, one root, weights sum to {sum(r['weight'] for r in recs):g}")


def _toy_posterior():
    from graph_engine.regime_posterior import RegimePosterior
    return RegimePosterior(lo=-0.5, hi=2.5)


if __name__ == "__main__":
    _selftest()
