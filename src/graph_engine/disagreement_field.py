#!/usr/bin/env python3
"""
disagreement_field.py — the interval between two claims is not empty. Two things fill it, and they can be compared.

A pair (subject, object) carries claims on boxes of a condition variable x. `regime_posterior` fills the whole domain
from those claims: P₊^claims(x), the posterior probability that dY/dX is + at x (telegraph prior: at most one sign
change, plus the collision family). That filling uses no mechanism — only the boxes, the signs and the lineage weights.

A first-principles MODEL fills the same interval independently: given y = model(x, **params) and the DECLARED parameter
box, the share of that box on which sign(dy/dx) is + at x is P₊^model(x) — the same scan `hidden_variable.from_mechanism`
runs over (x, θ), read as a field in x instead of a score per parameter. A model whose sign is certified on the whole box
(`claim_types.deductive_certificate` returns ok) gives 0 or 1 everywhere; an uncertified parameter box gives a ramp whose
slope is exactly the model's remaining ignorance about θ.

Where the two fillings disagree most is the stress point: either the claims are wrong there, or the model is, or the
declared parameter box is missing an axis. That x is a probe address that neither side produces alone — `best_probe`
maximizes the expected drop of the claims' own entropy and therefore never looks where the claims are confident and
wrong, and the model has no probe rule at all.

THE LINEAGE RULE. A probe answered by the MODEL instead of by the world is not evidence about the world; it is evidence
about what the model says, and every such answer has the same root: the model. `model_probes` enters K model answers with
`weight=1/K` each, so the K of them carry N_eff = 1 answer's worth in total — the ρ = 1 limit of the tempering rule
already in `RegimePosterior.add_probe` (N_eff = K / (1 + (K−1)ρ)). One model = one root = one independent answer, however
finely it is sampled. Synthetic data drawn from a model is never independent of the model. The lineage tag
"model:<name>" is recorded on the posterior (`model_probe_lineage`) so a later federation can see the root.

THE FACIT THAT IS NOT PROXIMITY. `LinkedWorld` plants hidden links: two pairs share a latent θ, so their transition
points coincide (t = g(θ_shared)) although nothing in the claim graph says so — the sources, the boxes and the graph
distance are drawn independently of θ. A throw "p ↔ q" is TRUE exactly when p and q share θ. The measurable consequence
is in the disagreement field: the model (right family, θ unknown) is the SAME field for both, so the two gaps
P₊^model − P₊^claims move together only when the two claim fields sit at the same transition. Ranking candidate pairs by
the correlation of their disagreement fields is therefore a throw rule that reads a shared mechanism, not a shared
neighbourhood; e37 measures it against random and against effective-resistance proximity in the claim graph.

Honest scope: when one model covers every pair (family known, θ unknown) the gap correlation equals the correlation of
the claim fields up to a common additive field, so the model contributes the frame, not extra signal — e37 reports the
claims-only correlation as a third arm rather than hiding it. The model earns separate signal only when the declared
parameter boxes differ per pair.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

__all__ = ["SignField", "Disagreement", "model_sign_field", "claims_sign_field", "disagreement_field",
           "stress_probe", "model_probes", "gap_correlation", "link_scores", "LinkedWorld", "harvest", "resonance"]


# ------------------------------------------------------------------------------------------------
# mechanism families (one transition: + then −, the peak at θ)
# ------------------------------------------------------------------------------------------------
def harvest(x: float, theta: float = 0.5) -> float:
    """Harvested yield against effort: y = x(1 − x/2θ). dy/dx = 1 − x/θ: + below θ, − above."""
    return float(x * (1.0 - x / (2.0 * theta)))


def resonance(x: float, theta: float = 0.5, damping: float = 0.04) -> float:
    """Resonance amplitude against drive frequency: a Lorentzian peaking at x = θ. Same sign structure as `harvest`
    (+ below θ, − above) through a completely different formula — the model field only sees the sign."""
    u = x / theta
    return float(1.0 / ((1.0 - u * u) ** 2 + damping))


# ------------------------------------------------------------------------------------------------
# the two fillings of the interval
# ------------------------------------------------------------------------------------------------
@dataclass
class SignField:
    xs: np.ndarray
    p_plus: np.ndarray
    source: str                      # "model:<name>" | "claims"
    meta: dict = field(default_factory=dict)


def _param_points(params: dict[str, tuple[float, float]], n_p: int, max_points: int, seed: int) -> list[dict]:
    """The declared box, scanned: the full product grid when it is small, else corners + midpoint + a fixed random
    sample (the same fallback `claim_types.deductive_certificate` uses, so the two agree on what "the box" means)."""
    names = sorted(params)
    if not names:
        return [{}]
    grids = [np.unique(np.linspace(float(params[k][0]), float(params[k][1]), n_p)) for k in names]
    total = int(np.prod([len(g) for g in grids]))
    if total <= max_points:
        return [dict(zip(names, map(float, c))) for c in itertools.product(*grids)]
    rng = np.random.default_rng(seed)
    pts = [dict(zip(names, map(float, c))) for c in itertools.product(*[[g[0], g[-1]] for g in grids])]
    pts.append({k: float(np.median(g)) for k, g in zip(names, grids)})
    while len(pts) < max_points:
        pts.append({k: float(rng.uniform(g[0], g[-1])) for k, g in zip(names, grids)})
    return pts


def model_sign_field(model: Callable[..., float], x_range: tuple[float, float], params: dict[str, tuple[float, float]],
                     n: int = 200, n_p: int = 21, max_points: int = 4000, fixed: dict | None = None,
                     name: str | None = None, seed: int = 0) -> SignField:
    """P₊^model(x): the share of the DECLARED parameter box on which sign(dy/dx) is + at x.

    The scan is the one `hidden_variable.from_mechanism` runs — central differences in x on a grid of the parameter box —
    read as a field in x instead of collapsed to one score per parameter. A model whose sign is certified over the whole
    box (`claim_types.deductive_certificate`, ok=True) returns 1.0 (sign +1) or 0.0 (sign −1) at every x; anything
    between is the model's own ignorance about θ, not noise. Points where dy/dx is exactly 0 count as ½ (a sign the
    model does not declare), so a peak's own apex does not bias the field to either side.

    `fixed` pins parameters out of the scan (they leave the declared box and stop contributing ignorance).
    """
    lo, hi = float(x_range[0]), float(x_range[1])
    if not hi > lo:
        raise ValueError("x_range must be non-degenerate")
    if n < 2:
        raise ValueError("n must be ≥ 2")
    fixed = dict(fixed or {})
    scan = {k: v for k, v in params.items() if k not in fixed}
    pts = _param_points(scan, n_p, max_points, seed)
    xs = np.linspace(lo, hi, n)
    h = (hi - lo) / (n * 10)
    acc = np.zeros(n)
    for p in pts:
        q = dict(fixed); q.update(p)
        d = np.array([model(float(x) + h, **q) - model(float(x) - h, **q) for x in xs])
        acc += np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))
    return SignField(xs, acc / len(pts), f"model:{name or getattr(model, '__name__', 'model')}",
                     {"n_param_points": len(pts), "scanned": sorted(scan), "fixed": fixed})


def claims_sign_field(rp, n: int = 200) -> SignField:
    """P₊^claims(x) from a RegimePosterior: its posterior sign probability, probes included, on n points of [lo, hi]."""
    xs = np.linspace(rp.lo, rp.hi, n)
    return SignField(xs, np.array([rp.p_plus(float(x)) for x in xs]), "claims",
                     {"n_claims": len(rp.claims), "n_probes": len(rp.probes)})


# ------------------------------------------------------------------------------------------------
# the disagreement between them
# ------------------------------------------------------------------------------------------------
def _h2(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1 - 1e-12)
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


@dataclass
class Disagreement:
    xs: np.ndarray
    p_model: np.ndarray
    p_claims: np.ndarray
    gap: np.ndarray                  # SIGNED: p_model − p_claims (this is what co-moves across pairs)
    abs_gap: np.ndarray              # |Δ|
    js: np.ndarray                   # pointwise Jensen–Shannon divergence between Bern(p_model), Bern(p_claims), bits
    integral_abs: float              # ∫|Δ| dx (weighted)
    integral_js: float               # ∫ JS dx (weighted), bits × units of x
    weight_total: float


def disagreement_field(p_model: SignField | np.ndarray, p_claims: SignField | np.ndarray,
                       xs: np.ndarray | None = None, w: np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None
                       ) -> Disagreement:
    """The pointwise disagreement between the two fillings and its integral.

    Two readings, both returned. |Δ| = |P₊^model − P₊^claims| is the measure of x that would be signed differently by
    the two, in the units the loop scores (`closed_loop.wrong_measure`). JS(Bern(p_model) ‖ Bern(p_claims)) is the same
    disagreement in BITS: H((p+q)/2) − (H(p) + H(q))/2, bounded by 1 bit, zero iff the fillings agree, and — unlike |Δ|
    — it grows fastest where both sides are confident and opposed (0.02 vs 0.98 is 0.86 bits; 0.48 vs 0.52 is 0.0006
    bits at a |Δ| twenty times smaller). Integrals by the trapezoid rule; `w` is an optional weight field (array or
    callable of x) for a domain where not every x costs the same to probe."""
    pm = p_model.p_plus if isinstance(p_model, SignField) else np.asarray(p_model, float)
    pc = p_claims.p_plus if isinstance(p_claims, SignField) else np.asarray(p_claims, float)
    if xs is None:
        xs = p_model.xs if isinstance(p_model, SignField) else (p_claims.xs if isinstance(p_claims, SignField) else None)
    if xs is None:
        raise ValueError("xs must be given when neither field is a SignField")
    xs = np.asarray(xs, float)
    if not (len(xs) == len(pm) == len(pc)):
        raise ValueError(f"length mismatch: xs {len(xs)}, model {len(pm)}, claims {len(pc)}")
    if callable(w):
        w = w(xs)
    ww = np.ones_like(xs) if w is None else np.asarray(w, float)
    gap = pm - pc
    js = _h2(0.5 * (pm + pc)) - 0.5 * (_h2(pm) + _h2(pc))
    js = np.clip(js, 0.0, 1.0)
    quad = lambda v: float(np.trapezoid(v * ww, xs)) if hasattr(np, "trapezoid") else float(np.trapz(v * ww, xs))
    return Disagreement(xs, pm, pc, gap, np.abs(gap), js, quad(np.abs(gap)), quad(js), quad(np.ones_like(xs)))


def stress_probe(d: Disagreement, measure: str = "js", tol: float = 1e-9) -> tuple[float, float]:
    """(x, value) — the x that maximizes the pointwise disagreement. `measure`: "js" (bits, default) or "abs".

    TIE RULE, needed because a certified model against a confident claim gives a flat maximal plateau: take the points
    within `tol` of the maximum, group them into contiguous runs, choose the LONGEST run (the widest stressed region,
    not an isolated grid point), and inside it the midpoint — rounded DOWN to the lower of the two middle grid points
    for an even-length run, and among equally long runs the one with the lowest x. Deterministic, and it puts the probe
    in the middle of the contested interval rather than at its edge where one answer settles less."""
    v = d.js if measure == "js" else d.abs_gap
    if measure not in ("js", "abs"):
        raise ValueError("measure must be 'js' or 'abs'")
    m = float(v.max())
    hit = np.flatnonzero(v >= m - tol)
    runs, start = [], hit[0]
    for a, b in zip(hit[:-1], hit[1:]):
        if b != a + 1:
            runs.append((start, a)); start = b
    runs.append((start, hit[-1]))
    lengths = [b - a + 1 for a, b in runs]
    k = int(np.argmax(lengths))                       # argmax takes the first = lowest x on ties
    a, b = runs[k]
    return float(d.xs[(a + b) // 2]), m


def model_probes(rp, p_model: SignField, xs=None, r_model: float = 0.9, name: str = "model",
                 include_undeclared: bool = False) -> list[dict]:
    """Enter the MODEL's answers into the claims posterior as probes with lineage "model:<name>".

    K answers enter with `weight = 1/K` each. Reason: they have one root. `RegimePosterior.add_probe` tempers K answers
    from one judge whose errors are correlated with ρ to N_eff = K / (1 + (K−1)ρ); a model is the ρ = 1 case — resample
    it a thousand times and it still tells you one thing — so N_eff = 1 and each answer enters with 1/K. Sampling the
    model more finely then changes WHERE the single answer's worth of evidence is spread, never how much there is.
    (Test: K answers at one x give exactly the posterior of one answer of the same reliability at that x.)

    The sign at x is + when P₊^model(x) > ½. Points where the model declares no sign (exactly ½ — an uncertified
    parameter box split down the middle) are skipped unless `include_undeclared`, in which case they enter as + (they
    contribute a likelihood ratio of 1 either way at r < 1 only in the symmetric-hypothesis limit, so skipping is the
    honest default). Returns the records, which are also appended to `rp.model_probe_lineage`.
    """
    if not 0.5 <= r_model < 1.0:
        raise ValueError(f"model reliability must be in [0.5, 1): {r_model}")
    if xs is None:
        xs = p_model.xs
        pp = p_model.p_plus
    else:
        xs = np.asarray(xs, float)
        pp = np.interp(xs, p_model.xs, p_model.p_plus)
    keep = [(float(x), float(p)) for x, p in zip(xs, pp) if include_undeclared or abs(p - 0.5) > 1e-12]
    if not keep:
        return []
    wt = 1.0 / len(keep)
    out = []
    for x, p in keep:
        sg = 1 if p >= 0.5 else -1
        rp.add_probe(x, sg, reliability=r_model, weight=wt)
        out.append({"x": x, "sign": sg, "reliability": r_model, "weight": wt, "lineage": f"model:{name}",
                    "p_plus_model": p})
    if not hasattr(rp, "model_probe_lineage"):
        rp.model_probe_lineage = []
    rp.model_probe_lineage.extend(out)
    return out


def gap_correlation(a: Disagreement, b: Disagreement, baseline: np.ndarray | None = None) -> float:
    """Pearson correlation of two SIGNED disagreement fields on a common grid. Two pairs whose model-vs-claims gaps
    co-move are candidates for a shared latent parameter: the model is the same for both, so the only thing that can
    align the gaps is the claims landing on the same mechanism. 0.0 when either field is flat.

    `baseline` (an array on the same grid, normally the COHORT MEAN gap, see `link_scores`) is subtracted from both
    first. Without it the statistic saturates: every pair of the same family has the same + → − shape, so raw gap
    correlations sit at 0.97–0.99 for linked and unlinked alike (measured, 8 worlds) and the transition location — the
    only thing θ controls — is a small residual on top of a large common field. Against the cohort mean the common
    field is gone and the residual is what is correlated."""
    ga, gb = a.gap, b.gap
    if baseline is not None:
        ga, gb = ga - np.asarray(baseline, float), gb - np.asarray(baseline, float)
    x, y = ga - ga.mean(), gb - gb.mean()
    den = float(np.sqrt((x @ x) * (y @ y)))
    return 0.0 if den <= 1e-300 else float((x @ y) / den)


def link_scores(ds: list[Disagreement]) -> np.ndarray:
    """The symmetric matrix of cohort-baselined gap correlations over a list of pairs' disagreement fields: the throw
    score for "p and q share a latent parameter". Baseline = the mean gap field over the cohort (what every pair of the
    family has in common, including the model's own ignorance ramp), so only the pair-specific residual is compared."""
    G = np.array([d.gap for d in ds], float)
    R = G - G.mean(0, keepdims=True)
    R = R - R.mean(1, keepdims=True)
    s = np.sqrt(np.einsum("ij,ij->i", R, R))
    s = np.where(s <= 1e-300, 1.0, s)
    C = (R @ R.T) / np.outer(s, s)
    np.fill_diagonal(C, 0.0)
    return C


# ------------------------------------------------------------------------------------------------
# a world with PLANTED HIDDEN LINKS
# ------------------------------------------------------------------------------------------------
from .closed_loop import World                                    # noqa: E402  (after __all__, before use)


@dataclass
class LinkedWorld(World):
    """`closed_loop.World` with LATENT SHARED PARAMETERS. `n_links` disjoint couples of pairs each draw one θ and both
    members put their transition at t = g(θ) = θ (plus `link_jitter` of measurement-scale noise); every other pair draws
    its own θ. All pairs are one-transition peaks (start +, the harvest/resonance family), so a model given the right
    family and an undeclared θ is correct for every pair and wrong about every pair in the same way.

    The claim graph cannot see θ: the boxes, the sources and the copy structure are drawn exactly as in `World`, before
    the planting, and the claims are then re-derived against the planted truth with the same source reliabilities and
    the same copy relations (a copy still repeats its parent's report). So `links` is a facit for a throw that CANNOT be
    read off graph proximity — which is what e37 tests.
    """
    p_transition: float = 1.0
    p_two: float = 0.0
    n_links: int = 3
    link_jitter: float = 0.02
    theta_range: tuple = (0.2, 0.8)
    links: list = field(default_factory=list)      # [(p, q)] the planted couples
    theta: list = field(default_factory=list)      # per pair: the latent parameter
    family: list = field(default_factory=list)     # per pair: "harvest" | "resonance"

    def __post_init__(self):
        super().__post_init__()
        if 2 * self.n_links > self.n_pairs:
            raise ValueError("n_links needs 2 pairs each")
        rng = np.random.default_rng(self.seed + 9901)
        self.theta = [0.0] * self.n_pairs
        self.family = ["harvest"] * self.n_pairs
        for i in range(self.n_links):
            p, q = 2 * i, 2 * i + 1
            th = float(rng.uniform(*self.theta_range))
            fam = "harvest" if rng.random() < 0.5 else "resonance"
            self.links.append((p, q))
            for r in (p, q):
                self.theta[r] = th
                self.family[r] = fam
                self.transitions[r] = [float(np.clip(th + rng.normal(0.0, self.link_jitter), 0.06, 0.94))]
                self.start_sign[r] = 1
        for r in range(2 * self.n_links, self.n_pairs):
            th = float(rng.uniform(*self.theta_range))
            self.theta[r] = th
            self.family[r] = "harvest" if rng.random() < 0.5 else "resonance"
            self.transitions[r] = [th]
            self.start_sign[r] = 1
        self._regenerate_claims(rng)

    def _regenerate_claims(self, rng) -> None:
        """Same boxes, same sources, same copies — signs re-drawn against the planted truth. One noisy draw per
        (pair, box, root), reused by that root's copies, so the lineage structure `World` built is preserved exactly."""
        rel = list(self.source_reliability)[: self.n_sources]
        seen: dict = {}
        out = []
        for p, a, b, _sg, roots in self.claims:
            key = (p, round(a, 9), round(b, 9), tuple(sorted(roots)))
            if key not in seen:
                s = self.majority_sign(p, a, b)
                if rng.random() > rel[min(roots)]:
                    s = -s
                seen[key] = s
            out.append((p, a, b, seen[key], roots))
        self.claims = out

    def is_link(self, p: int, q: int) -> bool:
        return (min(p, q), max(p, q)) in {(min(a, b), max(a, b)) for a, b in self.links}

    def true_links(self) -> set:
        return {(min(a, b), max(a, b)) for a, b in self.links}

    def model_of(self, p: int) -> Callable[..., float]:
        """The mechanism family of pair p as a callable model(x, theta) — the right family, θ still undeclared."""
        return harvest if self.family[p] == "harvest" else resonance
