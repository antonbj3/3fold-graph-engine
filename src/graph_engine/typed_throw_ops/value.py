"""value.py — what a SET of parts is worth for a target quantity, in the engine's own currency (bits), and why the
existing set value cannot see a mediator.

precision_form.best_set values a set by I(x; y_S) = ½ log det(I + H C Hᵀ/σ²). With conditionally independent rows
this is monotone SUBMODULAR (the docstring's 1 − 1/e argument): the gain of adding c can only shrink as the set grows.
A mediator is the opposite effect — c is worth nothing next to a alone and a lot next to {a, b} — so no submodular
valuation can ever rank a mediator triple above its pairs. `complementarity` below is ≤ 0 for every triple under the
unrestricted value (test_submodular_cannot_see_mediator) and positive under the target-restricted value.

The target-restricted value is the engine's own `value_bits(h, σ, Q)` (entropy drop of Q x only), summed along the
chain rule with `observe` between rows. It is not submodular: the GRAPH_WILD witness
(ASTRA_WAVE4 .../23_GRAPH_WILD_research.py, triple_boundary) already measured an exhaustive triple beating pair
receding by more than 5× on a target variance. Nothing here is a new valuation; it is the existing Q-value applied to
sets.

STRUCTURAL IMAGE of a set of parts (a model of the parts' PORTS, not of their physics):
    variables  one per quantity id
    prior      N(0, prior_var·I), prior_var large (an unknown quantity)
    part p     for each output o:  x_o − Σ_{required inputs i} x_i  = 0 ± σ_p     (AND: o is pinned only if every
               required input is pinned). σ_p by reach: operator 0.05, ports 0.2, words 1.0 — a words-level part can
               close a path only weakly.
    sources    x_s = 0 ± σ_src for quantities marked `given` in the parts (measured / supplied in their source)
    target     Q = e_t
The bits then say whether the set closes a route from given quantities to the target with every intermediate pinned,
and how tightly; they are not physical information. Unit elasticities (sum) are an explicit simplification.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .engine import PrecisionForm
from .parts import Part

SIGMA_BY_REACH = {"operator": 0.05, "ports": 0.2, "words": 1.0}


class StructuralImage:
    def __init__(self, parts: list[Part], prior_var: float = 1e4, sigma_src: float = 0.05,
                 extra_given: tuple[str, ...] = ()):
        self.parts = {p.id: p for p in parts}
        qs = sorted({x.q for p in parts for x in p.inputs + p.outputs})
        self.idx = {q: k for k, q in enumerate(qs)}
        self.d = len(qs)
        self.prior_var = prior_var
        self.sigma_src = sigma_src
        self.given = sorted({x.q for p in parts for x in p.inputs if x.given} | set(extra_given))

    def rows(self, part_ids) -> list[tuple[np.ndarray, float]]:
        out = []
        for pid in part_ids:
            p = self.parts[pid]
            s = SIGMA_BY_REACH[p.reach]
            for o in p.outputs:
                h = np.zeros(self.d)
                h[self.idx[o.q]] = 1.0
                for i in p.inputs:
                    if i.required and i.q != o.q:
                        h[self.idx[i.q]] -= 1.0
                out.append((h, s))
        return out

    def base_form(self, given=None) -> PrecisionForm:
        f = PrecisionForm.zeros(self.d)
        f.J = np.eye(self.d) / self.prior_var
        f._dirty()
        for q in (self.given if given is None else given):
            if q in self.idx:
                h = np.zeros(self.d); h[self.idx[q]] = 1.0
                f.observe(h, self.sigma_src)
        return f

    def value_Q(self, part_ids, target: str, given=None) -> float:
        """Bits of entropy drop of x_target from the part rows, given the source rows. Chain rule over the
        engine's value_bits(·, Q) with observe() between rows (exact for Gaussian rows)."""
        if target not in self.idx:
            return 0.0
        Q = np.zeros((1, self.d)); Q[0, self.idx[target]] = 1.0
        f = self.base_form(given)
        tot = 0.0
        for h, s in self.rows(part_ids):
            tot += f.value_bits(h, s, Q=Q)
            f.observe(h, s)
        return float(tot)

    def value_full(self, part_ids, given=None) -> float:
        """The unrestricted set value (precision_form.set_value_bits) of the same rows — what best_set maximizes."""
        rs = self.rows(part_ids)
        if not rs:
            return 0.0
        H = np.array([h for h, _ in rs]); s = np.array([s for _, s in rs])
        return float(self.base_form(given).set_value_bits(H, s, exact_bernoulli=False))


def triple_terms(v, a, b, c) -> dict:
    """Pair-irreducible parts of a set function v on the triple {a, b, c}.
       complementarity(c | a, b) = [v(abc) − v(ab)] − max(v(ac) − v(a), v(bc) − v(b))   > 0: c pays only with both
       excess_over_pairs         = v(abc) − max(v(ab), v(ac), v(bc))
       interaction (co-information sign convention) = v(abc) − v(ab) − v(ac) − v(bc) + v(a) + v(b) + v(c)
    For a monotone submodular v, complementarity ≤ 0 for every labelling (gains shrink)."""
    V = {}
    for S in [(a,), (b,), (c,), (a, b), (a, c), (b, c), (a, b, c)]:
        V[S] = v(list(S))
    comp = {}
    for x, (y, z) in ((a, (b, c)), (b, (a, c)), (c, (a, b))):
        yz = tuple(sorted((y, z), key=[a, b, c].index))
        xy = tuple(sorted((x, y), key=[a, b, c].index)); xz = tuple(sorted((x, z), key=[a, b, c].index))
        comp[x] = (V[(a, b, c)] - V[yz]) - max(V[xy] - V[(y,)], V[xz] - V[(z,)])
    return {"v": {"+".join(map(str, k)): float(val) for k, val in V.items()},
            "complementarity": {str(k): float(val) for k, val in comp.items()},
            "max_complementarity": float(max(comp.values())),
            "excess_over_pairs": float(V[(a, b, c)] - max(V[(a, b)], V[(a, c)], V[(b, c)])),
            "interaction": float(V[(a, b, c)] - V[(a, b)] - V[(a, c)] - V[(b, c)] + V[(a,)] + V[(b,)] + V[(c,)])}


def mediator_scores(img: StructuralImage, a: str, b: str, c: str, targets: list[str]) -> dict:
    """For each target quantity: triple terms under value_Q, plus the unrestricted-value terms for contrast."""
    out = {}
    for t in targets:
        tq = triple_terms(lambda S: img.value_Q(S, t), a, b, c)
        out[t] = tq
    out["__full__"] = triple_terms(lambda S: img.value_full(S), a, b, c)
    return out


def best_target_for(img: StructuralImage, members: list[str]) -> tuple[str | None, float]:
    """The quantity the set pins best (largest value_Q among outputs of the members)."""
    outs = {o.q for m in members for o in img.parts[m].outputs}
    best, bv = None, 0.0
    for t in sorted(outs):
        v = img.value_Q(members, t)
        if v > bv:
            best, bv = t, v
    return best, bv


# ---------------------------------------------------------------------------------------------------------------
# DIRECTED closure value (used by the corpus run). The Gaussian image above with sum rows can pin a quantity through
# an accidental linear cancellation between two parts that share inputs (inverse use of a port, measured on the
# corpus: 8 bits for a target whose operator input was never supplied). Ports have a direction; the closure below
# follows it: a quantity is available if given, or produced by a part whose REQUIRED inputs are all available.
# Its variance adds along the derivation (var(o) = σ_p² + Σ var(inputs)), which is what the Gaussian image gives on a
# derivation TREE, where no cancellation is possible. value = ½ log2(prior_var / var(target)) bits, 0 if unreachable.
# Monotone in the set; not submodular (AND nodes), so complementarity > 0 marks an irreducible triple.
# ---------------------------------------------------------------------------------------------------------------
def closure(parts: dict, member_ids, given: set, sigma_src: float = 0.05) -> dict:
    from .parts import regime_intersection, box_intersection_many, assumption_conflicts
    from .compose import link
    member_ids = tuple(member_ids)
    selected = [parts[pid] for pid in member_ids]
    if regime_intersection([p.regime for p in selected]) is None or box_intersection_many([p.box for p in selected]) is None:
        return {}
    if any(assumption_conflicts(a.assumptions, b.assumptions) for a,b in combinations(selected,2)):
        return {}
    if any(not l.ok for a in selected for b in selected if a is not b for l in link(a,b)):
        return {}
    var = {q: sigma_src ** 2 for q in given}
    changed = True
    while changed:
        changed = False
        for pid in member_ids:
            p = parts[pid]
            req = [i.q for i in p.inputs if i.required]
            if all(q in var for q in req):
                s2 = SIGMA_BY_REACH[p.reach] ** 2
                v = s2 + sum(var[q] for q in req)
                for o in p.outputs:
                    if o.q not in var or v < var[o.q] - 1e-15:
                        var[o.q] = v
                        changed = True
    return var


def closure_value(parts: dict, member_ids, target: str, given: set, prior_var: float = 1e4) -> float:
    var = closure(parts, member_ids, given)
    return max(0.0, float(0.5 * np.log2(prior_var / var[target]))) if target in var else 0.0
