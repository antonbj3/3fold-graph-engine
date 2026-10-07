#!/usr/bin/env python3
"""
claim_federation.py — combine two claim graphs and rank the next experiment.

Contract (plain JSON-able dicts; one per graph):
    {"graph_id": "A", "label": "TEST-GRAPH" | "CURATED-PUBLIC",
     "concepts": [{"id": "grain_size", "aliases": ["d"]}],
     "scales":   {"grain_size_nm": "log"},                       # optional; default linear
     "sources":  [{"id": "s1", "derives_from": []}, {"id": "s2", "derives_from": ["s1"]}],
     "claims":   [{"id": "c1", "subject": "grain_size", "object": "yield_strength", "sign": -1,
                   "validity": {"grain_size_nm": [20, 1e5]},      # box; absent variable = unbounded
                   "evidence": ["s1", "s2"], "cost": 1.0}]}

What the federation does, and the rule behind each step:

* INDEPENDENCE. Every source is traced to its root sources (no `derives_from`). Two evidence items are
  modelled as the mean of their roots' independent unit-variance errors: y = μ·1 + M ε, with M the
  row-normalized report×root incidence. The best linear unbiased estimate of μ has variance
  1/‖M⁺1‖², so the support of a claim side is  N_eff = ‖M⁺1‖²:
      K independent reports → K;   any number of copies of one origin → 1;
      Hall + Petch + a textbook citing both + a handbook citing Hall → 2.
  (Proof: M1 = 1 since rows sum to one, so for full column rank M⁺1_K = 1_roots and ‖·‖² = #roots;
  in general M⁺1 is the minimum-norm weighting of roots that reproduces every report.)
  Two forms were tried first and rejected on the Hall-Petch case: equal-weight Kish K²/ΣΣρ gave
  1.78, and 1ᵀC⁺1 with cosine correlations gave 1.53. Known limit of the model: one review citing two
  independent origins counts as 2. A shared origin ACROSS the two graphs is caught the
  same way, because lineage is resolved after the union.
  This is the STRUCTURAL layer only (who derives from whom). leg_decorrelation_lineage_gate.py calls that layer 1 and
  requires a measured error correlation (layer 2) before two legs are booked as independent; that layer is not composed here.
* VALIDITY. A claim holds inside a box over condition variables. Belief is evaluated AT A POINT: only
  claims whose box contains the point vote there.
* CONF vs BOUNDARY. Two opposite-sign claims on the same (subject, object): boxes intersect → a
  CONTRADICTION (README "conf": the node is opened, no winner is picked; the probe point is the centre
  of the intersection). Boxes disjoint → a REGIME BOUNDARY; the probe point is the middle of the gap,
  which is where the transition has to be.
* INFERRED links. s→x from one graph and x→o from the other, retaining asserted s→o conflicts: sign = product,
  validity = intersection of the two boxes. Empty intersection → rejected and kept in `rejected`
  with the reason. An inferred link carries its two claim ids, is never written into the claim set,
  and is flagged when its two legs share a root source.
* NEXT EXPERIMENT. Candidates are the probe points above plus single-origin claims (N_eff < 2, the
  admission rule "two decorrelated channels").
      value = min(p, 1−p) · extent / cost
  min(p, 1−p) is the expected gain in P(right sign) from a clean answer at the point (expected value
  of perfect information for a two-way decision); extent is the share of the condition domain whose
  sign the outcome newly decides, so the product is the expected number of point-signs set right.
  Ranking: points with no decided sign first (a gap, or a contradiction with balanced support), then
  by value. Measured on planted test graphs (examples/engine_experiments/e2, 8 seeds × 4 noise settings):
  this order beats random and a reach/cost order in every setting. Ranking by value alone matches it
  only when `reliability` is at or above the true source reliability (0.85 there) and loses below it;
  the open-points-first order does not depend on that prior. Tried and rejected: entropy × extent
  (worse than either), and multiplying by (1 + inferred links resting on the pair), which lowered
  accuracy in every setting — kept as `weight_dependents=True`, off by default.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = ["Federation", "Probe", "InferredLink", "box_intersection", "box_gap_point", "lineage_information"]

def lineage_information(root_sets, exact_rank: bool = False) -> float:
    """‖M⁺1‖² for reports whose errors are the means of their roots' independent unit errors (M = row-normalized
    report × root incidence). Which N_eff this is (the repository keeps several, by PURPOSE — see neff_form.py and
    the NEFF_BAR note in leg_decorrelation_lineage_gate.py): it belongs to the VARIANCE-REDUCTION family. It is the
    information the reports carry about one common quantity, in units of one single-root report, with optimal (GLS)
    weights; for exchangeable reports it equals Kish × (information of one report), and for one root per report it is
    the number of distinct roots (= paper_graph.pipeline.confidence_from_legs). It is NOT a decorrelation rank
    (participation ratio, ρ²) and NOT a false-accept exponent (2/(1+ρ)); do not feed it into thresholds defined for those.
    exact_rank=True computes the minimum-norm solution over Q before returning a float; the default keeps rcond."""
    import numpy as np
    R = [frozenset(r) for r in root_sets]
    if not R:
        return 0.0
    allroots = sorted(set().union(*R))
    if all(len(r) == 1 for r in R):
        return float(len(allroots))
    if exact_rank and all(R):
        from fractions import Fraction as _S4Fraction
        from ._exact_rank import _s4_pinv
        Mq = [[_S4Fraction(int(x in r), len(r)) for x in allroots] for r in R]
        Mp, _ = _s4_pinv(Mq)
        wq = Mp @ np.full(len(R), _S4Fraction(1), dtype=object)
        return float(wq @ wq)
    M = np.array([[1.0 if x in r else 0.0 for x in allroots] for r in R]); M /= M.sum(1, keepdims=True)
    w = np.linalg.pinv(M, rcond=1e-9) @ np.ones(len(R))
    return float(w @ w)


Box = dict[str, tuple[float, float]]
INF = float("inf")


def source_family_components(reports):
    """Conservative independent-unit count for verified, family-bound reports.

    Each report is {verified: bool, source_family: [family IDs]}. Reports sharing
    any family, including transitively overlapping mixed reports, are one unit.
    A multi-family report is never several independent observations. These family
    declarations must come from a trusted provenance registry, not node adjacency.
    """
    families = []
    if not isinstance(reports, (list, tuple)):
        return dict(valid=False, count=0, reason_codes=["PROVENANCE_MISSING"])
    for r in reports:
        if not isinstance(r, dict) or type(r.get("verified")) is not bool:
            return dict(valid=False, count=0, reason_codes=["PROVENANCE_MISSING"])
        if r["verified"] is not True:
            continue
        f = r.get("source_family")
        if not isinstance(f, (list, tuple)) or not f or any(not isinstance(x, str) or not x for x in f):
            return dict(valid=False, count=0, reason_codes=["PROVENANCE_MISSING"])
        families.append(set(f))
    components = []
    for f in families:
        intersecting = [c for c in components if c & f]
        for c in intersecting:
            f |= c; components.remove(c)
        components.append(f)
    return dict(valid=True, count=len(components), verified_reports=len(families),
                components=[sorted(c) for c in components], reason_codes=[])


# ------------------------------------------------------------------------------------------------
# box geometry
# ------------------------------------------------------------------------------------------------
def box_intersection(a: Box, b: Box) -> Box | None:
    out: Box = {}
    for v in set(a) | set(b):
        lo = max(a.get(v, (-INF, INF))[0], b.get(v, (-INF, INF))[0])
        hi = min(a.get(v, (-INF, INF))[1], b.get(v, (-INF, INF))[1])
        if lo > hi:
            return None
        out[v] = (lo, hi)
    return out


def _mid(lo: float, hi: float, log: bool = False) -> float:
    return math.sqrt(lo * hi) if (log and lo > 0) else (lo + hi) / 2


def _centre(box: Box, log_vars: frozenset = frozenset()) -> dict[str, float]:
    p = {}
    for v, (lo, hi) in box.items():
        if lo == -INF and hi == INF:
            continue
        p[v] = hi if lo == -INF else lo if hi == INF else _mid(lo, hi, v in log_vars)
    return p


def box_gap_point(a: Box, b: Box, log_vars: frozenset = frozenset()) -> dict[str, float]:
    """A point between two disjoint boxes: mid-gap on every separating variable, centre of the
    overlap on the others."""
    p = {}
    for v in set(a) | set(b):
        alo, ahi = a.get(v, (-INF, INF))
        blo, bhi = b.get(v, (-INF, INF))
        if ahi < blo:
            p[v] = _mid(ahi, blo, v in log_vars)
        elif bhi < alo:
            p[v] = _mid(bhi, alo, v in log_vars)
        else:
            p.update(_centre({v: (max(alo, blo), min(ahi, bhi))}, log_vars))
    return p


def _contains(box: Box, point: dict[str, float]) -> bool:
    """Is `point` inside `box`, checking only the axes the point actually carries?

    An EMPTY box is unrestricted, so it contains every point. A NON-EMPTY box that shares no axis with
    the point is a different case and must not read the same way: the filter checked nothing, and
    reporting containment there made `belief(..., point=...)` return unfiltered beliefs while its caller
    believed they were validity-filtered. The old form was `all(... if v in point)`, a universal over a
    filtered set that is vacuously true when the filter admits nothing -- the same shape as a conditional
    gate passing because its comparison had no comparands. Absence of a shared axis is not agreement,
    for the same reason that an absent validity box is not an identical one (margin_net.validity_relation).
    """
    shared = [(v, box[v]) for v in box if v in point]
    if box and not shared:
        return False
    return all(lo <= point[v] <= hi for v, (lo, hi) in shared)


def _h(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# ------------------------------------------------------------------------------------------------
@dataclass
class Probe:
    kind: str                     # CONTRADICTION | BOUNDARY | INFERRED | SINGLE-ORIGIN
    pair: tuple[str, str]
    point: dict[str, float]
    claims: tuple[str, ...]       # the claim ids that generated the probe (trace)
    entropy: float = 0.0
    dependents: int = 0
    cost: float = 1.0
    value: float = 0.0
    evpi: float = 0.0             # min(p, 1-p): expected gain in P(right decision) from a clean answer
    extent: float = 1.0           # share of the condition domain the outcome decides
    decides: list = field(default_factory=list)   # boxes whose sign is NEWLY decided, one per outcome
    region_if: dict = field(default_factory=dict)   # outcome sign → box the measurement then asserts


@dataclass
class InferredLink:
    subject: str
    via: str
    object: str
    sign: int
    validity: Box
    legs: tuple[str, str]         # claim ids
    p: float
    shared_origin: bool
    status: str = "INFERRED"      # never "ASSERTED"
    contradicts: tuple[str, ...] = ()
    instance: str = ""
    regime: dict = field(default_factory=dict)


@dataclass
class Federation:
    reliability: float = 0.75     # prior P(an independent origin reports the true sign)
    reliability_by_root: dict = field(default_factory=dict)   # per-origin overrides (source_reliability.estimate)
    transitivity: float = 0.6     # prior P(a two-step chain composes), used only to score inferred links
    use_lineage: bool = True      # False = every evidence item counts as independent (the naive baseline)
    use_validity: bool = True     # False = boxes ignored (plain ABC / plain vote baseline)
    claims: dict[str, dict[str, Any]] = field(default_factory=dict)
    _parents: dict[str, list[str]] = field(default_factory=dict)
    _alias: dict[str, str] = field(default_factory=dict)
    _roots_cache: dict[str, frozenset] = field(default_factory=dict)
    _pair_cache: Any = None
    scales: dict[str, str] = field(default_factory=dict)   # variable → "log" | "linear" (default)
    _neff_cache: dict = field(default_factory=dict)

    # -- loading -----------------------------------------------------------------------------------
    def add_graph(self, g: dict[str, Any]) -> None:
        gid = g["graph_id"]
        self.scales.update(g.get("scales", {}))
        for c in g.get("concepts", []):
            canon = self._alias.setdefault(_norm(c["id"]), _norm(c["id"]))
            for a in c.get("aliases", []):
                self._alias.setdefault(_norm(a), canon)
        for s in g.get("sources", []):
            # a source id is global: the same id in both graphs IS the same source
            self._parents.setdefault(s["id"], [])
            self._parents[s["id"]] = sorted(set(self._parents[s["id"]]) | set(s.get("derives_from", [])))
        self._roots_cache.clear()
        self._neff_cache.clear()
        for c in g["claims"]:
            cid = f"{gid}:{c['id']}"
            self.claims[cid] = {
                "id": cid, "graph": gid, "pair": (self.canon(c["subject"]), self.canon(c["object"])),
                "sign": 1 if c["sign"] > 0 else -1,
                "box": {v: (float(lo), float(hi)) for v, (lo, hi) in c.get("validity", {}).items()},
                "evidence": list(c.get("evidence", [])), "cost": float(c.get("cost", 1.0)),
                "instance": c.get("instance", ""), "regime": c.get("regime", {}),
                # Kept for claim_types.typecheck_link, which needs the declared units and any
                # per-claim scale override.  Dropping them here is why the typed gate could not fire
                # on real federated claims: typed_throw_ops.compose already emits `units`.
                "units": dict(c.get("units", {})) if isinstance(c.get("units"), dict) else {},
                "scales": dict(c.get("scales", {})) if isinstance(c.get("scales"), dict) else {},
                # `value` likewise: a claim that states a NUMBER does not compose with one that states
                # a SIGN, and the typed gate cannot see that unless the number survives ingest. Found
                # by spanning the gate's own rejection reasons instead of testing three of them.
                "value": c.get("value"),
            }

    @property
    def _logv(self) -> frozenset:
        return frozenset(v for v, k in self.scales.items() if k == "log")

    def canon(self, name: str) -> str:
        return self._alias.get(_norm(name), _norm(name))

    # -- independence ------------------------------------------------------------------------------
    def roots(self, source: str) -> frozenset:
        if source not in self._roots_cache:
            seen, stack, out = set(), [source], set()
            while stack:
                s = stack.pop()
                if s in seen:
                    continue
                seen.add(s)
                ps = self._parents.get(s, [])
                if ps:
                    stack.extend(ps)
                else:
                    out.add(s)
            self._roots_cache[source] = frozenset(out)
        return self._roots_cache[source]

    def n_eff(self, sources: Iterable[str]) -> float:
        src = tuple(sorted(sources))
        if not src:
            return 0.0
        if not self.use_lineage:
            return float(len(src))
        if src not in self._neff_cache:
            self._neff_cache[src] = lineage_information(self.roots(x) for x in src)
        return self._neff_cache[src]

    # -- belief at a point -------------------------------------------------------------------------
    def _by_pair(self) -> dict[tuple[str, str], list[dict[str, Any]]]:
        if self._pair_cache is None or self._pair_cache[0] != len(self.claims):
            d = defaultdict(list)
            for c in self.claims.values():
                d[c["pair"]].append(c)
            self._pair_cache = (len(self.claims), d)      # claims are only ever appended
        return self._pair_cache[1]

    def belief(self, pair: tuple[str, str], point: dict[str, float] | None = None,
               claims: list[dict[str, Any]] | None = None) -> tuple[float, float, float]:
        """(P(sign=+), N_eff⁺, N_eff⁻) from the claims valid at `point`."""
        cl = claims if claims is not None else self._by_pair().get(pair, [])
        if self.use_validity and point is not None:
            cl = [c for c in cl if _contains(c["box"], point)]
        pos = [s for c in cl if c["sign"] > 0 for s in c["evidence"]]
        neg = [s for c in cl if c["sign"] < 0 for s in c["evidence"]]
        npos, nneg = self.n_eff(pos), self.n_eff(neg)
        if self.reliability_by_root:
            # per-origin weights: each root votes once (copies collapsed), with log-odds log(r_root/(1−r_root))
            lo = 0.0
            for sign, srcs in ((1, pos), (-1, neg)):
                for root in {r for s in srcs for r in self.roots(s)}:
                    rr = self.reliability_by_root.get(root, self.reliability)
                    lo += sign * math.log(rr / (1 - rr))
        else:
            lo = (npos - nneg) * math.log(self.reliability / (1 - self.reliability))
        return 1 / (1 + math.exp(-lo)), npos, nneg

    # -- conf / boundary ---------------------------------------------------------------------------
    def stress_points(self) -> list[Probe]:
        out = []
        for pair, cl in self._by_pair().items():
            seen = set()
            for a in cl:
                for b in cl:
                    if a["sign"] > 0 > b["sign"]:
                        from .typed_throws import Claim, regime_intersection
                        if a.get("instance") and b.get("instance") and a["instance"] != b["instance"]:
                            continue
                        if regime_intersection([a.get("regime", {}), b.get("regime", {})]) is None:
                            continue
                        inter = box_intersection(a["box"], b["box"])
                        if inter is not None or not self.use_validity:
                            kind, pt = "CONTRADICTION", _centre(inter or a["box"], self._logv)
                            region = {1: inter or a["box"], -1: inter or a["box"]}
                            decides = [inter or a["box"]]
                        else:
                            kind, pt = "BOUNDARY", box_gap_point(a["box"], b["box"], self._logv)
                            region = {1: _span(a["box"], b["box"], pt), -1: _span(b["box"], a["box"], pt)}
                            decides = [_span(a["box"], b["box"], pt, gap_only=True), _span(b["box"], a["box"], pt, gap_only=True)]
                        key = (kind, tuple(sorted(pt.items())))
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(Probe(kind, pair, pt, (a["id"], b["id"]), cost=max(a["cost"], b["cost"]),
                                         region_if=region, decides=decides))
        from .typed_throws import Claim, frustrated_cycles, box_intersection_many
        signed = [Claim(c["id"], c["graph"], "sign", subject=c["pair"][0], object=c["pair"][1],
                        sign=c["sign"], box=c["box"] if self.use_validity else {},
                        regime=c.get("regime", {}), instance=c.get("instance", ""))
                  for c in self.claims.values()]
        for cycle in frustrated_cycles(signed):
            if cycle.cls != "contradiction" or len(cycle.claims) == 2:
                continue
            cl = [self.claims[cid] for cid in cycle.claims]
            region = box_intersection_many([c["box"] for c in cl]) if self.use_validity else {}
            out.append(Probe("CYCLE", cl[-1]["pair"], _centre(region or {}, self._logv), cycle.claims,
                             cost=max(c["cost"] for c in cl), decides=[region or {}],
                             region_if={1: region or {}, -1: region or {}}))
        return out

    # -- inferred links ----------------------------------------------------------------------------
    def inferred_links(self, cross_graph_only: bool = True,
                       typecheck: bool = False) -> tuple[list[InferredLink], list[dict]]:
        """`typecheck=True` runs claim_types.typecheck_link on every candidate pair before proposing it,
        so a link only enters when the shared concept resolves to ONE canonical id, the shared variable
        is on the same scale on both sides, and declared units agree dimensionally. Off by default: it
        removes links that the untyped path proposes, and callers that counted them must opt in."""
        bp = self._by_pair()
        by_subject = defaultdict(list)
        for c in self.claims.values():
            by_subject[c["pair"][0]].append(c)
        links, rejected = [], []
        for a in self.claims.values():
            s, x = a["pair"]
            for b in by_subject.get(x, []):
                o = b["pair"][1]
                if o == s or (cross_graph_only and a["graph"] == b["graph"]):
                    continue
                from .typed_throws import regime_intersection
                if a.get("instance") and b.get("instance") and a["instance"] != b["instance"]:
                    continue
                regime = regime_intersection([a.get("regime", {}), b.get("regime", {})])
                if regime is None:
                    rejected.append({"legs": (a["id"], b["id"]), "reason": "categorical regimes do not intersect"})
                    continue
                if typecheck:
                    tc = self._typecheck_pair(a, b)
                    if not tc["ok"]:
                        rejected.append({"legs": (a["id"], b["id"]),
                                         "reason": "typecheck: " + "; ".join(tc["reasons"])})
                        continue
                inter = box_intersection(a["box"], b["box"])
                if inter is None and self.use_validity:
                    rejected.append({"legs": (a["id"], b["id"]), "reason": "validity boxes do not intersect"})
                    continue
                box = inter if (inter is not None and self.use_validity) else dict(a["box"])
                pt = _centre(box, self._logv)
                pa, pb = self.belief(a["pair"], pt)[0], self.belief(b["pair"], pt)[0]
                pa, pb = (pa if a["sign"] > 0 else 1 - pa), (pb if b["sign"] > 0 else 1 - pb)
                ra = frozenset().union(*[self.roots(e) for e in a["evidence"]]) if a["evidence"] else frozenset()
                rb = frozenset().union(*[self.roots(e) for e in b["evidence"]]) if b["evidence"] else frozenset()
                instance = a.get("instance") or b.get("instance", "")
                opposite = []
                for c in bp.get((s, o), []):
                    if c["sign"] == a["sign"] * b["sign"]:
                        continue
                    if instance and c.get("instance") and instance != c["instance"]:
                        continue
                    if regime_intersection([regime, c.get("regime", {})]) is None:
                        continue
                    if self.use_validity and box_intersection(box, c["box"]) is None:
                        continue
                    opposite.append(c["id"])
                links.append(InferredLink(s, x, o, a["sign"] * b["sign"], box, (a["id"], b["id"]),
                                          self.transitivity * pa * pb, bool(ra & rb),
                                          contradicts=tuple(opposite), instance=instance,
                                          regime={k: sorted(v) for k,v in regime.items()}))
        return links, rejected

    def _typecheck_pair(self, a: dict, b: dict) -> dict:
        """Adapter. A federated claim stores `pair` and `box`; claim_types reads subject/object/validity.
        Concepts come from this federation's own alias table, so a name that reaches two canonical ids
        is reported as ambiguous rather than silently canonicalised."""
        from .claim_types import typecheck_link
        concepts: dict[str, list[str]] = {}
        for surface, canon in self._alias.items():
            concepts.setdefault(canon, []).append(surface)
        for c in (a, b):
            for name in c["pair"]:
                concepts.setdefault(name, [])
        def keyed(d: dict) -> dict:
            """Index under BOTH the declared spelling and this federation's canonical one. `canon`
            normalises `grain_size_nm` to `grain size nm`, while a units dict keeps the underscores,
            so a lookup by the canonical name silently missed and the gate passed anything."""
            out = dict(d or {})
            for k, v in list(out.items()):
                out.setdefault(self.canon(k), v)
            return out

        def shape(c: dict, subj: str, obj: str) -> dict:
            out = {"subject": subj, "object": obj, "validity": dict(c["box"]), "sign": c["sign"],
                   "units": keyed(c.get("units")), "scales": keyed(c.get("scales"))}
            if c.get("value") is not None:
                out["value"] = c["value"]
            return out
        return typecheck_link(shape(a, a["pair"][0], a["pair"][1]),
                              shape(b, b["pair"][0], b["pair"][1]),
                              concepts=concepts, scales=self.scales)

    # -- next experiment ---------------------------------------------------------------------------
    def next_experiments(self, include_inferred: bool = True, weight_dependents: bool = False) -> list[Probe]:
        links, _ = self.inferred_links()
        rest = defaultdict(set)                             # distinct inferred (subject, object) per probed pair
        for l in links:
            for leg in l.legs:
                rest[self.claims[leg]["pair"]].add((l.subject, l.object))
        resting = {k: len(v) for k, v in rest.items()}
        probes = self.stress_points()
        for c in self.claims.values():
            if self.n_eff(c["evidence"]) < 2.0:
                probes.append(Probe("SINGLE-ORIGIN", c["pair"], _centre(c["box"], self._logv), (c["id"],), cost=c["cost"],
                                    region_if={1: c["box"], -1: c["box"]}, decides=[c["box"]]))
        if include_inferred:
            for l in links:
                pr = Probe("INFERRED", (l.subject, l.object), _centre(l.validity, self._logv), l.legs,
                           cost=max(self.claims[x]["cost"] for x in l.legs), decides=[l.validity])
                pr.entropy, pr.evpi = _h(l.p), min(l.p, 1 - l.p)
                probes.append(pr)
        best: dict[tuple, Probe] = {}
        for pr in probes:
            if pr.kind != "INFERRED":
                pb = self.belief(pr.pair, pr.point)[0]
                pr.entropy, pr.evpi = _h(pb), min(pb, 1 - pb)
            pr.dependents = resting.get(pr.pair, 0)
            pr.extent = self._extent(pr)
            pr.value = pr.evpi * pr.extent * ((1 + pr.dependents) if weight_dependents else 1) / pr.cost
            key = (pr.pair, tuple(sorted(pr.point.items())))
            priority = (pr.kind == "CYCLE", pr.kind == "CONTRADICTION", pr.value)
            old = best.get(key)
            if old is None or priority > (old.kind == "CYCLE", old.kind == "CONTRADICTION", old.value):
                best[key] = pr
        # open points (no decided sign: a gap, or a balanced contradiction) come first; measured in e2, this
        # order does not depend on the reliability prior, while min(p,1−p) alone loses when the prior is too low.
        return sorted(best.values(), key=lambda p: (p.kind == "INFERRED" or p.evpi < 0.45, -p.value))

    def _extent(self, pr: Probe) -> float:
        dom: dict[str, list[float]] = {}
        for c in self.claims.values():
            for v, (lo, hi) in c["box"].items():
                d = dom.setdefault(v, [INF, -INF])
                if lo > -INF: d[0] = min(d[0], lo)
                if hi < INF: d[1] = max(d[1], hi)
        tot = 0.0
        for box in (pr.decides or [{}]):
            f = 1.0
            for v, (lo, hi) in box.items():
                d = dom.get(v)
                if d and d[1] > d[0] and lo > -INF and hi < INF:
                    # a variable declared "log" is measured in decades (grain size 2 nm–100 µm: on a linear
                    # axis the 8–25 nm gap is 0.02 % of the domain)
                    g = math.log if (self.scales.get(v) == "log" and d[0] > 0) else (lambda x: x)
                    f *= max(g(hi) - g(lo), 0.0) / (g(d[1]) - g(d[0]))
            tot += f
        return tot / max(len(pr.decides), 1)

    def record(self, probe: Probe, sign: int, source: str) -> str:
        """Write a measured outcome back as an asserted claim with a fresh root source. The box it
        asserts comes from the probe: the intersection for a CONTRADICTION; for a BOUNDARY, the span
        from the same-sign claim up to the probe point (one transition between the two claims ⇒
        everything on that side of the point has that sign), so repeated probing bisects the gap."""
        sign = 1 if sign > 0 else -1
        if probe.kind == "CYCLE":
            raise ValueError("a cycle witness needs an explicit edge-specific claim; use add_graph")
        cid = f"MEAS:{len(self.claims)}"
        self._parents.setdefault(source, [])
        box = probe.region_if.get(sign) or {v: (x, x) for v, x in probe.point.items()}
        self.claims[cid] = {"id": cid, "graph": "MEAS", "pair": probe.pair, "sign": sign, "box": dict(box),
                            "evidence": [source], "cost": probe.cost}
        return cid


def _span(same: Box, other: Box, point: dict[str, float], gap_only: bool = False) -> Box:
    """Box from `same` up to `point` on the variables that separate `same` from `other`.
    gap_only: only the part between `same` and `point` (what a measurement at `point` newly decides)."""
    out = {}
    for v in set(same) | set(other):
        lo, hi = same.get(v, (-INF, INF)); olo, ohi = other.get(v, (-INF, INF))
        if hi < olo:
            out[v] = (hi if gap_only else lo, point[v])
        elif ohi < lo:
            out[v] = (point[v], lo if gap_only else hi)
        else:
            out[v] = (max(lo, olo), min(hi, ohi))
    return out


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").replace("-", " ").split())
