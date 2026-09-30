"""triples.py — what a triple has that its three pairs do not.

1. MEDIATOR (port level). A and C do not compose directly in either direction (no ok link, or no shared quantity),
   but A→B and B→C (or C→B→A) do, AND the three parts' regimes intersect. B opens a port between two parts that do
   not go together. Graded by value.py: the target-restricted bits of {A,B,C} against every pair.

2. TRIPLE CONF — three claims, every pair jointly satisfiable, the three not. Four kinds are detected:
   sign_cycle   s→x (σ1), x→o (σ2), s→o (σ3) with σ1σ2 ≠ σ3. No two share (subject, object), so
                claim_federation's pairwise conf test cannot fire, and its inferred-link step SKIPS s→o because an
                asserted s→o exists (claim_federation.inferred_links: `(s, o) in bp` → continue). Verified in tests.
   interval     bounds on quantities + background relations (monotone maps, inequalities) from parts: interval
                propagation (HC4-lite) is empty for the three, non-empty for every two.
   order        strict orderings a<b, b<c, c<a: a directed 3-cycle (each pair consistent).
   transitions  sign_at claims along one axis with pattern + − + (two sign changes) against a part's declared
                "at most one transition" rule on that axis (Anton's ordningsmotsägelse; regime_posterior keeps a
                two-transition family at small prior — this names where it is needed).
   Each is then CLASSIFIED by the regimes of the three claims:
       contradiction    the three validity regions share a point: they cannot all hold there
       regime_boundary  every pair of regions intersects but the three do not (possible on categorical axes, or
                        when a mediator links non-adjacent boxes): no point where all three are asserted — the
                        boundary itself is the finding
       ordering         order/transitions kinds whose regions share a point
3. CHAIN REGIME CONF (compose.Chain.regime_conf): links compose pairwise, the whole chain has no regime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations

import numpy as np

from .compose import best_link
from .parts import Claim, Part, regime_intersection, box_intersection_many


# ------------------------------------------------------------------------------------------------ mediator (ports)
@dataclass
class Mediation:
    a: str
    b: str
    c: str
    direction: str                  # "a>b>c" or "c>b>a"
    direct_ac: bool
    regime_ok: bool
    shared: tuple[str, str]
    reasons_ac: list[str] = field(default_factory=list)


def _ok(A, B):
    l = best_link(A, B)
    return (l is not None and l.ok), l


def mediation(A: Part, B: Part, C: Part) -> Mediation | None:
    ab, lab = _ok(A, B); bc, lbc = _ok(B, C)
    cb, lcb = _ok(C, B); ba, lba = _ok(B, A)
    if ab and bc:
        direction, shared = "a>b>c", (lab.q, lbc.q)
    elif cb and ba:
        direction, shared = "c>b>a", (lcb.q, lba.q)
    else:
        return None
    ac, lac = _ok(A, C); ca, lca = _ok(C, A)
    reasons = []
    for l in (lac, lca):
        if l is not None and not l.ok:
            reasons += l.reasons
    reg = regime_intersection([A.regime, B.regime, C.regime]) is not None and \
        box_intersection_many([A.box, B.box, C.box]) is not None
    return Mediation(A.id, B.id, C.id, direction, ac or ca, reg, shared, reasons or (["no_shared_quantity"] if lac is None and lca is None else []))


# ------------------------------------------------------------------------------------------------ claim regions
def _region_ok(claims: list[Claim]) -> bool:
    return regime_intersection([c.regime for c in claims]) is not None and \
        box_intersection_many([c.box for c in claims]) is not None


def classify(claims: list[Claim], kind: str) -> str:
    if _region_ok(claims):
        return "ordering" if kind in ("order", "transitions") else "contradiction"
    pairwise = all(_region_ok([x, y]) for x, y in combinations(claims, 2))
    return "regime_boundary" if pairwise else "disjoint_regions"


@dataclass
class TripleConf:
    kind: str
    claims: tuple[str, str, str]
    cls: str
    detail: str
    parts: tuple[str, ...] = ()


# ------------------------------------------------------------------------------------------------ sign cycle
def sign_cycles(claims: list[Claim]) -> list[TripleConf]:
    S = [c for c in claims if c.kind == "sign" and c.sign in (1, -1)]
    by_pair: dict[tuple[str, str], list[Claim]] = {}
    for c in S:
        by_pair.setdefault((c.subject, c.object), []).append(c)
    out = []
    for a in S:
        for b in S:
            if b is a or b.subject != a.object:
                continue
            for c in by_pair.get((a.subject, b.object), []):
                if c is a or c is b or not same_instance([a, b, c]):
                    continue
                if a.sign * b.sign != c.sign:
                    trio = [a, b, c]
                    out.append(TripleConf("sign_cycle", (a.id, b.id, c.id), classify(trio, "sign"),
                                          f"{a.subject}->{a.object}({a.sign:+d}) · {b.subject}->{b.object}({b.sign:+d}) "
                                          f"vs asserted {c.subject}->{c.object}({c.sign:+d})",
                                          tuple(sorted({a.part, b.part, c.part}))))
    return out


# ------------------------------------------------------------------------------------------------ interval CSP
@dataclass
class Relation:
    """Background relation contributed by a part.  kind 'map': out = f(ins) with per-argument monotonicity
    mono[i] ∈ {+1, −1}; kind 'le': lhs ≤ rhs; kind 'implies': min(premises) ≤ conclusion on {0,1}."""
    part: str
    kind: str
    out: str = ""
    ins: tuple[str, ...] = ()
    f: object = None
    mono: tuple[int, ...] = ()
    lhs: str = ""
    rhs: str = ""


def _propagate(bounds: dict[str, list[float]], rels: list[Relation], iters: int = 50) -> bool:
    """Forward/backward narrowing; returns False on an empty interval."""
    for _ in range(iters):
        changed = False
        for r in rels:
            if r.kind == "map":
                if any(i not in bounds for i in r.ins):
                    continue
                lo_args = [bounds[i][0] if m > 0 else bounds[i][1] for i, m in zip(r.ins, r.mono)]
                hi_args = [bounds[i][1] if m > 0 else bounds[i][0] for i, m in zip(r.ins, r.mono)]
                try:
                    flo, fhi = float(r.f(*lo_args)), float(r.f(*hi_args))
                except (ValueError, ZeroDivisionError, OverflowError):
                    continue
                if not (np.isfinite(flo) or np.isfinite(fhi)):
                    continue
                cur = bounds.setdefault(r.out, [-np.inf, np.inf])
                nlo, nhi = max(cur[0], flo if np.isfinite(flo) else -np.inf), min(cur[1], fhi if np.isfinite(fhi) else np.inf)
                if (nlo, nhi) != tuple(cur):
                    cur[0], cur[1] = nlo, nhi; changed = True
            elif r.kind in ("le", "implies"):
                L = bounds.setdefault(r.lhs, [-np.inf, np.inf]); R = bounds.setdefault(r.rhs, [-np.inf, np.inf])
                if R[1] < L[1]:
                    L[1] = R[1]; changed = True
                if L[0] > R[0]:
                    R[0] = L[0]; changed = True
            for v in bounds.values():
                if v[0] > v[1] + 1e-12:
                    return False
        if not changed:
            break
    return all(v[0] <= v[1] + 1e-12 for v in bounds.values())


def _bounds_of(claims: list[Claim], domains: dict | None = None) -> dict[str, list[float]]:
    b: dict[str, list[float]] = {k: [float(v[0]), float(v[1])] for k, v in (domains or {}).items()}
    for c in claims:
        if c.kind == "bound":
            cur = b.setdefault(c.subject, [-np.inf, np.inf])
            cur[0], cur[1] = max(cur[0], c.lo), min(cur[1], c.hi)
        elif c.kind == "prop":
            cur = b.setdefault(c.subject, [0.0, 1.0])
            v = 1.0 if c.value else 0.0
            cur[0], cur[1] = max(cur[0], v), min(cur[1], v)
    return b


def satisfiable(claims: list[Claim], rels: list[Relation], domains: dict | None = None) -> bool:
    """domains: quantity -> (lo, hi) that holds before any claim (e.g. T1's premise e, r ∈ [0, 1))."""
    return _propagate(_bounds_of(claims, domains), rels)


def same_instance(cs) -> bool:
    named = {c.instance for c in cs if c.instance}
    return len(named) <= 1


def interval_triples(claims: list[Claim], rels: list[Relation], domains: dict | None = None) -> list[TripleConf]:
    C = [c for c in claims if c.kind in ("bound", "prop")]
    sat1 = {c.id: satisfiable([c], rels, domains) for c in C}
    sat2 = {}
    for x, y in combinations(C, 2):
        sat2[(x.id, y.id)] = satisfiable([x, y], rels, domains)
    out = []
    for x, y, z in combinations(C, 3):
        if not same_instance([x, y, z]):
            continue
        if not (sat1[x.id] and sat1[y.id] and sat1[z.id]):
            continue
        if not (sat2[(x.id, y.id)] and sat2[(x.id, z.id)] and sat2[(y.id, z.id)]):
            continue
        if not satisfiable([x, y, z], rels, domains):
            trio = [x, y, z]
            used = sorted({r.part for r in rels if {r.out, r.lhs, r.rhs, *r.ins} & {c.subject for c in trio}})
            out.append(TripleConf("interval", (x.id, y.id, z.id), classify(trio, "interval"),
                                  "pairwise satisfiable, jointly empty under relations from " + ",".join(used),
                                  tuple(sorted({c.part for c in trio} | set(used)))))
    return out


# ------------------------------------------------------------------------------------------------ order cycles
def order_cycles(claims: list[Claim]) -> list[TripleConf]:
    O = [c for c in claims if c.kind == "order"]          # subject < object
    out, seen = [], set()
    for a, b, c in permutations(O, 3):
        if a.object == b.subject and b.object == c.subject and c.object == a.subject:
            key = frozenset((a.id, b.id, c.id))
            if key in seen:
                continue
            seen.add(key)
            trio = [a, b, c]
            out.append(TripleConf("order", (a.id, b.id, c.id), classify(trio, "order"),
                                  f"{a.subject}<{a.object}<{b.object}<{c.object}", tuple(sorted({x.part for x in trio}))))
    return out


# ------------------------------------------------------------------------------------------------ transitions
def transition_triples(claims: list[Claim]) -> list[TripleConf]:
    """sign_at claims on (subject, object) along `axis`; rule claims kind 'sign_at' with max_transitions ≥ 0."""
    obs = [c for c in claims if c.kind == "sign_at" and c.max_transitions < 0]
    rules = [c for c in claims if c.kind == "sign_at" and c.max_transitions >= 0]
    out = []
    for r in rules:
        rel = [c for c in obs if c.subject == r.subject and c.object == r.object and c.axis == r.axis]
        for trio in combinations(sorted(rel, key=lambda c: c.at), 3):
            s = [c.sign for c in trio]
            changes = sum(1 for u, v in zip(s[:-1], s[1:]) if u != v)
            if changes > r.max_transitions:
                cls = classify(list(trio) + [r], "transitions")
                out.append(TripleConf("transitions", tuple(c.id for c in trio), cls,
                                      f"pattern {''.join('+' if x > 0 else '-' for x in s)} along {r.axis} at "
                                      f"{[c.at for c in trio]} vs rule {r.id} (≤{r.max_transitions} change)",
                                      tuple(sorted({c.part for c in trio} | {r.part}))))
    return out


# ------------------------------------------------------------------------------------------------ chain sign conf
def _frustrated_cycles(claims: list[Claim]) -> list[TripleConf]:
    """A CHAIN of signed claims s→x1→…→o against an asserted s→o of the other sign is a cycle with negative sign product
    in the (undirected) signed claim graph. Harary (1953): a signed graph has no such cycle iff its vertices 2-colour
    (balance). FB_N_GREEN_SET_DISTANCE measured the same condition on an operator's off-diagonal sign pattern: the Green
    walk certificate equals the exact coherence iff the pattern 2-colours, and a frustrated cycle is where it goes loose.
    BFS 2-colouring; every non-tree edge that breaks the colouring closes one frustrated fundamental cycle, reported with
    its claims. Length 3 = sign_cycles; longer = chain conf."""
    S = [c for c in claims if c.kind == "sign" and c.sign in (1, -1)]
    adj = {}
    for c in S:
        adj.setdefault(c.subject, []).append((c.object, c)); adj.setdefault(c.object, []).append((c.subject, c))
    col, par, out = {}, {}, []
    for root in adj:
        if root in col:
            continue
        from collections import deque
        col[root] = 1; par[root] = (None, None); queue = deque([root])
        while queue:
            u = queue.popleft()
            for v, c in adj[u]:
                if v not in col:
                    col[v] = col[u] * c.sign; par[v] = (u, c); queue.append(v)
                elif col[v] != col[u] * c.sign and par[u][1] is not c and par[v][1] is not c:
                    def path(x):
                        p = [x]
                        while par[p[-1]][0] is not None:
                            p.append(par[p[-1]][0])
                        return p
                    pu, pv = path(u), path(v)
                    common = next(x for x in pu if x in pv)
                    nodes = pu[:pu.index(common) + 1] + list(reversed(pv[:pv.index(common)]))
                    cyc = []
                    for a, b in zip(nodes[:-1], nodes[1:]):
                        e = par[a][1] if par[a][0] == b else par[b][1]
                        cyc.append(e)
                    cyc.append(c)
                    key = frozenset(x.id for x in cyc)
                    if any(frozenset(t.claims) == key for t in out) or not same_instance(cyc):
                        continue
                    out.append(TripleConf("frustrated_cycle", tuple(x.id for x in cyc), classify(cyc, "sign"),
                                          f"cycle of {len(cyc)} signed claims with negative product: " +
                                          " ; ".join(f"{x.subject}->{x.object}({x.sign:+d})" for x in cyc),
                                          tuple(sorted({x.part for x in cyc}))))
    return out


def frustrated_cycles(claims: list[Claim], max_contexts: int = 10000) -> list[TripleConf]:
    """Balance witnesses, with separate trees for compatible instances/regions.

    A tree across incompatible claims can mask a valid cycle. Run the reused
    BFS kernel on each distinct active edge set at box endpoints/categorical
    values. Any nonempty box intersection includes an endpoint of each finite
    constrained axis. Cost is linear per context plus witness reconstruction;
    region enumeration is bounded explicitly, never silently truncated.
    """
    from itertools import product
    S = [c for c in claims if c.kind == "sign" and c.sign in (1, -1)]
    out = _frustrated_cycles(S)
    seen = {frozenset(x.claims) for x in out}
    instances = sorted({c.instance for c in S if c.instance}) or [""]
    contexts, edge_sets = 0, set()
    for instance in instances:
        pool = [c for c in S if not c.instance or c.instance == instance]
        axes = sorted({ax for c in pool for ax in c.regime})
        variables = sorted({v for c in pool for v in c.box})
        choices = [sorted({x for c in pool for x in c.regime.get(ax, [])}) for ax in axes]
        choices += [sorted({x for c in pool for x in c.box.get(v, ()) if np.isfinite(x)}) or [0.0] for v in variables]
        for values in product(*choices):
            contexts += 1
            if contexts > max_contexts:
                raise ValueError("cycle region enumeration exceeds max_contexts")
            reg = dict(zip(axes, values[:len(axes)]))
            point = dict(zip(variables, values[len(axes):]))
            active = [c for c in pool if all(reg[ax] in vals for ax,vals in c.regime.items())
                      and all(lo <= point[v] <= hi for v,(lo,hi) in c.box.items())]
            key = frozenset(c.id for c in active)
            if key in edge_sets:
                continue
            edge_sets.add(key)
            for cycle in _frustrated_cycles(active):
                key = frozenset(cycle.claims)
                if key not in seen:
                    seen.add(key); out.append(cycle)
    return out


def all_triple_confs(claims: list[Claim], rels: list[Relation], domains: dict | None = None) -> list[TripleConf]:
    return sign_cycles(claims) + interval_triples(claims, rels, domains) + order_cycles(claims) + transition_triples(claims) + \
        [f for f in frustrated_cycles(claims) if len(f.claims) > 3]


# ------------------------------------------------------------------------------------------------ federation check
def federation_sees(claims: list[Claim]) -> dict:
    """Feed the sign claims to claim_federation (one graph per claim, independent sources) and report what IT finds:
    stress points (pairwise conf/boundary) and inferred links. Used to show the triple kinds are outside its reach."""
    from .engine import Federation
    fed = Federation()
    for k, c in enumerate(claims):
        if c.kind != "sign":
            continue
        box = {v: list(b) for v, b in c.box.items()}
        fed.add_graph({"graph_id": f"G{k}", "concepts": [{"id": c.subject}, {"id": c.object}],
                       "sources": [{"id": f"s{k}"}],
                       "claims": [{"id": c.id, "subject": c.subject, "object": c.object, "sign": c.sign,
                                   "validity": box, "evidence": [f"s{k}"], "cost": 1.0}]})
    links, rejected = fed.inferred_links()
    return {"stress_points": [(p.kind, p.pair) for p in fed.stress_points()],
            "inferred": [(l.subject, l.via, l.object, l.sign) for l in links],
            "rejected": rejected}
