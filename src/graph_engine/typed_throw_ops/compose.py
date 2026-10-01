"""compose.py — can part A hand a quantity to part B, and can a CHAIN A→B→C→D run as a whole?

LINK A→B (one shared quantity q, out-port of A = in-port of B):
    * claim_types.typecheck_link on the projection  a: (A's input) → q,  b: q → (B's output), carrying the units of
      q on each side and each part's continuous validity box. It checks shared concept, box intersection, scale and
      unit dimension. Its statement-kind check ("sign or number") is dropped: a part is an operator, not a signed
      claim; that reason is filtered and recorded as a note, nothing else is filtered.
    * port kind must match (a covariance matrix is not a scalar with the same name);
    * categorical regime intersection on every axis (parts.regime_intersection);
    * no declared assumption conflict (parts.CONFLICTS).
CHAIN [P1..Pk]: every adjacent link ok, every REQUIRED input of Pi (i ≥ 2) supplied by an earlier part in the chain
or marked `given` (AND semantics), and — the part pairwise checking never sees — the regime of the WHOLE chain is
non-empty. For boxes, pairwise-intersecting boxes always share a point (Helly number 2 for axis boxes), but a chain
only checks ADJACENT pairs, so P1∩P3 can be empty while P1∩P2 and P2∩P3 are not; on categorical axes even three
pairwise-intersecting value sets can have no common value. Such a chain composes link by link and has no regime to
run in: a regime conf located by its middle part.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .engine import typecheck_link
from .parts import Part, Port, assumption_conflicts, regime_intersection, box_intersection_many

_OPERATOR_ONLY_REASONS = ("claim_states_neither_sign_nor_value", "mixed_statement_kinds")


@dataclass
class Link:
    a: str
    b: str
    q: str
    ok: bool
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _port_claim(part: Part, port: Port, side: str) -> dict:
    other = (part.inputs[0].q if part.inputs else "__src__") if side == "a" else (part.outputs[0].q if part.outputs else "__sink__")
    units = {port.q: port.unit} if port.unit else {}
    if side == "a":
        return {"subject": other, "object": port.q, "validity": dict(part.box), "units": units}
    return {"subject": port.q, "object": other, "validity": dict(part.box), "units": units}


def _port_link(A: Part, o: Port, B: Part, i: Port) -> Link:
    """Check one supply occurrence, rather than another port with the same name."""
    reasons, notes = [], []
    tc = typecheck_link(_port_claim(A, o, "a"), _port_claim(B, i, "b"), concepts=[{"id": o.q}])
    for r in tc["reasons"]:
        (notes if r.startswith(_OPERATOR_ONLY_REASONS) else reasons).append(r)
    notes += tc["notes"]
    if o.kind != i.kind:
        reasons.append(f"port_kind_mismatch:{o.kind}!={i.kind}")
    if regime_intersection([A.regime, B.regime]) is None:
        bad = [ax for ax in set(A.regime) & set(B.regime) if not set(A.regime[ax]) & set(B.regime[ax])]
        reasons.append("regime_disjoint:" + ",".join(sorted(bad)))
    conf = assumption_conflicts(A.assumptions, B.assumptions)
    if conf:
        reasons.append("assumption_conflict:" + ";".join(f"{x}|{y}" for x, y in conf))
    return Link(A.id, B.id, o.q, not reasons, reasons, notes)


def link(A: Part, B: Part) -> list[Link]:
    """All candidate links A→B over shared quantities, each with its verdict."""
    return [_port_link(A, o, B, i) for o in A.outputs for i in B.inputs if o.q == i.q]


def best_link(A: Part, B: Part) -> Link | None:
    ls = link(A, B)
    if not ls:
        return None
    ok = [l for l in ls if l.ok]
    return ok[0] if ok else ls[0]


@dataclass
class Chain:
    parts: list[str]
    links: list[Link]
    ok_adjacent: bool
    inputs_closed: bool
    missing_inputs: list[tuple[str, str]]
    whole_regime: dict | None
    whole_box: dict | None
    reach: str

    @property
    def runs(self) -> bool:
        return self.ok_adjacent and self.inputs_closed and self.whole_regime is not None and self.whole_box is not None

    @property
    def regime_conf(self) -> bool:
        """Composes link by link, closes its inputs, but has no common regime."""
        return self.ok_adjacent and self.inputs_closed and (self.whole_regime is None or self.whole_box is None)


def evaluate_chain(parts: list[Part]) -> Chain:
    if not parts:
        raise ValueError("a typed chain needs at least one part")
    links, ok_adj = [], True
    for A, B in zip(parts[:-1], parts[1:]):
        l = best_link(A, B)
        if l is None:
            return Chain([p.id for p in parts], links, False, False, [], None, None, "none")
        links.append(l)
        ok_adj &= l.ok
    # The head's inputs are external supplies by the existing chain convention.
    # Keep their actual types, and every earlier output occurrence: quantity
    # names alone do not establish typed AND-input closure.
    avail = [(parts[0], i) for i in parts[0].inputs]
    missing = []
    for p in parts:
        for i in p.inputs:
            if (i.required and not i.given and p is not parts[0]
                    and not any(o.q == i.q and _port_link(a, o, p, i).ok for a, o in avail)):
                missing.append((p.id, i.q))
        avail.extend((p, o) for o in p.outputs)
    reg = regime_intersection([p.regime for p in parts])
    box = box_intersection_many([p.box for p in parts])
    from itertools import combinations
    if any(assumption_conflicts(a.assumptions, b.assumptions) for a, b in combinations(parts, 2)):
        ok_adj = False
    order = {"operator": 0, "ports": 1, "words": 2}
    reach = max((p.reach for p in parts), key=lambda r: order[r])
    return Chain([p.id for p in parts], links, ok_adj, not missing, missing, reg, box, reach)


def enumerate_chains(parts: list[Part], max_len: int = 4, min_len: int = 3, require_ok_links: bool = True,
                     max_out: int = 200000) -> list[Chain]:
    """Depth-first over the link graph (A→B iff some link A→B is ok, or exists when require_ok_links=False).
    Simple paths only; each chain is evaluated as a whole (evaluate_chain)."""
    by_id = {p.id: p for p in parts}
    succ: dict[str, list[str]] = {p.id: [] for p in parts}
    for A in parts:
        for B in parts:
            if A.id == B.id:
                continue
            l = best_link(A, B)
            if l is not None and (l.ok or not require_ok_links):
                succ[A.id].append(B.id)
    out: list[Chain] = []

    def dfs(path):
        if len(out) >= max_out:
            return
        if len(path) >= min_len:
            out.append(evaluate_chain([by_id[x] for x in path]))
        if len(path) == max_len:
            return
        for nxt in succ[path[-1]]:
            if nxt not in path:
                dfs(path + [nxt])

    for p in parts:
        dfs([p.id])
    return out
