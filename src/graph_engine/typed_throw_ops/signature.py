"""signature.py — a PORT signature of a part, next to (not instead of) mechanism_signature's fold signature.

mechanism_signature.Signature reads how an equilibrium loses uniqueness (limit points, exponents); GRAPH-01
(ASTRA_DISTILL jobs/BT-HARVEST28-GRAPH-01) measured that none of its 8 fields can record a port, an uncertainty
group or an observation set, so two mechanisms that differ exactly in what decides the answer sit at distance 0.
signature_vector already handles vector states, so "several state variables" is not the gap; the gap is operators
that are not F(x, μ) = 0 at all (a Schur conditioning, a cone projection, a greedy set rule).

The port signature is the tuple
    ops        operator-family tokens (parts.OPERATORS)
    in_shape   multiset of input port types:  dimension string of the unit (claim_types.dimension) for scalars,
               "<kind>:<q>" for operator / label / field objects
    out_shape  same for outputs
    arity      number of REQUIRED inputs (1 = pass-through, ≥2 = AND node: the part needs a second supplier)
    memory     'path_dependent' if the regime says so or a label/history port is carried, else 'memoryless'
    uncert     uncertainty_class values (GRAPH-01's PortGroup: independent box vs tied gain vs conditioned Schur)

distance(a, b) ∈ [0, ~3] or NaN:
    NaN when either part has reach 'words' (unknown is not far; see GRAPH_COMBINATORICS README).
    1 − J(ops) + 0.5·(1 − J(port types)) + 0.25·[arity class differs] + 0.25·[memory differs] + 0.5·[uncert differs]
It deliberately ignores subject, location and regime values: regime decides COMPOSABILITY (compose.py), not whether
two parts do the same thing.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .engine import dimension
from .parts import Part


def _ptype(p) -> str:
    if p.kind != "scalar":
        return f"{p.kind}:{p.q}"
    d = dimension(p.unit) if p.unit else None
    return f"dim:{d}" if d is not None else f"q:{p.q}"


@dataclass(frozen=True)
class PortSignature:
    ops: frozenset
    in_shape: tuple
    out_shape: tuple
    arity: int
    memory: str
    uncert: frozenset
    reach: str


def port_signature(p: Part) -> PortSignature:
    ins = Counter(_ptype(x) for x in p.inputs)
    outs = Counter(_ptype(x) for x in p.outputs)
    memory = "path_dependent" if ("path_dependent" in p.regime.get("history", []) and
                                  "memoryless" not in p.regime.get("history", [])) else "memoryless"
    if any(x.kind == "label" for x in p.inputs + p.outputs):
        memory = "path_dependent"
    return PortSignature(frozenset(p.operator), tuple(sorted(ins.items())), tuple(sorted(outs.items())),
                         sum(1 for x in p.inputs if x.required), memory,
                         frozenset(p.regime.get("uncertainty_class", [])), p.reach)


def _jacc(a, b) -> float:
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def distance(a: PortSignature, b: PortSignature) -> float:
    if a.reach == "words" or b.reach == "words":
        return float("nan")
    types_a = {t for t, _ in a.in_shape} | {t for t, _ in a.out_shape}
    types_b = {t for t, _ in b.in_shape} | {t for t, _ in b.out_shape}
    d = (1 - _jacc(a.ops, b.ops)) + 0.5 * (1 - _jacc(types_a, types_b))
    d += 0.25 * float((a.arity >= 2) != (b.arity >= 2))
    d += 0.25 * float(a.memory != b.memory)
    if a.uncert and b.uncert:
        d += 0.5 * float(not (a.uncert & b.uncert))
    return d


def word_distance(a: Part, b: Part) -> float:
    """Token Jaccard distance of the prose. A SUBJECT distance (throws.py: a negative filter), never mechanism."""
    import re
    tok = lambda s: {w for w in re.findall(r"[a-zåäö]{4,}", s.lower())}
    return 1 - _jacc(tok(a.does + " " + a.family), tok(b.does + " " + b.family))


def is_unknown(x: float) -> bool:
    return isinstance(x, float) and math.isnan(x)
