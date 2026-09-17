#!/usr/bin/env python3
"""
unlock_value.py — order of work on an unlock graph when open nodes may FAIL.

tools/anchor_graph_tools.priority ranks an actionable node by (weighted goals it transitively unblocks) / cost.
It does not use how likely the node is to hold. Here every open node i has p_i = P(it holds) and a cost c_i, and a
goal is reached only if every node under it holds. If one of them fails, the work spent on the others under that
goal was wasted (the goal has to be re-planned), so failures should be found early.

Theorem (series system; Mitten 1960, Butterworth 1972). For one goal standing on independent nodes that are worked
one at a time and abandoned at the first failure, the expected cost  Σ_k c_(k) Π_{j<k} p_(j)  is minimized by working
in DECREASING order of  (1 − p_i) / c_i.   Proof: swap two adjacent nodes a, b; the cost changes by
p-prefix · [c_a + p_a c_b − c_b − p_b c_a] = p-prefix · [c_a(1−p_b) − c_b(1−p_a)], which is ≤ 0 exactly when
(1−p_a)/c_a ≥ (1−p_b)/c_b. With equal costs this is "the hardest node first".

Precedence. When open nodes depend on each other the single-node rule is no longer optimal (review: suboptimal in 257 of
400 two-chain graphs, worst 1.81× the optimum, when blocked nodes were simply put last). For chain precedence the
optimal policy works the initial chain SEGMENT with the largest ratio
        (1 − Π_{j∈S} p_j) / Σ_{k∈S} c_k Π_{j<k} p_j
as one block, then repeats (Garey 1973; Sidney 1975). `rank` does this; a segment stops where the dependency structure
branches or joins, so for trees and general graphs the order is a heuristic (tests: brute-force optimal on chains).

Several goals sharing nodes. Each node's failure term is weighted by  stake_i = Σ_{g above i} w_g · C_g  (C_g = open cost
under goal g): the ratio becomes Σ_k stake_k (1 − p_k) Π_{j<k} p_j over the same denominator. With one goal the stake is
common to all nodes and cancels. This weighting is not proven optimal.

Independence. The proof needs the nodes to fail independently. oed_probe_selection_engine.py models the opposite case
(failure modes sharing common causes). With positively shared causes a failure of one node raises the failure probability of
its siblings, and the order above is no longer proven optimal; examples/engine_experiments/e11 reports what was measured.

p_i comes from wherever the node's evidence lives: margin_net (Φ(z)), regime_posterior / claim_federation (belief in
the asserted sign at the node's operating point), or RISK_PRIOR by risk class when there is none.
"""
from __future__ import annotations

from typing import Any

__all__ = ["rank", "expected_cost_of_order", "RISK_PRIOR"]

RISK_PRIOR = {"LOW": 0.9, "MED": 0.7, "HIGH": 0.5, "SOURCE-COND": 0.5}
_OPEN = ("OPEN", "ASSUMED")


def _below(nodes: dict[str, dict]) -> dict[str, set]:
    memo: dict[str, set] = {}

    def go(i: str) -> set:
        if i not in memo:
            memo[i] = set()
            for d in nodes[i].get("depends_on", []):
                if d in nodes:
                    memo[i] |= {d} | go(d)
        return memo[i]
    return {i: go(i) for i in nodes}


def p_of(n: dict, p_holds: dict[str, float]) -> float:
    if n.get("status") == "PROVEN":
        return 1.0
    if n.get("status") == "REFUTED":
        return 0.0
    return float(p_holds.get(n["id"], RISK_PRIOR.get(n.get("risk"), 0.7)))


def rank(graph: dict[str, Any], p_holds: dict[str, float] | None = None, cost: dict[str, float] | None = None) -> list[dict]:
    """Open nodes in the order to work them. Precedence (`depends_on` between open nodes) is respected by construction:
    the next block is the initial chain segment with the largest ratio (see module docstring), not the single node with
    the largest (1 − p)/c. Nodes that can never become available (regime gate closed, or standing on a node that is
    neither open nor proven) come last with actionable = False."""
    from .tools import anchor_graph_tools as agt
    nodes = {n["id"]: n for n in graph["nodes"]}; below = _below(nodes); p_holds = p_holds or {}; cost = cost or {}
    c = {i: float(cost.get(i, agt._node_cost(n))) for i, n in nodes.items()}
    gw = {**{g: 1.0 for g, n in nodes.items() if n.get("type") == "GOAL"}, **((graph.get("_meta") or {}).get("goal_weights") or {})}
    gw = {g: float(w) for g, w in gw.items() if g in nodes}
    base = {r["id"]: r["priority"] for r in agt.priority(graph)}
    open_ids = [i for i, n in nodes.items() if n.get("status") in _OPEN]
    open_cost = {g: sum(c[j] for j in below[g] | {g} if nodes[j].get("status") in _OPEN) for g in gw}
    stake = {i: sum(w * open_cost[g] for g, w in gw.items() if i == g or i in below[g]) for i in open_ids}
    p = {i: p_of(nodes[i], p_holds) for i in open_ids}

    def gated(i):
        rg = nodes[i].get("regime_gate"); return bool(rg) and not rg.get("open", False)
    dead = {i for i in open_ids if gated(i) or any(nodes[d].get("status") not in _OPEN + ("PROVEN",) for d in nodes[i].get("depends_on", []) if d in nodes)}
    changed = True
    while changed:                                            # anything standing on a dead node is dead too
        changed = False
        for i in open_ids:
            if i not in dead and any(d in dead for d in nodes[i].get("depends_on", [])):
                dead.add(i); changed = True
    succ = {i: [j for j in open_ids if i in nodes[j].get("depends_on", [])] for i in open_ids}
    done: set = set(); order: list[tuple[str, float]] = []
    live = [i for i in open_ids if i not in dead]

    def ready(i, extra=()):
        return all((d in done) or (d in extra) or nodes[d].get("status") == "PROVEN" for d in nodes[i].get("depends_on", []) if d in nodes)

    def best_segment(i):
        best, seg, alive, spent, gain = (-1.0, [i]), [], 1.0, 0.0, 0.0
        cur = i
        while cur is not None:
            seg.append(cur); spent += alive * c[cur]; gain += stake.get(cur, 0.0) * alive * (1 - p[cur]); alive *= p[cur]
            r = gain / spent if spent > 0 else 0.0
            if r > best[0]:
                best = (r, list(seg))
            nxt = [j for j in succ[cur] if j not in done and j not in seg and j not in dead and ready(j, seg)]
            cur = nxt[0] if len(nxt) == 1 else None             # follow a chain; stop where it branches or joins
        return best
    while len(done) < len(live):
        avail = [i for i in live if i not in done and ready(i)]
        if not avail:
            break
        r, seg = max((best_segment(i) for i in avail), key=lambda t: (t[0], -len(t[1])))
        for k in seg:
            order.append((k, r)); done.add(k)
    out = [{"id": i, "actionable": True, "p_holds": round(p[i], 3), "cost": c[i], "stake": round(stake[i], 3), "score": round(r, 4),
            "priority_existing": base.get(i, 0.0)} for i, r in order]
    out += [{"id": i, "actionable": False, "p_holds": round(p[i], 3), "cost": c[i], "stake": round(stake[i], 3), "score": 0.0,
             "priority_existing": base.get(i, 0.0)} for i in open_ids if i not in done]
    return out


def expected_cost_of_order(order: list[str], p: dict[str, float], c: dict[str, float]) -> float:
    """Σ_k c_(k) Π_{j<k} p_(j): expected spend on one goal when work stops at the first failed node."""
    tot, alive = 0.0, 1.0
    for i in order:
        tot += alive * c[i]; alive *= p[i]
    return tot
