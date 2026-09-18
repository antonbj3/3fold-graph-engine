#!/usr/bin/env python3
"""
plan_value.py — one currency for probing and working: expected cost to SETTLE the nodes under the goals.

A node is settled when its belief leaves the band (1 − τ, τ): p ≥ τ counts as holding, p ≤ 1 − τ as failed (τ = the admission
level). Each node can be settled by INSTRUMENTS (cost c, reliability r): a cheap judge, a measurement cell, a full study.
An instrument's yes/no answer updates p by Bayes; the cost to settle a node is the smallest expected spend over the order in
which its instruments are tried, stopping as soon as the band is left:
        S(p) = min_instrument [ c + Σ_o q_o · S(p_o) ]  with S(p) = 0 outside the band     (finite-horizon dynamic programme)
An expensive, reliable instrument settles in one step; a cheap one settles when its answer is clear enough and is otherwise
followed by the expensive one. The value of having the cheap instrument is S_expensive(p) − S_all(p) ≥ 0: it is the
cascade economics of e5/e6 applied per node.

Why not "criticality × E[Δp]": the posterior is a martingale, so that quantity is identically zero. Value comes only from
a decision the outcome can change — here which instrument to pay for next, and whether the goal is abandoned.

Order across nodes: with settle cost S_i and hold probability p_i the series-system rule of unlock_value applies with S_i in
place of c_i — work the node with the largest (1 − p_i)/S_i first (chain segments when precedence binds). The goal's
expected cost is Σ_k S_(k) Π_{j<k} p_(j); goals are summed with their weights, a node under two goals counted in both.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from .unlock_value import expected_cost_of_order, p_of, rank

__all__ = ["settle_cost", "plan", "expected_cost", "cheap_instrument_value"]


def _update(p: float, r: float, yes: bool) -> float:
    q = p * r + (1 - p) * (1 - r)
    return p * r / q if yes else p * (1 - r) / (1 - q)


def settle_cost(p: float, instruments: list[tuple[float, float]], tau: float = 0.9, max_steps: int = 6) -> tuple[float, str]:
    """(expected cost to settle a node with belief p, name of the first instrument to use).
    instruments: [(cost, reliability, name)] or [(cost, reliability)]. A reliability of 1.0 settles in one step."""
    ins = [(c, r, (t[2] if len(t) > 2 else f"i{k}")) for k, t in enumerate(instruments) for c, r in [t[:2]]]

    @lru_cache(maxsize=None)
    def S(pb: float, steps: int) -> tuple[float, str]:
        if pb >= tau or pb <= 1 - tau:
            return 0.0, ""
        if steps == 0:
            return float("inf"), ""
        best = (float("inf"), "")
        for c, r, name in ins:
            if r >= 1.0:
                tot = c
            else:
                q = pb * r + (1 - pb) * (1 - r)
                tot = c + q * S(round(_update(pb, r, True), 12), steps - 1)[0] + (1 - q) * S(round(_update(pb, r, False), 12), steps - 1)[0]
            if tot < best[0]:
                best = (tot, name)
        return best
    return S(round(float(p), 12), max_steps)


def cheap_instrument_value(p: float, cheap: tuple[float, float], expensive: tuple[float, float], tau: float = 0.9) -> float:
    """S with the expensive instrument only − S with both: what the cheap instrument is worth on this node, in cost units."""
    return settle_cost(p, [expensive], tau)[0] - settle_cost(p, [cheap, expensive], tau)[0]


def plan(graph: dict[str, Any], p_holds: dict[str, float], instruments: dict[str, list[tuple]], tau: float = 0.9) -> dict[str, Any]:
    """Order of work with settle costs as the cost of each node, and the instrument to start with on the first node."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    S = {i: settle_cost(p_of(nodes[i], p_holds), ins, tau) for i, ins in instruments.items()}
    order = rank(graph, p_holds, cost={i: max(s[0], 1e-9) for i, s in S.items()})
    first = next((r["id"] for r in order if r["actionable"]), None)
    return {"order": order, "settle_cost": {i: round(s[0], 4) for i, s in S.items()}, "first_node": first,
            "first_instrument": S[first][1] if first in S else None}


def expected_cost(graph: dict[str, Any], p_holds: dict[str, float], instruments: dict[str, list[tuple]], tau: float = 0.9) -> float:
    from .unlock_value import _below
    nodes = {n["id"]: n for n in graph["nodes"]}; below = _below(nodes)
    gw = {**{g: 1.0 for g, n in nodes.items() if n.get("type") == "GOAL"}, **((graph.get("_meta") or {}).get("goal_weights") or {})}
    S = {i: settle_cost(p_of(nodes[i], p_holds), ins, tau)[0] for i, ins in instruments.items()}
    order = [r["id"] for r in rank(graph, p_holds, cost={i: max(s, 1e-9) for i, s in S.items()}) if r["actionable"]]
    tot = 0.0
    for g, w in gw.items():
        if g not in nodes:
            continue
        members = [i for i in order if i in below[g] and i in S]
        tot += float(w) * expected_cost_of_order(members, {i: p_of(nodes[i], p_holds) for i in members}, S)
    return tot
