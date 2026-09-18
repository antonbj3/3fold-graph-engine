#!/usr/bin/env python3
"""E16: probing and working in one currency (plan_value). SIMULATION on unlock graphs of 8 open nodes under one goal.
Each node has a hidden truth (holds with its prior p), a cheap judge (cost 1, reliability r_j) and a cell (cost 20, exact).
A node is settled when its belief leaves (1−τ, τ). Wrong settlements (settled as holding but false, or the reverse) are
counted: cost alone can be gamed by a cheap judge that settles everything wrongly. Work stops at the first node settled
as failed, so a policy that finds the failure earlier works FEWER nodes — wrong settlements per goal then mixes the
instrument rule with the order (review finding). Both are reported: per goal and per node WORKED.

Arms. Order rules: existing priority · (1−p)/c · (1−p)/(judge+cell) [= equal costs, the judge_first order] · (1−p)/settle
cost [the plan_value order]. Instrument rules: cell only · judge first, then cell · the settle-cost programme's choice.
The four headline arms pair each instrument rule with its own order; the two FIXED-ORDER arms run each instrument rule
under the OTHER's order, which isolates the instrument rule from the order. All arms on one graph share common random
numbers: every node has its own seeded stream of judge draws, so the same node gets the same answers in every arm
whatever position it holds in the order.

Sweep: r_j ∈ {0.55, 0.60, 0.70, 0.80, 0.97} × τ ∈ {0.90, 0.99}, N graphs per cell, standard errors over graphs."""
import json, sys
from functools import lru_cache
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src/graph_engine/tools")); sys.path.insert(0, str(ROOT / "tests"))
import anchor_graph_tools as agt
from graph_engine.plan_value import settle_cost
from graph_engine.unlock_value import rank
from test_unlock_value import _graph
CJ, CC = 1.0, 20.0
N_GRAPHS, N_NODES = 600, 8
ARMS = ["priority_cell", "ratio_cell", "judge_first", "plan_value", "judge_first_at_plan_value_order", "plan_value_at_judge_first_order"]


def upd(p, r, yes):
    q = p * r + (1 - p) * (1 - r); return p * r / q if yes else p * (1 - r) / (1 - q)


@lru_cache(maxsize=None)
def _settle(p, rj, tau):
    """(expected cost to settle, first instrument) — cached across graphs; the beliefs reached inside one node repeat."""
    return settle_cost(p, [(CJ, rj, "judge"), (CC, 1.0, "cell")], tau)


def orders(p0, rj, tau):
    """The four orders, as lists of node ids. Equal costs (judge+cell) give the judge_first order."""
    ids = list(p0); g = _graph(list(p0.values()), [CC] * len(ids))
    strip = lambda rs: [r["id"] for r in rs if r["id"] != "G"]
    S = {i: max(_settle(float(p0[i]), rj, tau)[0], 1e-9) for i in ids}
    return {"priority": strip(agt.priority(g)), "ratio": strip(rank(g, p_holds=p0)),
            "judge_first": strip(rank(g, p_holds=p0, cost={i: CJ + CC for i in ids})),
            "plan_value": strip(rank(g, p_holds=p0, cost=S))}


def run(instrument, order, p0, truth, rj, tau, seeds):
    """Work `order` with `instrument` ∈ {cell, judge_first, plan_value}; stop at the first node settled as failed.
    Node i draws its judge answers from its OWN generator, seeded by seeds[i], so the answers do not depend on the node's
    position in the order — every arm sees the same node answer the same way (common random numbers).
    Returns (spend, wrong settlements, nodes worked) — a node counts as worked once a settlement verdict is reached for it."""
    gen = {i: np.random.default_rng(s) for i, s in seeds.items()}
    spend, wrong, worked = 0.0, 0, 0
    for i in order:
        p = float(p0[i]); worked += 1
        if p >= tau:
            wrong += not truth[i]; continue
        if p <= 1 - tau:
            return spend, wrong + truth[i], worked
        while (1 - tau) < p < tau:
            use = "judge" if instrument == "judge_first" else _settle(p, rj, tau)[1] if instrument == "plan_value" else "cell"
            if use == "cell":
                spend += CC; p = 1.0 if truth[i] else 0.0
            else:
                spend += CJ; ans = truth[i] if gen[i].random() < rj else not truth[i]; p = upd(p, rj, ans)
        settled_holds = p >= tau; wrong += settled_holds != truth[i]
        if not settled_holds:
            return spend, wrong, worked
    return spend, wrong, worked


def cell(tau, rj, n_graphs=N_GRAPHS, seed=0):
    rng = np.random.default_rng(seed); res = {k: [] for k in ARMS}
    for _ in range(n_graphs):
        p = rng.uniform(0.3, 0.98, N_NODES); ids = [f"n{i}" for i in range(N_NODES)]; p0 = dict(zip(ids, p))
        truth = {i: rng.random() < p0[i] for i in ids}; seeds = {i: int(s) for i, s in zip(ids, rng.integers(1 << 62, size=N_NODES))}
        o = orders(p0, rj, tau)
        for arm, (instr, order) in {"priority_cell": ("cell", o["priority"]), "ratio_cell": ("cell", o["ratio"]),
                                    "judge_first": ("judge_first", o["judge_first"]), "plan_value": ("plan_value", o["plan_value"]),
                                    "judge_first_at_plan_value_order": ("judge_first", o["plan_value"]),
                                    "plan_value_at_judge_first_order": ("plan_value", o["judge_first"])}.items():
            res[arm].append(run(instr, order, p0, truth, rj, tau, seeds))
    out = {}
    for arm, v in res.items():
        a = np.array(v, float); ratio = a[:, 1] / a[:, 2]
        m = lambda x: [round(float(np.mean(x)), 4), round(float(np.std(x, ddof=1) / np.sqrt(len(x))), 4)]
        out[arm] = {"spend": m(a[:, 0]), "wrong_per_goal": m(a[:, 1]), "wrong_per_node_worked": m(ratio),
                    "nodes_worked": m(a[:, 2]), "n_graphs": int(len(a))}
    return out


if __name__ == "__main__":
    out = {"_meta": {"n_graphs_per_cell": N_GRAPHS, "n_nodes": N_NODES, "judge_cost": CJ, "cell_cost": CC,
                     "note": "values are [mean, standard error over graphs]; common random numbers across arms within a graph"}}
    hdr = f"{'arm':34s}{'spend':>16s}{'wrong/goal':>16s}{'wrong/worked':>16s}{'worked':>14s}"
    for tau in (0.90, 0.99):
        for rj in (0.55, 0.60, 0.70, 0.80, 0.97):
            key = f"tau={tau},judge_reliability={rj}"; out[key] = cell(tau, rj)
            print(f"\n{key}\n{hdr}")
            for arm, v in out[key].items():
                f = lambda k: f"{v[k][0]:>10.3f}±{v[k][1]:<5.3f}"
                print(f"{arm:34s}{f('spend'):>16s}{f('wrong_per_goal'):>16s}{f('wrong_per_node_worked'):>16s}{f('nodes_worked'):>14s}")
            json.dump(out, open(Path(__file__).parent / "e16_results.json", "w"), indent=1)
