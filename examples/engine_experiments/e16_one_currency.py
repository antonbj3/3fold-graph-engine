#!/usr/bin/env python3
"""E16: probing and working in one currency (plan_value). SIMULATION on unlock graphs of 8 open nodes under one goal.
Each node has a hidden truth (holds with its prior p), a cheap judge (cost 1, reliability r_j) and a cell (cost 20, exact).
A node is settled when its belief leaves (1−τ, τ). Wrong settlements (settled as holding but false, or the reverse) are counted:
cost alone can be gamed by a cheap judge that settles everything wrongly. Work stops at the first node settled as failed.
Policies: existing priority order + cell only · (1−p)/c order + cell only · judge always first · plan_value (instrument
chosen per node by the settle-cost programme, nodes ordered by (1−p)/settle cost). Reported: mean spend per goal."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src/graph_engine/tools")); sys.path.insert(0, str(ROOT / "tests"))
import anchor_graph_tools as agt
from graph_engine.plan_value import settle_cost
from graph_engine.unlock_value import rank
from test_unlock_value import _graph
CJ, CC = 1.0, 20.0
def upd(p, r, yes):
    q = p * r + (1 - p) * (1 - r); return p * r / q if yes else p * (1 - r) / (1 - q)

def run_policy(policy, p0, truth, rj, rng):
    """Returns (spend, wrong settlements). Nodes worked in the policy's order; stop at the first node settled as failed."""
    ids = list(p0); spend = 0.0; wrong = 0
    if policy == "priority_cell": order = [r["id"] for r in agt.priority(_graph(list(p0.values()), [CC] * len(ids))) if r["id"] != "G"]
    elif policy == "ratio_cell": order = [r["id"] for r in rank(_graph(list(p0.values()), [CC] * len(ids)), p_holds=p0) if r["id"] != "G"]
    else:
        S = {i: settle_cost(p0[i], [(CJ, rj, "judge"), (CC, 1.0, "cell")], TAU)[0] for i in ids} if policy == "plan_value" else {i: CJ + CC for i in ids}
        order = [r["id"] for r in rank(_graph(list(p0.values()), [CC] * len(ids)), p_holds=p0, cost={i: max(S[i], 1e-9) for i in ids}) if r["id"] != "G"]
    for i in order:
        p = p0[i]
        if p >= TAU: wrong += not truth[i]; continue
        if p <= 1 - TAU: return spend, wrong + truth[i]
        while (1 - TAU) < p < TAU:
            use = "judge" if policy == "judge_first" else settle_cost(p, [(CJ, rj, "judge"), (CC, 1.0, "cell")], TAU)[1] if policy == "plan_value" else "cell"
            if use == "cell":
                spend += CC; p = 1.0 if truth[i] else 0.0
            else:
                spend += CJ; ans = truth[i] if rng.random() < rj else not truth[i]; p = upd(p, rj, ans)
        settled_holds = p >= TAU; wrong += settled_holds != truth[i]
        if not settled_holds: return spend, wrong
    return spend, wrong

out = {}
for TAU, rj in [(0.9, 0.8), (0.9, 0.97), (0.99, 0.8), (0.99, 0.97)]:
    rng = np.random.default_rng(0); res = {k: [] for k in ["priority_cell", "ratio_cell", "judge_first", "plan_value"]}
    for _ in range(600):
        p = rng.uniform(0.3, 0.98, 8); ids = [f"n{i}" for i in range(8)]; p0 = dict(zip(ids, p)); truth = {i: rng.random() < p0[i] for i in ids}
        for k in res: res[k].append(run_policy(k, p0, truth, rj, np.random.default_rng(rng.integers(1 << 30))))
    out[f"tau={TAU},judge_reliability={rj}"] = {k: {"spend": round(float(np.mean([a for a, _ in v])), 2), "wrong_settlements_per_goal": round(float(np.mean([b for _, b in v])), 3)} for k, v in res.items()}
    print(TAU, rj); [print("   ", k, v) for k, v in out[f"tau={TAU},judge_reliability={rj}"].items()]
json.dump(out, open(Path(__file__).parent / "e16_results.json", "w"), indent=1)
